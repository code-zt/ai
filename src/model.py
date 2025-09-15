from typing import Optional

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GATConv
    HAS_PYG = True
except Exception:
    HAS_PYG = False


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class OpenAPIGenerator(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 384, nhead: int = 6, num_layers: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.use_gnn = HAS_PYG
        if self.use_gnn:
            self.gat1 = GATConv(d_model, d_model, heads=2, concat=True)
            self.gat2 = GATConv(d_model * 2, d_model, heads=1, concat=True)
        else:
            self.graph_ff = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        tgt_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Encoder on input tokens
        enc_inp = self.pos_enc(self.token_emb(input_ids))
        memory = self.encoder(enc_inp)

        # Graph context over pooled representations per batch element
        pooled = memory.mean(dim=1)
        if self.use_gnn and edge_index is not None:
            graph_ctx = self.gat1(pooled, edge_index)
            graph_ctx = self.gat2(graph_ctx, edge_index)
        else:
            graph_ctx = self.graph_ff(pooled)
        memory = memory + graph_ctx.unsqueeze(1)

        if tgt_ids is None:
            # shift inputs for autoregressive decoding
            tgt_ids = input_ids[:, :-1]
            memory = memory[:, 1:, :]

        tgt_emb = self.pos_enc(self.token_emb(tgt_ids))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(1)).to(tgt_emb.device)
        decoded = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
        logits = self.lm_head(decoded)
        return logits


