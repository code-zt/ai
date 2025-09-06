import json
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import AdamW
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

class GoFunctionDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_length=512, max_output_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        go_code = item['go_code']
        swagger_json = item['swagger_json']

        instruction = (
            "task: generate_swagger for the given Go function. "
            "Return ONLY valid minified JSON with keys name, parameters, return_type. "
            "Go function:\n" + go_code
        )
        
        input_encoding = self.tokenizer(
            instruction,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            swagger_json,
            max_length=self.max_output_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = target_encoding['input_ids']
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': labels.flatten()
        }

class FunctionParser(pl.LightningModule):
    def __init__(self, model_name='t5-small', lr=3e-4):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.tokenizer.add_tokens(['{', '}', '[', ']', 'func', 'string', 'int', 'bool', 'float64'])
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.lr = lr
    
    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)

def main():
    pl.seed_everything(42, workers=True)

    with open('go_function_dataset.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # simple deterministic split instead of scikit-learn
    rng = random.Random(42)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx = indices[:split]
    val_idx = indices[split:]
    train_data = [dataset[i] for i in train_idx]
    val_data = [dataset[i] for i in val_idx]

    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    tokenizer.add_tokens(['{', '}', '[', ']', 'func', 'string', 'int', 'bool', 'float64'])

    train_dataset = GoFunctionDataset(train_data, tokenizer)
    val_dataset = GoFunctionDataset(val_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

    model = FunctionParser()

    os.makedirs('checkpoints', exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='function-parser-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=1,  # Короткое обучение для тестирования
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else 1,
        precision='16-mixed' if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=10,
        deterministic=True
    )

    trainer.fit(model, train_loader, val_loader)

    model.model.save_pretrained('function_parser_model')
    tokenizer.save_pretrained('function_parser_model')
    print("Model training completed! Saved to function_parser_model")

if __name__ == "__main__":
    main()