import json
from typing import Dict, List, Tuple

import torch


class APITokenizer:
    def __init__(self):
        self.special_tokens = {
            "[PAD]": 0,
            "[BOS]": 1,
            "[EOS]": 2,
            "[UNK]": 3,
            "[ENDPOINT]": 4,
            "[METHOD]": 5,
            "[PARAM]": 6,
            "[SCHEMA]": 7,
            "[FRAMEWORK]": 8,
        }
        self.vocab: Dict[str, int] = dict(self.special_tokens)
        self.inv_vocab: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.frozen: bool = False

    def _add_token(self, token: str) -> int:
        if token in self.vocab:
            return self.vocab[token]
        if self.frozen:
            return self.vocab["[UNK]"]
        idx = len(self.vocab)
        self.vocab[token] = idx
        self.inv_vocab[idx] = token
        return idx

    def _tokenize_path(self, path: str) -> List[int]:
        parts = [p for p in path.strip().split("/") if p]
        return [self._add_token(f"/" + p) for p in parts]

    def _tokenize_method(self, method: str) -> List[int]:
        return [self._add_token(method.upper())]

    def _tokenize_params(self, parameters: List[Dict]) -> List[int]:
        tokens: List[int] = []
        for param in parameters:
            tokens.append(self.special_tokens["[PARAM]"])
            tokens.append(self._add_token(f"name={param.get('name','')}"))
            tokens.append(self._add_token(f"type={param.get('type','string')}"))
            if "in" in param:
                tokens.append(self._add_token(f"in={param['in']}"))
            if "required" in param:
                tokens.append(self._add_token(f"required={param['required']}"))
        return tokens

    def tokenize(self, spec: Dict) -> torch.Tensor:
        tokens: List[int] = [self.special_tokens["[BOS]"]]
        # Framework hint
        fw = spec.get("framework")
        if fw:
            tokens.append(self.special_tokens["[FRAMEWORK]"])
            tokens.append(self._add_token(str(fw).upper()))
        tokens.append(self.special_tokens["[ENDPOINT]"])
        tokens.extend(self._tokenize_path(spec.get("path", "/")))

        tokens.append(self.special_tokens["[METHOD]"])
        tokens.extend(self._tokenize_method(spec.get("method", "GET")))

        tokens.extend(self._tokenize_params(spec.get("parameters", [])))

        tokens.append(self.special_tokens["[EOS]"])
        return torch.tensor(tokens, dtype=torch.long)

    def detokenize(self, tokens: List[int]) -> Dict:
        # Very simple detokenizer for demonstration/generation
        path_parts: List[str] = []
        method: str = "GET"
        parameters: List[Dict] = []
        current: Dict = {}
        mode: str = ""
        framework: str = ""

        for t in tokens:
            token = self.inv_vocab.get(int(t), "")
            if token in ("[BOS]", "[EOS]", "[PAD]"):
                continue
            if token == "[ENDPOINT]":
                mode = "path"
                continue
            if token == "[METHOD]":
                mode = "method"
                continue
            if token == "[FRAMEWORK]":
                mode = "framework"
                continue
            if token == "[PARAM]":
                if current:
                    parameters.append(current)
                current = {}
                mode = "param"
                continue

            if mode == "path" and token.startswith("/"):
                path_parts.append(token)
            elif mode == "method":
                method = token
            elif mode == "framework":
                framework = token
            elif mode == "param":
                if token.startswith("name="):
                    current["name"] = token[len("name="):]
                elif token.startswith("type="):
                    current["type"] = token[len("type="):]
                elif token.startswith("in="):
                    current["in"] = token[len("in="):]
                elif token.startswith("required="):
                    val = token[len("required="):]
                    current["required"] = val == "True"

        if current:
            parameters.append(current)

        return {
            "path": "/" + "/".join([p.lstrip("/") for p in path_parts]) if path_parts else "/",
            "method": method,
            "parameters": parameters,
            "framework": framework or None,
        }


