## OpenAPI 3.0 Spec Generator (BERT-GNN)

### Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Install torch-geometric wheels matching your torch version if desired
```

### Generate Synthetic Dataset
```bash
python -m src.synthesize_data
```

### Train
```bash
python -m src.train
```

### Generate Spec
```bash
python -m src.generate
```

### Visualize Endpoint Graph
```bash
python -m src.visualize_graph
```

Notes:
- The model integrates BERT encodings with optional GAT-based graph context across endpoints.
- Training uses a schema-aware loss mixing CE and JSON Schema validation.
- For GPU, ensure CUDA-enabled torch and matching torch-geometric wheels.


