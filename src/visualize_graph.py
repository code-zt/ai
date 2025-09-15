import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from .graph import build_endpoint_graph


def visualize(data_path: str = "./data/synthetic.json", out_path: str = "./artifacts/graph.png"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(data_path) as f:
        specs = json.load(f)
    g = build_endpoint_graph(specs)
    pos = nx.spring_layout(g, seed=42)
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(g, pos, node_size=500, node_color="#89cff0")
    nx.draw_networkx_edges(g, pos, arrows=True, alpha=0.5)
    nx.draw_networkx_labels(g, pos, font_size=7)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved graph to {out_path}")


if __name__ == "__main__":
    visualize()


