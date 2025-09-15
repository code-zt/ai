from typing import Dict, List, Tuple

import networkx as nx


def build_endpoint_graph(specs: List[Dict]) -> nx.DiGraph:
    """
    Build a directed graph where nodes are endpoint paths and edges connect
    endpoints that share parameters or reference similar schema tokens.
    """
    g = nx.DiGraph()
    for spec in specs:
        node = spec.get("path", "/") + "#" + spec.get("method", "GET")
        g.add_node(node, method=spec.get("method", "GET"))

    def param_signature(s: Dict) -> List[str]:
        sig = []
        for p in s.get("parameters", []):
            sig.append(f"{p.get('name','')}:{p.get('type','string')}:{p.get('in','query')}")
        return sorted(sig)

    nodes = list(g.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a = nodes[i]
            b = nodes[j]
            sa = next(x for x in specs if x.get("path", "/") + "#" + x.get("method", "GET") == a)
            sb = next(x for x in specs if x.get("path", "/") + "#" + x.get("method", "GET") == b)
            inter = set(param_signature(sa)).intersection(param_signature(sb))
            if inter:
                g.add_edge(a, b, weight=len(inter))
                g.add_edge(b, a, weight=len(inter))
    return g


