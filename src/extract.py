import ast
import re
from pathlib import Path
from typing import Dict, List


def _make_param(name: str, where: str = "path", ptype: str = "string", required: bool = True) -> Dict:
    return {"name": name, "in": where, "type": ptype, "required": required}


def extract_from_fastapi(project_dir: str) -> List[Dict]:
    specs: List[Dict] = []
    for py in Path(project_dir).rglob("*.py"):
        try:
            src = py.read_text(encoding="utf-8")
        except Exception:
            continue
        try:
            tree = ast.parse(src)
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.decorator_list:
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                        name = dec.func.attr.lower()
                        if name in {"get", "post", "put", "delete", "patch"}:
                            method = name.upper()
                            path_arg = dec.args[0].s if dec.args and isinstance(dec.args[0], ast.Constant) else None
                            if not path_arg:
                                continue
                            params: List[Dict] = []
                            for match in re.findall(r"\{(\w+)\}", path_arg):
                                params.append(_make_param(match, where="path", ptype="string", required=True))
                            specs.append({"path": path_arg, "method": method, "parameters": params, "framework": "FASTAPI"})
    return specs


def extract_from_flask(project_dir: str) -> List[Dict]:
    specs: List[Dict] = []
    route_re = re.compile(r"@\w*app\.route\(\s*['\"]([^'\"]+)['\"],\s*methods=\[([^\]]+)\]", re.I)
    for py in Path(project_dir).rglob("*.py"):
        try:
            src = py.read_text(encoding="utf-8")
        except Exception:
            continue
        for m in route_re.finditer(src):
            path = m.group(1)
            methods = [s.strip().strip("'\"").upper() for s in m.group(2).split(",")]
            params: List[Dict] = []
            for match in re.findall(r"<(?:(?:int|uuid|string):)?(\w+)>", path):
                ptype = "integer" if ":int" in path else "string"
                params.append(_make_param(match, where="path", ptype=ptype, required=True))
            for method in methods:
                specs.append({"path": path, "method": method, "parameters": params, "framework": "FLASK"})
    return specs


def extract_from_django(project_dir: str) -> List[Dict]:
    specs: List[Dict] = []
    pattern_re = re.compile(r"path\(\s*['\"]([^'\"]+)['\"]", re.I)
    for py in Path(project_dir).rglob("urls.py"):
        try:
            src = py.read_text(encoding="utf-8")
        except Exception:
            continue
        for m in pattern_re.finditer(src):
            raw = m.group(1)
            path = "/" + raw.strip("/")
            params: List[Dict] = []
            for match in re.findall(r"<(?:(?:int|uuid|str):)?(\w+)>", raw):
                ptype = "integer" if ":int" in raw else "string"
                params.append(_make_param(match, where="path", ptype=ptype, required=True))
            specs.append({"path": path, "method": "GET", "parameters": params, "framework": "DJANGO"})
    return specs


def extract_from_go(project_dir: str) -> List[Dict]:
    specs: List[Dict] = []
    # Basic patterns for gorilla/mux, chi, net/http HandleFunc
    mux_re = re.compile(r"HandleFunc\(\s*\"([^\"]+)\"\s*,", re.I)
    method_re = re.compile(r"Methods\(\s*\"(GET|POST|PUT|DELETE|PATCH)\"\s*\)")
    chi_re = re.compile(r"r\.(Get|Post|Put|Delete|Patch)\(\s*\"([^\"]+)\"", re.I)
    http_re = re.compile(r"http\.HandleFunc\(\s*\"([^\"]+)\"", re.I)
    for go in Path(project_dir).rglob("*.go"):
        try:
            src = go.read_text(encoding="utf-8")
        except Exception:
            continue
        # gorilla/mux style
        for mm in mux_re.finditer(src):
            path = mm.group(1)
            methods = method_re.findall(src) or ["GET"]
            params: List[Dict] = []
            for match in re.findall(r"\{(\w+)(?::[^}]+)?\}", path):
                params.append(_make_param(match, where="path", ptype="string", required=True))
            for mtd in methods:
                specs.append({"path": path, "method": mtd, "parameters": params, "framework": "GO"})
        # chi style
        for cm in chi_re.finditer(src):
            mtd = cm.group(1).upper()
            path = cm.group(2)
            params: List[Dict] = []
            for match in re.findall(r"\{(\w+)(?::[^}]+)?\}", path):
                params.append(_make_param(match, where="path", ptype="string", required=True))
            specs.append({"path": path, "method": mtd, "parameters": params, "framework": "GO"})
        # net/http
        for hm in http_re.finditer(src):
            path = hm.group(1)
            specs.append({"path": path, "method": "GET", "parameters": [], "framework": "GO"})
    return specs


def extract_endpoints(project_dir: str, framework: str) -> List[Dict]:
    fw = framework.strip().upper()
    if fw == "FASTAPI":
        return extract_from_fastapi(project_dir)
    if fw == "FLASK":
        return extract_from_flask(project_dir)
    if fw == "DJANGO":
        return extract_from_django(project_dir)
    if fw in {"GO", "GOLANG"}:
        return extract_from_go(project_dir)
    return []


