import json
import random
from pathlib import Path
from typing import Dict, List


METHODS = ["GET", "POST", "PUT", "DELETE"]
TYPES = ["string", "integer", "boolean"]
PARAM_IN = ["query", "path", "header"]
FRAMEWORKS = ["FASTAPI", "FLASK", "DJANGO", "GO"]


def random_param() -> Dict:
    return {
        "name": random.choice(["id", "age", "name", "active", "page", "limit"]),
        "type": random.choice(TYPES),
        "in": random.choice(PARAM_IN),
        "required": random.choice([True, False]),
    }


def random_spec() -> Dict:
    resource = random.choice(["users", "orders", "products", "sessions", "reports"]) 
    path = f"/api/{resource}"
    method = random.choice(METHODS)
    parameters = [random_param() for _ in range(random.randint(1, 3))]
    framework = random.choices(FRAMEWORKS, weights=[0.4, 0.25, 0.2, 0.15])[0]
    return {"path": path, "method": method, "parameters": parameters, "framework": framework}


def synthesize(n: int = 500) -> List[Dict]:
    data = [random_spec() for _ in range(n)]
    # induce correlations: parameters shared between related endpoints
    for res in ("users", "orders", "products"):
        for method in ("GET", "POST"):
            data.append({
                "path": f"/api/{res}",
                "method": method,
                "parameters": [
                    {"name": "id", "type": "integer", "in": "query", "required": False},
                    {"name": "page", "type": "integer", "in": "query", "required": False},
                ],
                "framework": "FASTAPI",
            })
    # Add framework-specific patterns
    data += [
        {"path": "/items/{item_id}", "method": "GET", "parameters": [
            {"name": "item_id", "type": "integer", "in": "path", "required": True}
        ], "framework": "FASTAPI"},
        {"path": "/users/<int:user_id>", "method": "GET", "parameters": [
            {"name": "user_id", "type": "integer", "in": "path", "required": True}
        ], "framework": "FLASK"},
        {"path": "/api/v1/users/<uuid:id>", "method": "DELETE", "parameters": [
            {"name": "id", "type": "string", "in": "path", "required": True}
        ], "framework": "DJANGO"},
    ]
    # Add GO-like routes
    data += [
        {"path": "/api/v1/todos/{id}", "method": "GET", "parameters": [
            {"name": "id", "type": "integer", "in": "path", "required": True}
        ], "framework": "GO"},
        {"path": "/api/v1/login", "method": "POST", "parameters": [
            {"name": "username", "type": "string", "in": "query", "required": True},
            {"name": "password", "type": "string", "in": "query", "required": True}
        ], "framework": "GO"},
    ]
    return data


def main():
    out_dir = Path("./data")
    out_dir.mkdir(parents=True, exist_ok=True)
    data = synthesize(600)
    with (out_dir / "synthetic.json").open("w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(data)} samples to {out_dir / 'synthetic.json'}")


if __name__ == "__main__":
    main()


