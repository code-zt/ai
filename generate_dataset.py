import json
import random
from typing import List, Dict, Any

class GoFunctionDatasetGenerator:
    def __init__(self):
        self.function_names = ["GetUser", "CreateUser", "UpdateUser", "DeleteUser", 
                              "GetProduct", "CreateProduct", "UpdateProduct", "DeleteProduct",
                              "GetOrder", "CreateOrder", "UpdateOrder", "DeleteOrder",
                              "Login", "Register", "Logout", "RefreshToken",
                              "Search", "Filter", "Sort", "Paginate"]
        self.param_types = ["int", "string", "bool", "float64", "[]byte", "time.Time"]
        self.return_types = ["error", "string", "int", "bool", "interface{}", "[]byte", 
                            "map[string]interface{}", "struct{}", "*http.Response"]
        
    def generate_go_function(self) -> str:
        func_name = random.choice(self.function_names)
        num_params = random.randint(0, 5)
        
        params = []
        for i in range(num_params):
            param_name = f"param{i+1}"
            param_type = random.choice(self.param_types)
            params.append(f"{param_name} {param_type}")
        
        return_type = random.choice(self.return_types)
        
        params_str = ", ".join(params)
        
        return f"func {func_name}({params_str}) {return_type} {{\n    // Implementation here\n    return nil\n}}"
    
    def generate_swagger_spec(self, go_code: str) -> Dict[str, Any]:
        lines = go_code.split('\n')
        func_line = lines[0]
        
        parts = func_line.split('(')
        func_decl = parts[0].replace('func ', '').strip()
        func_name = func_decl.split()[0] if ' ' in func_decl else func_decl
        
        params_part = parts[1].split(')')[0]
        return_type = parts[1].split(')')[1].strip()
        
        parameters = []
        if params_part.strip():
            param_list = [p.strip() for p in params_part.split(',')]
            for param in param_list:
                if ' ' in param:
                    name, p_type = param.split()
                    parameters.append({"name": name, "type": p_type})
                else:
                    parameters.append({"name": param, "type": "unknown"})
        
        # Синтетическое ограничение: привести return_type к более реалистичному HTTP-слою
        rt = return_type
        if '*' in rt or 'interface' in rt or 'struct' in rt:
            rt = "object"
        elif '[]byte' in rt:
            rt = "string"
        elif 'int' in rt:
            rt = "integer"
        elif 'bool' in rt:
            rt = "boolean"
        elif 'string' in rt:
            rt = "string"

        # маппинг типов параметров в JSON-типы
        def map_type(t: str) -> str:
            if 'int' in t:
                return 'integer'
            if 'bool' in t:
                return 'boolean'
            if 'float' in t:
                return 'number'
            if '[]byte' in t:
                return 'string'
            return 'string'

        parameters = [
            {"name": p["name"], "type": map_type(p["type"])} for p in parameters
        ]

        return {
            "name": func_name,
            "parameters": parameters,
            "return_type": rt
        }
    
    def generate_dataset(self, num_samples: int = 1200) -> List[Dict[str, Any]]:
        dataset = []
        for i in range(num_samples):
            go_code = self.generate_go_function()
            swagger_spec = self.generate_swagger_spec(go_code)
            
            dataset.append({
                "id": i,
                "go_code": go_code,
                "swagger_spec": swagger_spec,
                "swagger_json": json.dumps(swagger_spec, separators=(",", ":"))
            })
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1} samples")
        
        return dataset

if __name__ == "__main__":
    generator = GoFunctionDatasetGenerator()
    dataset = generator.generate_dataset(1200)
    
    with open('go_function_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print("Dataset generation completed! Saved to go_function_dataset.json")