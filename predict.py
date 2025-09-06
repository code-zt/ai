import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import json
import re

def _build_instruction(go_code: str) -> str:
    return (
        "task: generate_swagger for the given Go function. "
        "Return ONLY valid minified JSON with keys name, parameters, return_type. "
        "Go function:\n" + go_code
    )

def _extract_json_like(text: str) -> str:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else text

def parse_function(go_code):
    model = T5ForConditionalGeneration.from_pretrained('function_parser_model')
    tokenizer = T5Tokenizer.from_pretrained('function_parser_model')

    instruction = _build_instruction(go_code)

    encoded = tokenizer(
        instruction,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    generate_kwargs = dict(
        max_length=256,
        num_beams=5,
        length_penalty=0.0,
        early_stopping=True,
        no_repeat_ngram_size=3
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask'],
            **generate_kwargs
        )

    raw_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    cleaned = _extract_json_like(raw_text).strip()

    try:
        # Validate JSON; if fails, try minor fixes
        return json.dumps(json.loads(cleaned), separators=(",", ":"))
    except Exception:
        # Heuristics: remove trailing commas and whitespace
        fixed = re.sub(r",\s*([}\]])", r"\\1", cleaned)
        try:
            return json.dumps(json.loads(fixed), separators=(",", ":"))
        except Exception:
            return raw_text

if __name__ == "__main__":
    sample_go_code = "func CalculateSum(a int, b int) int {\n    return a + b\n}"
    result = parse_function(sample_go_code)
    print("Go function:", sample_go_code)
    print("Parsed result:", result)
    
    # Попробуем загрузить как JSON для красивого вывода
    try:
        parsed_json = json.loads(result)
        print("Formatted JSON:")
        print(json.dumps(parsed_json, indent=2))
    except:
        print("Result is not a valid JSON, raw output above.")