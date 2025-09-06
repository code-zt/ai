import json
import random
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from predict import parse_function
from typing import Tuple

def evaluate_model(test_data, num_samples=50) -> Tuple[float, float]:
    smoothie = SmoothingFunction().method4
    total_bleu = 0
    correct_count = 0
    
    for i in range(min(num_samples, len(test_data))):
        item = test_data[i]
        go_code = item['go_code']
        ground_truth = item['swagger_json']
        
        generated = parse_function(go_code)
        
        reference = [ground_truth.split()]
        candidate = generated.split()
        bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
        
        total_bleu += bleu_score
        
        # Простая проверка: если сгенерированный JSON парсится и содержит ожидаемые поля
        try:
            generated_json = json.loads(generated)
            # parameters must be list, return_type must exist
            if isinstance(generated_json.get('parameters'), list) and 'name' in generated_json and 'return_type' in generated_json:
                correct_count += 1
        except Exception:
            pass
        
        if i < 5:
            print(f"Go code: {go_code}")
            print(f"Generated: {generated}")
            print(f"Ground truth: {ground_truth}")
            print(f"BLEU: {bleu_score:.4f}")
            print("---")
    
    avg_bleu = total_bleu / min(num_samples, len(test_data))
    accuracy = correct_count / min(num_samples, len(test_data))
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    print(f"Accuracy (valid JSON with required fields): {accuracy:.4f}")
    
    return avg_bleu, accuracy

if __name__ == "__main__":
    with open('go_function_dataset.json', 'r', encoding='utf-8') as f:
        full_dataset = json.load(f)

    rng = random.Random(42)
    indices = list(range(len(full_dataset)))
    rng.shuffle(indices)
    test_size = max(1, int(0.1 * len(indices)))
    test_idx = indices[:test_size]
    test_data = [full_dataset[i] for i in test_idx]

    avg_bleu, accuracy = evaluate_model(test_data)