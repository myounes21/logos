import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from logos.data.filter import is_valid_reasoning

def validate():
    input_file = "data/filtered/train.jsonl" # Path where actual training file exists
    if not os.path.exists(input_file):
        input_file = "data/chunks/validated_dataset.jsonl"
    
    if not os.path.exists(input_file):
        print(f"Error: Could not find dataset at {input_file}")
        return

    dataset = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                dataset.append(json.loads(line))
            except Exception as e:
                print(f"JSON Parse Error: {e}")
                return

    bad_ids = ["85", "91", "92", "113"]
    valid_count = 0
    errors = 0

    for item in dataset:
        item_id = str(item.get("id", ""))
        if item_id in bad_ids:
            errors += 1
            continue
            
        think = item.get("think", "")
        if not is_valid_reasoning(think):
            errors += 1
            continue
            
        ans = item.get("answer", "")
        if not ans.strip():
            errors += 1
            continue
            
        valid_count += 1

    print(f"Validation Complete. Valid: {valid_count}, Errors: {errors}")

if __name__ == "__main__":
    validate()
