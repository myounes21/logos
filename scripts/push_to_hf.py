import os
import sys
from pathlib import Path
from datasets import load_dataset, DatasetDict

sys.path.append(str(Path(__file__).parent.parent / "src"))
from logos.config import settings

def main():
    if not settings.HF_TOKEN:
        print("Error: HF_TOKEN missing in .env. Please configure it to push to Hugging Face Hub.")
        return

    data_dir = Path(__file__).parent.parent / "data"
    
    files_to_push = {
        "train": data_dir / "train.jsonl",
        "test": data_dir / "test.jsonl",
        "r1": data_dir / "r1.jsonl",
    }
    
    dataset_dict = {}
    for split_name, file_path in files_to_push.items():
        if file_path.exists():
            print(f"Loading {split_name} split from {file_path}")
            dataset = load_dataset("json", data_files=str(file_path), split="train")
            dataset_dict[split_name] = dataset
        else:
            print(f"Warning: {file_path} not found. Skipping {split_name} split.")
            
    if not dataset_dict:
        print("No datasets found to push.")
        return
        
    ds = DatasetDict(dataset_dict)
    repo_name = "myounes21/logos-reasoning-dataset"
    print(f"Pushing dataset to {repo_name} on Hugging Face Hub...")
    ds.push_to_hub(repo_name, token=settings.HF_TOKEN)
    print("Push complete!")

if __name__ == "__main__":
    main()
