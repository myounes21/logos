import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import os

def merge_and_export():
    adapter_path = "./outputs/grpo_final"
    output_path = "./outputs/logos_full_merged"

    print(f"Loading adapter from {adapter_path}...")
    model_to_merge = AutoPeftModelForCausalLM.from_pretrained(
        adapter_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    print("Merging adapters into base model...")
    merged_model = model_to_merge.merge_and_unload()
    
    print(f"Saving merged model to {output_path}...")
    merged_model.save_pretrained(output_path)
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_path)
    
    print("Merge complete! To export to GGUF, use llama.cpp:")
    print(f"python3 llama.cpp/convert_hf_to_gguf.py {output_path}")

if __name__ == "__main__":
    merge_and_export()
