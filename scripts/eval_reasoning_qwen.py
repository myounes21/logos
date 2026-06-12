import re
import pandas as pd
from datasets import load_dataset
from openai import OpenAI
import time
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from logos.config import settings
import os
if settings.HF_TOKEN:
    os.environ["HF_TOKEN"] = settings.HF_TOKEN

def get_qwen_client():
    if not settings.QWEN_API_KEY:
        raise ValueError("QWEN_API_KEY is not set in config/.env")
    return OpenAI(api_key=settings.QWEN_API_KEY, base_url=settings.QWEN_BASE_URL)

def evaluate_reasoning(instruction, expected_answer, think_text, client, model_name):
    if pd.isna(think_text) or str(think_text).strip() == "":
        return 1.0  # Base score for no reasoning

    prompt = f"""You are an expert evaluator grading the internal "Chain-of-Thought" (reasoning trace) of an AI model trained to solve coding problems.
The model was trained to think out loud in Arabic before outputting the final Python code.

Original Problem / Instruction:
{instruction}

Ground Truth / Expected Solution:
{expected_answer}

Model's Internal Reasoning Trace (Arabic):
{think_text}

Your task is to evaluate the QUALITY of the reasoning trace above. Does the model clearly understand the problem? Does it formulate a correct logical plan that would naturally lead to the expected ground truth solution? 

Rate the trace on a scale of 1 to 10 based on the following criteria:
1. Understanding: Does the trace show a clear grasp of the problem requirements?
2. Logical Planning: Is the algorithmic plan sound, step-by-step, and logically correct?
3. Alignment: Does the reasoning correctly align with the ground truth solution?
4. Coherence (Arabic): Is the thought process easy to follow and clearly articulated in Arabic?

Output ONLY an integer between 1 and 10. No other text or explanation."""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a harsh but fair judge of logical reasoning."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=10
        )
        content = response.choices[0].message.content.strip()
        # Extract the first integer found in the response
        match = re.search(r'\b([1-9]|10)\b', content)
        if match:
            return float(match.group(1))
        else:
            return 1.0
    except Exception as e:
        print(f"Error calling API: {e}")
        return 1.0

def main():
    print("Loading datasets...")
    # Load original dataset to get instructions
    dataset = load_dataset("myounes21/logos-reasoning-dataset", split="train")
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    test_dataset = split_dataset['test']
    
    # Create mapping from Sample_ID to instruction and answer
    id_to_data = {i: {"instruction": ex['instruction'], "answer": ex['answer']} for i, ex in enumerate(test_dataset)}
    
    print("Loading raw results CSV...")
    df = pd.read_csv("results/logos_raw.csv")
    
    client = get_qwen_client()
    model_name = settings.QWEN_MODEL
    print(f"Using API with model: {model_name}")
    
    scores = []
    
    print("Evaluating reasoning traces...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            sample_id = int(row['Sample_ID'])
        except ValueError:
            scores.append(None)
            continue
        
        data = id_to_data.get(sample_id, {"instruction": "Unknown Instruction", "answer": ""})
        instruction = data["instruction"]
        expected_answer = data["answer"]
        think_text = row.get('think_text', "")
        
        score = evaluate_reasoning(instruction, expected_answer, think_text, client, model_name)
        scores.append(score)
        
        # Rate limit protection just in case
        time.sleep(0.1)
        
    df['reasoning_score'] = scores
    
    output_path = "results/logos_raw.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved detailed scores to {output_path}")
    
    print("\nAggregated Reasoning Scores (out of 10)")
    print("=" * 60)
    agg_scores = df.groupby('Model')['reasoning_score'].mean().reset_index()
    
    # Update logos_results.csv
    try:
        results_df = pd.read_csv("results/logos_results.csv")
        score_map = dict(zip(agg_scores['Model'], agg_scores['reasoning_score']))
        results_df['Reasoning Score (avg)'] = results_df['Model'].map(score_map).round(2)
        results_df.to_csv("results/logos_results.csv", index=False)
        print("Updated aggregated scores in results/logos_results.csv")
    except Exception as e:
        print(f"Could not update logos_results.csv: {e}")

    for _, row in agg_scores.iterrows():
        print(f"Model: {row['Model']:<25} | Avg Reasoning Score: {row['reasoning_score']:.2f}")

if __name__ == "__main__":
    main()
