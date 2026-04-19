from pathlib import Path
import json
import os
from src.logos.data.generate import generate_trace

BASE_DIR = Path(__file__).resolve().parents[1]

seeds_path = BASE_DIR / "data/raw/seeds.json"
output_path = BASE_DIR / "data/generated/raw_dataset.jsonl"


def load_seeds():
    with open(seeds_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save(sample):
    os.makedirs(output_path.parent, exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def run_tests(code, tests):
    try:
        env = {}
        exec(code, {}, env)

        func = next((v for v in env.values() if callable(v)), None)
        if not func:
            return False

        for inp, expected in tests:
            if func(inp) != expected:
                return False

        return True

    except Exception:
        return False



seeds = load_seeds()
total = len(seeds)
passed = 0

for i, seed in enumerate(seeds, 1):
    print(f"\n[{i}/{total}] generating...")

    sample = generate_trace(seed["instruction"])

    if not sample:
        print("rejected")
        continue

    if not run_tests(sample["answer"], seed["tests"]):
        print("failed tests")
        continue

    save(sample)
    passed += 1

    print("ok")

print("\n---")
print(f"saved: {passed}/{total}")
print("---")