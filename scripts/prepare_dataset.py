import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FINAL_FILE = DATA_DIR / "final_generated.jsonl"
RAW_FILE = DATA_DIR / "raw.jsonl"
OUTPUT_FILE = DATA_DIR / "logos_dataset.jsonl"

entries = []
with open(FINAL_FILE) as fh:
    for line in fh:
        if line.strip():
            entries.append(json.loads(line))

raw_entries = {}
with open(RAW_FILE) as fr:
    for i, line in enumerate(fr, 1):
        line = line.strip()
        if line.startswith("//"):
            line = line[2:].strip()
        if line:
            try:
                raw_entries[i] = json.loads(line)
            except json.JSONDecodeError:
                pass

good = []
for e in entries:
    oid = int(e["id"])
    think = e.get("think", "")
    answer = e.get("answer", "")
    think_w = len(think.split())
    if think_w < 50 or len(answer) < 30 or "pass" in answer:
        continue
    raw_e = raw_entries.get(oid, {})
    instruction = e.get("instruction", raw_e.get("instruction", ""))
    unit_tests = raw_e.get("unit_tests", e.get("unit_tests", []))

    # Format unit_tests for human-readable display
    tests_str = json.dumps(unit_tests, ensure_ascii=False)
    test_lines = []
    for t in unit_tests:
        inp = t.get("input", t.get("args", ""))
        exp = t.get("expected", "")
        test_lines.append(f"  {inp} -> {exp}")
    tests_display = "\n".join(test_lines) if test_lines else tests_str

    sharegpt = {
        "conversations": [
            {
                "from": "human",
                "value": f"المسألة:\n{instruction}\n\nالاختبارات:\n{tests_display}"
            },
            {
                "from": "gpt",
                "value": f"فكر: {think}\n\nالإجابة:\n```python\n{answer}\n```"
            }
        ]
    }
    good.append(sharegpt)

print(f"Total entries: {len(good)}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in good:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Saved to {OUTPUT_FILE}")
