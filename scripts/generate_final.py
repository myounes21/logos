#!/usr/bin/env python3

import json, sys, time
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings, DATA_DIR

RAW = DATA_DIR / "raw.jsonl"
FINAL = DATA_DIR / "final_generated.jsonl"
CHECKPOINT = DATA_DIR / "generation_final.checkpoint.json"
BATCH = 3

PROMPT = """أنت خبير خوارزميات وبرمجة. مهمتك: لكل مسألة، توليد think بالعربية الفصحى (150-350 كلمة، لا Markdown، لا قوائم، نثر متصل باستخدام "بما أن، إذن، بالتالي") و answer (كود Python فقط). أخرج JSONL فقط. كل سطر: {"id":"...","think":"...","answer":"...","unit_tests":[...]}.

لا تستخدم أي لغة غير العربية في think. لا تذكر unit_tests داخل think."""


def strip(line):
    line = line.strip()
    if line.startswith("//"):
        line = line[2:].strip()
    return line


def load_raw():
    entries = []
    with open(RAW, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            c = strip(line)
            if c:
                try:
                    entries.append((i, json.loads(c)))
                except json.JSONDecodeError:
                    pass
    return entries


def load_done():
    done = set()
    if FINAL.exists():
        with open(FINAL, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        done.add(int(json.loads(line).get("id", 0)))
                    except:
                        pass
    return done


def main():
    print("generate_final.py", flush=True)
    raw = load_raw()
    done = load_done()
    remaining = [(i, e) for i, e in raw if i not in done]
    print(f"raw={len(raw)} done={len(done)} remaining={len(remaining)}", flush=True)
    if not remaining:
        return

    cp = 0
    if CHECKPOINT.exists():
        try:
            cp = json.loads(CHECKPOINT.read_text()).get("lines_processed", 0)
        except:
            pass
    remaining = [(i, e) for i, e in remaining if i > cp]
    print(f"checkpoint={cp} to_process={len(remaining)}", flush=True)
    if not remaining:
        return

    client = OpenAI(api_key=settings.QWEN_API_KEY, base_url=settings.QWEN_BASE_URL)
    total = 0
    batches = (len(remaining) + BATCH - 1) // BATCH

    for b in range(0, len(remaining), BATCH):
        batch = remaining[b : b + BATCH]
        bn = b // BATCH + 1
        fst, lst = batch[0][0], batch[-1][0]
        print(f"\nBatch {bn}/{batches} ({fst}-{lst})", flush=True)

        parts = []
        for idx, entry in batch:
            o = {"id": str(idx)}
            o.update(entry)
            parts.append(json.dumps(o, ensure_ascii=False))

        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=settings.QWEN_MODEL,
                messages=[
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": "أخرج JSONL:\n" + "\n".join(parts)},
                ],
                temperature=0.2,
            )
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            break

        dt = time.time() - t0
        content = resp.choices[0].message.content or ""

        results = {}
        for line in content.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("`"):
                continue
            try:
                obj = json.loads(line)
                if "id" in obj and "think" in obj and "answer" in obj:
                    results[str(obj["id"])] = obj
            except:
                pass

        with open(FINAL, "a", encoding="utf-8") as f:
            for idx, entry in batch:
                sid = str(idx)
                if sid in results:
                    r = results[sid]
                    record = {
                        "id": sid,
                        "instruction": entry.get("instruction", ""),
                        "topic": entry.get("topic"),
                        "subtopic": entry.get("subtopic"),
                        "difficulty": entry.get("difficulty"),
                        "problem_type": entry.get("problem_type"),
                        "think": r.get("think", ""),
                        "answer": r.get("answer", ""),
                        "unit_tests": r.get("unit_tests", entry.get("unit_tests", [])),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    total += 1
                    print(f"  {idx}: {len(r['think'].split())}words", flush=True)
                else:
                    print(f"  {idx}: MISSING", flush=True)
                CHECKPOINT.write_text(json.dumps({"lines_processed": idx}))

        time.sleep(0.3)
        print(f"  ({dt:.0f}s)", flush=True)

    rem = len(raw) - (len(done) + total)
    print(f"\nDone: {total} processed, ~{rem} remaining", flush=True)


if __name__ == "__main__":
    main()
