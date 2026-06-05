LOGOS Dataset Pipeline

Minimal pipeline that supports:

1) Generate from chunk prompts into data/raw.jsonl
2) Validate schema into data/validated.jsonl
3) Apply quality filters into data/quality.jsonl
4) Report counts and distributions from data/quality.jsonl
5) Split data/quality.jsonl into train/test
6) Generate R1 responses into data/r1.jsonl (train only)

Quick start:

python -m logos_dataset.generate
python -m logos_dataset.validator
python -m logos_dataset.quality
python -m logos_dataset.report
python -m logos_dataset.split
python -c "from config import TRAIN_PATH, R1_PATH; from src.logos.data.generate import batch_generate; total = batch_generate(TRAIN_PATH, R1_PATH); print(f'Generated {total} records')"
