LOGOS Dataset Pipeline

Minimal pipeline that supports:

1) Generate from chunk prompts into data/raw.jsonl
2) Validate schema into data/validated.jsonl
3) Apply quality filters into data/quality.jsonl
4) Report counts and distributions from data/quality.jsonl

Quick start:

python -m logos_dataset.generate
python -m logos_dataset.validator
python -m logos_dataset.quality
python -m logos_dataset.report
