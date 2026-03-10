# Synthetic Agent Task Generator

This repository includes a runnable generator for benchmark-equivalent LLM-agent training tasks.

## Quick start

```bash
python3 src/generate_synthetic_tasks.py --count 240 --seed 7 --out artifacts/tasks.jsonl
```

## What is generated

Each JSONL line contains:
- `instruction`
- `validation_spec`
- `app_files` (contains family-aligned `/app/task.py` and `/app/eval.py`)

plus metadata fields (`id`, `family`, `difficulty`, `tags`) for dataset management.

## Coverage updates

- Expanded task families from 8 to 12 (`bitset_ops`, `window_stats`, `trie_like_lookup`, `schedule_sim` added).
- Expanded difficulty tiers from 4 to 5 (new `L5`) with stricter performance threshold and heavier robustness requirements.
