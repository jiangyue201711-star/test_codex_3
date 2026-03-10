# Synthetic Agent Task Generator

This repository includes a runnable generator for benchmark-equivalent LLM-agent training tasks.

## Quick start

```bash
python3 src/generate_synthetic_tasks.py --count 200 --seed 7 --out artifacts/tasks.jsonl
```

Each JSONL line contains:
- `instruction`
- `validation_spec`
- `app_files` (includes concrete `/app/task.py` and `/app/eval.py` contents)

plus metadata fields (`id`, `family`, `difficulty`, `tags`) for dataset management.
