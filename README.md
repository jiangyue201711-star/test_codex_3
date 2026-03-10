# Synthetic Agent Task Generator

This repository now includes an initial code implementation for generating benchmark-equivalent LLM-agent training tasks as specified in the PRD.

## Quick start

```bash
python3 src/generate_synthetic_tasks.py --count 200 --seed 7 --out artifacts/tasks.jsonl
```

Each JSONL line contains:
- `instruction`
- `validation_spec`

plus metadata fields (`id`, `family`, `difficulty`, `tags`) for dataset management.
