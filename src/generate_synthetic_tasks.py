#!/usr/bin/env python3
"""Generate benchmark-equivalent LLM agent training tasks.

Each output record contains:
- instruction
- validation_spec
- app_files: {"/app/task.py": ..., "/app/eval.py": ...}
"""

from __future__ import annotations

import argparse
import json
import random
import re
import textwrap
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

BANNED_KEYWORDS = {
    "dominant eigenvalue",
    "eigenvector",
    "power iteration",
    "rayleigh",
    "spectral radius",
}

FAMILIES = {
    "numeric_kernel": {
        "goal": "Optimize batched sum-of-squares with float tolerance.",
        "io": "Input is List[List[float]], output is List[float] of sum(x*x).",
        "kind": "numeric",
        "pitfalls": ["temporary allocations", "python loop overhead", "floating-point drift"],
    },
    "string_batch": {
        "goal": "Accelerate token-frequency counting over string batches.",
        "io": "Input is List[str], output is List[int] where each item counts token occurrences.",
        "kind": "discrete",
        "pitfalls": ["quadratic scanning", "edge punctuation", "repeated splitting"],
    },
    "graph_microkernel": {
        "goal": "Optimize weighted out-degree computation from adjacency lists.",
        "io": "Input is List[List[Tuple[int,int]]], output is List[int] weighted sums.",
        "kind": "numeric",
        "pitfalls": ["index bounds", "duplicate edges", "branch-heavy loops"],
    },
    "dp_operator": {
        "goal": "Improve throughput for longest non-decreasing run per sequence.",
        "io": "Input is List[List[int]], output is List[int] run lengths.",
        "kind": "discrete",
        "pitfalls": ["base-case handling", "state reset bugs", "empty sequence edge"],
    },
    "geometry_batch": {
        "goal": "Optimize batched Manhattan distance computation.",
        "io": "Input is List[Tuple[Tuple[float,float],Tuple[float,float]]], output is List[float].",
        "kind": "numeric",
        "pitfalls": ["coordinate unpacking overhead", "degenerate points", "precision tolerance"],
    },
    "prob_sampling": {
        "goal": "Optimize probability normalization for non-negative vectors.",
        "io": "Input is List[List[float]], output is List[List[float]] normalized rows.",
        "kind": "numeric",
        "pitfalls": ["divide-by-zero", "drift from sum=1", "unnecessary copies"],
    },
    "etl_transform": {
        "goal": "Accelerate missing-aware standardization.",
        "io": "Input is List[List[float|None]], output is standardized List[List[float]].",
        "kind": "numeric",
        "pitfalls": ["null handling", "repeated aggregation", "unstable variance"],
    },
    "dsl_executor": {
        "goal": "Optimize simple DSL accumulator execution.",
        "io": "Input is List[List[str]], output is List[int] final accumulator values.",
        "kind": "discrete",
        "pitfalls": ["dispatch overhead", "opcode parsing", "state mutation bugs"],
    },
}

DIFFICULTY = {
    "L1": {"alpha": 0.98, "seeds": 2, "hidden_tests": 8},
    "L2": {"alpha": 0.95, "seeds": 3, "hidden_tests": 12},
    "L3": {"alpha": 0.90, "seeds": 4, "hidden_tests": 16},
    "L4": {"alpha": 0.85, "seeds": 5, "hidden_tests": 20},
}
TARGET_RATIO = [("L1", 2), ("L2", 4), ("L3", 3), ("L4", 1)]


@dataclass
class TaskRecord:
    id: str
    family: str
    difficulty: str
    instruction: str
    validation_spec: Dict
    app_files: Dict[str, str]


def build_instruction(family: str, difficulty: str, rng: random.Random) -> str:
    profile = FAMILIES[family]
    pitfalls = ", ".join(rng.sample(profile["pitfalls"], k=2))
    return (
        f"You are given a partially implemented function in /app/task.py. "
        f"Family: {family}. Goal: {profile['goal']} "
        f"Difficulty: {difficulty}. I/O contract: {profile['io']} "
        "Keep public function signatures unchanged. You may modify only /app/task.py. "
        "Do not change /app/eval.py. Your solution must pass correctness checks and "
        "beat the reference implementation on median runtime. "
        f"Pay special attention to: {pitfalls}."
    )


def build_validation_spec(family: str, difficulty: str, rng: random.Random) -> Dict:
    d = DIFFICULTY[difficulty]
    numeric = FAMILIES[family]["kind"] == "numeric"
    return {
        "correctness": {
            "deterministic_tests": 24 + 4 * (difficulty in {"L3", "L4"}),
            "randomized_tests": 32 + 8 * (difficulty in {"L3", "L4"}),
            "numeric_tolerance": {"abs_err": 1e-8 if numeric else 0.0, "rel_err": 1e-6 if numeric else 0.0},
        },
        "performance": {
            "metric": "median_time_per_call",
            "warmup_calls": 20,
            "timed_calls": 120,
            "pass_condition": f"candidate_median <= {d['alpha']} * reference_median",
        },
        "robustness": {
            "seeds": [rng.randint(1, 10_000_000) for _ in range(d["seeds"])],
            "hidden_tests": d["hidden_tests"],
        },
        "anti_cheat": {"disallow_eval_introspection": True, "shuffle_input_order": True},
    }


def _task_template(family: str, body: str) -> str:
    body_i = textwrap.indent(textwrap.dedent(body).strip(), "    ")
    return (
        f'"""Auto-generated task starter for family: {family}."""\n'
        'from __future__ import annotations\n\n'
        f'FAMILY = "{family}"\n\n'
        'def solve(data):\n'
        '    """TODO: optimize while preserving exact output semantics."""\n'
        f"{body_i}\n"
    )


def _eval_template(family: str, alpha: float, dataset_fn: str, reference_fn: str) -> str:
    dataset_i = textwrap.indent(textwrap.dedent(dataset_fn).strip(), "    ")
    reference_i = textwrap.indent(textwrap.dedent(reference_fn).strip(), "    ")
    return (
        f'"""Auto-generated evaluator for family: {family}."""\n'
        'from __future__ import annotations\n\n'
        'import random\nimport statistics\nimport time\n\n'
        f'FAMILY = "{family}"\n\n'
        'def build_dataset(seed: int, n: int = 120):\n'
        f"{dataset_i}\n\n"
        'def reference_solve(data):\n'
        f"{reference_i}\n\n"
        'def check_correctness(candidate_fn, seed: int = 0):\n'
        '    data = build_dataset(seed)\n'
        '    ref = reference_solve(data)\n'
        '    got = candidate_fn(data)\n'
        '    if len(ref) != len(got):\n'
        '        return False\n'
        '    for a, b in zip(ref, got):\n'
        '        if isinstance(a, float):\n'
        '            if abs(a - b) > 1e-8:\n'
        '                return False\n'
        '        elif isinstance(a, list):\n'
        '            if len(a) != len(b):\n'
        '                return False\n'
        '            for x, y in zip(a, b):\n'
        '                if abs(x - y) > 1e-8:\n'
        '                    return False\n'
        '        else:\n'
        '            if a != b:\n'
        '                return False\n'
        '    return True\n\n'
        'def median_runtime(fn, seed: int = 0, repeats: int = 80):\n'
        '    data = build_dataset(seed)\n'
        '    costs = []\n'
        '    for _ in range(repeats):\n'
        '        t0 = time.perf_counter()\n'
        '        fn(data)\n'
        '        costs.append(time.perf_counter() - t0)\n'
        '    return statistics.median(costs)\n\n'
        'def evaluate(candidate_fn):\n'
        '    if not check_correctness(candidate_fn, seed=7):\n'
        '        return {"passed": False, "reason": "correctness", "family": FAMILY}\n'
        '    c = median_runtime(candidate_fn, seed=11)\n'
        '    r = median_runtime(reference_solve, seed=11)\n'
        f'    return {{"passed": bool(c <= {alpha} * r), "candidate": c, "reference": r, "alpha": {alpha}, "family": FAMILY}}\n'
    )


def build_task_py(family: str) -> str:
    task_bodies = {
        "numeric_kernel": """
out = []
for row in data:
    s = 0.0
    for x in row:
        fx = float(x)
        s += fx * fx
    out.append(s)
return out
""",
        "string_batch": """
out = []
for text in data:
    cnt = 0
    for token in text.split():
        if token == "x":
            cnt += 1
    out.append(cnt)
return out
""",
        "graph_microkernel": """
out = []
for edges in data:
    total = 0
    for _dst, w in edges:
        total += int(w)
    out.append(total)
return out
""",
        "dp_operator": """
out = []
for seq in data:
    if not seq:
        out.append(0)
        continue
    best = 1
    cur = 1
    prev = seq[0]
    for v in seq[1:]:
        if v >= prev:
            cur += 1
        else:
            cur = 1
        if cur > best:
            best = cur
        prev = v
    out.append(best)
return out
""",
        "geometry_batch": """
out = []
for (x1, y1), (x2, y2) in data:
    out.append(abs(float(x1) - float(x2)) + abs(float(y1) - float(y2)))
return out
""",
        "prob_sampling": """
out = []
for row in data:
    total = 0.0
    for x in row:
        total += float(x)
    if total <= 0.0:
        out.append([0.0 for _ in row])
    else:
        out.append([float(x) / total for x in row])
return out
""",
        "etl_transform": """
out = []
for row in data:
    vals = [float(x) for x in row if x is not None]
    if not vals:
        out.append([0.0 for _ in row])
        continue
    mean = sum(vals) / len(vals)
    var = sum((v - mean) * (v - mean) for v in vals) / len(vals)
    std = (var ** 0.5) if var > 0 else 1.0
    out.append([0.0 if x is None else (float(x) - mean) / std for x in row])
return out
""",
        "dsl_executor": """
out = []
for prog in data:
    acc = 0
    for ins in prog:
        op, val = ins.split(':')
        v = int(val)
        if op == 'ADD':
            acc += v
        elif op == 'SUB':
            acc -= v
        elif op == 'MUL':
            acc *= v
    out.append(acc)
return out
""",
    }
    return _task_template(family, task_bodies[family])


def build_eval_py(family: str, difficulty: str) -> str:
    alpha = DIFFICULTY[difficulty]["alpha"]
    dataset_builders = {
        "numeric_kernel": """
rng = random.Random(seed)
return [[rng.uniform(-3, 3) for _ in range(32)] for _ in range(n)]
""",
        "string_batch": """
rng = random.Random(seed)
vocab = ["x", "a", "b", "c", "d"]
return [" ".join(rng.choice(vocab) for _ in range(40)) for _ in range(n)]
""",
        "graph_microkernel": """
rng = random.Random(seed)
return [[(rng.randint(0, 31), rng.randint(1, 8)) for _ in range(24)] for _ in range(n)]
""",
        "dp_operator": """
rng = random.Random(seed)
return [[rng.randint(0, 20) for _ in range(48)] for _ in range(n)]
""",
        "geometry_batch": """
rng = random.Random(seed)
return [((rng.uniform(-10, 10), rng.uniform(-10, 10)), (rng.uniform(-10, 10), rng.uniform(-10, 10))) for _ in range(n)]
""",
        "prob_sampling": """
rng = random.Random(seed)
return [[rng.random() for _ in range(16)] for _ in range(n)]
""",
        "etl_transform": """
rng = random.Random(seed)
data = []
for _ in range(n):
    row = []
    for _ in range(12):
        row.append(None if rng.random() < 0.15 else rng.uniform(-5, 5))
    data.append(row)
return data
""",
        "dsl_executor": """
rng = random.Random(seed)
ops = ["ADD", "SUB", "MUL"]
return [[f"{rng.choice(ops)}:{rng.randint(1,4)}" for _ in range(18)] for _ in range(n)]
""",
    }

    reference_impl = {
        "numeric_kernel": """
out = []
for row in data:
    total = 0.0
    for x in row:
        fx = float(x)
        total += fx * fx
    out.append(total)
return out
""",
        "string_batch": """
out = []
for text in data:
    c = 0
    for token in text.split():
        if token == "x":
            c += 1
    out.append(c)
return out
""",
        "graph_microkernel": """
out = []
for edges in data:
    s = 0
    for _dst, w in edges:
        s += int(w)
    out.append(s)
return out
""",
        "dp_operator": """
out = []
for seq in data:
    if not seq:
        out.append(0)
        continue
    best = 1
    cur = 1
    prev = seq[0]
    for v in seq[1:]:
        if v >= prev:
            cur += 1
        else:
            cur = 1
        if cur > best:
            best = cur
        prev = v
    out.append(best)
return out
""",
        "geometry_batch": """
out = []
for (x1, y1), (x2, y2) in data:
    out.append(abs(float(x1) - float(x2)) + abs(float(y1) - float(y2)))
return out
""",
        "prob_sampling": """
out = []
for row in data:
    s = 0.0
    for x in row:
        s += float(x)
    if s <= 0.0:
        out.append([0.0 for _ in row])
    else:
        out.append([float(x) / s for x in row])
return out
""",
        "etl_transform": """
out = []
for row in data:
    vals = [float(x) for x in row if x is not None]
    if not vals:
        out.append([0.0 for _ in row])
        continue
    mean = sum(vals) / len(vals)
    var = sum((v - mean) * (v - mean) for v in vals) / len(vals)
    std = (var ** 0.5) if var > 0 else 1.0
    out.append([0.0 if x is None else (float(x) - mean) / std for x in row])
return out
""",
        "dsl_executor": """
out = []
for prog in data:
    acc = 0
    for ins in prog:
        op, val = ins.split(':')
        v = int(val)
        if op == 'ADD':
            acc += v
        elif op == 'SUB':
            acc -= v
        elif op == 'MUL':
            acc *= v
    out.append(acc)
return out
""",
    }
    return _eval_template(family, alpha, dataset_builders[family], reference_impl[family])


def build_app_files(family: str, difficulty: str) -> Dict[str, str]:
    return {"/app/task.py": build_task_py(family), "/app/eval.py": build_eval_py(family, difficulty)}


def contains_banned_phrase(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in BANNED_KEYWORDS)


def jaccard_ngrams(a: str, b: str, n: int = 3) -> float:
    def ngrams(s: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9_]+", s.lower())
        grams = set()
        for token in tokens:
            if len(token) < n:
                grams.add(token)
            else:
                grams.update(token[i : i + n] for i in range(len(token) - n + 1))
        return grams

    sa, sb = ngrams(a), ngrams(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def build_difficulty_schedule(total: int) -> List[str]:
    bucket: List[str] = []
    for name, weight in TARGET_RATIO:
        bucket.extend([name] * weight)
    return [bucket[i % len(bucket)] for i in range(total)]


def generate_records(count: int, seed: int) -> List[TaskRecord]:
    rng = random.Random(seed)
    families = list(FAMILIES.keys())
    schedule = build_difficulty_schedule(count)
    records: List[TaskRecord] = []

    for i in range(count):
        family = families[i % len(families)]
        difficulty = schedule[i]
        instruction = build_instruction(family, difficulty, rng)

        if contains_banned_phrase(instruction):
            raise ValueError("Generated instruction violated anti-contamination keyword policy.")

        rec = TaskRecord(
            id=str(uuid.uuid4()),
            family=family,
            difficulty=difficulty,
            instruction=instruction,
            validation_spec=build_validation_spec(family, difficulty, rng),
            app_files=build_app_files(family, difficulty),
        )

        if any(jaccard_ngrams(rec.instruction, prev.instruction) > 0.88 for prev in records):
            family = rng.choice(families)
            instruction = build_instruction(family, difficulty, rng)
            rec = TaskRecord(
                id=str(uuid.uuid4()),
                family=family,
                difficulty=difficulty,
                instruction=instruction,
                validation_spec=build_validation_spec(family, difficulty, rng),
                app_files=build_app_files(family, difficulty),
            )

        records.append(rec)

    return records


def to_export_dict(record: TaskRecord) -> Dict:
    return {
        "id": record.id,
        "instruction": record.instruction,
        "validation_spec": record.validation_spec,
        "app_files": record.app_files,
        "family": record.family,
        "difficulty": record.difficulty,
        "tags": ["correctness", "performance", "robustness", record.family, record.difficulty],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic agent-training tasks.")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("artifacts/synthetic_tasks.jsonl"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    records = generate_records(count=args.count, seed=args.seed)

    with args.out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(to_export_dict(rec), ensure_ascii=False) + "\n")

    print(f"Generated {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
