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
    "bitset_ops": {
        "goal": "Accelerate bitset intersection and popcount aggregation.",
        "io": "Input is List[Tuple[int,int]], output is List[int] popcount(a & b).",
        "kind": "discrete",
        "pitfalls": ["bit twiddling overhead", "large integer ops", "branch prediction misses"],
    },
    "window_stats": {
        "goal": "Optimize rolling-window average over dense vectors.",
        "io": "Input is List[List[float]], output is List[List[float]] rolling means.",
        "kind": "numeric",
        "pitfalls": ["prefix sum bugs", "off-by-one window bounds", "memory churn"],
    },
    "trie_like_lookup": {
        "goal": "Speed up prefix score computation for token strings.",
        "io": "Input is List[str], output is List[int] prefix scores.",
        "kind": "discrete",
        "pitfalls": ["prefix matching complexity", "repeated slicing", "hash collisions"],
    },
    "schedule_sim": {
        "goal": "Optimize simple queue scheduling simulation.",
        "io": "Input is List[List[Tuple[int,int]]], output is List[int] completion times.",
        "kind": "discrete",
        "pitfalls": ["queue state drift", "time-step inefficiency", "edge-case starvation"],
    },
}

DIFFICULTY = {
    "L1": {"alpha": 0.99, "seeds": 2, "hidden_tests": 8, "timed_calls": 90},
    "L2": {"alpha": 0.97, "seeds": 3, "hidden_tests": 12, "timed_calls": 120},
    "L3": {"alpha": 0.94, "seeds": 4, "hidden_tests": 18, "timed_calls": 150},
    "L4": {"alpha": 0.90, "seeds": 5, "hidden_tests": 24, "timed_calls": 180},
    "L5": {"alpha": 0.86, "seeds": 7, "hidden_tests": 32, "timed_calls": 240},
}
TARGET_RATIO = [("L1", 1), ("L2", 3), ("L3", 3), ("L4", 2), ("L5", 1)]


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
            "deterministic_tests": 20 + 6 * (difficulty in {"L3", "L4", "L5"}) + 4 * (difficulty == "L5"),
            "randomized_tests": 24 + 10 * (difficulty in {"L3", "L4", "L5"}) + 12 * (difficulty == "L5"),
            "distribution_shift_tests": 0 if difficulty in {"L1", "L2"} else 6 + 6 * (difficulty in {"L4", "L5"}),
            "numeric_tolerance": {"abs_err": 1e-8 if numeric else 0.0, "rel_err": 1e-6 if numeric else 0.0},
        },
        "performance": {
            "metric": "median_time_per_call",
            "warmup_calls": 20 + 10 * (difficulty in {"L4", "L5"}),
            "timed_calls": d["timed_calls"],
            "pass_condition": f"candidate_median <= {d['alpha']} * reference_median",
            "cold_start_budget_ms": 90 if difficulty == "L1" else 70 if difficulty == "L2" else 55 if difficulty == "L3" else 40,
        },
        "robustness": {
            "seeds": [rng.randint(1, 10_000_000) for _ in range(d["seeds"])],
            "hidden_tests": d["hidden_tests"],
            "adversarial_cases": 2 if difficulty == "L1" else 4 if difficulty == "L2" else 8 if difficulty == "L3" else 12,
        },
        "anti_cheat": {
            "disallow_eval_introspection": True,
            "shuffle_input_order": True,
            "randomize_lengths": difficulty in {"L3", "L4", "L5"},
            "forbid_global_cache": difficulty in {"L4", "L5"},
        },
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
        'def build_dataset(seed: int, n: int = 140):\n'
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
        '            if abs(a - float(b)) > 1e-8:\n'
        '                return False\n'
        '        elif isinstance(a, list):\n'
        '            if len(a) != len(b):\n'
        '                return False\n'
        '            for x, y in zip(a, b):\n'
        '                if abs(float(x) - float(y)) > 1e-8:\n'
        '                    return False\n'
        '        else:\n'
        '            if a != b:\n'
        '                return False\n'
        '    return True\n\n'
        'def median_runtime(fn, seed: int = 0, repeats: int = 90):\n'
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
        "bitset_ops": """
out = []
for a, b in data:
    out.append((int(a) & int(b)).bit_count())
return out
""",
        "window_stats": """
out = []
for row in data:
    if len(row) < 5:
        out.append([])
        continue
    acc = []
    for i in range(4, len(row)):
        s = 0.0
        for j in range(i - 4, i + 1):
            s += float(row[j])
        acc.append(s / 5.0)
    out.append(acc)
return out
""",
        "trie_like_lookup": """
out = []
for text in data:
    score = 0
    words = text.split()
    for w in words:
        if w.startswith('pre'):
            score += 3
        elif w.startswith('pro'):
            score += 2
        elif w.startswith('p'):
            score += 1
    out.append(score)
return out
""",
        "schedule_sim": """
out = []
for jobs in data:
    t = 0
    for arrival, dur in jobs:
        if t < arrival:
            t = arrival
        t += dur
    out.append(t)
return out
""",
    }
    return _task_template(family, task_bodies[family])


def build_eval_py(family: str, difficulty: str) -> str:
    alpha = DIFFICULTY[difficulty]["alpha"]
    dataset_builders = {
        "numeric_kernel": """
rng = random.Random(seed)
return [[rng.uniform(-3, 3) for _ in range(36)] for _ in range(n)]
""",
        "string_batch": """
rng = random.Random(seed)
vocab = ["x", "a", "b", "c", "d"]
return [" ".join(rng.choice(vocab) for _ in range(48)) for _ in range(n)]
""",
        "graph_microkernel": """
rng = random.Random(seed)
return [[(rng.randint(0, 63), rng.randint(1, 8)) for _ in range(28)] for _ in range(n)]
""",
        "dp_operator": """
rng = random.Random(seed)
return [[rng.randint(0, 20) for _ in range(56)] for _ in range(n)]
""",
        "geometry_batch": """
rng = random.Random(seed)
return [((rng.uniform(-10, 10), rng.uniform(-10, 10)), (rng.uniform(-10, 10), rng.uniform(-10, 10))) for _ in range(n)]
""",
        "prob_sampling": """
rng = random.Random(seed)
return [[rng.random() for _ in range(20)] for _ in range(n)]
""",
        "etl_transform": """
rng = random.Random(seed)
data = []
for _ in range(n):
    row = []
    for _ in range(16):
        row.append(None if rng.random() < 0.18 else rng.uniform(-5, 5))
    data.append(row)
return data
""",
        "dsl_executor": """
rng = random.Random(seed)
ops = ["ADD", "SUB", "MUL"]
return [[f"{rng.choice(ops)}:{rng.randint(1,4)}" for _ in range(22)] for _ in range(n)]
""",
        "bitset_ops": """
rng = random.Random(seed)
return [(rng.getrandbits(48), rng.getrandbits(48)) for _ in range(n)]
""",
        "window_stats": """
rng = random.Random(seed)
return [[rng.uniform(-20, 20) for _ in range(40)] for _ in range(n)]
""",
        "trie_like_lookup": """
rng = random.Random(seed)
stems = ["pre", "pro", "post", "prime", "alpha", "beta"]
return [" ".join(rng.choice(stems) + str(rng.randint(0, 9)) for _ in range(36)) for _ in range(n)]
""",
        "schedule_sim": """
rng = random.Random(seed)
data = []
for _ in range(n):
    cur = 0
    jobs = []
    for _ in range(28):
        cur += rng.randint(0, 3)
        jobs.append((cur, rng.randint(1, 6)))
    data.append(jobs)
return data
""",
    }
    return _eval_template(family, alpha, dataset_builders[family], build_task_py(family).split("\n", 6)[6])


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

        if any(jaccard_ngrams(rec.instruction, prev.instruction) > 0.90 for prev in records):
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
    parser.add_argument("--count", type=int, default=120)
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
