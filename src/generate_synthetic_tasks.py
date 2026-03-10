#!/usr/bin/env python3
"""Generate benchmark-equivalent LLM agent training tasks.

Each output record contains:
- instruction
- validation_spec
- app_files: {"/app/task.py": ..., "/app/eval.py": ...}

The generator enforces:
- family/difficulty diversity
- anti-contamination keyword filtering
- basic semantic deduplication by n-gram Jaccard
"""

from __future__ import annotations

import argparse
import json
import random
import re
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
        "goal": "Optimize a numerical kernel function while preserving tolerance-bound outputs.",
        "io": "Input is a batch of float64 arrays; output is per-item scalar statistics.",
        "pitfalls": ["catastrophic cancellation", "branch-heavy loops", "temporary allocations"],
        "kind": "numeric",
    },
    "string_batch": {
        "goal": "Accelerate batch string/byte processing with exact matching semantics.",
        "io": "Input is a list of strings and tokens; output is per-token counts.",
        "pitfalls": ["quadratic scanning", "encoding edge-cases", "large intermediate objects"],
        "kind": "discrete",
    },
    "graph_microkernel": {
        "goal": "Optimize a small graph propagation/counting micro-kernel.",
        "io": "Input is compact adjacency list; output is per-node weighted degree proxy.",
        "pitfalls": ["sparse-dense conversion", "index bounds", "duplicate edge handling"],
        "kind": "numeric",
    },
    "dp_operator": {
        "goal": "Improve throughput of a bounded dynamic-programming operator.",
        "io": "Input is short integer sequences; output is longest non-decreasing run length.",
        "pitfalls": ["state explosion", "poor cache locality", "incorrect base-case handling"],
        "kind": "discrete",
    },
    "geometry_batch": {
        "goal": "Optimize batched geometric predicate computation.",
        "io": "Input is point pairs; output is Euclidean distance summaries.",
        "pitfalls": ["floating-point tolerance", "degenerate geometry", "O(n^2) checks"],
        "kind": "numeric",
    },
    "prob_sampling": {
        "goal": "Optimize a probabilistic normalization/evaluation step.",
        "io": "Input is non-negative score vectors; output is normalized vectors.",
        "pitfalls": ["normalization drift", "seed misuse", "biased sampling shortcuts"],
        "kind": "numeric",
    },
    "etl_transform": {
        "goal": "Accelerate ETL-style normalization and feature transformation.",
        "io": "Input is numeric table with missing values; output is z-score-like normalization.",
        "pitfalls": ["datetime parsing cost", "null handling", "type instability"],
        "kind": "numeric",
    },
    "dsl_executor": {
        "goal": "Optimize execution of a tiny fixed-grammar DSL interpreter core.",
        "io": "Input is op string list; output is accumulator trajectory checksum.",
        "pitfalls": ["dispatch overhead", "state mutation bugs", "unchecked opcode paths"],
        "kind": "discrete",
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
        f"Goal: {profile['goal']} "
        f"Difficulty: {difficulty}. "
        f"I/O contract: {profile['io']} "
        "Keep public function signatures unchanged. You may modify only /app/task.py. "
        "Do not change /app/eval.py or test harness files. "
        "Your solution must pass correctness checks and beat the reference implementation "
        "on median runtime. "
        f"Pay special attention to: {pitfalls}."
    )


def build_validation_spec(family: str, difficulty: str, rng: random.Random) -> Dict:
    d = DIFFICULTY[difficulty]
    return {
        "correctness": {
            "deterministic_tests": 24 + 4 * (difficulty in {"L3", "L4"}),
            "randomized_tests": 32 + 8 * (difficulty in {"L3", "L4"}),
            "numeric_tolerance": {
                "abs_err": 1e-8 if FAMILIES[family]["kind"] == "numeric" else 0.0,
                "rel_err": 1e-6 if FAMILIES[family]["kind"] == "numeric" else 0.0,
            },
            "property_checks": [
                "shape and dtype invariants",
                "edge-case behavior on empty/minimal inputs",
                "stability under repeated calls with same seed",
            ],
        },
        "performance": {
            "metric": "median_time_per_call",
            "warmup_calls": 20,
            "timed_calls": 120,
            "pass_condition": f"candidate_median <= {d['alpha']} * reference_median",
            "max_single_case_timeout_ms": 200,
        },
        "robustness": {
            "seeds": [rng.randint(1, 10_000_000) for _ in range(d["seeds"])],
            "distribution_shift_tests": True,
            "hidden_tests": d["hidden_tests"],
        },
        "anti_cheat": {
            "disallow_eval_introspection": True,
            "shuffle_input_order": True,
            "randomize_lengths": True,
        },
    }


def build_task_py(family: str) -> str:
    return f'''"""Auto-generated starter for family: {family}."""

from __future__ import annotations


def solve(data):
    """TODO: optimize this implementation in-place.

    Constraints:
    - keep function signature unchanged
    - keep return type stable
    """
    # Slow baseline (intentionally loop-heavy)
    out = []
    for item in data:
        if isinstance(item, (list, tuple)):
            s = 0.0
            for x in item:
                s += float(x)
            out.append(s)
        elif isinstance(item, str):
            out.append(float(len(item.split())))
        else:
            out.append(float(item) if item is not None else 0.0)
    return out
'''


def build_eval_py(difficulty: str) -> str:
    alpha = DIFFICULTY[difficulty]["alpha"]
    return f'''"""Auto-generated evaluator template."""

from __future__ import annotations

import statistics
import time


def reference_solve(data):
    # Deliberately simple and usually slower than an optimized vectorized solution.
    out = []
    for item in data:
        if isinstance(item, (list, tuple)):
            total = 0.0
            for x in item:
                total += float(x)
            out.append(total)
        elif isinstance(item, str):
            out.append(float(len(item.split())))
        else:
            out.append(float(item) if item is not None else 0.0)
    return out


def check_correctness(candidate_fn, dataset):
    ref = reference_solve(dataset)
    got = candidate_fn(dataset)
    if len(ref) != len(got):
        return False
    return all(abs(a - b) <= 1e-8 for a, b in zip(ref, got))


def median_runtime(fn, dataset, repeats=80):
    costs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(dataset)
        costs.append(time.perf_counter() - t0)
    return statistics.median(costs)


def evaluate(candidate_fn, dataset):
    ok = check_correctness(candidate_fn, dataset)
    if not ok:
        return {{"passed": False, "reason": "correctness"}}

    c = median_runtime(candidate_fn, dataset)
    r = median_runtime(reference_solve, dataset)
    perf_ok = c <= {alpha} * r
    return {{"passed": bool(perf_ok), "candidate": c, "reference": r, "alpha": {alpha}}}
'''


def build_app_files(family: str, difficulty: str) -> Dict[str, str]:
    return {
        "/app/task.py": build_task_py(family),
        "/app/eval.py": build_eval_py(difficulty),
    }


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
