from src.generate_synthetic_tasks import DIFFICULTY, FAMILIES, build_difficulty_schedule, generate_records, to_export_dict


def test_record_count_and_shape():
    records = generate_records(36, 123)
    assert len(records) == 36
    for rec in records:
        assert rec.instruction
        assert rec.validation_spec["performance"]["metric"] == "median_time_per_call"
        assert "/app/task.py" in rec.app_files
        assert "/app/eval.py" in rec.app_files


def test_difficulty_constraints_and_l5_present():
    records = generate_records(50, 1)
    assert any(rec.difficulty == "L5" for rec in records)
    for rec in records:
        alpha = float(rec.validation_spec["performance"]["pass_condition"].split("<= ")[1].split(" *")[0])
        assert alpha == DIFFICULTY[rec.difficulty]["alpha"]


def test_export_contains_app_files():
    record = generate_records(1, 42)[0]
    payload = to_export_dict(record)
    assert "app_files" in payload
    assert "def solve(data):" in payload["app_files"]["/app/task.py"]
    assert "def evaluate(candidate_fn):" in payload["app_files"]["/app/eval.py"]


def test_family_instruction_and_app_file_match():
    records = generate_records(80, 9)
    for rec in records:
        assert f"Family: {rec.family}." in rec.instruction
        assert f'FAMILY = "{rec.family}"' in rec.app_files["/app/task.py"]
        assert f'FAMILY = "{rec.family}"' in rec.app_files["/app/eval.py"]


def test_generated_code_is_valid_python_for_all_families():
    records = generate_records(120, 7)
    for family in FAMILIES:
        rec = next(r for r in records if r.family == family)
        compile(rec.app_files["/app/task.py"], f"{family}_task.py", "exec")
        compile(rec.app_files["/app/eval.py"], f"{family}_eval.py", "exec")


def test_difficulty_schedule_has_expected_mix():
    schedule = build_difficulty_schedule(30)
    assert {"L1", "L2", "L3", "L4", "L5"}.issubset(set(schedule))
