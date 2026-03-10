from src.generate_synthetic_tasks import DIFFICULTY, FAMILIES, generate_records, to_export_dict


def test_record_count_and_shape():
    records = generate_records(25, 123)
    assert len(records) == 25
    for rec in records:
        assert rec.instruction
        assert rec.validation_spec["performance"]["metric"] == "median_time_per_call"
        assert "/app/task.py" in rec.app_files
        assert "/app/eval.py" in rec.app_files


def test_difficulty_constraints():
    records = generate_records(20, 1)
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
    records = generate_records(32, 9)
    for rec in records:
        assert f"Family: {rec.family}." in rec.instruction
        assert f'FAMILY = "{rec.family}"' in rec.app_files["/app/task.py"]
        assert f'FAMILY = "{rec.family}"' in rec.app_files["/app/eval.py"]


def test_generated_code_is_valid_python_for_all_families():
    records = generate_records(64, 7)
    for family in FAMILIES:
        rec = next(r for r in records if r.family == family)
        compile(rec.app_files["/app/task.py"], f"{family}_task.py", "exec")
        compile(rec.app_files["/app/eval.py"], f"{family}_eval.py", "exec")
