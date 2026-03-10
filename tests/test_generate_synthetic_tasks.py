from src.generate_synthetic_tasks import DIFFICULTY, generate_records, to_export_dict


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
    assert "def evaluate(candidate_fn, dataset):" in payload["app_files"]["/app/eval.py"]
