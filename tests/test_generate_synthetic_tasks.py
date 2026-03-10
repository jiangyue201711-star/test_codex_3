from src.generate_synthetic_tasks import DIFFICULTY, generate_records


def test_record_count_and_shape():
    records = generate_records(25, 123)
    assert len(records) == 25
    for rec in records:
        assert rec.instruction
        assert rec.validation_spec["performance"]["metric"] == "median_time_per_call"


def test_difficulty_constraints():
    records = generate_records(20, 1)
    for rec in records:
        alpha = float(rec.validation_spec["performance"]["pass_condition"].split("<= ")[1].split(" *")[0])
        assert alpha == DIFFICULTY[rec.difficulty]["alpha"]
