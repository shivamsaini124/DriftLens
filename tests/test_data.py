import numpy as np
from src.data import load_dataset, make_reference_and_batches


def test_data_load_shapes():
    X_train, X_test, y_train, y_test = load_dataset()
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    assert X_train.shape[1] == 30


def test_data_generation_scenarios():
    X_train, _, _, _ = load_dataset()
    scenarios = ["stable", "gradual", "sudden", "temporary"]

    for scenario in scenarios:
        reference, batches = make_reference_and_batches(
            X_reference=X_train,
            scenario=scenario,
            n_batches=10,
            batch_size=100,
            shift_strength=1.0,
            shift_start=4,
            shift_duration=3,
            random_state=42,
        )

        assert reference.shape == X_train.shape
        assert len(batches) == 10

        for item in batches:
            X_batch = item["X"]
            metadata = item["metadata"]
            assert X_batch.shape == (100, 30)
            assert "batch_number" in metadata
            assert "drift_injected" in metadata


def test_data_reproducibility():
    X_train, _, _, _ = load_dataset()

    _, first_run = make_reference_and_batches(
        X_train, scenario="sudden", random_state=123
    )
    _, second_run = make_reference_and_batches(
        X_train, scenario="sudden", random_state=123
    )

    for first, second in zip(first_run, second_run):
        assert np.array_equal(
            first["X"].to_numpy(),
            second["X"].to_numpy(),
        )
        assert first["metadata"] == second["metadata"]
