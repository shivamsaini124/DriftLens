
import numpy as np
import pandas as pd
import pytest

from src.data import load_dataset, make_reference_and_batches
from src.drift_detectors import (
    calculate_ks,
    calculate_psi,
    detect_drift,
)


@pytest.fixture
def reference():
    X_train, _, _, _ = load_dataset()
    return X_train


def test_detector_returns_expected_columns(reference):
    _, batches = make_reference_and_batches(
        reference,
        scenario="sudden",
        n_batches=5,
        batch_size=100,
        shift_strength=1.0,
        shift_start=3,
        random_state=42,
    )

    result = detect_drift(reference, batches[2]["X"])

    expected = {
        "feature",
        "ks_statistic",
        "ks_pvalue",
        "ks_adjusted_pvalue",
        "ks_significant",
        "psi",
        "psi_flag",
        "combined_feature_flag",
    }

    assert expected.issubset(result.columns)
    assert len(result) == reference.shape[1]


def test_identical_data_has_low_psi(reference):
    result = calculate_psi(reference, reference.copy())

    assert np.isfinite(result["psi"]).all()
    assert (result["psi"] < 1e-8).all()


def test_psi_remains_finite_with_zero_count_bins():
    reference_data = pd.DataFrame({
        "feature": np.arange(100, dtype=float)
    })

    current_data = pd.DataFrame({
        "feature": np.full(100, 50.0)
    })

    result = calculate_psi(
        reference_data,
        current_data,
        bins=10,
    )

    assert np.isfinite(result["psi"]).all()


def test_strong_shift_is_detectable():
    rng = np.random.default_rng(42)

    reference_data = pd.DataFrame({
        "feature": rng.normal(0, 1, 1000)
    })

    shifted_data = pd.DataFrame({
        "feature": rng.normal(3, 1, 1000)
    })

    result = detect_drift(
        reference_data,
        shifted_data,
        alpha=0.05,
        bins=10,
        psi_threshold=0.10,
    )

    assert bool(result.loc[0, "ks_significant"])
    assert bool(result.loc[0, "psi_flag"])
    assert bool(result.loc[0, "combined_feature_flag"])


def test_ks_adjusted_pvalues_are_valid(reference):
    _, batches = make_reference_and_batches(
        reference,
        scenario="stable",
        n_batches=2,
        batch_size=100,
        shift_start=2,
        random_state=42,
    )

    result = calculate_ks(reference, batches[0]["X"])

    assert result["ks_pvalue"].between(0, 1).all()
    assert result["ks_adjusted_pvalue"].between(0, 1).all()


def test_invalid_inputs_raise_errors():
    reference_data = pd.DataFrame({
        "feature": [1.0, 2.0, 3.0]
    })

    current_data = pd.DataFrame({
        "feature": [1.0, np.nan, 3.0]
    })

    with pytest.raises(ValueError):
        calculate_ks(reference_data, current_data)

    with pytest.raises(ValueError):
        calculate_psi(
            pd.DataFrame({"feature": [1.0, 2.0]}),
            pd.DataFrame({"feature": [1.0, 2.0]}),
            bins=0,
        )


def test_data_simulator_is_reproducible(reference):
    _, first_run = make_reference_and_batches(
        reference,
        scenario="sudden",
        n_batches=6,
        batch_size=100,
        shift_start=3,
        random_state=123,
    )

    _, second_run = make_reference_and_batches(
        reference,
        scenario="sudden",
        n_batches=6,
        batch_size=100,
        shift_start=3,
        random_state=123,
    )

    for first, second in zip(first_run, second_run):
        np.testing.assert_array_equal(
            first["X"].to_numpy(),
            second["X"].to_numpy(),
        )
        assert first["metadata"] == second["metadata"]


def test_temporary_shift_returns_to_stable(reference):
    _, batches = make_reference_and_batches(
        reference,
        scenario="temporary",
        n_batches=8,
        batch_size=100,
        shift_strength=1.0,
        shift_start=3,
        shift_duration=2,
        random_state=42,
    )

    assert batches[2]["metadata"]["drift_injected"] is True
    assert batches[3]["metadata"]["drift_injected"] is True
    assert batches[4]["metadata"]["drift_injected"] is False
    assert batches[4]["metadata"]["shift_multiplier"] == 0.0


def test_detector_rejects_mismatched_features():
    reference_data = pd.DataFrame({
        "feature_a": [1.0, 2.0, 3.0]
    })

    current_data = pd.DataFrame({
        "feature_b": [1.0, 2.0, 3.0]
    })

    with pytest.raises(ValueError):
        detect_drift(reference_data, current_data)
