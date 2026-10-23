import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.alert_engine import (
    batch_drift_signal,
    new_alert_state,
    run_strategies,
    update_alert_state,
)
from src.evaluation import evaluate_alert_strategies, feature_identification


def test_immediate_follows_signal():
    states = run_strategies([False, True, False, True, True])
    assert [s["immediate_alert"] for s in states] == [False, True, False, True, True]


def test_persistent_needs_two_consecutive():
    states = run_strategies([False, True, True, True])
    assert [s["persistent_alert"] for s in states] == [False, False, True, True]


def test_single_spike_never_triggers_persistent():
    states = run_strategies([False, True, False, True, False])
    assert not any(s["persistent_alert"] for s in states)


def test_counter_resets_after_non_drift_batch():
    states = run_strategies([True, True, False, True])
    assert states[2]["consecutive_count"] == 0
    assert states[3]["consecutive_count"] == 1
    assert states[3]["persistent_alert"] is False


def test_temporary_drift_leaves_counter_inactive():
    states = run_strategies([False, True, True, False, False, False])
    assert [s["persistent_alert"] for s in states] == [False, False, True, False, False, False]
    assert states[-1]["consecutive_count"] == 0


def test_state_is_not_mutated_and_batch_index_advances():
    s0 = new_alert_state()
    s1 = update_alert_state(True, s0)
    assert s0["consecutive_count"] == 0 and s0["batch_index"] == 0
    assert s1["batch_index"] == 1


def test_custom_persistence_k():
    states = run_strategies([True, True, True], persistence_k=3)
    assert [s["persistent_alert"] for s in states] == [False, False, True]


def test_batch_drift_signal_min_features():
    df = pd.DataFrame({"feature": ["a", "b", "c"], "combined_feature_flag": [True, False, True]})
    assert batch_drift_signal(df, min_features=2) is True
    assert batch_drift_signal(df, min_features=3) is False
    assert batch_drift_signal(df.iloc[0:0]) is False


def _toy_results():
    n = 10
    flags_imm = [False, False, True, False, False, True, True, True, False, False]
    # shift injected from batch 4 onward
    shifted = [b >= 4 for b in range(n)]
    states = run_strategies(flags_imm)
    return pd.DataFrame(
        {
            "scenario": "sudden",
            "seed": 0,
            "batch_id": range(n),
            "is_shifted": shifted,
            "immediate_alert": [s["immediate_alert"] for s in states],
            "persistent_alert": [s["persistent_alert"] for s in states],
        }
    )


def test_evaluation_metrics_on_toy_run():
    summary = evaluate_alert_strategies(_toy_results(), eval_window=5)
    imm = summary[summary.strategy == "immediate"].iloc[0]
    per = summary[summary.strategy == "persistent"].iloc[0]

    # clean batches are 0..3; immediate has one false alert (batch 2)
    assert imm["false_alert_batch_rate"] == pytest.approx(0.25)
    assert imm["false_alert_run_rate"] == 1.0
    assert per["false_alert_batch_rate"] == 0.0
    assert per["false_alert_run_rate"] == 0.0

    # drift starts at 4; immediate first alert at 5 (delay 1), persistent at 6 (delay 2)
    assert imm["detection_rate"] == 1.0 and imm["mean_detection_delay"] == 1
    assert per["detection_rate"] == 1.0 and per["mean_detection_delay"] == 2


def test_missed_detection_is_counted_and_excluded_from_delay():
    df = _toy_results()
    df["immediate_alert"] = False
    df["persistent_alert"] = False
    summary = evaluate_alert_strategies(df, eval_window=5)
    row = summary[summary.strategy == "immediate"].iloc[0]
    assert row["detection_rate"] == 0.0
    assert row["missed_runs"] == 1
    assert np.isnan(row["mean_detection_delay"])


def test_stable_scenario_has_no_detection_metrics():
    df = _toy_results()
    df["scenario"] = "stable"
    df["is_shifted"] = False
    summary = evaluate_alert_strategies(df)
    assert summary["detection_rate"].isna().all()


def test_feature_identification():
    df = pd.DataFrame(
        {
            "scenario": ["sudden", "sudden"],
            "is_shifted": [True, True],
            "shifted_features": ["a|b", "a|b"],
            "flagged_features": ["a|b", "a|c"],
        }
    )
    out = feature_identification(df).iloc[0]
    assert out["recall"] == pytest.approx(0.75)
    assert out["precision"] == pytest.approx(0.75)


def test_same_seed_is_reproducible():
    """Integration check; needs Person A's modules."""
    pytest.importorskip("src.data")
    pytest.importorskip("src.drift_detectors")
    from src.data import load_dataset, make_reference_and_batches
    from src.drift_detectors import detect_drift

    X_train, X_test, _, y_test = load_dataset()

    def run():
        ref, batches = make_reference_and_batches(
            X_train, X_test, y_test, scenario="sudden", n_batches=6, batch_size=60,
            shift_strength=1.0, seed=7, drift_start=3, drift_length=3,
        )
        return [detect_drift(ref, b["X"])["psi"].round(8).tolist() for b in batches]

    assert run() == run()
