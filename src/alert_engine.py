"""Alert strategies for DriftLens (Person B).

Both strategies run on the SAME batch-level drift signal:
  * immediate  : alert whenever the current batch has drift.
  * persistent : alert only after `persistence_k` consecutive drift batches.

Aggregation rule (document this in the README):
  batch signal = True if at least `min_features` features have
  combined_feature_flag == True (corrected KS significant AND PSI > threshold).

State is a plain dict so it can live in st.session_state. It is never mutated;
update_alert_state returns a NEW dict.
"""
from __future__ import annotations

import pandas as pd


def new_alert_state() -> dict:
    """Fresh state. batch_index = number of batches processed so far."""
    return {
        "batch_index": 0,
        "consecutive_count": 0,
        "immediate_alert": False,
        "persistent_alert": False,
    }


def batch_drift_signal(feature_results: pd.DataFrame, min_features: int = 1) -> bool:
    """Collapse the feature-level table into one True/False batch signal."""
    if feature_results is None or len(feature_results) == 0:
        return False
    return int(feature_results["combined_feature_flag"].sum()) >= int(min_features)


def update_alert_state(
    batch_drift_detected: bool,
    previous_state: dict | None = None,
    persistence_k: int = 2,
) -> dict:
    """Advance the alert state by one batch.

    The consecutive counter resets to 0 as soon as a batch has no drift.
    The persistent alert stays on while the counter is >= persistence_k.
    """
    prev = previous_state if previous_state is not None else new_alert_state()
    detected = bool(batch_drift_detected)
    count = prev["consecutive_count"] + 1 if detected else 0
    return {
        "batch_index": prev["batch_index"] + 1,
        "consecutive_count": count,
        "immediate_alert": detected,
        "persistent_alert": count >= persistence_k,
    }


def run_strategies(signals: list[bool], persistence_k: int = 2) -> list[dict]:
    """Convenience: feed a whole list of batch signals, return state after each batch."""
    state = new_alert_state()
    out = []
    for s in signals:
        state = update_alert_state(s, state, persistence_k)
        out.append(state)
    return out
