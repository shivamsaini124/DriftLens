"""Evaluation of immediate vs persistent alerts (Person B).

Input: a long DataFrame, one row per (scenario, seed, batch_id), with columns
  scenario, seed, batch_id, is_shifted, immediate_alert, persistent_alert
  (optional for feature identification: flagged_features, shifted_features,
   each a '|'-joined string of feature names)

Metric definitions (copy these into the report):
  false_alert_batch_rate : alerts on NON-shifted batches / number of non-shifted batches
                           (pooled over all seeds of that scenario).
  false_alert_run_rate   : fraction of runs with >= 1 alert on a non-shifted batch.
  detection_rate         : fraction of runs (that contain injected drift) with >= 1 alert
                           in [drift_start, drift_start + eval_window - 1].
  mean_detection_delay   : mean of (first alert batch - drift_start) over DETECTED runs only.
                           Missed runs are excluded from the mean and counted in missed_runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ALERT_COLS = {"immediate": "immediate_alert", "persistent": "persistent_alert"}


def _detection_outcome(run: pd.DataFrame, col: str, window: int):
    """Return None (no drift in run), (False, nan) if missed, (True, delay) if detected."""
    shifted = run[run["is_shifted"]]
    if shifted.empty:
        return None
    start = int(shifted["batch_id"].min())
    hits = run[
        (run["batch_id"] >= start) & (run["batch_id"] < start + window) & (run[col])
    ]
    if hits.empty:
        return (False, np.nan)
    return (True, int(hits["batch_id"].min()) - start)


def evaluate_alert_strategies(results: pd.DataFrame, eval_window: int = 5) -> pd.DataFrame:
    df = results.copy()
    df["is_shifted"] = df["is_shifted"].astype(bool)
    for col in ALERT_COLS.values():
        df[col] = df[col].astype(bool)

    rows = []
    for strategy, col in ALERT_COLS.items():
        for scenario, sdf in df.groupby("scenario", sort=False):
            seeds = sdf["seed"].unique()
            clean = sdf[~sdf["is_shifted"]]

            fa_batch = clean[col].mean() if len(clean) else np.nan
            fa_run = (
                clean.groupby("seed")[col].any().reindex(seeds, fill_value=False).mean()
            )

            outcomes = [
                _detection_outcome(run.sort_values("batch_id"), col, eval_window)
                for _, run in sdf.groupby("seed")
            ]
            outcomes = [o for o in outcomes if o is not None]
            if outcomes:
                detected = [o for o in outcomes if o[0]]
                det_rate = len(detected) / len(outcomes)
                delay = float(np.mean([o[1] for o in detected])) if detected else np.nan
                missed = len(outcomes) - len(detected)
            else:
                det_rate, delay, missed = np.nan, np.nan, 0

            rows.append(
                {
                    "strategy": strategy,
                    "scenario": scenario,
                    "n_runs": len(seeds),
                    "false_alert_batch_rate": fa_batch,
                    "false_alert_run_rate": fa_run,
                    "detection_rate": det_rate,
                    "mean_detection_delay": delay,
                    "missed_runs": missed,
                }
            )
    return pd.DataFrame(rows)


def _to_set(value) -> set:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    return {v for v in str(value).split("|") if v}


def feature_identification(results: pd.DataFrame) -> pd.DataFrame:
    """On shifted batches: how many truly shifted features were flagged (recall),
    and how many flagged features were truly shifted (precision)."""
    sh = results[results["is_shifted"].astype(bool)]
    recs = []
    for _, r in sh.iterrows():
        truth, flagged = _to_set(r["shifted_features"]), _to_set(r["flagged_features"])
        hit = len(truth & flagged)
        recs.append(
            {
                "scenario": r["scenario"],
                "recall": hit / len(truth) if truth else np.nan,
                "precision": hit / len(flagged) if flagged else np.nan,
            }
        )
    if not recs:
        return pd.DataFrame(columns=["scenario", "recall", "precision"])
    return pd.DataFrame(recs).groupby("scenario", sort=False).mean().reset_index()
