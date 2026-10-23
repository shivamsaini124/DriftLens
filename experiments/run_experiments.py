"""Run all scenarios over many seeds through both alert strategies.

Usage (from the project root):
    python experiments/run_experiments.py                      # validation seeds 0..19
    python experiments/run_experiments.py --seed-start 1000    # independent FINAL seeds

Outputs (results/):
    batch_results.csv   one row per scenario/seed/batch
    summary.csv         immediate vs persistent metrics
    feature_id.csv      feature identification precision/recall
    performance.csv     model metrics on shifted vs non-shifted batches
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.alert_engine import batch_drift_signal, new_alert_state, update_alert_state  # noqa: E402
from src.data import load_dataset, make_reference_and_batches  # noqa: E402
from src.drift_detectors import detect_drift  # noqa: E402
from src.evaluation import evaluate_alert_strategies, feature_identification  # noqa: E402
from src.model import evaluate_model, train_model  # noqa: E402

SCENARIOS = ["stable", "gradual", "sudden", "temporary"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--n-batches", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--shift-strength", type=float, default=1.0)
    p.add_argument("--drift-start", type=int, default=8)
    p.add_argument("--drift-length", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--psi-threshold", type=float, default=0.25)
    p.add_argument("--min-features", type=int, default=1)
    p.add_argument("--persistence-k", type=int, default=2)
    p.add_argument("--eval-window", type=int, default=5)
    p.add_argument("--out", default=str(ROOT / "results"))
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_dataset()
    model = train_model(X_train, y_train)
    print("Baseline metrics:", evaluate_model(model, X_test, y_test))

    rows = []
    for scenario in SCENARIOS:
        for seed in range(a.seed_start, a.seed_start + a.seeds):
            ref, batches = make_reference_and_batches(
                X_train, X_test, y_test,
                scenario=scenario, n_batches=a.n_batches, batch_size=a.batch_size,
                shift_strength=a.shift_strength, seed=seed,
                drift_start=a.drift_start, drift_length=a.drift_length,
            )
            state = new_alert_state()
            for b in batches:
                # Labels (b["y"]) are NOT passed to the detector.
                fr = detect_drift(ref, b["X"], alpha=a.alpha, psi_threshold=a.psi_threshold)
                signal = batch_drift_signal(fr, a.min_features)
                state = update_alert_state(signal, state, a.persistence_k)
                flagged = fr.loc[fr["combined_feature_flag"], "feature"].tolist()
                m = evaluate_model(model, b["X"], b["y"])
                rows.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "batch_id": b["batch_id"],
                        "is_shifted": bool(b["is_shifted"]),
                        "n_flagged": len(flagged),
                        "flagged_features": "|".join(flagged),
                        "shifted_features": "|".join(b["shifted_features"]),
                        "immediate_alert": state["immediate_alert"],
                        "persistent_alert": state["persistent_alert"],
                        **{f"model_{k}": v for k, v in m.items()},
                    }
                )

    results = pd.DataFrame(rows)
    summary = evaluate_alert_strategies(results, eval_window=a.eval_window)
    feat_id = feature_identification(results)
    perf = (
        results.groupby(["scenario", "is_shifted"])[
            ["model_accuracy", "model_precision", "model_recall", "model_f1"]
        ]
        .mean()
        .reset_index()
    )

    results.to_csv(out / "batch_results.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    feat_id.to_csv(out / "feature_id.csv", index=False)
    perf.to_csv(out / "performance.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== Immediate vs persistent ===")
    print(summary.round(3).to_string(index=False))
    print("\n=== Feature identification ===")
    print(feat_id.round(3).to_string(index=False))
    print(f"\nSaved CSVs to {out}")


if __name__ == "__main__":
    main()
