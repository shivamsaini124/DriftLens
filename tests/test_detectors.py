
from src.data import load_dataset, make_reference_and_batches
from src.drift_detectors import detect_drift

X_train, X_test, y_train, y_test = load_dataset()

for scenario in ["stable", "gradual", "sudden", "temporary"]:
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

    print(f"\n{'=' * 60}")
    print(f"SCENARIO: {scenario.upper()}")
    print("=" * 60)

    for item in batches:
        current = item["X"]
        metadata = item["metadata"]

        results = detect_drift(
            reference=reference,
            current=current,
            alpha=0.05,
            bins=10,
            psi_threshold=0.10,
        )

        flagged = results.loc[
            results["combined_feature_flag"], "feature"
        ].tolist()

        print(
            f"Batch {metadata['batch_number']:2} | "
            f"Injected shift: {metadata['drift_injected']} | "
            f"Flagged features: {len(flagged)}"
        )

        if flagged:
            print("  ", ", ".join(flagged))

        if metadata["batch_number"] == 4:
            print("\nTop features in batch 4:")
            print(
                results.sort_values("psi", ascending=False)
                .head(5)[
                    [
                        "feature",
                        "ks_statistic",
                        "ks_pvalue",
                        "ks_adjusted_pvalue",
                        "psi",
                        "combined_feature_flag",
                    ]
                ]
                .to_string(index=False)
            )
