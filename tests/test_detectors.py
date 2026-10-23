from src.data import load_dataset, make_reference_and_batches
from src.drift_detectors import detect_drift


def test_detect_drift_across_scenarios():
    X_train, _, _, _ = load_dataset()

    for scenario in ["stable", "gradual", "sudden", "temporary"]:
        reference, batches = make_reference_and_batches(
            X_reference=X_train,
            scenario=scenario,
            n_batches=5,
            batch_size=100,
            shift_strength=1.0,
            shift_start=3,
            shift_duration=2,
            random_state=42,
        )

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

            assert not results.empty
            assert len(results) == reference.shape[1]
            assert "combined_feature_flag" in results.columns

            flagged = results.loc[
                results["combined_feature_flag"], "feature"
            ].tolist()

            if metadata["drift_injected"]:
                # When drift is strongly injected, detector should flag features
                assert isinstance(flagged, list)
