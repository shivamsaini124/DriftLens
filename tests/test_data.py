
import numpy as np

from src.data import load_dataset, make_reference_and_batches


X_train, X_test, y_train, y_test = load_dataset()

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

    print(f"\nScenario: {scenario}")
    print("Reference shape:", reference.shape)
    print("Number of batches:", len(batches))

    for item in batches:
        X_batch = item["X"]
        metadata = item["metadata"]

        assert X_batch.shape == (100, 30)

        print(
            f"Batch {metadata['batch_number']:2}: "
            f"shifted={metadata['drift_injected']}, "
            f"multiplier={metadata['shift_multiplier']:.2f}, "
            f"features={metadata['shifted_features']}"
        )

# Check reproducibility.
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

print("\nAll basic checks passed!")
