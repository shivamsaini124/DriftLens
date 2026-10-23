
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd


def load_dataset(test_size: float = 0.2, random_state: int = 42):
    """
    Load the Breast Cancer Wisconsin dataset and split it into
    training and held-out test sets.

    Returns:
        X_train: Training features.
        X_test: Held-out test features.
        y_train: Training labels.
        y_test: Held-out test labels.
    """
    dataset = load_breast_cancer(as_frame=True)

    X = dataset.data
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test

def make_reference_and_batches(
    X_reference: pd.DataFrame,
    scenario: str = "stable",
    n_batches: int = 10,
    batch_size: int = 100,
    shift_strength: float = 1.0,
    shift_features: list[str] | None = None,
    shift_start: int = 4,
    shift_duration: int = 3,
    random_state: int = 42,
):
    """
    Generate reproducible incoming data batches for drift experiments.

    Args:
        X_reference: Reference features, normally X_train.
        scenario: stable, gradual, sudden, or temporary.
        n_batches: Number of incoming batches.
        batch_size: Rows per batch.
        shift_strength: Maximum shift measured in reference standard
            deviations.
        shift_features: Feature names to shift. Defaults to the first
            three features.
        shift_start: First shifted batch, using 1-based numbering.
        shift_duration: Number of shifted batches for temporary drift.
        random_state: Random seed.

    Returns:
        reference: Copy of the reference feature DataFrame.
        batches: List of dictionaries containing batch data and metadata.
    """
    allowed_scenarios = {"stable", "gradual", "sudden", "temporary"}

    if scenario not in allowed_scenarios:
        raise ValueError(
            f"scenario must be one of {sorted(allowed_scenarios)}"
        )

    if X_reference.empty:
        raise ValueError("X_reference must not be empty")

    if n_batches < 1 or batch_size < 1:
        raise ValueError("n_batches and batch_size must be positive")

    if shift_strength < 0:
        raise ValueError("shift_strength must be non-negative")

    if not 1 <= shift_start <= n_batches:
        raise ValueError("shift_start must be between 1 and n_batches")

    if shift_duration < 1:
        raise ValueError("shift_duration must be positive")

    if not all(pd.api.types.is_numeric_dtype(dtype)
               for dtype in X_reference.dtypes):
        raise ValueError("All reference features must be numeric")

    if not np.isfinite(X_reference.to_numpy(dtype=float)).all():
        raise ValueError("Reference features must contain finite values")

    reference = X_reference.copy().reset_index(drop=True)

    if shift_features is None:
        shift_features = list(reference.columns[:3])
    else:
        shift_features = list(shift_features)

    if not shift_features:
        raise ValueError("At least one shift feature is required")

    unknown_features = set(shift_features) - set(reference.columns)
    if unknown_features:
        raise ValueError(
            f"Unknown shift features: {sorted(unknown_features)}"
        )

    # Use reference statistics to determine the shift magnitude.
    feature_std = reference[shift_features].std(ddof=0)

    # A zero-variance feature cannot be shifted using a std-based offset.
    feature_std = feature_std.fillna(0.0)

    rng = np.random.default_rng(random_state)
    batches = []

    for batch_number in range(1, n_batches + 1):
        # Sample rows with replacement from the reference features.
        row_indices = rng.integers(
            low=0,
            high=len(reference),
            size=batch_size,
        )

        batch = reference.iloc[row_indices].copy().reset_index(drop=True)

        # Determine the shift multiplier for this batch.
        multiplier = 0.0

        if scenario == "gradual" and batch_number >= shift_start:
            remaining_batches = n_batches - shift_start + 1
            progress = (
                batch_number - shift_start + 1
            ) / remaining_batches
            multiplier = shift_strength * progress

        elif scenario == "sudden" and batch_number >= shift_start:
            multiplier = shift_strength

        elif scenario == "temporary":
            temporary_end = shift_start + shift_duration

            if shift_start <= batch_number < temporary_end:
                multiplier = shift_strength

        # Apply a constant offset to selected features.
        actually_shifted_features = []

        if multiplier > 0:
            for feature in shift_features:
                offset = multiplier * feature_std[feature]

                if offset > 0:
                    batch[feature] = batch[feature] + offset
                    actually_shifted_features.append(feature)

        batches.append(
            {
                "X": batch,
                "metadata": {
                    "scenario": scenario,
                    "batch_number": batch_number,
                    "drift_injected": bool(actually_shifted_features),
                    "shifted_features": actually_shifted_features,
                    "shift_multiplier": multiplier,
                },
            }
        )

    return reference, batches
