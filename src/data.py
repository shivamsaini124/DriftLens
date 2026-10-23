import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


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
    *args,
    scenario: str = "stable",
    n_batches: int = 10,
    batch_size: int = 100,
    shift_strength: float = 1.0,
    shift_features: list[str] | None = None,
    shift_start: int | None = None,
    shift_duration: int | None = None,
    drift_start: int | None = None,
    drift_length: int | None = None,
    random_state: int | None = None,
    seed: int | None = None,
    **kwargs,
):
    """
    Generate reproducible incoming data batches for drift experiments.

    Supports both Person A's calling interface (X_reference, scenario=..., ...)
    and Person B's calling interface (X_train, X_test, y_test, scenario=..., ...).

    Args:
        X_reference: Reference features (typically X_train).
        *args: Optional positional arguments:
            - If arg[0] is DataFrame: (X_stream, [y_stream], [scenario])
            - If arg[0] is str: (scenario, [n_batches], [batch_size], ...)
        scenario: 'stable', 'gradual', 'sudden', or 'temporary'.
        n_batches: Number of incoming batches.
        batch_size: Rows per batch.
        shift_strength: Multiplier for feature standard deviations.
        shift_features: Features to shift (defaults to first 3 columns).
        shift_start / drift_start: 1-based index of the first shifted batch.
        shift_duration / drift_length: Number of shifted batches in temporary drift.
        random_state / seed: Seed for random number generation.

    Returns:
        reference: Reference DataFrame.
        batches: List of batch dicts containing 'X', 'y', 'batch_id', 'is_shifted',
                 'shifted_features', 'shift_multiplier', and 'metadata'.
    """
    # Parse flexible positional arguments
    X_stream = kwargs.get("X_stream", None)
    y_stream = kwargs.get("y_stream", None)

    if len(args) >= 1:
        first_arg = args[0]
        if isinstance(first_arg, (pd.DataFrame, np.ndarray)):
            X_stream = pd.DataFrame(first_arg)
            if len(args) >= 2:
                y_stream = args[1]
            if len(args) >= 3 and isinstance(args[2], str):
                scenario = args[2]
        elif isinstance(first_arg, str):
            scenario = first_arg
            if len(args) >= 2 and isinstance(args[1], int):
                n_batches = args[1]
            if len(args) >= 3 and isinstance(args[2], int):
                batch_size = args[2]
            if len(args) >= 4 and isinstance(args[3], (int, float)):
                shift_strength = float(args[3])

    # Unify alias parameter names
    eff_seed = seed if seed is not None else (random_state if random_state is not None else 42)
    eff_start = drift_start if drift_start is not None else (shift_start if shift_start is not None else 4)
    eff_duration = drift_length if drift_length is not None else (shift_duration if shift_duration is not None else 3)

    allowed_scenarios = {"stable", "gradual", "sudden", "temporary"}
    if scenario not in allowed_scenarios:
        raise ValueError(f"scenario must be one of {sorted(allowed_scenarios)}")

    if X_reference.empty:
        raise ValueError("X_reference must not be empty")

    if n_batches < 1 or batch_size < 1:
        raise ValueError("n_batches and batch_size must be positive")

    if shift_strength < 0:
        raise ValueError("shift_strength must be non-negative")

    if not 1 <= eff_start <= n_batches:
        raise ValueError("shift_start must be between 1 and n_batches")

    if eff_duration < 1:
        raise ValueError("shift_duration must be positive")

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X_reference.dtypes):
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
        raise ValueError(f"Unknown shift features: {sorted(unknown_features)}")

    # Use reference statistics to determine shift magnitude
    feature_std = reference[shift_features].std(ddof=0).fillna(0.0)

    # Determine stream source distribution
    if X_stream is not None:
        source_X = pd.DataFrame(X_stream).copy().reset_index(drop=True)
        source_y = pd.Series(y_stream).copy().reset_index(drop=True) if y_stream is not None else None
    else:
        source_X = reference
        source_y = None

    rng = np.random.default_rng(eff_seed)
    batches = []

    for batch_idx in range(n_batches):
        batch_number = batch_idx + 1  # 1-based numbering
        batch_id = batch_idx          # 0-based indexing

        # Sample rows with replacement from source
        row_indices = rng.integers(low=0, high=len(source_X), size=batch_size)
        batch = source_X.iloc[row_indices].copy().reset_index(drop=True)

        if source_y is not None:
            batch_y = source_y.iloc[row_indices].copy().reset_index(drop=True)
        else:
            batch_y = pd.Series(np.zeros(batch_size, dtype=int))

        # Determine shift multiplier for this batch
        multiplier = 0.0

        if scenario == "gradual" and batch_number >= eff_start:
            remaining_batches = n_batches - eff_start + 1
            progress = (batch_number - eff_start + 1) / remaining_batches
            multiplier = shift_strength * progress

        elif scenario == "sudden" and batch_number >= eff_start:
            multiplier = shift_strength

        elif scenario == "temporary":
            temporary_end = eff_start + eff_duration
            if eff_start <= batch_number < temporary_end:
                multiplier = shift_strength

        # Apply offset to selected features
        actually_shifted_features = []
        if multiplier > 0:
            for feature in shift_features:
                offset = multiplier * feature_std[feature]
                if offset > 0:
                    batch[feature] = batch[feature] + offset
                    actually_shifted_features.append(feature)

        is_shifted = bool(actually_shifted_features)

        batches.append(
            {
                "X": batch,
                "y": batch_y,
                "batch_id": batch_id,
                "batch_number": batch_number,
                "is_shifted": is_shifted,
                "shifted_features": actually_shifted_features,
                "shift_multiplier": float(multiplier),
                "metadata": {
                    "scenario": scenario,
                    "batch_number": batch_number,
                    "batch_id": batch_id,
                    "drift_injected": is_shifted,
                    "is_shifted": is_shifted,
                    "shifted_features": actually_shifted_features,
                    "shift_multiplier": float(multiplier),
                },
            }
        )

    return reference, batches
