
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp
from statsmodels.stats.multitest import multipletests


def _validate_inputs(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> None:
    """Validate the reference and incoming feature DataFrames."""
    if not isinstance(reference, pd.DataFrame):
        raise TypeError("reference must be a pandas DataFrame")

    if not isinstance(current, pd.DataFrame):
        raise TypeError("current must be a pandas DataFrame")

    if reference.empty or current.empty:
        raise ValueError("Reference and current data must not be empty")

    if not reference.columns.is_unique or not current.columns.is_unique:
        raise ValueError("Feature column names must be unique")

    if set(reference.columns) != set(current.columns):
        raise ValueError(
            "Reference and current data must contain the same features"
        )

    if not all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in reference.dtypes
    ):
        raise ValueError("Reference features must be numeric")

    if not all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in current.dtypes
    ):
        raise ValueError("Current features must be numeric")

    # Keep feature order consistent with the reference.
    current_columns = list(reference.columns)

    if list(current.columns) != current_columns:
        # Column order does not affect the result, but align it explicitly.
        current = current[current_columns]

    for name, frame in (
        ("reference", reference),
        ("current", current),
    ):
        values = frame.to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"{name} data must not contain NaN or infinite values"
            )


def calculate_ks(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Run a two-sample KS test for each feature and apply
    Benjamini-Hochberg false discovery rate correction.

    Returns one row per feature.
    """
    _validate_inputs(reference, current)

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    results = []

    for feature in reference.columns:
        statistic, pvalue = ks_2samp(
            reference[feature].to_numpy(dtype=float),
            current[feature].to_numpy(dtype=float),
            alternative="two-sided",
            method="auto",
        )

        results.append(
            {
                "feature": feature,
                "ks_statistic": float(statistic),
                "ks_pvalue": float(pvalue),
            }
        )

    result = pd.DataFrame(results)

    # Correct across all feature-wise KS tests.
    reject, adjusted_pvalues, _, _ = multipletests(
        result["ks_pvalue"].to_numpy(),
        alpha=alpha,
        method="fdr_bh",
    )

    result["ks_adjusted_pvalue"] = adjusted_pvalues
    result["ks_significant"] = reject

    return result


def _reference_bin_edges(
    reference_values: np.ndarray,
    bins: int,
) -> np.ndarray:
    """
    Create reusable PSI bin edges from reference quantiles.

    Infinite outer edges ensure incoming values outside the
    reference range are still counted.
    """
    if bins < 1:
        raise ValueError("bins must be at least 1")

    quantiles = np.linspace(0, 1, bins + 1)

    reference_quantiles = np.quantile(
        reference_values,
        quantiles,
    )

    # Duplicate quantiles can occur with discrete or constant data.
    internal_edges = np.unique(reference_quantiles[1:-1])

    return np.concatenate(
        (
            [-np.inf],
            internal_edges,
            [np.inf],
        )
    )


def _calculate_feature_psi(
    reference_values: np.ndarray,
    current_values: np.ndarray,
    bins: int,
    epsilon: float,
) -> float:
    """Calculate PSI for one feature using reference-derived bins."""
    edges = _reference_bin_edges(reference_values, bins)

    reference_counts, _ = np.histogram(
        reference_values,
        bins=edges,
    )

    current_counts, _ = np.histogram(
        current_values,
        bins=edges,
    )

    # Smooth zero-count bins, then normalize the proportions.
    reference_proportions = (
        reference_counts.astype(float) + epsilon
    )
    current_proportions = current_counts.astype(float) + epsilon

    reference_proportions /= reference_proportions.sum()
    current_proportions /= current_proportions.sum()

    psi = np.sum(
        (current_proportions - reference_proportions)
        * np.log(current_proportions / reference_proportions)
    )

    return float(psi)


def calculate_psi(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    bins: int = 10,
    threshold: float = 0.10,
    epsilon: float = 1e-6,
) -> pd.DataFrame:
    """
    Calculate feature-level PSI with fixed reference-derived
    binning for the given reference dataset.

    Returns one row per feature.
    """
    _validate_inputs(reference, current)

    if bins < 1:
        raise ValueError("bins must be at least 1")

    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")

    results = []

    for feature in reference.columns:
        reference_values = reference[feature].to_numpy(dtype=float)
        current_values = current[feature].to_numpy(dtype=float)

        psi = _calculate_feature_psi(
            reference_values=reference_values,
            current_values=current_values,
            bins=bins,
            epsilon=epsilon,
        )

        results.append(
            {
                "feature": feature,
                "psi": psi,
                "psi_flag": psi >= threshold,
            }
        )

    return pd.DataFrame(results)


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    alpha: float = 0.05,
    bins: int = 10,
    psi_threshold: float = 0.10,
) -> pd.DataFrame:
    """
    Combine feature-level KS and PSI results.

    A feature is flagged only when its adjusted KS p-value
    is significant AND its PSI reaches the configured threshold.
    """
    ks_results = calculate_ks(
        reference=reference,
        current=current,
        alpha=alpha,
    )

    psi_results = calculate_psi(
        reference=reference,
        current=current,
        bins=bins,
        threshold=psi_threshold,
    )

    results = ks_results.merge(
        psi_results,
        on="feature",
        how="inner",
        validate="one_to_one",
    )

    results["combined_feature_flag"] = (
        results["ks_significant"] & results["psi_flag"]
    )

    return results[
        [
            "feature",
            "ks_statistic",
            "ks_pvalue",
            "ks_adjusted_pvalue",
            "ks_significant",
            "psi",
            "psi_flag",
            "combined_feature_flag",
        ]
    ]
