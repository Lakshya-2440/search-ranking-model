"""
Interaction-level features.
CTR, dwell time, skip rate, position bias correction.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def extract_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract interaction features from aggregated click data.

    Features computed:
        - ctr: Click-through rate (already in aggregated data)
        - avg_dwell_time: Average dwell time on clicked docs
        - skip_rate: Fraction of impressions where doc was skipped
        - position_bias_correction: IPW weight (already computed)
        - historical_ctr: Smoothed CTR with Bayesian prior
        - click_entropy: Entropy of click distribution for query
        - dwell_time_norm: Normalized dwell time relative to query average

    Args:
        df: Aggregated DataFrame with ctr, avg_dwell, skip_rate columns

    Returns:
        DataFrame with interaction features added
    """
    logger.info("Extracting interaction features...")
    result = df.copy()

    # CTR already exists from aggregation
    # Ensure it's properly bounded
    result["ctr"] = result["ctr"].clip(0, 1)

    # Average dwell time — already exists, normalize
    max_dwell = result["avg_dwell"].quantile(0.99)
    result["avg_dwell_time_norm"] = (result["avg_dwell"] / max(max_dwell, 1)).clip(0, 1)

    # Skip rate — already exists
    result["skip_rate"] = result["skip_rate"].clip(0, 1)

    # Position bias correction (IPW weight)
    result["position_bias_correction"] = result["avg_ipw_weight"]

    # Historical CTR with Bayesian smoothing
    # Additive smoothing: (clicks + alpha) / (impressions + alpha + beta)
    alpha = 1.0  # Prior clicks
    beta = 10.0  # Prior impressions
    result["historical_ctr"] = (
        (result["total_clicks"] + alpha) /
        (result["total_impressions"] + alpha + beta)
    )

    # Click entropy per query (how spread out are clicks?)
    query_click_entropy = result.groupby("qid").apply(
        _compute_click_entropy, include_groups=False
    ).reset_index()
    query_click_entropy.columns = ["qid", "click_entropy"]
    result = result.merge(query_click_entropy, on="qid", how="left")
    result["click_entropy"] = result["click_entropy"].fillna(0)

    # Dwell time normalized by query average
    query_avg_dwell = result.groupby("qid")["avg_dwell"].transform("mean")
    result["dwell_time_norm"] = np.where(
        query_avg_dwell > 0,
        result["avg_dwell"] / query_avg_dwell,
        0.0
    )
    result["dwell_time_norm"] = result["dwell_time_norm"].clip(0, 5)

    # Impression count feature (log-scaled)
    result["log_impressions"] = np.log1p(result["total_impressions"])

    logger.info(f"  Interaction features extracted for {len(result):,} rows")
    return result


def _compute_click_entropy(group: pd.DataFrame) -> float:
    """Compute entropy of click distribution within a query group."""
    ctr_values = group["ctr"].values
    if len(ctr_values) == 0:
        return 0.0
    # Normalize to distribution
    total = ctr_values.sum()
    if total == 0:
        return 0.0
    probs = ctr_values / total
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    return entropy
