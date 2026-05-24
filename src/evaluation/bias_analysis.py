"""
Position bias analysis.
Quantify bias, compare models with/without IPW correction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging

from src.evaluation.metrics import evaluate_ranking

logger = logging.getLogger(__name__)


def quantify_position_bias(click_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze position bias in click data.

    Computes CTR, avg dwell, click probability by position.

    Args:
        click_logs: Raw click log DataFrame

    Returns:
        DataFrame with per-position statistics
    """
    logger.info("Quantifying position bias...")

    pos_stats = click_logs.groupby("position").agg(
        impressions=("clicked", "count"),
        clicks=("clicked", "sum"),
        avg_dwell=("dwell_time", "mean"),
        avg_relevance=("true_relevance", "mean"),
        skip_count=("skipped", "sum"),
    ).reset_index()

    pos_stats["ctr"] = pos_stats["clicks"] / pos_stats["impressions"]
    pos_stats["skip_rate"] = pos_stats["skip_count"] / pos_stats["impressions"]

    # Normalized CTR (relative to position 0)
    ctr_pos0 = pos_stats.loc[pos_stats["position"] == 0, "ctr"].values[0]
    pos_stats["normalized_ctr"] = pos_stats["ctr"] / ctr_pos0

    logger.info("\nPosition Bias Analysis:")
    logger.info(f"{'Pos':>4} {'CTR':>8} {'NormCTR':>8} {'AvgDwell':>9} {'SkipRate':>9} {'AvgRel':>8}")
    logger.info("-" * 52)
    for _, row in pos_stats.head(15).iterrows():
        logger.info(
            f"{int(row['position']):>4} {row['ctr']:>8.4f} {row['normalized_ctr']:>8.4f} "
            f"{row['avg_dwell']:>9.1f} {row['skip_rate']:>9.4f} {row['avg_relevance']:>8.4f}"
        )

    return pos_stats


def compare_ipw_effectiveness(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_store,
    model_config: dict,
) -> Dict[str, Dict[str, float]]:
    """
    Compare model trained with vs without IPW correction.

    Trains two LambdaMART models:
    1. With position_bias_correction feature (IPW-corrected)
    2. Without position_bias_correction feature

    Args:
        train_df: Training data
        test_df: Test data
        feature_store: FeatureStore instance
        model_config: Model config

    Returns:
        Dict with metrics for both variants
    """
    from src.models.lambdamart import LambdaMARTRanker
    import copy

    logger.info("\n" + "=" * 60)
    logger.info("IPW EFFECTIVENESS ANALYSIS")
    logger.info("=" * 60)

    results = {}

    # Model WITH IPW
    logger.info("\n--- Training with IPW correction ---")
    all_features = feature_store.feature_names
    X_train = feature_store.get_feature_matrix(train_df, all_features)
    y_train = feature_store.get_labels(train_df)
    qid_train = feature_store.get_qids(train_df)
    X_test = feature_store.get_feature_matrix(test_df, all_features)

    model_with = LambdaMARTRanker(copy.deepcopy(model_config))
    model_with.fit(X_train, y_train, qid_train, feature_names=all_features)

    scores_with = model_with.predict(X_test)
    metrics_with = evaluate_ranking(test_df, scores_with, model_name="With IPW")
    results["with_ipw"] = metrics_with

    # Model WITHOUT IPW
    logger.info("\n--- Training without IPW correction ---")
    features_no_ipw = [f for f in all_features if f != "position_bias_correction"]
    X_train_no = feature_store.get_feature_matrix(train_df, features_no_ipw)
    X_test_no = feature_store.get_feature_matrix(test_df, features_no_ipw)

    model_without = LambdaMARTRanker(copy.deepcopy(model_config))
    model_without.fit(X_train_no, y_train, qid_train, feature_names=features_no_ipw)

    scores_without = model_without.predict(X_test_no)
    metrics_without = evaluate_ranking(test_df, scores_without, model_name="Without IPW")
    results["without_ipw"] = metrics_without

    # Print comparison
    print("\n" + "=" * 55)
    print("  IPW EFFECTIVENESS COMPARISON")
    print("=" * 55)
    print(f"{'Metric':<20} {'With IPW':>12} {'Without IPW':>14} {'Δ':>8}")
    print("-" * 55)
    for metric in metrics_with:
        w = metrics_with[metric]
        wo = metrics_without[metric]
        delta = w - wo
        print(f"{metric:<20} {w:>12.4f} {wo:>14.4f} {delta:>+8.4f}")
    print("=" * 55)

    return results


def analyze_fairness_by_position(
    test_df: pd.DataFrame,
    bm25_scores: np.ndarray,
    lmart_scores: np.ndarray,
) -> pd.DataFrame:
    """
    Analyze if LambdaMART ranking is fairer across positions
    compared to BM25.

    Args:
        test_df: Test DataFrame
        bm25_scores: BM25 predicted scores
        lmart_scores: LambdaMART predicted scores

    Returns:
        DataFrame with fairness metrics per position bucket
    """
    df = test_df.copy()
    df["bm25_score"] = bm25_scores
    df["lmart_score"] = lmart_scores

    # Bucket by original average position
    df["pos_bucket"] = pd.cut(
        df["avg_position"],
        bins=[0, 2, 5, 10, float("inf")],
        labels=["top_2", "3_5", "6_10", "10+"],
    )

    fairness = df.groupby("pos_bucket", observed=True).agg(
        count=("qid", "count"),
        avg_true_rel=("relevance_grade", "mean"),
        avg_bm25=("bm25_score", "mean"),
        avg_lmart=("lmart_score", "mean"),
    ).reset_index()

    logger.info("\nFairness Analysis by Position:")
    logger.info(fairness.to_string(index=False))

    return fairness
