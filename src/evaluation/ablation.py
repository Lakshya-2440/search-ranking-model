"""
Feature ablation study.
Measures impact of each feature group by training without it.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
import copy

from src.models.lambdamart import LambdaMARTRanker
from src.evaluation.metrics import evaluate_ranking
from src.features.feature_store import FeatureStore, FEATURE_REGISTRY

logger = logging.getLogger(__name__)


def run_ablation_study(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_store: FeatureStore,
    model_config: dict,
    full_model_metrics: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    """
    Feature group ablation study.

    For each feature group (query, document, interaction):
    1. Remove all features in that group
    2. Retrain LambdaMART
    3. Evaluate on test set
    4. Measure NDCG drop vs full model

    Args:
        train_df: Training data with features
        val_df: Validation data
        test_df: Test data
        feature_store: Fitted FeatureStore
        model_config: LambdaMART config
        full_model_metrics: Metrics from full model for comparison

    Returns:
        Dict[group_name → Dict[metric → value]] with ablation results
    """
    logger.info("=" * 60)
    logger.info("ABLATION STUDY: Feature group importance")
    logger.info("=" * 60)

    groups = ["query", "document", "interaction"]
    results = {"full_model": full_model_metrics}

    for group in groups:
        logger.info(f"\n--- Ablating: {group} features ---")

        # Get features WITHOUT this group
        features_without = feature_store.get_features_excluding_group(group)
        removed = feature_store.get_features_by_group(group)
        logger.info(f"  Removed: {removed}")
        logger.info(f"  Remaining: {len(features_without)} features")

        if len(features_without) == 0:
            logger.warning(f"  No features remaining after removing {group}! Skipping.")
            continue

        # Build feature matrices without group
        X_train = feature_store.get_feature_matrix(train_df, features_without)
        y_train = feature_store.get_labels(train_df)
        qid_train = feature_store.get_qids(train_df)

        X_val = feature_store.get_feature_matrix(val_df, features_without)
        y_val = feature_store.get_labels(val_df)
        qid_val = feature_store.get_qids(val_df)

        X_test = feature_store.get_feature_matrix(test_df, features_without)

        # Train ablated model
        ablated_config = copy.deepcopy(model_config)
        ablated_config["n_estimators"] = min(model_config.get("n_estimators", 1000), 500)

        ablated_model = LambdaMARTRanker(ablated_config)
        ablated_model.fit(
            X_train, y_train, qid_train,
            X_val, y_val, qid_val,
            feature_names=features_without,
        )

        # Evaluate
        ablated_scores = ablated_model.predict(X_test)
        ablated_metrics = evaluate_ranking(
            test_df, ablated_scores,
            model_name=f"Without {group}",
        )

        # Compute deltas
        for metric in ablated_metrics:
            full_val = full_model_metrics.get(metric, 0)
            abl_val = ablated_metrics[metric]
            if full_val > 0:
                drop = (full_val - abl_val) / full_val * 100
            else:
                drop = 0
            ablated_metrics[f"{metric}_drop_pct"] = drop

        results[f"without_{group}"] = ablated_metrics

    # Print ablation summary
    _print_ablation_summary(results, full_model_metrics)

    return results


def _print_ablation_summary(
    results: Dict[str, Dict[str, float]],
    full_metrics: Dict[str, float],
) -> None:
    """Print formatted ablation results table."""
    print("\n" + "=" * 75)
    print("  ABLATION STUDY RESULTS")
    print("=" * 75)
    print(f"{'Configuration':<25} {'NDCG@10':>10} {'Δ NDCG@10':>12} {'MAP':>10} {'MRR':>10}")
    print("-" * 75)

    # Full model
    print(f"{'Full model':<25} {full_metrics.get('NDCG@10', 0):>10.4f} {'---':>12} "
          f"{full_metrics.get('MAP', 0):>10.4f} {full_metrics.get('MRR', 0):>10.4f}")

    # Ablated models
    for key in ["without_query", "without_document", "without_interaction"]:
        if key in results:
            metrics = results[key]
            group_name = key.replace("without_", "– ") + " features"
            ndcg = metrics.get("NDCG@10", 0)
            drop = metrics.get("NDCG@10_drop_pct", 0)
            map_score = metrics.get("MAP", 0)
            mrr = metrics.get("MRR", 0)
            print(f"{group_name:<25} {ndcg:>10.4f} {drop:>+11.1f}% "
                  f"{map_score:>10.4f} {mrr:>10.4f}")

    print("=" * 75)


def run_single_feature_importance(
    test_df: pd.DataFrame,
    feature_store: FeatureStore,
    model: LambdaMARTRanker,
    n_shuffles: int = 5,
) -> Dict[str, float]:
    """
    Permutation-based single feature importance.

    For each feature, shuffle it and measure NDCG@10 drop.
    Average over n_shuffles for stability.

    Args:
        test_df: Test data with features
        feature_store: FeatureStore
        model: Trained LambdaMART model
        n_shuffles: Number of shuffle iterations

    Returns:
        Dict[feature_name → importance_score (NDCG drop)]
    """
    logger.info("\nComputing permutation feature importance...")

    all_features = feature_store.feature_names
    X_test = feature_store.get_feature_matrix(test_df)

    # Baseline score
    base_scores = model.predict(X_test)
    base_metrics = evaluate_ranking(test_df, base_scores, model_name="Baseline")
    base_ndcg = base_metrics["NDCG@10"]

    importance = {}
    for feat_idx, feat_name in enumerate(all_features):
        if feat_name not in test_df.columns:
            continue

        drops = []
        for _ in range(n_shuffles):
            X_shuffled = X_test.copy()
            np.random.shuffle(X_shuffled[:, feat_idx])

            shuffled_scores = model.predict(X_shuffled)
            shuffled_metrics = evaluate_ranking(
                test_df, shuffled_scores, model_name=f"Shuffled {feat_name}"
            )
            drop = base_ndcg - shuffled_metrics["NDCG@10"]
            drops.append(drop)

        importance[feat_name] = float(np.mean(drops))
        logger.info(f"  {feat_name}: {importance[feat_name]:.4f} NDCG@10 drop")

    # Sort by importance
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    return importance
