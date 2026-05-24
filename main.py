#!/usr/bin/env python3
"""
Search Ranking Model — Full Pipeline Orchestrator

End-to-end LambdaMART learning-to-rank system.
Generates data, engineers features, trains models, evaluates, and simulates deployment.

Usage:
    python main.py
    python main.py --config configs/config.yaml
    python main.py --skip-ablation  # Skip ablation study (faster)
"""

import argparse
import logging
import os
import sys
import time
import yaml
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML config."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Config loaded from {config_path}")
    return config


def run_pipeline(config: dict, skip_ablation: bool = False) -> None:
    """
    Full pipeline execution.

    Steps:
    1. Generate synthetic data (2M+ query-doc pairs)
    2. Extract labels from click signals
    3. Time-based train/val/test split
    4. Feature engineering
    5. Train BM25 baseline
    6. Train LambdaMART
    7. Evaluate both models
    8. Feature ablation study
    9. Position bias analysis
    10. Shadow deployment simulation
    11. A/B test simulation
    12. Generate all visualizations
    13. Print final results
    """
    from src.data.generate_synthetic import generate_full_dataset
    from src.data.label_extraction import extract_labels
    from src.data.splitter import time_based_split
    from src.features.feature_store import FeatureStore
    from src.models.bm25_baseline import BM25Baseline
    from src.models.lambdamart import LambdaMARTRanker
    from src.models.model_utils import (
        save_results, print_comparison_table, ensure_output_dirs,
    )
    from src.evaluation.metrics import evaluate_ranking, per_query_metrics
    from src.evaluation.ablation import run_ablation_study
    from src.evaluation.bias_analysis import (
        quantify_position_bias, analyze_fairness_by_position,
    )
    from src.serving.shadow_deploy import run_shadow_deployment
    from src.serving.ab_testing import ABTestSimulator
    from src.visualization.plots import (
        plot_feature_importance, plot_ndcg_comparison, plot_position_bias,
        plot_ablation_heatmap, plot_per_query_distribution,
        plot_ab_test_metrics, plot_label_distribution,
        plot_all_metrics_comparison,
    )

    start_time = time.time()

    # Create output directories
    ensure_output_dirs(config)
    plots_dir = config["output"]["plots_dir"]
    results_dir = config["output"]["results_dir"]
    model_dir = config["output"]["model_dir"]

    # ================================================================
    # STEP 1: Generate synthetic data
    # ================================================================
    queries_df, docs_df, click_logs_df = generate_full_dataset(config["data"])

    # ================================================================
    # STEP 2: Extract labels
    # ================================================================
    labeled_df = extract_labels(click_logs_df, config["data"])

    # Plot label distribution
    plot_label_distribution(labeled_df, os.path.join(plots_dir, "label_distribution.png"))

    # ================================================================
    # STEP 3: Time-based split
    # ================================================================
    split_cfg = config["split"]
    train_df, val_df, test_df = time_based_split(
        labeled_df,
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        test_ratio=split_cfg["test_ratio"],
    )

    # ================================================================
    # STEP 4: Feature engineering
    # ================================================================
    feature_store = FeatureStore()

    # Fit on training doc titles
    all_doc_titles = train_df["doc_title"].tolist()
    feature_store.fit(all_doc_titles)

    # Transform all splits
    logger.info("\nTransforming training data...")
    train_df = feature_store.transform(train_df)
    logger.info("\nTransforming validation data...")
    val_df = feature_store.transform(val_df)
    logger.info("\nTransforming test data...")
    test_df = feature_store.transform(test_df)

    # ================================================================
    # STEP 5: Train BM25 baseline
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: BM25 Baseline")
    logger.info("=" * 60)

    bm25_cfg = config["model"]["bm25"]
    bm25_model = BM25Baseline(k1=bm25_cfg["k1"], b=bm25_cfg["b"])
    bm25_model.fit(
        train_df["doc_id"].tolist(),
        train_df["doc_title"].tolist(),
    )

    # BM25 scores on test set
    bm25_test_scores = bm25_model.predict_scores_for_df(test_df)

    # Evaluate BM25
    bm25_metrics = evaluate_ranking(
        test_df, bm25_test_scores,
        cutoffs=config["evaluation"]["ndcg_cutoffs"],
        model_name="BM25 Baseline",
    )

    # ================================================================
    # STEP 6: Train LambdaMART
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: LambdaMART Training")
    logger.info("=" * 60)

    lmart_cfg = config["model"]["lambdamart"]

    # Sort by qid — XGBoost requires qid in non-decreasing order
    train_df = feature_store.sort_by_qid(train_df)
    val_df = feature_store.sort_by_qid(val_df)
    test_df = feature_store.sort_by_qid(test_df)

    # Recompute BM25 scores after sorting (indices changed)
    bm25_test_scores = bm25_model.predict_scores_for_df(test_df)

    # Re-evaluate BM25 on sorted test
    bm25_metrics = evaluate_ranking(
        test_df, bm25_test_scores,
        cutoffs=config["evaluation"]["ndcg_cutoffs"],
        model_name="BM25 Baseline",
    )

    # Prepare feature matrices
    feature_names = feature_store.feature_names
    X_train = feature_store.get_feature_matrix(train_df)
    y_train = feature_store.get_labels(train_df)
    qid_train = feature_store.get_qids(train_df)

    X_val = feature_store.get_feature_matrix(val_df)
    y_val = feature_store.get_labels(val_df)
    qid_val = feature_store.get_qids(val_df)

    X_test = feature_store.get_feature_matrix(test_df)

    logger.info(f"  X_train shape: {X_train.shape}")
    logger.info(f"  X_val shape:   {X_val.shape}")
    logger.info(f"  X_test shape:  {X_test.shape}")

    # Train
    lmart_model = LambdaMARTRanker(lmart_cfg)
    lmart_model.fit(
        X_train, y_train, qid_train,
        X_val, y_val, qid_val,
        feature_names=feature_names,
    )

    # Save model
    lmart_model.save(os.path.join(model_dir, "lambdamart_model.json"))

    # ================================================================
    # STEP 7: Evaluate both models
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 7: Model Evaluation")
    logger.info("=" * 60)

    lmart_test_scores = lmart_model.predict(X_test)

    lmart_metrics = evaluate_ranking(
        test_df, lmart_test_scores,
        cutoffs=config["evaluation"]["ndcg_cutoffs"],
        model_name="LambdaMART",
    )

    # Print comparison
    comparison_str = print_comparison_table(bm25_metrics, lmart_metrics)

    # Feature importance
    importance = lmart_model.get_feature_importance("gain")
    logger.info("\nTop Feature Importance (Gain):")
    for feat, imp in list(importance.items())[:10]:
        logger.info(f"  {feat}: {imp:.1f}")

    # Per-query metrics
    lmart_per_query = per_query_metrics(test_df, lmart_test_scores, k=10)

    # Plots
    plot_ndcg_comparison(bm25_metrics, lmart_metrics,
                         os.path.join(plots_dir, "ndcg_comparison.png"))
    plot_feature_importance(importance,
                            os.path.join(plots_dir, "feature_importance.png"))
    plot_per_query_distribution(lmart_per_query,
                                os.path.join(plots_dir, "per_query_ndcg.png"))
    plot_all_metrics_comparison(bm25_metrics, lmart_metrics,
                                os.path.join(plots_dir, "all_metrics_comparison.png"))

    # Save metrics
    save_results({"bm25": bm25_metrics, "lambdamart": lmart_metrics},
                 os.path.join(results_dir, "evaluation_metrics.json"))
    save_results(importance, os.path.join(results_dir, "feature_importance.json"))

    # ================================================================
    # STEP 8: Ablation study
    # ================================================================
    if not skip_ablation:
        ablation_results = run_ablation_study(
            train_df, val_df, test_df,
            feature_store, lmart_cfg, lmart_metrics,
        )
        plot_ablation_heatmap(ablation_results,
                              os.path.join(plots_dir, "ablation_heatmap.png"))
        save_results(ablation_results,
                     os.path.join(results_dir, "ablation_results.json"))
    else:
        logger.info("\nSkipping ablation study (--skip-ablation flag)")

    # ================================================================
    # STEP 9: Position bias analysis
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 9: Position Bias Analysis")
    logger.info("=" * 60)

    pos_stats = quantify_position_bias(click_logs_df)
    plot_position_bias(pos_stats, os.path.join(plots_dir, "position_bias.png"))

    fairness = analyze_fairness_by_position(test_df, bm25_test_scores, lmart_test_scores)

    # ================================================================
    # STEP 10: Shadow deployment
    # ================================================================
    shadow_cfg = config["serving"]["shadow"]
    shadow_results = run_shadow_deployment(
        test_df, bm25_test_scores, lmart_test_scores,
        n_queries=shadow_cfg["n_test_queries"],
    )
    save_results(shadow_results,
                 os.path.join(results_dir, "shadow_deploy_results.json"))

    # ================================================================
    # STEP 11: A/B test simulation
    # ================================================================
    ab_cfg = config["serving"]["ab_test"]
    ab_sim = ABTestSimulator(
        traffic_split=ab_cfg["traffic_split"],
        duration_days=ab_cfg["duration_days"],
        significance_level=ab_cfg["significance_level"],
    )
    ab_results = ab_sim.run_test(test_df, bm25_test_scores, lmart_test_scores)
    plot_ab_test_metrics(ab_results, os.path.join(plots_dir, "ab_test_results.png"))
    save_results(ab_results, os.path.join(results_dir, "ab_test_results.json"))

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    elapsed = time.time() - start_time
    _print_final_summary(bm25_metrics, lmart_metrics, elapsed, config)


def _print_final_summary(
    bm25_metrics: dict,
    lmart_metrics: dict,
    elapsed: float,
    config: dict,
) -> None:
    """Print final pipeline summary."""
    ndcg_improvement = 0
    bm25_ndcg10 = bm25_metrics.get("NDCG@10", 0)
    lmart_ndcg10 = lmart_metrics.get("NDCG@10", 0)
    if bm25_ndcg10 > 0:
        ndcg_improvement = (lmart_ndcg10 - bm25_ndcg10) / bm25_ndcg10 * 100

    mrr_improvement = 0
    bm25_mrr = bm25_metrics.get("MRR", 0)
    lmart_mrr = lmart_metrics.get("MRR", 0)
    if bm25_mrr > 0:
        mrr_improvement = (lmart_mrr - bm25_mrr) / bm25_mrr * 100

    print("\n" + "█" * 65)
    print("█" + " " * 63 + "█")
    print("█" + "   SEARCH RANKING MODEL — FINAL RESULTS".center(63) + "█")
    print("█" + " " * 63 + "█")
    print("█" * 65)
    print(f"")
    print(f"  📊 Data: {config['data']['n_pairs']:,}+ query-doc pairs")
    print(f"  🔧 Features: {len(config['features']['query']) + len(config['features']['document']) + len(config['features']['interaction'])} engineered features")
    print(f"  🌲 Model: LambdaMART (XGBoost rank:ndcg)")
    print(f"")
    print(f"  ┌─────────────────────────────────────────────┐")
    print(f"  │  NDCG@10 Improvement: {ndcg_improvement:>+.1f}% over BM25     │")
    print(f"  │  Target: ≥18%                               │")
    target_met = "✅ TARGET MET" if ndcg_improvement >= 18 else "❌ TARGET MISSED"
    print(f"  │  Status: {target_met:<35}│")
    print(f"  └─────────────────────────────────────────────┘")
    print(f"")
    print(f"  Metric Results:")
    print(f"    NDCG@10:  BM25={bm25_ndcg10:.4f} → LambdaMART={lmart_ndcg10:.4f} ({ndcg_improvement:+.1f}%)")
    print(f"    MRR:      BM25={bm25_mrr:.4f} → LambdaMART={lmart_mrr:.4f} ({mrr_improvement:+.1f}%)")
    print(f"    MAP:      BM25={bm25_metrics.get('MAP', 0):.4f} → LambdaMART={lmart_metrics.get('MAP', 0):.4f}")
    print(f"")
    print(f"  📁 Outputs saved to: {config['output']['plots_dir']}")
    print(f"  ⏱  Pipeline completed in {elapsed:.1f}s")
    print(f"")
    print("█" * 65)

    # Resume bullet verification
    print(f"\n  📝 RESUME BULLETS VERIFICATION:")
    n_pairs = config['data']['n_pairs']
    pairs_str = f"{n_pairs/1_000_000:.0f}M+" if n_pairs >= 1_000_000 else f"{n_pairs/1_000:.0f}K+"
    print(f"    ✓ Built a learning-to-rank model using LambdaMART on {pairs_str} query-doc pairs")
    print(f"    {'✓' if ndcg_improvement >= 18 else '✗'} Improved NDCG@10 by {ndcg_improvement:.0f}% over BM25 baseline in offline evaluation")
    print()


def main():
    parser = argparse.ArgumentParser(description="Search Ranking Model Pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    parser.add_argument("--skip-ablation", action="store_true", help="Skip ablation study")
    args = parser.parse_args()

    config = load_config(args.config)
    run_pipeline(config, skip_ablation=args.skip_ablation)


if __name__ == "__main__":
    main()
