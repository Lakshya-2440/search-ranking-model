"""
Shadow deployment simulation.
Runs LambdaMART alongside BM25 on test queries without user impact.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats
import logging

from src.evaluation.metrics import evaluate_ranking, ndcg_at_k

logger = logging.getLogger(__name__)


def run_shadow_deployment(
    test_df: pd.DataFrame,
    bm25_scores: np.ndarray,
    lmart_scores: np.ndarray,
    n_queries: int = 1000,
    seed: int = 42,
) -> Dict:
    """
    Simulate shadow deployment.

    Run LambdaMART alongside BM25 on test queries.
    Compare rankings without affecting users.

    Args:
        test_df: Test DataFrame
        bm25_scores: BM25 predicted scores
        lmart_scores: LambdaMART predicted scores
        n_queries: Number of test queries to evaluate
        seed: Random seed for query sampling

    Returns:
        Dict with shadow deployment analysis results
    """
    logger.info("=" * 60)
    logger.info("SHADOW DEPLOYMENT SIMULATION")
    logger.info("=" * 60)

    rng = np.random.RandomState(seed)

    df = test_df.copy()
    df["bm25_score"] = bm25_scores
    df["lmart_score"] = lmart_scores

    # Sample queries
    unique_qids = df["qid"].unique()
    sample_size = min(n_queries, len(unique_qids))
    sampled_qids = rng.choice(unique_qids, size=sample_size, replace=False)
    df_sample = df[df["qid"].isin(sampled_qids)]

    logger.info(f"  Evaluating {sample_size:,} queries ({len(df_sample):,} query-doc pairs)")

    # Per-query comparison
    comparisons = []
    rank_disagreements = 0
    total_queries = 0

    for qid, group in df_sample.groupby("qid"):
        if len(group) < 2:
            continue

        total_queries += 1
        rels = group["relevance_grade"].values

        # BM25 ranking
        bm25_order = np.argsort(-group["bm25_score"].values)
        bm25_ranked_rels = rels[bm25_order]

        # LambdaMART ranking
        lmart_order = np.argsort(-group["lmart_score"].values)
        lmart_ranked_rels = rels[lmart_order]

        bm25_ndcg = ndcg_at_k(bm25_ranked_rels, 10)
        lmart_ndcg = ndcg_at_k(lmart_ranked_rels, 10)

        # Check rank agreement (top-3)
        if not np.array_equal(bm25_order[:3], lmart_order[:3]):
            rank_disagreements += 1

        comparisons.append({
            "qid": qid,
            "bm25_ndcg10": bm25_ndcg,
            "lmart_ndcg10": lmart_ndcg,
            "ndcg_diff": lmart_ndcg - bm25_ndcg,
            "lmart_wins": int(lmart_ndcg > bm25_ndcg),
            "bm25_wins": int(bm25_ndcg > lmart_ndcg),
            "ties": int(bm25_ndcg == lmart_ndcg),
        })

    comp_df = pd.DataFrame(comparisons)

    # Summary statistics
    results = {
        "n_queries_evaluated": total_queries,
        "lmart_wins": int(comp_df["lmart_wins"].sum()),
        "bm25_wins": int(comp_df["bm25_wins"].sum()),
        "ties": int(comp_df["ties"].sum()),
        "lmart_win_rate": float(comp_df["lmart_wins"].mean()),
        "avg_ndcg_diff": float(comp_df["ndcg_diff"].mean()),
        "median_ndcg_diff": float(comp_df["ndcg_diff"].median()),
        "rank_disagreement_rate": rank_disagreements / max(total_queries, 1),
        "avg_bm25_ndcg10": float(comp_df["bm25_ndcg10"].mean()),
        "avg_lmart_ndcg10": float(comp_df["lmart_ndcg10"].mean()),
    }

    # Statistical significance (paired t-test)
    t_stat, p_value = stats.ttest_rel(
        comp_df["lmart_ndcg10"].values,
        comp_df["bm25_ndcg10"].values,
    )
    results["t_statistic"] = float(t_stat)
    results["p_value"] = float(p_value)
    results["significant_at_005"] = p_value < 0.05

    # Score distribution analysis
    results["bm25_score_std"] = float(df_sample["bm25_score"].std())
    results["lmart_score_std"] = float(df_sample["lmart_score"].std())

    # Rank correlation (Spearman)
    rank_corr, _ = stats.spearmanr(
        df_sample["bm25_score"].values,
        df_sample["lmart_score"].values,
    )
    results["rank_correlation"] = float(rank_corr)

    # Print report
    _print_shadow_report(results)

    return results


def _print_shadow_report(results: Dict) -> None:
    """Print formatted shadow deployment report."""
    print("\n" + "=" * 60)
    print("  SHADOW DEPLOYMENT REPORT")
    print("=" * 60)
    print(f"  Queries evaluated:     {results['n_queries_evaluated']:>8,}")
    print(f"  LambdaMART wins:       {results['lmart_wins']:>8,} ({results['lmart_win_rate']*100:.1f}%)")
    print(f"  BM25 wins:             {results['bm25_wins']:>8,}")
    print(f"  Ties:                  {results['ties']:>8,}")
    print(f"")
    print(f"  Avg NDCG@10 (BM25):    {results['avg_bm25_ndcg10']:>8.4f}")
    print(f"  Avg NDCG@10 (LMart):   {results['avg_lmart_ndcg10']:>8.4f}")
    print(f"  Avg NDCG@10 diff:      {results['avg_ndcg_diff']:>+8.4f}")
    print(f"")
    print(f"  Rank disagreement:     {results['rank_disagreement_rate']*100:>7.1f}%")
    print(f"  Rank correlation:      {results['rank_correlation']:>8.4f}")
    print(f"")
    print(f"  t-statistic:           {results['t_statistic']:>8.3f}")
    print(f"  p-value:               {results['p_value']:>8.6f}")
    sig = "YES ✓" if results['significant_at_005'] else "NO"
    print(f"  Significant (α=0.05):  {sig:>8}")
    print("=" * 60)
