"""
Ranking evaluation metrics.
NDCG@k, MAP, MRR implementations.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    """
    Compute Discounted Cumulative Gain at position k.

    DCG@k = Σ (2^rel_i - 1) / log2(i + 2), for i = 0..k-1

    Args:
        relevances: Array of relevance scores in ranked order
        k: Cutoff position

    Returns:
        DCG@k score
    """
    relevances = np.array(relevances)[:k]
    if len(relevances) == 0:
        return 0.0
    discounts = np.log2(np.arange(len(relevances)) + 2)
    gains = (2 ** relevances - 1) / discounts
    return float(np.sum(gains))


def ndcg_at_k(relevances: np.ndarray, k: int) -> float:
    """
    Compute Normalized DCG at position k.

    NDCG@k = DCG@k / IDCG@k

    Args:
        relevances: Array of relevance scores in ranked order
        k: Cutoff position

    Returns:
        NDCG@k score in [0, 1]
    """
    actual_dcg = dcg_at_k(relevances, k)

    # Ideal DCG: sort relevances descending
    ideal_relevances = np.sort(relevances)[::-1]
    ideal_dcg = dcg_at_k(ideal_relevances, k)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def mean_average_precision(relevances_list: List[np.ndarray], threshold: int = 1) -> float:
    """
    Compute Mean Average Precision (MAP).

    Args:
        relevances_list: List of relevance arrays (one per query), in ranked order
        threshold: Relevance threshold for "relevant" (default: grade >= 1)

    Returns:
        MAP score
    """
    aps = []
    for relevances in relevances_list:
        binary = (np.array(relevances) >= threshold).astype(float)
        if binary.sum() == 0:
            aps.append(0.0)
            continue

        precisions = []
        n_relevant = 0
        for i, rel in enumerate(binary):
            if rel == 1:
                n_relevant += 1
                precisions.append(n_relevant / (i + 1))

        aps.append(np.mean(precisions) if precisions else 0.0)

    return float(np.mean(aps))


def mean_reciprocal_rank(relevances_list: List[np.ndarray], threshold: int = 1) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    Args:
        relevances_list: List of relevance arrays (one per query), in ranked order
        threshold: Relevance threshold for "relevant"

    Returns:
        MRR score
    """
    rrs = []
    for relevances in relevances_list:
        binary = (np.array(relevances) >= threshold).astype(float)
        relevant_positions = np.where(binary == 1)[0]
        if len(relevant_positions) > 0:
            rrs.append(1.0 / (relevant_positions[0] + 1))
        else:
            rrs.append(0.0)

    return float(np.mean(rrs))


def evaluate_ranking(
    df: pd.DataFrame,
    scores: np.ndarray,
    cutoffs: List[int] = [1, 5, 10],
    model_name: str = "Model",
) -> Dict[str, float]:
    """
    Full ranking evaluation.

    For each query, sort documents by predicted scores,
    then compute NDCG@k, MAP, MRR.

    Args:
        df: DataFrame with 'qid' and 'relevance_grade' columns
        scores: Predicted relevance scores (higher = more relevant)
        cutoffs: NDCG cutoff positions
        model_name: Name for logging

    Returns:
        Dict of metric_name → score
    """
    logger.info(f"\nEvaluating {model_name}...")

    df_eval = df.copy()
    df_eval["pred_score"] = scores

    results = {}
    all_ndcg = {k: [] for k in cutoffs}
    all_relevances = []

    # Evaluate per query
    for qid, group in df_eval.groupby("qid"):
        # Sort by predicted score (descending)
        ranked = group.sort_values("pred_score", ascending=False)
        relevances = ranked["relevance_grade"].values

        all_relevances.append(relevances)

        # NDCG at each cutoff
        for k in cutoffs:
            all_ndcg[k].append(ndcg_at_k(relevances, k))

    # Aggregate
    for k in cutoffs:
        results[f"NDCG@{k}"] = float(np.mean(all_ndcg[k]))

    results["MAP"] = mean_average_precision(all_relevances)
    results["MRR"] = mean_reciprocal_rank(all_relevances)

    # Log results
    logger.info(f"  {model_name} Results:")
    for metric, value in results.items():
        logger.info(f"    {metric}: {value:.4f}")

    return results


def per_query_metrics(
    df: pd.DataFrame,
    scores: np.ndarray,
    k: int = 10,
) -> pd.DataFrame:
    """
    Compute per-query NDCG@k for detailed analysis.

    Args:
        df: DataFrame with 'qid' and 'relevance_grade'
        scores: Predicted scores
        k: Cutoff position

    Returns:
        DataFrame with qid and ndcg score per query
    """
    df_eval = df.copy()
    df_eval["pred_score"] = scores

    per_query = []
    for qid, group in df_eval.groupby("qid"):
        ranked = group.sort_values("pred_score", ascending=False)
        relevances = ranked["relevance_grade"].values
        ndcg = ndcg_at_k(relevances, k)
        per_query.append({"qid": qid, f"ndcg@{k}": ndcg, "n_docs": len(group)})

    return pd.DataFrame(per_query)
