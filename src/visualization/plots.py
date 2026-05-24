"""
Visualization module.
All charts for the search ranking pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import os
import logging

logger = logging.getLogger(__name__)

# Style config
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "bm25": "#E74C3C",
    "lambdamart": "#2ECC71",
    "primary": "#3498DB",
    "secondary": "#9B59B6",
    "accent": "#F39C12",
}


def plot_feature_importance(
    importance: Dict[str, float],
    save_path: str,
    top_k: int = 20,
    title: str = "LambdaMART Feature Importance (Gain)",
) -> None:
    """Plot horizontal bar chart of feature importance."""
    sorted_imp = dict(list(importance.items())[:top_k])

    fig, ax = plt.subplots(figsize=(10, 8))
    features = list(sorted_imp.keys())[::-1]
    values = list(sorted_imp.values())[::-1]

    colors = [COLORS["primary"] if v > np.median(values) else COLORS["secondary"]
              for v in values]

    ax.barh(features, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Importance (Gain)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.tick_params(axis="y", labelsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {save_path}")


def plot_ndcg_comparison(
    bm25_metrics: Dict[str, float],
    lmart_metrics: Dict[str, float],
    save_path: str,
) -> None:
    """Plot NDCG@k comparison between BM25 and LambdaMART."""
    cutoffs = [1, 5, 10]
    bm25_vals = [bm25_metrics.get(f"NDCG@{k}", 0) for k in cutoffs]
    lmart_vals = [lmart_metrics.get(f"NDCG@{k}", 0) for k in cutoffs]

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(cutoffs))
    width = 0.35

    bars1 = ax.bar(x - width/2, bm25_vals, width, label="BM25",
                   color=COLORS["bm25"], edgecolor="white")
    bars2 = ax.bar(x + width/2, lmart_vals, width, label="LambdaMART",
                   color=COLORS["lambdamart"], edgecolor="white")

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Cutoff k", fontsize=12)
    ax.set_ylabel("NDCG@k", fontsize=12)
    ax.set_title("NDCG@k: BM25 vs LambdaMART", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"@{k}" for k in cutoffs])
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(max(bm25_vals), max(lmart_vals)) * 1.2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {save_path}")


def plot_position_bias(
    pos_stats: pd.DataFrame,
    save_path: str,
) -> None:
    """Plot position bias: CTR and normalized CTR by position."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # CTR by position
    ax1.bar(pos_stats["position"], pos_stats["ctr"],
            color=COLORS["primary"], alpha=0.8, edgecolor="white")
    ax1.set_xlabel("Position", fontsize=12)
    ax1.set_ylabel("Click-Through Rate", fontsize=12)
    ax1.set_title("CTR by Position (Position Bias)", fontsize=13, fontweight="bold")

    # Normalized CTR + examination probability curve
    ax2.plot(pos_stats["position"], pos_stats["normalized_ctr"],
             "o-", color=COLORS["bm25"], label="Observed (Normalized CTR)", linewidth=2)

    # Theoretical curve: 1/(1+k)^0.7
    k_range = np.arange(len(pos_stats))
    theoretical = 1.0 / (1.0 + k_range) ** 0.7
    ax2.plot(k_range, theoretical, "--", color=COLORS["secondary"],
             label="Theoretical P(examine)", linewidth=2)

    ax2.set_xlabel("Position", fontsize=12)
    ax2.set_ylabel("Normalized CTR / P(examine)", fontsize=12)
    ax2.set_title("Position Bias: Observed vs Theoretical", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {save_path}")


def plot_ablation_heatmap(
    ablation_results: Dict[str, Dict[str, float]],
    save_path: str,
) -> None:
    """Plot heatmap of ablation study results."""
    metrics = ["NDCG@1", "NDCG@5", "NDCG@10", "MAP", "MRR"]
    groups = []
    data = []

    for key in ["full_model", "without_query", "without_document", "without_interaction"]:
        if key in ablation_results:
            label = key.replace("without_", "– ").replace("_", " ").title()
            if key == "full_model":
                label = "Full Model"
            groups.append(label)
            row = [ablation_results[key].get(m, 0) for m in metrics]
            data.append(row)

    if not data:
        logger.warning("No ablation data to plot")
        return

    df = pd.DataFrame(data, index=groups, columns=metrics)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(df, annot=True, fmt=".4f", cmap="RdYlGn", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Score"})
    ax.set_title("Feature Ablation Study", fontsize=14, fontweight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {save_path}")


def plot_per_query_distribution(
    per_query_df: pd.DataFrame,
    save_path: str,
    model_name: str = "LambdaMART",
) -> None:
    """Plot distribution of per-query NDCG@10 scores."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(per_query_df["ndcg@10"], bins=30, color=COLORS["lambdamart"],
             alpha=0.8, edgecolor="white")
    ax1.axvline(per_query_df["ndcg@10"].mean(), color=COLORS["bm25"],
                linestyle="--", linewidth=2, label=f"Mean: {per_query_df['ndcg@10'].mean():.4f}")
    ax1.set_xlabel("NDCG@10", fontsize=12)
    ax1.set_ylabel("Number of Queries", fontsize=12)
    ax1.set_title(f"Per-Query NDCG@10 Distribution ({model_name})",
                  fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)

    # Box plot by query doc count
    per_query_df["doc_count_bin"] = pd.cut(
        per_query_df["n_docs"],
        bins=[0, 5, 10, 20, float("inf")],
        labels=["1-5", "6-10", "11-20", "20+"]
    )
    per_query_df.boxplot(column="ndcg@10", by="doc_count_bin", ax=ax2)
    ax2.set_xlabel("Documents per Query", fontsize=12)
    ax2.set_ylabel("NDCG@10", fontsize=12)
    ax2.set_title(f"NDCG@10 by Query Size ({model_name})",
                  fontsize=13, fontweight="bold")
    plt.suptitle("")  # Remove auto-title

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {save_path}")


def plot_ab_test_metrics(
    daily_results: Dict,
    save_path: str,
) -> None:
    """Plot A/B test metric comparison."""
    metrics = ["ctr", "ndcg10", "avg_dwell", "no_click_rate"]
    labels = ["CTR (Top-3)", "NDCG@10", "Avg Dwell (s)", "No-Click Rate"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[i]
        control = daily_results.get(f"control_avg_{metric}", 0)
        treatment = daily_results.get(f"treatment_avg_{metric}", 0)
        delta = daily_results.get(f"delta_{metric}_pct", 0)

        bars = ax.bar(["BM25\n(Control)", "LambdaMART\n(Treatment)"],
                      [control, treatment],
                      color=[COLORS["bm25"], COLORS["lambdamart"]],
                      edgecolor="white", width=0.5)

        # Add value labels
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=10)

        ax.set_title(f"{label} (Δ={delta:+.1f}%)", fontsize=12, fontweight="bold")
        ax.set_ylabel(label)

        # Significance marker
        sig = daily_results.get("significance", {}).get(metric, {})
        if sig.get("significant", False):
            ax.annotate("p < 0.05 ✓", xy=(0.95, 0.95), xycoords="axes fraction",
                        ha="right", va="top", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#2ECC71", alpha=0.3))

    plt.suptitle("A/B Test Results: BM25 vs LambdaMART",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {save_path}")


def plot_label_distribution(
    df: pd.DataFrame,
    save_path: str,
) -> None:
    """Plot relevance grade distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))

    grade_dist = df["relevance_grade"].value_counts().sort_index()
    colors = ["#E74C3C", "#E67E22", "#F1C40F", "#2ECC71", "#27AE60"]

    ax.bar(grade_dist.index, grade_dist.values, color=colors[:len(grade_dist)],
           edgecolor="white", width=0.6)

    for i, (grade, count) in enumerate(grade_dist.items()):
        ax.text(grade, count + max(grade_dist.values) * 0.02,
                f"{count:,}\n({count/len(df)*100:.1f}%)",
                ha="center", fontsize=10)

    ax.set_xlabel("Relevance Grade", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Relevance Grade Distribution", fontsize=14, fontweight="bold")
    ax.set_xticks(range(5))
    ax.set_xticklabels(["0: Skip", "1: Short Click", "2: Medium", "3: Long Click", "4: Very Long"],
                       rotation=15, ha="right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {save_path}")


def plot_all_metrics_comparison(
    bm25_metrics: Dict[str, float],
    lmart_metrics: Dict[str, float],
    save_path: str,
) -> None:
    """Plot radar/bar chart comparing all metrics."""
    metrics = list(bm25_metrics.keys())
    bm25_vals = [bm25_metrics[m] for m in metrics]
    lmart_vals = [lmart_metrics[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, bm25_vals, width, label="BM25",
                   color=COLORS["bm25"], alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width/2, lmart_vals, width, label="LambdaMART",
                   color=COLORS["lambdamart"], alpha=0.85, edgecolor="white")

    # Delta annotations
    for i in range(len(metrics)):
        if bm25_vals[i] > 0:
            delta = (lmart_vals[i] - bm25_vals[i]) / bm25_vals[i] * 100
            ax.annotate(f"{delta:+.1f}%",
                        xy=(x[i] + width/2, lmart_vals[i]),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=9, fontweight="bold",
                        color=COLORS["lambdamart"] if delta > 0 else COLORS["bm25"])

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Full Metrics Comparison: BM25 vs LambdaMART",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.legend(fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {save_path}")
