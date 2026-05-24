"""
Model utilities.
Save/load, feature importance analysis, model comparison.
"""

import numpy as np
import pandas as pd
import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def save_results(results: Dict[str, Any], path: str) -> None:
    """Save evaluation results to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert_value(value):
        if isinstance(value, (np.integer,)):
            return int(value)
        elif isinstance(value, (np.floating,)):
            return float(value)
        elif isinstance(value, (np.bool_,)):
            return bool(value)
        elif isinstance(value, bool):
            return value
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, dict):
            return {k: convert_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [convert_value(v) for v in value]
        return value

    serializable = convert_value(results)

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Results saved to {path}")


def load_results(path: str) -> Dict[str, Any]:
    """Load evaluation results from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def print_comparison_table(
    bm25_metrics: Dict[str, float],
    lmart_metrics: Dict[str, float],
    title: str = "Model Comparison",
) -> str:
    """
    Print side-by-side comparison of BM25 vs LambdaMART metrics.

    Returns:
        Formatted comparison string
    """
    lines = []
    lines.append("=" * 65)
    lines.append(f"  {title}")
    lines.append("=" * 65)
    lines.append(f"{'Metric':<25} {'BM25':>12} {'LambdaMART':>12} {'Δ (%)':>10}")
    lines.append("-" * 65)

    all_metrics = sorted(set(list(bm25_metrics.keys()) + list(lmart_metrics.keys())))

    for metric in all_metrics:
        bm25_val = bm25_metrics.get(metric, 0.0)
        lmart_val = lmart_metrics.get(metric, 0.0)

        if bm25_val > 0:
            delta_pct = (lmart_val - bm25_val) / bm25_val * 100
        else:
            delta_pct = 0.0

        marker = " ✓" if delta_pct > 0 else ""
        lines.append(
            f"{metric:<25} {bm25_val:>12.4f} {lmart_val:>12.4f} {delta_pct:>+9.1f}%{marker}"
        )

    lines.append("=" * 65)

    output = "\n".join(lines)
    print(output)
    return output


def ensure_output_dirs(config: dict) -> None:
    """Create output directories if they don't exist."""
    for key in ["model_dir", "plots_dir", "results_dir", "logs_dir"]:
        path = config.get("output", {}).get(key)
        if path:
            os.makedirs(path, exist_ok=True)
            logger.info(f"  Output dir ready: {path}")
