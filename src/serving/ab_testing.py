"""
A/B testing framework simulation.
Simulates 5% traffic split, metric tracking, significance testing.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional
import logging

from src.evaluation.metrics import ndcg_at_k

logger = logging.getLogger(__name__)


class ABTestSimulator:
    """
    Simulates A/B test between BM25 (control) and LambdaMART (treatment).
    """

    def __init__(
        self,
        traffic_split: float = 0.05,
        duration_days: int = 14,
        significance_level: float = 0.05,
        seed: int = 42,
    ):
        """
        Args:
            traffic_split: Fraction of traffic for treatment (default 5%)
            duration_days: Test duration in simulated days
            significance_level: Alpha for significance testing
            seed: Random seed
        """
        self.traffic_split = traffic_split
        self.duration_days = duration_days
        self.significance_level = significance_level
        self.rng = np.random.RandomState(seed)

    def run_test(
        self,
        test_df: pd.DataFrame,
        bm25_scores: np.ndarray,
        lmart_scores: np.ndarray,
    ) -> Dict:
        """
        Simulate full A/B test.

        Args:
            test_df: Test DataFrame
            bm25_scores: BM25 scores (control)
            lmart_scores: LambdaMART scores (treatment)

        Returns:
            Dict with A/B test results
        """
        logger.info("=" * 60)
        logger.info("A/B TEST SIMULATION")
        logger.info(f"  Traffic split: {self.traffic_split*100:.0f}% treatment")
        logger.info(f"  Duration: {self.duration_days} days")
        logger.info("=" * 60)

        df = test_df.copy()
        df["bm25_score"] = bm25_scores
        df["lmart_score"] = lmart_scores

        unique_qids = df["qid"].unique()
        n_queries = len(unique_qids)

        # Simulate daily traffic
        daily_results = []

        for day in range(self.duration_days):
            # Random subset of queries for this day
            n_daily = max(1, n_queries // self.duration_days)
            daily_qids = self.rng.choice(unique_qids, size=n_daily, replace=True)

            # Split into control / treatment
            assignments = self.rng.random(len(daily_qids)) < self.traffic_split
            control_qids = daily_qids[~assignments]
            treatment_qids = daily_qids[assignments]

            # Compute metrics for control (BM25)
            control_metrics = self._compute_daily_metrics(
                df, control_qids, "bm25_score", day
            )

            # Compute metrics for treatment (LambdaMART)
            treatment_metrics = self._compute_daily_metrics(
                df, treatment_qids, "lmart_score", day
            )

            daily_results.append({
                "day": day + 1,
                "control_queries": len(control_qids),
                "treatment_queries": len(treatment_qids),
                **{f"control_{k}": v for k, v in control_metrics.items()},
                **{f"treatment_{k}": v for k, v in treatment_metrics.items()},
            })

        daily_df = pd.DataFrame(daily_results)

        # Aggregate results
        results = self._aggregate_results(daily_df)

        # Statistical tests
        results["significance"] = self._run_significance_tests(daily_df)

        # Print report
        self._print_ab_report(results, daily_df)

        return results

    def _compute_daily_metrics(
        self,
        df: pd.DataFrame,
        qids: np.ndarray,
        score_col: str,
        day: int,
    ) -> Dict[str, float]:
        """Compute ranking metrics for a set of queries using given scores."""
        daily_df = df[df["qid"].isin(qids)]

        if len(daily_df) == 0:
            return {"ctr": 0, "avg_dwell": 0, "ndcg10": 0, "no_click_rate": 0}

        ndcg_scores = []
        ctr_values = []
        dwell_values = []
        no_click_counts = 0
        total_queries = 0

        for qid, group in daily_df.groupby("qid"):
            total_queries += 1

            # Rank by score
            ranked = group.sort_values(score_col, ascending=False)
            rels = ranked["relevance_grade"].values

            # NDCG@10
            ndcg_scores.append(ndcg_at_k(rels, 10))

            # Simulate CTR from top-3 (based on relevance)
            top3_rels = rels[:3]
            # Higher relevance → higher simulated CTR
            sim_ctr = np.mean(top3_rels) / 4.0 * 0.6 + 0.1  # Scale to 0.1-0.7 range
            ctr_values.append(sim_ctr)

            # Avg dwell from relevance
            avg_dwell_sim = np.mean(top3_rels) * 30 + self.rng.normal(0, 5)
            dwell_values.append(max(0, avg_dwell_sim))

            # No-click (top result has zero relevance)
            if rels[0] == 0:
                no_click_counts += 1

        return {
            "ctr": float(np.mean(ctr_values)),
            "avg_dwell": float(np.mean(dwell_values)),
            "ndcg10": float(np.mean(ndcg_scores)),
            "no_click_rate": no_click_counts / max(total_queries, 1),
        }

    def _aggregate_results(self, daily_df: pd.DataFrame) -> Dict:
        """Aggregate daily results into overall metrics."""
        control_cols = [c for c in daily_df.columns if c.startswith("control_")]
        treatment_cols = [c for c in daily_df.columns if c.startswith("treatment_")]

        results = {
            "total_days": len(daily_df),
            "total_control_queries": int(daily_df["control_queries"].sum()),
            "total_treatment_queries": int(daily_df["treatment_queries"].sum()),
        }

        # Mean metrics
        for col in control_cols:
            metric = col.replace("control_", "")
            results[f"control_avg_{metric}"] = float(daily_df[col].mean())

        for col in treatment_cols:
            metric = col.replace("treatment_", "")
            results[f"treatment_avg_{metric}"] = float(daily_df[col].mean())

        # Deltas
        for metric in ["ctr", "avg_dwell", "ndcg10", "no_click_rate"]:
            control = results.get(f"control_avg_{metric}", 0)
            treatment = results.get(f"treatment_avg_{metric}", 0)
            if control > 0:
                delta_pct = (treatment - control) / control * 100
            else:
                delta_pct = 0
            results[f"delta_{metric}_pct"] = delta_pct

        return results

    def _run_significance_tests(self, daily_df: pd.DataFrame) -> Dict:
        """Run statistical significance tests on daily metrics."""
        sig_results = {}

        for metric in ["ctr", "ndcg10", "avg_dwell", "no_click_rate"]:
            control = daily_df[f"control_{metric}"].values
            treatment = daily_df[f"treatment_{metric}"].values

            # Welch's t-test
            t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)

            # Bootstrap confidence interval
            n_bootstrap = 1000
            diffs = []
            for _ in range(n_bootstrap):
                c_sample = self.rng.choice(control, size=len(control), replace=True)
                t_sample = self.rng.choice(treatment, size=len(treatment), replace=True)
                diffs.append(t_sample.mean() - c_sample.mean())

            ci_lower = np.percentile(diffs, 2.5)
            ci_upper = np.percentile(diffs, 97.5)

            sig_results[metric] = {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < self.significance_level,
                "ci_95_lower": float(ci_lower),
                "ci_95_upper": float(ci_upper),
            }

        return sig_results

    def _print_ab_report(self, results: Dict, daily_df: pd.DataFrame) -> None:
        """Print formatted A/B test report."""
        print("\n" + "=" * 70)
        print("  A/B TEST RESULTS")
        print("=" * 70)
        print(f"  Duration:              {results['total_days']} days")
        print(f"  Control queries:       {results['total_control_queries']:,}")
        print(f"  Treatment queries:     {results['total_treatment_queries']:,}")
        print(f"  Traffic split:         {self.traffic_split*100:.0f}% treatment")
        print()
        print(f"{'Metric':<20} {'Control':>10} {'Treatment':>10} {'Δ (%)':>10} {'p-value':>10} {'Sig?':>6}")
        print("-" * 70)

        for metric in ["ctr", "ndcg10", "avg_dwell", "no_click_rate"]:
            control = results.get(f"control_avg_{metric}", 0)
            treatment = results.get(f"treatment_avg_{metric}", 0)
            delta = results.get(f"delta_{metric}_pct", 0)
            sig = results.get("significance", {}).get(metric, {})
            p_val = sig.get("p_value", 1.0)
            is_sig = "YES ✓" if sig.get("significant", False) else "NO"

            print(f"{metric:<20} {control:>10.4f} {treatment:>10.4f} {delta:>+9.1f}% {p_val:>10.4f} {is_sig:>6}")

        print("=" * 70)

        # Decision recommendation
        ndcg_delta = results.get("delta_ndcg10_pct", 0)
        ctr_delta = results.get("delta_ctr_pct", 0)
        ndcg_sig = results.get("significance", {}).get("ndcg10", {}).get("significant", False)

        print("\n  RECOMMENDATION:")
        if ndcg_delta > 0 and ctr_delta > 0 and ndcg_sig:
            print("  ✅ ROLLOUT — Treatment shows significant improvement across metrics.")
        elif ndcg_delta > 0 and ndcg_sig:
            print("  ⚠️  CONSIDER ROLLOUT — NDCG improved but other metrics mixed.")
        else:
            print("  ❌ ROLLBACK — Insufficient evidence of improvement.")
        print()
