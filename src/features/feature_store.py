"""
Feature store — registry and pipeline for all features.
Manages feature groups, builds final feature matrix.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import logging

from src.features.query_features import QueryFeatureExtractor
from src.features.doc_features import extract_doc_features
from src.features.interaction_features import extract_interaction_features

logger = logging.getLogger(__name__)

# Feature registry: name → group mapping
FEATURE_REGISTRY = {
    # Query features
    "query_length_words": "query",
    "query_length_chars": "query",
    "query_type_nav": "query",
    "query_type_info": "query",
    "query_type_trans": "query",
    "tfidf_score": "query",
    "bm25_score": "query",

    # Document features
    "pagerank_score": "document",
    "freshness_score": "document",
    "title_match_exact": "document",
    "title_match_partial": "document",
    "title_overlap_ratio": "document",
    "anchor_text_match": "document",
    "doc_length_norm": "document",
    "url_depth_norm": "document",

    # Interaction features
    "ctr": "interaction",
    "avg_dwell_time_norm": "interaction",
    "skip_rate": "interaction",
    "position_bias_correction": "interaction",
    "historical_ctr": "interaction",
    "click_entropy": "interaction",
    "dwell_time_norm": "interaction",
    "log_impressions": "interaction",
}

ALL_FEATURE_NAMES = list(FEATURE_REGISTRY.keys())

FEATURE_GROUPS = {
    "query": [k for k, v in FEATURE_REGISTRY.items() if v == "query"],
    "document": [k for k, v in FEATURE_REGISTRY.items() if v == "document"],
    "interaction": [k for k, v in FEATURE_REGISTRY.items() if v == "interaction"],
}


class FeatureStore:
    """
    Central feature store for the ranking pipeline.
    Manages feature extraction, registration, and matrix construction.
    """

    def __init__(self):
        self.query_extractor = QueryFeatureExtractor()
        self.feature_names = ALL_FEATURE_NAMES
        self.feature_groups = FEATURE_GROUPS
        self._fitted = False

    def fit(self, doc_titles: List[str]) -> "FeatureStore":
        """Fit feature extractors on document corpus."""
        self.query_extractor.fit(doc_titles)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full feature extraction pipeline.

        1. Extract query features
        2. Extract document features
        3. Extract interaction features

        Args:
            df: Aggregated DataFrame from label extraction

        Returns:
            DataFrame with all features added
        """
        logger.info("=" * 60)
        logger.info("STEP 4: Feature engineering")
        logger.info("=" * 60)

        # Query features
        result = self.query_extractor.extract_features(df)

        # Document features
        result = extract_doc_features(result)

        # Interaction features
        result = extract_interaction_features(result)

        # Verify all features present
        missing = [f for f in self.feature_names if f not in result.columns]
        if missing:
            logger.warning(f"Missing features: {missing}")

        # Log feature statistics
        logger.info(f"\nFeature matrix: {len(result):,} rows × {len(self.feature_names)} features")
        logger.info(f"Feature groups:")
        for group, features in self.feature_groups.items():
            present = [f for f in features if f in result.columns]
            logger.info(f"  {group}: {len(present)} features")

        return result

    def get_feature_matrix(
        self,
        df: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Extract feature matrix (X) from DataFrame.

        Args:
            df: DataFrame with feature columns
            feature_names: Subset of features to use (None = all)

        Returns:
            numpy array of shape (n_samples, n_features)
        """
        if feature_names is None:
            feature_names = self.feature_names

        available = [f for f in feature_names if f in df.columns]
        X = df[available].values.astype(np.float32)

        # Handle NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

        return X

    def get_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Extract label array."""
        return df["relevance_grade"].values.astype(np.float32)

    def get_query_groups(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get query group sizes for XGBoost ranking.
        Documents must be grouped by qid, this returns count per group.
        """
        groups = df.groupby("qid").size().values
        return groups

    def get_qids(self, df: pd.DataFrame) -> np.ndarray:
        """Get query ID array (for XGBRanker qid parameter)."""
        # Encode string qids to integers
        qid_map = {qid: idx for idx, qid in enumerate(sorted(df["qid"].unique()))}
        return df["qid"].map(qid_map).values.astype(np.int32)

    @staticmethod
    def sort_by_qid(df: pd.DataFrame) -> pd.DataFrame:
        """Sort DataFrame by qid — required by XGBoost ranking."""
        return df.sort_values("qid").reset_index(drop=True)

    def get_features_by_group(self, group: str) -> List[str]:
        """Get feature names for a specific group."""
        return self.feature_groups.get(group, [])

    def get_features_excluding_group(self, exclude_group: str) -> List[str]:
        """Get all feature names except those in specified group."""
        return [f for f in self.feature_names if FEATURE_REGISTRY.get(f) != exclude_group]
