"""
BM25 baseline ranker.
Ranks documents purely by BM25 keyword matching score.
"""

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class BM25Baseline:
    """
    BM25 baseline ranker for search results.
    Uses BM25Okapi for keyword-based relevance scoring.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.doc_ids = None
        self.doc_titles = None
        self._fitted = False

    def fit(self, doc_ids: List[str], doc_titles: List[str]) -> "BM25Baseline":
        """
        Fit BM25 model on document corpus.

        Args:
            doc_ids: List of document IDs
            doc_titles: List of document title strings
        """
        logger.info("Fitting BM25 baseline...")
        self.doc_ids = doc_ids
        self.doc_titles = doc_titles

        # Tokenize corpus
        tokenized = [title.lower().split() for title in doc_titles]
        self.bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)

        self._fitted = True
        logger.info(f"  BM25 fitted on {len(doc_titles):,} documents")
        return self

    def score(self, query: str) -> np.ndarray:
        """
        Score all documents for a query.

        Args:
            query: Query string

        Returns:
            Array of BM25 scores for all documents
        """
        if not self._fitted:
            raise RuntimeError("BM25 model not fitted. Call fit() first.")
        query_tokens = query.lower().split()
        return self.bm25.get_scores(query_tokens)

    def rank(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Rank documents for a query, return top-k.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of (doc_id, score) tuples, sorted by score descending
        """
        scores = self.score(query)
        top_indices = np.argsort(-scores)[:top_k]
        return [(self.doc_ids[i], scores[i]) for i in top_indices]

    def predict_scores_for_df(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate BM25 ranking scores for each row in DataFrame.
        Used for evaluation against LambdaMART.

        Args:
            df: DataFrame with 'query_text' and 'doc_title' columns

        Returns:
            Array of BM25 scores
        """
        logger.info("Computing BM25 scores for evaluation...")
        scores = np.zeros(len(df))

        for idx, row in df.iterrows():
            query_tokens = row["query_text"].lower().split()
            doc_tokens = row["doc_title"].lower().split()

            # Direct BM25-style scoring: term overlap with TF saturation
            overlap = set(query_tokens) & set(doc_tokens)
            if not query_tokens:
                scores[idx] = 0.0
                continue

            # Simplified BM25 formula
            score = 0.0
            doc_len = len(doc_tokens)
            avg_dl = 10  # Approximate average doc title length

            for term in query_tokens:
                if term in doc_tokens:
                    tf = doc_tokens.count(term)
                    # BM25 TF component
                    tf_norm = (tf * (self.k1 + 1)) / (
                        tf + self.k1 * (1 - self.b + self.b * doc_len / avg_dl)
                    )
                    # IDF approximation (use overlap ratio as proxy)
                    idf = np.log(1 + 1.0 / max(len(overlap), 1))
                    score += tf_norm * idf

            # Add PageRank boost
            if "doc_pagerank" in row.index:
                score += row["doc_pagerank"] * 0.1

            scores[idx] = score

        return scores
