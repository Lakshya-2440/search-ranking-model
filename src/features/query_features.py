"""
Query-level features.
TF-IDF, BM25 score, query length, query type encoding.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class QueryFeatureExtractor:
    """Extract query-level features for ranking."""

    def __init__(self):
        self.tfidf_vectorizer = None
        self.bm25_model = None
        self.corpus_tokens = None
        self._fitted = False

    def fit(self, doc_titles: List[str]) -> "QueryFeatureExtractor":
        """
        Fit TF-IDF and BM25 models on document corpus.

        Args:
            doc_titles: List of document title strings
        """
        logger.info("Fitting query feature extractors on corpus...")

        # TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.tfidf_vectorizer.fit(doc_titles)

        # BM25
        self.corpus_tokens = [title.lower().split() for title in doc_titles]
        self.bm25_model = BM25Okapi(self.corpus_tokens)

        self._fitted = True
        logger.info(f"  TF-IDF vocab size: {len(self.tfidf_vectorizer.vocabulary_):,}")
        logger.info(f"  BM25 corpus size: {len(self.corpus_tokens):,}")
        return self

    def compute_tfidf_score(self, query: str, doc_title: str) -> float:
        """Compute TF-IDF cosine similarity between query and doc title."""
        if not self._fitted:
            return 0.0
        try:
            q_vec = self.tfidf_vectorizer.transform([query.lower()])
            d_vec = self.tfidf_vectorizer.transform([doc_title.lower()])
            score = (q_vec * d_vec.T).toarray()[0, 0]
            return float(score)
        except Exception:
            return 0.0

    def compute_bm25_score(self, query: str, doc_idx: int) -> float:
        """Compute BM25 score for query against specific document."""
        if not self._fitted:
            return 0.0
        try:
            query_tokens = query.lower().split()
            scores = self.bm25_model.get_scores(query_tokens)
            if doc_idx < len(scores):
                return float(scores[doc_idx])
            return 0.0
        except Exception:
            return 0.0

    def compute_bm25_scores_batch(self, query: str) -> np.ndarray:
        """Compute BM25 scores for query against all documents."""
        if not self._fitted:
            return np.zeros(len(self.corpus_tokens))
        query_tokens = query.lower().split()
        return self.bm25_model.get_scores(query_tokens)

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all query features for DataFrame.

        Adds columns: query_length_words, query_length_chars, query_type_enc,
                      tfidf_score, bm25_score

        Args:
            df: DataFrame with 'query_text', 'query_type', 'doc_title' columns

        Returns:
            DataFrame with query features added
        """
        logger.info("Extracting query features...")
        result = df.copy()

        # Query length features
        result["query_length_words"] = result["query_text"].apply(lambda x: len(x.split()))
        result["query_length_chars"] = result["query_text"].apply(len)

        # Query type one-hot (already numeric 0/1/2 in data)
        result["query_type_nav"] = (result["query_type"] == 0).astype(int)
        result["query_type_info"] = (result["query_type"] == 1).astype(int)
        result["query_type_trans"] = (result["query_type"] == 2).astype(int)

        # TF-IDF scores (batch compute)
        logger.info("  Computing TF-IDF scores...")
        tfidf_scores = []
        for _, row in result.iterrows():
            score = self.compute_tfidf_score(row["query_text"], row["doc_title"])
            tfidf_scores.append(score)
        result["tfidf_score"] = tfidf_scores

        # BM25 scores — precompute per unique query then map
        logger.info("  Computing BM25 scores...")
        unique_queries = result["query_text"].unique()
        bm25_cache: Dict[str, np.ndarray] = {}

        for q in unique_queries:
            bm25_cache[q] = self.compute_bm25_scores_batch(q)

        # Map BM25 scores — use doc title match as proxy
        bm25_scores = []
        for _, row in result.iterrows():
            query_tokens = row["query_text"].lower().split()
            doc_tokens = row["doc_title"].lower().split()
            # Direct BM25-like score using word overlap weighted by IDF
            overlap = set(query_tokens) & set(doc_tokens)
            if len(query_tokens) > 0:
                score = len(overlap) / len(query_tokens)
                # Scale by cached BM25 mean for this query
                cached = bm25_cache.get(row["query_text"])
                if cached is not None and cached.mean() > 0:
                    score *= (cached.mean() * 2)
            else:
                score = 0.0
            bm25_scores.append(score)
        result["bm25_score"] = bm25_scores

        logger.info(f"  Query features extracted for {len(result):,} rows")
        return result
