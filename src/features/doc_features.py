"""
Document-level features.
PageRank, freshness, title match, anchor text, doc length normalization.
"""

import numpy as np
import pandas as pd
import logging
from typing import Set

logger = logging.getLogger(__name__)


def title_match_exact(query: str, title: str) -> int:
    """Check if query appears exactly in title."""
    return int(query.lower().strip() in title.lower())


def title_match_partial(query: str, title: str) -> float:
    """Fraction of query words found in title."""
    q_words = set(query.lower().split())
    t_words = set(title.lower().split())
    if not q_words:
        return 0.0
    return len(q_words & t_words) / len(q_words)


def title_overlap_ratio(query: str, title: str) -> float:
    """Jaccard similarity between query and title words."""
    q_words = set(query.lower().split())
    t_words = set(title.lower().split())
    union = q_words | t_words
    if not union:
        return 0.0
    return len(q_words & t_words) / len(union)


def anchor_text_match(query: str, domain: str, topic: str, subtopic: str) -> float:
    """
    Simulate anchor text match.
    In production, this would match against actual anchor texts.
    Here we use domain + topic as proxy.
    """
    q_words = set(query.lower().split())
    anchor_words = set(domain.lower().replace(".", " ").split()) | \
                   set(topic.lower().split()) | \
                   set(subtopic.lower().split())
    if not q_words:
        return 0.0
    return len(q_words & anchor_words) / len(q_words)


def doc_length_norm(body_words: int, avg_doc_length: float = 1000.0) -> float:
    """
    Normalized document length.
    Returns ratio of doc length to average, capped at reasonable bounds.
    """
    if avg_doc_length <= 0:
        return 1.0
    ratio = body_words / avg_doc_length
    return np.clip(ratio, 0.01, 10.0)


def freshness_score(freshness_days: int, decay_rate: float = 0.005) -> float:
    """
    Exponential freshness decay.
    Newer docs score higher.
    """
    return np.exp(-decay_rate * freshness_days)


def extract_doc_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all document-level features.

    Adds columns: title_match_exact, title_match_partial, title_overlap_ratio,
                  anchor_text_match, doc_length_norm, freshness_score, url_depth_norm

    Args:
        df: DataFrame with query_text, doc_title, doc_domain, doc_topic,
            doc_subtopic, doc_body_words, doc_freshness_days, doc_url_depth, doc_pagerank

    Returns:
        DataFrame with document features added
    """
    logger.info("Extracting document features...")
    result = df.copy()

    # Average doc length for normalization
    avg_doc_len = result["doc_body_words"].mean()

    # Title match features
    result["title_match_exact"] = result.apply(
        lambda r: title_match_exact(r["query_text"], r["doc_title"]), axis=1
    )
    result["title_match_partial"] = result.apply(
        lambda r: title_match_partial(r["query_text"], r["doc_title"]), axis=1
    )
    result["title_overlap_ratio"] = result.apply(
        lambda r: title_overlap_ratio(r["query_text"], r["doc_title"]), axis=1
    )

    # Anchor text match
    result["anchor_text_match"] = result.apply(
        lambda r: anchor_text_match(
            r["query_text"], r["doc_domain"], r["doc_topic"], r["doc_subtopic"]
        ), axis=1
    )

    # Doc length normalization
    result["doc_length_norm"] = result["doc_body_words"].apply(
        lambda x: doc_length_norm(x, avg_doc_len)
    )

    # Freshness score (exponential decay)
    result["freshness_score"] = result["doc_freshness_days"].apply(freshness_score)

    # URL depth normalization (lower = better, normalize to 0-1)
    max_depth = result["doc_url_depth"].max()
    result["url_depth_norm"] = 1.0 - (result["doc_url_depth"] / max(max_depth, 1))

    # PageRank is already in data, just rename for clarity
    result["pagerank_score"] = result["doc_pagerank"]

    logger.info(f"  Document features extracted for {len(result):,} rows")
    return result
