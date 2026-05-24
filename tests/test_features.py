"""
Tests for feature engineering.
"""

import numpy as np
import pandas as pd
import pytest
from src.features.doc_features import (
    title_match_exact, title_match_partial, title_overlap_ratio,
    doc_length_norm, freshness_score,
)


class TestDocFeatures:
    def test_title_match_exact_present(self):
        assert title_match_exact("python tutorial", "Best Python Tutorial 2024") == 1

    def test_title_match_exact_absent(self):
        assert title_match_exact("java guide", "Python Tutorial") == 0

    def test_title_match_partial(self):
        score = title_match_partial("python machine learning", "Python Tutorial ML")
        assert 0 <= score <= 1

    def test_title_match_partial_no_overlap(self):
        assert title_match_partial("java", "python tutorial") == 0.0

    def test_title_overlap_ratio(self):
        # Same words → Jaccard = 1.0
        assert title_overlap_ratio("python", "python") == pytest.approx(1.0)

    def test_title_overlap_ratio_no_match(self):
        assert title_overlap_ratio("java", "python") == 0.0

    def test_doc_length_norm_average(self):
        """Average length doc → norm ≈ 1.0."""
        assert doc_length_norm(1000, 1000.0) == pytest.approx(1.0)

    def test_doc_length_norm_capped(self):
        """Very long doc → capped at 10."""
        assert doc_length_norm(100000, 100.0) == 10.0

    def test_freshness_score_new(self):
        """Brand new doc → score close to 1."""
        assert freshness_score(0) == pytest.approx(1.0)

    def test_freshness_score_old(self):
        """Old doc → score close to 0."""
        assert freshness_score(1000) < 0.01

    def test_freshness_score_bounded(self):
        """Score always in (0, 1]."""
        for days in [0, 1, 10, 100, 365, 3650]:
            s = freshness_score(days)
            assert 0 < s <= 1.0
