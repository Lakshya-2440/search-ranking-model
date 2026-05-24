"""
Tests for evaluation metrics.
"""

import numpy as np
import pytest
from src.evaluation.metrics import dcg_at_k, ndcg_at_k, mean_average_precision, mean_reciprocal_rank


class TestDCG:
    def test_perfect_ranking(self):
        """Perfect ranking should have maximum DCG."""
        rels = np.array([4, 3, 2, 1, 0])
        dcg = dcg_at_k(rels, 5)
        assert dcg > 0

    def test_empty(self):
        """Empty input → 0."""
        assert dcg_at_k(np.array([]), 5) == 0.0

    def test_single_item(self):
        """Single relevant item."""
        dcg = dcg_at_k(np.array([3]), 1)
        expected = (2**3 - 1) / np.log2(2)  # 7 / 1 = 7
        assert abs(dcg - expected) < 1e-6

    def test_k_larger_than_list(self):
        """k > len(rels) should work."""
        rels = np.array([1, 2])
        dcg = dcg_at_k(rels, 10)
        assert dcg > 0


class TestNDCG:
    def test_perfect_ranking(self):
        """Perfect ranking → NDCG = 1.0."""
        rels = np.array([4, 3, 2, 1, 0])
        assert ndcg_at_k(rels, 5) == pytest.approx(1.0)

    def test_worst_ranking(self):
        """Reverse order → NDCG < 1.0."""
        rels = np.array([0, 1, 2, 3, 4])
        assert ndcg_at_k(rels, 5) < 1.0

    def test_all_zero(self):
        """All zeros → NDCG = 0."""
        rels = np.array([0, 0, 0, 0])
        assert ndcg_at_k(rels, 4) == 0.0

    def test_bounded(self):
        """NDCG always in [0, 1]."""
        for _ in range(100):
            rels = np.random.randint(0, 5, size=10)
            ndcg = ndcg_at_k(rels, 10)
            assert 0 <= ndcg <= 1.0 + 1e-9


class TestMAP:
    def test_perfect(self):
        """All relevant at top → MAP = 1."""
        rels = [np.array([3, 2, 1, 0, 0])]
        assert mean_average_precision(rels) == pytest.approx(1.0)

    def test_no_relevant(self):
        """No relevant docs → MAP = 0."""
        rels = [np.array([0, 0, 0, 0])]
        assert mean_average_precision(rels) == 0.0

    def test_multiple_queries(self):
        """Multiple queries averaged."""
        rels = [np.array([1, 0, 1]), np.array([0, 0, 1])]
        map_score = mean_average_precision(rels)
        assert 0 < map_score < 1


class TestMRR:
    def test_first_position(self):
        """First result relevant → MRR = 1."""
        rels = [np.array([3, 0, 0])]
        assert mean_reciprocal_rank(rels) == pytest.approx(1.0)

    def test_second_position(self):
        """Second result relevant → MRR = 0.5."""
        rels = [np.array([0, 2, 0])]
        assert mean_reciprocal_rank(rels) == pytest.approx(0.5)

    def test_no_relevant(self):
        """No relevant → MRR = 0."""
        rels = [np.array([0, 0, 0])]
        assert mean_reciprocal_rank(rels) == 0.0
