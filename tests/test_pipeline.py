"""
Tests for data pipeline.
"""

import numpy as np
import pandas as pd
import pytest
from src.data.generate_synthetic import generate_queries, generate_documents
from src.data.label_extraction import compute_relevance_grades


class TestDataGeneration:
    def test_generate_queries(self):
        df = generate_queries(100, seed=42)
        assert len(df) == 100
        assert "qid" in df.columns
        assert "query_text" in df.columns
        assert "query_type" in df.columns
        assert df["query_type"].isin([0, 1, 2]).all()

    def test_generate_documents(self):
        df = generate_documents(50, seed=42)
        assert len(df) == 50
        assert "doc_id" in df.columns
        assert "pagerank" in df.columns
        assert all(0 <= pr <= 1 for pr in df["pagerank"])

    def test_query_uniqueness(self):
        df = generate_queries(200, seed=42)
        assert df["qid"].nunique() == len(df)


class TestLabelExtraction:
    def test_relevance_grades(self):
        data = pd.DataFrame({
            "clicked": [1, 1, 1, 1, 0],
            "dwell_time": [90, 45, 15, 3, 0],
            "skipped": [0, 0, 0, 0, 1],
        })
        config = {
            "labels": {
                "grade_4_dwell": 60,
                "grade_3_dwell": 30,
                "grade_2_dwell": 10,
            }
        }
        result = compute_relevance_grades(data, config)
        assert result["relevance_grade"].tolist() == [4, 3, 2, 1, 0]

    def test_grades_bounded(self):
        data = pd.DataFrame({
            "clicked": np.random.randint(0, 2, 100),
            "dwell_time": np.random.exponential(30, 100),
            "skipped": np.random.randint(0, 2, 100),
        })
        config = {"labels": {"grade_4_dwell": 60, "grade_3_dwell": 30, "grade_2_dwell": 10}}
        result = compute_relevance_grades(data, config)
        assert all(0 <= g <= 4 for g in result["relevance_grade"])
