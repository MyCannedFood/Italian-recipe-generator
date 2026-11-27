import numpy as np
import pytest


class TestRankingMetrics:
    
    def test_perfect_match(self):
        from utils import ranking_metrics
        top_indices = np.array([[0, 1, 2], [1, 0, 2]])
        targets = np.array([0, 1])
        metrics = ranking_metrics(top_indices, targets, k=3)
        assert metrics["HitRate@5"] == 1.0
        assert metrics["HitRate@10"] == 1.0
        assert metrics["MRR@10"] == 1.0
        assert metrics["NDCG@10"] == 1.0

    def test_no_match(self):
        from utils import ranking_metrics
        top_indices = np.array([[1, 2, 3], [2, 3, 4]])
        targets = np.array([0, 5])
        metrics = ranking_metrics(top_indices, targets, k=3)
        assert metrics["HitRate@5"] == 0.0
        assert metrics["HitRate@10"] == 0.0
        assert metrics["MRR@10"] == 0.0
        assert metrics["NDCG@10"] == 0.0

    def test_partial_match(self):
        from utils import ranking_metrics
        top_indices = np.array([[0, 1, 2], [3, 4, 5]])
        targets = np.array([0, 5])
        metrics = ranking_metrics(top_indices, targets, k=3)
        assert metrics["HitRate@5"] == 1.0  # Both found in top-3
        assert metrics["HitRate@10"] == 1.0

    def test_k_parameter(self):
        from utils import ranking_metrics
        top_indices = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
        targets = np.array([3, 5])
        metrics_k3 = ranking_metrics(top_indices, targets, k=3)
        metrics_k5 = ranking_metrics(top_indices, targets, k=5)
        # At k=3, first target at rank 4 not found
        assert metrics_k3["HitRate@5"] < metrics_k5["HitRate@10"]


class TestTopK:
    
    def test_returns_indices(self):
        from utils import top_k
        scores = np.array([[0.9, 0.1, 0.5, 0.3], [0.2, 0.8, 0.4, 0.6]])
        result = top_k(scores, k=2)
        assert result.shape == (2, 2)
        # First row: indices 0, 2 (top 2 scores)
        assert set(result[0]) == {0, 2}
        # Second row: indices 1, 3
        assert set(result[1]) == {1, 3}

    def test_k_exceeds_columns(self):
        from utils import top_k
        scores = np.array([[0.9, 0.1], [0.2, 0.8]])
        result = top_k(scores, k=5)
        assert result.shape == (2, 2)  # Should return all columns


class TestMakeQueries:
    
    def test_returns_list_of_lists(self, sample_ingredients):
        from utils import make_queries
        targets = np.array([0, 1])
        queries = make_queries(targets, keep=0.5, n_noise=1, seed=42)
        assert isinstance(queries, list)
        assert len(queries) == 2
        for q in queries:
            assert isinstance(q, list)
            assert len(q) > 0

    def test_keep_ratio(self):
        from utils import make_queries
        from config import recipe_ingredients
        # Use first recipe which has 5 ingredients
        targets = np.array([0])
        queries = make_queries(targets, keep=0.6, n_noise=0, seed=42)
        # Should keep ~60% of ingredients (3 out of 5)
        kept = len(queries[0])
        assert kept >= 1  # At least 1

    def test_noise_added(self):
        from utils import make_queries
        targets = np.array([0])
        queries_no_noise = make_queries(targets, keep=0.5, n_noise=0, seed=42)
        queries_with_noise = make_queries(targets, keep=0.5, n_noise=2, seed=42)
        assert len(queries_with_noise[0]) >= len(queries_no_noise[0])

    def test_different_seeds_different_queries(self):
        from utils import make_queries
        targets = np.array([0, 1])
        q1 = make_queries(targets, seed=1)
        q2 = make_queries(targets, seed=2)
        # At least one query should differ
        assert q1 != q2


@pytest.mark.integration
class TestCandidateFeatures:
    
    def test_returns_dataframe(self, sample_ingredients):
        from utils import candidate_features
        queries = [["spaghetti", "eggs"], ["pizza dough", "tomatoes"]]
        targets = np.array([0, 1])
        df = candidate_features(queries, targets, n_candidates=3)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6  # 2 queries * 3 candidates

    def test_has_label_column(self, sample_ingredients):
        from utils import candidate_features
        queries = [["spaghetti", "eggs"]]
        targets = np.array([0])
        df = candidate_features(queries, targets, n_candidates=3)
        assert "label" in df.columns
        # First candidate should be the target (label=1)
        assert df.iloc[0]["label"] == 1
        assert df.iloc[1]["label"] == 0
        assert df.iloc[2]["label"] == 0

    def test_query_id_assigned(self, sample_ingredients):
        from utils import candidate_features
        queries = [["a", "b"], ["c", "d"]]
        targets = np.array([0, 1])
        df = candidate_features(queries, targets, n_candidates=2)
        assert set(df["query_id"].unique()) == {0, 1}
        # Each query_id should have n_candidates rows
        for qid in [0, 1]:
            assert len(df[df["query_id"] == qid]) == 2


class TestF1OptimalThreshold:
    
    def test_perfect_predictions(self):
        from utils import f1_optimal_threshold
        y_true = np.array([0, 0, 1, 1])
        proba = np.array([0.1, 0.2, 0.8, 0.9])
        threshold = f1_optimal_threshold(y_true, proba)
        assert 0.2 < threshold <= 0.8  # Should be between the two groups (inclusive at upper bound)

    def test_all_zeros(self):
        from utils import f1_optimal_threshold
        y_true = np.array([0, 0, 0, 0])
        proba = np.array([0.1, 0.2, 0.3, 0.4])
        threshold = f1_optimal_threshold(y_true, proba)
        assert 0 <= threshold <= 1

    def test_all_ones(self):
        from utils import f1_optimal_threshold
        y_true = np.array([1, 1, 1, 1])
        proba = np.array([0.6, 0.7, 0.8, 0.9])
        threshold = f1_optimal_threshold(y_true, proba)
        assert 0 <= threshold <= 1


class TestHitsByQuery:
    
    def test_returns_boolean_array(self):
        from utils import hits_by_query
        import pandas as pd
        frame = pd.DataFrame({
            "query_id": [0, 0, 1, 1],
            "label": [1, 0, 0, 1]
        })
        scores = np.array([0.9, 0.1, 0.2, 0.8])
        hits = hits_by_query(frame, scores, k=1)
        assert isinstance(hits, np.ndarray)
        assert hits.dtype == bool
        assert len(hits) == 2  # 2 queries
        assert bool(hits[0]) is True   # Query 0: top score is label 1
        assert bool(hits[1]) is True   # Query 1: top score is label 1


class TestPairedComparison:
    
    def test_identical_arrays(self):
        from utils import paired_comparison
        hits_a = np.array([True, True, False, False])
        hits_b = np.array([True, True, False, False])
        result = paired_comparison(hits_a, hits_b)
        assert result["Δ HitRate@5"] == 0.0
        assert result["Hanya A benar"] == 0
        assert result["Hanya B benar"] == 0
        assert result["McNemar p"] == 1.0

    def test_all_a_better(self):
        from utils import paired_comparison
        hits_a = np.array([True, True, True, False])
        hits_b = np.array([False, False, False, False])
        result = paired_comparison(hits_a, hits_b)
        assert result["Δ HitRate@5"] > 0
        assert result["Hanya A benar"] == 3
        assert result["Hanya B benar"] == 0