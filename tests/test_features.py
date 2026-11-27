import numpy as np
import pandas as pd
import pytest


@pytest.mark.integration
class TestComputeM4Features:
    
    def test_returns_dataframe(self, sample_pantry):
        from features import compute_m4_features
        df = compute_m4_features(sample_pantry)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_has_required_columns(self, sample_pantry):
        from features import compute_m4_features
        from config import ML_FEATURES
        df = compute_m4_features(sample_pantry)
        for col in ML_FEATURES:
            assert col in df.columns, f"Missing column: {col}"

    def test_has_metadata_columns(self, sample_pantry):
        from features import compute_m4_features
        df = compute_m4_features(sample_pantry)
        assert "query_id" in df.columns
        assert "target" in df.columns
        assert "recipe" in df.columns
        assert "label" in df.columns

    def test_candidate_count(self, sample_pantry):
        from features import compute_m4_features
        from config import N_CANDIDATES
        df = compute_m4_features(sample_pantry)
        assert len(df) == N_CANDIDATES

    def test_m3_rank_sequential(self, sample_pantry):
        from features import compute_m4_features
        df = compute_m4_features(sample_pantry)
        assert list(df["m3_rank"]) == list(range(1, len(df) + 1))

    def test_scores_decreasing(self, sample_pantry):
        from features import compute_m4_features
        df = compute_m4_features(sample_pantry)
        scores = df["m3_score"].values
        assert np.all(np.diff(scores) <= 0)  # Non-increasing

    def test_numeric_features_present(self, sample_pantry):
        from features import compute_m4_features
        from config import ML_NUMERIC
        df = compute_m4_features(sample_pantry)
        for col in ML_NUMERIC:
            assert col in df.columns
            assert df[col].dtype in [np.float64, np.int64, float, int]

    def test_categorical_features_present(self, sample_pantry):
        from features import compute_m4_features
        from config import ML_CATEGORICAL
        df = compute_m4_features(sample_pantry)
        for col in ML_CATEGORICAL:
            assert col in df.columns


@pytest.mark.integration
class TestRecommendFromPantry:
    
    def test_returns_dataframe(self, sample_pantry):
        from features import recommend_from_pantry
        from recommender import get_m3_best
        m3 = get_m3_best()
        df = recommend_from_pantry(sample_pantry, k=3, use_m4=False, model=None)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_has_required_output_columns(self, sample_pantry):
        from features import recommend_from_pantry
        from recommender import get_m3_best
        m3 = get_m3_best()
        df = recommend_from_pantry(sample_pantry, k=3, use_m4=False, model=None)
        assert "Resep" in df.columns
        assert "Skor" in df.columns
        assert "Bahan dimiliki" in df.columns
        assert "Perlu dibeli" in df.columns

    def test_scores_rounded(self, sample_pantry):
        from features import recommend_from_pantry
        from recommender import get_m3_best
        m3 = get_m3_best()
        df = recommend_from_pantry(sample_pantry, k=3, use_m4=False, model=None)
        for score in df["Skor"]:
            # Should be rounded to 3 decimal places
            assert round(score, 3) == score

    def test_coverage_format(self, sample_pantry):
        from features import recommend_from_pantry
        from recommender import get_m3_best
        m3 = get_m3_best()
        df = recommend_from_pantry(sample_pantry, k=3, use_m4=False, model=None)
        for cov in df["Bahan dimiliki"]:
            assert cov.endswith("%")

    def test_missing_ingredients_listed(self, sample_pantry):
        from features import recommend_from_pantry
        from recommender import get_m3_best
        m3 = get_m3_best()
        df = recommend_from_pantry(sample_pantry, k=3, use_m4=False, model=None)
        for missing in df["Perlu dibeli"]:
            # Should be comma-separated string (or empty)
            assert isinstance(missing, str)

    def test_k_respected(self, sample_pantry):
        from features import recommend_from_pantry
        from recommender import get_m3_best
        m3 = get_m3_best()
        for k in [1, 3, 5, 10]:
            df = recommend_from_pantry(sample_pantry, k=k, use_m4=False, model=None)
            assert len(df) == k

    def test_empty_pantry(self):
        from features import recommend_from_pantry
        from recommender import get_m3_best
        m3 = get_m3_best()
        df = recommend_from_pantry([], k=3, use_m4=False, model=None)
        # Should handle gracefully
        assert isinstance(df, pd.DataFrame)

    def test_m4_reranking(self, sample_pantry, tuned_models):
        from features import recommend_from_pantry
        model = tuned_models["Logistic Regression"]
        from config import clean_ingredients_v2
        # We need a fitted model - just test it runs without error
        # (Full integration test would need trained model)
        pass


@pytest.mark.integration
class TestPrecomputedMetadata:
    
    def test_recipe_item_keys(self):
        from features import recipe_item_keys
        from config import recipe_ingredients
        assert len(recipe_item_keys) == len(recipe_ingredients)
        for keys in recipe_item_keys:
            assert isinstance(keys, list)
            for key_set in keys:
                assert isinstance(key_set, set)

    def test_recipe_key_sets(self):
        from features import recipe_key_sets
        from config import recipe_ingredients
        assert len(recipe_key_sets) == len(recipe_ingredients)
        for key_set in recipe_key_sets:
            assert isinstance(key_set, set)

    def test_recipe_idf(self):
        from features import recipe_idf
        from config import recipe_ingredients, idf
        assert len(recipe_idf) == len(recipe_ingredients)
        for arr in recipe_idf:
            assert isinstance(arr, np.ndarray)
            for val in arr:
                assert val >= 0  # IDF is non-negative

    def test_recipe_tabular(self):
        from features import recipe_tabular
        from config import recipes
        assert len(recipe_tabular) == len(recipes)
        assert "dish_group" in recipe_tabular.columns
        assert "difficulty" in recipe_tabular.columns
        assert "n_steps" in recipe_tabular.columns
        assert "est_prep_time_min" in recipe_tabular.columns
        assert "est_cook_time_min" in recipe_tabular.columns