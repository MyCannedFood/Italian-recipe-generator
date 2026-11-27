import numpy as np
import pytest


class TestItalianRecipeRecommender:
   
    def test_init_defaults(self):
        from recommender import ItalianRecipeRecommender
        model = ItalianRecipeRecommender()
        assert model.mode == "both"
        assert model.use_idf is True
        assert model.drop_staples is False
        assert model.alpha == 1.0

    def test_init_custom(self):
        from recommender import ItalianRecipeRecommender
        model = ItalianRecipeRecommender(mode="ingredient", use_idf=False, drop_staples=True, alpha=0.5)
        assert model.mode == "ingredient"
        assert model.use_idf is False
        assert model.drop_staples is True
        assert model.alpha == 0.5

    def test_fit_sets_attributes(self, m3_model, sample_ingredients):
        assert m3_model.recipes_ is not None
        assert m3_model.vectorizer_ is not None
        assert m3_model.item_matrix_ is not None
        assert m3_model.ingredient_index_ is not None
        assert m3_model.key_index_ is not None
        assert m3_model.ingredient_keys_ is not None
        assert m3_model.recipe_ingredients_ is not None
        assert m3_model.recipe_lengths_ is not None
        assert len(m3_model.recipes_) == len(sample_ingredients)

    def test_score_returns_array(self, m3_model):
        query = [["spaghetti", "eggs", "parmesan"]]
        scores = m3_model.score(query)
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (1, len(m3_model.recipes_))
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)

    def test_recommend_returns_tuples(self, m3_model):
        query = ["spaghetti", "eggs", "parmesan"]
        results = m3_model.recommend(query, k=2)
        assert len(results) == 2
        for idx, score, coverage in results:
            assert isinstance(idx, (int, np.integer))
            assert isinstance(score, float)
            assert isinstance(coverage, float)
            assert 0 <= coverage <= 1

    def test_recommend_respects_k(self, m3_model):
        query = ["spaghetti", "eggs", "parmesan"]
        results = m3_model.recommend(query, k=1)
        assert len(results) == 1

    def test_alpha_zero_uses_coverage_only(self, sample_ingredients):
        from recommender import ItalianRecipeRecommender
        model = ItalianRecipeRecommender(mode="both", use_idf=True, drop_staples=False, alpha=0.0)
        model.fit(sample_ingredients)
        query = ["spaghetti", "eggs", "parmesan"]
        results = model.recommend(query, k=2)
        # With alpha=0, score should equal coverage
        for idx, score, coverage in results:
            assert abs(score - coverage) < 1e-6

    def test_alpha_one_uses_cosine_only(self, sample_ingredients):
        from recommender import ItalianRecipeRecommender
        model = ItalianRecipeRecommender(mode="both", use_idf=True, drop_staples=False, alpha=1.0)
        model.fit(sample_ingredients)
        query = ["spaghetti", "eggs", "parmesan"]
        scores = model.score([query])[0]
        results = model.recommend(query, k=2)
        for idx, score, coverage in results:
            assert abs(score - scores[idx]) < 1e-6


@pytest.mark.integration
class TestOverlapTopK:
    
    def test_returns_indices(self, sample_ingredients):
        from recommender import overlap_top_k
        queries = [["spaghetti", "eggs"]]
        result = overlap_top_k(queries, k=2)
        assert result.shape == (1, 2)
        assert np.all(result >= 0)
        assert np.all(result < len(sample_ingredients))

    def test_k_respected(self, sample_ingredients):
        from recommender import overlap_top_k
        queries = [["spaghetti"]]
        result = overlap_top_k(queries, k=1)
        assert result.shape == (1, 1)


@pytest.mark.integration
class TestKNJaccardTopK:
    
    def test_returns_indices(self, sample_ingredients):
        from recommender import knn_jaccard_top_k
        queries = [["spaghetti", "eggs"]]
        result = knn_jaccard_top_k(queries, k=2)
        assert result.shape == (1, 2)
        assert np.all(result >= 0)
        assert np.all(result < len(sample_ingredients))


@pytest.mark.integration
class TestGetM3Best:
    
    def test_returns_fitted_model(self):
        from recommender import get_m3_best
        model = get_m3_best()
        assert isinstance(model, ItalianRecipeRecommender)
        assert model.vectorizer_ is not None
        assert model.item_matrix_ is not None

    def test_best_params_match_notebook(self):
        from recommender import get_m3_best
        model = get_m3_best()
        # Notebook best params: mode="both", use_idf=True, drop_staples=False, alpha=0.6
        assert model.mode == "both"
        assert model.use_idf is True
        assert model.drop_staples is False
        assert model.alpha == 0.6