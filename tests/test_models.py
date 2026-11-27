import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator


class TestGroupTimeImputer:
    
    def test_fit_transform(self):
        from models import GroupTimeImputer
        df = pd.DataFrame({
            "dish_group": ["Pasta", "Pasta", "Pizza", "Pizza"],
            "est_prep_time_min": [10, 20, np.nan, 15],
            "est_cook_time_min": [15, np.nan, 12, 18],
        })
        imputer = GroupTimeImputer()
        result = imputer.fit_transform(df)
        
        assert result.loc[0, "est_prep_time_min"] == 10  # Already filled
        assert result.loc[1, "est_prep_time_min"] == 20
        assert result.loc[2, "est_prep_time_min"] == 15  # Imputed from Pizza median
        assert result.loc[3, "est_prep_time_min"] == 15

    def test_global_fallback(self):
        from models import GroupTimeImputer
        # Group with all NaN should fall back to global median
        df = pd.DataFrame({
            "dish_group": ["Pasta", "Unknown"],
            "est_prep_time_min": [10, np.nan],
            "est_cook_time_min": [15, np.nan],
        })
        imputer = GroupTimeImputer()
        result = imputer.fit_transform(df)
        # Unknown group has no median, falls back to global (10)
        assert result.loc[1, "est_prep_time_min"] == 10

    def test_fit_transform_returns_dataframe(self):
        from models import GroupTimeImputer
        df = pd.DataFrame({
            "dish_group": ["Pasta", "Pasta"],
            "est_prep_time_min": [10, 20],
            "est_cook_time_min": [15, 25],
        })
        imputer = GroupTimeImputer()
        result = imputer.fit_transform(df)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == list(df.columns)


class TestMakePreprocessor:
    
    def test_returns_column_transformer(self):
        from models import make_preprocessor
        from config import ML_NUMERIC, ML_CATEGORICAL
        preprocessor = make_preprocessor(scale=True)
        assert hasattr(preprocessor, "fit_transform")
        assert hasattr(preprocessor, "transform")

    def test_transforms_numeric_and_categorical(self):
        from models import make_preprocessor
        from config import ML_NUMERIC, ML_CATEGORICAL
        import pandas as pd
        import numpy as np
        preprocessor = make_preprocessor(scale=True)
        # Create dummy M4 feature data matching ML_FEATURES
        n_samples = 10
        X = pd.DataFrame({
            **{col: np.random.rand(n_samples) for col in ML_NUMERIC},
            **{col: np.random.choice(["A", "B", "C"], n_samples) for col in ML_CATEGORICAL},
        })
        X_transformed = preprocessor.fit_transform(X)
        assert isinstance(X_transformed, np.ndarray)
        assert X_transformed.shape[0] == n_samples
        # Numeric features scaled + categorical one-hot
        assert X_transformed.shape[1] > len(ML_NUMERIC)


class TestMakePipeline:
    
    def test_returns_pipeline(self):
        from models import make_pipeline
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(random_state=42)
        pipeline = make_pipeline(clf, scale=True)
        assert hasattr(pipeline, "fit")
        assert hasattr(pipeline, "predict")
        assert hasattr(pipeline, "predict_proba")

    def test_pipeline_steps(self):
        from models import make_pipeline
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(random_state=42)
        pipeline = make_pipeline(clf, scale=True)
        step_names = [name for name, _ in pipeline.steps]
        assert "time_impute" in step_names
        assert "prep" in step_names
        assert "clf" in step_names


class TestDefaultModels:
    
    def test_returns_four_models(self):
        from models import get_default_models
        models = get_default_models()
        assert len(models) == 4
        assert "Logistic Regression" in models
        assert "Random Forest" in models
        assert "HistGradientBoosting" in models
        assert "XGBoost" in models

    def test_all_are_pipelines(self):
        from models import get_default_models
        from sklearn.pipeline import Pipeline
        models = get_default_models()
        for name, model in models.items():
            assert isinstance(model, Pipeline), f"{name} is not a Pipeline"


class TestTunedModels:
    
    def test_returns_four_models(self):
        from models import get_tuned_models
        models = get_tuned_models()
        assert len(models) == 4
        assert "Logistic Regression" in models
        assert "Random Forest" in models
        assert "HistGradientBoosting" in models
        assert "XGBoost" in models

    def test_logistic_regression_params(self):
        from models import get_tuned_models, TUNED_PARAMS
        models = get_tuned_models()
        lr = models["Logistic Regression"]
        # Check that tuned params are used
        clf = lr.named_steps["clf"]
        assert clf.C == TUNED_PARAMS["Logistic Regression"]["C"]
        assert clf.penalty == TUNED_PARAMS["Logistic Regression"]["penalty"]


class TestXGBQueryRanker:
    
    def test_init(self):
        from models import XGBQueryRanker
        ranker = XGBQueryRanker(n_estimators=10, max_depth=3)
        assert ranker.n_estimators == 10
        assert ranker.max_depth == 3

    def test_fit_predict(self):
        from models import XGBQueryRanker
        from config import ML_FEATURES, ML_NUMERIC, ML_CATEGORICAL
        # Create dummy data
        np.random.seed(42)
        n_samples = 20
        X = pd.DataFrame({
            "query_id": np.repeat([0, 1], 10),
            **{col: np.random.rand(n_samples) for col in ML_NUMERIC},
            **{col: np.random.choice(["A", "B"], n_samples) for col in ML_CATEGORICAL},
        })
        y = np.random.randint(0, 2, n_samples)
        ranker = XGBQueryRanker(n_estimators=5, max_depth=2, random_state=42)
        ranker.fit(X, y)
        scores = ranker.decision_function(X)
        assert scores.shape == (n_samples,)
        proba = ranker.predict_proba(X)
        assert proba.shape == (n_samples, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_classes_attribute(self):
        from models import XGBQueryRanker
        from config import ML_FEATURES, ML_NUMERIC, ML_CATEGORICAL
        np.random.seed(42)
        n_samples = 20
        X = pd.DataFrame({
            "query_id": np.repeat([0, 1], 10),
            **{col: np.random.rand(n_samples) for col in ML_NUMERIC},
            **{col: np.random.choice(["A", "B"], n_samples) for col in ML_CATEGORICAL},
        })
        y = np.random.randint(0, 2, n_samples)
        ranker = XGBQueryRanker(n_estimators=5, max_depth=2, random_state=42)
        ranker.fit(X, y)
        assert hasattr(ranker, "classes_")
        assert list(ranker.classes_) == [0, 1]


# Import sample_recipes fixture from conftest
from tests.conftest import sample_recipes  # noqa: F401