import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config as cfg
from config import (
    clean_ingredients_v2, ingredient_keys, to_features,
    RANDOM_STATE, N_CANDIDATES, ML_FEATURES, ML_NUMERIC, ML_CATEGORICAL,
    RECIPES_CSV, dataset_sha256,
    load_data,
)
from recommender import ItalianRecipeRecommender, get_m3_best
from models import get_default_models, get_tuned_models, make_preprocessor, make_pipeline


# Expose constants as fixtures (use cfg to avoid name collision)
@pytest.fixture
def GENERIC_WORDS():
    return cfg.GENERIC_WORDS


@pytest.fixture
def STAPLES():
    return cfg.STAPLES


@pytest.fixture
def FRACTIONS():
    return cfg.FRACTIONS


@pytest.fixture
def UNITS():
    return cfg.UNITS


@pytest.fixture
def PREPARATION():
    return cfg.PREPARATION


@pytest.fixture
def PASTA_NAMES():
    return cfg.PASTA_NAMES


@pytest.fixture
def ALIASES():
    return cfg.ALIASES


@pytest.fixture
def DATA_TYPOS():
    return cfg.DATA_TYPOS


@pytest.fixture
def NO_SINGULAR():
    return cfg.NO_SINGULAR


@pytest.fixture
def SALT_PEPPER():
    return cfg.SALT_PEPPER


@pytest.fixture(scope="session", autouse=True)
def _load_data():
    load_data()


@pytest.fixture(scope="session")
def sample_recipes():
    """Minimal recipe data for testing."""
    return pd.DataFrame({
        "recipe_title": [
            "Spaghetti Carbonara",
            "Margherita Pizza",
            "Chicken Parmesan",
        ],
        "ingredients": [
            '["spaghetti", "eggs", "pancetta", "parmesan", "black pepper"]',
            '["pizza dough", "tomatoes", "mozzarella", "basil", "olive oil"]',
            '["chicken breast", "bread crumbs", "parmesan", "marinara sauce", "mozzarella"]',
        ],
        "dish_group": ["Pasta", "Pizza", "Chicken"],
        "difficulty": ["Medium", "Easy", "Medium"],
        "n_steps": [5, 4, 6],
        "est_prep_time_min": [10, 15, 20],
        "est_cook_time_min": [15, 12, 25],
    })


@pytest.fixture(scope="session")
def sample_ingredients():
    """Sample ingredient lists for testing."""
    return [
        ["spaghetti", "eggs", "pancetta", "parmesan", "black pepper"],
        ["pizza dough", "tomatoes", "mozzarella", "basil", "olive oil"],
        ["chicken breast", "bread crumbs", "parmesan", "marinara sauce", "mozzarella"],
    ]


@pytest.fixture
def sample_pantry():
    return ["spaghetti", "eggs", "parmesan", "black pepper", "olive oil"]


@pytest.fixture
def m3_model(sample_ingredients):
    model = ItalianRecipeRecommender(mode="both", use_idf=True, drop_staples=False, alpha=0.6)
    model.fit(sample_ingredients)
    return model


@pytest.fixture
def default_models():
    return get_default_models()


@pytest.fixture
def tuned_models():
    return get_tuned_models()


@pytest.fixture
def preprocessor():
    return make_preprocessor(scale=True)