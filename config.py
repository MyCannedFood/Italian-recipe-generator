from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# Paths
DATA_DIR = Path(__file__).parent
RECIPES_CSV = DATA_DIR / "recipes_extended.csv"

# Preprocessing constants
GENERIC_WORDS = {
    "pepper", "sauce", "oil", "cheese", "powder", "juice", "seasoning", "vinegar", "broth",
    "stock", "paste", "extract", "flake", "seed", "leaf", "mix", "dressing", "crumb", "zest", "wine",
}

STAPLES = {"salt", "black pepper", "water", "cooking spray"}

RANDOM_STATE = 42
N_JOBS = 2
N_CANDIDATES = 50

# M4 feature columns 
ML_NUMERIC = [
    "m3_rank", "m3_score", "score_gap",
    "cos_both", "cos_ingredient",
    "exact_overlap", "jaccard",
    "coverage", "n_missing",
    "query_used", "idf_owned_mean", "idf_owned_max", "idf_missing_mean",
    "recipe_size",
    "n_steps", "est_prep_time_min", "est_cook_time_min",
]

ML_CATEGORICAL = ["dish_group", "difficulty"]

ML_FEATURES = ML_NUMERIC + ML_CATEGORICAL

# Lazy-loaded data
recipes: pd.DataFrame = None
recipe_ingredients: list[list[str]] = None
vocab: list[str] = None
vocab_set: set[str] = None
ingredient_index: dict[str, int] = None
key_index: dict[str, int] = None
ingredient_keys_matrix: csr_matrix = None
recipe_ingredients_matrix: csr_matrix = None
recipe_lengths: np.ndarray = None
idf: dict[str, float] = None
ingredient_frequency: dict[str, int] = None

# Model placeholders
m3_best = None
best_model = None
best_threshold = None


def ingredient_keys(name: str) -> set[str]:
    """Soft-match keys: full name + word suffixes ('yellow onion' -> {'yellow onion', 'onion'})."""
    words = name.split()
    keys = {" ".join(words[i:]) for i in range(len(words))}
    return {key for key in keys if key not in GENERIC_WORDS}


def to_features(ingredients: list[str], mode: str, drop_staples: bool) -> list[str]:
    """Convert ingredient list to TF-IDF analyzer tokens."""
    items = [item for item in ingredients if not (drop_staples and item in STAPLES)]
    if mode == "ingredient":
        return items
    words = [f"w_{word}" for item in items for word in item.split()]
    return words if mode == "word" else items + words


def load_data() -> None:
    """Load recipes CSV and build all shared data structures. Call once at startup."""
    global recipes, recipe_ingredients, vocab, vocab_set, ingredient_index, key_index
    global ingredient_keys_matrix, recipe_ingredients_matrix, recipe_lengths, idf, ingredient_frequency

    if recipes is not None:
        return  # already loaded

    recipes = pd.read_csv(RECIPES_CSV)

    # Normalize ingredient lists
    def parse_ingredients(s):
        try:
            return json.loads(s) if isinstance(s, str) else []
        except Exception:
            return []

    recipe_ingredients = recipes["ingredients"].apply(parse_ingredients).tolist()

    # Build vocabulary
    vocab = sorted({item for items in recipe_ingredients for item in items})
    vocab_set = set(vocab)
    ingredient_index = {item: i for i, item in enumerate(vocab)}

    # Soft-match key index
    keys = sorted({key for item in vocab for key in ingredient_keys(item)})
    key_index = {key: i for i, key in enumerate(keys)}

    # ingredient -> keys matrix (for fast coverage computation)
    rows, cols = [], []
    for item, i in ingredient_index.items():
        for key in ingredient_keys(item):
            rows.append(i)
            cols.append(key_index[key])
    ingredient_keys_matrix = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(vocab), len(keys)))

    # recipe -> ingredients matrix
    rows, cols = [], []
    for r, items in enumerate(recipe_ingredients):
        for item in set(items):
            rows.append(r)
            cols.append(ingredient_index[item])
    recipe_ingredients_matrix = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(recipe_ingredients), len(vocab)))
    recipe_lengths = np.asarray(recipe_ingredients_matrix.sum(axis=1)).ravel()

    # IDF for ingredients
    ingredient_frequency = {}
    for items in recipe_ingredients:
        for item in set(items):
            ingredient_frequency[item] = ingredient_frequency.get(item, 0) + 1
    n_recipes = len(recipe_ingredients)
    idf = {item: np.log(n_recipes / count) for item, count in ingredient_frequency.items()}


def clean_ingredients_v2(raw: list[str]) -> list[str]:
    """Normalize user pantry input (same regex as notebook)."""
    import re
    cleaned = []
    for x in raw:
        x = x.lower().strip()
        x = re.sub(r"^(\d+\.?\d*\s*)?(cups?|tbsp|tsp|grams?|g|kg|oz|lb|ml|l|pinch|dash|clove|slice|piece)s?\s+", "", x)
        x = re.sub(r"[^a-z0-9\s-]", "", x)
        x = re.sub(r"\s+", " ", x).strip()
        if x and x not in STAPLES:
            cleaned.append(x)
    return cleaned