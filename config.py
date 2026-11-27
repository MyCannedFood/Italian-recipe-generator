from pathlib import Path
import json
import re
import unicodedata
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

# --- clean_ingredients_v2 constants (from notebook Cell 6) ---
FRACTIONS = "¼½¾⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞"
UNITS = (
    "cups?|tablespoons?|tbsps?|teaspoons?|tsps?|ounces?|oz|pounds?|lbs?|"
    "grams?|g|kilograms?|kg|milliliters?|ml|liters?|l|quarts?|qt|pints?|"
    "pinch(?:es)?|dash(?:es)?|handfuls?|sprigs?|sticks?|stalks?|slices?|pieces?|"
    "cans?|packages?|packets?|envelopes?|jars?|bottles?|containers?|cartons?|"
    "boxes|box|bags?|bunch(?:es)?|heads?|inch(?:es)?|fluid|balls?|links?|loaf|loaves"
)
PREPARATION = (
    "chopped|diced|minced|sliced|crushed|grated|shredded|freshly|finely|roughly|"
    "coarsely|thinly|lightly|peeled|seeded|deseeded|drained|rinsed|divided|optional|"
    "beaten|softened|melted|cubed|halved|quartered|julienned|torn|trimmed|packed|"
    "zested|juiced|cooked|uncooked|thawed|frozen|toasted|boneless|skinless|"
    "fresh|large|small|medium|canned|whole|ripe|hulled|extra lean|lean|bite sized|"
    "to taste|at room temperature|room temperature|for dredging|cut into"
)
# "ground" dibuang untuk bumbu, tapi dipertahankan untuk daging (ground beef ≠ beef)
GROUND_NOT_MEAT = r"\bground\b(?!\s+(?:beef|pork|turkey|chicken|lamb|veal|sausage))"
PASTA_NAMES = {
    "spaghetti", "penne", "rigatoni", "fettuccine", "fettuccini", "linguine",
    "macaroni", "manicotti", "fusilli", "farfalle", "tagliatelle", "orzo",
    "lasagna noodle", "lasagne noodle", "lasagna", "ziti", "rotini", "orecchiette",
}
ALIASES = {
    "extra virgin olive oil": "olive oil", "virgin olive oil": "olive oil",
    "kosher salt": "salt", "sea salt": "salt", "table salt": "salt",
    "parmesan cheese": "parmesan", "parmigiano reggiano": "parmesan",
    "parmigiano reggiano cheese": "parmesan",
    "mozzarella cheese": "mozzarella", "ricotta cheese": "ricotta",
    "all purpose flour": "flour", "pepper": "black pepper",
    "unsalted butter": "butter", "salted butter": "butter",
    "dried basil leaf": "dried basil", "scallion": "green onion", "spring onion": "green onion",
}
# Salah ketik yang ditemukan di data mentah (huruf pertama hilang saat scraping)
DATA_TYPOS = {"armesan": "parmesan", "oma": "roma", "enoa": "genoa"}
NO_SINGULAR = {"asparagus", "couscous", "molasses", "hummus", "swiss", "gnocchi", "lees"}

SALT_PEPPER = re.compile(r"(?:salt and (?:black )?pepper|(?:black )?pepper and salt)")

RANDOM_STATE = 42
N_JOBS = 2
N_CANDIDATES = 50

# Dataset integrity check
dataset_sha256 = "cba06a7cfbb1fdf7d1e3915191fc99e42d6cbf21ed6e486858c04b5971b674d8"

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

    # --- Derived columns (matching notebook preprocessing) ---
    # dish_group from subcategory/category
    if "dish_group" not in recipes.columns:
        # Map subcategory to dish group
        subcat_to_group = {
            "Antipasti": "Antipasti", "Antipasto": "Antipasti", "Bruschetta": "Antipasti",
            "Biscotti": "Dessert", "Calzones": "Pizza", "Chicken Parmesan": "Chicken",
            "Eggplant Parmesan": "Vegetarian", "Fettuccini": "Pasta", "Frittatas": "Eggs",
            "Gnocchi": "Pasta", "Homemade Pasta": "Pasta", "Lasagna": "Pasta",
            "Meat Lasagna": "Pasta", "Chicken Lasagna": "Pasta", "Spinach Lasagna": "Pasta",
            "Vegetarian Lasagna": "Pasta", "Zucchini Lasagna": "Pasta", "Eggplant Lasagna": "Pasta",
            "Linguine": "Pasta", "Manicotti": "Pasta", "Italian Meatballs": "Meat",
            "Minestrone Soups": "Soup", "Pasta Carbonara": "Pasta", "Pasta Primavera": "Pasta",
            "Pestos": "Sauce", "Pizza": "Pizza", "Pepperoni Pizza": "Pizza", "Veggie Pizza": "Pizza",
        }
        def map_dish_group(row):
            subcat = str(row.get("subcategory", "")).strip()
            if subcat in subcat_to_group:
                return subcat_to_group[subcat]
            cat = str(row.get("category", "")).strip()
            if cat == "Italian":
                return "Other"
            return "Other"
        recipes["dish_group"] = recipes.apply(map_dish_group, axis=1)

    # n_steps from num_steps
    if "n_steps" not in recipes.columns and "num_steps" in recipes.columns:
        recipes["n_steps"] = pd.to_numeric(recipes["num_steps"], errors="coerce").fillna(0).astype(int)

    # Ensure est_prep_time_min and est_cook_time_min are numeric
    for col in ["est_prep_time_min", "est_cook_time_min"]:
        if col in recipes.columns:
            recipes[col] = pd.to_numeric(recipes[col], errors="coerce")

    # Ensure difficulty is string
    if "difficulty" in recipes.columns:
        recipes["difficulty"] = recipes["difficulty"].fillna("Unknown").astype(str).str.strip().str.title()


def singularize(name: str) -> str:
    """Singularisasi sederhana pada kata terakhir (tomatoes -> tomato, leaves -> leaf)."""
    words = name.split()
    last = words[-1]
    if last in NO_SINGULAR or len(last) <= 3:
        return name
    if re.search(r"(?:tomato|potato)es$", last):
        last = last[:-2]
    elif re.search(r"(?:ch|sh|x)es$", last):
        last = last[:-2]
    elif last.endswith("ies"):
        last = last[:-3] + "y"
    elif last.endswith("leaves"):
        last = last[:-6] + "leaf"
    elif re.search(r"[^su]s$", last):
        last = last[:-1]
    return " ".join(words[:-1] + [last])


def normalize_ingredient_v2(value: str) -> str:
    name = re.sub(r"[\w'’]+\s*[®™]", " ", str(value))                  # merek dagang: Hellmann's®
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode().lower()          # jalapeño -> jalapeno
    name = re.sub(r"\([^)]*\)", " ", name)                           # (8 ounce)
    name = re.split(r",|\s-\s|\bwith\b", name)[0]                      # buang catatan setelah koma / ' - ' / with
    name = re.split(r"\b(?:such as|plus more|for)\b", name)[0]          # for garnish/serving/frying
    name = re.sub(rf"[{FRACTIONS}]", " ", name)
    name = re.sub(r"\d+(?:[./]\d+)?\s*(?:to|or|-)\s*(?=\d)", " ", name)   # rentang: 1 to 2, 1 or 2, 2-3
    name = re.sub(r"\d+(?:[./-]\d+)?(?:\s*-\s*inch(?:es)?)?", " ", name)  # 1, 1/2, 1.5, 1-inch
    name = re.sub(r"\bcloves?\s+(?:of\s+)?(?=garlic\b)", " ", name)  # cloves garlic -> garlic
    name = re.sub(r"\b(garlic)\s+cloves?\b", r"\1", name)            # garlic cloves -> garlic
    name = re.sub(rf"\b(?:{UNITS})\b\.?", " ", name)
    name = name.replace("-", " ")
    name = re.sub(r"\b(?:warm|hot|cold|lukewarm|boiling|ice)\s+(?=water\b)", " ", name)
    name = re.sub(GROUND_NOT_MEAT, " ", name)
    name = re.sub(rf"\b(?:{PREPARATION})\b", " ", name)
    name = re.sub(r"[^a-z\s]", " ", name)                            # tanda baca & hyphen -> spasi
    name = next((part for part in re.split(r"\bor\b", name) if part.strip()), "")  # butter or margarine -> butter
    name = re.sub(r"^(?:\s*(?:of|and|a|an)\b)+|\b(?:of|and)\s*$", " ", name)
    name = re.sub(r"\b(?!bay\b)(\w+)\s+lea(?:f|ves)$", r"\1", name)  # basil leaves -> basil
    name = re.sub(r"\b(?:%s)\b" % "|".join(DATA_TYPOS), lambda m: DATA_TYPOS[m.group()], name)
    name = re.sub(r"\s+", " ", name).strip()
    if not name:
        return ""
    name = singularize(name)
    name = ALIASES.get(name, name)
    if name in PASTA_NAMES or name.endswith(" pasta"):
        return "pasta"
    return name


def clean_ingredients_v2(raw: list[str]) -> list[str]:
    """Normalize user pantry input (full notebook implementation)."""
    cleaned = []
    for value in raw:
        name = normalize_ingredient_v2(value)
        if SALT_PEPPER.fullmatch(re.sub(r"\b(?:kosher|sea)\s+", "", name)):
            cleaned.extend(["salt", "black pepper"])
        elif name:
            cleaned.append(name)
    return list(dict.fromkeys(cleaned))