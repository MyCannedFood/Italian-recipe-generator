import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from config import (
    N_CANDIDATES, ML_FEATURES, ML_NUMERIC, ML_CATEGORICAL,
    load_data, clean_ingredients_v2, ingredient_keys,
    recipe_ingredients, vocab, vocab_set, ingredient_index, key_index,
    ingredient_keys_matrix, recipe_ingredients_matrix, recipe_lengths,
    idf, ingredient_frequency, recipes,
)
from recommender import get_m3_best, ItalianRecipeRecommender

# Ensure data is loaded
load_data()


# Pre-computed recipe metadata for features
recipe_item_keys = [[ingredient_keys(item) for item in items] for items in recipe_ingredients]
recipe_key_sets = [set().union(*keys) for keys in recipe_item_keys]
recipe_idf = [np.array([idf[item] for item in items]) for items in recipe_ingredients]
recipe_tabular = recipes[["dish_group", "difficulty", "n_steps", "est_prep_time_min", "est_cook_time_min"]].reset_index(drop=True)

# M3 model (cached)
_m3_model: ItalianRecipeRecommender = None


def get_m3_model() -> ItalianRecipeRecommender:
    global _m3_model
    if _m3_model is None:
        _m3_model = get_m3_best()
    return _m3_model


def compute_m4_features(pantry: list[str]) -> pd.DataFrame:
    load_data()
    query = clean_ingredients_v2(pantry)
    m3 = get_m3_model()

    # Get M3 scores and top-N candidates
    m3_scores = m3.score([query])[0]
    candidates = np.argsort(-m3_scores)[:N_CANDIDATES]

    # Query keys for coverage computation
    query_keys = {k for item in query for k in ingredient_keys(item)}
    query_key_list = [ingredient_keys(item) for item in query]
    query_key_union = set().union(*query_key_list) if query_key_list else set()

    # Cosine similarities
    cos_both = (m3.vectorizer_.transform([query]) @ m3.item_matrix_.T).toarray()[0]
    
    # M2 (ingredient-only) cosine
    from recommender import ItalianRecipeRecommender
    m2 = ItalianRecipeRecommender(mode="ingredient", use_idf=True, drop_staples=False, alpha=1.0)
    m2.fit(recipe_ingredients)
    cos_ingredient = m2.score([query])[0]

    # Exact overlap (binary)
    from sklearn.preprocessing import MultiLabelBinarizer
    binarizer = MultiLabelBinarizer(classes=vocab)
    recipe_binary = binarizer.fit_transform(recipe_ingredients).astype(bool)
    query_binary = binarizer.transform([query]).astype(bool)
    exact_overlap = (query_binary @ recipe_binary.T)[0]

    top_score = m3_scores[candidates[0]]
    rows = []

    for rank, r in enumerate(candidates, start=1):
        owned = np.array([bool(keys & query_key_union) for keys in recipe_item_keys[r]])
        size = len(owned)
        n_owned = owned.sum()
        n_missing = size - n_owned

        # IDF stats
        idf_vals = recipe_idf[r]
        idf_owned_mean = idf_vals[owned].mean() if n_owned > 0 else 0.0
        idf_owned_max = idf_vals[owned].max() if n_owned > 0 else 0.0
        idf_missing_mean = idf_vals[~owned].mean() if n_missing > 0 else 0.0

        # Query used: proportion of query ingredients that appear in recipe
        query_used = np.mean([bool(qk & recipe_key_sets[r]) for qk in query_key_list]) if query_key_list else 0.0

        # Jaccard
        jaccard = exact_overlap[r] / (len(query) + size - exact_overlap[r]) if (len(query) + size - exact_overlap[r]) > 0 else 0.0

        rows.append({
            "query_id": 0,  # single query
            "target": -1,   # unknown for user query
            "recipe": r,
            "label": 0,     # not used for inference
            "m3_rank": rank,
            "m3_score": float(m3_scores[r]),
            "score_gap": float(top_score - m3_scores[r]),
            "cos_both": float(cos_both[r]),
            "cos_ingredient": float(cos_ingredient[r]),
            "exact_overlap": float(exact_overlap[r]),
            "jaccard": float(jaccard),
            "coverage": float(n_owned / size) if size > 0 else 0.0,
            "n_missing": int(n_missing),
            "query_used": float(query_used),
            "idf_owned_mean": float(idf_owned_mean),
            "idf_owned_max": float(idf_owned_max),
            "idf_missing_mean": float(idf_missing_mean),
            "recipe_size": int(size),
        })

    frame = pd.DataFrame(rows)
    # Join tabular recipe features
    frame = frame.join(recipe_tabular, on="recipe")
    return frame


def recommend_from_pantry(pantry: list[str], k: int = 5, use_m4: bool = True, model=None, threshold: float = None) -> pd.DataFrame:
    load_data()
    query = clean_ingredients_v2(pantry)
    m3 = get_m3_model()

    # Get M3 top candidates (more than k for M4 re-ranking)
    m3_candidates = m3.recommend(query, k=max(k * 3, N_CANDIDATES))
    candidate_indices = [idx for idx, _, _ in m3_candidates]

    if use_m4 and model is not None:
        # Compute M4 features for candidates
        feat_df = compute_m4_features(pantry)
        feat_df = feat_df[feat_df["recipe"].isin(candidate_indices)].copy()
        
        # Predict probabilities
        proba = model.predict_proba(feat_df[ML_FEATURES])[:, 1]
        feat_df["m4_proba"] = proba
        
        # Rank by M4 probability
        feat_df["m4_rank"] = feat_df["m4_proba"].rank(ascending=False, method="first").astype(int)
        feat_df = feat_df.sort_values("m4_proba", ascending=False).head(k)
        
        score_col = "m4_proba"
    else:
        # M3 only
        feat_df = pd.DataFrame(m3_candidates[:k], columns=["recipe", "score", "coverage"])
        score_col = "score"

    # Build output
    results = []
    for _, row in feat_df.iterrows():
        idx = int(row["recipe"])
        have = {k for item in query for k in ingredient_keys(item)}
        missing = [item for item in recipe_ingredients[idx] if not (ingredient_keys(item) & have)]
        results.append({
            "Resep": recipes.loc[idx, "recipe_title"],
            "Skor": round(float(row[score_col]), 3),
            "Bahan dimiliki": f"{row['coverage']:.0%}",
            "Perlu dibeli": ", ".join(missing[:6]),
        })
    
    return pd.DataFrame(results)