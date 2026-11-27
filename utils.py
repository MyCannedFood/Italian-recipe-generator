import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from typing import Callable


def ranking_metrics(top_indices: np.ndarray, targets: np.ndarray, k: int = 10) -> dict:
    hits = np.asarray(top_indices)[:, :k] == np.asarray(targets)[:, None]
    found = hits.any(axis=1)
    rank = np.where(found, hits.argmax(axis=1) + 1, np.inf)
    return {
        "HitRate@5": float(np.mean(rank <= 5)),
        "HitRate@10": float(np.mean(found)),
        "MRR@10": float(np.mean(np.where(found, 1.0 / rank, 0.0))),
        "NDCG@10": float(np.mean(np.where(found, 1.0 / np.log2(rank + 1), 0.0))),
    }


def top_k(scores: np.ndarray, k: int = 10) -> np.ndarray:
    return np.argsort(-scores, axis=1)[:, :k]


def relevance_metrics(queries: list[list[str]], top_indices: np.ndarray, k: int = 5) -> dict:
    from config import recipe_ingredients, ingredient_keys, recipe_lengths
    from recommender import get_m3_model
    
    m3 = get_m3_model()
    coverage_matrix = m3.coverage(queries)
    top = np.asarray(top_indices)[:, :k]
    cov = np.take_along_axis(coverage_matrix, top, axis=1)
    missing = np.round((1 - cov) * recipe_lengths[top])
    return {
        "Coverage@5": float(cov.mean()),
        "Missing@5": float(missing.mean()),
    }


def rerank_metrics(frame: pd.DataFrame, scores: np.ndarray) -> dict:
    frame = frame[["query_id", "label"]].assign(score=np.asarray(scores))
    frame["rank"] = frame.groupby("query_id")["score"].rank(ascending=False, method="first")
    ranks = (frame.loc[frame["label"] == 1].set_index("query_id")["rank"]
             .reindex(frame["query_id"].unique()).fillna(np.inf).to_numpy())
    found = ranks <= 10
    return {
        "HitRate@5": float(np.mean(ranks <= 5)),
        "HitRate@10": float(np.mean(found)),
        "MRR@10": float(np.mean(np.where(found, 1.0 / ranks, 0.0))),
        "NDCG@10": float(np.mean(np.where(found, 1.0 / np.log2(ranks + 1), 0.0))),
    }


def bootstrap_ci(data: np.ndarray, n_resamples: int = 2000, confidence_level: float = 0.95) -> tuple[float, float]:
    if len(data) == 0:
        return (0.0, 0.0)
    res = bootstrap((data,), np.mean, n_resamples=n_resamples, confidence_level=confidence_level, random_state=42)
    return (float(res.confidence_interval.low), float(res.confidence_interval.high))


def fmt_ci(ci: tuple[float, float]) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def verdict(p_value: float) -> str:
    if p_value < 0.01:
        return "signifikan"
    elif p_value < 0.05:
        return "signifikan (tipis)"
    else:
        return "tidak signifikan"


def make_queries(
    target_indices: np.ndarray,
    keep: float = 0.5,
    n_noise: int = 2,
    p_general: float = 0.5,
    seed: int = 42,
) -> list[list[str]]:
    from config import recipe_ingredients, GENERIC_WORDS, STAPLES, vocab
    import random
    
    rng = np.random.default_rng(seed)
    all_ingredients = list(vocab)
    general_ingredients = [x for x in all_ingredients if x in GENERIC_WORDS]
    specific_ingredients = [x for x in all_ingredients if x not in GENERIC_WORDS and x not in STAPLES]
    
    queries = []
    for idx in target_indices:
        target_items = recipe_ingredients[idx]
        n_keep = max(1, int(len(target_items) * keep))
        kept = rng.choice(target_items, size=n_keep, replace=False).tolist()
        
        noise = []
        for _ in range(n_noise):
            if rng.random() < p_general and general_ingredients:
                noise.append(rng.choice(general_ingredients))
            elif specific_ingredients:
                noise.append(rng.choice(specific_ingredients))
        
        query = kept + noise
        rng.shuffle(query)
        queries.append(query)
    
    return queries


def candidate_features(queries: list[list[str]], targets: np.ndarray, n_candidates: int = 50) -> pd.DataFrame:
    from config import (
        recipe_ingredients, vocab, vocab_set, ingredient_keys, idf,
        recipe_item_keys, recipe_key_sets, recipe_idf, recipe_tabular,
        ingredient_index, key_index, ingredient_keys_matrix, recipe_ingredients_matrix,
        N_CANDIDATES
    )
    from recommender import get_m3_best, ItalianRecipeRecommender
    from sklearn.preprocessing import MultiLabelBinarizer
    import numpy as np
    
    m3_best = get_m3_best()
    m2_default = ItalianRecipeRecommender(mode="ingredient", use_idf=True, drop_staples=False, alpha=1.0)
    m2_default.fit(recipe_ingredients)
    
    m3_scores = m3_best.score(queries)
    cos_both = (m3_best.vectorizer_.transform(queries) @ m3_best.item_matrix_.T).toarray()
    cos_ingredient = m2_default.score(queries)
    
    binarizer = MultiLabelBinarizer(classes=vocab)
    recipe_binary = binarizer.fit_transform(recipe_ingredients).astype(bool)
    query_binary = binarizer.transform([[x for x in q if x in vocab_set] for q in queries]).astype(bool)
    exact_overlap = (query_binary @ recipe_binary.T).astype(float)
    
    candidates = top_k(m3_scores, n_candidates)
    rows = []
    
    for q, (query, target) in enumerate(zip(queries, targets)):
        query_keys = [ingredient_keys(item) for item in query]
        query_key_union = set().union(*query_keys)
        top_score = m3_scores[q, candidates[q, 0]]
        
        for rank, r in enumerate(candidates[q], start=1):
            owned = np.array([bool(keys & query_key_union) for keys in recipe_item_keys[r]])
            size = len(owned)
            n_owned = owned.sum()
            
            idf_vals = recipe_idf[r]
            idf_owned_mean = idf_vals[owned].mean() if n_owned > 0 else 0.0
            idf_owned_max = idf_vals[owned].max() if n_owned > 0 else 0.0
            idf_missing_mean = idf_vals[~owned].mean() if (size - n_owned) > 0 else 0.0
            
            query_used = np.mean([bool(qk & recipe_key_sets[r]) for qk in query_keys]) if query_keys else 0.0
            jaccard = exact_overlap[q, r] / (len(query) + size - exact_overlap[q, r]) if (len(query) + size - exact_overlap[q, r]) > 0 else 0.0
            
            rows.append({
                "query_id": q, "target": target, "recipe": r, "label": int(r == target),
                "m3_rank": rank, "m3_score": m3_scores[q, r], "score_gap": top_score - m3_scores[q, r],
                "cos_both": cos_both[q, r], "cos_ingredient": cos_ingredient[q, r],
                "exact_overlap": exact_overlap[q, r], "jaccard": jaccard,
                "coverage": n_owned / size if size > 0 else 0.0,
                "n_missing": size - n_owned,
                "query_used": query_used,
                "idf_owned_mean": idf_owned_mean,
                "idf_owned_max": idf_owned_max,
                "idf_missing_mean": idf_missing_mean,
                "recipe_size": size,
            })
    
    frame = pd.DataFrame(rows)
    return frame.join(recipe_tabular.reset_index(drop=True), on="recipe")


def f1_optimal_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[np.argmax(f1)])


def hits_by_query(frame: pd.DataFrame, scores: np.ndarray, k: int = 5) -> np.ndarray:
    tmp = frame[["query_id", "label"]].assign(score=np.asarray(scores))
    tmp["rank"] = tmp.groupby("query_id")["score"].rank(ascending=False, method="first")
    ranks = (tmp.loc[tmp["label"] == 1].set_index("query_id")["rank"]
             .reindex(sorted(frame["query_id"].unique())).fillna(np.inf).to_numpy())
    return ranks <= k


def paired_comparison(hits_a: np.ndarray, hits_b: np.ndarray) -> dict:
    from scipy.stats import binomtest
    only_a = int((hits_a & ~hits_b).sum())
    only_b = int((~hits_a & hits_b).sum())
    p_value = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    delta = hits_a.mean() - hits_b.mean()
    ci = bootstrap_ci(hits_a.astype(int) - hits_b.astype(int))
    return {
        "Δ HitRate@5": float(delta),
        "CI 95% bootstrap": fmt_ci(ci),
        "Hanya A benar": only_a,
        "Hanya B benar": only_b,
        "McNemar p": float(p_value),
        "Keputusan": verdict(p_value),
    }