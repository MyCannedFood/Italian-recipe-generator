from typing import Optional
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer

from config import (
    GENERIC_WORDS, STAPLES, RANDOM_STATE, N_JOBS,
    ingredient_keys, to_features, load_data,
    recipe_ingredients, vocab, vocab_set, ingredient_index, key_index,
    ingredient_keys_matrix, recipe_ingredients_matrix, recipe_lengths,
    idf, ingredient_frequency,
)


class ItalianRecipeRecommender:
    def __init__(self, mode: str = "both", use_idf: bool = True, drop_staples: bool = False, alpha: float = 1.0):
        self.mode = mode
        self.use_idf = use_idf
        self.drop_staples = drop_staples
        self.alpha = alpha
        self.recipes_: list[list[str]] = None
        self.vectorizer_: TfidfVectorizer = None
        self.item_matrix_: csr_matrix = None
        self.binarizer_ = MultiLabelBinarizer(classes=vocab)
        self.recipe_binary_: csr_matrix = None
        self.knn_jaccard_: NearestNeighbors = None

    def _analyzer(self, ingredients: list[str]) -> list[str]:
        return to_features(ingredients, self.mode, self.drop_staples)

    def fit(self, recipe_ingredients_list: list[list[str]]) -> "ItalianRecipeRecommender":
        self.recipes_ = [list(items) for items in recipe_ingredients_list]

        # TF-IDF matrix
        self.vectorizer_ = TfidfVectorizer(
            analyzer=self._analyzer, use_idf=self.use_idf, sublinear_tf=True, norm="l2"
        )
        self.item_matrix_ = self.vectorizer_.fit_transform(self.recipes_)

        # Binary matrix for M0/M1 (exact overlap / Jaccard)
        self.recipe_binary_ = self.binarizer_.fit_transform(self.recipes_).astype(bool)
        self.knn_jaccard_ = NearestNeighbors(metric="jaccard", algorithm="brute").fit(self.recipe_binary_)

        return self

    def score(self, queries: list[list[str]]) -> np.ndarray:
        query_vecs = self.vectorizer_.transform(queries)
        return (query_vecs @ self.item_matrix_.T).toarray()

    def recommend(self, query: list[str], k: int = 10) -> list[tuple[int, float, float]]:
        scores = self.score([query])[0]
        top_idx = np.argsort(-scores)[:k]

        # Coverage: proportion of recipe ingredients user has (soft match)
        query_keys = {k for item in query for k in ingredient_keys(item)}
        results = []
        for idx in top_idx:
            recipe_items = self.recipes_[idx]
            owned = sum(1 for item in recipe_items if ingredient_keys(item) & query_keys)
            coverage = owned / len(recipe_items) if recipe_items else 0.0
            # Combined score: alpha * cosine + (1 - alpha) * coverage
            combined = self.alpha * scores[idx] + (1 - self.alpha) * coverage
            results.append((idx, float(combined), float(coverage)))
        return results


# M0: Exact ingredient overlap (binary)
def overlap_top_k(queries: list[list[str]], k: int = 10) -> np.ndarray:
    load_data()
    binarizer = MultiLabelBinarizer(classes=vocab)
    recipe_binary = binarizer.fit_transform(recipe_ingredients).astype(bool)

    def binary_queries(qs):
        return binarizer.transform([[x for x in q if x in vocab_set] for q in qs]).astype(bool)

    shared = binary_queries(queries).astype(float) @ recipe_binary.T.astype(float)
    # Tie-break: shorter recipes first
    return np.argsort(-(shared - 1e-3 * recipe_binary.sum(axis=1)), axis=1)[:, :k]


# M1: Jaccard similarity (binary)
knn_jaccard = None

def knn_jaccard_top_k(queries: list[list[str]], k: int = 10) -> np.ndarray:
    global knn_jaccard
    load_data()
    if knn_jaccard is None:
        binarizer = MultiLabelBinarizer(classes=vocab)
        recipe_binary = binarizer.fit_transform(recipe_ingredients).astype(bool)
        knn_jaccard = NearestNeighbors(metric="jaccard", algorithm="brute").fit(recipe_binary)

    def binary_queries(qs):
        return binarizer.transform([[x for x in q if x in vocab_set] for q in qs]).astype(bool)

    return knn_jaccard.kneighbors(binary_queries(queries), n_neighbors=k, return_distance=False)


# Default optimized M2/M3 model (parameters from notebook grid search)
def get_m3_best() -> ItalianRecipeRecommender:
    # Best params from notebook: mode="both", use_idf=True, drop_staples=False, alpha=0.6
    model = ItalianRecipeRecommender(mode="both", use_idf=True, drop_staples=False, alpha=0.6)
    model.fit(recipe_ingredients)
    return model


# Convenience: load data on import
load_data()