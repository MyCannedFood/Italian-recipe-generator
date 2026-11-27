from typing import Optional
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer

import config as cfg
from config import (
    GENERIC_WORDS, STAPLES, RANDOM_STATE, N_JOBS,
    ingredient_keys, to_features, load_data,
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
        # Sparse matrices for coverage computation (notebook style)
        self.ingredient_index_: dict[str, int] = None
        self.key_index_: dict[str, int] = None
        self.ingredient_keys_: csr_matrix = None
        self.recipe_ingredients_: csr_matrix = None
        self.recipe_lengths_: np.ndarray = None

    def _analyzer(self, ingredients: list[str]) -> list[str]:
        return to_features(ingredients, self.mode, self.drop_staples)

    def fit(self, recipe_ingredients_list: list[list[str]]) -> "ItalianRecipeRecommender":
        self.recipes_ = [list(items) for items in recipe_ingredients_list]

        # TF-IDF matrix
        self.vectorizer_ = TfidfVectorizer(
            analyzer=self._analyzer, use_idf=self.use_idf, sublinear_tf=True, norm="l2"
        )
        self.item_matrix_ = self.vectorizer_.fit_transform(self.recipes_)

        # Build sparse matrices for coverage (matching notebook)
        vocab = sorted({item for items in self.recipes_ for item in items})
        self.ingredient_index_ = {item: i for i, item in enumerate(vocab)}
        keys = sorted({key for item in vocab for key in ingredient_keys(item)})
        self.key_index_ = {key: i for i, key in enumerate(keys)}

        # ingredient -> keys matrix
        rows, cols = [], []
        for item, i in self.ingredient_index_.items():
            for key in ingredient_keys(item):
                rows.append(i)
                cols.append(self.key_index_[key])
        self.ingredient_keys_ = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(vocab), len(keys)))

        # recipe -> ingredients matrix
        rows, cols = [], []
        for r, items in enumerate(self.recipes_):
            for item in set(items):
                rows.append(r)
                cols.append(self.ingredient_index_[item])
        self.recipe_ingredients_ = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(self.recipes_), len(vocab)))
        self.recipe_lengths_ = np.asarray(self.recipe_ingredients_.sum(axis=1)).ravel()

        return self

    def coverage(self, queries: list[list[str]]) -> np.ndarray:
        rows, cols = [], []
        for q, items in enumerate(queries):
            for key in {k for item in items for k in ingredient_keys(item)}:
                if key in self.key_index_:
                    rows.append(q)
                    cols.append(self.key_index_[key])
        query_keys = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(queries), len(self.key_index_)))
        available = (query_keys @ self.ingredient_keys_.T) > 0
        covered = (available.astype(float) @ self.recipe_ingredients_.T).toarray()
        return covered / self.recipe_lengths_

    def score(self, queries: list[list[str]]) -> np.ndarray:
        similarity = (self.vectorizer_.transform(queries) @ self.item_matrix_.T).toarray()
        if self.alpha >= 1:
            return similarity
        return self.alpha * similarity + (1 - self.alpha) * self.coverage(queries)

    def recommend(self, ingredients: list[str], k: int = 10) -> list[tuple[int, float, float]]:
        scores = self.score([ingredients])[0]
        top = np.argsort(-scores)[:k]
        coverage = self.coverage([ingredients])[0]
        return [(int(i), float(scores[i]), float(coverage[i])) for i in top]


# M0: Exact ingredient overlap (binary)
def overlap_top_k(queries: list[list[str]], k: int = 10) -> np.ndarray:
    load_data()
    binarizer = MultiLabelBinarizer(classes=cfg.vocab)
    recipe_binary = binarizer.fit_transform(cfg.recipe_ingredients).astype(bool)

    def binary_queries(qs):
        return binarizer.transform([[x for x in q if x in cfg.vocab_set] for q in qs]).astype(bool)

    shared = binary_queries(queries).astype(float) @ recipe_binary.T.astype(float)
    # Tie-break: shorter recipes first
    return np.argsort(-(shared - 1e-3 * recipe_binary.sum(axis=1)), axis=1)[:, :k]


# M1: Jaccard similarity (binary)
knn_jaccard = None

def knn_jaccard_top_k(queries: list[list[str]], k: int = 10) -> np.ndarray:
    global knn_jaccard
    load_data()
    if knn_jaccard is None:
        binarizer = MultiLabelBinarizer(classes=cfg.vocab)
        recipe_binary = binarizer.fit_transform(cfg.recipe_ingredients).astype(bool)
        knn_jaccard = NearestNeighbors(metric="jaccard", algorithm="brute").fit(recipe_binary)

    def binary_queries(qs):
        return binarizer.transform([[x for x in q if x in cfg.vocab_set] for q in qs]).astype(bool)

    return knn_jaccard.kneighbors(binary_queries(queries), n_neighbors=k, return_distance=False)


# Default optimized M2/M3 model (parameters from notebook grid search)
def get_m3_best() -> ItalianRecipeRecommender:
    # Best params from notebook: mode="both", use_idf=True, drop_staples=False, alpha=0.6
    load_data()  # Ensure data is loaded
    model = ItalianRecipeRecommender(mode="both", use_idf=True, drop_staples=False, alpha=0.6)
    model.fit(cfg.recipe_ingredients)
    return model


# Convenience: load data on import
load_data()