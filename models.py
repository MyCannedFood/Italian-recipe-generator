from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from config import ML_NUMERIC, ML_CATEGORICAL, N_JOBS, RANDOM_STATE


class GroupTimeImputer(BaseEstimator):
    def __init__(self, group_col: str = "dish_group", time_cols: tuple = ("est_prep_time_min", "est_cook_time_min")):
        self.group_col = group_col
        self.time_cols = time_cols
        self.group_medians_ = None
        self.global_medians_ = None

    def fit(self, X: pd.DataFrame, y=None) -> "GroupTimeImputer":
        df = X.copy()
        self.group_medians_ = df.groupby(self.group_col)[list(self.time_cols)].median()
        self.global_medians_ = df[list(self.time_cols)].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col in self.time_cols:
            group_fill = df[self.group_col].map(self.group_medians_[col])
            df[col] = df[col].fillna(group_fill).fillna(self.global_medians_[col])
        return df

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X).transform(X)


def make_preprocessor(scale: bool = False) -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()) if scale else ("passthrough", "passthrough"),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, ML_NUMERIC),
        ("cat", categorical_pipe, ML_CATEGORICAL),
    ])


def make_pipeline(clf, scale: bool = False) -> Pipeline:
    return Pipeline([
        ("time_impute", GroupTimeImputer()),
        ("prep", make_preprocessor(scale=scale)),
        ("clf", clf),
    ])


# Default M4 models (matching notebook ML_MODELS)
def get_default_models() -> dict:
    return {
        "Logistic Regression": make_pipeline(
            LogisticRegression(
                C=0.1, penalty="l2", solver="lbfgs", max_iter=3000,
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1
            ),
            scale=True,
        ),
        "Random Forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=300, max_depth=15, min_samples_leaf=5,
                max_features="sqrt", class_weight="balanced_subsample",
                n_jobs=N_JOBS, random_state=RANDOM_STATE
            ),
            scale=False,
        ),
        "HistGradientBoosting": make_pipeline(
            HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=200, max_leaf_nodes=31,
                l2_regularization=1.0, class_weight="balanced",
                random_state=RANDOM_STATE
            ),
            scale=False,
        ),
        "XGBoost": make_pipeline(
            XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                reg_lambda=1.0, objective="binary:logistic", tree_method="hist",
                n_jobs=N_JOBS, random_state=RANDOM_STATE, eval_metric="aucpr",
                scale_pos_weight=1.0  # will be overridden by class_weight in pipeline
            ),
            scale=False,
        ),
    }


class XGBQueryRanker(BaseEstimator):
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 5,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        objective: str = "rank:pairwise",
        n_jobs: int = N_JOBS,
        random_state: int = RANDOM_STATE,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.objective = objective
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "XGBQueryRanker":
        from xgboost import XGBRanker
        qid = X["query_id"].to_numpy()
        order = np.argsort(qid, kind="stable")
        X_sorted = X.iloc[order]
        y_sorted = np.asarray(y)[order]
        qid_sorted = qid[order]

        self.pre_ = Pipeline([
            ("time_impute", GroupTimeImputer()),
            ("prep", make_preprocessor(scale=False)),
        ])
        features = self.pre_.fit_transform(X_sorted[ML_NUMERIC + ML_CATEGORICAL])

        self.model_ = XGBRanker(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda,
            gamma=self.gamma,
            objective=self.objective,
            tree_method="hist",
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self.model_.fit(features, y_sorted, qid=qid_sorted)
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        features = self.pre_.transform(X[ML_NUMERIC + ML_CATEGORICAL])
        return np.asarray(self.model_.predict(features), dtype=float)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Sigmoid mapping for compatibility with classifier API
        scores = self.decision_function(X)
        p = 1.0 / (1.0 + np.exp(-np.clip(scores, -30, 30)))
        return np.column_stack([1 - p, p])


# Tuned model parameters (from notebook Optuna results)
TUNED_PARAMS = {
    "Logistic Regression": {
        "C": 0.029, "penalty": "elasticnet", "l1_ratio": 0.58, "solver": "saga",
        "max_iter": 3000, "class_weight": "balanced", "random_state": RANDOM_STATE, "n_jobs": 1
    },
    "Random Forest": {
        "n_estimators": 250, "max_depth": 20, "min_samples_leaf": 15,
        "max_features": "log2", "class_weight": "balanced_subsample",
        "n_jobs": N_JOBS, "random_state": RANDOM_STATE
    },
    "HistGradientBoosting": {
        "learning_rate": 0.0193, "max_iter": 139, "max_leaf_nodes": 15,
        "l2_regularization": 1.93, "class_weight": "balanced", "random_state": RANDOM_STATE
    },
    "XGBoost": {
        "n_estimators": 300, "max_depth": 4, "learning_rate": 0.014,
        "subsample": 0.8, "colsample_bytree": 0.5, "min_child_weight": 16,
        "reg_lambda": 5.0, "objective": "binary:logistic", "tree_method": "hist",
        "n_jobs": N_JOBS, "random_state": RANDOM_STATE, "eval_metric": "aucpr"
    },
}


def get_tuned_models() -> dict:
    models = {}
    for name, params in TUNED_PARAMS.items():
        if name == "Logistic Regression":
            clf = LogisticRegression(**params)
            models[name] = make_pipeline(clf, scale=True)
        elif name == "Random Forest":
            clf = RandomForestClassifier(**params)
            models[name] = make_pipeline(clf, scale=False)
        elif name == "HistGradientBoosting":
            clf = HistGradientBoostingClassifier(**params)
            models[name] = make_pipeline(clf, scale=False)
        elif name == "XGBoost":
            clf = XGBClassifier(**params)
            models[name] = make_pipeline(clf, scale=False)
    return models