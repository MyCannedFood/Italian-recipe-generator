import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.base import clone

from config import (
    RECIPES_CSV, RANDOM_STATE, N_CANDIDATES, ML_FEATURES,
    load_data, recipes, recipe_ingredients, vocab, idf,
    ingredient_frequency, dataset_sha256,
)
from recommender import get_m3_best
from models import get_tuned_models, XGBQueryRanker
from utils import candidate_features
from features import compute_m4_features
from utils import make_queries, f1_optimal_threshold, ranking_metrics


def main():
    print("Loading data...")
    load_data()
    
    print(f"Recipes: {len(recipes)}")
    print(f"Vocab size: {len(vocab)}")
    
    # Split data (same as notebook)
    all_indices = np.arange(len(recipes))
    train_idx, test_idx = train_test_split(all_indices, test_size=0.2, random_state=RANDOM_STATE, shuffle=True)
    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
    
    # Train M3
    print("\nTraining M3 (TF-IDF + Cosine + Coverage)...")
    m3_best = get_m3_best()
    print(f"M3 fitted. Matrix shape: {m3_best.item_matrix_.shape}")
    
    # Build M4 training data
    print("\nBuilding M4 training data...")
    train_queries = make_queries(train_idx, keep=0.5, n_noise=2, p_general=0.5, seed=7)
    train_df = candidate_features(train_queries, train_idx, n_candidates=N_CANDIDATES)
    X_train = train_df[ML_FEATURES]
    y_train = train_df["label"].to_numpy()
    print(f"Training pairs: {len(train_df)}, Positive rate: {y_train.mean():.3%}")
    
    # Train M4 models (use tuned Logistic Regression as default - best in notebook)
    print("\nTraining M4 models...")
    tuned_models = get_tuned_models()
    
    fitted_models = {}
    thresholds = {}
    
    for name, model in tuned_models.items():
        print(f"  Fitting {name}...")
        fitted = clone(model).fit(X_train, y_train)
        fitted_models[name] = fitted
        
        # Compute OOF threshold (approximate - using train predictions)
        proba = fitted.predict_proba(X_train)[:, 1]
        threshold = f1_optimal_threshold(y_train, proba)
        thresholds[name] = threshold
        print(f"    F1-optimal threshold: {threshold:.4f}")
    
    # Select best model (Logistic Regression was best in notebook)
    best_model_name = "Logistic Regression"
    best_model = fitted_models[best_model_name]
    best_threshold = thresholds[best_model_name]
    
    print(f"\nBest model: {best_model_name} (threshold={best_threshold:.4f})")
    
    # Quick validation on test set
    print("\nValidating on test set...")
    test_queries = make_queries(test_idx, keep=0.5, n_noise=2, p_general=0.5, seed=7)
    test_df = candidate_features(test_queries, test_idx, n_candidates=N_CANDIDATES)
    X_test = test_df[ML_FEATURES]
    y_test = test_df["label"].to_numpy()
    
    test_proba = best_model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= best_threshold).astype(int)
    
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
    print(f"  ROC-AUC: {roc_auc_score(y_test, test_proba):.4f}")
    print(f"  PR-AUC: {average_precision_score(y_test, test_proba):.4f}")
    print(f"  Accuracy: {accuracy_score(y_test, test_pred):.4f}")
    print(f"  Balanced Acc: {balanced_accuracy_score(y_test, test_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"  Recall: {recall_score(y_test, test_pred):.4f}")
    print(f"  F1: {f1_score(y_test, test_pred):.4f}")
    
    # Ranking metrics
    m3_ranking = ranking_metrics(top_k(-test_df.groupby("query_id")["m3_rank"].apply(list).to_numpy()), test_idx)
    m4_ranking = ranking_metrics(test_proba, test_idx)  # This needs fixing
    
    # Save bundle
    print("\nSaving model bundle...")
    bundle = {
        "m3": m3_best,
        "m4": best_model,
        "threshold": best_threshold,
        "model_name": best_model_name,
        "all_models": fitted_models,
        "all_thresholds": thresholds,
        "features": ML_FEATURES,
        "numeric_features": [c for c in ML_FEATURES if c in ML_FEATURES[:21]],  # ML_NUMERIC
        "categorical_features": [c for c in ML_FEATURES if c in ML_FEATURES[21:]],  # ML_CATEGORICAL
        "recipes": recipes,
        "recipe_ingredients": recipe_ingredients,
        "vocab": vocab,
        "idf": idf,
        "ingredient_frequency": ingredient_frequency,
        "N_CANDIDATES": N_CANDIDATES,
        "dataset_sha256": dataset_sha256,
    }
    
    output_path = Path(__file__).parent / "model_bundle.joblib"
    joblib.dump(bundle, output_path, compress=3)
    print(f"Saved to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    
    # Also save a lightweight recipes CSV for reference
    export_cols = [
        "recipe_title", "category", "subcategory", "description", "ingredients", "directions",
        "dish_group", "est_prep_time_min", "est_cook_time_min", "difficulty", "ingredient_count"
    ]
    export_df = recipes[export_cols].copy()
    export_df["ingredient_list_normalized"] = recipes["ingredients"]
    export_df.to_csv("recipes_italia_final.csv", index=False, encoding="utf-8-sig")
    print("Saved recipes_italia_final.csv")
    
    print("\nDone! Bundle ready for Streamlit.")


if __name__ == "__main__":
    main()