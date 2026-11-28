"""Load notebook's model bundle and repackage for Streamlit (app.py).

Training is owned by the notebook (Group3_Italian_Recipe_Recommender.ipynb,
Cell 133 exports model_bundle.joblib). This script loads it and saves a
Streamlit-ready bundle at the project root.
"""
import joblib
from pathlib import Path
import sys

import config as cfg
from config import load_data, dataset_sha256
from recommender import ItalianRecipeRecommender  # noqa: F401  needed for unpickling notebook bundle
from models import GroupTimeImputer  # noqa: F401  needed for unpickling notebook bundle (M4 pipeline step)

NB_BUNDLE_PATH = Path(__file__).parent / "output" / "uts_audited" / "model_bundle.joblib"
OUT_BUNDLE_PATH = Path(__file__).parent / "model_bundle.joblib"


def load_notebook_bundle() -> dict:
    """Load and verify the notebook's model bundle."""
    if not NB_BUNDLE_PATH.exists():
        raise FileNotFoundError(
            f"Notebook bundle not found: {NB_BUNDLE_PATH}\n"
            "Run Group3_Italian_Recipe_Recommender.ipynb with EXPORT_FILES = True first."
        )

    bundle = joblib.load(NB_BUNDLE_PATH)

    # Verify required keys from notebook
    required_keys = ["m3", "m4", "threshold", "model_name", "features", "vocab", "recipes", "dataset_sha256"]
    missing = [k for k in required_keys if k not in bundle]
    if missing:
        raise KeyError(f"Notebook bundle missing required keys: {missing}")

    if bundle.get("dataset_sha256") != dataset_sha256:
        raise ValueError(
            f"Dataset SHA256 mismatch: bundle={bundle.get('dataset_sha256')}, "
            f"config={dataset_sha256}. Re-run the notebook."
        )

    return bundle


def build_streamlit_bundle(nb_bundle: dict) -> dict:
    """Build the bundle format expected by app.py."""
    load_data()  # Build vocab, matrices, IDF, etc.

    return {
        # Models from notebook
        "m3": nb_bundle["m3"],
        "m4": nb_bundle["m4"],
        "threshold": nb_bundle["threshold"],
        "model_name": nb_bundle["model_name"],
        "features": nb_bundle["features"],

        # Data from config (built from parquet)
        "recipes": cfg.recipes,
        "recipe_ingredients": cfg.recipe_ingredients,
        "vocab": cfg.vocab,
        "idf": cfg.idf,
        "ingredient_frequency": cfg.ingredient_frequency,
        "N_CANDIDATES": cfg.N_CANDIDATES,
        "dataset_sha256": nb_bundle["dataset_sha256"],
    }


def main():
    print("Loading notebook bundle...")
    nb_bundle = load_notebook_bundle()
    print(f"  Model: {nb_bundle['model_name']}")
    print(f"  Threshold: {nb_bundle['threshold']:.4f}")
    print(f"  Recipes in notebook bundle: {len(nb_bundle['recipes'])}")

    print("Building Streamlit bundle...")
    streamlit_bundle = build_streamlit_bundle(nb_bundle)
    print(f"  Recipes in Streamlit bundle: {len(streamlit_bundle['recipes'])}")
    print(f"  Vocab size: {len(streamlit_bundle['vocab'])}")
    print(f"  Features: {len(streamlit_bundle['features'])}")

    joblib.dump(streamlit_bundle, OUT_BUNDLE_PATH, compress=3)
    print(f"\nSaved Streamlit bundle to {OUT_BUNDLE_PATH} ({OUT_BUNDLE_PATH.stat().st_size / 1e6:.1f} MB)")
    print("Ready for `streamlit run app.py`")


if __name__ == "__main__":
    main()