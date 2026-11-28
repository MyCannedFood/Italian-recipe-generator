"""Verify and load the pre-trained model bundle from the notebook.

Training is owned by the notebook (Group3_Italian_Recipe_Recommender.ipynb,
Cell 133 exports model_bundle.joblib). This script only checks that the
bundle exists and loads it for use by app.py and other consumers.

Run the notebook first to generate model_bundle.joblib, then run this
script to verify the bundle is loadable.
"""
import joblib
from pathlib import Path

import config as cfg
from config import load_data, dataset_sha256


BUNDLE_PATH = Path(__file__).parent / "output" / "uts_audited" / "model_bundle.joblib"


def verify_bundle() -> dict:
    """Load and verify the model bundle. Returns the bundle dict."""
    load_data()

    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(
            f"Model bundle not found: {BUNDLE_PATH}\n"
            "Run Group3_Italian_Recipe_Recommender.ipynb (set EXPORT_FILES = True)\n"
            "to generate model_bundle.joblib."
        )

    bundle = joblib.load(BUNDLE_PATH)

    # Verify required keys
    required_keys = [
        "m3", "m4", "threshold", "model_name", "features",
        "vocab", "recipes", "dataset_sha256",
    ]
    missing = [k for k in required_keys if k not in bundle]
    if missing:
        raise KeyError(f"Bundle missing required keys: {missing}")

    # Verify dataset integrity
    if bundle.get("dataset_sha256") != dataset_sha256:
        raise ValueError(
            f"Dataset SHA256 mismatch: bundle={bundle.get('dataset_sha256')}, "
            f"config={dataset_sha256}. Re-run the notebook."
        )

    print(f"Bundle verified: {BUNDLE_PATH}")
    print(f"  Model: {bundle['model_name']}")
    print(f"  Threshold: {bundle['threshold']:.4f}")
    print(f"  Features: {len(bundle['features'])}")
    print(f"  Recipes: {len(bundle['recipes'])}")
    print(f"  Vocab: {len(bundle['vocab'])}")
    print(f"  Dataset SHA256: {bundle['dataset_sha256'][:16]}...")

    return bundle


def main():
    verify_bundle()


if __name__ == "__main__":
    main()