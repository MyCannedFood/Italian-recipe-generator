# Streamlit Deployment TODO

## 1. Extract code from notebook to Python modules
- [ ] `recommender.py` — `ItalianRecipeRecommender` class (M0-M3)
- [ ] `features.py` — `candidate_features()`, `make_queries()`, `clean_ingredients_v2()`, `ingredient_keys()`, `to_features()`
- [ ] `models.py` — `XGBQueryRanker`, `GroupTimeImputer`, `make_preprocessor()`, `make_pipeline()`
- [ ] `utils.py` — `ranking_metrics()`, `relevance_metrics()`, `bootstrap_ci()`, `verdict()`, `fmt_ci()`
- [ ] `config.py` — constants: `ML_FEATURES`, `ML_NUMERIC`, `ML_CATEGORICAL`, `N_CANDIDATES`, `STAPLES`, `GENERIC_WORDS`, `vocab`, `idf`, `recipe_ingredients`, `recipes` dataframe

## 2. Add single-query M4 feature computation
- [ ] `compute_m4_features(pantry: list[str]) -> pd.DataFrame` — computes coverage, n_missing, idf_owned_mean, query_used, cos_both, etc. for ONE user query against all recipes (or top-N M3 candidates)

## 3. Save model artifacts (run once, commit outputs)
- [ ] Create `save_model.py` script that:
  - Loads data, trains M3 (`m3_best`), trains M4 (`best_model`), finds `best_threshold`
  - Saves `model_bundle.joblib` with: M3, M4, threshold, features, vocab, idf, recipe_ingredients, recipes df
- [ ] Run it and commit `model_bundle.joblib` + `recipes_italia_final.csv` to repo

## 4. Create Streamlit app
- [ ] `app.py` — main entry point
  - `@st.cache_resource` load model bundle
  - UI: text input for ingredients (comma-separated), slider for k
  - Button "Rekomendasikan" → call `recommend_from_pantry()` + M4 re-rank
  - Display results: recipe title, score, coverage %, missing ingredients
- [ ] `requirements.txt` — pinned versions from notebook

## 5. Deploy to Streamlit Community Cloud
- [ ] Push to GitHub (public repo)
- [ ] Connect at share.streamlit.io
- [ ] Set `RECIPES_CSV` path or bundle data in model artifact
- [ ] Test deployed app

## 6. Optional improvements
- [ ] Add "Ideal vs Realistis" query mode toggle
- [ ] Show M3 vs M4 comparison
- [ ] Cache M3 candidates to speed up repeated queries
- [ ] Add error handling for empty input / no candidates