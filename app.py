import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
import sys

# Ensure classes are importable for unpickling model bundle (notebook saves them in __main__)
sys.path.insert(0, str(Path(__file__).parent))
from recommender import ItalianRecipeRecommender  # noqa: F401  M3 model class
from models import GroupTimeImputer  # noqa: F401  M4 pipeline step

# Page config
st.set_page_config(
    page_title="Italian Recipe Recommender",
    page_icon="🍝",
    layout="wide",
)

# Load model bundle (cached)
@st.cache_resource
def load_bundle():
    bundle_path = Path(__file__).parent / "model_bundle.joblib"
    if not bundle_path.exists():
        st.error(f"Model bundle not found at {bundle_path}. Run `python save_model.py` first.")
        st.stop()
    return joblib.load(bundle_path)


bundle = load_bundle()
m3 = bundle["m3"]
m4 = bundle["m4"]
threshold = bundle["threshold"]
model_name = bundle["model_name"]
recipes = bundle["recipes"]
recipe_ingredients = bundle["recipe_ingredients"]
vocab = bundle["vocab"]
idf = bundle["idf"]
N_CANDIDATES = bundle["N_CANDIDATES"]
ML_FEATURES = bundle["features"]

# Import after bundle loads (uses config from bundle)
from features import recommend_from_pantry, compute_m4_features
from config import clean_ingredients_v2, ingredient_keys


# UI
st.title("🍝 Italian Recipe Recommender")
st.caption(f"M3 (TF-IDF + Cosine + Coverage) → M4 Re-ranking ({model_name})")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    use_m4 = st.checkbox("Use M4 re-ranking", value=True, help="Enable ML re-ranker on top of M3")
    k = st.slider("Number of recommendations", 1, 10, 5)
    show_debug = st.checkbox("Show debug info", value=False)
    
    st.divider()
    st.header("📊 Model Info")
    st.write(f"**M3**: TF-IDF (both) + Cosine + Coverage (α=0.6)")
    st.write(f"**M4**: {model_name}")
    st.write(f"**Threshold**: {threshold:.3f}")
    st.write(f"**Vocab size**: {len(vocab)}")
    st.write(f"**Recipes**: {len(recipes)}")
    st.write(f"**Candidates**: Top-{N_CANDIDATES} from M3")

# Main input
st.header("🥘 What's in your pantry?")
st.write("Enter ingredients (comma-separated), e.g.: `spaghetti, tomatoes, garlic, basil, parmesan, olive oil`")

pantry_input = st.text_area(
    "Ingredients",
    placeholder="spaghetti, tomatoes, garlic, basil, parmesan, olive oil",
    height=100,
)

col1, col2 = st.columns([1, 4])
with col1:
    recommend_btn = st.button("🔍 Recommend", type="primary", use_container_width=True)
with col2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.rerun()

# Recommendation logic
if recommend_btn and pantry_input.strip():
    # Parse input
    raw_items = [x.strip() for x in pantry_input.split(",") if x.strip()]
    
    with st.spinner("Finding recipes..."):
        try:
            results = recommend_from_pantry(
                pantry=raw_items,
                k=k,
                use_m4=use_m4,
                model=m4 if use_m4 else None,
                threshold=threshold if use_m4 else None,
            )
        except Exception as e:
            st.error(f"Error: {e}")
            if show_debug:
                import traceback
                st.code(traceback.format_exc())
            st.stop()
    
    if results.empty:
        st.warning("No recipes found. Try different ingredients.")
    else:
        st.success(f"Found {len(results)} recommendations")
        
        # Display results
        for i, row in results.iterrows():
            with st.container(border=True):
                cols = st.columns([3, 1, 1])
                with cols[0]:
                    st.subheader(row["Resep"])
                with cols[1]:
                    st.metric("Score", f"{row['Skor']:.3f}")
                with cols[2]:
                    st.metric("Coverage", row["Bahan dimiliki"])
                
                if row["Perlu dibeli"]:
                    st.caption(f"🛒 Need to buy: {row['Perlu dibeli']}")
                else:
                    st.caption("✅ You have all ingredients!")
                
                # Show M4 probability if available
                if "M4_proba" in results.columns and use_m4:
                    st.caption(f"M4 probability: {row.get('M4_proba', 0):.3f} | Rank: {row.get('M4_rank', 'N/A')}")

elif recommend_btn:
    st.warning("Please enter some ingredients first.")

# Debug / Demo section
if show_debug:
    st.divider()
    st.header("🔧 Debug")
    
    if pantry_input.strip():
        raw_items = [x.strip() for x in pantry_input.split(",") if x.strip()]
        cleaned = clean_ingredients_v2(raw_items)
        st.write("**Raw input:**", raw_items)
        st.write("**Cleaned:**", cleaned)
        st.write("**Keys:**", [ingredient_keys(item) for item in cleaned])
        
        # Show M3 candidates
        m3_cands = m3.recommend(cleaned, k=10)
        m3_df = pd.DataFrame(m3_cands, columns=["Index", "Score", "Coverage"])
        m3_df["Title"] = m3_df["Index"].apply(lambda i: recipes.loc[i, "recipe_title"])
        st.write("**Top-10 M3 Candidates:**")
        st.dataframe(m3_df[["Title", "Score", "Coverage"]], use_container_width=True)
        
        # Show M4 features for top candidate
        if use_m4:
            feat_df = compute_m4_features(raw_items)
            st.write("**M4 Features (top candidate):**")
            st.dataframe(feat_df.head(1)[ML_FEATURES].T, use_container_width=True)

# Footer
st.divider()
st.caption("Italian Recipe Recommender — M3 + M4 Learning to Rank | IS411 UAS")