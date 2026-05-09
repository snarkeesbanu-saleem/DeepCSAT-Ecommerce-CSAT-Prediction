import streamlit as st
import joblib
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# ────────────────────────────────────────────────
# Paths — all files should be in the same folder as app.py
# ────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent

MODEL_PATH      = BASE_DIR / "deepcsat_xgb_model.joblib"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.joblib"
SVD_PATH        = BASE_DIR / "svd_reducer.joblib"
COLUMNS_PATH    = BASE_DIR / "training_columns.pkl"

# ────────────────────────────────────────────────
# Load models and training columns
# ────────────────────────────────────────────────

@st.cache_resource
def load_resources():
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        svd = joblib.load(SVD_PATH)

        # Try to load exact training column names
        try:
            with open(COLUMNS_PATH, "rb") as f:
                training_columns = pickle.load(f)
            st.session_state['training_columns'] = training_columns
            st.session_state['use_exact_columns'] = True
        except FileNotFoundError:
            st.session_state['use_exact_columns'] = False
            st.session_state['training_columns'] = None

        return model, vectorizer, svd
    except Exception as e:
        st.error(f"Failed to load model files: {str(e)}")
        st.info("Please make sure the following files exist in the same folder as app.py:")
        st.code("""
- deepcsat_xgb_model.joblib
- tfidf_vectorizer.joblib
- svd_reducer.joblib
- training_columns.pkl (optional but strongly recommended)
        """)
        st.stop()

model, vectorizer, svd = load_resources()

# ────────────────────────────────────────────────
# Page config & layout
# ────────────────────────────────────────────────

st.set_page_config(page_title="DeepCSAT Predictor", layout="wide")

st.title("DeepCSAT – Customer Satisfaction Score Predictor")
st.markdown("Enter the customer interaction details below and click **Predict**")

# ─── Input fields ───
col1, col2 = st.columns([3, 2])

with col1:
    remark = st.text_area(
        "Customer Remark",
        value="Very bad service, order delayed, pathetic support, not resolved",
        height=140,
        key="remark_input"
    )

with col2:
    resp_time = st.slider(
        "First Response Time (minutes)",
        min_value=1, max_value=180, value=30, step=5,
        key="resp_time"
    )
    handling_time = st.slider(
        "Handling Time (minutes)",
        min_value=1, max_value=60, value=8, step=1,
        key="handling_time"
    )
    item_price = st.slider(
        "Item Price (₹)",
        min_value=100, max_value=50000, value=2500, step=100,
        key="item_price"
    )

shift = st.selectbox(
    "Agent Shift",
    ["Morning", "Afternoon", "Evening", "Split", "Night"],
    index=2,
    key="shift_select"
)

category = st.selectbox(
    "Category",
    ["Order Related", "Returns", "Product Queries", "Cancellation", "Payments related"],
    index=0,
    key="category_select"
)

# ─── Predict button ───
if st.button("Predict CSAT", type="primary", key="predict_button_unique"):

    with st.spinner("Running prediction..."):

        try:
            # Prepare input data
            input_data = pd.DataFrame([{
                'remark': remark.strip() or "no_remarks",
                'response_time_min': resp_time,
                'connected_handling_time': handling_time,
                'Item_price': item_price,
                'order_to_survey_days': 2,  # fixed value – you can make it a slider
                'Agent Shift': shift,
                'category': category,
            }])

            # ─── Text features ───
            text_vec = vectorizer.transform(input_data['remark'])
            text_svd = svd.transform(text_vec)

            # ─── Tabular features ───
            num_cols = ['response_time_min', 'connected_handling_time', 'Item_price', 'order_to_survey_days']
            cat_cols = ['Agent Shift', 'category']

            df_num = input_data[num_cols]
            df_cat = pd.get_dummies(input_data[cat_cols], drop_first=True)

            new_tabular = pd.concat([df_num, df_cat], axis=1)

            # ─── Column alignment (most important part) ───
            if st.session_state.get('use_exact_columns', False):
                new_tabular = new_tabular.reindex(
                    columns=st.session_state['training_columns'],
                    fill_value=0
                )
                st.caption("Using exact training columns from pickle file")
            else:
                st.warning("training_columns.pkl not found → using current columns (may cause shape mismatch)")

            # Combine features
            new_X = np.hstack([text_svd, new_tabular.values.astype(np.float32)])

            # ─── Force shape match (safety net) ───
            expected = model.n_features_in_
            current = new_X.shape[1]

            if current != expected:
                st.warning(f"Shape mismatch: expected {expected}, got {current}")
                if current > expected:
                    new_X = new_X[:, :expected]
                    st.info(f"Trimmed to {expected} columns")
                else:
                    padding = np.zeros((new_X.shape[0], expected - current))
                    new_X = np.hstack([new_X, padding])
                    st.info(f"Padded to {expected} columns")

            # Predict
            pred_mapped = model.predict(new_X)[0]
            pred_csat = int(pred_mapped) + 1

            probabilities = model.predict_proba(new_X)[0]

            # ─── Display results ───
            st.subheader(f"Predicted CSAT Score: ★ {pred_csat} / 5")

            st.markdown("**Class Probabilities**")
            for i, p in enumerate(probabilities, 1):
                st.write(f"CSAT {i}: **{p:.1%}**")

            if pred_csat <= 2:
                st.error("⚠️ HIGH RISK – Consider escalating to supervisor")
            elif pred_csat >= 4:
                st.success("✓ Positive interaction – Well handled!")
            else:
                st.info("→ Neutral outcome – Monitor follow-up")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.info("Common causes:\n"
                    "• Model files not found\n"
                    "• training_columns.pkl missing or incorrect\n"
                    "• Column count mismatch after get_dummies()")