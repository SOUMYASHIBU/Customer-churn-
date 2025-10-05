import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

MODEL_PATH = "model/model.pkl"

st.set_page_config(page_title="Churn Predictor", layout="centered")

st.title("📊 Telecom Churn Prediction")
st.markdown("Interactive demo: enter customer attributes and get churn probability & explanation.")

# Load model
@st.cache_data(show_spinner=False)
def load_model(path):
    data = joblib.load(path)
    return data

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Run `train_and_save_model.py` first to create model/model.pkl.")
    st.stop()

data = load_model(MODEL_PATH)
pipeline = data['pipeline']
label_encoder = data['label_encoder']
artifact_metrics = data.get('metrics', {})

# Display model metrics
st.sidebar.header("Model performance (on hold-out test set)")
st.sidebar.write(f"Accuracy: **{artifact_metrics.get('accuracy', 'N/A')}**")
st.sidebar.write(f"ROC AUC: **{artifact_metrics.get('roc_auc', 'N/A')}**")
conf = artifact_metrics.get('confusion_matrix', None)
if conf:
    st.sidebar.write("Confusion Matrix:")
    st.sidebar.write(conf)

# Get feature lists
num_features = data['feature_names']['num_features']
cat_features = data['feature_names']['cat_features']

# For nicer UI show default values by sampling if possible
# Attempt to load a sample row for defaults if test_data exists
defaults = {}
try:
    X_test, y_test = data.get('test_data', (None, None))
    if X_test is not None:
        sample = X_test.iloc[0]
        for f in num_features + cat_features:
            defaults[f] = sample.get(f, "")
except Exception:
    defaults = {}

st.header("Input customer data")

with st.form("predict_form"):
    cols = st.columns(2)
    inputs = {}
    # Numeric inputs
    for i, feat in enumerate(num_features):
        default_val = float(defaults.get(feat, 0)) if defaults.get(feat, 0) != "" else 0.0
        inputs[feat] = cols[i % 2].number_input(feat, value=default_val)

    # Categorical inputs: get unique categories from training pipeline if possible
    # Attempt to extract categories from fitted OneHotEncoder inside pipeline
    encoder_categories = {}
    try:
        preprocessor = pipeline.named_steps['preprocessor']
        # cat transformer is preprocessor.transformers_[1][1].named_steps['onehot']
        # But we will be defensive:
        cat_trans = None
        for name, trans, cols_in in preprocessor.transformers:
            if name == 'cat':
                cat_trans = trans
                cat_cols = cols_in
                break
        if cat_trans is not None:
            onehot = None
            # cat_trans might be Pipeline
            if hasattr(cat_trans, 'named_steps'):
                onehot = cat_trans.named_steps.get('onehot', None)
            elif cat_trans.__class__.__name__ == 'OneHotEncoder':
                onehot = cat_trans
            if onehot is not None and hasattr(onehot, 'categories_'):
                # categories_ is list aligned with cat_features
                for col_name, cats in zip(cat_cols, onehot.categories_):
                    encoder_categories[col_name] = list(cats)
    except Exception:
        encoder_categories = {}

    for feat in cat_features:
        options = encoder_categories.get(feat, None)
        default_val = defaults.get(feat, "")
        if options:
            if default_val in options:
                inputs[feat] = st.selectbox(feat, options, index=options.index(default_val))
            else:
                inputs[feat] = st.selectbox(feat, options)
        else:
            # fallback text input
            inputs[feat] = st.text_input(feat, value=str(default_val))

    submitted = st.form_submit_button("Predict churn")

if submitted:
    # Build single-row DataFrame
    input_df = pd.DataFrame({k: [v] for k, v in inputs.items()})
    # Run through pipeline
    try:
        prob = pipeline.predict_proba(input_df)[:, 1][0] if hasattr(pipeline, "predict_proba") else pipeline.predict(input_df)[0]
        pred_label = pipeline.predict(input_df)[0]
        pred_label_str = label_encoder.inverse_transform([pred_label])[0] if label_encoder is not None else str(pred_label)

        st.subheader("Prediction")
        st.metric("Churn probability", f"{prob:.4f}")
        st.write(f"Predicted label: **{pred_label_str}**")

        # Explain top contributing features (simple approach: use pipeline.feature_selector scores)
        try:
            fs = pipeline.named_steps.get('feature_selector', None)
            pre = pipeline.named_steps.get('preprocessor', None)
            if fs is not None and pre is not None:
                # Get feature names after preprocessing
                try:
                    # sklearn >= 1.0: get_feature_names_out exists on ColumnTransformer
                    feature_names = pre.get_feature_names_out()
                except Exception:
                    # fallback: create feature names manually (not perfect)
                    feature_names = []
                sel = fs.get_support(indices=True)
                selected = [feature_names[i] for i in sel] if len(feature_names) > 0 else []
                if selected:
                    st.write("Selected features used by model:")
                    st.write(selected)
        except Exception:
            pass

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.write("Model & pipeline loaded from `model/model.pkl`. To retrain, run `train_and_save_model.py` and re-upload model.pkl.")
