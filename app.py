import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.set_page_config(page_title="Recruitment Decision Explainer", layout="wide")
st.title("Recruitment Decision Explainer")

st.markdown("""
This app trains a Random Forest on your data and lets you explore:
- **Global** explanations (beeswarm of SHAP values)
- **Local** explanations (per-row waterfall plot)
""")

df = pd.read_csv("recruitment_data.csv")

st.write("### Preview of data")
st.dataframe(df.head())




@st.cache_resource
def load_shap():
    # Load model and X_train
    with open("rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("X_train.pkl", "rb") as f:
        X_train = pickle.load(f)

    # Load dataset again (fast)
    df = pd.read_csv("recruitment_data.csv")
    X_test = df.drop("HiringDecision", axis=1)

    # Build SHAP explainer and full SHAP values
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    return model, X_train, X_test, shap_values


model, X_train, X_test, shap_values = load_shap()

# Select class 1 SHAP values properly
class1_shap_values = shap_values[:, :, 1]






view = st.sidebar.radio("What do you want to explore?", ["Global SHAP", "Single candidate"])

if view == "Global SHAP":
    st.subheader("Global feature importance (class 1)")

    max_display = st.slider(
        "Number of features to display", 
        min_value=1, 
        max_value=len(X_train.columns),
        value=min(10, len(X_train.columns))
    )

    # beeswarm (need to capture fig for Streamlit)
    shap_values_for_plot = class1_shap_values
    with st.spinner("Rendering SHAP beeswarm..."):
        fig, ax = plt.subplots()
        shap.plots.beeswarm(
            shap_values_for_plot, 
            max_display=max_display, 
            show=False
        )
        st.pyplot(fig)

elif view == "Single candidate":
    st.subheader("Local explanation for one candidate")

    # pick index in test set
    idx = st.slider("Pick a test sample index", 0, len(X_test) - 1, 0)
    x_row = X_test.iloc[[idx]]
    st.write("#### Candidate features")
    st.dataframe(x_row.T)

    # prediction
    pred_proba = model.predict_proba(x_row)[0]
    pred_class = model.predict(x_row)[0]

    st.write(f"**Predicted class**: {int(pred_class)}")
    st.write(f"**P(class=1)**: {pred_proba[1]:.3f},  **P(class=0)**: {pred_proba[0]:.3f}")

    # local SHAP waterfall
    st.write("#### SHAP waterfall (class 1)")
    shap_value_row = class1_shap_values[idx]

    with st.spinner("Rendering local SHAP plot..."):
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_value_row, show=False)
        st.pyplot(fig)