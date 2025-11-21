import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import pickle
from dotenv import load_dotenv
import google.generativeai as genai
import os
import json
from sklearn.metrics import accuracy_score

VOTE_FILE = "votes.json"

def load_votes():
    if os.path.exists(VOTE_FILE):
        with open(VOTE_FILE, "r") as f:
            return json.load(f)
    else:
        return {"biased": 0, "not_biased": 0}

def save_votes(votes):
    with open(VOTE_FILE, "w") as f:
        json.dump(votes, f)

# load gemini API key
load_dotenv()  
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY is None:
    st.error("GEMINI_API_KEY not found in .env")

genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel("gemini-2.5-flash")

# function to explain SHAP values using Gemini
def explain_shap_with_gemini(feature_names, shap_values, prediction, proba):
    prompt = (
        "Explain these SHAP values for a hiring decision model.\n\n"
        f"Model predicted class: {prediction} (probability {proba:.3f})\n\n"
        "Feature contributions (positive = pushes toward hire, negative = pushes toward reject):\n"
    )
    for name, val in zip(feature_names, shap_values):
        prompt += f"- {name}: {val:.4f}\n"

    response = model_gemini.generate_content(prompt)
    return response.text

st.set_page_config(page_title="Recruitment XAI Demo", layout="wide")


# Page Navigation State

if "page" not in st.session_state:
    st.session_state.page = "intro"

def go_to_explorer():
    st.session_state.page = "explorer"


# Page 1: Intro Page

if st.session_state.page == "intro":

    # CSS
    st.markdown("""
    <style>
        .hero-title {
            font-size: 2.6rem;
            font-weight: 700;
            text-align: center;
            color: #2b2b2b;
            margin-bottom: 0.2rem;
        }
        .hero-subtitle {
            text-align: center;
            color: #666;
            font-size: 20rem;
            margin-bottom: 2rem;
        }
        .info-card {
            background: #f8f9fc;
            padding: 1.2rem 1.5rem;
            border-radius: 12px;
            border: 1px solid #e6e6e6;
            margin-bottom: 1.2rem;
        }
        .section-title {
            text-align: center;
            font-size: 1.4rem;
            font-weight: 600;
            margin-top: 2rem;
            color: #333;
        }
        .divider {
            margin: 1.5rem 0;
            border-top: 1px solid #e6e6e6;
        }
    </style>
    """, unsafe_allow_html=True)


    
    st.markdown("<h1 class='hero-title'>Explainable AI for Recruitment Decisions</h1>", unsafe_allow_html=True)
    st.markdown("<h5 class='hero-subtitle'>Peek inside how machine learning models make hiring decisions.</p>", unsafe_allow_html=True)


    
    # Introduction Box 
    st.markdown("""
    <div class='info-card'>
        Welcome to the Recruitment Decision Explainer!  
        This interactive demo shows <b>why</b> a model predicts the way it does using SHAP — 
        a powerful framework for interpreting complex models.
        <br><br>
        Here's what you’ll explore:
        <ul>
            <li>How the model reasons at a <b>global</b> level</li>
            <li>Why it made a specific <b>hiring decision</b></li>
            <li>Which features <b>helped</b> or <b>hurt</b> a candidate</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)



    # SHAP Explanation 
    st.markdown("<div class='section-title'>What is SHAP?</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-card'>
        <b>SHAP (SHapley Additive exPlanations)</b> is a game theoretic approach to explain the output of any machine learning model.
        <br><br>
        In simple terms:  
        <b>SHAP tells you which features pushed the model toward yes, and which pushed it toward no.</b>
    </div>
    """, unsafe_allow_html=True)

    # Data
    st.markdown("<div class='section-title'>Data</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-card'>
    This demo uses a recruitment dataset containing candidate features and a hiring decision label.
    The goal is to understand how each feature contributes to the model’s prediction.
    <br><br>
    You can view the dataset here:  
    <a href="https://www.kaggle.com/datasets/rabieelkharoua/predicting-hiring-decisions-in-recruitment-data" target="_blank">💼 recruitment_data.csv</a>
    </div>
    """, unsafe_allow_html=True)

    # Divider
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Explore Button 
    st.markdown("<h3 style='text-align: center'>Ready to explore?</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([4, 1, 4])

    with col2:
        st.button("Explore the Model", on_click=go_to_explorer)

    st.stop()




# Page 2: SHAP Explorer


st.title("Recruitment Decision Explainer")

df = pd.read_csv("recruitment_data.csv")
st.write("### Preview of data")
st.dataframe(df.head())

# load model, data, SHAP values
@st.cache_resource
def load_shap():
    with open("rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("X_train.pkl", "rb") as f:
        X_train = pickle.load(f)

    #X_test = df.drop("HiringDecision", axis=1)
    with open("X_test.pkl", "rb") as f:
        X_test = pickle.load(f)

    with open("y_test.pkl", "rb") as f:
        y_test = pickle.load(f)

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    return model, X_train, X_test, y_test, shap_values, explainer

model, X_train, X_test, y_test, shap_values, explainer = load_shap()

y_test_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_test_pred)
st.write(f"### Model Accuracy on Test Set: {accuracy:.2%}")



# Extract class-1 SHAP values
class1_shap_values = shap_values[:, :, 1]

# Sidebar Navigation 
view = st.sidebar.radio("View", ["Global Explanation", "Local Explanation", "Customize Your Own"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Do you think the model is biased?")

votes = load_votes()

vote_choice = st.sidebar.radio(
    "Your Vote",
    ["I think it IS biased", "I think it is NOT biased"],
    index=None,
    key="bias_vote_selection"
)

if st.sidebar.button("Submit Vote"):
    if vote_choice is None:
        st.sidebar.warning("Please choose an option before voting.")
    else:
        if vote_choice == "I think it IS biased":
            votes["biased"] += 1
        else:
            votes["not_biased"] += 1

        save_votes(votes)
        st.sidebar.success("Thanks for voting!")

# Show vote results
total_votes = votes["biased"] + votes["not_biased"]

if total_votes > 0:
    pct_biased = votes["biased"] / total_votes * 100
    pct_not_biased = votes["not_biased"] / total_votes * 100

    st.sidebar.markdown("### Vote Results")
    st.sidebar.write(f"**Biased:** {pct_biased:.1f}%")
    st.sidebar.write(f"**Not Biased:** {pct_not_biased:.1f}%")

else:
    st.sidebar.info("No votes yet.")

# Dataset Details
with st.expander("Dataset Details"):
    st.markdown("""
    ### Features Overview

    **Age**  
    • Description: Age of the candidate  
    • Range: 20–50  
    • Type: Integer  

    **Gender**  
    • Description: Gender of the candidate  
    • Categories: 0 = Male, 1 = Female  
    • Type: Binary  

    **Education Level**  
    • Categories: 1 = Bachelor's (Type 1), 2 = Bachelor's (Type 2), 3 = Master's, 4 = PhD  
    • Type: Categorical  

    **Experience Years**  
    • Range: 0–15  
    • Type: Integer  

    **Previous Companies Worked**  
    • Range: 1–5  
    • Type: Integer  

    **Distance From Company**  
    • Range: 1–50 km  
    • Type: Float  

    **Interview Score**  
    • Range: 0–100  
    • Type: Integer  

    **Skill Score**  
    • Range: 0–100  
    • Type: Integer  

    **Personality Score**  
    • Range: 0–100  
    • Type: Integer  

    **Recruitment Strategy**  
    • Categories: 1 = Aggressive, 2 = Moderate, 3 = Conservative  
    • Type: Categorical  

    ---
    ### Target Variable

    **Hiring Decision**  
    • 0 = Not hired  
    • 1 = Hired  
    • Type: Binary  

    ---
    ### Dataset Info  

    • Records: 1500  
    • Features: 10  
    • Target: HiringDecision  
    """)


if view == "Global Explanation":
    st.subheader("Global Feature Importance (class 1: Hired)")

    
    # SHAP Beeswarm Plot
    fig, ax = plt.subplots()
    shap.plots.beeswarm(class1_shap_values,  show=False)
    st.pyplot(fig)

    mean_shap = np.abs(class1_shap_values.values).mean(axis=0)

    # Button for Gemini explanation
    if st.button("Explain Global SHAP with Gemini"):
        # Build prompt text
        feature_names = X_train.columns.tolist()
        prompt = "Explain the global SHAP importance of features for a hiring decision model.\n\n"
        prompt += "Mean absolute SHAP values:\n"

        # Below two lines were generated by GPT-5 on November 20, 2025 at 14:15 PM
        for name, val in zip(feature_names, mean_shap):
            prompt += f"- {name}: {val:.4f}\n"

        prompt += (
            "\nSummarize which features are most influential, "
            "which ones matter least, and give an HR-friendly interpretation."
        )

        # Call Gemini API
        try:
            response = model_gemini.generate_content(prompt)
            st.markdown("### Gemini Explanation")
            st.write(response.text)
        except Exception as e:
            st.error(f"Gemini API Error: {e}")

elif view == "Local Explanation":
    st.subheader("Local Explanation")
    st.markdown("Select a candidate from the test set to see their individual prediction and SHAP explanation.")
    idx = st.slider("Pick a sample", 0, len(X_test) - 1, 0)
    x_row = X_test.iloc[[idx]]

    pred = model.predict(x_row)[0]
    proba = model.predict_proba(x_row)[0]

    st.write(f"**Predicted Class:** {pred}")
    st.write(f"Probability (Hired = 1): {proba[1]:.3f}")

    # SHAP Explanation
    shap_row = class1_shap_values[idx]
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_row, show=False)
    st.pyplot(fig)

    # Gemini Explanation Button
    if st.button("Explain with Gemini"):
        text = explain_shap_with_gemini(
            X_train.columns,
            shap_row.values,
            pred,
            proba[1]
        )
        st.markdown("### Gemini Explanation")
        st.write(text)

elif view == "Customize Your Own":
    st.subheader("Customize Your Own Candidate")

    st.markdown("Adjust the values below to create your own candidate profile.")

    # User Input Form 

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 20, 50, 30)
    with col2:
        gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")

    col3, col4 = st.columns(2)
    with col3:
        edu = st.selectbox("Education Level", [1, 2, 3, 4], 
                        format_func=lambda x: {1:"Bachelor (Type 1)", 2:"Bachelor (Type 2)", 3:"Master", 4:"PhD"}[x])
    with col4:
        exp_years = st.slider("Experience Years", 0, 15, 3)

    col5, col6 = st.columns(2)
    with col5:
        prev_comp = st.slider("Previous Companies Worked", 1, 5, 2)
    with col6:
        distance = st.number_input("Distance From Company (km)", min_value=1.0, max_value=50.0, value=10.0, step=0.5)

    col7, col8 = st.columns(2)
    with col7:
        interview = st.slider("Interview Score", 0, 100, 80)
    with col8:
        skill = st.slider("Skill Score", 0, 100, 75)

    col9, col10 = st.columns(2)
    with col9:
        personality = st.slider("Personality Score", 0, 100, 65)
    with col10:
        strategy = st.selectbox(
            "Recruitment Strategy", 
            [1, 2, 3], 
            format_func=lambda x: {1:"Aggressive", 2:"Moderate", 3:"Conservative"}[x]
        )


    # Create a dataframe with the SAME column order as X_train
    input_df = pd.DataFrame([[
        age, gender, edu, exp_years, prev_comp, distance, interview, skill, personality, strategy
    ]], columns=X_train.columns)

    st.write("### Candidate Profile")
    st.dataframe(input_df.T)

    # Make Prediction
    pred_proba = model.predict_proba(input_df)[0]
    pred_class = model.predict(input_df)[0]

    st.write(f"**Predicted Class:** {pred_class}")
    st.write(f"**Probability (Hired = 1):** {pred_proba[1]:.3f}")

    # SHAP Explanation 
    st.write("### SHAP Explanation for Your Custom Input")

    custom_shap = explainer(input_df)[:, :, 1][0]   

    fig, ax = plt.subplots()
    shap.plots.waterfall(custom_shap, show=False)
    st.pyplot(fig)

    # Gemini Explanation Button
    if st.button("Explain with Gemini (AI Explanation)"):
        try:
            gemini_text = explain_shap_with_gemini(
                X_train.columns,
                custom_shap.values,
                pred_class,
                pred_proba[1]
            )
            st.markdown("### Gemini Explanation")
            st.write(gemini_text)
        except Exception as e:
            st.error(f"Gemini API Error: {e}")


