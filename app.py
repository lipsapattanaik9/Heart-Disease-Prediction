import streamlit as st
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Prediction")
st.write("A simple, production-ready Streamlit app to predict heart disease risk using a trained ML model.")

MODEL_PATH = Path("model/heart_model.pkl")

@st.cache_resource(show_spinner=False)
def load_model():
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

with st.sidebar:
    st.header("Navigation")
    st.page_link("app.py", label="Predict", icon="❤️")
    st.page_link("pages/1_Train_and_Evaluate.py", label="Train & Evaluate", icon="🧠")
    st.page_link("pages/2_Dataset_Explorer.py", label="Dataset Explorer", icon="📊")
    st.page_link("pages/3_About.py", label="About", icon="ℹ️")
    st.divider()
    if model is None:
        st.warning("No saved model found. Go to **Train & Evaluate** to train or upload one.")
    else:
        st.success("Model loaded ✔️")

st.subheader("Patient Details")

with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=45, step=1)
        sex = st.selectbox("Sex", ["Female", "Male"])
        cp = st.selectbox("Chest pain type (cp)", options=[0,1,2,3], 
                          help="0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic")
        trestbps = st.number_input("Resting blood pressure (mm Hg)", min_value=70, max_value=220, value=130)
        chol = st.number_input("Serum cholesterol (mg/dl)", min_value=100, max_value=700, value=230)
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs)", options=[0,1])

    with col2:
        restecg = st.selectbox("Resting ECG (restecg)", options=[0,1,2], 
                               help="0: Normal, 1: ST-T abnormality, 2: LV hypertrophy")
        thalach = st.number_input("Max heart rate achieved (thalach)", min_value=60, max_value=250, value=150)
        exang = st.selectbox("Exercise induced angina (exang)", options=[0,1])
        oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox("Slope of peak exercise ST (slope)", options=[0,1,2])
        ca = st.selectbox("Major vessels colored by fluoroscopy (ca)", options=[0,1,2,3,4])
        thal = st.selectbox("Thalassemia (thal)", options=[0,1,2,3])

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    if model is None:
        st.error("No model available. Please go to **Train & Evaluate** to create one or upload a saved model (model/heart_model.pkl).")
    else:
        # Encode inputs
        sex_val = 1 if sex == "Male" else 0
        features = np.array([[
            age, sex_val, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal
        ]], dtype=float)

        try:
            # Works for raw estimators or sklearn Pipelines
            proba = None
            if hasattr(model, "predict_proba"):
                proba = float(model.predict_proba(features)[0,1])
            pred = int(model.predict(features)[0])

            st.markdown("### Result")
            if pred == 1:
                st.error(f"⚠️ **High risk** of heart disease. (Prediction = 1){' | Probability: {:.2%}'.format(proba) if proba is not None else ''}")
            else:
                st.success(f"✅ **Low risk** of heart disease. (Prediction = 0){' | Probability: {:.2%}'.format(proba) if proba is not None else ''}")
        except Exception as e:
            st.exception(e)
            st.info("Tip: If you trained in Colab, save a **Pipeline** (e.g., `StandardScaler` + estimator) to avoid feature-scaling mismatches.")
