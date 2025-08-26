import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️")

st.title("ℹ️ About this Project")
st.markdown("""
**Heart Disease Prediction** web app built with **Streamlit** and **scikit-learn**.

- **Predict** page: Enter patient details and get a risk prediction.
- **Train & Evaluate**: Upload a dataset, train models (LogReg, RF, GBDT) and save a Pipeline.
- **Dataset Explorer**: Quick EDA with histograms and correlations.

### Tips for Best Results
- Save a **sklearn Pipeline** (scaler + model) as `model/heart_model.pkl`.
- Use consistent feature order in both training and prediction.

### Credits
- UCI Heart Disease dataset/Cleveland variant is commonly used for demos.
""")
