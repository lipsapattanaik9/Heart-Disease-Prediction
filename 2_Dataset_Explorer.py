import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dataset Explorer", page_icon="📊", layout="wide")

st.title("📊 Dataset Explorer")
st.write("Upload a dataset to explore distributions and correlations.")

with st.sidebar:
    st.page_link("app.py", label="← Back to Predict", icon="❤️")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if not uploaded:
    st.info("Upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded)
st.subheader("Preview")
st.dataframe(df.head())

st.subheader("Summary")
st.write(df.describe())

num_cols = df.select_dtypes(include='number').columns.tolist()
if num_cols:
    sel = st.selectbox("Choose a numeric column to plot", num_cols)
    fig = plt.figure()
    df[sel].hist(bins=30)
    plt.title(f"Histogram of {sel}")
    st.pyplot(fig)

    st.subheader("Correlation (numeric only)")
    corr = df[num_cols].corr()
    st.dataframe(corr.style.background_gradient(cmap="Blues"))
else:
    st.warning("No numeric columns found to plot.")
