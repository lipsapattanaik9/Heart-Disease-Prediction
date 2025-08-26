import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt

st.set_page_config(page_title="Train & Evaluate", page_icon="🧠", layout="wide")

st.title("🧠 Train & Evaluate")
st.write("Upload a heart dataset or use your own features to train a model. The app will save a **Pipeline** to `model/heart_model.pkl`.")

MODEL_PATH = Path("model/heart_model.pkl")
DEFAULT_FEATURES = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']

with st.sidebar:
    st.page_link("app.py", label="← Back to Predict", icon="❤️")

st.subheader("1) Upload dataset")
uploaded = st.file_uploader("Upload CSV", type=["csv"], help="Expected columns include features and a binary target column")
target_col = st.text_input("Target column name", value="target")

algo = st.selectbox("Select algorithm", ["LogisticRegression", "RandomForest", "GradientBoosting"])
test_size = st.slider("Test size", 0.1, 0.4, value=0.2, step=0.05)
random_state = st.number_input("Random state", value=42, step=1)

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Preview:", df.head())

    # Guess features
    cols = df.columns.tolist()
    if target_col not in cols:
        st.error(f"Target column '{target_col}' not found in uploaded CSV.")
        st.stop()

    feature_candidates = [c for c in cols if c != target_col]
    selected_features = st.multiselect("Select features to use", options=feature_candidates, default=[c for c in DEFAULT_FEATURES if c in feature_candidates])

    if not selected_features:
        st.warning("Select at least one feature.")
        st.stop()

    X = df[selected_features].copy()
    y = df[target_col].copy()

    # Basic cleaning
    X = X.replace({"?": np.nan}).astype(float)
    if y.dtype == object:
        y = y.replace({"yes":1,"no":0,"true":1,"false":0}).astype(int)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Build pipeline
    if algo == "LogisticRegression":
        clf = LogisticRegression(max_iter=1000)
    elif algo == "RandomForest":
        clf = RandomForestClassifier(n_estimators=300, random_state=random_state)
    else:
        clf = GradientBoostingClassifier(random_state=random_state)

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False) if (X_train.min().min() < 0) else StandardScaler()),  # simple heuristic
        ("model", clf)
    ])

    # Train
    with st.spinner("Training..."):
        pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
    }
    # Probabilities for ROC if available
    try:
        y_proba = pipe.predict_proba(X_test)[:,1]
        metrics["ROC AUC"] = roc_auc_score(y_test, y_proba)
    except Exception:
        y_proba = None

    st.subheader("2) Metrics")
    st.write(pd.DataFrame([metrics]))

    st.subheader("3) Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    for (i, j), z in np.ndenumerate(cm):
        plt.text(j, i, str(z), ha='center', va='center')
    st.pyplot(fig)

    if y_proba is not None:
        st.subheader("4) ROC Curve")
        fig2 = plt.figure()
        RocCurveDisplay.from_predictions(y_test, y_proba)
        st.pyplot(fig2)

    st.subheader("5) Save model")
    if st.button("💾 Save to model/heart_model.pkl"):
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(pipe, f)
        st.success("Saved! You can now use the **Predict** page.")
else:
    st.info("Upload a CSV to start. Expected features often include: " + ", ".join(DEFAULT_FEATURES) + " and binary target column 'target'.")
