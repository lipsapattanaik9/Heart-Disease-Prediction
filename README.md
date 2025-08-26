# Heart Disease Prediction – Streamlit App

A complete, production-ready Streamlit frontend for heart disease prediction. You can use your existing model from Google Colab or train a new one inside the app.

## 🔧 Local Setup (No Streamlit account required)

1. **Install Python 3.8+** and Git (optional).
2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate    # Windows
   source .venv/bin/activate # macOS/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. If you already have a trained model from Colab, save it as `model/heart_model.pkl`:
   ```python
   # In Colab after training
   import pickle
   from google.colab import files

   # If you used scaling/encoding, wrap them in a Pipeline and save the Pipeline
   with open("heart_model.pkl", "wb") as f:
       pickle.dump(pipeline_or_model, f)

   files.download("heart_model.pkl")
   ```
   Put this file into the `model/` folder locally.
5. Run the app:
   ```bash
   streamlit run app.py
   ```

## ☁️ Deploy Online (Streamlit Community Cloud)

> Requires a free Streamlit account.

1. **Create a GitHub repo** and push this project.
2. Go to **https://streamlit.io/cloud** → Sign in → **New app**.
3. Select your repo, branch (e.g., `main`), and **file path**: `app.py`.
4. Click **Deploy**. Streamlit will read `requirements.txt`, install deps, and run the app.
5. To include your trained model, commit `model/heart_model.pkl` to the repo or upload it via the Cloud app's file manager.

## 📁 Project Structure

```
heart_disease_streamlit_project/
├── app.py
├── pages/
│   ├── 1_Train_and_Evaluate.py
│   ├── 2_Dataset_Explorer.py
│   └── 3_About.py
├── model/
│   └── heart_model.pkl   # (optional; created after training or copied from Colab)
├── utils/
│   └── __init__.py
├── requirements.txt
└── README.md
```

## 🧠 Notes on the Model

- Recommended: use a **scikit-learn Pipeline** (e.g., `StandardScaler` + classifier) so the app can safely handle scaling/encoding.
- Default feature order in the Predict page follows the common UCI heart dataset:
  `['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']`
- The **Train & Evaluate** page lets you upload any CSV, choose the target column, pick features, and train a model. It saves a `Pipeline` to `model/heart_model.pkl`.

## 🔐 Secrets (Optional)

If you later integrate remote storage or APIs:
- Locally, create `.streamlit/secrets.toml`.
- On Streamlit Cloud, go to **App → Settings → Secrets** and paste the same keys.

## 🛠 Troubleshooting

- **ModuleNotFoundError**: Make sure `pip install -r requirements.txt` completed successfully.
- **Model not found**: Train with the **Train & Evaluate** page or copy `model/heart_model.pkl` from Colab.
- **Different feature order**: Ensure the features in training and prediction match; Pipelines help avoid this.
- **Large CSVs**: Preprocess offline to keep the Cloud app responsive.
