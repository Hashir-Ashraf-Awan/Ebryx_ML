import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Wine Classifier", layout="centered")
st.title("🍷 Wine Class Predictor (KNN / NB / DT / LR)")

# Load models
knn_model = joblib.load("knn_model.pkl")
nb_model = joblib.load("nb_model.pkl")
dt_model = joblib.load("dt_model.pkl")
lr_model = joblib.load("lr_model.pkl")

model_choice = st.selectbox("Choose Classifier", ["KNN", "Naive Bayes", "Decision Tree", "Logistic Regression"])

st.subheader("🎛️ Set Feature Values")
def safe_float_input(label, default, min_val=None, max_val=None):
    val = st.text_input(label, value=str(default))
    try:
        float_val = float(val)
        if min_val is not None and float_val < min_val:
            st.error(f"❌ {label} must be at least {min_val}.")
            st.stop()
        if max_val is not None and float_val > max_val:
            st.error(f"❌ {label} must be at most {max_val}.")
            st.stop()
        return float_val
    except ValueError:
        st.error(f"❌ Please enter a valid number for {label}.")
        st.stop()
# Feature sliders
Alcohol = safe_float_input("Alcohol", 13.0, 11.0, 15.5)
Malic_acid = safe_float_input("Malic acid", 2.0, 0.5, 5.8)
Acl = safe_float_input("Acl (Ash)", 2.3, 1.0, 3.5)
Mg = safe_float_input("Mg (Magnesium)", 100, 70, 162)
Phenols = safe_float_input("Total Phenols", 2.0, 0.9, 4.0)
Flavanoids = safe_float_input("Flavanoids", 2.0, 0.3, 5.0)
Nonflavanoid_phenols = safe_float_input("Nonflavanoid Phenols", 0.3, 0.1, 1.0)
Proanth = safe_float_input("Proanthocyanins", 1.5, 0.4, 4.0)
Color_int = safe_float_input("Color Intensity", 5.0, 1.0, 13.0)
Hue = safe_float_input("Hue", 1.0, 0.5, 1.7)
OD = safe_float_input("OD280/OD315 of Diluted Wines", 2.5, 1.0, 4.0)
Proline = safe_float_input("Proline", 800, 280, 1700)

input_data = np.array([[Alcohol, Malic_acid, Acl, Mg, Phenols, Flavanoids,
                        Nonflavanoid_phenols, Proanth, Color_int, Hue, OD, Proline]])

if st.button("Predict"):
    if model_choice == "KNN":
        model = knn_model
    elif model_choice == "Naive Bayes":
        model = nb_model
    elif model_choice == "Decision Tree":
        model = dt_model
    else:
        model = lr_model

    pred = model.predict(input_data)[0]
    st.success(f"🍷 Predicted Wine Class: **{pred}**")

@st.cache_data
def load_test_data():
    df = pd.read_csv("test_data.csv")
    X = df.drop(['label'], axis=1).values
    y = df['label'].values
    return X, y

st.subheader("📊 Model Evaluation on Test Set")

X_test, y_test = load_test_data()

if model_choice == "KNN":
    model = knn_model
elif model_choice == "Naive Bayes":
    model = nb_model
elif model_choice == "Decision Tree":
    model = dt_model
else:
    model = lr_model

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
labels = sorted(np.unique(y_test))

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
st.pyplot(fig)

report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.style.format({'precision': "{:.2f}", 'recall': "{:.2f}", 'f1-score': "{:.2f}"}))
