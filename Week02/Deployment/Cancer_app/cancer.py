import streamlit as st
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Load models
knn_model = joblib.load('knn_model.pkl')
nb_model = joblib.load('nb_model.pkl')
dt_model = joblib.load('dt_model.pkl')

# Load label encoder
label_encoder = joblib.load("label_encoder.pkl")  # Make sure you saved it

st.title("🔬 Breast Cancer Prediction App")

# Dropdown to select model
model_choice = st.selectbox("Select a Model", ['KNN', 'Naive Bayes', 'Decision Tree'])

# Define input fields
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst'
]

# Input fields
user_input = []
st.subheader("🧬 Input Features:")
for name in feature_names:
    val = st.number_input(f"{name}", value=0.0)
    user_input.append(val)

# Predict button
if st.button("Predict"):
    X = np.array(user_input).reshape(1, -1)

    if model_choice == 'KNN':
        prediction = knn_model.predict(X)
    elif model_choice == 'Naive Bayes':
        prediction = nb_model.predict(X)
    else:
        prediction = dt_model.predict(X)

    result = "Malignant" if prediction[0] == 1 else "Benign"
    st.success(f"🩺 Prediction: **{result}**")

st.write(f"✅ You selected: **{model_choice}** model")

@st.cache_data
def load_test_data():
    df = pd.read_csv("test_data.csv")  # Must have 'label' column
    X = df.drop(['label'], axis=1).values
    y = df['label'].values
    return X, y

st.subheader("📊 Model Evaluation on Test Set")

X_test, y_test = load_test_data()

# Select model
if model_choice == "KNN":
    model = knn_model
elif model_choice == "Naive Bayes":
    model = nb_model
else:
    model = dt_model

# Predict
y_pred = model.predict(X_test)
y_test_decoded = label_encoder.inverse_transform(y_test)
y_pred_decoded = label_encoder.inverse_transform(y_pred)

# Confusion Matrix
labels = label_encoder.classes_
cm = confusion_matrix(y_test_decoded, y_pred_decoded)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
st.pyplot(fig)

# Classification Report
report = classification_report(y_test_decoded, y_pred_decoded, target_names=labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.style.format({'precision': "{:.2f}", 'recall': "{:.2f}", 'f1-score': "{:.2f}"}))
