import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🚢 Titanic Survival Predictor (KNN / NB / DT)")

knn_model = joblib.load("knn_model.pkl")
nb_model = joblib.load("nb_model.pkl")
dt_model = joblib.load("dt_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

model_choice = st.selectbox("Choose Classifier", ["KNN", "Naive Bayes", "Decision Tree"])

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

st.subheader("🧾 Enter Passenger Details")

pclass = safe_float_input("Pclass (1 = 1st, 2 = 2nd, 3 = 3rd)", 3, 1, 3)
sex = st.radio("Sex", ["Male", "Female"])
sex_val = 1 if sex == "Male" else 0
age = safe_float_input("Age", 30, 0, 120)
sibsp = safe_float_input("Siblings/Spouses Aboard (SibSp)", 0, 0)
parch = safe_float_input("Parents/Children Aboard (Parch)", 0, 0)
fare = safe_float_input("Fare", 32.2, 0, 1000)

input_data = np.array([[pclass, sex_val, age, sibsp, parch, fare]])

if st.button("Predict"):
    if model_choice == "KNN":
        model = knn_model
    elif model_choice == "Naive Bayes":
        model = nb_model
    else:
        model = dt_model

    pred_encoded = model.predict(input_data)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]
    status = "Not Survived" if pred_label == 0 else "Survived"
    st.success(f"🧬 Predicted Class: {status}")

@st.cache_data
def load_test_data():
    df = pd.read_csv("test_data.csv")  # Should include 'label' column
    X = df.drop(['label'], axis=1).values
    y = df['label'].values
    return X, y

st.subheader("📊 Model Evaluation on Test Set")

X_test, y_test = load_test_data()

if model_choice == "KNN":
    model = knn_model
elif model_choice == "Naive Bayes":
    model = nb_model
else:
    model = dt_model

y_pred = model.predict(X_test)
y_test_decoded = label_encoder.inverse_transform(y_test)
y_pred_decoded = label_encoder.inverse_transform(y_pred)

labels = label_encoder.classes_
cm = confusion_matrix(y_test_decoded, y_pred_decoded)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
st.pyplot(fig)

report = classification_report(y_test_decoded, y_pred_decoded, target_names=labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.style.format({'precision': "{:.2f}", 'recall': "{:.2f}", 'f1-score': "{:.2f}"}))
