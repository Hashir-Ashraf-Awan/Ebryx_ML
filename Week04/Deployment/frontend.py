import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🩺 Lung Cancer Prediction App")
st.markdown("Enter patient data to predict lung cancer likelihood.")

user_input = []

age = st.number_input("AGE", min_value=0.0, step=1.0)
user_input.append(age)

gender = st.radio("GENDER", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
user_input.append(gender)

smoking_status = st.number_input("SMOKING_STATUS", min_value=0.0, step=1.0)
user_input.append(smoking_status)

for feature in [
    'YELLOW_SKIN', 'ANXIETY', 'PEER_PRESSURE', 'FATIGUE', 'ALLERGY',
    'WHEEZING', 'ALCOHOL_CONSUMPTION', 'WEEKLY_GLASSES_OF_ALCOHOL',
    'COUGHING', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN'
]:
    val = st.number_input(feature, min_value=0.0, step=1.0)
    user_input.append(val)

if st.button("Predict"):
    response = requests.post("http://127.0.0.1:8000/predict/", json={"features": user_input})
    if response.status_code == 200:
        preds = response.json()
        st.subheader("Prediction Results")
        st.write(f"✅ KNN Prediction: {'Cancer' if preds['knn'] == 2 else 'Not Cancer'}")
        st.write(f"✅ Decision Tree Prediction: {'Cancer' if preds['decision_tree'] == 2 else 'Not Cancer'}")
        st.write(f"✅ Random Forest Prediction: {'Cancer' if preds['random_forest'] == 2 else 'Not Cancer'}")
    else:
        st.error("❌ Failed to get prediction.")

# Show metrics
st.header("📊 Model Evaluation Metrics")

def show_metrics(model_name):
    response = requests.get(f"http://127.0.0.1:8000/metrics/{model_name}")
    if response.status_code == 200:
        data = response.json()
        cm = data['confusion_matrix']
        report = data['classification_report']

        st.subheader(f"{model_name.upper()} Confusion Matrix")
        df_cm = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
        fig, ax = plt.subplots()
        sns.heatmap(df_cm, annot=True, cmap="Blues", fmt="d", ax=ax)
        st.pyplot(fig)

        st.subheader(f"{model_name.upper()} Classification Report")
        st.json(report)
    else:
        st.error("❌ Error loading metrics.")

model_to_check = st.selectbox("Select a model to view metrics", ["knn", "decision_tree", "random_forest"])
if st.button("Show Metrics"):
    show_metrics(model_to_check)
