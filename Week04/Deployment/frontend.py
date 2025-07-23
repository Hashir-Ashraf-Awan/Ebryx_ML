import streamlit as st
import requests

st.title("🩺 Lung Cancer Prediction App")
st.markdown("Enter patient data to predict lung cancer likelihood.")

# Feature input
user_input = []

# 1. AGE
age = st.number_input("AGE", min_value=0.0, step=1.0)
user_input.append(age)

# 2. GENDER (radio)
gender = st.radio("GENDER", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
user_input.append(gender)

# 3. SMOKING_STATUS
smoking_status = st.number_input("SMOKING_STATUS", min_value=0.0, step=1.0)
user_input.append(smoking_status)

# 4–14: remaining features
for feature in [
    'YELLOW_SKIN', 'ANXIETY', 'PEER_PRESSURE', 'FATIGUE', 'ALLERGY',
    'WHEEZING', 'ALCOHOL_CONSUMPTION', 'WEEKLY_GLASSES_OF_ALCOHOL',
    'COUGHING', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN'
]:
    val = st.number_input(feature, min_value=0.0, step=1.0)
    user_input.append(val)

# Submit
if st.button("Predict"):
    payload = {"features": user_input}
    response = requests.post("http://127.0.0.1:8000/predict/", json=payload)

    if response.status_code == 200:
        preds = response.json()
        st.subheader("Prediction Results")
        st.write(f"KNN Prediction: {'Cancer' if preds['knn'] == 2 else 'Not Cancer'}")
        st.write(f"Decision Tree Prediction: {'Cancer' if preds['decision_tree'] == 2 else 'Not Cancer'}")
        st.write(f"Random Forest Prediction: {'Cancer' if preds['random_forest'] == 2 else 'Not Cancer'}")
    else:
        st.error("Failed to get prediction from FastAPI server.")
