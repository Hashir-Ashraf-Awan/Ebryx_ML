from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and scaler
knn_model = joblib.load("knn_model.pkl")
dt_model = joblib.load("dt_model.pkl")
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load predictions from CSV
pred_df = pd.read_csv("model_predictions.csv")

def get_metrics(model_name):
    model_df = pred_df[pred_df['Model'] == model_name]
    y_true = model_df['Actual']
    y_pred = model_df['Predicted']
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, output_dict=True)
    return cm, report

from pydantic import BaseModel
from typing import List

class PatientFeatures(BaseModel):
    features: List[float]

@app.post("/predict/")
def predict(data: PatientFeatures):
    input_data = np.array(data.features).reshape(1, -1)
    scaled_data = scaler.transform(input_data)

    pred_knn = knn_model.predict(scaled_data)[0]
    pred_dt = dt_model.predict(scaled_data)[0]
    pred_rf = rf_model.predict(scaled_data)[0]

    return {
        "knn": int(pred_knn),
        "decision_tree": int(pred_dt),
        "random_forest": int(pred_rf)
    }

@app.get("/metrics/{model_name}")
def metrics(model_name: str):
    model_name_map = {
        "knn": "KNN",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest"
    }
    if model_name not in model_name_map:
        return {"error": "Invalid model name"}
    cm, report = get_metrics(model_name_map[model_name])
    return {"confusion_matrix": cm, "classification_report": report}
