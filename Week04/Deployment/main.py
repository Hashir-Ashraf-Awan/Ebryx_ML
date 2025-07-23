from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained models and scaler
knn_model = joblib.load("knn_model.pkl")
dt_model = joblib.load("dt_model.pkl")
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

class PatientFeatures(BaseModel):
    features: list  # List of 7 values

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
