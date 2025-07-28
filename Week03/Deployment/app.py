from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load scaler and models
scaler = joblib.load("scaler.pkl")
models = {
    "KNN": joblib.load("knn_model.pkl"),
    "Naive Bayes": joblib.load("nb_model.pkl"),
    "Decision Tree": joblib.load("dt_model.pkl")
}

label_map = {
    0: "Frugal Elders",
    1: "Impulsive Youth",
    2: "Conservative Adults",
    3: "Enthusiastic Shoppers"
}

PREDICTION_LOG = "predictions_log.csv"

# Helper to save prediction
def save_prediction(actual, predicted, model_name):
    new_row = pd.DataFrame([{
        "Actual": actual,
        "Predicted": predicted,
        "Model": model_name
    }])
    if os.path.exists(PREDICTION_LOG):
        df = pd.read_csv(PREDICTION_LOG)
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row
    df.to_csv(PREDICTION_LOG, index=False)

# Generate confusion matrix image and classification report text
def generate_cm_and_report(model_name):
    if not os.path.exists(PREDICTION_LOG):
        return None, None

    df = pd.read_csv(PREDICTION_LOG)
    model_df = df[df["Model"] == model_name]

    if len(model_df) < 2:
        return None, None  # not enough data

    y_true = model_df["Actual"]
    y_pred = model_df["Predicted"]

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} Confusion Matrix")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    cm_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Classification report as text
    report = classification_report(y_true, y_pred)

    return cm_base64, report

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "models": models.keys()
    })

@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    Age: float = Form(...),
    Income: float = Form(...),
    Gender: int = Form(...),
    Spending: float = Form(...),
    model_name: str = Form(...)
):
    try:
        input_data = np.array([[Gender, Age, Income, Spending]])
        scaled_input = scaler.transform(input_data)
        model = models.get(model_name)
        if model is None:
            raise ValueError("Invalid model selected.")

        pred = model.predict(scaled_input)[0]
        cluster = label_map.get(pred, str(pred))

        # Save prediction (actual = predicted for now)
        save_prediction(pred, pred, model_name)

        # Generate CM and report
        cm_img, report_txt = generate_cm_and_report(model_name)

        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": cluster,
            "selected_model": model_name,
            "models": models.keys(),
            "cm_img": cm_img,
            "report": report_txt
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": f"❌ Error: {str(e)}",
            "models": models.keys()
        })
