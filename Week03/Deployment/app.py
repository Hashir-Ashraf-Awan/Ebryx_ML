from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load scaler and models
scaler = joblib.load("scaler.pkl")
models = {
    "KNN": joblib.load("knn_model.pkl"),
    "Naive Bayes": joblib.load("nb_model.pkl"),
    "Decision Tree": joblib.load("dt_model.pkl")
}

# Label mapping (edit as per your cluster labeling)
label_map = {
    0: "Frugal Elders",
    1: "Impulsive Youth",
    2: "Conservative Adults",
    3: "Enthusiastic Shoppers"
}

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
        # Prepare input
        input_data = np.array([[Age, Income, Gender, Spending]])
        scaled_input = scaler.transform(input_data)

        # Get selected model
        model = models.get(model_name)
        if model is None:
            raise ValueError("Invalid model name selected.")

        # Predict
        pred = model.predict(scaled_input)[0]
        cluster = label_map.get(pred, f"Cluster {pred}")

        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": cluster,
            "selected_model": model_name,
            "models": models.keys()
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": f"❌ Error: {str(e)}",
            "models": models.keys()
        })
