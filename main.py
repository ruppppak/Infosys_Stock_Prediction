from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ==========================
# Load model and scaler
# ==========================
model = load_model("lstm_close_model.h5", compile=False)
scaler = joblib.load("lstm_scaler.pkl")

# ==========================
# Helper: prepare sequence
# ==========================
def prepare_input_sequence(recent_prices, n_steps=10):
    arr = np.array(recent_prices).reshape(-1, 1)
    scaled = scaler.transform(arr)
    X_input = np.array([scaled[-n_steps:, 0]])  # take last n_steps
    return X_input.reshape(1, n_steps, 1)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, prices: str = Form(...)):
    """
    Input example: "100,101,102,103,104,105,106,107,108,109"
    (last 10 closing prices)
    """
    try:
        price_list = [float(p.strip()) for p in prices.split(",")]
        if len(price_list) < 10:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "prediction": "❌ Please enter at least 10 comma-separated prices."
            })

        X_input = prepare_input_sequence(price_list)
        pred_scaled = model.predict(X_input)
        pred_price = scaler.inverse_transform(pred_scaled)[0][0]

        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": f"📈 Predicted next close price: {pred_price:.2f}"
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": f"Error: {e}"
        })

# Run with: uvicorn main:app --reload
