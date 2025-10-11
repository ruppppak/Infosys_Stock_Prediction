from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# --- App Initialization ---
app = FastAPI()
templates = Jinja2Templates(directory=".")

# --- Load Models and Scalers ---
try:
    lstm_model = load_model("lstm_close_model.h5", compile=False)
    lstm_scaler = joblib.load("lstm_scaler.pkl")
    gru_model = load_model("gru_close_model.h5", compile=False)
    gru_scaler = joblib.load("gru_scaler.pkl")
    print("✅ Models and scalers loaded successfully!")
    print(f"   LSTM Scaler range: [{lstm_scaler.data_min_[0]:.2f}, {lstm_scaler.data_max_[0]:.2f}]")
    print(f"   GRU Scaler range: [{gru_scaler.data_min_[0]:.2f}, {gru_scaler.data_max_[0]:.2f}]")
except Exception as e:
    print(f"❌ Fatal Error: Could not load models or scalers. {e}")
    exit()


# --- Helper Function for Forecasting ---
def generate_forecast(model, scaler, initial_sequence: list, days_to_predict: int = 5) -> list:
    """
    Generates a multi-day forecast using a walk-forward prediction method.

    Args:
        model: The trained Keras model.
        scaler: The scaler used for the model.
        initial_sequence: The starting list of 10 prices.
        days_to_predict: The number of future days to forecast.

    Returns:
        A list of forecasted prices.
    """
    future_preds = []
    current_sequence = list(initial_sequence)

    for _ in range(days_to_predict):
        # Take last 10 values from current sequence
        seq_arr = np.array(current_sequence[-10:]).reshape(-1, 1)

        # Scale the sequence
        scaled_seq = scaler.transform(seq_arr).reshape(1, 10, 1)

        # Predict the next value (scaled)
        next_pred_scaled = model.predict(scaled_seq, verbose=0)

        # Inverse transform to get the actual price
        next_pred = scaler.inverse_transform(next_pred_scaled)[0, 0]

        # Store prediction
        future_preds.append(float(next_pred))

        # Append the prediction to the sequence for the next iteration
        current_sequence.append(float(next_pred))

    return future_preds


def validate_input_prices(prices: list, scaler) -> tuple:
    """
    Validate if input prices are within reasonable range of the model's training data.
    Returns: (is_valid, warning_message)
    """
    scaler_min = scaler.data_min_[0]
    scaler_max = scaler.data_max_[0]

    price_min = min(prices)
    price_max = max(prices)
    price_mean = np.mean(prices)

    # Check if prices are way outside training range (more than 50% deviation)
    range_buffer = (scaler_max - scaler_min) * 0.5

    if price_max < (scaler_min - range_buffer) or price_min > (scaler_max + range_buffer):
        return False, f"Input prices (₹{price_min:.2f} - ₹{price_max:.2f}) are far outside the model's training range (₹{scaler_min:.2f} - ₹{scaler_max:.2f}). Predictions may be unreliable."

    if price_mean < scaler_min * 0.5:
        return False, f"Input prices are too low (mean: ₹{price_mean:.2f}). Model was trained on prices ranging from ₹{scaler_min:.2f} to ₹{scaler_max:.2f}. Please use prices in a similar range."

    if price_mean > scaler_max * 2:
        return False, f"Input prices are too high (mean: ₹{price_mean:.2f}). Model was trained on prices ranging from ₹{scaler_min:.2f} to ₹{scaler_max:.2f}. Please use prices in a similar range."

    return True, None


# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serves the main index.html page with default context."""
    context = {
        "request": request,
        "lstm_prediction": "",
        "gru_prediction": "",
        "historical_data": [],
        "lstm_forecast_data": [],
        "gru_forecast_data": [],
        "lstm_trend": "",
        "lstm_change_percent": "",
        "gru_trend": "",
        "gru_change_percent": "",
        "error": None,
        "warning": None
    }
    return templates.TemplateResponse("index.html", context)


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, prices: str = Form(...)):
    """Handles prediction logic and returns results with trend analysis."""
    try:
        # 1. Parse and validate input
        input_prices = [float(x.strip()) for x in prices.split(",")]

        if len(input_prices) != 10:
            raise ValueError("Please provide exactly 10 historical prices.")

        # Check for valid price values
        if any(price <= 0 for price in input_prices):
            raise ValueError("All prices must be positive numbers.")

        # Validate against model training range
        is_valid, warning_msg = validate_input_prices(input_prices, lstm_scaler)

        if not is_valid:
            raise ValueError(warning_msg)

        last_price = input_prices[-1]
        arr = np.array(input_prices).reshape(-1, 1)

        # 2. Get single-day predictions (Day +1)
        lstm_input = lstm_scaler.transform(arr).reshape(1, 10, 1)
        lstm_pred_raw = lstm_model.predict(lstm_input, verbose=0)
        lstm_pred = float(lstm_scaler.inverse_transform(lstm_pred_raw)[0, 0])

        gru_input = gru_scaler.transform(arr).reshape(1, 10, 1)
        gru_pred_raw = gru_model.predict(gru_input, verbose=0)
        gru_pred = float(gru_scaler.inverse_transform(gru_pred_raw)[0, 0])

        # 3. Generate 5-day forecasts using the helper function
        lstm_future_preds = generate_forecast(lstm_model, lstm_scaler, input_prices, days_to_predict=5)
        gru_future_preds = generate_forecast(gru_model, gru_scaler, input_prices, days_to_predict=5)

        # 4. Calculate trend and percentage change
        lstm_change = lstm_pred - last_price
        gru_change = gru_pred - last_price

        lstm_change_pct = (lstm_change / last_price) * 100
        gru_change_pct = (gru_change / last_price) * 100

        # 5. Sanity check on predictions
        # If prediction change is more than 50%, it's likely unreliable
        warning = None
        if abs(lstm_change_pct) > 50 or abs(gru_change_pct) > 50:
            warning = f"⚠️ Warning: Large prediction changes detected. This may indicate the input prices are outside the model's reliable prediction range. For best results, use prices similar to the training data (₹{lstm_scaler.data_min_[0]:.2f} - ₹{lstm_scaler.data_max_[0]:.2f})."

        # 6. Prepare context for template
        context = {
            "request": request,
            "lstm_prediction": f"{lstm_pred:.2f}",
            "gru_prediction": f"{gru_pred:.2f}",
            "historical_data": [round(float(p), 2) for p in input_prices],

            # Forecast data - ensure all values are Python floats for JSON serialization
            "lstm_forecast_data": [round(float(p), 2) for p in lstm_future_preds],
            "gru_forecast_data": [round(float(p), 2) for p in gru_future_preds],

            # Trend analysis data
            "lstm_trend": "Increase" if lstm_change > 0 else "Decrease",
            "lstm_change_percent": f"{lstm_change_pct:+.2f}%",
            "gru_trend": "Increase" if gru_change > 0 else "Decrease",
            "gru_change_percent": f"{gru_change_pct:+.2f}%",

            "error": None,
            "warning": warning
        }

        # Debug logging
        print(f"\n📊 Prediction Results:")
        print(f"   Input range: ₹{min(input_prices):.2f} - ₹{max(input_prices):.2f}")
        print(f"   LSTM: ₹{lstm_pred:.2f} ({lstm_change_pct:+.2f}%)")
        print(f"   GRU:  ₹{gru_pred:.2f} ({gru_change_pct:+.2f}%)")
        print(f"   LSTM Forecast: {[f'{p:.2f}' for p in lstm_future_preds]}")
        print(f"   GRU Forecast:  {[f'{p:.2f}' for p in gru_future_preds]}")
        if warning:
            print(f"   ⚠️  {warning}")

        return templates.TemplateResponse("index.html", context)

    except ValueError as ve:
        # Handle validation errors
        context = {
            "request": request,
            "lstm_prediction": "",
            "gru_prediction": "",
            "historical_data": [],
            "lstm_forecast_data": [],
            "gru_forecast_data": [],
            "lstm_trend": "",
            "lstm_change_percent": "",
            "gru_trend": "",
            "gru_change_percent": "",
            "error": str(ve),
            "warning": None
        }
        return templates.TemplateResponse("index.html", context)

    except Exception as e:
        # Handle unexpected errors
        print(f"❌ Error during prediction: {e}")
        context = {
            "request": request,
            "lstm_prediction": "",
            "gru_prediction": "",
            "historical_data": [],
            "lstm_forecast_data": [],
            "gru_forecast_data": [],
            "lstm_trend": "",
            "lstm_change_percent": "",
            "gru_trend": "",
            "gru_change_percent": "",
            "error": f"An error occurred during prediction: {str(e)}",
            "warning": None
        }
        return templates.TemplateResponse("index.html", context)


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 70)
    print("🚀 Starting StonksAI Stock Prediction Server")
    print("=" * 70)
    print(f"   LSTM Model: Loaded")
    print(f"   GRU Model: Loaded")
    print(f"   Server: http://localhost:8000")
    print("=" * 70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)