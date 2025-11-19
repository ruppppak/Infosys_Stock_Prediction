from fastapi import FastAPI, Request, Form, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.middleware.sessions import SessionMiddleware
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import sqlite3
import hashlib
import secrets
from datetime import datetime
import os

# --- App Initialization ---
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=secrets.token_hex(32))
templates = Jinja2Templates(directory=".")

# --- Database Setup ---
DATABASE_NAME = "stocksai_users.db"


def init_database():
    """Initialize SQLite database with users table"""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    """)

    # Create predictions history table (optional - for future use)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            input_prices TEXT,
            lstm_prediction REAL,
            gru_prediction REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    conn.commit()
    conn.close()
    print("✅ Database initialized successfully!")


# Initialize database on startup
init_database()

# --- Load Models and Scalers ---
try:
    lstm_model = load_model("lstm_close_model.h5", compile=False)
    lstm_scaler = joblib.load("lstm_scaler.pkl")
    gru_model = load_model("gru_close_model.h5", compile=False)
    gru_scaler = joblib.load("gru_scaler.pkl")
    print("✅ Models and scalers loaded successfully!")
except Exception as e:
    print(f"❌ Fatal Error: Could not load models or scalers. {e}")
    # Create dummy models for testing without actual model files
    lstm_model = None
    lstm_scaler = None
    gru_model = None
    gru_scaler = None


# --- Authentication Helper Functions ---
def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return hash_password(plain_password) == hashed_password


def create_user(username: str, email: str, password: str) -> tuple:
    """Create a new user in the database"""
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()

        password_hash = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, password_hash)
        )

        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return True, user_id
    except sqlite3.IntegrityError:
        return False, "Username or email already exists"
    except Exception as e:
        return False, str(e)


def authenticate_user(username: str, password: str) -> tuple:
    """Authenticate user credentials"""
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            (username,)
        )
        user = cursor.fetchone()

        if user and verify_password(password, user[2]):
            # Update last login
            cursor.execute(
                "UPDATE users SET last_login = ? WHERE id = ?",
                (datetime.now(), user[0])
            )
            conn.commit()
            conn.close()
            return True, {"id": user[0], "username": user[1]}

        conn.close()
        return False, "Invalid username or password"
    except Exception as e:
        return False, str(e)


def get_current_user(request: Request):
    """Get current logged-in user from session"""
    return request.session.get("user")


def require_login(request: Request):
    """Dependency to require login"""
    user = get_current_user(request)
    if not user:
        return None
    return user


# --- Helper Function for Forecasting ---
def generate_forecast(model, scaler, initial_sequence: list, days_to_predict: int = 5) -> list:
    """Generates a multi-day forecast using a walk-forward prediction method."""
    if model is None or scaler is None:
        # Return dummy data for testing
        base = initial_sequence[-1]
        return [base * (1 + 0.01 * i) for i in range(1, days_to_predict + 1)]

    future_preds = []
    current_sequence = list(initial_sequence)

    for _ in range(days_to_predict):
        seq_arr = np.array(current_sequence[-10:]).reshape(-1, 1)
        scaled_seq = scaler.transform(seq_arr).reshape(1, 10, 1)
        next_pred_scaled = model.predict(scaled_seq, verbose=0)
        next_pred = scaler.inverse_transform(next_pred_scaled)[0, 0]
        future_preds.append(float(next_pred))
        current_sequence.append(float(next_pred))

    return future_preds


def validate_input_prices(prices: list, scaler) -> tuple:
    """Validate if input prices are within reasonable range"""
    if scaler is None:
        return True, None

    scaler_min = scaler.data_min_[0]
    scaler_max = scaler.data_max_[0]
    price_min = min(prices)
    price_max = max(prices)
    price_mean = np.mean(prices)
    range_buffer = (scaler_max - scaler_min) * 0.5

    if price_max < (scaler_min - range_buffer) or price_min > (scaler_max + range_buffer):
        return False, f"Input prices are outside the model's training range."

    return True, None


# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Redirect to login if not authenticated, otherwise show dashboard"""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    context = {
        "request": request,
        "user": user,
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


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Show login page"""
    user = get_current_user(request)
    if user:
        return RedirectResponse(url="/", status_code=303)

    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/login", response_class=HTMLResponse)
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Handle login form submission"""
    success, result = authenticate_user(username, password)

    if success:
        request.session["user"] = result
        return RedirectResponse(url="/", status_code=303)
    else:
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": result
        })


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """Show registration page"""
    user = get_current_user(request)
    if user:
        return RedirectResponse(url="/", status_code=303)

    return templates.TemplateResponse("register.html", {"request": request, "error": None})


@app.post("/register", response_class=HTMLResponse)
async def register(
        request: Request,
        username: str = Form(...),
        email: str = Form(...),
        password: str = Form(...),
        confirm_password: str = Form(...)
):
    """Handle registration form submission"""
    # Validate input
    if len(username) < 3:
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error": "Username must be at least 3 characters long"
        })

    if len(password) < 6:
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error": "Password must be at least 6 characters long"
        })

    if password != confirm_password:
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error": "Passwords do not match"
        })

    # Create user
    success, result = create_user(username, email, password)

    if success:
        # Auto-login after registration
        request.session["user"] = {"id": result, "username": username}
        return RedirectResponse(url="/", status_code=303)
    else:
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error": result
        })


@app.get("/logout")
async def logout(request: Request):
    """Handle logout"""
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, prices: str = Form(...)):
    """Handle prediction logic"""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    try:
        input_prices = [float(x.strip()) for x in prices.split(",")]

        if len(input_prices) != 10:
            raise ValueError("Please provide exactly 10 historical prices.")

        if any(price <= 0 for price in input_prices):
            raise ValueError("All prices must be positive numbers.")

        is_valid, warning_msg = validate_input_prices(input_prices, lstm_scaler)
        if not is_valid:
            raise ValueError(warning_msg)

        last_price = input_prices[-1]

        # Generate predictions (with dummy data if models not loaded)
        if lstm_model and gru_model:
            arr = np.array(input_prices).reshape(-1, 1)
            lstm_input = lstm_scaler.transform(arr).reshape(1, 10, 1)
            lstm_pred_raw = lstm_model.predict(lstm_input, verbose=0)
            lstm_pred = float(lstm_scaler.inverse_transform(lstm_pred_raw)[0, 0])

            gru_input = gru_scaler.transform(arr).reshape(1, 10, 1)
            gru_pred_raw = gru_model.predict(gru_input, verbose=0)
            gru_pred = float(gru_scaler.inverse_transform(gru_pred_raw)[0, 0])
        else:
            # Dummy predictions for testing
            lstm_pred = last_price * 1.025
            gru_pred = last_price * 1.023

        lstm_future_preds = generate_forecast(lstm_model, lstm_scaler, input_prices, days_to_predict=5)
        gru_future_preds = generate_forecast(gru_model, gru_scaler, input_prices, days_to_predict=5)

        lstm_change = lstm_pred - last_price
        gru_change = gru_pred - last_price
        lstm_change_pct = (lstm_change / last_price) * 100
        gru_change_pct = (gru_change / last_price) * 100

        warning = None
        if abs(lstm_change_pct) > 50 or abs(gru_change_pct) > 50:
            warning = "⚠️ Warning: Large prediction changes detected."

        # Save prediction to database
        try:
            conn = sqlite3.connect(DATABASE_NAME)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO predictions (user_id, input_prices, lstm_prediction, gru_prediction) VALUES (?, ?, ?, ?)",
                (user["id"], ",".join(map(str, input_prices)), lstm_pred, gru_pred)
            )
            conn.commit()
            conn.close()
        except:
            pass  # Ignore if saving fails

        context = {
            "request": request,
            "user": user,
            "lstm_prediction": f"{lstm_pred:.2f}",
            "gru_prediction": f"{gru_pred:.2f}",
            "historical_data": [round(float(p), 2) for p in input_prices],
            "lstm_forecast_data": [round(float(p), 2) for p in lstm_future_preds],
            "gru_forecast_data": [round(float(p), 2) for p in gru_future_preds],
            "lstm_trend": "Increase" if lstm_change > 0 else "Decrease",
            "lstm_change_percent": f"{lstm_change_pct:+.2f}%",
            "gru_trend": "Increase" if gru_change > 0 else "Decrease",
            "gru_change_percent": f"{gru_change_pct:+.2f}%",
            "error": None,
            "warning": warning
        }

        return templates.TemplateResponse("index.html", context)

    except ValueError as ve:
        context = {
            "request": request,
            "user": user,
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
        print(f"❌ Error during prediction: {e}")
        context = {
            "request": request,
            "user": user,
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
    print("🚀 Starting StocksAI Stock Prediction Server with Authentication")
    print("=" * 70)
    print(f"   Database: {DATABASE_NAME}")
    print(f"   Server: http://localhost:8000")
    print("=" * 70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)