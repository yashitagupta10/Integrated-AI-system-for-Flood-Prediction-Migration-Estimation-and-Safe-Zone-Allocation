"""
hybrid_model.py — Hybrid ARIMA + LSTM Flood Prediction Model
============================================================
Architecture:
  1. ARIMA captures linear trend and seasonality in rainfall time-series
  2. LSTM learns residual non-linear patterns ARIMA misses
  3. Final prediction = ARIMA output + LSTM(residuals)

This is a DYNAMIC model — it retrains/fine-tunes whenever new real-time
data arrives, so it improves over time rather than being static.

Requirements:
    pip install tensorflow numpy pandas statsmodels scikit-learn joblib
"""

import os
import numpy as np
import pandas as pd
import warnings
import joblib
import json
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Optional imports with graceful fallback ──────────────────────────────────
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    print("[hybrid_model] WARNING: statsmodels not installed. ARIMA disabled.")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[hybrid_model] WARNING: TensorFlow not installed. LSTM disabled.")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_DIR = Path("models_saved/hybrid")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 14   # 14 days lookback for LSTM
ARIMA_ORDER = (2, 1, 2)  # (p, d, q) — auto-selected if ADF test fails
LSTM_UNITS = 64
DROPOUT_RATE = 0.2
EPOCHS = 50
BATCH_SIZE = 16


# ═══════════════════════════════════════════════════════════════════════════
# ARIMA Component
# ═══════════════════════════════════════════════════════════════════════════
class ARIMAComponent:
    def __init__(self, order=ARIMA_ORDER):
        self.order = order
        self.model_fit = None
        self.last_train_data = None

    def _select_order(self, series):
        """Auto-select differencing order via ADF test."""
        if not ARIMA_AVAILABLE:
            return self.order
        try:
            result = adfuller(series, autolag='AIC')
            d = 0 if result[1] < 0.05 else 1
            return (self.order[0], d, self.order[2])
        except Exception:
            return self.order

    def fit(self, series: np.ndarray):
        """Fit ARIMA to a rainfall time-series."""
        if not ARIMA_AVAILABLE:
            self.last_train_data = series
            return self
        order = self._select_order(series)
        try:
            model = ARIMA(series, order=order)
            self.model_fit = model.fit()
            self.last_train_data = series
            print(f"  [ARIMA] Fitted with order={order}, AIC={self.model_fit.aic:.2f}")
        except Exception as e:
            print(f"  [ARIMA] Fit failed ({e}), using order={self.order}")
            try:
                model = ARIMA(series, order=(1, 1, 1))
                self.model_fit = model.fit()
                self.last_train_data = series
            except Exception as e2:
                print(f"  [ARIMA] Fallback also failed: {e2}")
        return self

    def predict(self, steps: int = 7) -> np.ndarray:
        """Forecast next `steps` values."""
        if not ARIMA_AVAILABLE or self.model_fit is None:
            # Naive forecast: repeat last value with slight decay
            last = self.last_train_data[-1] if self.last_train_data is not None else 0
            return np.array([last * (0.9 ** i) for i in range(steps)])
        try:
            forecast = self.model_fit.forecast(steps=steps)
            return np.array(forecast)
        except Exception:
            last = self.last_train_data[-1]
            return np.full(steps, last)

    def get_residuals(self) -> np.ndarray:
        """Return in-sample residuals for LSTM to learn from."""
        if not ARIMA_AVAILABLE or self.model_fit is None:
            return np.zeros(len(self.last_train_data) if self.last_train_data is not None else 10)
        return np.array(self.model_fit.resid)


# ═══════════════════════════════════════════════════════════════════════════
# LSTM Component
# ═══════════════════════════════════════════════════════════════════════════
class LSTMComponent:
    def __init__(self, seq_len=SEQUENCE_LENGTH, units=LSTM_UNITS):
        self.seq_len = seq_len
        self.units = units
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False

    def _build_model(self, n_features=1):
        if not TF_AVAILABLE:
            return None
        model = Sequential([
            Input(shape=(self.seq_len, n_features)),
            LSTM(self.units, return_sequences=True),
            Dropout(DROPOUT_RATE),
            LSTM(self.units // 2),
            Dropout(DROPOUT_RATE),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='huber')
        return model

    def _make_sequences(self, data: np.ndarray):
        X, y = [], []
        for i in range(self.seq_len, len(data)):
            X.append(data[i - self.seq_len:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def fit(self, residuals: np.ndarray):
        """Train LSTM on ARIMA residuals."""
        if not TF_AVAILABLE:
            print("  [LSTM] TensorFlow not available, skipping.")
            self.is_fitted = False
            return self

        if len(residuals) < self.seq_len + 5:
            print(f"  [LSTM] Not enough data ({len(residuals)} samples). Need {self.seq_len + 5}+")
            self.is_fitted = False
            return self

        # Scale
        scaled = self.scaler.fit_transform(residuals.reshape(-1, 1))
        X, y = self._make_sequences(scaled)

        if len(X) < 5:
            self.is_fitted = False
            return self

        self.model = self._build_model(n_features=1)
        es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

        split = max(1, int(len(X) * 0.85))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        val_data = (X_val, y_val) if len(X_val) > 0 else None

        self.model.fit(
            X_train, y_train,
            validation_data=val_data,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[es] if val_data else [],
            verbose=0,
        )
        self.is_fitted = True
        self._last_sequence = scaled[-self.seq_len:]
        print(f"  [LSTM] Trained on {len(X_train)} sequences.")
        return self

    def predict(self, steps: int = 7) -> np.ndarray:
        """Predict residual corrections for next `steps` days."""
        if not TF_AVAILABLE or not self.is_fitted or self.model is None:
            return np.zeros(steps)

        predictions = []
        seq = self._last_sequence.copy()

        for _ in range(steps):
            x_input = seq.reshape(1, self.seq_len, 1)
            pred_scaled = self.model.predict(x_input, verbose=0)[0, 0]
            predictions.append(pred_scaled)
            seq = np.append(seq[1:], [[pred_scaled]], axis=0)

        # Inverse transform
        preds_inv = self.scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()
        return preds_inv

    def save(self, path: str):
        if self.model:
            self.model.save(path + "_lstm.keras")
            joblib.dump(self.scaler, path + "_scaler.pkl")

    def load(self, path: str):
        lstm_path = path + "_lstm.keras"
        scaler_path = path + "_scaler.pkl"
        if TF_AVAILABLE and os.path.exists(lstm_path):
            self.model = load_model(lstm_path)
            self.scaler = joblib.load(scaler_path)
            self.is_fitted = True
        return self


# ═══════════════════════════════════════════════════════════════════════════
# Hybrid ARIMA + LSTM Model
# ═══════════════════════════════════════════════════════════════════════════
class HybridARIMALSTM:
    """
    Main hybrid model. Usage:
        model = HybridARIMALSTM(district="Pune")
        model.train(historical_rainfall_series)
        result = model.predict_flood_risk(current_rainfall, affected_area, census_data)
    """

    def __init__(self, district: str):
        self.district = district
        self.arima = ARIMAComponent()
        self.lstm = LSTMComponent()
        self.is_trained = False
        self.training_history = []
        self.model_path = str(MODEL_DIR / district.replace(" ", "_"))

    def train(self, rainfall_series: np.ndarray, force_retrain: bool = False):
        """
        Train the hybrid model on historical rainfall data.
        rainfall_series: 1D numpy array of daily rainfall (mm)
        """
        if len(rainfall_series) < 30:
            print(f"[Hybrid] Need at least 30 data points. Got {len(rainfall_series)}.")
            return self

        if not force_retrain and self._load_if_exists():
            print(f"[Hybrid:{self.district}] Loaded saved model.")
            return self

        print(f"\n[Hybrid:{self.district}] Training on {len(rainfall_series)} data points...")

        # Step 1: Fit ARIMA
        self.arima.fit(rainfall_series)

        # Step 2: Get residuals → train LSTM
        residuals = self.arima.get_residuals()
        self.lstm.fit(residuals)

        self.is_trained = True
        self.training_history.append({
            "timestamp": datetime.now().isoformat(),
            "data_points": len(rainfall_series),
            "arima_available": ARIMA_AVAILABLE,
            "lstm_available": TF_AVAILABLE and self.lstm.is_fitted,
        })

        self._save()
        print(f"[Hybrid:{self.district}] Training complete.")
        return self

    def forecast_rainfall(self, steps: int = 7) -> np.ndarray:
        """
        Forecast next `steps` days of rainfall.
        Returns combined ARIMA + LSTM forecast.
        """
        arima_forecast = self.arima.predict(steps)
        lstm_correction = self.lstm.predict(steps)

        combined = arima_forecast + lstm_correction
        combined = np.clip(combined, 0, None)  # rainfall can't be negative
        return combined

    def predict_flood_risk(
        self,
        current_rainfall: float,
        affected_area_km2: float,
        population: int = 100000,
        elevation_risk: float = 0.5,
    ) -> dict:
        """
        Main prediction function called by the Flask API.
        Returns full risk assessment dict compatible with existing app.py response format.
        """
        # Get 7-day forecast
        forecast = self.forecast_rainfall(7) if self.is_trained else np.zeros(7)
        total_forecast_7d = float(np.sum(forecast))
        peak_forecast = float(np.max(forecast)) if len(forecast) > 0 else 0.0

        # ── Risk scoring (0–100) ────────────────────────────────────────────
        # Rainfall component (0–40 pts)
        if current_rainfall < 50:
            rain_score = current_rainfall * 0.4
        elif current_rainfall < 150:
            rain_score = 20 + (current_rainfall - 50) * 0.15
        elif current_rainfall < 300:
            rain_score = 35 + (current_rainfall - 150) * 0.033
        else:
            rain_score = 40

        # Forecast trend component (0–20 pts)
        forecast_score = min(total_forecast_7d / 500 * 20, 20)

        # Area component (0–20 pts)
        area_score = min(affected_area_km2 / 10000 * 20, 20)

        # Elevation risk (0–20 pts)
        elev_score = elevation_risk * 20

        total_score = rain_score + forecast_score + area_score + elev_score
        total_score = min(total_score, 100)

        # ── Risk level ──────────────────────────────────────────────────────
        if total_score < 25:
            risk_level = "Low"
            risk_color = "#22c55e"
        elif total_score < 50:
            risk_level = "Moderate"
            risk_color = "#f59e0b"
        elif total_score < 75:
            risk_level = "High"
            risk_color = "#ef4444"
        else:
            risk_level = "Extreme"
            risk_color = "#7c3aed"

        # ── Estimated affected population ───────────────────────────────────
        impact_fraction = total_score / 100 * 0.4
        affected_population = int(population * impact_fraction)

        # ── Shelter capacity ────────────────────────────────────────────────
        shelters_needed = max(1, affected_population // 500)

        return {
            "district": self.district,
            "risk_score": round(total_score, 1),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "current_rainfall_mm": round(current_rainfall, 2),
            "affected_area_km2": round(affected_area_km2, 2),
            "forecast_7day_mm": [round(f, 2) for f in forecast.tolist()],
            "total_forecast_7day_mm": round(total_forecast_7d, 2),
            "peak_forecast_mm": round(peak_forecast, 2),
            "affected_population_estimate": affected_population,
            "shelters_needed": shelters_needed,
            "model_type": "Hybrid ARIMA+LSTM",
            "arima_component": ARIMA_AVAILABLE,
            "lstm_component": TF_AVAILABLE and self.lstm.is_fitted,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "recommendations": self._get_recommendations(risk_level, current_rainfall),
        }

    def _get_recommendations(self, risk_level: str, rainfall: float) -> list:
        base = {
            "Low": [
                "Monitor weather forecasts regularly.",
                "Check drainage systems are clear.",
                "Keep emergency contacts handy.",
            ],
            "Moderate": [
                "Alert local emergency services.",
                "Prepare evacuation routes.",
                "Move valuables to higher floors.",
                "Stock emergency supplies (3 days).",
            ],
            "High": [
                "🚨 Issue flood warning to residents.",
                "Activate emergency response teams.",
                "Open evacuation shelters immediately.",
                "Block flood-prone roads.",
                "Evacuate low-lying areas.",
            ],
            "Extreme": [
                "🔴 CRITICAL: Mass evacuation required.",
                "Deploy NDRF/SDRF teams immediately.",
                "Activate all emergency protocols.",
                "Coordinate with state disaster authority.",
                "Issue media alerts across all channels.",
                "Establish relief camps with medical units.",
            ],
        }
        return base.get(risk_level, [])

    def update_with_new_data(self, new_rainfall_point: float):
        """
        Online learning — append new real-time data point and fine-tune.
        Call this daily to keep the model fresh.
        """
        if hasattr(self.arima, 'last_train_data') and self.arima.last_train_data is not None:
            updated = np.append(self.arima.last_train_data, new_rainfall_point)
            # Keep last 2 years max
            if len(updated) > 730:
                updated = updated[-730:]
            self.train(updated, force_retrain=True)

    def _save(self):
        """Persist model to disk."""
        try:
            self.lstm.save(self.model_path)
            meta = {
                "district": self.district,
                "is_trained": self.is_trained,
                "training_history": self.training_history,
                "arima_last_data": self.arima.last_train_data.tolist()
                    if self.arima.last_train_data is not None else [],
            }
            with open(self.model_path + "_meta.json", "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            print(f"  [Hybrid] Save warning: {e}")

    def _load_if_exists(self) -> bool:
        """Load model from disk if available."""
        meta_path = self.model_path + "_meta.json"
        if not os.path.exists(meta_path):
            return False
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            self.is_trained = meta.get("is_trained", False)
            self.training_history = meta.get("training_history", [])
            last_data = meta.get("arima_last_data", [])
            if last_data:
                self.arima.last_train_data = np.array(last_data)
            self.lstm.load(self.model_path)
            return self.is_trained
        except Exception as e:
            print(f"  [Hybrid] Load warning: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════
# Model Registry — one model per district, cached in memory
# ═══════════════════════════════════════════════════════════════════════════
_MODEL_REGISTRY: dict[str, HybridARIMALSTM] = {}


def get_model(district: str) -> HybridARIMALSTM:
    """Returns the hybrid model for a district (creates if not cached)."""
    if district not in _MODEL_REGISTRY:
        _MODEL_REGISTRY[district] = HybridARIMALSTM(district)
    return _MODEL_REGISTRY[district]


def train_model_from_dataframe(district: str, df: pd.DataFrame,
                               date_col="date", rainfall_col="rainfall") -> HybridARIMALSTM:
    """
    Convenience function: train model from a pandas DataFrame.
    df must have columns: date (str/datetime), rainfall (float, mm)
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    series = df[rainfall_col].fillna(0).values.astype(float)

    model = get_model(district)
    model.train(series, force_retrain=True)
    return model


def quick_predict(district: str, rainfall_series: np.ndarray,
                  current_rainfall: float, affected_area: float) -> dict:
    """
    One-shot: train (or load) + predict.
    Perfect for integration with app.py.
    """
    model = get_model(district)
    if not model.is_trained:
        model.train(rainfall_series)
    return model.predict_flood_risk(current_rainfall, affected_area)


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Hybrid ARIMA+LSTM Quick Test ===")
    np.random.seed(42)
    # Simulate 2 years of daily rainfall with seasonal pattern
    t = np.arange(365 * 2)
    synthetic_rainfall = (
        30 * np.sin(2 * np.pi * t / 365 + np.pi / 2) +   # seasonal
        10 * np.random.randn(len(t)) +                     # noise
        20                                                 # base
    ).clip(0)

    model = HybridARIMALSTM("TestDistrict")
    model.train(synthetic_rainfall, force_retrain=True)

    result = model.predict_flood_risk(
        current_rainfall=120.5,
        affected_area_km2=450.0,
        population=500000,
    )

    print(f"\nDistrict:     {result['district']}")
    print(f"Risk Level:   {result['risk_level']} ({result['risk_score']}/100)")
    print(f"7-Day FC:     {result['forecast_7day_mm']}")
    print(f"Model:        {result['model_type']}")
    print(f"ARIMA:        {result['arima_component']}")
    print(f"LSTM:         {result['lstm_component']}")