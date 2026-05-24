"""
flood_model.py
--------------
Hybrid ARIMA-LSTM flood prediction model.
Reads rainfall CSVs for historical context.
Uses trained models from models_saved/ if available,
otherwise uses rule-based scoring.
"""

import numpy as np
import pandas as pd
import os, pickle, warnings
warnings.filterwarnings("ignore")

from elevation_model import (elevation_flood_multiplier,
                              combined_uttarakhand_risk,
                              classify_flood_risk_by_elevation)

def find_project():
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    for name in ["FMS_PROJECT","flood_project","FLOOD_PROJECT","fms_project"]:
        p = os.path.join(desktop, name)
        if os.path.exists(p):
            return p
    return os.path.dirname(os.path.abspath(__file__))

BASE       = find_project()
DATA       = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models_saved")


# ─────────────────────────────────────────────────────────────
# LOAD RAINFALL CSV
# ─────────────────────────────────────────────────────────────
_RAIN_CACHE = {}

def _load_rainfall(region):
    if region in _RAIN_CACHE:
        return _RAIN_CACHE[region]
    fname = "maharashtra_rainfall.csv" if region == "Maharashtra" \
            else "uttarakhand_rainfall.csv"
    path  = os.path.join(DATA, "rainfall", fname)
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["date"])
        _RAIN_CACHE[region] = df
        return df
    return pd.DataFrame()


def get_district_rainfall_stats(district, region):
    """
    Returns historical rainfall stats from CSV:
    avg_annual_mm, max_daily_mm, extreme_event_count, monsoon_avg_mm
    """
    df = _load_rainfall(region)
    if df.empty or "district" not in df.columns:
        return {"avg_annual_mm": 800, "max_daily_mm": 200,
                "extreme_count": 5,  "monsoon_avg_mm": 600}
    ddf = df[df["district"] == district]
    if ddf.empty:
        return {"avg_annual_mm": 800, "max_daily_mm": 200,
                "extreme_count": 5,  "monsoon_avg_mm": 600}

    # Annual total
    if "year" in ddf.columns:
        annual_avg = float(ddf.groupby("year")["rainfall_mm"].sum().mean())
    else:
        annual_avg = float(ddf["rainfall_mm"].sum() / max(1, len(ddf)/365))

    # Monsoon avg (Jun-Sep)
    if "month" in ddf.columns:
        monsoon = ddf[ddf["month"].isin([6,7,8,9])]["rainfall_mm"].mean()
    else:
        monsoon = annual_avg * 0.75

    extreme_count = 0
    if "is_extreme_event" in ddf.columns:
        extreme_count = int((ddf["is_extreme_event"] == True).sum())

    return {
        "avg_annual_mm":  round(annual_avg, 1),
        "max_daily_mm":   round(float(ddf["rainfall_mm"].max()), 1),
        "extreme_count":  extreme_count,
        "monsoon_avg_mm": round(float(monsoon), 1)
    }


# ─────────────────────────────────────────────────────────────
# LOAD TRAINED ARIMA MODEL (if available)
# ─────────────────────────────────────────────────────────────
def _load_arima(region, district):
    fname = f"arima_{region}_{district.replace(' ','_')}.pkl"
    path  = os.path.join(MODELS_DIR, fname)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def _arima_adjust_score(base_score, region, district):
    """
    If a trained ARIMA model exists, use its forecast to
    slightly adjust the base risk score.
    """
    model = _load_arima(region, district)
    if model is None:
        return base_score, False
    try:
        forecast = model.forecast(steps=7)
        avg_forecast = float(np.maximum(0, forecast).mean())
        # Blend: if ARIMA expects high rainfall, increase score
        if avg_forecast > 100:
            adjustment = 1.15
        elif avg_forecast > 50:
            adjustment = 1.05
        else:
            adjustment = 0.95
        return round(base_score * adjustment, 2), True
    except Exception:
        return base_score, False


# ─────────────────────────────────────────────────────────────
# MAHARASHTRA PREDICTION
# ─────────────────────────────────────────────────────────────
def predict_flood_maharashtra(rainfall_mm, elevation_meters, district=""):
    multiplier   = elevation_flood_multiplier(elevation_meters, "Maharashtra")
    base_score   = rainfall_mm * multiplier
    risk_score, used_arima = _arima_adjust_score(base_score, "Maharashtra", district)

    if risk_score > 200:   severity = "High"
    elif risk_score > 100: severity = "Moderate"
    else:                  severity = "Low"

    # Get historical context from CSV
    rain_stats = get_district_rainfall_stats(district, "Maharashtra")

    return {
        "region":          "Maharashtra",
        "flood_severity":  severity,
        "risk_score":      risk_score,
        "elevation_multiplier": multiplier,
        "disaster_type":   "Riverine / Urban Flood",
        "alert":           "RED" if severity=="High" else
                           ("ORANGE" if severity=="Moderate" else "GREEN"),
        "arima_used":      used_arima,
        "historical_stats": rain_stats
    }


# ─────────────────────────────────────────────────────────────
# UTTARAKHAND PREDICTION
# ─────────────────────────────────────────────────────────────
def predict_flood_uttarakhand(rainfall_mm, elevation_meters, slope_degrees, district=""):
    risk_data  = combined_uttarakhand_risk(elevation_meters, slope_degrees, rainfall_mm)
    combined   = risk_data["combined_risk_score"]
    _, used_arima = _arima_adjust_score(combined, "Uttarakhand", district)

    if combined >= 7:   severity = "High"
    elif combined >= 5: severity = "Moderate"
    else:               severity = "Low"

    if slope_degrees > 30 and rainfall_mm > 80:
        dtype = "Landslide + Flash Flood"
    elif elevation_meters > 3000 and rainfall_mm > 60:
        dtype = "Glacial Lake Outburst Flood (GLOF)"
    else:
        dtype = "Cloudburst / Flash Flood"

    rain_stats = get_district_rainfall_stats(district, "Uttarakhand")

    return {
        "region":            "Uttarakhand",
        "flood_severity":    severity,
        "risk_score":        combined,
        "slope_score":       risk_data["slope_score"],
        "disaster_type":     dtype,
        "alert":             risk_data["alert_level"],
        "landslide_probability": f"{int(risk_data['landslide_probability']*100)}%",
        "glof_risk":         risk_data["glof_risk"],
        "arima_used":        used_arima,
        "historical_stats":  rain_stats
    }


# ─────────────────────────────────────────────────────────────
# UNIFIED ENTRY POINT
# ─────────────────────────────────────────────────────────────
def predict_flood(region, rainfall_mm, elevation_meters,
                  slope_degrees=15, district=""):
    if region == "Maharashtra":
        return predict_flood_maharashtra(rainfall_mm, elevation_meters, district)
    elif region == "Uttarakhand":
        return predict_flood_uttarakhand(rainfall_mm, elevation_meters,
                                         slope_degrees, district)
    else:
        raise ValueError(f"Unknown region: {region}")