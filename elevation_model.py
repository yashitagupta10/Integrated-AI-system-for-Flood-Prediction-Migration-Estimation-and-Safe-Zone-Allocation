"""
elevation_model.py
------------------
Reads elevation CSV data and computes flood/landslide risk scores.
Used by flood_model.py and app.py
"""

import pandas as pd
import numpy as np
import os

def find_project():
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    for name in ["FMS_PROJECT","flood_project","FLOOD_PROJECT","fms_project"]:
        p = os.path.join(desktop, name)
        if os.path.exists(p):
            return p
    return os.path.dirname(os.path.abspath(__file__))

BASE = find_project()
DATA = os.path.join(BASE, "data")

# ─────────────────────────────────────────────────────────────
# LOAD ELEVATION CSVs
# ─────────────────────────────────────────────────────────────
def _load_elevation_csv(region):
    fname = "maharashtra_elevation.csv" if region == "Maharashtra" else "uttarakhand_elevation.csv"
    path  = os.path.join(DATA, "elevation", fname)
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

_ELEV_CACHE = {}

def get_elevation_df(region):
    if region not in _ELEV_CACHE:
        _ELEV_CACHE[region] = _load_elevation_csv(region)
    return _ELEV_CACHE[region]


# ─────────────────────────────────────────────────────────────
# DISTRICT ELEVATION STATS FROM CSV
# ─────────────────────────────────────────────────────────────
def get_district_elevation(district, region):
    """
    Returns avg elevation, slope, and terrain breakdown
    for a district using the elevation CSV.
    """
    df = get_elevation_df(region)
    if df.empty or "district" not in df.columns:
        return {"avg_elevation_m": 500, "avg_slope_deg": 15,
                "min_elevation_m": 200, "max_elevation_m": 1000,
                "high_risk_pct": 20.0}
    ddf = df[df["district"] == district]
    if ddf.empty:
        return {"avg_elevation_m": 500, "avg_slope_deg": 15,
                "min_elevation_m": 200, "max_elevation_m": 1000,
                "high_risk_pct": 20.0}

    high_risk_pct = 0.0
    if "landslide_risk" in ddf.columns:
        high_risk_pct = round(
            ddf["landslide_risk"].isin(["High","Very High","Extreme"]).mean() * 100, 1
        )
    return {
        "avg_elevation_m": round(float(ddf["elevation_m"].mean()), 1),
        "avg_slope_deg":   round(float(ddf["slope_degrees"].mean()), 1),
        "min_elevation_m": round(float(ddf["elevation_m"].min()), 1),
        "max_elevation_m": round(float(ddf["elevation_m"].max()), 1),
        "high_risk_pct":   high_risk_pct
    }


# ─────────────────────────────────────────────────────────────
# ELEVATION FLOOD MULTIPLIER
# ─────────────────────────────────────────────────────────────
def elevation_flood_multiplier(elevation_m, region="Maharashtra"):
    """
    Returns a multiplier for flood risk based on elevation.
    Lower elevation = higher multiplier (more flood prone).
    """
    if region == "Maharashtra":
        if elevation_m < 50:    return 2.0
        elif elevation_m < 150: return 1.7
        elif elevation_m < 300: return 1.4
        elif elevation_m < 600: return 1.0
        else:                   return 0.6
    else:  # Uttarakhand — valley floors most dangerous
        if elevation_m < 500:   return 2.5
        elif elevation_m < 1000:return 2.0
        elif elevation_m < 2000:return 1.5
        elif elevation_m < 3500:return 1.0
        else:                   return 0.4


# ─────────────────────────────────────────────────────────────
# CLASSIFY RISK BY ELEVATION
# ─────────────────────────────────────────────────────────────
def classify_flood_risk_by_elevation(elevation_m, region="Maharashtra"):
    if region == "Maharashtra":
        if elevation_m < 100:   return "Very High"
        elif elevation_m < 300: return "High"
        elif elevation_m < 600: return "Moderate"
        elif elevation_m < 900: return "Low"
        else:                   return "Very Low"
    else:
        if elevation_m < 500:   return "Very High"
        elif elevation_m < 1000:return "High"
        elif elevation_m < 2000:return "Moderate"
        elif elevation_m < 3500:return "Low"
        else:                   return "Very Low"


# ─────────────────────────────────────────────────────────────
# SLOPE RISK (Uttarakhand)
# ─────────────────────────────────────────────────────────────
def slope_risk_score(slope_degrees):
    """Returns 0-10 score and landslide probability from slope angle."""
    if slope_degrees > 45:   return 9, 0.90
    elif slope_degrees > 30: return 7, 0.70
    elif slope_degrees > 20: return 5, 0.45
    elif slope_degrees > 10: return 3, 0.20
    else:                    return 1, 0.05


# ─────────────────────────────────────────────────────────────
# COMBINED UTTARAKHAND RISK
# ─────────────────────────────────────────────────────────────
def combined_uttarakhand_risk(elevation_m, slope_degrees, rainfall_mm):
    """
    Combines slope + rainfall + elevation into one risk score (0-10).
    Formula from research paper:
      combined = (slope_score×0.4) + (rain_score×0.35) + (elev_score×0.25)
    """
    slope_score, landslide_prob = slope_risk_score(slope_degrees)

    if elevation_m < 500:    elev_score = 9
    elif elevation_m < 1000: elev_score = 7
    elif elevation_m < 2000: elev_score = 5
    elif elevation_m < 3500: elev_score = 3
    else:                    elev_score = 1

    if rainfall_mm > 150:   rain_score = 9
    elif rainfall_mm > 100: rain_score = 7
    elif rainfall_mm > 60:  rain_score = 5
    elif rainfall_mm > 30:  rain_score = 3
    else:                   rain_score = 1

    combined = round((slope_score*0.4) + (rain_score*0.35) + (elev_score*0.25), 2)

    if combined >= 7:    alert = "RED"
    elif combined >= 5:  alert = "ORANGE"
    elif combined >= 3:  alert = "YELLOW"
    else:                alert = "GREEN"

    glof_risk = "High"     if elevation_m > 3000 and rainfall_mm > 60  else \
                "Moderate" if elevation_m > 2000 and rainfall_mm > 80  else "Low"

    return {
        "combined_risk_score": combined,
        "slope_score":         slope_score,
        "elev_score":          elev_score,
        "rain_score":          rain_score,
        "landslide_probability": landslide_prob,
        "alert_level":         alert,
        "glof_risk":           glof_risk
    }