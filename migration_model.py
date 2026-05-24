"""
migration_model.py
------------------
Estimates migration / displacement from flood events.
Uses migration_data.csv and census_data.csv for real rates.
Formula from paper:
  M = β0 + β1*Severity + β2*Rainfall + β3*Elevation + β4*Population + β5*Area + β6*Slope
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
# LOAD CSVs
# ─────────────────────────────────────────────────────────────
def _load(fname):
    path = os.path.join(DATA, fname)
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

_MIGRATION_DF = None
_CENSUS_MH    = None
_CENSUS_UK    = None

def _get_migration_df():
    global _MIGRATION_DF
    if _MIGRATION_DF is None:
        _MIGRATION_DF = _load("migration/migration_data.csv")
    return _MIGRATION_DF

def _get_census(region):
    global _CENSUS_MH, _CENSUS_UK
    if region == "Maharashtra":
        if _CENSUS_MH is None:
            _CENSUS_MH = _load("census/maharashtra_census.csv")
        return _CENSUS_MH
    else:
        if _CENSUS_UK is None:
            _CENSUS_UK = _load("census/uttarakhand_census.csv")
        return _CENSUS_UK


# ─────────────────────────────────────────────────────────────
# GET BASE MIGRATION RATE FROM CSV
# ─────────────────────────────────────────────────────────────
def get_base_rate_from_csv(severity, state):
    """
    Reads migration_data.csv and returns the average historical
    migration rate for a given severity level and state.
    """
    df = _get_migration_df()
    fallback = {"High": 0.55, "Moderate": 0.25, "Low": 0.08}

    if df.empty or "flood_severity" not in df.columns:
        return fallback.get(severity, 0.20)

    sub = df[df["flood_severity"] == severity]
    if "state" in df.columns:
        state_sub = sub[sub["state"] == state]
        if not state_sub.empty:
            sub = state_sub

    if sub.empty or "migration_rate_pct" not in sub.columns:
        return fallback.get(severity, 0.20)

    return round(float(sub["migration_rate_pct"].mean()) / 100, 4)


# ─────────────────────────────────────────────────────────────
# GET VULNERABILITY FACTOR FROM CENSUS CSV
# ─────────────────────────────────────────────────────────────
def get_vulnerability(district, region):
    """
    Returns vulnerability index from census CSV.
    Higher index = more likely to migrate.
    """
    df = _get_census(region)
    if df.empty or "district" not in df.columns:
        return 3.0
    row = df[df["district"] == district]
    if row.empty:
        return 3.0
    return float(row.iloc[0].get("vulnerability_index", 3.0))


# ─────────────────────────────────────────────────────────────
# MAIN MIGRATION ESTIMATION
# ─────────────────────────────────────────────────────────────
def estimate_migration(flood_severity, population, elevation_meters,
                       affected_area_km2, rainfall_mm, region,
                       slope_degrees=15, district=""):
    """
    Estimates displaced population using:
    1. Base rate from migration_data.csv (real historical data)
    2. Adjusted by elevation, slope, vulnerability from census CSV
    3. Returns full breakdown by vulnerable groups
    """

    # Step 1: Base rate from CSV
    base_rate = get_base_rate_from_csv(flood_severity, region)

    # Step 2: Elevation adjustment
    if elevation_meters < 100:    elev_factor = 1.25
    elif elevation_meters < 400:  elev_factor = 1.10
    elif elevation_meters < 1000: elev_factor = 1.00
    else:                         elev_factor = 0.85

    # Step 3: Slope adjustment (Uttarakhand mainly)
    if region == "Uttarakhand":
        if slope_degrees > 40:    slope_factor = 1.30
        elif slope_degrees > 30:  slope_factor = 1.20
        elif slope_degrees > 20:  slope_factor = 1.10
        else:                     slope_factor = 1.00
    else:
        slope_factor = 1.00

    # Step 4: Vulnerability from census CSV
    vuln = get_vulnerability(district, region)
    vuln_factor = 1.0 + (vuln - 3.0) * 0.05  # 3.0 is neutral

    # Step 5: Area factor (larger area = more displacement)
    area_factor = min(1.30, 1.0 + (affected_area_km2 / 200))

    # Step 6: Final rate
    final_rate = base_rate * elev_factor * slope_factor * vuln_factor * area_factor
    final_rate = round(min(0.95, max(0.02, final_rate)), 4)  # cap between 2% and 95%

    migrants = int(population * final_rate)
    staying  = population - migrants

    # Step 7: Breakdown by vulnerable groups (from migration CSV averages)
    mdf = _get_migration_df()
    women_pct    = 0.50
    children_pct = 0.33
    elderly_pct  = 0.12
    disabled_pct = 0.04
    if not mdf.empty:
        if "women_displaced"    in mdf.columns: women_pct    = mdf["women_displaced"].sum()    / max(1, mdf["people_displaced"].sum())
        if "children_displaced" in mdf.columns: children_pct = mdf["children_displaced"].sum() / max(1, mdf["people_displaced"].sum())
        if "elderly_displaced"  in mdf.columns: elderly_pct  = mdf["elderly_displaced"].sum()  / max(1, mdf["people_displaced"].sum())
        if "disabled_displaced" in mdf.columns: disabled_pct = mdf["disabled_displaced"].sum() / max(1, mdf["people_displaced"].sum())

    # Average return timeline from CSV
    avg_days = 30
    return_7d_pct  = 35.0
    return_30d_pct = 70.0
    if not mdf.empty:
        if "avg_days_displaced"        in mdf.columns: avg_days       = float(mdf["avg_days_displaced"].mean())
        if "returned_within_7days_pct" in mdf.columns: return_7d_pct  = float(mdf["returned_within_7days_pct"].mean())
        if "returned_within_30days_pct"in mdf.columns: return_30d_pct = float(mdf["returned_within_30days_pct"].mean())

    return {
        "flood_severity":       flood_severity,
        "total_population":     population,
        "base_migration_rate":  round(base_rate * 100, 1),
        "adjusted_rate":        round(final_rate * 100, 1),
        "estimated_migrants":   migrants,
        "people_staying":       staying,
        # Adjustments breakdown
        "elevation_factor":     elev_factor,
        "slope_factor":         slope_factor,
        "vulnerability_index":  round(vuln, 2),
        # Vulnerable groups
        "women_displaced":      int(migrants * women_pct),
        "children_displaced":   int(migrants * children_pct),
        "elderly_displaced":    int(migrants * elderly_pct),
        "disabled_displaced":   int(migrants * disabled_pct),
        # Return timeline
        "avg_days_displaced":   round(avg_days, 0),
        "return_7days_pct":     round(return_7d_pct, 1),
        "return_30days_pct":    round(return_30d_pct, 1),
        # Note
        "data_source":          "migration_data.csv + census CSV" if not mdf.empty else "Rule-based fallback"
    }