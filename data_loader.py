"""
data_loader.py
--------------
Central loader for all 12 CSV datasets.
Called by app.py at startup to preload everything into memory.
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
# INDIVIDUAL LOADERS
# ─────────────────────────────────────────────────────────────
def _read(rel_path):
    full = os.path.join(DATA, rel_path)
    if not os.path.exists(full):
        print(f"  ⚠️  Missing: {full}")
        return pd.DataFrame()
    df = pd.read_csv(full)
    print(f"  ✅ {rel_path}: {len(df):,} rows, {len(df.columns)} cols")
    return df

def load_all():
    """Load all 12 CSV files and return as a dict."""
    print(f"\n{'='*50}")
    print(f"  Loading all 12 CSV datasets")
    print(f"  From: {DATA}")
    print(f"{'='*50}")

    data = {
        "rainfall_mh":    _read("rainfall/maharashtra_rainfall.csv"),
        "rainfall_uk":    _read("rainfall/uttarakhand_rainfall.csv"),
        "census_mh":      _read("census/maharashtra_census.csv"),
        "census_uk":      _read("census/uttarakhand_census.csv"),
        "elevation_mh":   _read("elevation/maharashtra_elevation.csv"),
        "elevation_uk":   _read("elevation/uttarakhand_elevation.csv"),
        "shelters_mh":    _read("shelters/maharashtra_shelters.csv"),
        "shelters_uk":    _read("shelters/uttarakhand_shelters.csv"),
        "flood_history":  _read("flood_history/historical_flood_events.csv"),
        "migration":      _read("migration/migration_data.csv"),
        "rivers":         _read(os.path.join("rivers", "river_levels.csv")),
    }

    loaded   = sum(1 for v in data.values() if not v.empty)
    missing  = sum(1 for v in data.values() if v.empty)
    total_rows = sum(len(v) for v in data.values())

    print(f"\n  ✅ Loaded:  {loaded}/11 files")
    if missing:
        print(f"  ⚠️  Missing: {missing} files (will use fallback values)")
    print(f"  📊 Total rows in memory: {total_rows:,}")
    print(f"{'='*50}\n")
    return data


# ─────────────────────────────────────────────────────────────
# BUILD DISTRICT LOOKUP FROM CSVs
# ─────────────────────────────────────────────────────────────
def build_district_lookup(data):
    """
    Combines census + elevation CSVs into a fast lookup dict:
    { "Sangli": { population, elevation, slope, lat, lon, ... } }
    """
    lookup = {}

    for region, census_key, elev_key in [
        ("Maharashtra", "census_mh", "elevation_mh"),
        ("Uttarakhand", "census_uk", "elevation_uk")
    ]:
        census_df = data.get(census_key, pd.DataFrame())
        elev_df   = data.get(elev_key,   pd.DataFrame())

        if census_df.empty:
            continue

        for _, row in census_df.iterrows():
            dist = row["district"]

            # Get elevation/slope from elevation CSV
            if not elev_df.empty and "district" in elev_df.columns:
                edist = elev_df[elev_df["district"] == dist]
                avg_elev  = float(edist["elevation_m"].mean())   if not edist.empty else 500.0
                avg_slope = float(edist["slope_degrees"].mean()) if not edist.empty else 15.0
                high_risk = float(
                    edist["landslide_risk"].isin(["High","Very High","Extreme"]).mean() * 100
                ) if (not edist.empty and "landslide_risk" in edist.columns) else 20.0
            else:
                avg_elev, avg_slope, high_risk = 500.0, 15.0, 20.0

            lookup[dist] = {
                "state":        region,
                "population":   int(row.get("total_population",   100000)),
                "male_pop":     int(row.get("male_population",     50000)),
                "female_pop":   int(row.get("female_population",   50000)),
                "urban_pop":    int(row.get("urban_population",    30000)),
                "rural_pop":    int(row.get("rural_population",    70000)),
                "elevation":    round(avg_elev, 0),
                "slope":        round(avg_slope, 1),
                "high_risk_pct":round(high_risk, 1),
                "latitude":     float(row.get("latitude",  0)),
                "longitude":    float(row.get("longitude", 0)),
                "flood_prone":  bool(row.get("flood_prone", False)),
                "river":        str(row.get("primary_river", "")),
                "literacy":     float(row.get("literacy_rate_pct",       75.0)),
                "poverty_pct":  float(row.get("below_poverty_line_pct",  20.0)),
                "vuln_index":   float(row.get("vulnerability_index",      3.0)),
                "area_km2":     float(row.get("area_km2",               1000.0)),
            }

    print(f"  ✅ District lookup built: {len(lookup)} districts total")
    return lookup


# ─────────────────────────────────────────────────────────────
# RIVER STATUS LOOKUP
# ─────────────────────────────────────────────────────────────
def get_river_status(rivers_df, river_name):
    """Return latest status for a named river from rivers CSV."""
    if rivers_df.empty or "river" not in rivers_df.columns:
        return None
    rdf = rivers_df[rivers_df["river"].str.contains(river_name, na=False, case=False)]
    if rdf.empty:
        return None
    if "year" in rdf.columns and "month" in rdf.columns:
        rdf = rdf.sort_values(["year","month"])
    latest = rdf.iloc[-1]
    return {
        "river":       str(latest.get("river", river_name)),
        "level_m":     float(latest.get("water_level_m", 0)),
        "status":      str(latest.get("status", "Normal")),
        "flood_alert": bool(latest.get("flood_alert", False)),
        "warning_m":   float(latest.get("warning_level_m", 0)),
        "danger_m":    float(latest.get("danger_level_m",  0)),
    }


# ─────────────────────────────────────────────────────────────
# FLOOD HISTORY LOOKUP
# ─────────────────────────────────────────────────────────────
def get_worst_event(flood_hist_df, district, state):
    """Return worst historical flood event for a district."""
    if flood_hist_df.empty:
        return None
    df = flood_hist_df[
        (flood_hist_df.get("district","") == district) &
        (flood_hist_df.get("state","")    == state)
    ] if "district" in flood_hist_df.columns else pd.DataFrame()

    if df.empty and "state" in flood_hist_df.columns:
        df = flood_hist_df[flood_hist_df["state"] == state]
    if df.empty:
        return None

    worst = df.loc[df["deaths"].idxmax()] if "deaths" in df.columns else df.iloc[-1]
    return {
        "year":      int(worst.get("year",              0)),
        "event":     str(worst.get("event_name",       "Unknown")),
        "severity":  str(worst.get("severity",         "High")),
        "deaths":    int(worst.get("deaths",            0)),
        "displaced": int(worst.get("people_displaced",  0)),
        "rainfall":  int(worst.get("peak_rainfall_mm",  0)),
        "loss_crore":float(worst.get("economic_loss_crore", 0)),
        "army_deployed": bool(worst.get("army_deployed", False)),
    }