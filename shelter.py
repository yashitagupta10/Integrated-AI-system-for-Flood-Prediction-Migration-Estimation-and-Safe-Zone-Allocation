"""
shelter.py
----------
Allocates displaced people to shelters.
Reads shelter CSVs (maharashtra_shelters.csv / uttarakhand_shelters.csv).
Prioritises road-accessible shelters, then helicopter-only for Uttarakhand.
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
# LOAD SHELTER CSVs
# ─────────────────────────────────────────────────────────────
_SHELTER_CACHE = {}

def load_shelters(region):
    if region in _SHELTER_CACHE:
        return _SHELTER_CACHE[region]
    fname = "maharashtra_shelters.csv" if region == "Maharashtra" \
            else "uttarakhand_shelters.csv"
    path  = os.path.join(DATA, "shelters", fname)
    if os.path.exists(path):
        df = pd.read_csv(path)
        _SHELTER_CACHE[region] = df
        return df
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# PRIORITISE SHELTERS FOR A DISTRICT
# ─────────────────────────────────────────────────────────────
def get_prioritised_shelters(region, district=""):
    """
    Returns shelters sorted by:
    1. Same district first
    2. Good structural condition
    3. Highest capacity
    Road shelters before helicopter-only.
    """
    df = load_shelters(region)
    if df.empty:
        return df

    # Prefer same-district shelters
    if district and "district" in df.columns:
        same = df[df["district"] == district].copy()
        others = df[df["district"] != district].copy()
        df = pd.concat([same, others]).reset_index(drop=True)

    # Sort by condition, then capacity
    if "structural_condition" in df.columns:
        cond_order = {"Good": 0, "Fair": 1, "Needs Repair": 2}
        df["_cond_rank"] = df["structural_condition"].map(cond_order).fillna(1)
        df = df.sort_values(["_cond_rank", "capacity_persons"],
                            ascending=[True, False])

    return df


# ─────────────────────────────────────────────────────────────
# MAIN ALLOCATION FUNCTION
# ─────────────────────────────────────────────────────────────
def allocate_shelters(migrants, region, district=""):
    """
    Allocate `migrants` people across shelters from CSV.
    Returns detailed allocation with per-shelter breakdown.
    """
    df = get_prioritised_shelters(region, district)

    if df.empty:
        return {
            "region": region,
            "total_migrants":    migrants,
            "total_sheltered":   0,
            "unallocated":       migrants,
            "total_capacity":    0,
            "shelters_used":     0,
            "allocations":       [],
            "heli_needed":       False,
            "warning": "⚠️ No shelter data available. Please check shelters CSV."
        }

    # Separate road vs helicopter shelters
    heli_col = "helicopter_access_only"
    if heli_col in df.columns:
        road_df = df[df[heli_col] == False].copy()
        heli_df = df[df[heli_col] == True].copy()
    else:
        road_df = df.copy()
        heli_df = pd.DataFrame()

    remaining   = migrants
    allocations = []
    heli_needed = False

    def fill(shelter_df, access_type):
        nonlocal remaining
        for _, s in shelter_df.iterrows():
            if remaining <= 0:
                break
            cap      = int(s.get("capacity_persons", 500))
            assigned = min(remaining, cap)
            remaining -= assigned
            pct = round(assigned / cap * 100, 1)

            # Build shelter info dict
            entry = {
                "shelter_id":   str(s.get("shelter_id", "")),
                "name":         str(s.get("name", "Shelter")),
                "district":     str(s.get("district", "")),
                "type":         str(s.get("shelter_type", "Shelter")),
                "assigned":     assigned,
                "capacity":     cap,
                "occupancy_pct":pct,
                "access":       access_type,
                "has_medical":  bool(s.get("has_medical_facility", False)),
                "has_water":    bool(s.get("has_water_supply", True)),
                "has_electricity": bool(s.get("has_electricity", True)),
                "condition":    str(s.get("structural_condition", "Good")),
                "contact":      str(s.get("contact_number", "N/A")),
                "lat":          float(s.get("latitude", 0)),
                "lon":          float(s.get("longitude", 0)),
            }
            allocations.append(entry)

    # Fill road shelters first
    fill(road_df, "Road")

    # Fill helicopter shelters if still people left (Uttarakhand)
    if remaining > 0 and not heli_df.empty:
        heli_needed = True
        fill(heli_df, "Helicopter Only")

    # Total capacity of all shelters in CSV
    total_cap = int(df["capacity_persons"].sum()) if "capacity_persons" in df.columns else 0

    result = {
        "region":          region,
        "total_migrants":  migrants,
        "total_sheltered": migrants - remaining,
        "unallocated":     remaining,
        "total_capacity":  total_cap,
        "shelters_used":   len(allocations),
        "allocations":     allocations,
        "heli_needed":     heli_needed,
        "data_source":     f"{region.lower()}_shelters.csv"
    }

    if remaining > 0:
        result["warning"] = (
            f"⚠️ {remaining:,} people could not be allocated to shelters. "
            f"Additional emergency relief needed!"
        )

    return result