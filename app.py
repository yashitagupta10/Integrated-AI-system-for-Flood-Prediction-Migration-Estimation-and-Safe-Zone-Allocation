"""
app.py  (Updated with Authentication)
--------------------------------------
All routes now protected by login.
Users stored in users.json (no database needed).
New routes added:
  GET/POST  /login    - Login page
  GET/POST  /register - Register page
  GET       /logout   - Logout
All existing routes remain unchanged.
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os

from data_loader     import load_all, build_district_lookup, get_river_status, get_worst_event
from flood_model     import predict_flood
from migration_model import estimate_migration
from shelter         import allocate_shelters
from auth            import login_user, register_user   # ← NEW

app = Flask(__name__)
app.secret_key = "fms_flood_secret_key_2024"           # ← required for sessions

# ─────────────────────────────────────────────────────────────
# LOAD ALL 12 CSVs AT STARTUP
# ─────────────────────────────────────────────────────────────
print("Starting Flood Management System...")
DATA   = load_all()
LOOKUP = build_district_lookup(DATA)
MH_DISTRICTS = {k: v for k, v in LOOKUP.items() if v["state"] == "Maharashtra"}
UK_DISTRICTS = {k: v for k, v in LOOKUP.items() if v["state"] == "Uttarakhand"}
print(f"Maharashtra: {len(MH_DISTRICTS)} districts | Uttarakhand: {len(UK_DISTRICTS)} districts")


# ─────────────────────────────────────────────────────────────
# LOGIN REQUIRED DECORATOR
# ─────────────────────────────────────────────────────────────
from functools import wraps

def login_required(f):
    """Redirect to login page if user is not logged in."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ─────────────────────────────────────────────────────────────
# AUTH ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    # Already logged in → go to dashboard
    if "username" in session:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        success, result = login_user(username, password)
        if success:
            session["username"] = result["username"]
            session["name"]     = result["name"]
            return redirect(url_for("home"))
        else:
            return render_template("login.html", error=result, username=username)

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if "username" in session:
        return redirect(url_for("home"))

    if request.method == "POST":
        username         = request.form.get("username", "").strip()
        password         = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()
        name             = request.form.get("name", "").strip()

        success, message = register_user(username, password, confirm_password, name)
        if success:
            return render_template("register.html", success=message)
        else:
            return render_template("register.html", error=message,
                                   username=username, name=name)

    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ─────────────────────────────────────────────────────────────
# PROTECTED ROUTES (all need login)
# ─────────────────────────────────────────────────────────────

@app.route("/")
@login_required
def home():
    return render_template("index.html",
                           user_name=session.get("name", "User"),
                           username=session.get("username", ""))


@app.route("/get_districts/<region>")
@login_required
def get_districts(region):
    districts_map = MH_DISTRICTS if region == "Maharashtra" else UK_DISTRICTS
    center = [18.5, 76.0] if region == "Maharashtra" else [30.0, 79.0]
    zoom   = 6            if region == "Maharashtra" else 7
    districts = []
    for name, info in districts_map.items():
        districts.append({
            "name":        name,
            "population":  info["population"],
            "elevation":   info["elevation"],
            "slope":       info["slope"],
            "lat":         info["latitude"],
            "lon":         info["longitude"],
            "flood_prone": info["flood_prone"],
            "river":       info["river"],
            "vuln_index":  info["vuln_index"],
            "poverty_pct": info["poverty_pct"],
        })
    return jsonify({"districts": districts, "center": center, "zoom": zoom})


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    try:
        body     = request.json
        region   = body.get("region",   "Maharashtra")
        district = body.get("district", "")
        rainfall = float(body.get("rainfall", 0))
        area     = float(body.get("area",     20))
        slope_in = body.get("slope", None)

        info       = LOOKUP.get(district, {})
        population = info.get("population",  100000)
        elevation  = info.get("elevation",   500)
        slope      = float(slope_in) if slope_in else info.get("slope", 15)
        lat        = info.get("latitude",   18.5 if region == "Maharashtra" else 30.0)
        lon        = info.get("longitude",  76.0 if region == "Maharashtra" else 79.0)
        river      = info.get("river", "")

        flood = predict_flood(
            region=region, rainfall_mm=rainfall,
            elevation_meters=elevation, slope_degrees=slope, district=district
        )
        migration = estimate_migration(
            flood_severity=flood["flood_severity"], population=population,
            elevation_meters=elevation, affected_area_km2=area,
            rainfall_mm=rainfall, region=region, slope_degrees=slope, district=district
        )
        shelter = allocate_shelters(
            migrants=migration["estimated_migrants"], region=region, district=district
        )
        worst        = get_worst_event(DATA["flood_history"], district, region)
        river_status = get_river_status(DATA["rivers"], river) if river else None

        vuln_info = {
            "vulnerability_index":      info.get("vuln_index",  3.0),
            "poverty_pct":              info.get("poverty_pct", 20.0),
            "literacy_pct":             info.get("literacy",    75.0),
            "flood_prone":              info.get("flood_prone", False),
            "high_landslide_risk_pct":  info.get("high_risk_pct", 20.0),
        }

        return jsonify({
            "status":       "success",
            "region":       region,
            "district":     district,
            "inputs": {
                "rainfall_mm": rainfall, "elevation_m": elevation,
                "slope_deg": slope, "population": population, "area_km2": area,
            },
            "flood":         flood,
            "migration":     migration,
            "shelter":       shelter,
            "worst_event":   worst,
            "river_status":  river_status,
            "vulnerability": vuln_info,
            "coordinates":   [lat, lon],
            "predicted_by":  session.get("name", "Unknown"),   # ← tracks who ran prediction
        })

    except Exception as e:
        import traceback
        return jsonify({"status": "error", "message": str(e),
                        "trace": traceback.format_exc()}), 500


@app.route("/flood_history/<region>")
@login_required
def flood_history(region):
    df = DATA.get("flood_history")
    if df is None or df.empty:
        return jsonify({"events": []})
    region_df = df[df["state"] == region] if "state" in df.columns else df
    cols = [c for c in ["year","district","event_name","severity",
                        "peak_rainfall_mm","people_displaced",
                        "deaths","economic_loss_crore","event_type"]
            if c in region_df.columns]
    return jsonify({"region": region, "count": len(region_df),
                    "events": region_df[cols].to_dict("records")})


@app.route("/river_levels/<region>")
@login_required
def river_levels(region):
    df = DATA.get("rivers")
    if df is None or df.empty:
        return jsonify({"rivers": []})
    rdf = df[df["state"] == region] if "state" in df.columns else df
    if "year" in rdf.columns and "month" in rdf.columns:
        latest = rdf.sort_values(["year","month"]).groupby("river").last().reset_index()
    else:
        latest = rdf.groupby("river").last().reset_index()
    cols = [c for c in ["river","water_level_m","status","flood_alert",
                        "warning_level_m","danger_level_m"] if c in latest.columns]
    return jsonify({"region": region, "rivers": latest[cols].to_dict("records")})


@app.route("/summary")
@login_required
def summary():
    return jsonify({
        "logged_in_as":          session.get("username"),
        "maharashtra_districts": len(MH_DISTRICTS),
        "uttarakhand_districts": len(UK_DISTRICTS),
        "datasets_loaded": {k: len(v) for k, v in DATA.items() if v is not None}
    })


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  FLOOD DISASTER MANAGEMENT SYSTEM")
    print("  Authentication: ENABLED ✅")
    print("  Default login  → admin / admin123")
    print("  Open: http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=True, port=5000)