"""
app_routes_patch.py
===================
NEW ROUTES to add to your existing app.py (FMS_PROJECT).

INSTRUCTIONS:
1. Copy these route functions into your app.py
2. Add these imports at the top of app.py:
       from weather_service import get_district_weather_package
       from hybrid_model import get_model, train_model_from_dataframe, quick_predict
       import numpy as np
3. Call  init_hybrid_models(DATA)  after your DATA loading block.
4. Place weather_service.py and hybrid_model.py in the same folder as app.py.

The routes added:
  GET  /get_realtime_weather/<district>    → auto-fills rainfall + area
  POST /analyze_hybrid                    → hybrid ARIMA+LSTM prediction
  POST /update_model/<district>           → online learning with new data point
  GET  /model_status                      → shows which models are trained
"""

# ── ADD THESE IMPORTS to the top of app.py ──────────────────────────────────
# from weather_service import get_district_weather_package
# from hybrid_model import get_model, train_model_from_dataframe, quick_predict
# import numpy as np

# ── ADD THIS FUNCTION after DATA is loaded in app.py ────────────────────────
def init_hybrid_models(DATA: dict):
    """
    Pre-train hybrid models for all districts using existing DATA.
    Call this once at startup after DATA is populated.

    Example — add to app.py right after your data loading:
        init_hybrid_models(DATA)
    """
    from hybrid_model import get_model
    import pandas as pd

    rainfall_df = DATA.get("rainfall")
    if rainfall_df is None:
        print("[Hybrid Init] No rainfall DataFrame found in DATA.")
        return

    # Detect column names (adjust if yours differ)
    date_col = next((c for c in rainfall_df.columns
                     if "date" in c.lower()), None)
    rain_col = next((c for c in rainfall_df.columns
                     if "rain" in c.lower() or "precip" in c.lower()), None)
    dist_col = next((c for c in rainfall_df.columns
                     if "dist" in c.lower() or "region" in c.lower()), None)

    if not rain_col:
        print("[Hybrid Init] Could not find rainfall column.")
        return

    districts = rainfall_df[dist_col].unique() if dist_col else ["All"]

    trained = 0
    for district in districts:
        try:
            if dist_col:
                subset = rainfall_df[rainfall_df[dist_col] == district].copy()
            else:
                subset = rainfall_df.copy()

            if date_col:
                subset = subset.sort_values(date_col)

            series = subset[rain_col].fillna(0).values.astype(float)

            if len(series) < 30:
                continue

            model = get_model(district)
            if not model.is_trained:
                model.train(series)
                trained += 1
        except Exception as e:
            print(f"  [Hybrid Init] Skipped {district}: {e}")

    print(f"[Hybrid Init] Trained {trained} new district models.")


# ── NEW FLASK ROUTES ─────────────────────────────────────────────────────────
# Paste these functions into your app.py Flask app

def register_new_routes(app, DATA, auth_required):
    """
    Call this in app.py:  register_new_routes(app, DATA, login_required_decorator)
    Or just copy-paste the route functions directly into app.py.
    """
    from flask import jsonify, request, session
    from weather_service import get_district_weather_package
    from hybrid_model import get_model, quick_predict
    import numpy as np

    # ── 1. Real-time weather auto-fill ──────────────────────────────────────
    @app.route("/get_realtime_weather/<district>")
    def get_realtime_weather(district):
        """
        Called by the frontend when user selects a district.
        Auto-fills rainfall (mm) and affected area (km²).
        
        Frontend JS usage:
            fetch(`/get_realtime_weather/${district}`)
              .then(r => r.json())
              .then(data => {
                  document.getElementById('rainfall-input').value = data.rainfall_for_model;
                  document.getElementById('area-input').value    = data.affected_area_km2;
                  updateRainfallSlider(data.rainfall_for_model);
              });
        """
        try:
            pkg = get_district_weather_package(district)
            return jsonify({
                "success": True,
                "district": district,
                "rainfall_mm": pkg["rainfall_for_model"],         # 24h accumulated
                "current_rainfall_mm": pkg["current_rainfall_mm"], # this-hour
                "affected_area_km2": pkg["affected_area_km2"],
                "humidity": pkg["humidity"],
                "forecast_7day": pkg["forecast_7day"],
                "lat": pkg["lat"],
                "lon": pkg["lon"],
                "timestamp": pkg["timestamp"],
                "data_source": "Open-Meteo (real-time)",
            })
        except ConnectionError as e:
            return jsonify({"success": False, "error": str(e), "fallback": True}), 503
        except ValueError as e:
            return jsonify({"success": False, "error": str(e)}), 404
        except Exception as e:
            return jsonify({"success": False, "error": f"Unexpected error: {e}"}), 500

    # ── 2. Hybrid ARIMA+LSTM prediction ─────────────────────────────────────
    @app.route("/analyze_hybrid", methods=["POST"])
    def analyze_hybrid():
        """
        Drop-in replacement (or complement) to your existing /analyze endpoint.
        Uses real-time weather + hybrid model.

        POST body (JSON):
            {
                "district": "Pune",
                "rainfall": 120.5,         ← optional; fetched live if omitted
                "affected_area": 450.0,    ← optional; computed if omitted
                "use_realtime": true       ← set false to use manual values
            }
        """
        body = request.get_json(silent=True) or {}
        district = body.get("district")
        use_realtime = body.get("use_realtime", True)

        if not district:
            return jsonify({"error": "district is required"}), 400

        # ── Get rainfall & area (real-time or manual) ──────────────────────
        if use_realtime:
            try:
                pkg = get_district_weather_package(district)
                rainfall = pkg["rainfall_for_model"]
                affected_area = pkg["affected_area_km2"]
                weather_data = pkg
            except Exception as e:
                # Fallback to manual values
                rainfall = float(body.get("rainfall", 0))
                affected_area = float(body.get("affected_area", 50))
                weather_data = {}
                print(f"  [analyze_hybrid] Weather fetch failed, using manual: {e}")
        else:
            rainfall = float(body.get("rainfall", 0))
            affected_area = float(body.get("affected_area", 50))
            weather_data = {}

        # ── Get or create hybrid model for district ────────────────────────
        model = get_model(district)

        if not model.is_trained:
            # Bootstrap from DATA if available
            rainfall_df = DATA.get("rainfall")
            if rainfall_df is not None:
                try:
                    dist_col = next(
                        (c for c in rainfall_df.columns if "dist" in c.lower()), None
                    )
                    rain_col = next(
                        (c for c in rainfall_df.columns
                         if "rain" in c.lower() or "precip" in c.lower()), None
                    )
                    if dist_col and rain_col:
                        subset = rainfall_df[rainfall_df[dist_col] == district]
                        series = subset[rain_col].fillna(0).values.astype(float)
                        if len(series) >= 30:
                            model.train(series)
                except Exception as e:
                    print(f"  [analyze_hybrid] Model bootstrap failed: {e}")

        # ── Get population from census ─────────────────────────────────────
        population = 100000
        census = DATA.get("census")
        if census is not None:
            try:
                dist_col = next(
                    (c for c in census.columns if "dist" in c.lower()), None
                )
                pop_col = next(
                    (c for c in census.columns if "pop" in c.lower()), None
                )
                if dist_col and pop_col:
                    row = census[census[dist_col].str.lower() == district.lower()]
                    if not row.empty:
                        population = int(row[pop_col].values[0])
            except Exception:
                pass

        # ── Run prediction ─────────────────────────────────────────────────
        if model.is_trained:
            prediction = model.predict_flood_risk(
                current_rainfall=rainfall,
                affected_area_km2=affected_area,
                population=population,
            )
        else:
            # Fallback: rule-based prediction (no model)
            prediction = _rule_based_fallback(district, rainfall, affected_area, population)

        # Merge weather data into response
        if weather_data:
            prediction["realtime_weather"] = {
                "source": "Open-Meteo",
                "humidity": weather_data.get("humidity"),
                "forecast_7day_openmeteo": weather_data.get("forecast_7day", []),
                "fetched_at": weather_data.get("timestamp"),
            }

        return jsonify(prediction)

    # ── 3. Online learning — update model with new data ──────────────────────
    @app.route("/update_model/<district>", methods=["POST"])
    def update_model(district):
        """
        Call this daily (e.g., via a cron job) to keep the model fresh.
        POST body: {"rainfall": 45.2}
        """
        body = request.get_json(silent=True) or {}
        new_value = float(body.get("rainfall", 0))
        model = get_model(district)
        model.update_with_new_data(new_value)
        return jsonify({
            "success": True,
            "district": district,
            "message": f"Model updated with rainfall={new_value}mm",
        })

    # ── 4. Model status dashboard ────────────────────────────────────────────
    @app.route("/model_status")
    def model_status():
        """Shows training status for all loaded hybrid models."""
        from hybrid_model import _MODEL_REGISTRY
        status = {}
        for district, model in _MODEL_REGISTRY.items():
            status[district] = {
                "is_trained": model.is_trained,
                "arima_available": bool(model.arima.last_train_data is not None),
                "lstm_available": model.lstm.is_fitted,
                "training_runs": len(model.training_history),
                "last_trained": model.training_history[-1]["timestamp"]
                    if model.training_history else None,
            }
        return jsonify({"models": status, "total": len(status)})


# ── RULE-BASED FALLBACK (used when model not yet trained) ────────────────────
def _rule_based_fallback(district, rainfall, affected_area, population):
    """Simple rule-based prediction used as fallback."""
    if rainfall < 50:
        risk_level, risk_score, risk_color = "Low", 20, "#22c55e"
    elif rainfall < 150:
        risk_level, risk_score, risk_color = "Moderate", 45, "#f59e0b"
    elif rainfall < 300:
        risk_level, risk_score, risk_color = "High", 70, "#ef4444"
    else:
        risk_level, risk_score, risk_color = "Extreme", 90, "#7c3aed"

    return {
        "district": district,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "current_rainfall_mm": rainfall,
        "affected_area_km2": affected_area,
        "affected_population_estimate": int(population * risk_score / 250),
        "shelters_needed": max(1, int(population * risk_score / 125000)),
        "forecast_7day_mm": [0] * 7,
        "total_forecast_7day_mm": 0,
        "model_type": "Rule-Based (Hybrid model training in progress)",
        "timestamp": str(__import__("datetime").datetime.now()),
        "recommendations": [],
    }