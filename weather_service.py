"""
weather_service.py — Real-time Rainfall & Affected Area Fetcher
Uses Open-Meteo API (free, no API key required)
Covers all Maharashtra and Uttarakhand districts with lat/lon coordinates.
"""

import requests
from datetime import datetime, timedelta
from functools import lru_cache
import math

# ── District coordinates lookup ──────────────────────────────────────────────
DISTRICT_COORDS = {
    # Maharashtra
    "Ahmednagar":   (19.0948, 74.7480),
    "Akola":        (20.7070, 77.0082),
    "Amravati":     (20.9320, 77.7523),
    "Aurangabad":   (19.8762, 75.3433),
    "Beed":         (18.9890, 75.7600),
    "Bhandara":     (21.1667, 79.6500),
    "Buldhana":     (20.5293, 76.1833),
    "Chandrapur":   (19.9615, 79.2961),
    "Dhule":        (20.9020, 74.7749),
    "Gadchiroli":   (20.1809, 80.0000),
    "Gondia":       (21.4600, 80.2000),
    "Hingoli":      (19.7171, 77.1497),
    "Jalgaon":      (21.0077, 75.5626),
    "Jalna":        (19.8350, 75.8800),
    "Kolhapur":     (16.7050, 74.2433),
    "Latur":        (18.4088, 76.5604),
    "Mumbai City":  (18.9388, 72.8354),
    "Mumbai Suburban": (19.1731, 72.9555),
    "Nagpur":       (21.1458, 79.0882),
    "Nanded":       (19.1383, 77.3210),
    "Nandurbar":    (21.3666, 74.2433),
    "Nashik":       (19.9975, 73.7898),
    "Osmanabad":    (18.1860, 76.0389),
    "Palghar":      (19.6967, 72.7697),
    "Parbhani":     (19.2704, 76.7747),
    "Pune":         (18.5204, 73.8567),
    "Raigad":       (18.5158, 73.1167),
    "Ratnagiri":    (16.9902, 73.3120),
    "Sangli":       (16.8524, 74.5815),
    "Satara":       (17.6805, 73.9946),
    "Sindhudurg":   (16.3495, 73.5523),
    "Solapur":      (17.6599, 75.9064),
    "Thane":        (19.2183, 72.9781),
    "Wardha":       (20.7453, 78.6022),
    "Washim":       (20.1072, 77.1472),
    "Yavatmal":     (20.3888, 78.1204),
    # Uttarakhand
    "Almora":       (29.5971, 79.6591),
    "Bageshwar":    (29.8367, 79.7734),
    "Chamoli":      (30.4087, 79.3220),
    "Champawat":    (29.3337, 80.0920),
    "Dehradun":     (30.3165, 78.0322),
    "Haridwar":     (29.9457, 78.1642),
    "Nainital":     (29.3919, 79.4542),
    "Pauri Garhwal":(29.8817, 78.7754),
    "Pithoragarh":  (29.5831, 80.2181),
    "Rudraprayag":  (30.2847, 78.9813),
    "Tehri Garhwal":(30.3783, 78.4322),
    "Udham Singh Nagar": (28.9835, 79.5069),
    "Uttarkashi":   (30.7268, 78.4354),
}

# ── Average district area in km² (used when auto-computing affected area) ──
DISTRICT_AREA_KM2 = {
    # Maharashtra (approximate)
    "Ahmednagar": 17048, "Akola": 5431, "Amravati": 12235, "Aurangabad": 10107,
    "Beed": 10693, "Bhandara": 3895, "Buldhana": 9661, "Chandrapur": 11443,
    "Dhule": 7195, "Gadchiroli": 14412, "Gondia": 5234, "Hingoli": 4526,
    "Jalgaon": 11765, "Jalna": 7718, "Kolhapur": 7685, "Latur": 7157,
    "Mumbai City": 157, "Mumbai Suburban": 446, "Nagpur": 9892, "Nanded": 10528,
    "Nandurbar": 5955, "Nashik": 15530, "Osmanabad": 7569, "Palghar": 5344,
    "Parbhani": 6511, "Pune": 15643, "Raigad": 7152, "Ratnagiri": 8208,
    "Sangli": 8572, "Satara": 10480, "Sindhudurg": 5207, "Solapur": 14895,
    "Thane": 4214, "Wardha": 6309, "Washim": 5152, "Yavatmal": 13582,
    # Uttarakhand (approximate)
    "Almora": 3144, "Bageshwar": 2246, "Chamoli": 8030, "Champawat": 1766,
    "Dehradun": 3088, "Haridwar": 2360, "Nainital": 4251, "Pauri Garhwal": 5438,
    "Pithoragarh": 7110, "Rudraprayag": 1984, "Tehri Garhwal": 4085,
    "Udham Singh Nagar": 2912, "Uttarkashi": 8016,
}


def fetch_realtime_rainfall(district: str) -> dict:
    """
    Fetches real-time and recent rainfall data for a district using Open-Meteo.
    Returns:
        {
          "current_rainfall_mm": float,   # today's accumulated precipitation mm
          "hourly_rainfall_mm": float,    # last 24h sum
          "forecast_7day": list[float],   # 7-day daily precipitation forecast
          "humidity": float,
          "lat": float,
          "lon": float,
          "timestamp": str,
        }
    """
    coords = DISTRICT_COORDS.get(district)
    if not coords:
        raise ValueError(f"District '{district}' not found in coordinate database.")

    lat, lon = coords

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ["precipitation", "relative_humidity_2m", "rain"],
        "hourly": ["precipitation"],
        "daily": ["precipitation_sum"],
        "timezone": "Asia/Kolkata",
        "forecast_days": 7,
        "past_days": 1,          # include yesterday for 24h sum
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise ConnectionError(f"Open-Meteo API error: {e}")

    # Current precipitation (mm right now this hour)
    current_precip = data.get("current", {}).get("precipitation", 0.0) or 0.0
    current_humidity = data.get("current", {}).get("relative_humidity_2m", 60.0) or 60.0

    # Sum last 24 hourly values for 24h rainfall
    hourly_precip = data.get("hourly", {}).get("precipitation", [])
    # last 24 entries
    last_24h = hourly_precip[-24:] if len(hourly_precip) >= 24 else hourly_precip
    rainfall_24h = round(sum(v for v in last_24h if v is not None), 2)

    # 7-day daily forecast
    daily_precip = data.get("daily", {}).get("precipitation_sum", [])
    forecast_7day = [round(v, 2) if v is not None else 0.0 for v in daily_precip[:7]]

    return {
        "current_rainfall_mm": round(current_precip, 2),
        "hourly_rainfall_mm": rainfall_24h,
        "forecast_7day": forecast_7day,
        "humidity": round(current_humidity, 1),
        "lat": lat,
        "lon": lon,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def estimate_affected_area(district: str, rainfall_mm: float, humidity: float = 60.0) -> float:
    """
    Estimates the flood-affected area (km²) based on:
      - Rainfall intensity
      - District total area
      - Humidity modifier

    Uses a sigmoid-style scaling so that:
      - 0–50 mm  → 0–5% of district area
      - 50–150   → 5–25%
      - 150–300  → 25–60%
      - 300+     → 60–85%
    """
    total_area = DISTRICT_AREA_KM2.get(district, 5000)

    # Humidity amplifier: higher humidity = more runoff
    humidity_factor = 1.0 + (max(humidity - 50, 0) / 200.0)  # 1.0 – 1.25

    # Sigmoid-based flood fraction
    # Using a logistic curve: fraction = 1 / (1 + exp(-k*(x - x0)))
    k = 0.02
    x0 = 150  # inflection at 150 mm
    fraction = 1.0 / (1.0 + math.exp(-k * (rainfall_mm - x0)))

    # Scale to 0–85% of district
    affected_fraction = fraction * 0.85 * humidity_factor
    affected_fraction = min(affected_fraction, 0.90)  # cap at 90%

    affected_km2 = round(total_area * affected_fraction, 1)
    return max(affected_km2, 0.0)


def get_district_weather_package(district: str) -> dict:
    """
    One-call function that returns rainfall + auto-computed affected area.
    This is what app.py calls for the auto-fill endpoint.
    """
    weather = fetch_realtime_rainfall(district)

    # Use 24h rainfall for flood risk (more meaningful than instantaneous)
    rainfall_for_model = weather["hourly_rainfall_mm"]

    affected_area = estimate_affected_area(
        district,
        rainfall_for_model,
        weather["humidity"]
    )

    return {
        **weather,
        "rainfall_for_model": rainfall_for_model,
        "affected_area_km2": affected_area,
        "district": district,
    }


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_district = "Pune"
    print(f"\nFetching real-time weather for: {test_district}")
    result = get_district_weather_package(test_district)
    for k, v in result.items():
        print(f"  {k}: {v}")