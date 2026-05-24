"""
test_everything.py
------------------
Run this BEFORE running app.py to make sure everything is working.

HOW TO RUN:
  python test_everything.py

If all tests pass → run python app.py
If any test fails → it tells you exactly what to fix
"""

import os
import sys

print("=" * 55)
print("  FLOOD PROJECT — SYSTEM TEST")
print("=" * 55)

errors   = []
warnings = []
passed   = 0

BASE = os.path.join(os.path.expanduser("~"), "Desktop", "flood_project")

# ─────────────────────────────────────────────────────────────
# TEST 1: Check all required files exist
# ─────────────────────────────────────────────────────────────
print("\n[1] Checking project files...")

required_files = {
    "app.py":              os.path.join(BASE, "app.py"),
    "flood_model.py":      os.path.join(BASE, "flood_model.py"),
    "elevation_model.py":  os.path.join(BASE, "elevation_model.py"),
    "migration_model.py":  os.path.join(BASE, "migration_model.py"),
    "shelter.py":          os.path.join(BASE, "shelter.py"),
    "data_loader.py":      os.path.join(BASE, "data_loader.py"),
    "train_model.py":      os.path.join(BASE, "train_model.py"),
    "templates/index.html":os.path.join(BASE, "templates", "index.html"),
}

for name, path in required_files.items():
    if os.path.exists(path):
        print(f"  ✅ {name}")
        passed += 1
    else:
        print(f"  ❌ MISSING: {name}")
        errors.append(f"Missing file: {path}")

# ─────────────────────────────────────────────────────────────
# TEST 2: Check all CSV datasets exist
# ─────────────────────────────────────────────────────────────
print("\n[2] Checking CSV datasets...")

required_csvs = {
    "Maharashtra Rainfall":  "data/rainfall/maharashtra_rainfall.csv",
    "Uttarakhand Rainfall":  "data/rainfall/uttarakhand_rainfall.csv",
    "Maharashtra Census":    "data/census/maharashtra_census.csv",
    "Uttarakhand Census":    "data/census/uttarakhand_census.csv",
    "Maharashtra Elevation": "data/elevation/maharashtra_elevation.csv",
    "Uttarakhand Elevation": "data/elevation/uttarakhand_elevation.csv",
    "Maharashtra Shelters":  "data/shelters/maharashtra_shelters.csv",
    "Uttarakhand Shelters":  "data/shelters/uttarakhand_shelters.csv",
    "Flood History":         "data/flood_history/historical_flood_events.csv",
    "Migration Data":        "data/migration/migration_data.csv",
    "River Levels":          "data/rivers/river_levels.csv",
}

for name, rel_path in required_csvs.items():
    full_path = os.path.join(BASE, rel_path)
    if os.path.exists(full_path):
        size_kb = os.path.getsize(full_path) // 1024
        print(f"  ✅ {name} ({size_kb} KB)")
        passed += 1
    else:
        print(f"  ❌ MISSING: {name}")
        print(f"     Expected at: {full_path}")
        errors.append(f"Missing CSV: {full_path}")

# ─────────────────────────────────────────────────────────────
# TEST 3: Check Python libraries installed
# ─────────────────────────────────────────────────────────────
print("\n[3] Checking Python libraries...")

libs = {
    "flask":        "Flask (web framework)",
    "pandas":       "Pandas (data processing)",
    "numpy":        "NumPy (math)",
    "sklearn":      "Scikit-learn (ML utilities)",
    "statsmodels":  "Statsmodels (ARIMA)",
    "matplotlib":   "Matplotlib (graphs)",
}
optional_libs = {
    "tensorflow":   "TensorFlow (LSTM neural network)",
    "rasterio":     "Rasterio (GeoTIFF elevation files)",
}

for lib, desc in libs.items():
    try:
        __import__(lib)
        print(f"  ✅ {desc}")
        passed += 1
    except ImportError:
        print(f"  ❌ MISSING: {desc}")
        errors.append(f"Missing library: pip install {lib}")

for lib, desc in optional_libs.items():
    try:
        __import__(lib)
        print(f"  ✅ {desc} (optional)")
        passed += 1
    except ImportError:
        print(f"  ⚠️  OPTIONAL MISSING: {desc}")
        warnings.append(f"Optional library missing: pip install {lib}")

# ─────────────────────────────────────────────────────────────
# TEST 4: Try loading a CSV file
# ─────────────────────────────────────────────────────────────
print("\n[4] Testing CSV loading...")

try:
    import pandas as pd
    rain_path = os.path.join(BASE, "data/rainfall/maharashtra_rainfall.csv")
    if os.path.exists(rain_path):
        df = pd.read_csv(rain_path)
        print(f"  ✅ Maharashtra rainfall loaded: {len(df):,} rows, {len(df.columns)} columns")
        print(f"     Columns: {', '.join(df.columns.tolist())}")
        print(f"     Districts: {df['district'].nunique()}")
        passed += 1
    else:
        print("  ⚠️  Rainfall CSV not found — skipping load test")
except Exception as e:
    print(f"  ❌ Failed to load CSV: {e}")
    errors.append(str(e))

# ─────────────────────────────────────────────────────────────
# TEST 5: Test flood model
# ─────────────────────────────────────────────────────────────
print("\n[5] Testing flood model...")

try:
    sys.path.insert(0, BASE)
    from flood_model import predict_flood

    result = predict_flood("Maharashtra", 150, 80, 10)
    assert "flood_severity" in result
    print(f"  ✅ Maharashtra prediction works: severity = {result['flood_severity']}")
    passed += 1

    result2 = predict_flood("Uttarakhand", 90, 1200, 35)
    assert "flood_severity" in result2
    print(f"  ✅ Uttarakhand prediction works: severity = {result2['flood_severity']}, "
          f"landslide = {result2.get('landslide_probability','N/A')}")
    passed += 1
except Exception as e:
    print(f"  ❌ Flood model error: {e}")
    errors.append(str(e))

# ─────────────────────────────────────────────────────────────
# TEST 6: Check trained models exist
# ─────────────────────────────────────────────────────────────
print("\n[6] Checking trained models...")

models_dir = os.path.join(BASE, "models_saved")
if os.path.exists(models_dir):
    model_files = os.listdir(models_dir)
    if model_files:
        print(f"  ✅ {len(model_files)} trained model files found")
        for f in model_files[:5]:
            print(f"     • {f}")
        passed += 1
    else:
        print("  ⚠️  No trained models yet — run: python train_model.py")
        warnings.append("Models not trained yet. Run: python train_model.py")
else:
    print("  ⚠️  models_saved folder not found — run: python train_model.py")
    warnings.append("Run: python train_model.py to train models")

# ─────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print(f"  TEST RESULTS: {passed} passed")
print("=" * 55)

if errors:
    print(f"\n❌ {len(errors)} ERROR(S) — Fix these before running app.py:\n")
    for i, e in enumerate(errors, 1):
        print(f"  {i}. {e}")
    print("\nFIX COMMANDS:")
    missing_libs = [e.replace("Missing library: ","") for e in errors if "Missing library" in e]
    if missing_libs:
        print(f"  pip install {' '.join(missing_libs)}")
    missing_files = [e for e in errors if "Missing file" in e]
    if missing_files:
        print("  → Re-download missing code files from Claude")
    missing_csvs = [e for e in errors if "Missing CSV" in e]
    if missing_csvs:
        print("  → Place downloaded CSV files in correct folders")

if warnings:
    print(f"\n⚠️  {len(warnings)} WARNING(S) — Optional but recommended:")
    for w in warnings:
        print(f"  • {w}")

if not errors:
    print("\n✅ ALL CRITICAL TESTS PASSED!")
    print("\n📌 NEXT STEPS:")
    if any("train" in w.lower() for w in warnings):
        print("  1. Run: python train_model.py   (trains ARIMA-LSTM, ~10 mins)")
        print("  2. Run: python app.py           (starts website)")
    else:
        print("  1. Run: python app.py           (starts website)")
    print("  2. Open browser: http://127.0.0.1:5000")
    print("  3. Select a region and district, enter rainfall → click Predict!")