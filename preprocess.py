"""
calculate_slope.py
------------------
Calculates slope angle from SRTM elevation GeoTIFF.
Run this ONCE after downloading the Uttarakhand SRTM file.

Usage:
    python calculate_slope.py

Output:
    data/elevation/uttarakhand/slope.tif
"""

import numpy as np
import os

def calculate_slope_from_dem(dem_path, output_path):
    """
    Calculate slope in degrees from a Digital Elevation Model (DEM).
    Uses the standard Horn's method (same as GIS software).
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds

        print(f"Reading DEM from: {dem_path}")

        with rasterio.open(dem_path) as src:
            elevation = src.read(1).astype(float)
            transform = src.transform
            profile = src.profile
            cell_size = abs(transform[0])   # pixel size in degrees

        # Convert cell size from degrees to meters (approximate)
        # 1 degree latitude ≈ 111,000 meters
        cell_size_m = cell_size * 111000

        print(f"DEM shape: {elevation.shape}")
        print(f"Cell size: {cell_size_m:.1f} meters")
        print("Calculating slope...")

        # Compute slope using numpy gradient
        dy, dx = np.gradient(elevation, cell_size_m, cell_size_m)

        # Slope in degrees
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

        # Save output
        profile.update(dtype='float32', count=1)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(slope.astype('float32'), 1)

        print(f"Slope saved to: {output_path}")
        print(f"Slope range: {slope.min():.1f}° to {slope.max():.1f}°")
        print("Done!")
        return True

    except ImportError:
        print("ERROR: rasterio not installed. Run: pip install rasterio")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def get_slope_at_point(lat, lon, slope_tif_path):
    """Get slope in degrees at a specific location."""
    try:
        import rasterio
        with rasterio.open(slope_tif_path) as src:
            row, col = src.index(lon, lat)
            slope = src.read(1)[row, col]
        return float(slope)
    except Exception as e:
        print(f"Could not read slope at ({lat},{lon}): {e}")
        return 20.0   # Default fallback


if __name__ == "__main__":
    # Default paths — change if your files are named differently
    dem_path    = "data/elevation/uttarakhand/srtm_uttarakhand.tif"
    output_path = "data/elevation/uttarakhand/slope.tif"

    if not os.path.exists(dem_path):
        print(f"DEM file not found at: {dem_path}")
        print("Please download the SRTM GeoTIFF from earthexplorer.usgs.gov")
        print("and save it as:", dem_path)
    else:
        calculate_slope_from_dem(dem_path, output_path)