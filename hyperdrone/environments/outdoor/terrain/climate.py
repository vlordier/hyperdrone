"""Climate system ported from threejs-worldgen.

Maps latitude/longitude to temperature and moisture
using procedural noise for local variation.
"""

from __future__ import annotations

import numpy as np

from .noise import fbm


def climate_temperature(
    seed: int,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    latitude_scale: float = 0.001,
) -> np.ndarray:
    """Calculate temperature based on latitude (y) with local noise variation.

    Args:
        seed: Random seed for noise generation.
        x: X coordinates (longitude-like).
        y: Y coordinates (latitude-like).
        z: Z coordinates (unused but kept for API consistency).
        latitude_scale: Scale factor for latitude influence.

    Returns:
        Temperature array in degrees Celsius.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    # Latitude-based temperature (0 at equator, -90 at poles)
    lat = np.arctan2(y, x) * 180.0 / np.pi  # Approximate latitude
    base_temp = 30.0 - 60.0 * abs(lat) / 90.0  # 30C at equator, -30C at poles

    # Local variation from noise
    local = fbm(seed, x * latitude_scale, y * latitude_scale, z * latitude_scale, octaves=3, frequency=1.0)
    local = local * 10.0  # +/- 10C variation

    return base_temp + local


def climate_moisture(
    seed: int,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    frequency: float = 0.002,
) -> np.ndarray:
    """Calculate moisture based on noise patterns.

    Args:
        seed: Random seed for noise generation.
        x: X coordinates.
        y: Y coordinates.
        z: Z coordinates.
        frequency: Scale of moisture patterns.

    Returns:
        Moisture array in range [0, 1].
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    # Multi-octave noise for moisture patterns
    moisture = fbm(seed + 100, x * frequency, y * frequency, z * frequency, octaves=5, frequency=1.0)
    moisture = (moisture + 1.0) / 2.0  # Normalize to [0, 1]

    return np.clip(moisture, 0.0, 1.0)


def get_biome(
    temperature: np.ndarray,
    moisture: np.ndarray,
    height: np.ndarray,
    sea_level: float = 0.0,
) -> np.ndarray:
    """Select biome based on temperature, moisture, and height.

    Args:
        temperature: Temperature array in degrees Celsius.
        moisture: Moisture array in range [0, 1].
        height: Height array (normalized).
        sea_level: Sea level threshold for ocean classification.

    Returns:
        Array of biome IDs matching the shape of inputs.
    """
    temperature = np.asarray(temperature, dtype=np.float64)
    moisture = np.asarray(moisture, dtype=np.float64)
    height = np.asarray(height, dtype=np.float64)

    biome = np.zeros_like(temperature, dtype=np.int32)

    # Ocean
    ocean_mask = height < sea_level
    biome[ocean_mask] = 0  # deepOcean

    # Glacier
    glacier_mask = (~ocean_mask) & (temperature < -15)
    biome[glacier_mask] = 8  # glacier

    # Tundra
    tundra_mask = (~ocean_mask) & (~glacier_mask) & (temperature < -5) & (moisture < 0.4)
    biome[tundra_mask] = 7  # tundra

    # Taiga
    taiga_mask = (~ocean_mask) & (~glacier_mask) & (~tundra_mask) & (temperature < 0) & (moisture > 0.3)
    biome[taiga_mask] = 5  # taiga

    # Alpine meadow
    alpine_mask = (~ocean_mask) & (~glacier_mask) & (~tundra_mask) & (~taiga_mask) & (height > 0.7)
    biome[alpine_mask] = 8  # alpineMeadow (reuse index)

    # Mountain peaks
    peaks_mask = (~ocean_mask) & (~glacier_mask) & (height > 0.85)
    biome[peaks_mask] = 9  # mountainPeaks

    # Desert
    desert_mask = (~ocean_mask) & (~glacier_mask) & (~tundra_mask) & (~taiga_mask) & (~alpine_mask) & (~peaks_mask) & (temperature > 20) & (moisture < 0.15)
    biome[desert_mask] = 11  # desert

    # Savanna
    savanna_mask = (~ocean_mask) & (~glacier_mask) & (~tundra_mask) & (~taiga_mask) & (~alpine_mask) & (~peaks_mask) & (~desert_mask) & (temperature > 20) & (moisture < 0.45)
    biome[savanna_mask] = 12  # savanna

    # Grassland
    grassland_mask = (~ocean_mask) & (~glacier_mask) & (~tundra_mask) & (~taiga_mask) & (~alpine_mask) & (~peaks_mask) & (~desert_mask) & (~savanna_mask) & (temperature < 15) & (moisture < 0.5)
    biome[grassland_mask] = 2  # grassland

    # Mediterranean
    med_mask = (~ocean_mask) & (~glacier_mask) & (~tundra_mask) & (~taiga_mask) & (~alpine_mask) & (~peaks_mask) & (~desert_mask) & (~savanna_mask) & (~grassland_mask) & (temperature > 15) & (moisture < 0.35)
    biome[med_mask] = 15  # mediterranean

    # Deciduous forest
    deciduous_mask = (~ocean_mask) & (~glacier_mask) & (~tundra_mask) & (~taiga_mask) & (~alpine_mask) & (~peaks_mask) & (~desert_mask) & (~savanna_mask) & (~grassland_mask) & (~med_mask) & (moisture > 0.5)
    biome[deciduous_mask] = 3  # deciduousForest

    # Jungle
    jungle_mask = (~ocean_mask) & (~glacier_mask) & (~tundra_mask) & (~taiga_mask) & (~alpine_mask) & (~peaks_mask) & (~desert_mask) & (~savanna_mask) & (~grassland_mask) & (~med_mask) & (~deciduous_mask) & (temperature > 22) & (moisture > 0.7)
    biome[jungle_mask] = 14  # jungle

    # Coast/beach
    coast_mask = (~ocean_mask) & (height < sea_level + 0.02)
    biome[coast_mask] = 1  # coast

    # Wetland
    wetland_mask = (~ocean_mask) & (~coast_mask) & (moisture > 0.85) & (height < sea_level + 0.05)
    biome[wetland_mask] = 17  # wetland

    return biome
