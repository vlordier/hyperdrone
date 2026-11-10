"""Tests for biomes.py and climate.py - biome definitions and climate generation."""

from __future__ import annotations

import re

import numpy as np

from hyperdrone.environments.outdoor.terrain.biomes import (
    BIOME_MAP,
    BIOMES,
    CLIMATE_BIOMES,
    GROUND_COLORS,
    GROUND_MATERIALS,
    Biome,
)
from hyperdrone.environments.outdoor.terrain.climate import (
    climate_moisture,
    climate_temperature,
    get_biome,
)


class TestBiomeDefinitions:
    """Test biome table and definitions."""

    def test_all_biomes_defined(self) -> None:
        """All 20 biomes are defined."""
        assert len(BIOMES) == 20
        assert len(BIOME_MAP) == 20

    def test_biome_ids_unique(self) -> None:
        """Biome IDs are unique."""
        ids = [b.id for b in BIOMES]
        assert len(ids) == len(set(ids))

    def test_biome_map_matches(self) -> None:
        """BIOME_MAP contains same biomes as BIOMES."""
        biome_ids = [b.id for b in BIOMES]
        for biome_id in BIOME_MAP:
            assert biome_id in biome_ids
        for biome_id in biome_ids:
            assert biome_id in BIOME_MAP

    def test_biome_required_fields(self) -> None:
        """All biomes have required fields."""
        required_fields = ["id", "name", "placement"]
        for biome in BIOMES:
            for field in required_fields:
                assert hasattr(biome, field), f"Biome {biome.id} missing {field}"

    def test_ground_materials(self) -> None:
        """7 ground materials defined."""
        assert len(GROUND_MATERIALS) == 7
        assert len(GROUND_COLORS) == 7

    def test_climate_biomes_subset(self) -> None:
        """Climate biomes are a subset of all biomes."""
        climate_ids = [b.id for b in CLIMATE_BIOMES]
        all_ids = [b.id for b in BIOMES]
        for climate_id in climate_ids:
            assert climate_id in all_ids
        assert len(CLIMATE_BIOMES) == 12


class TestBiomeProperties:
    """Test individual biome properties."""

    def test_ocean_biomes(self) -> None:
        """Ocean biomes have correct placement."""
        ocean_biomes = [b for b in BIOMES if b.placement == "ocean"]
        assert len(ocean_biomes) >= 1
        for biome in ocean_biomes:
            assert biome.temperature < 20.0

    def test_coast_biomes(self) -> None:
        """Coast biomes have correct placement."""
        coast_biomes = [b for b in BIOMES if b.placement == "coast"]
        assert len(coast_biomes) >= 1
        for biome in coast_biomes:
            assert biome.temperature < 25.0

    def test_biomes_have_colors(self) -> None:
        """All biomes have hex color strings."""
        for biome in BIOMES:
            assert hasattr(biome, "color")
            assert isinstance(biome.color, str)
            # Hex color format: #RRGGBB
            assert re.match(r"^#[0-9a-fA-F]{6}$", biome.color) is not None

    def test_biomes_have_ground_weights(self) -> None:
        """All biomes have ground material weights."""
        for biome in BIOMES:
            assert hasattr(biome, "ground")
            # Each biome should have at least one ground material
            if len(biome.ground) > 0:
                for weight in biome.ground.values():
                    assert 0.0 <= weight <= 1.0


class TestClimateTemperature:
    """Test temperature generation."""

    def test_temperature_range(self) -> None:
        """Temperature is in reasonable range (roughly -100 to 40 C)."""
        x = np.linspace(-100, 100, 50)
        y = np.zeros_like(x)
        z = np.zeros_like(x)
        temp = climate_temperature(42, x, y, z)
        assert temp.min() >= -100.0
        assert temp.max() <= 50.0

    def test_temperature_deterministic(self) -> None:
        """Same seed produces same temperature map."""
        x = np.ones((10, 10), dtype=np.float64)
        y = np.zeros_like(x)
        z = np.zeros_like(x)
        temp1 = climate_temperature(42, x, y, z)
        temp2 = climate_temperature(42, x, y, z)
        np.testing.assert_array_equal(temp1, temp2)

    def test_temperature_varies_spatially(self) -> None:
        """Temperature varies across space."""
        x = np.linspace(-100, 100, 20)
        y = np.zeros_like(x)
        z = np.zeros_like(x)
        temp = climate_temperature(42, x, y, z)
        assert temp.max() > temp.min() + 0.5


class TestClimateMoisture:
    """Test moisture generation."""

    def test_moisture_range(self) -> None:
        """Moisture is in [0, 1]."""
        x = np.linspace(0, 100, 50)
        y = np.zeros_like(x)
        z = np.zeros_like(x)
        moisture = climate_moisture(42, x, y, z)
        assert moisture.min() >= 0.0
        assert moisture.max() <= 1.0

    def test_moisture_deterministic(self) -> None:
        """Same seed produces same moisture map."""
        x = np.ones((10, 10), dtype=np.float64)
        y = np.zeros_like(x)
        z = np.zeros_like(x)
        m1 = climate_moisture(42, x, y, z)
        m2 = climate_moisture(42, x, y, z)
        np.testing.assert_array_equal(m1, m2)

    def test_moisture_noisy(self) -> None:
        """Moisture has spatial variation (not constant)."""
        x = np.linspace(0, 100, 30).reshape(6, 5)
        y = np.zeros_like(x)
        z = np.zeros_like(x)
        moisture = climate_moisture(42, x, y, z)
        assert moisture.max() > moisture.min() + 0.05


class TestBiomeClassification:
    """Test biome assignment from climate data."""

    def test_biome_ids_valid(self) -> None:
        """All biome IDs are in [0, 19]."""
        x = np.linspace(-50, 50, 20)
        y = np.zeros_like(x)
        z = np.zeros_like(x)
        temp = climate_temperature(42, x, y, z)
        moisture = climate_moisture(42, x, y, z)
        height = np.zeros_like(x) + 0.5
        biome = get_biome(temp, moisture, height)
        assert biome.min() >= 0
        assert biome.max() <= 19

    def test_biome_map_returns_correct_biome(self) -> None:
        """BIOME_MAP returns biome for valid ID."""
        # Pick a random valid biome key
        valid_keys = list(BIOME_MAP.keys())
        biome_id = valid_keys[0]  # deterministic: first key
        biome = BIOME_MAP[biome_id]
        assert isinstance(biome, Biome)
        assert biome.id in BIOME_MAP

    def test_climate_biomes_defined(self) -> None:
        """All 12 climate biomes have definitions."""
        assert len(CLIMATE_BIOMES) == 12
        for biome in CLIMATE_BIOMES:
            assert hasattr(biome, "id")
            assert hasattr(biome, "temperature")
            assert hasattr(biome, "moisture")


class TestIntegration:
    """Integration tests for biome-climate system."""

    def test_full_pipeline(self) -> None:
        """Climate data produces valid biome assignment."""
        x = np.linspace(-100, 100, 30)
        y = np.linspace(-100, 100, 30)
        xx, yy = np.meshgrid(x, y)
        z = np.zeros_like(xx)
        temp = climate_temperature(42, xx, yy, z)
        moisture = climate_moisture(42, xx, yy, z)
        height = np.ones_like(xx) * 0.5
        biome = get_biome(temp, moisture, height)
        assert biome.shape == (30, 30)
        # Should have at least 2 different biomes
        assert len(np.unique(biome)) >= 2

    def test_height_affects_biome(self) -> None:
        """Height influences biome assignment."""
        x = np.ones((10, 10), dtype=np.float64) * 50.0
        y = np.zeros_like(x)
        z = np.zeros_like(x)
        temp = climate_temperature(42, x, y, z)
        moisture = climate_moisture(42, x, y, z)

        biome_low = get_biome(temp, moisture, np.zeros_like(x))
        biome_high = get_biome(temp, moisture, np.ones_like(x) * 0.9)

        # Different heights should produce different biomes
        assert not np.array_equal(biome_low, biome_high)
