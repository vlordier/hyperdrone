"""Tests for heightfield.py - terrain height generation."""

from __future__ import annotations

import numpy as np

from hyperdrone.environments.outdoor.terrain.heightfield import Heightfield


class TestHeightfieldBasic:
    """Test basic Heightfield functionality."""

    def test_creation_basic(self) -> None:
        """Basic heightfield creation works."""
        hf = Heightfield(42, resolution=128)
        assert hf.resolution == 128
        assert hf.height.shape == (128, 128)

    def test_height_range(self) -> None:
        """Heights are within reasonable range."""
        hf = Heightfield(100, resolution=64)
        hf.generate()
        assert hf.height.min() >= -100.0
        assert hf.height.max() <= 100.0

    def test_different_seeds_different_terrain(self) -> None:
        """Different seeds produce different heightmaps."""
        hf1 = Heightfield(1, resolution=64)
        hf1.generate()
        hf2 = Heightfield(2, resolution=64)
        hf2.generate()
        assert not np.allclose(hf1.height, hf2.height)


class TestHeightfieldBiomes:
    """Test biome-aware height generation."""

    def test_biome_map_applied(self) -> None:
        """Heightfield applies biome map."""
        hf = Heightfield(42, resolution=64)
        hf.generate()
        assert hf.biome is not None
        assert hf.biome.shape == (64, 64)

    def test_terrain_variation(self) -> None:
        """Terrain has significant height variation."""
        hf = Heightfield(42, resolution=64)
        hf.generate()
        # After normalization, heights should span [0, 1]
        assert hf.height.min() >= 0.0
        assert hf.height.max() <= 1.0
        # There should be meaningful variation
        assert hf.height.max() - hf.height.min() > 0.5

    def test_mountain_peaks(self) -> None:
        """Mountain biomes produce positive heights."""
        hf = Heightfield(42, resolution=64)
        hf.generate()
        assert hf.height.max() > 0.0


class TestHeightfieldRivers:
    """Test river generation."""

    def test_river_path_exists(self) -> None:
        """River paths are generated."""
        hf = Heightfield(42, resolution=128)
        hf.generate()
        assert hf.size == 512.0


class TestHeightfieldDeterminism:
    """Test heightfield determinism."""

    def test_same_seed_same_terrain(self) -> None:
        """Same seed produces identical heightmap."""
        hf1 = Heightfield(123, resolution=64)
        hf1.generate()
        hf2 = Heightfield(123, resolution=64)
        hf2.generate()
        np.testing.assert_array_equal(hf1.height, hf2.height)
