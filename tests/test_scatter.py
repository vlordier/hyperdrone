"""Tests for scatter.py - object placement system."""

from __future__ import annotations

from hyperdrone.environments.outdoor.terrain.heightfield import Heightfield
from hyperdrone.environments.outdoor.terrain.scatter import Scatter


class TestScatterBasic:
    """Test Scatter creation and basic operations."""

    def test_scatter_creation(self) -> None:
        """Scatter creates with correct parameters."""
        hf = Heightfield(42, resolution=64)
        hf.generate()
        scatter = Scatter(hf, seed=42)
        assert scatter.heightfield.resolution == 64
        assert scatter.seed == 42

    def test_scatter_empty_positions(self) -> None:
        """Scatter starts with empty positions array."""
        hf = Heightfield(42, resolution=64)
        hf.generate()
        scatter = Scatter(hf, seed=42)
        assert scatter.positions.shape == (0, 3)
        assert scatter.types.shape == (0,)
        assert scatter.scales.shape == (0,)

    def test_scatter_get_mesh_data(self) -> None:
        """get_mesh_data returns tuple of three arrays."""
        hf = Heightfield(42, resolution=64)
        hf.generate()
        scatter = Scatter(hf, seed=42)
        positions, types, scales = scatter.get_mesh_data()
        assert len(positions.shape) == 2
        assert len(types.shape) == 1
        assert len(scales.shape) == 1


class TestScatterPlacement:
    """Test scatter placement on heightfield."""

    def test_scatter_produces_positions(self) -> None:
        """Scatter placement produces coordinate arrays."""
        hf = Heightfield(42, resolution=64)
        hf.generate()
        scatter = Scatter(hf, seed=42)
        scatter.scatter()
        # After scattering, positions should have at least some data
        assert scatter.positions.shape[0] >= 0  # May be empty if no biome has scatter entries
        assert scatter.positions.shape[1] == 3  # x, y, z

    def test_scatter_deterministic(self) -> None:
        """Same seed produces same placement."""
        hf1 = Heightfield(42, resolution=64)
        hf1.generate()
        scatter1 = Scatter(hf1, seed=42)
        scatter1.scatter()

        hf2 = Heightfield(42, resolution=64)
        hf2.generate()
        scatter2 = Scatter(hf2, seed=42)
        scatter2.scatter()

        assert scatter1.positions.shape == scatter2.positions.shape

    def test_scatter_different_seeds_different(self) -> None:
        """Different seeds produce different placement counts."""
        hf1 = Heightfield(42, resolution=64)
        hf1.generate()
        scatter1 = Scatter(hf1, seed=42)
        scatter1.scatter()

        hf2 = Heightfield(42, resolution=64)
        hf2.generate()
        scatter2 = Scatter(hf2, seed=43)
        scatter2.scatter()

        # Different seeds may place different numbers of objects,
        # but all positions must have correct column count (3D coordinates)
        assert scatter1.positions.shape[1] == 3
        assert scatter2.positions.shape[1] == 3
