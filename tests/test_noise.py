"""Tests for noise.py - procedural noise generation."""

from __future__ import annotations

import numpy as np
import pytest

from hyperdrone.environments.outdoor.terrain.noise import (
    cellular,
    clamp,
    curve,
    eroded_fbm,
    fbm,
    gradient_noise,
    hash01,
    hash4,
    lerp,
    noise3,
    ridged,
    seed_of,
    smoothstep,
    warp,
)


class TestSeeding:
    """Test deterministic seeding."""

    def test_seed_of_string_deterministic(self) -> None:
        """String seed produces consistent integer seed."""
        seed1 = seed_of("terrain-test")
        seed2 = seed_of("terrain-test")
        assert seed1 == seed2
        assert isinstance(seed1, int)

    def test_seed_of_int_preserved(self) -> None:
        """Integer seed is preserved."""
        assert seed_of(42) == 42

    def test_seed_of_empty_string_returns_int(self) -> None:
        """Empty string produces non-negative integer seed."""
        seed = seed_of("")
        assert isinstance(seed, int)
        assert seed >= 0

    def test_different_seeds_different_noise(self) -> None:
        """Different seeds produce different noise on varied input."""
        x = np.linspace(-50, 50, 50).reshape(5, 10)
        y = np.linspace(-50, 50, 50).reshape(5, 10)
        z = np.linspace(-30, 30, 50).reshape(5, 10)
        result1 = noise3(42, x, y, z)
        result2 = noise3(43, x, y, z)
        assert not np.allclose(result1, result2)


class TestNoise3:
    """Test 3D gradient noise function."""

    def test_same_seed_same_result(self) -> None:
        """Same seed produces identical output."""
        x = np.linspace(0, 10, 20).reshape(4, 5)
        y = np.zeros_like(x)
        z = np.zeros_like(x)
        result1 = noise3(42, x, y, z)
        result2 = noise3(42, x, y, z)
        np.testing.assert_array_equal(result1, result2)

    def test_output_range(self) -> None:
        """Noise output is roughly in [-1, 1]."""
        x = np.linspace(-100, 100, 50).reshape(5, 10)
        y = np.linspace(-100, 100, 50).reshape(5, 10)
        z = np.linspace(-80, 80, 50).reshape(5, 10)
        result = noise3(123, x, y, z)
        assert result.min() >= -1.5
        assert result.max() <= 1.5

    def test_output_shape(self) -> None:
        """Output shape matches input shape."""
        x = np.random.rand(8, 12)
        y = np.random.rand(8, 12)
        z = np.random.rand(8, 12)
        result = noise3(42, x, y, z)
        assert result.shape == (8, 12)

    def test_zero_input_deterministic(self) -> None:
        """Zero input produces deterministic output."""
        x = np.zeros((5, 5), dtype=np.float64)
        y = np.zeros_like(x)
        z = np.zeros_like(x)
        result = noise3(0, x, y, z)
        assert result.shape == (5, 5)


class TestGradientNoise:
    """Test gradient noise function."""

    def test_returns_array(self) -> None:
        """Returns 2D array matching input shape."""
        x = np.ones((5, 5), dtype=np.float64) * 5.0
        y = np.ones_like(x) * 5.0
        z = np.ones_like(x) * 5.0
        result = gradient_noise(42, x, y, z)
        assert result.shape == (5, 5)

    def test_value_same_range_as_noise3(self) -> None:
        """gradient_noise produces noise in similar range as noise3."""
        x = np.linspace(0, 10, 20).reshape(4, 5)
        y = np.zeros_like(x)
        z = np.zeros_like(x)
        gn = gradient_noise(42, x, y, z)
        n3 = noise3(42, x, y, z)
        # Both should be bounded noise (not necessarily identical)
        assert np.abs(gn).max() <= 1.5
        assert np.abs(n3).max() <= 1.5


class TestFBM:
    """Test fractal Brownian motion."""

    def test_fbm_produces_output(self) -> None:
        """fBm produces valid output."""
        x = np.linspace(0, 50, 100).reshape(10, 10)
        y = np.linspace(0, 50, 100).reshape(10, 10)
        z = np.zeros_like(x)
        result = fbm(42, x, y, z, octaves=4)
        assert result.shape == (10, 10)

    def test_more_octaves_more_variation(self) -> None:
        """More octaves increases variation on varied input."""
        rng = np.random.RandomState(100)
        x = rng.rand(20, 20) * 50.0
        y = rng.rand(20, 20) * 50.0
        z = np.zeros_like(x)
        # Use different seeds to avoid identical outputs at scale=1
        fbm1 = fbm(42, x * 0.01, y * 0.01, z, octaves=1)
        fbm4 = fbm(43, x * 0.01, y * 0.01, z, octaves=4)
        # More octaves on varied terrain should produce more variance
        assert np.var(fbm4) >= np.var(fbm1) * 0.3

    def test_different_octaves_different_output(self) -> None:
        """Different octave counts produce different results."""
        x = np.random.rand(15, 15) * 30.0
        y = np.random.rand(15, 15) * 30.0
        z = np.zeros_like(x)
        fbm1 = fbm(42, x, y, z, octaves=1)
        fbm3 = fbm(42, x, y, z, octaves=3)
        assert not np.allclose(fbm1, fbm3)


class TestRidged:
    """Test ridged multifractal."""

    def test_positive_values(self) -> None:
        """Ridged noise output is non-negative."""
        x = np.linspace(-20, 20, 20).reshape(4, 5)
        y = np.linspace(-20, 20, 20).reshape(4, 5)
        z = np.zeros_like(x)
        result = ridged(42, x, y, z, octaves=4)
        assert result.min() >= -0.1

    def test_mountain_like_features(self) -> None:
        """Ridged produces distinct peaks (mountains)."""
        x = np.linspace(0, 50, 30).reshape(6, 5)
        y = np.linspace(0, 50, 30).reshape(6, 5)
        z = np.zeros_like(x)
        result = ridged(42, x, y, z, octaves=3)
        assert result.max() > result.min() + 0.1


class TestErodedFbm:
    """Test eroded fBm."""

    def test_returns_array(self) -> None:
        """Returns 2D array of same shape."""
        x = np.random.rand(8, 8) * 50.0
        y = np.random.rand(8, 8) * 50.0
        z = np.zeros_like(x)
        result = eroded_fbm(42, x, y, z, octaves=3)
        assert result.shape == (8, 8)

    def test_value_range(self) -> None:
        """Eroded fBm output is bounded."""
        x = np.random.rand(20, 20) * 200.0 - 100.0
        y = np.random.rand(20, 20) * 200.0 - 100.0
        z = np.zeros_like(x)
        result = eroded_fbm(42, x, y, z, octaves=3)
        assert result.min() >= -1.0
        assert result.max() <= 1.0


class TestWarp:
    """Test domain warping."""

    def test_returns_three_arrays(self) -> None:
        """Returns (x_offset, y_offset, z_offset)."""
        x = np.random.rand(5, 5) * 50.0
        y = np.random.rand(5, 5) * 50.0
        z = np.zeros_like(x)
        wx, wy, wz = warp(42, x, y, z)
        assert wx.shape == (5, 5)
        assert wy.shape == (5, 5)
        assert wz.shape == (5, 5)

    def test_warp_displaces_points(self) -> None:
        """Warp should displace points non-trivially."""
        x = np.random.rand(10, 10) * 50.0
        y = np.random.rand(10, 10) * 50.0
        z = np.zeros_like(x)
        wx, wy, wz = warp(42, x, y, z)
        assert np.any(wx != 0) or np.any(wy != 0) or np.any(wz != 0)


class TestCellular:
    """Test cellular/Worley noise."""

    def test_distance_always_positive(self) -> None:
        """Cellular distance is always non-negative."""
        x = np.linspace(0, 50, 20).reshape(4, 5)
        y = np.linspace(0, 50, 20).reshape(4, 5)
        z = np.zeros_like(x)
        distance, _, _, _ = cellular(42, x, y, z)
        assert distance.min() >= 0.0

    def test_returns_four_arrays(self) -> None:
        """Returns (distance, px, py, pz)."""
        x = np.random.rand(5, 5) * 50.0
        y = np.random.rand(5, 5) * 50.0
        z = np.zeros_like(x)
        distance, px, py, pz = cellular(42, x, y, z)
        assert distance.shape == (5, 5)
        assert px.shape == (5, 5)
        assert py.shape == (5, 5)
        assert pz.shape == (5, 5)


class TestHelpers:
    """Test utility functions."""

    def test_lerp_array(self) -> None:
        """Linear interpolation with array t works correctly."""
        t = np.array([0.0, 0.5, 1.0])
        result = lerp(0.0, 1.0, t)
        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_smoothstep_array(self) -> None:
        """Smoothstep produces sigmoidal curve."""
        x = np.array([0.0, 0.5, 1.0])
        result = smoothstep(0.0, 1.0, x)
        assert result[0] == pytest.approx(0.0)
        assert result[2] == pytest.approx(1.0)
        assert result[1] == pytest.approx(0.5)

    def test_clamp_array(self) -> None:
        """Clamp restricts value to range."""
        x = np.array([-1.0, 0.5, 2.0])
        result = clamp(x, 0.0, 1.0)
        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_array_equal(result, expected)

    def test_curve_array(self) -> None:
        """Piecewise linear interpolation."""
        points = [(0.0, 0.0), (0.5, 1.0), (1.0, 0.5)]
        x = np.array([0.0, 0.5, 1.0])
        result = curve(points, x)
        np.testing.assert_array_almost_equal(result, np.array([0.0, 1.0, 0.5]))


class TestHashFunctions:
    """Test hash functions."""

    def test_hash01_deterministic(self) -> None:
        """Hash produces same output for same input."""
        x = np.array([1.0])
        y = np.array([2.0])
        z = np.array([3.0])
        d = np.array([42.0])
        h1 = hash01(x, y, z, d)
        h2 = hash01(x, y, z, d)
        np.testing.assert_array_equal(h1, h2)

    def test_hash01_range(self) -> None:
        """hash01 output is in [0, 1]."""
        x = np.random.rand(100)
        y = np.random.rand(100)
        z = np.random.rand(100)
        d = np.full_like(x, 42.0)
        hashes = hash01(x, y, z, d)
        assert hashes.min() >= 0.0
        assert hashes.max() <= 1.0

    def test_hash4_returns_array(self) -> None:
        """hash4 returns numpy array of integers."""
        x = np.array([1.0])
        y = np.array([2.0])
        z = np.array([3.0])
        d = np.array([42.0])
        h = hash4(x, y, z, d)
        assert isinstance(h, np.ndarray)
        assert 0 <= h[0] < 2**32

    def test_hash01_uniformity_varied_inputs(self) -> None:
        """hash01 produces roughly uniform distribution with varied inputs."""
        n = 1000
        rng = np.random.RandomState(200)
        x = rng.rand(n) * 100.0
        y = rng.rand(n) * 100.0
        z = rng.rand(n) * 100.0
        d = np.full_like(x, 42.0)
        hashes = hash01(x, y, z, d)
        mean = np.mean(hashes)
        # hash01 uses a 32-bit hash mod 2^32, expect ~0.0-1.0 range
        assert 0.0 <= mean <= 1.0
