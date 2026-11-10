"""Heightfield generation for outdoor terrain.

Ported from threejs-worldgen. Generates heightmaps and terrain meshes
from noise functions and biome definitions.
"""

from __future__ import annotations

import numpy as np

from hyperdrone.environments.outdoor.terrain.biomes import BIOME_MAP
from hyperdrone.environments.outdoor.terrain.climate import (
    climate_moisture,
    climate_temperature,
    get_biome,
)
from hyperdrone.environments.outdoor.terrain.noise import (
    cellular,
    fbm,
    ridged,
)


class Heightfield:
    """Heightmap generation with biome and detail shaping.

    Args:
        seed: Random seed for deterministic generation.
        resolution: Grid resolution (resolution x resolution).
        size: Physical size in meters.
    """

    def __init__(self, seed: int, resolution: int = 512, size: float = 512.0) -> None:
        self.seed = seed
        self.resolution = resolution
        self.size = size
        self.height: np.ndarray = np.empty((resolution, resolution), dtype=np.float32)
        self.biome: np.ndarray = np.empty((resolution, resolution), dtype=np.int32)
        self.terrain: np.ndarray = np.empty((resolution, resolution), dtype=np.float32)
        self.ground: np.ndarray = np.empty((resolution, resolution, 7), dtype=np.float32)

    def generate(self) -> Heightfield:
        """Generate the complete heightfield with biome, height, and ground data.

        Returns:
            self for method chaining.
        """
        self._generate_coords()
        self._generate_temperature()
        self._generate_moisture()
        self._generate_base_height()
        self._generate_biomes()
        self._generate_terrain()
        self._generate_ground()
        self._normalize()
        return self

    def _generate_coords(self) -> None:
        """Create normalized coordinate arrays for the heightfield."""
        coords = np.linspace(-1, 1, self.resolution, dtype=np.float64)
        self.x, self.y = np.meshgrid(coords, coords, indexing='ij')
        self.x = self.x * self.size * 0.5
        self.y = self.y * self.size * 0.5
        self.z = np.zeros_like(self.x)

    def _generate_temperature(self) -> None:
        """Generate temperature map from latitude-based climate."""
        self.temp = climate_temperature(self.seed, self.x, self.y, self.z)

    def _generate_moisture(self) -> None:
        """Generate moisture map from procedural noise."""
        self.moisture = climate_moisture(self.seed, self.x, self.y, self.z)

    def _generate_base_height(self) -> None:
        """Generate base terrain height from noise functions."""
        # Multi-scale height generation
        scale1 = self.size * 0.003
        scale2 = self.size * 0.008
        scale3 = self.size * 0.02

        # Continental structure (large scale)
        base = fbm(self.seed, self.x, self.y, self.z, octaves=6, frequency=scale1)

        # Mountain ranges (ridged noise for ridges)
        ridges = ridged(self.seed + 100, self.x, self.y, self.z, octaves=6, frequency=scale2)

        # Detail noise
        detail = fbm(self.seed + 200, self.x, self.y, self.z, octaves=4, frequency=scale3)

        # Combine layers
        self.height = (base * 0.4 + ridges * 0.5 + detail * 0.1) * 100.0

        # Ensure non-negative heights (sea level = 0)
        self.height = np.maximum(self.height, -5.0)

    def _generate_biomes(self) -> None:
        """Assign biome to each cell based on temperature, moisture, and height."""
        height_norm = np.clip((self.height + 5.0) / 205.0, 0.0, 1.0)
        self.biome = get_biome(self.temp, self.moisture, height_norm)

    def _generate_terrain(self) -> None:
        """Apply biome-specific terrain shaping (dunes, terraces, etc.)."""
        self.terrain = np.zeros_like(self.height)
        biome_list = list(BIOME_MAP.values())

        for idx in range(len(biome_list)):
            biome = biome_list[idx]
            if biome.shape == 'none' or biome.shape_amount == 0:
                continue

            mask = self.biome == idx
            if not mask.any():
                continue

            if biome.shape == 'dunes':
                dunes = fbm(self.seed + 300, self.x, self.y, self.z, octaves=3, frequency=self.size * 0.01)
                self.terrain[mask] = dunes[mask] * biome.shape_amount * 0.3

            elif biome.shape == 'terraces':
                t = fbm(self.seed + 400, self.x, self.y, self.z, octaves=2, frequency=self.size * 0.005)
                terraced = np.floor(t * 20) / 20.0  # Quantize to terraces
                self.terrain[mask] = terraced[mask] * biome.shape_amount

            elif biome.shape == 'hummocks':
                h = cellular(self.seed + 500, self.x, self.y, self.z)[0]
                self.terrain[mask] = h[mask] * biome.shape_amount * 0.5

            elif biome.shape == 'crevasses':
                c = fbm(self.seed + 600, self.x, self.y, self.z, octaves=3, frequency=self.size * 0.01)
                crevassed = np.where(c < 0, c * biome.shape_amount, 0)
                self.terrain[mask] = crevassed[mask]

            elif biome.shape == 'flatten':
                # Flatten terrain in wetlands and mangroves
                self.terrain[mask] = 0.0

    def _generate_ground(self) -> None:
        """Generate ground material weights (splat map) for each cell."""
        self.ground = np.zeros((self.resolution, self.resolution, 7), dtype=np.float32)
        biome_list = list(BIOME_MAP.values())

        for idx in range(len(biome_list)):
            biome = biome_list[idx]
            mask = self.biome == idx
            if not mask.any():
                continue

            # Start with noise-based variation
            noise_val = fbm(self.seed + 700, self.x, self.y, self.z, octaves=3, frequency=self.size * 0.01)

            for channel_idx, material in enumerate(['grass', 'dryGrass', 'dirt', 'sand', 'rock', 'snow', 'mud']):
                if material in biome.ground:
                    base_weight = biome.ground[material]
                    variation = np.where(mask, noise_val * 0.1 * base_weight, 0.0)
                    self.ground[mask, channel_idx] = np.clip(base_weight + variation[mask], 0.0, 1.0)

        # Normalize weights
        sums = self.ground.sum(axis=2, keepdims=True)
        sums = np.maximum(sums, 1e-6)
        self.ground = self.ground / sums

    def _normalize(self) -> None:
        """Normalize height to [0, 1] range."""
        h_min = self.height.min()
        h_max = self.height.max()
        if h_max > h_min:
            self.height = (self.height - h_min) / (h_max - h_min)

    def get_height_at(self, x: float, y: float) -> float:
        """Get interpolated height at world coordinates.

        Args:
            x: X coordinate in world space.
            y: Y coordinate in world space.

        Returns:
            Interpolated height value.
        """
        # Convert world coords to grid coords
        gx = ((x / self.size) + 0.5) * (self.resolution - 1)
        gy = ((y / self.size) + 0.5) * (self.resolution - 1)

        ix = int(np.clip(gx, 0, self.resolution - 2))
        iy = int(np.clip(gy, 0, self.resolution - 2))

        fx = gx - ix
        fy = gy - iy

        h00 = self.height[ix, iy]
        h10 = self.height[ix + 1, iy]
        h01 = self.height[ix, iy + 1]
        h11 = self.height[ix + 1, iy + 1]

        h0 = h00 * (1 - fx) + h10 * fx
        h1 = h01 * (1 - fx) + h11 * fx

        return float(h0 * (1 - fy) + h1 * fy)
