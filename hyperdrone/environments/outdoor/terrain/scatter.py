"""Scatter system ported from threejs-worldgen.

Places scattered objects (trees, rocks, plants) on terrain
based on biome definitions and noise-based filtering.
"""

from __future__ import annotations

import numpy as np

from hyperdrone.environments.outdoor.terrain.biomes import BIOME_MAP
from hyperdrone.environments.outdoor.terrain.heightfield import Heightfield
from hyperdrone.environments.outdoor.terrain.noise import fbm


class Scatter:
    """Places scattered objects on terrain based on biome definitions.

    Args:
        heightfield: The heightfield to scatter objects onto.
        seed: Random seed for deterministic placement.
    """

    def __init__(self, heightfield: Heightfield, seed: int) -> None:
        self.heightfield = heightfield
        self.seed = seed
        self.positions: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.types: np.ndarray = np.empty((0,), dtype=object)
        self.scales: np.ndarray = np.empty((0,), dtype=np.float32)

    def scatter(self) -> Scatter:
        """Scatter objects based on biome definitions.

        Returns:
            self for method chaining.
        """
        hf = self.heightfield
        rng = np.random.default_rng(self.seed)

        all_positions = []
        all_types = []
        all_scales = []

        for biome_idx, _ in enumerate(BIOME_MAP.items()):
            biome = list(BIOME_MAP.values())[biome_idx]
            if not biome.scatter:
                continue

            mask = hf.biome == biome_idx
            if not mask.any():
                continue

            # Create coordinate grid where biome is present
            ys, xs = np.where(mask)
            num_points = len(xs)

            for entry in biome.scatter:
                # Determine number of instances based on probability and area
                num_instances = int(num_points * entry.probability)
                if num_instances <= 0:
                    continue

                # Randomly sample positions
                indices = rng.choice(num_points, size=num_instances, replace=False)
                sampled_x = xs[indices]
                sampled_y = ys[indices]

                # Filter by noise-based probability (cluster effect)
                cluster_noise = fbm(
                    self.seed + biome_idx * 100,
                    sampled_x.astype(np.float64),
                    sampled_y.astype(np.float64),
                    np.zeros_like(sampled_x, dtype=np.float64),
                    octaves=2,
                    frequency=0.05,
                )
                probability_mask = cluster_noise > (1.0 - entry.cluster)

                valid_indices = np.where(probability_mask)[0]
                if len(valid_indices) <= 0:
                    continue

                # Convert grid coordinates to world coordinates
                world_x = (sampled_x[valid_indices] / (hf.resolution - 1)) * hf.size - hf.size * 0.5
                world_y = (sampled_y[valid_indices] / (hf.resolution - 1)) * hf.size - hf.size * 0.5
                world_z = np.array([
                    hf.get_height_at(float(world_x[i]), float(world_y[i])) * 100.0
                    for i in range(len(world_x))
                ])

                # Scale based on spacing (smaller spacing = larger objects)
                scale = 10.0 / entry.spacing

                all_positions.append(np.column_stack([world_x, world_y, world_z]))
                all_types.extend([entry.species] * len(valid_indices))
                all_scales.extend([scale] * len(valid_indices))

        if all_positions:
            self.positions = np.vstack(all_positions).astype(np.float32)
            self.types = np.array(all_types)
            self.scales = np.array(all_scales, dtype=np.float32)

        return self

    def get_mesh_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get scatter mesh data.

        Returns:
            Tuple of (positions, types, scales).
        """
        return self.positions, self.types, self.scales
