"""Mesh generation for outdoor terrain.

Creates vertices, indices, colors, and UVs for rendering
terrain with biome-based ground textures.
"""

from __future__ import annotations

import numpy as np

from hyperdrone.environments.outdoor.terrain.biomes import BIOME_MAP
from hyperdrone.environments.outdoor.terrain.heightfield import Heightfield


class TerrainMesh:
    """Generates terrain mesh from heightfield data.

    Args:
        heightfield: The heightfield to generate mesh from.
    """

    def __init__(self, heightfield: Heightfield) -> None:
        self.heightfield = heightfield
        self.vertices: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.normals: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.uvs: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self.colors: np.ndarray = np.empty((0, 4), dtype=np.float32)
        self.indices: np.ndarray = np.empty((0,), dtype=np.uint32)

    def generate(self) -> TerrainMesh:
        """Generate the terrain mesh.

        Returns:
            self for method chaining.
        """
        self._generate_vertices()
        self._generate_normals()
        self._generate_uvs()
        self._generate_colors()
        self._generate_indices()
        return self

    def _generate_vertices(self) -> None:
        """Generate vertex positions from heightfield."""
        hf = self.heightfield
        res = hf.resolution
        size = hf.size

        # Create grid coordinates
        x = np.linspace(-size * 0.5, size * 0.5, res, dtype=np.float32)
        y = np.linspace(-size * 0.5, size * 0.5, res, dtype=np.float32)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        zz = (hf.height * 100.0).astype(np.float32)  # Scale height to meters

        # Flatten and stack
        self.vertices = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    def _generate_normals(self) -> None:
        """Compute vertex normals using finite differences."""
        hf = self.heightfield
        res = hf.resolution

        # Reshape height to 2D
        h = hf.height.reshape(res, res)

        # Compute normals using central differences
        dy, dx = np.gradient(h, hf.size / res, hf.size / res)

        # Normalize
        n = np.stack([-dx, -dy, np.ones_like(h)], axis=-1)
        lengths = np.linalg.norm(n, axis=-1, keepdims=True)
        lengths = np.maximum(lengths, 1e-6)
        normals_2d = n / lengths

        # Flatten
        self.normals = normals_2d.reshape(-1, 3).astype(np.float32)

    def _generate_uvs(self) -> None:
        """Generate UV coordinates."""
        hf = self.heightfield
        res = hf.resolution

        # Create UV grid
        u = np.linspace(0, 1, res, dtype=np.float32)
        v = np.linspace(0, 1, res, dtype=np.float32)
        uu, vv = np.meshgrid(u, v, indexing='ij')

        self.uvs = np.column_stack([uu.ravel(), vv.ravel()])

    def _generate_colors(self) -> None:
        """Generate vertex colors from ground material weights."""
        hf = self.heightfield

        # Get ground weights (already normalized)
        ground = hf.ground.reshape(-1, 7)

        # Convert ground weights to RGB colors
        colors = np.zeros((ground.shape[0], 4), dtype=np.float32)

        # Ground material colors
        mat_colors = np.array([
            [0.31, 0.48, 0.17],   # grass
            [0.66, 0.60, 0.35],   # dryGrass
            [0.42, 0.32, 0.22],   # dirt
            [0.85, 0.76, 0.58],   # sand
            [0.48, 0.46, 0.43],   # rock
            [0.95, 0.96, 0.97],   # snow
            [0.25, 0.23, 0.17],   # mud
        ], dtype=np.float32)

        # Multiply by biome ground tint and compute color
        for biome_idx in range(len(hf.biome.flat)):
            biome_id = int(hf.biome.flat[biome_idx])
            biome = list(BIOME_MAP.values())[biome_id] if biome_id < len(BIOME_MAP) else None

            tint = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            if biome and biome.ground_tint:
                hex_color = biome.ground_tint.lstrip('#')
                tint = np.array([
                    int(hex_color[i:i+2], 16) / 255.0
                    for i in (0, 2, 4)
                ], dtype=np.float32)

            rgb = ground[biome_idx] @ mat_colors * tint
            colors[biome_idx, :3] = rgb
            colors[biome_idx, 3] = 1.0  # Alpha

        self.colors = colors

    def _generate_indices(self) -> None:
        """Generate triangle indices for the terrain grid."""
        hf = self.heightfield
        res = hf.resolution

        # Create index grid
        idx = np.arange(res * res, dtype=np.uint32).reshape(res, res)

        # Top-left to bottom-right triangles
        i00 = idx[:-1, :-1]
        i01 = idx[:-1, 1:]
        i10 = idx[1:, :-1]
        i11 = idx[1:, 1:]

        # Two triangles per grid cell
        top_left = np.column_stack([i00.ravel(), i01.ravel(), i10.ravel()])
        bottom_right = np.column_stack([i01.ravel(), i11.ravel(), i10.ravel()])

        self.indices = np.vstack([top_left, bottom_right]).ravel().astype(np.uint32)

    def get_mesh_data(self) -> dict:
        """Get all mesh data.

        Returns:
            Dictionary with vertices, normals, uvs, colors, indices.
        """
        return {
            'vertices': self.vertices,
            'normals': self.normals,
            'uvs': self.uvs,
            'colors': self.colors,
            'indices': self.indices,
        }
