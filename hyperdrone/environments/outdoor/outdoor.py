"""Outdoor world composition module.

Combines heightfield, scatter, and mesh generation into a complete
outdoor world suitable for drone simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hyperdrone.environments.outdoor.terrain.heightfield import Heightfield
from hyperdrone.environments.outdoor.terrain.mesh import TerrainMesh
from hyperdrone.environments.outdoor.terrain.scatter import Scatter


@dataclass
class OutdoorWorld:
    """Complete outdoor world with terrain, scatter, and mesh data.

    Args:
        seed: Random seed for deterministic generation.
        resolution: Grid resolution for terrain (default 512).
        size: Physical size in meters (default 512).
    """

    seed: int
    resolution: int = 512
    size: float = 512.0

    heightfield: Heightfield = field(init=False, repr=False)
    scatter: Scatter = field(init=False, repr=False)
    mesh: TerrainMesh = field(init=False, repr=False)
    generated: bool = field(default=False, repr=False)

    def generate(self) -> OutdoorWorld:
        """Generate the complete outdoor world.

        Returns:
            self for method chaining.
        """
        self.heightfield = Heightfield(self.seed, self.resolution, self.size)
        self.heightfield.generate()

        self.scatter = Scatter(self.heightfield, self.seed)
        self.scatter.scatter()

        self.mesh = TerrainMesh(self.heightfield)
        self.mesh.generate()

        self.generated = True
        return self

    def validate(self) -> bool:
        """Validate that the world was generated correctly.

        Returns:
            True if all data is present and valid.
        """
        if not self.generated:
            return False

        hf = self.heightfield
        scatter = self.scatter
        mesh_data = self.mesh.get_mesh_data()

        # Check heightfield
        if hf.height.size == 0:
            return False
        if hf.biome.size == 0:
            return False
        if hf.ground.size == 0:
            return False

        # Check scatter
        if scatter.positions.shape[0] == 0:
            return False

        # Check mesh
        if mesh_data['vertices'].shape[0] == 0:
            return False

        return mesh_data['indices'].shape[0] != 0

    def get_height_at(self, x: float, y: float) -> float:
        """Get terrain height at world coordinates.

        Args:
            x: X coordinate in world space.
            y: Y coordinate in world space.

        Returns:
            Height at the specified location.
        """
        return self.heightfield.get_height_at(x, y)
