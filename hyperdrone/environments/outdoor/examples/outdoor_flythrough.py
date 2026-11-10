"""Outdoor drone flythrough example.

Demonstrates procedural outdoor world generation with terrain,
biomes, scatter objects, and a drone flythrough camera path.

Run: uv run python examples/outdoor_flythrough.py
"""

from __future__ import annotations

import numpy as np

from hyperdrone.environments.outdoor.outdoor import OutdoorWorld
from hyperdrone.environments.outdoor.terrain.biomes import BIOME_MAP


def create_flythrough_path(world: OutdoorWorld, num_frames: int = 256) -> np.ndarray:
    """Create a drone flythrough path above the terrain.

    Args:
        world: The generated outdoor world.
        num_frames: Number of path waypoints.

    Returns:
        Path array of shape (num_frames, 3) with (x, y, z) coordinates.
    """
    t = np.linspace(0, 1, num_frames, dtype=np.float64)

    # Elliptical flythrough path
    radius_x = world.size * 0.35
    radius_y = world.size * 0.35
    height = 50.0 + 20.0 * np.sin(2 * np.pi * t * 3)  # Varying altitude

    x = radius_x * np.cos(2 * np.pi * t)
    y = radius_y * np.sin(2 * np.pi * t)

    # Sample terrain height and add clearance
    z = np.array([world.get_height_at(float(x[i]), float(y[i])) * 100.0 + height[i] + 10.0 for i in range(len(x))])

    path = np.column_stack([x, y, z])
    return path


def create_cameras_from_path(
    path: np.ndarray,
    look_at: np.ndarray,
    fov: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create camera matrices from a flythrough path.

    Args:
        path: Path array of shape (N, 3).
        look_at: Look-at point in world coordinates.
        fov: Field of view in degrees.

    Returns:
        Tuple of (positions, matrices) arrays.
    """
    positions = path.astype(np.float32)
    fov_rad = np.radians(fov).astype(np.float32)

    # Compute look directions
    directions = look_at - positions
    lengths = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.maximum(lengths, 1e-6)

    # Compute up vectors (simplified)
    up = np.array([[0, 1, 0]], dtype=np.float32).repeat(len(path), axis=0)

    # Compute right vectors
    right = np.cross(directions, up)
    right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-6)

    # Recompute true up
    up = np.cross(right, directions)

    # Build view matrices (simplified - returning positions and directions)
    view_data = np.column_stack([positions, directions, up, right, fov_rad])

    return positions, view_data


def main() -> None:
    """Generate outdoor world and print statistics."""
    print('Generating outdoor world...')

    # Generate world
    world = OutdoorWorld(seed=42, resolution=512, size=512.0)
    world.generate()

    # Validate
    if not world.validate():
        print('ERROR: World validation failed!')
        return

    # Print world statistics
    print(f'World generated: {world.size}m resolution {world.resolution}x{world.resolution}')

    # Heightfield stats
    hf = world.heightfield
    print(f'  Height range: [{hf.height.min():.3f}, {hf.height.max():.3f}]')
    unique_biomes = np.unique(hf.biome)
    biome_names = [list(BIOME_MAP.values())[int(i)].id for i in unique_biomes]
    print(f'  Biomes present: {len(unique_biomes)} ({biome_names})')

    # Scatter stats
    scatter = world.scatter
    pos, types, _scales = scatter.get_mesh_data()
    print(f'  Scatter objects: {len(pos)}')
    if len(types) > 0:
        unique_types = np.unique(types)
        print(f'    Types: {unique_types[:5]}...' if len(unique_types) > 5 else f'    Types: {unique_types}')

    # Mesh stats
    mesh_data = world.mesh.get_mesh_data()
    print(f'  Mesh vertices: {mesh_data["vertices"].shape[0]}')
    print(f'  Mesh indices: {mesh_data["indices"].shape[0]}')

    # Create flythrough path
    print('\nCreating flythrough path...')
    path = create_flythrough_path(world, num_frames=128)
    print(f'  Path waypoints: {len(path)}')
    print(f'  Path bounds: x=[{path[:, 0].min():.1f}, {path[:, 0].max():.1f}]')
    print(f'                   y=[{path[:, 1].min():.1f}, {path[:, 1].max():.1f}]')
    print(f'                   z=[{path[:, 2].min():.1f}, {path[:, 2].max():.1f}]')

    # Print sample terrain heights
    print('\nSample terrain heights:')
    for px, py in [(0, 0), (100, 100), (-100, -100), (200, -200)]:
        h = world.get_height_at(float(px), float(py)) * 100.0
        print(f'  ({px:>4}, {py:>4}): {h:.1f}m')

    print('\nOutdoor flythrough example completed!')


if __name__ == '__main__':
    # Import BIOME_MAP for display
    from hyperdrone.environments.outdoor.terrain.biomes import BIOME_MAP

    main()
