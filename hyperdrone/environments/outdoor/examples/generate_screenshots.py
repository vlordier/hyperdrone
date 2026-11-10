"""Generate high-resolution screenshots of procedural outdoor worlds.

Creates multiple views of the terrain:
  - Top-down biome map (color-coded)
  - 3D perspective terrain view
  - Heightfield heatmap
  - Scatter overlay (trees/rocks)

Run: uv run python -m hyperdrone.examples.outdoor_screenshots
"""

from __future__ import annotations

import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba

from hyperdrone.environments.outdoor.outdoor import OutdoorWorld
from hyperdrone.environments.outdoor.terrain.biomes import BIOME_MAP


def save_biome_map(world: OutdoorWorld, path: str) -> None:
    """Save a top-down biome color map.

    Args:
        world: The generated outdoor world.
        path: Output file path (.png recommended).
    """
    biome_grid = world.heightfield.biome
    h, w = biome_grid.shape

    # Build RGBA image from biome colors
    img = np.zeros((h, w, 4), dtype=np.float64)
    for idx, biome in enumerate(BIOME_MAP.values()):
        mask = biome_grid == idx
        rgba = to_rgba(biome.color)
        img[mask] = rgba

    plt.figure(figsize=(12, 12), dpi=150)
    plt.imshow(img)
    plt.title('Procedural Outdoor World — Biome Map')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved biome map: {path}')


def save_heightfield_heatmap(world: OutdoorWorld, path: str) -> None:
    """Save a heightfield heatmap.

    Args:
        world: The generated outdoor world.
        path: Output file path.
    """
    height = world.heightfield.height

    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    extent: tuple[float, float, float, float] = (-world.size / 2, world.size / 2, -world.size / 2, world.size / 2)
    im = ax.imshow(height, cmap='terrain', extent=extent)
    ax.set_title('Procedural Outdoor World — Heightfield')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    fig.colorbar(im, ax=ax, label='Normalized height [0, 1]')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved heightfield: {path}')


def save_3d_terrain(world: OutdoorWorld, path: str) -> None:
    """Save a 3D perspective view of the terrain heightfield.

    Args:
        world: The generated outdoor world.
        path: Output file path.
    """
    height = world.heightfield.height  # (res, res)
    size = world.size
    res = height.shape[0]

    # Build coordinate grids
    x = np.linspace(-size / 2, size / 2, res)
    y = np.linspace(-size / 2, size / 2, res)
    xx, yy = np.meshgrid(x, y)
    zz = height * size  # Scale to world units

    fig = plt.figure(figsize=(14, 10), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    # Downsample for visualization
    step = 4
    ax.plot_surface(xx[::step, ::step], yy[::step, ::step], zz[::step, ::step],
                    cmap='terrain', edgecolor='none', alpha=0.9)
    ax.set_title('Procedural Outdoor World — 3D Terrain')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Height (m)')
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved 3D terrain: {path}')


def save_scatter_overlay(world: OutdoorWorld, path: str) -> None:
    """Save terrain with scatter objects overlaid.

    Args:
        world: The generated outdoor world.
        path: Output file path.
    """
    # Heightfield background
    height = world.heightfield.height

    # Scatter positions
    scatter = world.scatter
    positions, _types, _scales = scatter.get_mesh_data()

    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    extent2: tuple[float, float, float, float] = (-world.size / 2, world.size / 2, -world.size / 2, world.size / 2)
    im = ax.imshow(height, cmap='terrain', extent=extent2)

    # Plot scatter objects
    if len(positions) > 0:
        # Scatter objects are in world coords; filter to visible range
        mask = (np.abs(positions[:, 0]) < world.size / 2) & (np.abs(positions[:, 1]) < world.size / 2)
        visible = positions[mask]
        if len(visible) > 0:
            ax.scatter(visible[:, 0], visible[:, 1], c='red', s=1, alpha=0.5, label='Scatter objects')

    ax.set_title('Procedural Outdoor World — Scatter Overlay')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    fig.colorbar(im, ax=ax, label='Height')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved scatter overlay: {path}')


def main() -> None:
    """Generate all screenshots for a seed."""
    seeds = [42, 123, 999]
    output_dir = 'screenshots'

    os.makedirs(output_dir, exist_ok=True)

    for seed in seeds:
        print(f'\nGenerating world (seed={seed})...')
        world = OutdoorWorld(seed=seed, resolution=256, size=512.0)
        world.generate()

        prefix = f'{output_dir}/world_seed{seed}'

        print('  Generating biome map...')
        save_biome_map(world, f'{prefix}_biome.png')

        print('  Generating heightfield heatmap...')
        save_heightfield_heatmap(world, f'{prefix}_height.png')

        print('  Generating 3D terrain view...')
        save_3d_terrain(world, f'{prefix}_3d.png')

        print('  Generating scatter overlay...')
        save_scatter_overlay(world, f'{prefix}_scatter.png')

        print(f'  World stats: height=[{world.heightfield.height.min():.3f}, {world.heightfield.height.max():.3f}], '
              f'scatter={len(world.scatter.positions)}')

    print(f'\nAll screenshots saved to {output_dir}/')


if __name__ == '__main__':
    main()
