"""Generate high-resolution trimesh renders of procedural outdoor worlds.

Creates:
  - 3D perspective terrain render (PNG via PIL)
  - OBJ mesh file for further work in Blender/Maya
  - Heightfield heatmap (PNG via PIL, no matplotlib)

Run: uv run python -m hyperdrone.examples.generate_screenshots
"""

from __future__ import annotations

import os

import numpy as np
import trimesh
from PIL import Image
from PIL.ImageDraw import Draw

from hyperdrone.environments.outdoor.outdoor import OutdoorWorld


def colormap_from_biome_grid(
    biome_grid: np.ndarray,
    colors: dict[int, tuple[float, float, float]],
) -> np.ndarray:
    """Convert a biome grid to an RGB image using a color lookup.

    Args:
        biome_grid: 2D array of biome indices.
        colors: Mapping from biome index to (R, G, B) floats in [0, 1].

    Returns:
        RGB image as uint8 HxWx3.
    """
    h, w = biome_grid.shape
    img = np.zeros((h, w, 3), dtype=np.float32)
    for bidx, (r, g, b) in colors.items():
        mask = biome_grid == bidx
        img[mask, 0] = r
        img[mask, 1] = g
        img[mask, 2] = b
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def save_biome_map(world: OutdoorWorld, path: str) -> None:
    """Save a top-down biome color map as PNG.

    Args:
        world: The generated outdoor world.
        path: Output file path.
    """
    biome_grid = world.heightfield.biome

    # Color palette from BIOMES definitions
    colors: dict[int, tuple[float, float, float]] = {
        0: (0.07, 0.13, 0.30),   # Deep ocean
        1: (0.12, 0.38, 0.60),   # Beach and coast
        2: (0.40, 0.55, 0.20),   # Temperate grassland
        3: (0.15, 0.40, 0.12),   # Temperate deciduous forest
        4: (0.05, 0.30, 0.08),   # Temperate rainforest
        5: (0.10, 0.25, 0.15),   # Boreal forest
        6: (0.70, 0.75, 0.80),   # Tundra
        7: (0.90, 0.92, 0.95),   # Glacier
        8: (0.35, 0.45, 0.25),   # Alpine meadow
        9: (0.50, 0.45, 0.40),   # Mountain peaks
        10: (0.80, 0.70, 0.35),  # Hot desert
        11: (0.65, 0.55, 0.35),  # Rocky desert
        12: (0.55, 0.60, 0.15),  # Savanna
        13: (0.05, 0.35, 0.10),  # Tropical rainforest
        14: (0.15, 0.45, 0.12),  # Tropical dry forest
        15: (0.45, 0.55, 0.25),  # Mediterranean
        16: (0.55, 0.55, 0.30),  # Cold steppe
        17: (0.25, 0.40, 0.20),  # Wetland
        18: (0.20, 0.35, 0.18),  # Mangrove
        19: (0.15, 0.12, 0.10),  # Volcanic
    }

    img = colormap_from_biome_grid(biome_grid, colors)
    pil_img = Image.fromarray(img, mode="RGB")
    pil_img.save(path, "PNG")
    print(f"  Saved biome map: {path}")


def save_heightfield_heatmap(world: OutdoorWorld, path: str) -> None:
    """Save a heightfield heatmap as PNG using PIL (no matplotlib).

    Uses a terrain-like sequential colormap computed from numpy.

    Args:
        world: The generated outdoor world.
        path: Output file path.
    """
    height = world.heightfield.height  # Normalized [0, 1]

    # Build a terrain colormap: deep water -> sand -> grass -> rock -> snow
    def terrain_color(t: np.ndarray) -> np.ndarray:
        """Map normalized height [0, 1] to RGB using terrain gradient."""
        r = np.zeros_like(t)
        g = np.zeros_like(t)
        b = np.zeros_like(t)

        # Deep water (0.0 - 0.15)
        m = (t >= 0.0) & (t < 0.15)
        u = (t[m] - 0.0) / 0.15 if m.any() else 0.0
        r[m] = 0.07 + 0.05 * u
        g[m] = 0.13 + 0.25 * u
        b[m] = 0.30 + 0.30 * u

        # Shallow water / beach (0.15 - 0.25)
        m = (t >= 0.15) & (t < 0.25)
        u = (t[m] - 0.15) / 0.10 if m.any() else 0.0
        r[m] = 0.12 + 0.68 * u
        g[m] = 0.38 + 0.32 * u
        b[m] = 0.60 - 0.25 * u

        # Grass / lowland (0.25 - 0.50)
        m = (t >= 0.25) & (t < 0.50)
        u = (t[m] - 0.25) / 0.25 if m.any() else 0.0
        r[m] = 0.20 + 0.20 * u
        g[m] = 0.60 + 0.00 * u
        b[m] = 0.35 - 0.15 * u

        # Forest / mid elevation (0.50 - 0.70)
        m = (t >= 0.50) & (t < 0.70)
        u = (t[m] - 0.50) / 0.20 if m.any() else 0.0
        r[m] = 0.30 + 0.35 * u
        g[m] = 0.50 - 0.15 * u
        b[m] = 0.25 - 0.20 * u

        # Rock / high elevation (0.70 - 0.85)
        m = (t >= 0.70) & (t < 0.85)
        u = (t[m] - 0.70) / 0.15 if m.any() else 0.0
        r[m] = 0.55 + 0.25 * u
        g[m] = 0.40 + 0.20 * u
        b[m] = 0.25 + 0.60 * u

        # Snow / peaks (0.85 - 1.0)
        m = (t >= 0.85) & (t <= 1.0)
        u = (t[m] - 0.85) / 0.15 if m.any() else 0.0
        r[m] = 0.80 + 0.15 * u
        g[m] = 0.70 + 0.22 * u
        b[m] = 0.60 + 0.35 * u

        return np.stack([r, g, b], axis=-1)

    img = terrain_color(height)
    pil_img = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8), mode="RGB")
    pil_img.save(path, "PNG")
    print(f"  Saved heightfield heatmap: {path}")


def save_3d_terrain(world: OutdoorWorld, path: str) -> None:
    """Save a 3D perspective render of the terrain using trimesh + PIL.

    Uses a fast perspective projection with simple lighting computed per triangle.
    """
    height = world.heightfield.height
    size = world.size
    res = min(height.shape[0], 128)  # Downsample for render speed

    # Build mesh
    x = np.linspace(-size / 2, size / 2, res)
    y = np.linspace(-size / 2, size / 2, res)
    xx, yy = np.meshgrid(x, y)
    zz = height[:res, :res] * size * 0.3

    row = np.arange(res)
    i, j = np.meshgrid(row[:-1], row[:-1], indexing="ij")
    triangles = np.stack([
        np.c_[i.ravel(), j.ravel(), ((i + 1) * res + (j + 1)).ravel()],
        np.c_[i.ravel(), ((i + 1) * res + (j + 1)).ravel(), ((i + 1) * res + j).ravel()],
    ], axis=1).reshape(-1, 3)

    vertices = np.stack([xx.ravel(), zz.ravel(), -yy.ravel()], axis=1)

    # Lighting
    light_dir = np.array([0.5, 0.3, 1.0]) / np.linalg.norm([0.5, 0.3, 1.0])
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    face_normals = face_normals / (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8)
    lighting = np.clip(np.sum(face_normals * light_dir, axis=1), 0, 1)

    # Projection: simple perspective to 2D
    h_img, w_img = 1200, 1600
    fov = 0.6
    center_x, center_y = w_img // 2, h_img // 2
    scale = min(w_img, h_img * 0.8) / size

    # Create image and draw polygons
    img = Image.new("RGB", (w_img, h_img), (15, 20, 35))  # Dark sky
    draw = Draw(img)

    for t_idx in range(len(triangles)):
        v0s = vertices[triangles[t_idx, 0]]
        v1s = vertices[triangles[t_idx, 1]]
        v2s = vertices[triangles[t_idx, 2]]

        # Project to 2D with perspective
        def project(v: np.ndarray) -> tuple[int, int]:
            z = max(v[1], 0.01)
            px = center_x + v[0] * scale * fov / z
            py = center_y - (v[1] * 0.5 + v[2] * scale * 0.3) / z * h_img / (size * 0.3)
            return int(px), int(py)

        p0 = project(v0s)
        p1 = project(v1s)
        p2 = project(v2s)

        # Height-based color with lighting
        avg_height = np.mean([v0s[1], v1s[1], v2s[1]])
        ht_norm = np.clip(avg_height / (size * 0.3), 0, 1)
        shade = lighting[t_idx]
        r = int((0.2 + 0.6 * ht_norm) * shade * 0.9 + 0.1) * 255 // 255
        g = int((0.5 - 0.2 * ht_norm) * shade * 0.9 + 0.1) * 255 // 255
        b = int((0.3 + 0.3 * ht_norm) * shade * 0.9 + 0.05) * 255 // 255
        color = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
        draw.polygon([p0, p1, p2], fill=color)

    img.save(path, "PNG")

    # Also save OBJ for Blender/Maya
    obj_path = path.replace(".png", ".obj")
    trimesh.Trimesh(vertices=vertices, faces=triangles, process=False).export(obj_path, file_type="obj")
    print(f"  Saved 3D render: {path}")
    print(f"  Saved OBJ mesh: {obj_path}")


def main() -> None:
    """Generate renders for multiple seeds."""
    seeds = [42, 123, 999]
    output_dir = "screenshots"
    resolution = 256

    os.makedirs(output_dir, exist_ok=True)

    for seed in seeds:
        print(f"\nGenerating world (seed={seed})...")
        world = OutdoorWorld(seed=seed, resolution=resolution, size=512.0)
        world.generate()

        prefix = f"{output_dir}/world_seed{seed}"

        print("  Biome map...")
        save_biome_map(world, f"{prefix}_biome.png")

        print("  Heightfield heatmap...")
        save_heightfield_heatmap(world, f"{prefix}_height.png")

        print("  3D terrain render...")
        save_3d_terrain(world, f"{prefix}_terrain.png")

        h = world.heightfield.height
        print(f"  World: height=[{h.min():.3f}, {h.max():.3f}], "
              f"biomes={len(np.unique(world.heightfield.biome))}, "
              f"resolution={resolution}x{resolution}")

    print(f"\nDone! All renders in {output_dir}/")


if __name__ == "__main__":
    main()
