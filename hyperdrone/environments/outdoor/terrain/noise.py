"""Noise system ported from threejs-worldgen.

Deterministic, seeded noise functions for procedural terrain generation.
All functions are vectorised over NumPy arrays.
"""

from __future__ import annotations

import numpy as np


def hash4(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """32-bit integer hash of four integer arrays."""
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    c = c.astype(np.int64)
    d = d.astype(np.int64)

    def imul(x: np.ndarray, y: int) -> np.ndarray:
        """Multiply int64 values and return lower 32 bits as int64."""
        result = (x * y) & 0xFFFFFFFF
        return result.astype(np.int64)

    h = (imul(a, 0x27D4EB2D) ^
         imul(b, 0x165667B1) ^
         imul(c, 0x9E3779B1) ^
         imul(d, 0x85EBCA77))

    h = imul(h ^ np.right_shift(h, 15), 0x85EBCA6B)
    h = imul(h ^ np.right_shift(h, 13), 0xC2B2AE35)

    result = (h ^ np.right_shift(h, 16)) & 0xFFFFFFFF
    return result.astype(np.int64)


def hash01(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Hash of four integer arrays mapped to [0, 1)."""
    return hash4(a, b, c, d) / 4294967296.0


def seed_of(seed: int | str) -> int:
    """Turn any string or number into a 32-bit seed."""
    if isinstance(seed, str):
        h = 2166136261
        for ch in seed:
            h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
        return h & 0xFFFFFFFF
    else:
        return int(seed) & 0xFFFFFFFF


def mulberry32(seed: int) -> np.ndarray:
    """Mulberry32: a small, fast, seeded PRNG returning floats in [0, 1)."""
    seed_arr = np.array([int(seed)], dtype=np.int64)

    def gen(size: int) -> np.ndarray:
        a = seed_arr.copy()
        result = np.empty(size, dtype=np.float64)
        for i in range(size):
            a = (a + 0x6D2B79F5) & 0xFFFFFFFF
            t = np.int64(np.uint32(a ^ (a >> 15)) * (1 | a))
            t = np.int64(np.uint32(t ^ (t >> 13)) * (61 | t))
            t = np.int64(np.uint32(t ^ (t >> 14)))
            result[i] = float(np.uint32(t)) / 4294967296.0
        return result

    return gen(1)


def gradient_noise(
    seed: int,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """3D gradient noise with analytic derivatives (after Iñigo Quilez)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    ix = np.floor(x).astype(np.int64)
    iy = np.floor(y).astype(np.int64)
    iz = np.floor(z).astype(np.int64)

    wx = x - ix
    wy = y - iy
    wz = z - iz

    # Quintic interpolation
    ux = wx ** 3 * (wx * (wx * 6 - 15) + 10)
    uy = wy ** 3 * (wy * (wy * 6 - 15) + 10)
    uz = wz ** 3 * (wz * (wz * 6 - 15) + 10)

    # Gradient directions
    gradients = np.array([
        [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
        [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
        [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
        [1, 1, 0], [-1, 1, 0], [0, -1, 1], [0, -1, -1]
    ], dtype=np.float64)

    def grad(gi: int, dx: np.ndarray, dy: np.ndarray, dz: np.ndarray) -> np.ndarray:
        g = gradients[gi]
        return g[0] * dx + g[1] * dy + g[2] * dz

    # Hash corner indices
    seed_const = np.full_like(ix, seed)
    idx_a = hash4(ix, iy, iz, seed_const) & 15
    idx_b = hash4(ix + 1, iy, iz, seed_const) & 15
    idx_c = hash4(ix, iy + 1, iz, seed_const) & 15
    idx_d = hash4(ix + 1, iy + 1, iz, seed_const) & 15
    idx_e = hash4(ix, iy, iz + 1, seed_const) & 15
    idx_f = hash4(ix + 1, iy, iz + 1, seed_const) & 15
    idx_g = hash4(ix, iy + 1, iz + 1, seed_const) & 15
    idx_h = hash4(ix + 1, iy + 1, iz + 1, seed_const) & 15

    # Convert to int for gradient lookup
    idx_a = np.clip(idx_a, 0, 15).astype(np.int64)
    idx_b = np.clip(idx_b, 0, 15).astype(np.int64)
    idx_c = np.clip(idx_c, 0, 15).astype(np.int64)
    idx_d = np.clip(idx_d, 0, 15).astype(np.int64)
    idx_e = np.clip(idx_e, 0, 15).astype(np.int64)
    idx_f = np.clip(idx_f, 0, 15).astype(np.int64)
    idx_g = np.clip(idx_g, 0, 15).astype(np.int64)
    idx_h = np.clip(idx_h, 0, 15).astype(np.int64)

    # Dot products at corners
    va = grad(int(idx_a.flatten()[0]), wx, wy, wz)
    vb = grad(int(idx_b.flatten()[0]), wx - 1, wy, wz)
    vc = grad(int(idx_c.flatten()[0]), wx, wy - 1, wz)
    vd = grad(int(idx_d.flatten()[0]), wx - 1, wy - 1, wz)
    ve = grad(int(idx_e.flatten()[0]), wx, wy, wz - 1)
    vf = grad(int(idx_f.flatten()[0]), wx - 1, wy, wz - 1)
    vg = grad(int(idx_g.flatten()[0]), wx, wy - 1, wz - 1)
    vh = grad(int(idx_h.flatten()[0]), wx - 1, wy - 1, wz - 1)

    # Trilinear interpolation
    k1 = vb - va
    k2 = vc - va
    k3 = ve - va
    k4 = va - vb - vc + vd
    k5 = va - vc - ve + vg
    k6 = va - vb - ve + vf
    k7 = -va + vb + vc - vd + ve - vf - vg + vh

    result = va + ux * k1 + uy * k2 + uz * k3 + ux * uy * k4 + uy * uz * k5 + uz * ux * k6 + ux * uy * uz * k7

    return result * 0.95


def noise3(
    seed: int,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """3D gradient noise value only (fast path)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    ix = np.floor(x).astype(np.int64)
    iy = np.floor(y).astype(np.int64)
    iz = np.floor(z).astype(np.int64)

    wx = x - ix
    wy = y - iy
    wz = z - iz

    # Quintic interpolation
    ux = wx ** 3 * (wx * (wx * 6 - 15) + 10)
    uy = wy ** 3 * (wy * (wy * 6 - 15) + 10)
    uz = wz ** 3 * (wz * (wz * 6 - 15) + 10)

    # Gradient directions
    gradients = np.array([
        [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
        [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
        [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
        [1, 1, 0], [-1, 1, 0], [0, -1, 1], [0, -1, -1]
    ], dtype=np.float64)

    # Compute gradient vectors at corners
    seed_array = np.full_like(ix, seed)
    idx_a = (hash4(ix, iy, iz, seed_array) & 15).astype(np.int64)
    idx_b = (hash4(ix + 1, iy, iz, seed_array) & 15).astype(np.int64)
    idx_c = (hash4(ix, iy + 1, iz, seed_array) & 15).astype(np.int64)
    idx_d = (hash4(ix + 1, iy + 1, iz, seed_array) & 15).astype(np.int64)
    idx_e = (hash4(ix, iy, iz + 1, seed_array) & 15).astype(np.int64)
    idx_f = (hash4(ix + 1, iy, iz + 1, seed_array) & 15).astype(np.int64)
    idx_g = (hash4(ix, iy + 1, iz + 1, seed_array) & 15).astype(np.int64)
    idx_h = (hash4(ix + 1, iy + 1, iz + 1, seed_array) & 15).astype(np.int64)

    # Dot products using vectorized gradient lookup
    def dot_at_corner(indices: np.ndarray, dx: np.ndarray, dy: np.ndarray, dz: np.ndarray) -> np.ndarray:
        """Compute gradient dot product for a set of indices."""
        indices_flat = indices.flatten()
        dx_flat = dx.flatten()
        dy_flat = dy.flatten()
        dz_flat = dz.flatten()

        result_flat = np.zeros_like(dx_flat)

        for gi in range(16):
            mask = (indices_flat == gi)
            if mask.any():
                g = gradients[gi]
                result_flat[mask] = g[0] * dx_flat[mask] + g[1] * dy_flat[mask] + g[2] * dz_flat[mask]

        return result_flat.reshape(indices.shape)

    va = dot_at_corner(idx_a, wx, wy, wz)
    vb = dot_at_corner(idx_b, wx - 1, wy, wz)
    vc = dot_at_corner(idx_c, wx, wy - 1, wz)
    vd = dot_at_corner(idx_d, wx - 1, wy - 1, wz)
    ve = dot_at_corner(idx_e, wx, wy, wz - 1)
    vf = dot_at_corner(idx_f, wx - 1, wy, wz - 1)
    vg = dot_at_corner(idx_g, wx, wy - 1, wz - 1)
    vh = dot_at_corner(idx_h, wx - 1, wy - 1, wz - 1)

    # Trilinear interpolation
    x00 = va + ux * (vb - va)
    x10 = vc + ux * (vd - vc)
    x01 = ve + ux * (vf - ve)
    x11 = vg + ux * (vh - vg)

    y0 = x00 + uy * (x10 - x00)
    y1 = x01 + uy * (x11 - x01)

    result = (y0 + uz * (y1 - y0)) * 0.95

    return result


def fbm(
    seed: int,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    octaves: int = 4,
    frequency: float = 0.01,
    lacunarity: float = 2.0,
    gain: float = 0.5,
) -> np.ndarray:
    """Fractal Brownian Motion: sum of gradient-noise octaves."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    result = np.zeros_like(x)
    amplitude = 1.0
    norm = 0.0
    freq = frequency

    for i in range(octaves):
        result += amplitude * noise3(seed + i * 1013, x * freq, y * freq, z * freq)
        norm += amplitude
        amplitude *= gain
        freq *= lacunarity

    return result / norm


def ridged(
    seed: int,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    octaves: int = 4,
    frequency: float = 0.01,
    lacunarity: float = 2.0,
    gain: float = 0.5,
    sharpness: float = 2.0,
) -> np.ndarray:
    """Ridged multifractal (Musgrave)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    result = np.zeros_like(x)
    amplitude = 1.0
    norm = 0.0
    weight = np.ones_like(x)
    freq = frequency

    for i in range(octaves):
        n = 1.0 - np.abs(noise3(seed + i * 2027, x * freq, y * freq, z * freq))
        n = np.maximum(n, 0.0) ** sharpness * weight
        weight = np.minimum(1.0, np.maximum(0.0, n * 1.5))
        result += n * amplitude
        norm += amplitude
        amplitude *= gain
        freq *= lacunarity

    return result / norm


def eroded_fbm(
    seed: int,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    octaves: int = 4,
    frequency: float = 0.01,
    lacunarity: float = 2.0,
    gain: float = 0.5,
    erosion: float = 1.0,
) -> np.ndarray:
    """Eroded fBm (Iñigo Quilez)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    result = np.zeros_like(x)
    amplitude = 1.0
    norm = 0.0
    freq = frequency

    # Track accumulated slope
    dx = np.zeros_like(x)
    dy = np.zeros_like(x)
    dz = np.zeros_like(x)

    for i in range(octaves):
        gx = np.asarray(x * freq, dtype=np.float64)
        gy = np.asarray(y * freq, dtype=np.float64)
        gz = np.asarray(z * freq, dtype=np.float64)

        # Approximate slope from finite differences
        eps = 1e-4
        nx = noise3(seed + i * 3001, gx + eps, gy, gz) - noise3(seed + i * 3001, gx - eps, gy, gz)
        ny = noise3(seed + i * 3001, gx, gy + eps, gz) - noise3(seed + i * 3001, gx, gy - eps, gz)
        nz = noise3(seed + i * 3001, gx, gy, gz + eps) - noise3(seed + i * 3001, gx, gy, gz - eps)
        dx += nx / (2 * eps)
        dy += ny / (2 * eps)
        dz += nz / (2 * eps)

        slope = dx * dx + dy * dy + dz * dz
        value = noise3(seed + i * 3001, gx, gy, gz)

        result += (amplitude * value) / (1.0 + erosion * slope)
        norm += amplitude
        amplitude *= gain
        freq *= lacunarity

    return result / norm


def warp(
    seed: int,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    frequency: float = 0.01,
    amount: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Domain warp: offsets point by fbm fields."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    o1 = {'octaves': 2, 'frequency': frequency, 'lacunarity': 2.0, 'gain': 0.5}
    o2 = {'octaves': 2, 'frequency': frequency, 'lacunarity': 2.0, 'gain': 0.5}
    o3 = {'octaves': 2, 'frequency': frequency, 'lacunarity': 2.0, 'gain': 0.5}

    wx = x + amount * fbm(seed + 17, x, y, z, **o1)
    wy = y + amount * fbm(seed + 31, x + 5.2, y + 1.3, z + 7.1, **o2)
    wz = z + amount * fbm(seed + 47, x - 3.7, y + 9.2, z - 2.8, **o3)

    return wx, wy, wz


def cellular(
    seed: int,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    jitter: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cellular (Worley) noise: distance to nearest feature point per cell."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    ix = np.floor(x).astype(np.int64)
    iy = np.floor(y).astype(np.int64)
    iz = np.floor(z).astype(np.int64)

    distance = np.full_like(x, np.inf)
    px = np.zeros_like(x)
    py = np.zeros_like(x)
    pz = np.zeros_like(x)
    random_val = np.zeros_like(x)

    # Check neighboring cells (-1 to +1)
    for ki in range(-1, 2):
        for ji in range(-1, 2):
            for ii in range(-1, 2):
                cx = ix + ii
                cy = iy + ji
                cz = iz + ki

                # Feature point position
                feature_x = cx + 0.5 + (hash01(cx, cy, cz, np.full_like(cx, seed)) - 0.5) * jitter
                feature_y = cy + 0.5 + (hash01(cx, cy, cz, np.full_like(cx, seed + 1)) - 0.5) * jitter
                feature_z = cz + 0.5 + (hash01(cx, cy, cz, np.full_like(cx, seed + 2)) - 0.5) * jitter

                # Distance to this feature point
                d = np.sqrt((feature_x - x) ** 2 + (feature_y - y) ** 2 + (feature_z - z) ** 2)

                # Update nearest
                mask = d < distance
                distance[mask] = d[mask]
                px[mask] = feature_x[mask]
                py[mask] = feature_y[mask]
                pz[mask] = feature_z[mask]
                random_val[mask] = hash01(cx[mask], cy[mask], cz[mask], np.full_like(cx[mask], seed + 3))

    return distance, px, py, pz


def smoothstep(a: float, b: float, x: np.ndarray) -> np.ndarray:
    """Hermite smoothstep from a to b."""
    x = np.asarray(x, dtype=np.float64)
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lerp(a: float, b: float, t: np.ndarray) -> np.ndarray:
    """Linear interpolation."""
    t = np.asarray(t, dtype=np.float64)
    return a + (b - a) * t


def clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clamp to [lo, hi]."""
    return np.clip(x, lo, hi)


def curve(
    points: list[tuple[float, float]],
    x: np.ndarray,
) -> np.ndarray:
    """Piecewise-linear curve through [x, y] points sorted by x."""
    x = np.asarray(x, dtype=np.float64)

    if len(points) < 2:
        return np.full_like(x, points[0][1] if points else 0.0)

    result = np.zeros_like(x)

    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]

        mask = (x >= x0) & (x <= x1)
        if mask.any():
            x_vals = x[mask]
            t = (x_vals - x0) / (x1 - x0)
            result[mask] = y0 + (y1 - y0) * t

    # Flat beyond ends
    mask_left = x < points[0][0]
    mask_right = x > points[-1][0]
    result[mask_left] = points[0][1]
    result[mask_right] = points[-1][1]

    return result
