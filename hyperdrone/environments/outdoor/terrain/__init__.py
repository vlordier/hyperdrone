"""Procedural terrain generation for outdoor worlds.

Ported from threejs-worldgen (askalice/threejs-worldgen).
Provides noise, biome, climate, heightfield, mesh, and scatter modules.
"""

from .biomes import (
    BIOME_MAP,
    BIOMES,
    CLIMATE_BIOMES,
    GROUND_COLORS,
    GROUND_MATERIALS,
    Biome,
    ScatterEntry,
)
from .climate import (
    climate_moisture,
    climate_temperature,
    get_biome,
)
from .heightfield import Heightfield
from .mesh import TerrainMesh
from .noise import (
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
from .scatter import Scatter

__all__ = [
    'BIOMES',
    'BIOME_MAP',
    'CLIMATE_BIOMES',
    'GROUND_COLORS',
    'GROUND_MATERIALS',
    'Biome',
    'Heightfield',
    'Scatter',
    'ScatterEntry',
    'TerrainMesh',
    'cellular',
    'clamp',
    'climate_moisture',
    'climate_temperature',
    'curve',
    'eroded_fbm',
    'fbm',
    'get_biome',
    'gradient_noise',
    'hash01',
    'hash4',
    'lerp',
    'noise3',
    'ridged',
    'seed_of',
    'smoothstep',
    'warp',
]
