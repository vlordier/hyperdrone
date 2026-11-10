"""
Biome table ported from threejs-worldgen.

The 20 biome definitions with climate-driven placement rules,
ground materials, and terrain shaping parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScatterEntry:
    """Something scattered over a biome: tree, plant, or rock."""
    species: str
    spacing: float
    probability: float
    cluster: float = 0.3


@dataclass
class BiomeGrass:
    """Grass configuration for a biome."""
    preset: str | None = None
    coverage: float = 1.0
    height: float = 1.0


@dataclass
class Biome:
    """A single biome definition."""
    id: str
    name: str
    placement: str  # 'climate', 'ocean', 'coast', 'mangrove', 'wetland', 'alpine', 'peaks', 'glacier', 'volcanic'
    temperature: float = 0.0  # °C at biome center
    moisture: float = 0.5  # 0-1
    color: str = '#808080'  # debug map color
    ground: dict[str, float] = field(default_factory=dict)  # ground material weights
    ground_tint: str = '#ffffff'  # color tint
    shape: str = 'none'  # 'none', 'dunes', 'terraces', 'hummocks', 'flatten', 'crevasses'
    shape_amount: float = 0.0  # meters
    grass: BiomeGrass = field(default_factory=BiomeGrass)
    scatter: list[ScatterEntry] = field(default_factory=list)


# Ground materials in splat-channel order
GROUND_MATERIALS: list[str] = ['grass', 'dryGrass', 'dirt', 'sand', 'rock', 'snow', 'mud']

# Default ground material colors (#rrggbb), multiplied by each biome's groundTint
GROUND_COLORS: dict[str, str] = {
    'grass': '#4f7a2c',
    'dryGrass': '#a99a5a',
    'dirt': '#6b5138',
    'sand': '#d8c393',
    'rock': '#7a756e',
    'snow': '#f2f5f8',
    'mud': '#3f3a2c',
}


def _s(species: str, spacing: float, probability: float, cluster: float = 0.3) -> ScatterEntry:
    """Helper to create a ScatterEntry."""
    return ScatterEntry(species=species, spacing=spacing, probability=probability, cluster=cluster)


def _g(preset: str | None = None, coverage: float = 1.0, height: float = 1.0) -> BiomeGrass:
    """Helper to create a BiomeGrass."""
    return BiomeGrass(preset=preset, coverage=coverage, height=height)


# The 20 built-in biomes
BIOMES: list[Biome] = [
    Biome(
        id='deepOcean',
        name='Deep ocean',
        placement='ocean',
        temperature=15,
        moisture=1.0,
        color='#1d3f73',
        ground={'sand': 0.6, 'rock': 0.4},
        ground_tint='#7f8a8c',
        shape='none',
        shape_amount=0,
        grass=_g(None, 0),
        scatter=[],
    ),
    Biome(
        id='coast',
        name='Beach and coast',
        placement='coast',
        temperature=15,
        moisture=0.5,
        color='#e8d9a0',
        ground={'sand': 1},
        ground_tint='#f2e3bd',
        shape='none',
        shape_amount=0,
        grass=_g('bermudaGrass', 0.25, 0.7),
        scatter=[_s('palm', 14, 0.25, 0.5), _s('pebble', 3, 0.15)],
    ),
    Biome(
        id='grassland',
        name='Temperate grassland',
        placement='climate',
        temperature=12,
        moisture=0.4,
        color='#9bc25b',
        ground={'grass': 0.85, 'dirt': 0.15},
        ground_tint='#ffffff',
        shape='none',
        shape_amount=0,
        grass=_g('kentuckyBluegrass', 1, 1.1),
        scatter=[_s('oak', 40, 0.25, 0.7), _s('shrub', 12, 0.15, 0.6), _s('rock', 18, 0.15, 0.5)],
    ),
    Biome(
        id='deciduousForest',
        name='Temperate deciduous forest',
        placement='climate',
        temperature=12,
        moisture=0.68,
        color='#3f8a3a',
        ground={'grass': 0.4, 'dirt': 0.5, 'mud': 0.1},
        ground_tint='#c9b58a',
        shape='none',
        shape_amount=0,
        grass=_g('fineFescue', 0.55, 0.8),
        scatter=[
            _s('oak', 7, 0.8, 0.6),
            _s('beech', 7, 0.7, 0.6),
            _s('maple', 8, 0.6, 0.6),
            _s('fern', 3, 0.35, 0.7),
            _s('boulder', 25, 0.2, 0.5),
        ],
    ),
    Biome(
        id='temperateRainforest',
        name='Temperate rainforest',
        placement='climate',
        temperature=10,
        moisture=0.93,
        color='#1f5e3d',
        ground={'mud': 0.4, 'grass': 0.4, 'dirt': 0.2},
        ground_tint='#7f9a6a',
        shape='none',
        shape_amount=0,
        grass=_g('perennialRyegrass', 0.6, 0.7),
        scatter=[
            _s('redwood', 10, 0.8, 0.5),
            _s('fir', 8, 0.6, 0.5),
            _s('fern', 2.5, 0.6, 0.6),
            _s('boulder', 15, 0.35, 0.5),
        ],
    ),
    Biome(
        id='taiga',
        name='Boreal forest (taiga)',
        placement='climate',
        temperature=0,
        moisture=0.55,
        color='#2f5f4f',
        ground={'grass': 0.3, 'dirt': 0.5, 'snow': 0.2},
        ground_tint='#a6a483',
        shape='none',
        shape_amount=0,
        grass=_g('fineFescue', 0.45, 0.7),
        scatter=[
            _s('spruce', 6, 0.85, 0.5),
            _s('pine', 7, 0.5, 0.5),
            _s('birch', 10, 0.3, 0.7),
            _s('rock', 14, 0.2, 0.4),
        ],
    ),
    Biome(
        id='tundra',
        name='Tundra',
        placement='climate',
        temperature=-8,
        moisture=0.35,
        color='#93a59a',
        ground={'dryGrass': 0.4, 'dirt': 0.3, 'rock': 0.15, 'snow': 0.15},
        ground_tint='#b7b39a',
        shape='hummocks',
        shape_amount=0.6,
        grass=_g('frostbite', 0.7, 0.45),
        scatter=[
            _s('dwarfShrub', 4, 0.5, 0.6),
            _s('rock', 9, 0.35, 0.4),
            _s('pebble', 3, 0.3, 0.3),
        ],
    ),
    Biome(
        id='glacier',
        name='Glacier and ice sheet',
        placement='glacier',
        temperature=-20,
        moisture=0.4,
        color='#e9f4fb',
        ground={'snow': 1},
        ground_tint='#f4f8ff',
        shape='crevasses',
        shape_amount=2.5,
        grass=_g(None, 0),
        scatter=[],
    ),
    Biome(
        id='alpineMeadow',
        name='Alpine meadow',
        placement='alpine',
        temperature=2,
        moisture=0.5,
        color='#a9c47f',
        ground={'grass': 0.6, 'rock': 0.3, 'dirt': 0.1},
        ground_tint='#d6dcb0',
        shape='none',
        shape_amount=0,
        grass=_g('fineFescue', 0.85, 0.5),
        scatter=[
            _s('boulder', 12, 0.35, 0.5),
            _s('rock', 6, 0.35, 0.4),
            _s('dwarfShrub', 5, 0.25, 0.6),
        ],
    ),
    Biome(
        id='mountainPeaks',
        name='Mountain peaks',
        placement='peaks',
        temperature=-5,
        moisture=0.4,
        color='#8c8a88',
        ground={'rock': 0.8, 'snow': 0.2},
        ground_tint='#9a958e',
        shape='none',
        shape_amount=0,
        grass=_g(None, 0),
        scatter=[
            _s('cliffChunk', 14, 0.35, 0.4),
            _s('scree', 4, 0.3, 0.5),
        ],
    ),
    Biome(
        id='desert',
        name='Hot desert (erg)',
        placement='climate',
        temperature=27,
        moisture=0.04,
        color='#efcf86',
        ground={'sand': 1},
        ground_tint='#f0cc8c',
        shape='dunes',
        shape_amount=9,
        grass=_g(None, 0),
        scatter=[_s('deadSnag', 90, 0.15, 0.3)],
    ),
    Biome(
        id='badlands',
        name='Rocky desert and badlands',
        placement='climate',
        temperature=21,
        moisture=0.14,
        color='#c46a3f',
        ground={'rock': 0.55, 'dirt': 0.3, 'sand': 0.15},
        ground_tint='#c9744a',
        shape='terraces',
        shape_amount=14,
        grass=_g('goldenSavanna', 0.12, 0.6),
        scatter=[
            _s('cactus', 18, 0.35, 0.4),
            _s('thornShrub', 10, 0.25, 0.5),
            _s('boulder', 14, 0.3, 0.5),
        ],
    ),
    Biome(
        id='savanna',
        name='Savanna',
        placement='climate',
        temperature=26,
        moisture=0.36,
        color='#c8b45a',
        ground={'dryGrass': 0.75, 'dirt': 0.25},
        ground_tint='#d7a96a',
        shape='none',
        shape_amount=0,
        grass=_g('goldenSavanna', 1, 1.2),
        scatter=[
            _s('acacia', 45, 0.45, 0.2),
            _s('baobab', 160, 0.15, 0.1),
            _s('thornShrub', 16, 0.2, 0.6),
            _s('rock', 30, 0.1, 0.5),
        ],
    ),
    Biome(
        id='jungle',
        name='Tropical rainforest',
        placement='climate',
        temperature=26,
        moisture=0.92,
        color='#1d7a2a',
        ground={'mud': 0.45, 'grass': 0.35, 'dirt': 0.2},
        ground_tint='#6f8a4a',
        shape='none',
        shape_amount=0,
        grass=_g('stAugustineGrass', 0.7, 1),
        scatter=[
            _s('jungleEmergent', 16, 0.6, 0.4),
            _s('palm', 7, 0.6, 0.5),
            _s('fern', 2.2, 0.7, 0.5),
            _s('shrub', 4, 0.5, 0.5),
        ],
    ),
    Biome(
        id='dryForest',
        name='Tropical dry forest',
        placement='climate',
        temperature=25,
        moisture=0.62,
        color='#7f9a3a',
        ground={'dirt': 0.5, 'dryGrass': 0.4, 'grass': 0.1},
        ground_tint='#b59a64',
        shape='none',
        shape_amount=0,
        grass=_g('zoysiaGrass', 0.6, 0.8),
        scatter=[
            _s('maple', 9, 0.55, 0.5),
            _s('thornShrub', 6, 0.4, 0.5),
            _s('baobab', 120, 0.2, 0.2),
            _s('rock', 20, 0.15, 0.5),
        ],
    ),
    Biome(
        id='mediterranean',
        name='Mediterranean shrubland',
        placement='climate',
        temperature=17,
        moisture=0.3,
        color='#a8a45a',
        ground={'dryGrass': 0.5, 'dirt': 0.3, 'rock': 0.2},
        ground_tint='#c9b483',
        shape='none',
        shape_amount=0,
        grass=_g('buffaloGrass', 0.7, 0.8),
        scatter=[
            _s('olive', 14, 0.5, 0.5),
            _s('shrub', 3.5, 0.55, 0.6),
            _s('cypress', 30, 0.15, 0.6),
            _s('rock', 10, 0.3, 0.5),
        ],
    ),
    Biome(
        id='steppe',
        name='Cold steppe',
        placement='climate',
        temperature=5,
        moisture=0.2,
        color='#b8b07a',
        ground={'dryGrass': 0.7, 'dirt': 0.3},
        ground_tint='#c4b88c',
        shape='none',
        shape_amount=0,
        grass=_g('autumnHaze', 0.9, 0.7),
        scatter=[
            _s('dwarfShrub', 8, 0.25, 0.5),
            _s('rock', 25, 0.15, 0.5),
        ],
    ),
    Biome(
        id='wetland',
        name='Wetland (marsh and swamp)',
        placement='wetland',
        temperature=14,
        moisture=0.9,
        color='#4f7a5c',
        ground={'mud': 0.6, 'grass': 0.4},
        ground_tint='#7d8a62',
        shape='flatten',
        shape_amount=1,
        grass=_g('tallFescue', 0.8, 1.3),
        scatter=[
            _s('reed', 1.6, 0.7, 0.7),
            _s('cypress', 12, 0.4, 0.6),
            _s('deadSnag', 40, 0.2, 0.4),
        ],
    ),
    Biome(
        id='mangrove',
        name='Mangrove',
        placement='mangrove',
        temperature=27,
        moisture=0.95,
        color='#2f6b55',
        ground={'mud': 0.8, 'sand': 0.2},
        ground_tint='#6e6a52',
        shape='flatten',
        shape_amount=0.5,
        grass=_g(None, 0),
        scatter=[_s('mangrove', 6, 0.75, 0.5)],
    ),
    Biome(
        id='volcanic',
        name='Volcanic',
        placement='volcanic',
        temperature=20,
        moisture=0.3,
        color='#3b2f2f',
        ground={'rock': 0.7, 'dirt': 0.3},
        ground_tint='#3c3633',
        shape='none',
        shape_amount=0,
        grass=_g(None, 0),
        scatter=[
            _s('deadSnag', 25, 0.25, 0.5),
            _s('boulder', 10, 0.4, 0.5),
            _s('scree', 4, 0.3, 0.5),
        ],
    ),
]


# Create lookup dictionaries
BIOME_MAP: dict[str, Biome] = {b.id: b for b in BIOMES}
CLIMATE_BIOMES = [b for b in BIOMES if b.placement == 'climate']
