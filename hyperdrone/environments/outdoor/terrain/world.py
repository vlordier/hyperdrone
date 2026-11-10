"""World composition: the complete World class from threejs-worldgen.

Combines continents, mountains, hills, volcanoes, rivers, climate,
biome rules, ground materials, detail, and micro-relief into one
(x, z) -> y height function plus biome evaluation.

Ported from AskAlice's threejs-worldgen `world.ts`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from hyperdrone.environments.outdoor.terrain.biomes import BIOMES, GROUND_MATERIALS
from hyperdrone.environments.outdoor.terrain.noise import (
    cellular,
    clamp,
    eroded_fbm,
    fbm,
    lerp,
    ridged,
    seed_of,
    smoothstep,
    warp,
)
from hyperdrone.environments.outdoor.terrain.noise import (
    curve as curve_fn,
)

if TYPE_CHECKING:
    from typing import Any

# ---------------------------------------------------------------------------
# Default world settings (from world.ts DEFAULT_WORLD)
# ---------------------------------------------------------------------------

DEFAULT_CONTINENTS: dict[str, Any] = {
    "scale": 6000,
    "octaves": 5,
    "warp": 0.4,
    "landBias": 0.12,
    "curve": [
        [-1, -140],
        [-0.45, -70],
        [-0.2, -12],
        [-0.06, -1],
        [0.02, 1.5],
        [0.15, 8],
        [0.45, 30],
        [1, 70],
    ],
}

DEFAULT_MOUNTAINS: dict[str, Any] = {
    "height": 950,
    "scale": 2400,
    "octaves": 8,
    "sharpness": 2,
    "rangeScale": 7000,
    "threshold": 0.5,
    "erosionDamping": 0.75,
}

DEFAULT_HILLS: dict[str, Any] = {
    "height": 55,
    "scale": 900,
    "octaves": 6,
    "erosion": 1.5,
}

DEFAULT_RIVERS: dict[str, Any] = {
    "enabled": True,
    "scale": 3400,
    "width": 0.018,
    "valleyWidth": 0.11,
    "depth": 3,
    "valleyDepth": 0.7,
    "maxHeight": 320,
}

DEFAULT_CLIMATE: dict[str, Any] = {
    "baseTemperature": 15,
    "temperatureRange": 16,
    "temperatureScale": 9000,
    "equatorTemperature": 28,
    "poleTemperature": -18,
    "latitudeGradient": 0,
    "lapseRate": 6.5,
    "moistureScale": 5200,
    "moistureBias": 0,
    "rainShadow": 0.35,
    "windDirection": [1, 0],
    "coastalMoisture": 0.12,
    "riverMoisture": 0.25,
}

DEFAULT_RULES: dict[str, Any] = {
    "blend": 0.2,
    "maxBiomes": 4,
    "coastHeight": 1.6,
    "deepWater": 6,
    "treelineTemperature": 2,
    "alpineMinHeight": 350,
    "peakShare": 0.42,
    "glacierTemperature": -14,
    "wetlandMoisture": 0.72,
    "wetlandMaxHeight": 8,
    "mangroveTemperature": 23,
}

DEFAULT_VOLCANOES: dict[str, Any] = {
    "chance": 0.05,
    "regionSize": 14000,
    "radius": 2200,
    "height": 1100,
    "crater": 0.14,
}

DEFAULT_GROUND: dict[str, Any] = {
    "rockSlope": 0.85,
    "snowTemperature": -3,
    "sandBand": 1.2,
}

DEFAULT_DETAIL: dict[str, Any] = {
    "height": 1.6,
    "scale": 48,
    "octaves": 8,
}

DEFAULT_MICRO: dict[str, Any] = {
    "height": 0.05,
    "scale": 0.8,
    "octaves": 7,
}


def _default_options(seed: int | str = 42) -> dict[str, Any]:
    """Return the default world options with the given seed."""
    return {
        "seed": seed,
        "seaLevel": 0,
        "continents": DEFAULT_CONTINENTS,
        "erosion": {"scale": 2600, "octaves": 3},
        "mountains": DEFAULT_MOUNTAINS,
        "hills": DEFAULT_HILLS,
        "detail": DEFAULT_DETAIL,
        "micro": DEFAULT_MICRO,
        "rivers": DEFAULT_RIVERS,
        "climate": DEFAULT_CLIMATE,
        "rules": DEFAULT_RULES,
        "volcanoes": DEFAULT_VOLCANOES,
        "ground": DEFAULT_GROUND,
        "biomes": list(BIOMES),
        "modifiers": [],
    }


# ---------------------------------------------------------------------------
# WorldSample: data returned by sample()
# ---------------------------------------------------------------------------


@dataclass
class WorldSample:
    """Everything known about one point of the world.

    Attributes:
        elevation: Height above the base surface, metres.
        continentalness: -1 (deep ocean) to 1 (deep inland).
        erosion: Erosion field 0 (rugged) to 1 (flat).
        mountain: Mountain contribution 0 to 1.
        temperature: Temperature at this altitude, C.
        moisture: Moisture 0 to 1.
        valley: River valley strength 0 to 1.
        river: River channel strength 0 to 1.
        water_level: Water surface height (sea or river), or NaN where dry.
        biomes: Biome indices, strongest first.
        weights: Biome weights, summing to 1.
    """

    elevation: float = 0.0
    continentalness: float = 0.0
    erosion: float = 0.0
    mountain: float = 0.0
    temperature: float = 0.0
    moisture: float = 0.0
    valley: float = 0.0
    river: float = 0.0
    water_level: float = float("nan")
    biomes: list[int] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# World — the main composition class
# ---------------------------------------------------------------------------

# Override biome placement order (module-level constant)
_OVERRIDE_ORDER: list[str] = [
    "volcanic",
    "ocean",
    "glacier",
    "peaks",
    "alpine",
    "mangrove",
    "coast",
    "wetland",
]


class World:
    """A procedural world: climate, height, rivers, volcanoes, 20 blended biomes.

    Works as an infinite flat world, and is cheap enough to query per vertex,
    per blade of grass, or per tree.

    Usage::

        world = World(seed=42, resolution=512, size=512)
        world.generate()  # in-place: fills elevation, biome arrays

    Args:
        seed: Random seed (int or str).
        resolution: Grid resolution for bulk generation (N by N).
        size: Physical size in metres.
        sea_level: Sea level in metres (default 0).
        continents: Continent noise options dict.
        mountains: Mountain noise options dict.
        hills: Hill noise options dict.
        rivers: River noise options dict.
        climate: Climate options dict.
        rules: Biome override rule options dict.
        volcanoes: Volcano options dict.
        detail: Ground detail options dict.
        micro: Micro-relief options dict.
        ground: Ground material options dict.
        biomes: Biome definitions list (default: the 20 built-in BIOMES).
    """

    def __init__(
        self,
        seed: int | str = 42,
        resolution: int = 512,
        size: float = 512.0,
        sea_level: float = 0,
        continents: dict[str, Any] | None = None,
        mountains: dict[str, Any] | None = None,
        hills: dict[str, Any] | None = None,
        rivers: dict[str, Any] | None = None,
        climate: dict[str, Any] | None = None,
        rules: dict[str, Any] | None = None,
        volcanoes: dict[str, Any] | None = None,
        detail: dict[str, Any] | None = None,
        micro: dict[str, Any] | None = None,
        ground: dict[str, Any] | None = None,
        biomes: list[dict[str, Any]] | None = None,
    ) -> None:
        self.seed = seed_of(seed)
        self.resolution = resolution
        self.size = size

        opts = _default_options(seed)
        if continents:
            opts["continents"] = {**DEFAULT_CONTINENTS, **continents}
        if mountains:
            opts["mountains"] = {**DEFAULT_MOUNTAINS, **mountains}
        if hills:
            opts["hills"] = {**DEFAULT_HILLS, **hills}
        if rivers:
            opts["rivers"] = {**DEFAULT_RIVERS, **rivers}
        if climate:
            opts["climate"] = {**DEFAULT_CLIMATE, **climate}
        if rules:
            opts["rules"] = {**DEFAULT_RULES, **rules}
        if volcanoes:
            opts["volcanoes"] = {**DEFAULT_VOLCANOES, **volcanoes}
        if detail:
            opts["detail"] = {**DEFAULT_DETAIL, **detail}
        if micro:
            opts["micro"] = {**DEFAULT_MICRO, **micro}
        if ground:
            opts["ground"] = {**DEFAULT_GROUND, **ground}
        if biomes:
            opts["biomes"] = biomes
        if sea_level is not None:
            opts["seaLevel"] = sea_level

        self.options: dict[str, Any] = opts
        self._prepare()

        # Pre-allocated arrays for bulk generation
        max_biomes = opts["rules"]["maxBiomes"]
        self.elevation = np.empty((resolution, resolution), dtype=np.float64)
        self.biome = np.empty((resolution, resolution), dtype=np.int32)
        self.biome_weights = np.empty((resolution, resolution, max_biomes), dtype=np.float32)
        self.temp = np.empty((resolution, resolution), dtype=np.float64)
        self.moist = np.empty((resolution, resolution), dtype=np.float64)
        self.continentalness = np.empty((resolution, resolution), dtype=np.float64)
        self.mountain_share = np.empty((resolution, resolution), dtype=np.float64)
        self.valley = np.empty((resolution, resolution), dtype=np.float64)
        self.river = np.empty((resolution, resolution), dtype=np.float64)
        self.water_level = np.empty((resolution, resolution), dtype=np.float64)
        self.erosion = np.empty((resolution, resolution), dtype=np.float64)

    def _prepare(self) -> None:
        """Set up lookup tables from options."""
        self._seed = self.seed
        biomes_list = self.options.get("biomes", BIOMES)

        # Index override biomes by placement type
        self._override_idx: dict[str, int] = {}
        self._climate_idx: list[int] = []
        for i, b in enumerate(biomes_list):
            placement = getattr(b, "placement", "climate")
            if placement == "climate":
                self._climate_idx.append(i)
            elif placement not in self._override_idx:
                self._override_idx[placement] = i

        # Ground material mix per biome
        self._ground_mix: list[dict[str, float]] = []
        for b in biomes_list:
            ground = getattr(b, "ground", {})
            total = sum(ground.get(m, 0) for m in GROUND_MATERIALS)
            mix: dict[str, float] = {}
            if total > 0:
                for m in GROUND_MATERIALS:
                    val = ground.get(m, 0)
                    mix[m] = val / total if total else 0
            self._ground_mix.append(mix)

    # ------------------------------------------------------------------
    # Bulk generation helpers
    # ------------------------------------------------------------------

    def _compute_continents(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray, co: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute continentalness and base terrain."""
        q = warp(
            self._seed + 101,
            x,
            y,
            z,
            1.0 / (co["scale"] * 1.7),
            co["warp"] * co["scale"],
        )
        continental = clamp(
            fbm(self._seed + 1, q[0], q[1], q[2], co["octaves"], 1.0 / co["scale"]) * 1.25
            + co["landBias"],
            -1,
            1,
        )
        self.continentalness[:] = continental
        base = curve_fn(co["curve"], continental)
        land = smoothstep(-0.08, 0.1, continental)
        return base, land, continental

    def _compute_erosion(
        self, eo: dict[str, Any], x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> np.ndarray:
        """Compute erosion field."""
        erosion_field = 0.5 + 0.5 * _spread(
            fbm(self._seed + 2, x, y, z, eo["octaves"], 1.0 / eo["scale"])
        )
        self.erosion[:] = erosion_field
        return erosion_field

    def _compute_mountains(
        self,
        mo: dict[str, Any],
        erosion_field: np.ndarray,
        land: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute mountain terrain."""
        range_field = 0.5 + 0.5 * _spread(fbm(self._seed + 3, x, y, z, 3, 1.0 / mo["rangeScale"]))
        range_mask = (
            smoothstep(mo["threshold"], mo["threshold"] + 0.25, range_field)
            * (1 - erosion_field * mo["erosionDamping"])
            * land
        )
        ridge = ridged(
            self._seed + 4,
            x,
            y,
            z,
            mo["octaves"],
            1.0 / mo["scale"],
            2.0,
            0.5,
            mo["sharpness"],
        )
        mountain_share = ridge * range_mask
        mountains = mo["height"] * mountain_share
        self.mountain_share[:] = mountain_share
        return mountains, mountain_share, range_mask

    def _compute_hills(
        self,
        ho: dict[str, Any],
        erosion_field: np.ndarray,
        land: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        """Compute hill terrain."""
        return (
            ho["height"]
            * eroded_fbm(
                self._seed + 5,
                x,
                y,
                z,
                ho["octaves"],
                1.0 / ho["scale"],
                ho["erosion"],
            )
            * (0.35 + 0.65 * (1 - erosion_field))
            * (0.25 + 0.75 * land)
        )

    def _compute_volcanoes(
        self, vo: dict[str, Any], land: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute volcano terrain."""
        volcano = np.zeros_like(x)
        volcanic = np.zeros_like(x)
        if vo["chance"] > 0:
            cx, cy, cz = x / vo["regionSize"], y / vo["regionSize"], z / vo["regionSize"]
            dist, _, _, rand_val = cellular(self._seed + 6, cx, cy, cz, jitter=0.8)
            cell = np.where(rand_val < vo["chance"], dist * vo["regionSize"], np.inf)
            t = smoothstep(vo["radius"], 0, cell)
            crater = smoothstep(
                vo["radius"] * vo["crater"], vo["radius"] * vo["crater"] * 0.4, cell
            )
            volcano = vo["height"] * (t**1.6 - crater * 0.35) * land
            volcanic = smoothstep(vo["radius"] * 1.15, vo["radius"] * 0.7, cell) * land
        return volcano, volcanic

    def _compute_rivers(
        self,
        ro: dict[str, Any],
        h: np.ndarray,
        base: np.ndarray,
        hills: np.ndarray,
        mountains: np.ndarray,
        volcanic: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        sea: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute river valleys and channels."""
        valley = np.zeros_like(x)
        river = np.zeros_like(x)
        water_level = np.where(h < sea, sea, np.nan)
        if ro["enabled"]:
            rq = warp(
                self._seed + 201,
                x,
                y,
                z,
                1.0 / (ro["scale"] * 0.8),
                ro["scale"] * 0.35,
            )
            rn = np.abs(fbm(self._seed + 7, rq[0], rq[1], rq[2], 3, 1.0 / ro["scale"]))
            height_mask = (
                smoothstep(sea + 0.5, sea + 4, base + hills * 0.5)
                * (1 - smoothstep(ro["maxHeight"] * 0.6, ro["maxHeight"], h))
                * (1 - volcanic)
            )
            valley = smoothstep(ro["valleyWidth"], 0, rn) * height_mask
            river = smoothstep(ro["width"], ro["width"] * 0.35, rn) * height_mask
            self.valley[:] = valley
            self.river[:] = river
            if valley.max() > 0:
                low = base + hills * 0.35 + mountains * 0.15
                surface = np.full_like(h, sea) + np.maximum(
                    0.4, (low - sea) * (1 - ro["valleyDepth"])
                )
                depth_arr = np.full_like(h, ro["depth"])
                h_max = np.minimum(h, surface + 0.6)
                h = np.where(
                    valley > 0,
                    np.where(
                        river > 0,
                        lerp(h, surface - depth_arr, river),
                        lerp(h, h_max, valley),
                    ),
                    h,
                )
                river_arr = np.where(river > 0.02, surface, water_level)
                water_level = np.where((river > 0.02) & (h < surface), river_arr, water_level)
        return valley, river, water_level

    def _compute_climate(
        self,
        cl: dict[str, Any],
        mo: dict[str, Any],
        h: np.ndarray,
        continental: np.ndarray,
        valley: np.ndarray,
        mountains: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute temperature and moisture fields."""
        sea = 0.0  # default
        above = np.maximum(0, h - sea)

        # Temperature (flat world)
        temperature = (
            cl["baseTemperature"]
            + cl["temperatureRange"]
            * _spread(fbm(self._seed + 8, x, y, z, 3, 1.0 / cl["temperatureScale"]))
            + (cl["latitudeGradient"] * z) / 1000
        )
        temperature -= (cl["lapseRate"] * above) / 1000
        self.temp[:] = temperature

        # Moisture
        moisture = (
            0.5
            + 0.55 * _spread(fbm(self._seed + 9, x, y, z, 3, 1.0 / cl["moistureScale"]))
            + cl["moistureBias"]
        )
        moisture += cl["coastalMoisture"] * smoothstep(0.25, -0.02, continental)
        moisture += cl["riverMoisture"] * valley

        # Rain shadow
        if cl["rainShadow"] > 0 and mo["height"] > 0:
            wind = cl["windDirection"]
            wl = np.hypot(wind[0], wind[1]) or 1
            wx, wz = wind[0] / wl, wind[1] / wl
            upwind = np.zeros_like(x)
            for dist in [mo["scale"] * 0.6, mo["scale"] * 1.4]:
                ux = x - wx * dist
                uy = y
                uz = z - wz * dist
                rf = 0.5 + 0.5 * _spread(fbm(self._seed + 3, ux, uy, uz, 3, 1.0 / mo["rangeScale"]))
                rmask = smoothstep(mo["threshold"], mo["threshold"] + 0.25, rf)
                candidate = np.where(
                    rmask > 0.001,
                    mo["height"]
                    * rmask
                    * ridged(
                        self._seed + 4,
                        ux,
                        uy,
                        uz,
                        3,
                        1.0 / mo["scale"],
                        2.0,
                        0.5,
                        mo["sharpness"],
                    ),
                    0,
                )
                upwind = np.maximum(upwind, candidate)
            moisture -= cl["rainShadow"] * clamp((upwind - mountains) / mo["height"], 0, 1)

        moisture = clamp(moisture, 0, 1)
        self.moist[:] = moisture
        return temperature, moisture

    def _assign_biomes(
        self,
        sea: float,
        h: np.ndarray,
        mountain_share: np.ndarray,
        temperature: np.ndarray,
        moisture: np.ndarray,
        erosion_field: np.ndarray,
        volcanic: np.ndarray,
        continental: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assign biomes based on rules and climate."""
        rules = self.options["rules"]
        max_biomes = rules.get("maxBiomes", 4)

        biome_indices: list[np.ndarray] = [
            np.full_like(h, -1, dtype=np.int32) for _ in range(max_biomes)
        ]
        biome_weight_sum = np.zeros_like(h)
        biome_out_weights: list[np.ndarray] = [np.zeros_like(h) for _ in range(max_biomes)]

        self._assign_override_biomes(
            h,
            rules,
            sea,
            mountain_share,
            temperature,
            moisture,
            erosion_field,
            volcanic,
            continental,
            biome_indices,
            biome_weight_sum,
            biome_out_weights,
            max_biomes,
        )
        self._assign_climate_biomes(
            h,
            rules,
            self.options.get("biomes", BIOMES),
            temperature,
            moisture,
            biome_indices,
            biome_weight_sum,
            biome_out_weights,
            max_biomes,
        )
        return self._build_biome_output(biome_indices, biome_out_weights, max_biomes)

    def _compute_override_strength(
        self,
        kind: str,
        sea: float,
        h: np.ndarray,
        mountain_share: np.ndarray,
        temperature: np.ndarray,
        moisture: np.ndarray,
        erosion_field: np.ndarray,
        volcanic: np.ndarray,
        continental: np.ndarray,
        rules: dict[str, Any],
    ) -> np.ndarray:
        """Compute biome strength for an override biome type."""
        depths = sea - h
        above_sea = h - sea
        on_land = smoothstep(sea - 0.5, sea + 0.3, h)

        if kind == "volcanic":
            return volcanic
        if kind == "ocean":
            return smoothstep(rules["deepWater"] * 0.5, rules["deepWater"], depths)
        if kind == "glacier":
            return (
                smoothstep(
                    rules["glacierTemperature"] + 3, rules["glacierTemperature"] - 2, temperature
                )
                * on_land
            )
        if kind == "peaks":
            return smoothstep(rules["peakShare"], rules["peakShare"] + 0.15, mountain_share)
        if kind == "alpine":
            return smoothstep(
                rules["treelineTemperature"] + 1.5, rules["treelineTemperature"] - 1.5, temperature
            ) * smoothstep(rules["alpineMinHeight"] * 0.7, rules["alpineMinHeight"], above_sea)
        if kind in ("mangrove", "coast"):
            shore = (
                smoothstep(rules["coastHeight"], rules["coastHeight"] * 0.4, above_sea)
                * smoothstep(rules["deepWater"], rules["deepWater"] * 0.5, depths)
                * smoothstep(0.12, -0.02, continental)
            )
            temp_strength = smoothstep(
                rules["mangroveTemperature"] - 2, rules["mangroveTemperature"] + 2, temperature
            ) * smoothstep(0.6, 0.8, moisture)
            if kind == "mangrove":
                return shore * temp_strength
            mangrove_strength = shore * temp_strength
            remaining_clean = np.maximum(1e-4, 1 - mangrove_strength)
            return (shore - mangrove_strength) / remaining_clean
        if kind == "wetland":
            return (
                smoothstep(
                    rules["wetlandMoisture"] - 0.06, rules["wetlandMoisture"] + 0.06, moisture
                )
                * smoothstep(rules["wetlandMaxHeight"], rules["wetlandMaxHeight"] * 0.4, above_sea)
                * smoothstep(0.45, 0.7, erosion_field)
                * on_land
            )
        return np.zeros_like(h)

    def _assign_override_biomes(
        self,
        h: np.ndarray,
        rules: dict[str, Any],
        sea: float,
        mountain_share: np.ndarray,
        temperature: np.ndarray,
        moisture: np.ndarray,
        erosion_field: np.ndarray,
        volcanic: np.ndarray,
        continental: np.ndarray,
        biome_indices: list[np.ndarray],
        biome_weight_sum: np.ndarray,
        biome_out_weights: list[np.ndarray],
        max_biomes: int,
    ) -> None:
        """Assign override biomes (volcanic, ocean, glacier, etc.)."""
        remaining = np.ones_like(h)
        for kind in _OVERRIDE_ORDER:
            if kind not in self._override_idx:
                continue
            if remaining.max() < 1e-4:
                break
            idx = self._override_idx[kind]
            strength = self._compute_override_strength(
                kind,
                sea,
                h,
                mountain_share,
                temperature,
                moisture,
                erosion_field,
                volcanic,
                continental,
                rules,
            )
            w = np.clip(strength, 0, 1) * remaining
            mask = w > 1e-4
            if not np.any(mask):
                continue
            slot = self._find_free_slot(biome_indices, max_biomes)
            if slot is not None:
                biome_indices[slot][mask] = idx
                biome_out_weights[slot][mask] = w[mask]
                biome_weight_sum[mask] += w[mask]
                remaining = np.maximum(0, remaining - w)

    def _find_free_slot(self, biome_indices: list[np.ndarray], max_biomes: int) -> int | None:
        """Find the first free slot in biome_indices, or None if full."""
        for s in range(max_biomes):
            if np.all(biome_indices[s] == -1):
                return s
        return None

    def _assign_climate_biomes(
        self,
        h: np.ndarray,
        rules: dict[str, Any],
        biomes_list: list,
        temperature: np.ndarray,
        moisture: np.ndarray,
        biome_indices: list[np.ndarray],
        biome_weight_sum: np.ndarray,
        biome_out_weights: list[np.ndarray],
        max_biomes: int,
    ) -> None:
        """Assign climate-driven biomes from remaining weight."""
        remaining = np.ones_like(h) - biome_weight_sum
        if remaining.max() < 1e-4 or not self._climate_idx:
            return
        blend = rules["blend"]
        b2 = blend * blend
        for idx in self._climate_idx:
            biome_def = biomes_list[idx]
            temp_c = getattr(biome_def, "temperature", 15)
            moist_c = getattr(biome_def, "moisture", 0.5)
            dt = (temperature - temp_c) / 25
            dm = (moisture - moist_c) / 0.45
            score = np.exp(-(dt * dt + dm * dm) / b2)
            for slot in range(max_biomes):
                if biome_indices[slot][0, 0] == -1:
                    mask = (biome_weight_sum <= 1e-4) & (score > 1e-5)
                    biome_indices[slot][mask] = idx
                    biome_out_weights[slot][mask] = score[mask]
                    biome_weight_sum[mask] += score[mask]
                    break

    def _build_biome_output(
        self,
        biome_indices: list[np.ndarray],
        biome_out_weights: list[np.ndarray],
        max_biomes: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build final biomes/weights output arrays."""
        _ref = biome_indices[0]
        out_biomes = np.full((self.resolution, self.resolution), -1, dtype=np.int32)
        out_weights = np.zeros((self.resolution, self.resolution, max_biomes), dtype=np.float32)
        for slot in range(max_biomes):
            mask = biome_indices[slot] != -1
            out_biomes[mask] = biome_indices[slot][mask].astype(np.int32)
            out_weights[mask, slot] = biome_out_weights[slot][mask].astype(np.float32)
        w_sums = out_weights.sum(axis=-1, keepdims=True)
        w_sums = np.maximum(w_sums, 1e-6)
        out_weights = out_weights / w_sums
        return out_biomes, out_weights

    def _apply_biome_shaping(
        self,
        h: np.ndarray,
        out_biomes: np.ndarray,
        out_weights: np.ndarray,
        sea: float,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        biomes_list: list,
    ) -> np.ndarray:
        """Apply biome-specific terrain shaping."""
        max_biomes = out_weights.shape[-1]
        for slot in range(max_biomes):
            w = out_weights[:, :, slot]
            active = w > 1e-4
            if not active.any():
                continue

            bidx = int(out_biomes.flat[active.ravel().argmax()]) if active.any() else 0
            bdef = biomes_list[bidx]
            shape = getattr(bdef, "shape", "none")
            shape_amount = getattr(bdef, "shape_amount", 0)

            if shape == "none" or shape_amount == 0:
                continue

            if shape == "dunes":
                cl_dir = self.options["climate"]["windDirection"]
                wl2 = np.hypot(cl_dir[0], cl_dir[1]) or 1
                phase = (x * (cl_dir[0] / wl2) + z * (cl_dir[1] / wl2)) / 70 + 1.8 * fbm(
                    self._seed + 10, x, y, z, 3, 1.0 / 260
                )
                crest = 1 - np.abs(np.sin(phase * np.pi))
                h = np.where(
                    active,
                    h
                    + w
                    * shape_amount
                    * (
                        crest * crest * (0.6 + 0.4 * fbm(self._seed + 11, x, y, z, 2, 1.0 / 500))
                        - 0.3
                    ),
                    h,
                )
            elif shape == "terraces":
                step = shape_amount
                k = (h - sea) / step
                f = k - np.floor(k)
                terraced = sea + (np.floor(k) + smoothstep(0.75, 1, f)) * step
                h = np.where(active, lerp(h, terraced, w * 0.85), h)
            elif shape == "hummocks":
                h = np.where(
                    active, h + w * shape_amount * fbm(self._seed + 12, x, y, z, 2, 1.0 / 9), h
                )
            elif shape == "flatten":
                target = sea + 0.3 + (h - sea) * 0.25
                h = np.where(active, lerp(h, target, w * 0.8), h)
            elif shape == "crevasses":
                abs_fbm = np.abs(fbm(self._seed + 13, x, y, z, 3, 1.0 / 70))
                h = np.where(active, h - w * shape_amount * smoothstep(0.05, 0, abs_fbm), h)
        return h

    def _apply_detail_and_micro(
        self, h: np.ndarray, mountain_share: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> np.ndarray:
        """Apply detail and micro-relief."""
        do = self.options["detail"]
        if do["height"] > 0:
            h += (
                do["height"]
                * fbm(self._seed + 14, x, y, z, do["octaves"], 1.0 / do["scale"])
                * (0.5 + mountain_share * 2)
            )

        uo = self.options["micro"]
        if uo["height"] > 0:
            h += uo["height"] * fbm(self._seed + 15, x, y, z, uo["octaves"], 1.0 / uo["scale"])
        return h

    # ------------------------------------------------------------------
    # Bulk generation: fills the pre-allocated arrays in-place
    # ------------------------------------------------------------------

    def generate(self) -> World:
        """Generate the complete world into the pre-allocated arrays.

        Returns:
            self for method chaining.
        """
        res = self.resolution
        x = np.linspace(-1, 1, res, dtype=np.float64)
        self._x, self._y = np.meshgrid(x, x, indexing="ij")
        self._x = self._x * self.size * 0.5
        self._y = self._y * self.size * 0.5
        self._z = np.zeros_like(self._x)

        self._sample_all(self._x, self._y)
        return self

    def _sample_all(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray | None = None,
    ) -> World:
        """Sample the world on a grid. Fills arrays in-place."""
        if z is None:
            z = np.zeros_like(x)

        sea = self.options["seaLevel"]
        o = self.options

        # Compute terrain layers
        base, land, continental = self._compute_continents(x, y, z, o["continents"])
        erosion_field = self._compute_erosion(o["erosion"], x, y, z)
        mountains, mountain_share, _ = self._compute_mountains(
            o["mountains"], erosion_field, land, x, y, z
        )
        hills = self._compute_hills(o["hills"], erosion_field, land, x, y, z)
        volcano, volcanic = self._compute_volcanoes(o["volcanoes"], land, x, y, z)

        # Base height
        h = base + mountains + hills + volcano

        # Rivers
        valley, _river, water_level = self._compute_rivers(
            o["rivers"], h, base, hills, mountains, volcanic, x, y, z, sea
        )

        # Climate
        temperature, moisture = self._compute_climate(
            o["climate"], o["mountains"], h, continental, valley, mountains, x, y, z
        )

        # Biomes
        out_biomes, out_weights = self._assign_biomes(
            sea, h, mountain_share, temperature, moisture, erosion_field, volcanic, continental
        )

        # Biome shaping
        h = self._apply_biome_shaping(
            h, out_biomes, out_weights, sea, x, y, z, o.get("biomes", BIOMES)
        )

        # Detail and micro
        h = self._apply_detail_and_micro(h, mountain_share, x, y, z)

        # Water level final
        water_level = np.where(h < sea, sea, water_level)
        water_level = np.where((h >= sea) & (~np.isfinite(water_level)), np.nan, water_level)

        # Store results
        self.elevation[:] = h
        self.biome[:] = out_biomes
        self.biome_weights[:] = out_weights.astype(np.float32)
        self.water_level[:] = water_level
        return self

    # ------------------------------------------------------------------
    # Single-point helpers
    # ------------------------------------------------------------------

    def _sample_single_continents(
        self, cx: float, cy: float, co: dict[str, Any]
    ) -> tuple[float, float, float]:
        """Compute continent data for a single point."""
        q = warp(
            self._seed + 101,
            np.array([cx]),
            np.array([cy]),
            np.array([0.0]),
            1.0 / (co["scale"] * 1.7),
            co["warp"] * co["scale"],
        )
        continental = clamp(
            fbm(self._seed + 1, q[0], q[1], q[2], co["octaves"], 1.0 / co["scale"]) * 1.25
            + co["landBias"],
            -1,
            1,
        )[0]
        base_val = curve_fn(co["curve"], np.array([continental]))[0]
        land = smoothstep(-0.08, 0.1, np.array([continental]))[0]
        return base_val, land, continental

    def _sample_single_mountains(
        self, cx: float, cy: float, erosion: float, mo: dict[str, Any]
    ) -> tuple[float, float]:
        """Compute mountain data for a single point."""
        range_f = (
            0.5
            + 0.5
            * _spread(
                fbm(
                    self._seed + 3,
                    np.array([cx]),
                    np.array([cy]),
                    np.array([0.0]),
                    3,
                    1.0 / mo["rangeScale"],
                )
            )[0]
        )
        range_mask = (
            smoothstep(mo["threshold"], mo["threshold"] + 0.25, np.array([range_f]))[0]
            * (1 - erosion * mo["erosionDamping"])
            * max(0, 1)
        )
        ridge = ridged(
            self._seed + 4,
            np.array([cx]),
            np.array([cy]),
            np.array([0.0]),
            mo["octaves"],
            1.0 / mo["scale"],
            2.0,
            0.5,
            mo["sharpness"],
        )[0]
        mountain_share = ridge * max(range_mask, 0)
        mountains = mo["height"] * mountain_share
        return mountains, mountain_share

    def _sample_single_hills(
        self, cx: float, cy: float, erosion: float, land: float, ho: dict[str, Any]
    ) -> float:
        """Compute hills for a single point."""
        return (
            ho["height"]
            * eroded_fbm(
                self._seed + 5,
                np.array([cx]),
                np.array([cy]),
                np.array([0.0]),
                ho["octaves"],
                1.0 / ho["scale"],
                ho["erosion"],
            )[0]
            * (0.35 + 0.65 * (1 - erosion))
            * (0.25 + 0.75 * land)
        )

    def _sample_single_volcanoes(
        self, cx: float, cy: float, land: float, vo: dict[str, Any]
    ) -> tuple[float, float]:
        """Compute volcano data for a single point."""
        volcano_val = 0.0
        volcanic_val = 0.0
        if vo["chance"] > 0:
            x_, y_, z_ = cx / vo["regionSize"], cy / vo["regionSize"], 0.0
            dist, _, _, rand_v = cellular(
                self._seed + 6, np.array([x_]), np.array([y_]), np.array([z_]), jitter=0.8
            )
            if rand_v[0] < vo["chance"]:
                d = dist[0] * vo["regionSize"]
                t = smoothstep(vo["radius"], 0, np.array([d]))[0]
                crater = smoothstep(
                    vo["radius"] * vo["crater"], vo["radius"] * vo["crater"] * 0.4, np.array([d])
                )[0]
                volcano_val = vo["height"] * (t**1.6 - crater * 0.35) * land
                volcanic_val = (
                    smoothstep(vo["radius"] * 1.15, vo["radius"] * 0.7, np.array([d]))[0] * land
                )
        return volcano_val, volcanic_val

    def _sample_single_biomes(
        self,
        temperature: float,
        moisture: float,
        above_sea: float,
        depths: float,
        _mountain_share: float,
        _erosion: float,
        _volcanic_val: float,
        _land: float,
        _sea: float,
        _h: float,
        _continental: float,
    ) -> tuple[list[int], list[float]]:
        rules = self.options["rules"]
        biomes_list = self.options.get("biomes", BIOMES)
        biomes_out: list[int] = []
        weights_out: list[float] = []

        for kind in [
            "volcanic",
            "ocean",
            "glacier",
            "peaks",
            "alpine",
            "mangrove",
            "coast",
            "wetland",
        ]:
            if kind not in self._override_idx:
                continue
            idx = self._override_idx[kind]
            strength = self._compute_override_strength_scalar(
                kind,
                rules,
                temperature,
                moisture,
                above_sea,
                depths,
            )
            w = min(max(strength, 0), 1)
            if w > 1e-4 and len(biomes_out) < rules.get("maxBiomes", 4):
                biomes_out.append(idx)
                weights_out.append(w)

        # Climate biomes
        remaining = 1.0 - sum(weights_out)
        if remaining > 1e-4 and self._climate_idx:
            for idx in self._climate_idx:
                if len(biomes_out) >= rules.get("maxBiomes", 4):
                    break
                bdef = biomes_list[idx]
                temp_c = getattr(bdef, "temperature", 15)
                moist_c = getattr(bdef, "moisture", 0.5)
                dt = (temperature - temp_c) / 25
                dm = (moisture - moist_c) / 0.45
                b2 = rules["blend"] * rules["blend"]
                w = float(np.exp(-(dt * dt + dm * dm) / b2))
                if w > 1e-5:
                    biomes_out.append(idx)
                    weights_out.append(w)

        # Normalise
        if biomes_out:
            total = sum(weights_out)
            if total > 0:
                paired = sorted(zip(biomes_out, weights_out, strict=False), key=lambda x: -x[1])
                paired = list(paired)[: rules.get("maxBiomes", 4)]
                biomes_out = [b for b, _ in paired]
                weights_out = [w for _, w in paired]
                weights_out = [w / total for w in weights_out]

        return biomes_out, weights_out

    def _compute_override_strength_scalar(
        self,
        kind: str,
        rules: dict[str, Any],
        temperature: float,
        moisture: float,
        above_sea: float,
        depths: float,
    ) -> float:
        """Scalar version of _compute_override_strength for sample()."""
        _moisture = moisture  # for mangrove/coast
        if kind == "ocean":
            return smoothstep(rules["deepWater"] * 0.5, rules["deepWater"], np.array([depths]))[0]
        if kind == "glacier":
            return smoothstep(
                rules["glacierTemperature"] + 3,
                rules["glacierTemperature"] - 2,
                np.array([temperature]),
            )[0]
        if kind == "peaks":
            return smoothstep(rules["peakShare"], rules["peakShare"] + 0.15, np.array([above_sea]))[
                0
            ]
        if kind == "alpine":
            return smoothstep(
                rules["treelineTemperature"] + 1.5,
                rules["treelineTemperature"] - 1.5,
                np.array([temperature]),
            )[0]
        return 0.0

    # ------------------------------------------------------------------
    # Single-point sample
    # ------------------------------------------------------------------

    def sample(self, cx: float, cy: float) -> WorldSample:
        """Sample the world at a single (x, y) point.

        Args:
            cx: X coordinate in world space.
            cy: Y coordinate in world space.

        Returns:
            WorldSample with elevation, climate, biome, etc.
        """
        o = self.options
        sea = o["seaLevel"]

        # Continents
        base_val, land, continental = self._sample_single_continents(cx, cy, o["continents"])

        # Erosion
        eo = o["erosion"]
        erosion = (
            0.5
            + 0.5
            * _spread(
                fbm(
                    self._seed + 2,
                    np.array([cx]),
                    np.array([cy]),
                    np.array([0.0]),
                    eo["octaves"],
                    1.0 / eo["scale"],
                )
            )[0]
        )

        # Mountains
        mountains, mountain_share = self._sample_single_mountains(cx, cy, erosion, o["mountains"])

        # Hills
        hills_val = self._sample_single_hills(cx, cy, erosion, land, o["hills"])

        # Volcanoes
        volcano_val, volcanic_val = self._sample_single_volcanoes(cx, cy, land, o["volcanoes"])

        # Base height
        h = base_val + mountains + hills_val + volcano_val

        # Rivers (simplified)
        valley = river = 0.0
        water_level = float("nan")
        ro = o["rivers"]
        if ro["enabled"]:
            rq = warp(
                self._seed + 201,
                np.array([cx]),
                np.array([cy]),
                np.array([0.0]),
                1.0 / (ro["scale"] * 0.8),
                ro["scale"] * 0.35,
            )
            rn = np.abs(fbm(self._seed + 7, rq[0], rq[1], rq[2], 3, 1.0 / ro["scale"]))[0]
            height_mask = (
                smoothstep(sea + 0.5, sea + 4, np.array([base_val + hills_val * 0.5]))[0]
                * (1 - smoothstep(ro["maxHeight"] * 0.6, ro["maxHeight"], np.array([h])))[0]
                * (1 - volcanic_val)
            )
            valley = smoothstep(ro["valleyWidth"], 0, np.array([rn]))[0] * height_mask
            river = smoothstep(ro["width"], ro["width"] * 0.35, np.array([rn]))[0] * height_mask
            if valley > 0:
                low = base_val + hills_val * 0.35 + mountains * 0.15
                surface = sea + max(0.4, (low - sea) * (1 - ro["valleyDepth"]))
                h = min(h, surface + 0.6) * valley + h * (1 - valley)
                if river > 0:
                    h = surface - ro["depth"]
                    if h < surface:
                        water_level = surface

        # Climate
        cl = o["climate"]
        above = max(0, h - sea)
        temperature = (
            cl["baseTemperature"]
            + cl["temperatureRange"]
            * _spread(
                fbm(
                    self._seed + 8,
                    np.array([cx]),
                    np.array([cy]),
                    np.array([0.0]),
                    3,
                    1.0 / cl["temperatureScale"],
                )
            )[0]
        )
        temperature -= (cl["lapseRate"] * above) / 1000

        moisture = (
            0.5
            + 0.55
            * _spread(
                fbm(
                    self._seed + 9,
                    np.array([cx]),
                    np.array([cy]),
                    np.array([0.0]),
                    3,
                    1.0 / cl["moistureScale"],
                )
            )[0]
            + cl["moistureBias"]
        )
        moisture += cl["coastalMoisture"] * smoothstep(0.25, -0.02, np.array([continental]))[0]
        moisture += cl["riverMoisture"] * valley
        moisture = clamp(moisture, 0, 1)[0]

        if h < sea:
            water_level = sea

        # Biomes
        depths = sea - h
        above_sea = h - sea
        biomes_out, weights_out = self._sample_single_biomes(
            temperature,
            moisture,
            above_sea,
            depths,
            mountain_share,
            erosion,
            volcanic_val,
            land,
            sea,
            h,
            continental,
        )
        biomes_out, weights_out = self._sample_single_biomes(
            temperature,
            moisture,
            mountain_share,
            erosion,
            volcanic_val,
            land,
            sea,
            h,
            continental,
            above_sea,
            depths,
        )

        return WorldSample(
            elevation=h,
            continentalness=float(continental),
            erosion=float(erosion),
            mountain=float(mountain_share),
            temperature=float(temperature),
            moisture=float(moisture),
            valley=float(valley),
            river=float(river),
            water_level=float(water_level) if np.isfinite(water_level) else float("nan"),
            biomes=biomes_out,
            weights=weights_out,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_height_at(self, x: float, y: float) -> float:
        """Get terrain height at world coordinates.

        Args:
            x: X coordinate in world space.
            y: Y coordinate in world space.

        Returns:
            Height in metres.
        """
        return self.sample(x, y).elevation

    def get_biome_at(self, x: float, y: float) -> int:
        """Get the primary biome index at world coordinates.

        Args:
            x: X coordinate in world space.
            y: Y coordinate in world space.

        Returns:
            Biome index (integer), or -1 if unknown.
        """
        s = self.sample(x, y)
        return s.biomes[0] if s.biomes else -1

    def get_height_interpolated(self, x: float, y: float) -> float:
        """Get interpolated height from the generated grid.

        Args:
            x: X coordinate in world space.
            y: Y coordinate in world space.

        Returns:
            Interpolated height value.
        """
        raise_msg = "Call generate() first"
        if not hasattr(self, "elevation"):
            raise RuntimeError(raise_msg)
        gx = ((x / self.size) + 0.5) * (self.resolution - 1)
        gy = ((y / self.size) + 0.5) * (self.resolution - 1)
        ix = int(np.clip(gx, 0, self.resolution - 2))
        iy = int(np.clip(gy, 0, self.resolution - 2))
        fx = gx - ix
        fy = gy - iy
        h00 = self.elevation[ix, iy]
        h10 = self.elevation[ix + 1, iy]
        h01 = self.elevation[ix, iy + 1]
        h11 = self.elevation[ix + 1, iy + 1]
        h0 = h00 * (1 - fx) + h10 * fx
        h1 = h01 * (1 - fx) + h11 * fx
        return float(h0 * (1 - fy) + h1 * fy)


# ---------------------------------------------------------------------------
# Helpers (ported from world.ts)
# ---------------------------------------------------------------------------


def _spread(x: np.ndarray) -> np.ndarray:
    """Symmetric spread: |x| with sign preserved."""
    return np.sign(x) * x * x
