export const meta = { name: "outdoor-worlds-impl", description: "Implement procedural outdoor worlds for hyperdrone" };

// Phase 1: Port noise system and biome table (independent, run in parallel)
const [noise_result, biome_result] = await Promise.all([
  agent(
    `## PORT NOISE SYSTEM FROM threejs-worldgen

You are implementing /Users/vincent/Work/hyperdrone/hyperdrone/environments/outdoor/terrain/noise.py — a pure Python/NumPy vectorised port of threejs-worldgen/biomes/noise.ts.

Step 1: Fetch and read the TypeScript source:
curl -s https://raw.githubusercontent.com/askalice/threejs-worldgen/main/packages/biomes/src/noise.ts

Step 2: Create the directory structure:
mkdir -p /Users/vincent/Work/hyperdrone/hyperdrone/environments/outdoor/terrain

Step 3: Implement all functions from the TypeScript source as vectorised Python/NumPy:
- hash4, hash01, seedOf, mulberry32
- gradientNoise (3D Perlin with derivatives), noise3 (fast path)
- fbm, ridged, erodedFbm (fractal sums)
- warp (domain warping), cellular (Worley noise)
- helpers: smoothstep, lerp, clamp, curve

Rules:
- np.float64 for determinism, np.int32 for hash computations  
- ALL vectorised: accept (H,W) or (N,) arrays, return same shape
- Pure functions, no global state, Google docstrings
- Import from __future__ import annotations

Step 4: Write tests at hyperdrone/environments/outdoor/terrain/test_noise.py

Step 5: Verify importability:
cd /Users/vincent/Work/hyperdrone && uv pip install numpy pytest
python -c "from hyperdrone.environments.outdoor.terrain.noise import noise3, fbm, ridged, erodedFbm; import numpy as np; print('OK, shape:', fbm(42, np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), {'frequency': 0.01, 'octaves': 4, 'lacunarity': 2, 'gain': 0.5}).shape)"

Return the full source code of noise.py and test_noise.py when done.`,
    { label: "noise-porter", effort: "high" }
  ),
  agent(
    `## PORT BIOME TABLE AND CLIMATE FROM threejs-worldgen

You are implementing:
1. /Users/vincent/Work/hyperdrone/hyperdrone/environments/outdoor/terrain/biomes.py — the 20-biome table
2. /Users/vincent/Work/hyperdrone/hyperdrone/environments/outdoor/terrain/climate.py — climate maps
3. /Users/vincent/Work/hyperdrone/hyperdrone/environments/outdoor/terrain/map.py — biome blending

Step 1: Fetch all three TypeScript sources:
- Biome table: curl -s https://raw.githubusercontent.com/askalice/threejs-worldgen/main/packages/biomes/src/biomes.ts
- World/climate: curl -s https://raw.githubusercontent.com/askalice/threejs-worldgen/main/packages/biomes/src/world.ts
- Map: curl -s https://raw.githubusercontent.com/askalice/threejs-worldgen/main/packages/biomes/src/map.ts

Step 2: Read all three files carefully.

Step 3: biomes.py — Port ALL 20 biomes as list of dicts. Define GROUND_MATERIALS and GROUND_COLORS.

Step 4: climate.py — Temperature map (latitude + elevation), moisture map (ocean proximity), continent map (fbm), river map.

Step 5: map.py — Biome weight blending from (temperature, moisture) → dominant biome index → RGB color map.

Rules:
- All vectorised over (H, W) grids
- np.float64 for determinism
- Pure functions, Google docstrings
- Import from __future__ import annotations

Step 6: Write tests for all three modules.

Step 7: Verify:
cd /Users/vincent/Work/hyperdrone && uv pip install numpy pytest
python -c "from hyperdrone.environments.outdoor.terrain.biomes import BIOMES; print(f'{len(BIOMES)} biomes loaded')"
python -c "from hyperdrone.environments.outdoor.terrain.map import assign_biomes; print('map OK')"

Return the full source code when done.`,
    { label: "biome-porter", effort: "high" }
  )
]);

// Log results
log("Noise port result: " + (noise_result || "FAILED") + "\n" + (noise_result?.substring(0, 200) || ""));
log("Biome port result: " + (biome_result || "FAILED") + "\n" + (biome_result?.substring(0, 200) || ""));

// Return summary
return "Phase 1 complete. Check hyperdrone/environments/outdoor/terrain/ for noise.py, biomes.py, climate.py, map.py and tests.";