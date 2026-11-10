# AGENTS.md — HyperDrone Fork (vlordier)

## Python Tooling

**Always use `uv` for Python package management.**

- Installing packages: `uv pip install <package>`, `uv pip install hyperdrone[examples]`
- Running code: `uv run python ...` or `uv run python -m hyperdrone.examples.xxx`
- Creating virtual envs: `uv venv`
- All `pip` or `python3` bare commands are forbidden unless `uv` is explicitly unavailable
- The `.venv/` directory is managed by uv and should never be modified by hand

## Coding Practices

Adopt these practices from the HyperDrone codebase:

### Determinism
- All procedural generation must be seeded. Same seed → same world, on any machine, any OS.
- Use `np.random.default_rng(seed)` or equivalent for NumPy random. Never use global `np.random` or `random.random()`.
- Noise functions (fbm, ridged, cellular) must be pure functions of `(seed, x, y, z)`.

### Performance
- Vectorise NumPy operations over arrays — never loop over grid cells in Python when a vectorised operation works.
- Heightfield sampling should return `(H, W)` arrays, not per-pixel calls.
- Mesh generation uses NumPy `stack`, `repeat`, broadcasting — batch vertex creation.

### Style
- Type hints everywhere (`from __future__ import annotations`).
- Docstrings on all public functions and classes (Google style, like HyperDrone's).
- Pure functions for noise, biome evaluation, climate — no side effects.
- One function, one purpose. Noise → heightfield → biome → mesh: each in its own function.
- Use `dataclasses` or plain dicts for configuration objects.

### Architecture
- Each module is a standalone, testable unit.
- `hyperdrone.environments.outdoor.*` mirrors HyperDrone's `hyperdrone.dynamics.*` structure.
- Example scripts live in `hyperdrone/examples/` and are runnable via `python -m hyperdrone.examples.outdoor_flythrough`.
- Never hardcode constants — parameterise everything (seed, resolution, biome, terrain parameters).

### Testing
- Each pure function has a simple smoke test in its test file.
- Noise functions tested against known values: `noise3(seed=0, x=0, y=0, z=0)` is deterministic.
- Tests run under `pytest` and are importable without a GPU.

### Git
- Small, focused commits. One feature / one fix per commit.
- Descriptive commit messages: "Add ridged multifractal noise function" not "fix stuff".
- Feature branches named `feature/outdoor-worlds`.

## Current Feature: Procedural Outdoor Worlds

Replacing indoor rooms with procedural outdoor terrain inspired by threejs-worldgen.

**Goal:** Generate deterministically seeded outdoor worlds with biomes (20 climate-blended biomes from
threejs-worldgen's biome table), terrain shaping (mountains, dunes, terraces), and scattered objects,
renderable in HyperLux.

### Design

```
hyperdrone/
└── hyperdrone/
    ├── environments/
    │   ├── outdoor.py          # composition: World(seed, biome, resolution, ...)
    │   └── indoor.py           # current implementation (kept for backward compat)
    └── ...
```

Each module is ported from threejs-worldgen's TypeScript to pure NumPy:

| Module | Source | Description |
|--------|--------|-------------|
| `terrain/noise.py` | `threejs-worldgen/biomes/noise.ts` | hash4, hash01, seedOf, noise3, fbm, ridged, eroded, warp, cellular |
| `terrain/heightfield.py` | `threejs-worldgen/biomes/terrain.ts` | Height function builder from noise layers |
| `terrain/climate.py` | `threejs-worldgen/biomes/world.ts` | Continent, mountain, river climate maps |
| `terrain/biomes.py` | `threejs-worldgen/biomes/biomes.ts` | 20 biome table, weight blending, color mapping |
| `terrain/mesh.py` | Python-only | Heightfield → mesh (vertices, indices, colors) |
| `terrain/scatter.py` | `threejs-worldgen/biomes/detail.ts` | Blue-noise placement of trees, rocks, shrubs |
| `outdoor.py` | Python-only | Composition module: `build_world(...)` → `hyperlux.Scene` |
| `examples/outdoor_flythrough.py` | Python-only | Example script |

### Key Constraints
- No JavaScript, no npm, no TypeScript runtime dependency. Pure Python + NumPy.
- Outputs meshes compatible with `hyperlux.Scene.add_mesh()`.
- Same seed produces identical output across platforms (float64 arithmetic).
- Resolution is configurable (e.g., `resolution=256` for 256×256 height grid).

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:46cd31e7 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/core-concepts/sync-concepts.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   bd dolt push
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->

<!-- BEGIN BEADS CODEX SETUP: generated by bd setup codex -->
## Beads Issue Tracker

Use Beads (`bd`) for durable task tracking in repositories that include it. Use the `beads` skill at `.agents/skills/beads/SKILL.md` (project install) or `~/.agents/skills/beads/SKILL.md` (global install) for Beads workflow guidance, then use the `bd` CLI for issue operations.

### Quick Reference

```bash
bd ready                # Find available work
bd show <id>            # View issue details
bd update <id> --claim  # Claim work
bd close <id>           # Complete work
bd prime                # Refresh Beads context
```

### Rules

- Use `bd` for all task tracking; do not create markdown TODO lists.
- Run `bd prime` when Beads context is missing or stale. Codex 0.129.0+ can load Beads context automatically through native hooks; use `/hooks` to inspect or toggle them.
- Keep persistent project memory in Beads via `bd remember`; do not create ad hoc memory files.

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/core-concepts/sync-concepts.md for details and anti-patterns.
<!-- END BEADS CODEX SETUP -->
