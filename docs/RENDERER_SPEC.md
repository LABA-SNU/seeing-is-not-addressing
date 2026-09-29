# Optional Blender renderer specification

Renderer code and generated images are intentionally outside this release. An independent Blender implementation must write `data/factoratlas/metadata.json` and each record's PNG under `images/`.

Each record requires `file`, `pattern`, `hue`, `shape`, `context`, `renderer_seed`, and deterministic integer `seed`. Generate the full 10 patterns x 12 hues x 8 shapes x 12 contexts x 2 seeds product (23,040 records). Contexts 0–3 are calibration, 4–5 validation, and 6–11 test.

For each fixed `(hue, shape, context, renderer_seed)` cell, all ten patterns must have exactly the same camera, object pose, light rig, world illumination, roughness, and random state; only texture pattern changes. Across contexts vary camera elevation (0.14–0.98 rad), distance (3.65–6.35), lens (42–66), pose, three area lights, world strength (0.08–0.72), and roughness (0.25–0.90). Render 224x224 PNGs.

Before embedding, verify unique paths, image existence, full vocabularies, 23,040 records, and balanced factor counts within every context x renderer-seed cell.
