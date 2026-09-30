# Optional Blender renderer specification

Renderer code and generated images are intentionally outside this release. An independent Blender implementation must write `data/factoratlas/metadata.json` and each record's PNG under `images/`.

Each record requires `file`, `pattern`, `hue`, `shape`, `context`, `renderer_seed`, and deterministic integer `seed`. Generate the full 10 patterns x 12 hues x 8 shapes x 12 contexts x 2 seeds product (23,040 records). Contexts 0–3 are calibration, 4–5 validation, and 6–11 test.


Before embedding, verify unique paths, image existence, full vocabularies, 23,040 records, and balanced factor counts within every context x renderer-seed cell.
