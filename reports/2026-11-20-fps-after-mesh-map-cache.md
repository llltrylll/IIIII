# Engineering assessment — ~25 FPS after mesh-map cache load

Date: 2026-11-20. Evidence source: owner diagnostic snapshot and static code inspection on the active development line.

## Current reading

The snapshot reports 24.946 FPS (40.086 ms interval), UI CPU EMA 30.907 ms and recent peak 31.265 ms. GPU composite is 18.8 ms average / 148 ms peak. F5 was not captured.

The 13 built-in maps were loaded from cache (from_cache=true); the bake was not freshly executed in this capture.

## Strongest leads

### 1. Parked Texture Set residency

Known Engine allocations total ~2.11 GiB by the application's own estimator, plus a separately reported 64 MiB shared Undo readback cache. Six parked Engines account for ~1.52 GiB. Their inactive-working-set release drops temporary scratch, prefix, filters and thumbnail scratch, but retains surface maps, packed mesh maps/fixes and one accumulation set.

This is a meaningful memory-pressure hypothesis on the owner's integrated GPU/shared-memory machine, not proof of physical VRAM exhaustion. The viewport binds per-Engine final textures, five built-in mesh-map textures, custom maps and surface maps, so simply releasing parked maps is not safe while all Texture Sets remain visible.

Investigate a renderer-aware residency strategy: distinguish resources required for the current viewport preview from resources only required for editing/recomposition. Preserve exact visual parity and fast Texture Set switching.

### 2. Full-texture Undo snapshot at stroke release

F6 records 87.7 ms in snapshot encoding and 89.5 ms total end time. Code path:
- Engine::end_stroke calls Snapshot::encode for the target texture list.
- Snapshot::encode copies each complete texture to staging buffers.
- The stroke dirty/scissor rectangles are not passed to the snapshot function.

The measured paint area was only 3.3% with four scissors. A region-aware before/after snapshot may substantially reduce bytes copied, but needs a correctness design for disconnected regions, blending, masks, channels, UV strokes, undo/redo, and compressed-size accounting.

This is currently the most concrete isolated source-level optimization candidate.

### 3. Composite/viewport frame cost

Composite average is 18.8 ms, but peak is 148 ms. One full recomposition occurred since F6 was enabled. F5 is missing, so separate persistent composite cost from occasional full-rebuild spikes with a controlled profile. Test a simple display mode to see whether the heavy material/mesh-map preview shader materially contributes.

## Recommended order

1. Add targeted instrumentation for Snapshot::encode: texture count, dimensions/formats, total copied bytes, pool hits/misses, encoder CPU time and asynchronous map latency. Avoid GPU waits.
2. Prototype region-aware Undo snapshots in test lane A; retain a full-snapshot fallback and compare byte-identical undo/redo results.
3. Separately profile parked-set residency and determine which baked maps the active viewport shader actually reads per display mode. Do not release them blindly.
4. Capture F5 for the current 10-layer state and an F6 on/off idle A/B. Keep all settings identical.

## Claims not supported yet

- No proof that 13 cached mesh maps themselves directly cause 25 FPS.
- No physical VRAM reading.
- No proof that the 87.7 ms snapshot cost occurs on every frame; it is measured at stroke end.
- No controlled F6-off comparison for the new snapshot.