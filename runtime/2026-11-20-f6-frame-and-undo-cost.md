# Application runtime diagnostic — 2026-11-20

Source: owner-provided technical-panel diagnostic snapshot. App 0.1.6.

The owner reported that the frame rate was around 25 FPS after loading 13 cached mesh maps. The snapshot records 24.946 FPS / 40.086 ms frame interval.

## Frame and profiler measurements

- UI frame CPU EMA: 30.906784 ms.
- Recent UI frame CPU peak: 31.264847 ms.
- F5 full-composite profile: not captured.
- F6 was enabled.
- GPU composite total: 18.8 ms average / 148 ms peak.
- GPU composite pass body: 18.7 ms average / 148 ms peak.
- GPU stamp: 0.3 ms average; GPU apply: 0.2 ms average.
- Paint area: 3.3% of texture; four scissors.
- Composite area: 100%; one scissor.
- One full recomposition and three thumbnail updates since F6 enabled.
- Mask cache: 4 hits, 0 misses, 0 GPU passes.
- Fill FBM cache: 1 hit, 0 misses, 0 GPU passes.

## Stroke-end CPU cost

- End: 89.5 ms.
- Engine end: 89.4 ms.
- Snapshot encoding: 87.7 ms.
- Submit/map request: 1.0 ms.
- Cleanup: 0.4 ms.
- Recomposition after release: 9.4 ms.

The snapshot timing is a stroke-end CPU spike. It must not be interpreted as an 89 ms cost on every frame.

Code inspection: Engine::end_stroke creates an encoder, calls Snapshot::encode on the full target texture list, submits the copy and requests asynchronous mapping. Snapshot::encode loops through every texture and records a full-texture copy. It does not currently receive the stroke dirty scissors/region. This supports investigating region-aware Undo snapshots, but any change must preserve exact restore semantics and project correctness.

## Estimated GPU resource ownership

- Sum of known Engine allocations: 2,266,046,342 bytes (~2.11 GiB).
- Shared Undo staging-buffer cache: 64 MiB, separately reported.
- Active Engine (chassi): 635,370,746 bytes (~606 MiB).
- Six parked Engines together: ~1.52 GiB.
- Typical parked Engine: ~244 MiB. The parked extra.001 Engine: ~334 MiB.

Each parked Engine still retains roughly 64 MiB of surface maps, 112 MiB of mesh maps/fixes and 68 MiB of accumulation; one also retains paint layers and Fill images. Inactive-working-set release currently clears transient working resources, but retains viewport-visible per-set results/maps.

Important: this is an application-side allocation estimate, not physical VRAM telemetry or a measure of actual unified-memory residency.

## Viewport dependency

The viewport bind group receives each Engine's final material textures, five packed built-in mesh-map textures, custom maps and surface maps. TextureSets::set_engines exposes all sets to viewport rendering. Blindly dropping parked map textures would break the multi-Texture-Set preview. Any residency optimization needs a fallback/reload strategy or a renderer architecture that preserves visual output.

## Limits

No F5 capture in this session. No controlled A/B with F6 off. No physical VRAM/system-memory telemetry. No claim that loading cached maps alone caused the 25 FPS.