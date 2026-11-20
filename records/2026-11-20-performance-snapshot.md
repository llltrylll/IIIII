# Owner benchmark record — 2026-11-20

This report is sanitized for the public repository. Exact machine and project identifiers remain only in the owner's local LOCAL_CONTEXT.md.

## Snapshot A — prior reference

- Layer count: 8
- F5 total: 177 ms (CPU preparation 7 ms; GPU 170 ms)
- Composite GPU: 167.41 ms; body: 167.29 ms
- Layer passes: 7
- Micro Roughness: 55.50 ms
- Estimated FPS: 23.886; frame interval: 41.865 ms
- UI CPU EMA: 30.976 ms; observed peak: 510 ms
- F6: disabled

## Snapshot B — 2026-11-20

- Application version: 0.1.6
- Texture Set: chassi, 2048 × 2048
- Texture Sets: 7
- Layer count: 10
- F5 total: 221 ms (CPU preparation 23 ms; GPU 198 ms)
- Composite GPU: 193.568 ms; body: 193.443 ms
- Layer passes: 8; profiler stages: 9
- Micro Roughness: 46.5458 ms
- Silver: 42.1528 ms
- Scratches: 32.5890 ms
- Polished Edges: 20.2453 ms
- Fill layer: 13.5950 ms
- Layer 3: 9.8595 ms
- laranja: 9.8504 ms
- white and black: 9.4525 ms
- Thumbnails: 7.3048 ms
- Mask cache: 4 hits, 0 misses, 0 GPU passes
- F6: disabled
- Frame time EMA: 46.287 ms; peak: 47.067 ms
- Frame interval: 58.946 ms; derived FPS: approximately 16.965

## Interpretation and caveats

The two runs are not equivalent: Snapshot A has 8 layers; Snapshot B has 10. Do not attribute the overall F5/FPS change to one optimization. Micro Roughness fell from 55.50 to 46.55 ms (~16.1%), while total composite GPU rose from 167.41 to 193.57 ms and F5 rose from 177 to 221 ms. More layers and workload differences prevent causal attribution. Silver and Scratches are additional high-cost stages worth investigating.

F6 was disabled in Snapshot B, so detailed cache counters for the new roughness FBM path were not captured. Collect a new run with F6 enabled and preserve the exact same project state, layer count, resolution, viewport and operation when comparing builds.

The owner-machine hardware identity and local project path are intentionally withheld from this public report.
