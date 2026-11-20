# Application runtime observations — 2026-11-20

Sanitized observations from application diagnostic output. This branch stores application-generated technical evidence, not source-code changes.

## Composite profiler

- F5 total: 221 ms.
- CPU preparation: 23 ms.
- GPU: 198 ms.
- Composite GPU: 193.568 ms.
- Composite body: 193.443 ms.
- Layer passes: 8.
- Profiler stages: 9.
- Mask cache: 4 hits, 0 misses, 0 GPU passes.

## Costly stages

| Stage | Time |
|---|---:|
| Micro Roughness | 46.5458 ms |
| Silver | 42.1528 ms |
| Scratches | 32.5890 ms |
| Polished Edges | 20.2453 ms |
| Fill layer | 13.5950 ms |
| Layer 3 | 9.8595 ms |
| laranja | 9.8504 ms |
| white and black | 9.4525 ms |
| Thumbnails | 7.3048 ms |

## Memory accounting reported by the application

- Estimated GPU resource allocations: 2,266,242,950 bytes.
- Active Texture Set allocation: 635,567,354 bytes.
- Active set accumulation: 136 MiB.
- Prefix cache: 68 MiB.
- Filtered masks: 56 MiB.
- Procedural variation cache: 8 MiB.
- Paint layers/masks: 104 MiB.
- Thumbnails: 12.62 MiB.
- Stroke snapshots: 32 MiB.
- Surface maps: 72 MiB.
- Mesh maps/fixes: 112 MiB.
- Fill images: 5.33 MiB.
- Shared undo readback cache: 64 MiB.
- Undo memory: 13.5 MB (4 steps; configured limit 512 MB).

Important: these are application-side allocation estimates, not measured physical VRAM telemetry. Do not label them as actual VRAM use.

## Runtime frame metrics

- Frame time EMA: 46.287 ms.
- Frame time peak: 47.067 ms.
- Frame interval: 58.946 ms.
- Derived FPS: approximately 16.965.
- F6 was disabled during this capture.

## Next diagnostic capture

Repeat the representative workload with F6 enabled while painting. Capture cache hit/miss/pass counters, undo snapshot costs, and frame timing. Keep project state, visible layers, texture resolution and viewport mode unchanged for comparisons.
