# Owner-machine benchmark — cached mesh maps and ~25 FPS

Date: 2026-11-20
App: 0.1.6
Evidence: owner-provided diagnostic snapshot (technical panel copy)

## Scenario

After loading 13 cached built-in mesh maps for a low-poly production asset:
- Split layout: 3D + UV.
- 3D view: 615 × 862 px; UV view: 376 × 862 px.
- 7 texture sets; active set index 1 (chassi).
- 2048² texture resolution.
- 10 visible layers / 10 total layers.
- Paint tool, Screen projection, 70 pt brush.
- F6 meter enabled.

Exact project name, local path and machine-specific adapter identity are kept in the owner's local context, not this public report.

## Observed frame performance

| Metric | Observed |
|---|---:|
| Frame interval | 40.086 ms |
| Estimated FPS | 24.946 |
| UI frame CPU EMA | 30.907 ms |
| UI frame CPU recent peak | 31.265 ms |

## Mesh-map cache evidence

- All 13 built-in maps are present (baked_map_mask=0x1fff).
- Bake report says from_cache=true.
- Asset statistics: 57,943 triangles, 442 parts, 2048 resolution.
- Cache report has no missed or filled texels.
- The reported 11.41154 s GPU time is the cached bake report, not evidence that a bake ran during this measurement.

## Comparison caveat

A prior snapshot on the same project and 10-layer stack reported approximately 16.965 FPS with F6 disabled. The current snapshot reports 24.946 FPS with F6 enabled. These are observations, not a controlled A/B test; do not attribute the difference to the map cache or profiler.

## Next controlled capture

Repeat the same project, same active Texture Set, same layout, same view mode and no layer edits:
1. Record idle frame interval with F6 disabled.
2. Enable F6 and capture the same idle interval.
3. Paint one short Screen stroke, then release and capture F6 again.
4. Repeat once with a simple/unlit or Base Color display mode, without changing the asset or texture resolution.
5. Capture F5 full-composite timing separately.

Record whether maps were loaded from cache and keep the exact workload constant.