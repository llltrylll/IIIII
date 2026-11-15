# Codebase Audit — 2026-11-15

Branch reviewed: gpt/integrated-three-fronts  
Audit base: a9c8879cb3c7f2a359ae311c88551b5d02160075

## Executive summary

The latest user test confirms that Screen painting now reaches the intended surface, but Material viewport shading remains wrong and still shows red edge artifacts. These are separate contracts and must stay separate.

The earlier first-person camera implementation used pointer-position delta while the cursor was locked. That cannot provide reliable look movement because the cursor's window position is intentionally stationary. The current branch corrects this to use egui raw mouse motion, with pointer delta as a fallback. The current commit's CI/GPU/Windows workflows are running; local camera behavior still requires the user's test.

The highest-confidence performance candidates from source review are:
1. Bind-group and temporary-vector creation on every 3D viewport frame.
2. GPU resources replicated for every texture-set Engine.
3. Blocking GPU readback in export and one-pixel sampling.
4. Synchronous project save/open/autosave and startup Smart Material thumbnail loading.
5. All stale layer thumbnails being rendered together after a stroke.
6. Rebuilding graph evaluation plans and formatting parameter values into strings on each evaluation.

These are code-based candidates, not claimed benchmark wins. Measure them on the user's real 50k–100k triangle / approximately 6-set workload before broad changes.

## Repository inventory

The current src tree contains 157 files: 76 Rust files, 78 WGSL files, two JP2 support files and one bundled licence. Largest source hotspots:

| File | Approx. size | Responsibility |
|---|---:|---|
| src/app.rs | 244 KiB | Application state, actions, orchestration and diagnostics |
| src/preview_test.rs | 240 KiB | GPU regression fixtures |
| src/engine.rs | 220 KiB | Painting, layers, masks, compositing and caches |
| src/bake.rs | 126 KiB | Mesh-map bake preparation and GPU execution |
| src/presets.rs | 95 KiB | Built-in material and generator presets |
| src/app/graph_ui.rs | 79 KiB | Graph editor UI |
| src/shaders/bake.wgsl | 64 KiB | Bake shader |
| src/app/mcp_tools.rs | 56 KiB | MCP integration |
| src/viewport.rs | 52 KiB | Raster viewport and viewport resources |
| src/import.rs | 46 KiB | OBJ/FBX import and normal provenance |

The repository tree was inventoried and the major hot paths were read in detail. This is a broad audit of the codebase and its critical paths, not a formal proof that every bundled preset or every shader line is defect-free.

## P0 — correctness and active user-visible issues

### N-01 — Material viewport normal/shading mismatch is still unresolved

Evidence:
- src/shaders/view.wgsl, mesh_shade_inputs: derives a geometric normal from screen-space world-position derivatives, aligns the imported shading normal to it, reconstructs an invalid tangent frame from UV derivatives, decodes tangent-space normal detail, and clamps the mapped normal to the viewer-facing hemisphere.
- src/shaders/pathtrace.wgsl: independently builds the mesh shading frame and applies a related viewer-facing correction.
- src/mesh.rs, mikk_tangents: uses MikkTSpace with the V coordinate flipped to the mesh UV convention and flips tangent handedness accordingly.
- src/import.rs, load_fbx: reverses triangle corner order when the parity of the node's negative transform and ufbx winding conversion requires it.
- The user's latest test still shows incorrect Material viewport shading and red slivers around narrow tips. Therefore the generic corrections have not solved the actual asset's visual mismatch.

Test limitation:
- raster_view_is_stable_with_inverted_shading_normals compares a synthetic plane using +Z, -Z and zero normals. It tests face-forward fallback, not Blender parity.
- raster_normal_map_is_stable_with_inverted_imported_normals uses a synthetic rounded cube and a generated normal texture, not the user's FBX.
- raster_normal_map_is_clamped_to_view_hemisphere_at_grazing_angle analytically computes a clamped reference from the same correction rule and checks the shader applies it. This does not establish that the rule matches Blender/InstaMAT for the real asset.
- path_trace_pipeline_compiles_with_grazing_normal_correction is only a pipeline construction/compile test, not a path-traced pixel comparison.

Next action:
Use the exact failing imported asset or a small FBX exported from the offending Airfox objects. Capture per-corner position, normal, UV, tangent and handedness, then compare controlled normal-map shading against Blender/InstaMAT. Verify V convention, green-channel convention, tangent interpolation/orthogonalization, backface basis orientation and the sign used when the shading normal is aligned to the geometric normal. Add a GPU regression from the captured data. Do not apply global normal flips or more generic clamps without a reproducer.

### PNT-01 — Preserve the working Screen paint selection path

The paint-only depth target in src/viewport.rs::render_3d is distinct from visible viewport depth and is built from imported normal facing. The paint-depth shader discards inward-facing surfaces; app.rs supplies this target to Screen stamping. The user confirmed that painted surfaces are now correct.

Do not couple viewport display/shading changes to this paint-depth rule. Keep paint selection and displayed shading as separate contracts. Retain the existing stacked-UV, thin-occluder, inverted-front and sloped-surface GPU regressions.

### N-02 — Fallback normal welding remains a narrow risk

src/mesh.rs::Mesh::from_parts calls smooth_normals when no normals are supplied. smooth_normals welds positions on a 1e-4 quantization grid and accumulates face normals across the welded positions. In that fallback route, disconnected coincident shells or intentional hard edges can influence one another. Imported OBJ/FBX Builder fallback is now per-object and uses source vertex identity, so do not assume this explains the Airfox if it has valid authored normals. Add a fixture only for missing-normal paths with coincident disconnected shells and hard edges.

## P1 — performance opportunities with concrete source evidence

### PERF-01 — Viewport bind-group and temporary allocation churn every frame

Location: src/viewport.rs::ViewRenderer::render_3d and src/app/texture_sets.rs::engines.

The 3D render path builds a new vector of bind groups and calls bind_group once per texture-set engine each frame. It also creates a fresh visible-set vector and vector-backed shadow/paint-depth cache keys each frame. texture_sets::engines returns a newly allocated Vec at its call sites.

Why it matters: the expected six-set production workload creates several bind groups and temporary allocations every frame, even when resources have not changed.

Next action: profile CPU/driver time with one versus six sets. Cache bind groups by the identity/version of their referenced texture views and environment resources. Reuse visible-set/index scratch buffers and fixed-size/reusable cache keys. Invalidate on texture replacement, visibility, environment change and mesh reload. Never reuse a bind group after one of its texture views changes.

### PERF-02 — GPU memory scales with each texture-set Engine

Location: src/app/texture_sets.rs::rebuild_sets and src/engine.rs::Engine::with_pipelines.

Each set owns an Engine. Each Engine allocates position, normal and tangent maps, accumulation targets, stroke targets, mesh maps, custom-map resources, buffers and cache state. Immutable render pipelines are already shared through Arc<Pipelines>, which is good; the large per-set texture/buffer state is not shared.

Next action: capture per-set VRAM and peak memory at the actual document resolution. Consider lazy allocation for inactive sets, sharing truly immutable mesh-wide resources and releasing derived caches that can be regenerated. Do not merge per-set paint textures or bake maps if that changes texture-set semantics.

### PERF-03 — Blocking GPU readback in export and sampling

Location: src/gpu.rs::read_texture, src/export.rs::snapshot, src/engine.rs::sample_final.

gpu::read_texture submits a copy, maps a staging buffer and calls device.poll(wait_indefinitely). Export snapshot calls this separately for final channels, mesh maps, world normal and position before the export worker starts. The worker only handles CPU conversion and file writing after the blocking readbacks. sample_final also creates a staging buffer and waits for a one-pixel read.

Next action: batch export texture copies into one encoder and map staging buffers asynchronously; pass owned byte arrays to the export worker. Use an asynchronous one-pixel request for sampling and update the UI when data arrives. Keep device operations on the owning thread and avoid waits inside a UI frame.

### PERF-04 — Save/open/autosave serialize and access files on the UI thread

Location: src/app/project_io.rs::write_project, autosave_if_due and open_project; src/app.rs::process_actions.

Autosave calls write_project directly. Save serializes mesh/layer/set data and writes the project synchronously. Open synchronously reads the whole file, parses the manifest/blob table, decodes mesh data and then rebuilds/uploads GPU resources. Action processing invokes these paths directly.

Next action: split project save/open into CPU preparation and GPU upload phases. Keep GPU readback on the GPU-owning thread, but move byte serialization, compression, disk I/O and decode to workers. Preserve atomic temp-file replacement, recovery, autosave semantics and project-v3 compatibility. Add interrupted-save and large multi-set round-trip tests first.

### PERF-05 — Smart Material thumbnails are loaded synchronously during startup

Location: src/app.rs::PainterApp::new calls src/app/smart_io.rs::list.

smart_io::list reads the Smart Material directory, opens each ZIP, decompresses thumb.png, decodes it and registers an egui texture in the constructor path. This is not part of the CPU shelf preview worker.

Next action: enumerate paths and decode thumbnails on a worker; send decoded pixel buffers back; create egui textures on the UI thread. Show placeholders until results arrive. Measure against the user's actual Smart Material directory.

### PERF-06 — Stale layer thumbnails can create a release-time GPU spike

Location: src/engine/thumbs.rs::Engine::update_thumbs, called from end_stroke.

Rebuilding is skipped during a stroke, which is good. On release, the method collects every stale Fill/masked-layer thumbnail and records all those render passes in one encoder. Fill thumbnails are 512x512 composites.

Next action: schedule a small number of thumbnails per frame after the stroke. Prioritize selected/visible layers and mask previews; keep a pending queue so all stale thumbnails eventually update. Measure maximum post-stroke frame time.

### PERF-07 — Graph runtime rebuilds plans and allocates debug strings during evaluation

Location: src/graph/runtime.rs::Runtime::evaluate.

Every evaluation rebuilds graph order, plan HashMap, per-node values/inputs vectors, reader counts and needed sets. The node cache hash includes format!("{values:?}"), allocating a debug string per node. In non-lean mode, rendered node thumbnails also allocate GPU textures and bind groups.

Next action: deterministic typed hashing for node parameters; reuse scratch structures where graph evaluation is frequent; profile dirty-node propagation before changing cache semantics. Preserve the lean texture pool and graph round-trip/preset tests.

### PERF-08 — Multi-set path tracing recopies changed map arrays

Location: src/raytrace.rs::RayTracer::update_copies.

For multiple sets, path tracing resamples changed engine maps into texture-array layers when generation/maps_version changes. This is cached between edits, but an edit can trigger several map copies and per-layer texture-view creation. It matters only when ray tracing is enabled.

Next action: instrument copied bytes/passes. Reuse array-layer views and update only changed channels/sets if the version/dependency model supports it. Do not prioritize this ahead of normal viewport/brush responsiveness.

### PERF-09 — Startup still has synchronous work before the first usable frame

Location: src/app.rs::PainterApp::new.

The constructor synchronously builds the initial Engine (including UV bake), creates the ViewRenderer, decodes/uploads the default environment, renders procedural brush/texture thumbnails and lists user Smart Materials. Some startup spans exist and some preview work is already worker-backed. Use MP_STARTUP_TRACE spans before moving work to workers; GPU resource creation must remain on the device-owning thread.

## P2 — architecture, tests and cross-system considerations

### ARCH-01 — Core orchestration modules are large

app.rs (~5.4k lines), engine.rs (~5.2k lines), bake.rs (~3k lines) and graph_ui.rs (~1.7k lines) each span multiple responsibilities. UI panels are already partly split, so this is not a request for a wholesale rewrite. Further splits should follow stable boundaries: document/history, GPU compositing, mask evaluation, viewport resource cache, project serialization and startup services. Move one boundary at a time with tests.

### TEST-01 — GPU regression coverage does not measure owner-machine performance or actual Blender parity

The workflow runs a focused Mesa Lavapipe suite, and Windows CI runs build/unit tests. This is useful regression coverage, but it does not measure Vega 8 driver cost, six-set VRAM peaks, post-stroke frame spikes or the owner's real FBX shading. Some GPU tests are ignored and only those named in gpu-regression.yml are executed there.

Next action: create a representative benchmark project within the agreed 50k–100k triangles / approximately six sets / six UV maps envelope, with realistic Paint/Fill layers, nested masks, filters and history. Capture F6 frame time, dirty/scissor pixels, mask cache hits/misses, thumbnail passes and VRAM per set.

### DATA-01 — Do not weaken project durability to improve responsiveness

The current save path uses atomic temporary-file replacement and versioned manifest/blob handling. Any async refactor must preserve project-v3 compatibility, recovery/autosave semantics, set mapping, and safety of worker-owned CPU snapshots. Do not free/reuse GPU data while a worker still reads its CPU snapshot.

### CAM-01 — Raw mouse-look fix

The first-person mode originally read pointer delta while the cursor was locked. The current branch now uses PointerState::motion() (raw device motion), falling back to pointer delta when raw motion is unavailable. egui 0.37 documents motion() as unfiltered movement not clamped by screen edges. Camera math tests cover position-preserving look and translation-preserving movement. CI/GPU/Windows workflows for the audit commit are running; user validation is still required.

CAMERA DESIGN CAVEAT: OrbitCam represents a pose as target + yaw/pitch + distance. First-person look moves the target to a point in front of the camera to preserve its eye position. On exit, orbit mode therefore inherits this new look-at target. If the owner still finds the pivot unintuitive after mouse-look works, keep a separate first-person pose or explicitly preserve/restore the orbit pivot. Do not change this until the owner tests the raw-motion fix.

## Foundations to preserve

- BVH-based Screen stroke candidate selection and reusable candidate-index storage.
- Per-face Screen-to-UV stamping and normal-aware cached paint depth.
- Sparse UV scissors, conservative cap/merge, incremental composite prefix path.
- Explicit mask cache invalidation.
- Thumbnail invalidation semantics and skip-during-stroke behavior.
- Shared immutable Engine pipelines via Arc<Pipelines>.
- Lazy ray-tracer creation and generation/maps-version cache for multi-set copies.
- CPU shelf preview worker, throttled shelf GPU work, startup trace.
- Async undo snapshot mapping/compression (immediate undo may still wait for pending readback).

## Work order for next session

Follow AGENTS.md and PROJECT_SCOPE.md: Phase 1 before broad Phase 2 optimization.

1. Normal parity: reproduce the actual model's shading error from captured geometry/tangent/UV data. No global flips. This is the single focused, high-difficulty Claude assignment.
2. Owner-machine baseline: use the same representative project and F6 before/after each optimization. Capture startup trace and per-set VRAM on the actual GPU.
3. Phase 1 closure: verify 3D and UV strokes, Vertex ID, painted masks, nested folders/filters, cache invalidation, undo/redo and save/reopen in realistic stacks.
4. Only then Phase 2: prioritize by measured impact — viewport bind-group cache, batched async readback, per-set resource memory, post-stroke thumbnail scheduling, async project I/O and startup Smart Material loading.
5. Leave PR #3 and PR #4 untouched unless the owner explicitly authorizes a change.
