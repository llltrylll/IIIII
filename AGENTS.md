# Guide for AI coding agents

Read LOCAL_CONTEXT.md if it exists in the working copy. It is machine-local, ignored by Git, and contains the owner's private project scope and hardware details. Never create, commit, upload, or quote its contents into tracked files, PRs, issues, logs, or external services. If absent, ask the owner for missing private context rather than guessing. This file is the shared operating guide for GPT, Claude, Kimi, and any other coding agent. Humans: see CONTRIBUTING.md too.

## Current priority — 60 FPS frame-time campaign

**Owner's current top priority: make the editor sustain 60 FPS (16.67 ms/frame) on the owner's real machine, starting from the reported ~20 FPS while the program is open.** This priority supersedes the previous Phase 1 / Phase 2 ordering below. Do not postpone frame-time work behind broad painting-feature completion.

The goal is not a benchmark spike. Target a stable 60 FPS in representative normal editing conditions, including idle editor, loaded production project, camera interaction, brush interaction, and normal viewport effects. Record separate results for each scenario. Do not promise the target until measured on the owner's Vega 8 machine.

### Required first step: establish the bottleneck

Before broad rewrites, build a reproducible frame-time baseline and identify the dominant costs. Separate:
- UI/app CPU time per frame;
- viewport CPU preparation and cache-key work;
- GPU command submission and GPU execution;
- driver/device polling and synchronization;
- background jobs, async completion polling, and repaint scheduling;
- memory allocation, uploads, readbacks, and resource churn.

Use F6/F5 and existing startup tracing where appropriate, but do not mistake paint/composite timings for total frame time. Add low-overhead stage instrumentation where the current measurements cannot isolate a bottleneck. Instrumentation must be disabled or cheap in normal use.

The first investigation must explicitly test idle behavior: whether the app continuously repaints, which subsystems request repaints, whether viewport cache hits are actually occurring, and whether the app does unnecessary CPU/GPU work while visually unchanged. The existing 3D/2D viewport caches are a starting point, not proof that the whole frame is cheap.

### Implementation mandate

The owner explicitly authorizes substantial architectural changes to the frame loop, repaint scheduling, viewport/render path, GPU submission, and background work when measurements justify them. Prefer changes capable of materially reducing frame time over cosmetic or sub-millisecond micro-optimizations. Do not preserve an inefficient architecture merely to keep diffs small.

Potential avenues to investigate (not assumptions or permission to implement blindly):
- demand-driven/dirty-driven rendering and repaint scheduling;
- avoiding unconditional per-frame work and redundant GPU polling;
- separating UI updates from expensive viewport refreshes;
- eliminating repeated cache-key construction and per-frame allocations;
- reducing GPU pass count, synchronization, unnecessary uploads/readbacks, and resource recreation;
- moving expensive work off the UI thread or incrementally budgeting it;
- adaptive quality only where it does not compromise texture correctness or the user's expected visual output.

### Hard invariants

- Preserve exact visual output and shading unless the owner explicitly approves a quality tradeoff. Existing pixel-regression failures are release blockers, not acceptable collateral.
- Keep old projects loadable. Do not alter project schemas casually; any schema change requires migration and compatibility tests.
- Protect user data. No destructive project operations, silent data loss, unbounded caches, uncontrolled GPU allocation, or unbounded background queues.
- Keep painting responsive: brush strokes, UV painting, undo/redo, masks, generators, baking, and export must remain correct.
- Never block the UI thread on disk, decoding, expensive CPU work, GPU readback, or a wait-for-idle operation in the normal frame loop.
- Do not remove GPU polling, callbacks, cache invalidation, or repaint requests until their lifecycle requirements are understood and tests cover them.
- No sleeps, frame-counting, or timeout-based timing hacks.
- WGSL must compile on DX12, Vulkan, and Metal. Shader compilation under Naga alone is not visual validation.
- Builds must remain warning-free under -D warnings; avoid unnecessary dependencies.

### Measurement and acceptance

For every optimization, record:
1. baseline and post-change measurements from equivalent workloads;
2. average, median, p95 and worst/recent-peak frame time where available;
3. CPU frame time separately from GPU time;
4. GPU/backend/driver and viewport resolution;
5. workload state (project, triangle count, texture sets, UV maps, texture resolution, layers, masks, effects, workspace);
6. correctness tests and any remaining owner-machine validation.

A 60 FPS claim requires sustained frame intervals at or below 16.67 ms in the stated scenario, not a momentary FPS counter. Report idle and active-editing results separately. If CI hardware differs from the owner's Vega 8, label it clearly. CI passing is not proof of a real-machine FPS gain.

### Execution order

1. Capture current HEAD, open PRs, branch state, CI and diffs. Preserve the owner's master-nosso-reserva-2026-11-20 recovery anchor as read-only.
2. Establish the idle and representative-workload frame-time baseline; inspect repaint sources, app-frame stages, viewport cache hit/miss behavior, and GPU submission/polling.
3. Identify the top one or two measured bottlenecks. Write a short design note with expected gain, risks, invalidation rules and rollback strategy.
4. Implement the highest-impact bounded change first; add regression tests and compare frame-time evidence.
5. Continue through the next measured bottleneck. Do not accumulate unrelated micro-optimizations.
6. Validate build, unit tests, relevant GPU tests, output images, project-open compatibility, and real-machine behavior. Clearly mark anything not tested.

## Previous production-foundation requirements (remain mandatory)

The former painting/masking foundation remains a correctness requirement even though it is no longer the gating priority for performance work. Continue to protect 3D brush latency and spikes, UV/2D painting correctness, Vertex ID and painted masks, smart-mask generators, mask layers/folders/filters, cache invalidation, incremental recomposition, and CPU/GPU work directly involved. Validate realistic layer/mask/filter stacks and the representative production workload, not empty or nearly unpainted models.

## Target workload

Personal production fork for Roblox asset texturing on the owner's machine:
- Normal production workload: 50,000–100,000 triangles, around 6 MeshParts and 6 UV maps.
- Absolute ceiling: 200,000 triangles, around 6 MeshParts and up to 12 UV maps.
- Air Fox is an example asset, not a mandatory benchmark or special target.
- Realistic production layer stacks, masks, and filters within that envelope.

Projects around 100 items are a very distant possibility, not a current target. A major HUD redesign is deferred for months or until the owner explicitly reprioritizes it. Do not optimize beyond the stated envelope without approval.

The owner explicitly permits major, risky, or breaking internal refactors when they materially improve production reliability and performance. Do not avoid a needed rewrite just to keep a diff small. Protect user data, add tests, preserve project compatibility deliberately, inspect rendered results, and be explicit about unvalidated runtime behavior.

## Shared work across GPT, Claude, and Kimi

GPT, Claude, Kimi, and any other coding agent share this repository and must read this file. Claude must also read CLAUDE.md. Read the ignored LOCAL_CONTEXT.md only from the owner's local working copy; never push it.

At the start of each task, inspect current branch, HEAD SHA, open PRs, changed files/diff, CI, and the other agent's recent commits/comments. Re-fetch files and current SHAs before writing. Never assume previous chat context equals current repository state.

Do not concurrently modify the same files/subsystem. Establish task ownership in the active PR or a concise handoff comment. Never overwrite the other agent's work. Do not create agent-specific branches; all production code work stays on the active branch and all diagnostic evidence uses the three designated evidence branches. Commit coherent units and leave a handoff stating changes, invariants, tests, gaps, and next steps. Keep code understandable for the other agent; no agent-specific abstractions.

## What this is

Material Painter is a PBR texture painter (in the spirit of Substance Painter) in **Rust**, with **eframe/egui 0.37** for the UI and **wgpu 30** for all GPU work. One binary crate, edition 2024. Goals: fast, low memory, and artist-first UI.

## Commands

```sh
cargo build                     # debug build (dependencies are optimized, see Cargo.toml)
cargo test                      # unit tests; CI runs these with RUSTFLAGS="-D warnings"
cargo test --bin material_painter -- --ignored --test-threads 1   # GPU tests, offscreen
cargo test --bin material_painter <name> -- --ignored               # one GPU test
```

GPU tests render without a window and write images to `target/bake_test/`.

For frame-time diagnosis on the owner's machine, launch with `MP_STARTUP_TRACE=1` (or `MP_STARTUP_TRACE=<file>`) and optionally `MP_STARTUP_FRAMES=300`. The report includes per-frame UI work, frame-to-frame cadence, named spans and viewport cache hit/miss counters. Compare equivalent workloads and share the sanitized report through `diagnostics/user-benchmarks` or `reports/engineering`; never include local paths or private project data.
Look at the images after changing rendering code; a passing test alone does
not prove the picture is right. `ui_screenshots` renders the real UI
offscreen with egui_kittest into `target/bake_test/ui/`.

## Map of the code

| Where | What |
|---|---|
| `src/main.rs` | Entry point, window options, panic hook (`crash.rs`) |
| `src/app.rs` + `src/app/` | The UI shell: workspaces (Material Paint, Material Designer, Shader Maker), docking (`dock.rs`), panels, input, actions (`Action` enum, processed once per frame), projects (`project_io.rs`), preferences, export window, welcome window |
| `src/app/{properties_ui,layers_ui,shelf_ui,view_ui,windows_ui,mask_ui,bake_jobs,color_ui}.rs` | UI implementation split by area: Properties, Layers, Shelf, 3D/2D viewports, windows, masks and Vertex ID plumbing, bake jobs, and colour pickers |
| `src/app/vertex_id.rs` | Vertex Color ID mask: palette, per-corner vertex colors, overlay, and legacy FBX/OBJ re-import |
| `src/engine.rs` | Texture-set engine: layer stack, masks/filters, painting, GPU compositing and undo hooks |
| `src/app/texture_sets.rs` | One `Engine` per mesh material/texture set |
| `src/cage.rs`, `src/app/cage_paint.rs` | Bake cage and cage painting |
| `src/polyfill.rs` | Polygon-fill selection: triangles, quads, loops, flat areas, parts, objects, UV islands and texture sets |
| `src/bake.rs`, `src/bake_cache.rs` | Mesh-map baking and on-disk bake cache |
| `src/viewport.rs`, `src/post.rs`, `src/raytrace.rs` | 3D/2D viewport rendering, post-processing and path tracer |
| `src/graph/` | Material graph: node definitions, WGSL, GPU runtime and presets |
| `src/shaders/*.wgsl` | Engine shaders for compositing, stamps, baking and views |
| `src/presets.rs` | Smart materials, materials and mask generators |
| `src/mesh.rs`, `src/import.rs` | Mesh data, primitives, OBJ/FBX import, tangents |
| `src/export.rs` | Texture export templates and file writing |
| `src/undo.rs` | Undo/redo |
| `src/mcp.rs`, `src/app/mcp_tools.rs` | Optional MCP server for app control (off by default) |

## Conventions

- No sleeps, frame-counting, or timeout-based timing hacks. Use existing worker/channel, GPU completion callback, repaint, and action-queue patterns.
- Never block the UI thread on disk, decoding, expensive CPU work, or GPU readback. Use worker threads and incremental GPU work where appropriate.
- Every document edit must go through undo-aware engine paths.
- Preserve dirty-region updates; do not replace partial recomposition with full recomposition without measured justification.
- Keep old projects loadable when changing project data formats; design and test migrations deliberately.
- WGSL must compile on DX12, Vulkan, and Metal. A shader compiling under Naga is not enough; run GPU tests and inspect images.
- Keep builds warning-free under `-D warnings`; do not silence unfinished code with `#[allow(dead_code)]`.
- Avoid new dependencies without a clear, documented need.
- Keep cross-platform paths and settings correct.

## Common mistakes

- Reformatting unrelated code or rejecting a justified subsystem rewrite just because the diff is large.
- Adding abstractions for one-off operations without a real subsystem boundary.
- Reimplementing existing GPU, theme, or engine edit helpers.
- Treating shader compilation as visual validation.
- Fixing 3D painting while breaking UV painting, path tracing, masks, undo/redo, or project persistence.

## Before finishing

1. Run `cargo build` with no warnings and `cargo test`.
2. Run relevant GPU tests for rendering, baking, painting, graph, masks, and export changes.
3. Inspect GPU test output images.
4. Run the app when possible; state clearly when owner-machine testing is still required.
5. Report measurements, test status, and unresolved risk. Never claim unperformed tests.
6. Ensure the diff contains only task-related changes.


## Current branch and recovery policy

Repository branch inventory has accumulated historical agent/test branches; do not assume the old ten-branch inventory is current, and do not delete or clean up branches.

- Active frame-time/performance work: perf/postfx-bind-group-cache-20261120 (PR #26). Continue the current task here unless the owner explicitly changes the active branch.
- PR #26 is stacked on perf/vulkan-fill-color-fbm-cache-20261120. Do not merge it independently of its base; preserve the stack relationship.
- Immutable owner recovery anchor: master-nosso-reserva-2026-11-20. It was verified identical to the working tree when created. Never commit, force-push, reset, or otherwise modify it.
- Existing diagnostic evidence branches remain report-only: diagnostics/user-benchmarks, diagnostics/app-runtime, reports/engineering.
- Existing test lanes test/agent-lab-a and test/agent-lab-b are reserved for bounded experiments only. Do not create more branches for individual agents or experiments.
- Never delete, reset, force-push, or rewrite any existing branch. Only the owner performs branch cleanup.
- Before any write, re-fetch the target branch HEAD and file SHA. Do not overwrite another agent's newer work.

### Evidence and privacy

- Sanitize diagnostic reports before pushing. The repository is public: exclude local file paths, personal identifiers, credentials, project file contents, private assets and machine-local information.
- Mark evidence as CI, software Vulkan/Lavapipe, other hardware, or owner's Vega 8 runtime. CI compilation is not real-GPU validation.
- Keep app allocation estimates separate from physical GPU/VRAM telemetry.

### Multi-agent coordination

- GPT, Claude, Kimi and other agents must read this file. Claude must also read CLAUDE.md.
- At task start, inspect current branch, HEAD SHA, open PRs, changed files/diff, CI, and other agents' recent commits/comments.
- Do not concurrently modify the same files/subsystem. State ownership in the active PR or a concise handoff comment.
- Commit coherent units and leave a handoff with changes, invariants, tests, gaps and next steps.

## Owner-local data boundary

LOCAL_CONTEXT.md is intentionally ignored by Git. It is the only place for personal project details, exact computer specifications, local paths, project files, and machine-specific performance baselines. Keep confirmed and unknown hardware fields distinct. Never invent CPU, RAM, GPU memory, storage, or driver details. Do not put personal project names, private workflow notes, local paths, or machine specifications in tracked documentation. The repository is public; all tracked content, commits, PRs, and CI logs must be treated as public.
