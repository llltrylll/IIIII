# Guide for AI coding agents

Read LOCAL_CONTEXT.md if it exists in the working copy. It is machine-local, ignored by Git, and contains the owner's private project scope and hardware details. Never create, commit, upload, or quote its contents into tracked files, PRs, issues, logs, or external services. If absent, ask the owner for missing private context rather than guessing. This file is the shared operating guide for GPT, Claude, Kimi, and any other coding agent. Humans: see CONTRIBUTING.md too.

## Current priority — establish the production foundation

**Phase 1: finish and validate the core painting and mask systems.** Focus on 3D brush latency and spikes, UV/2D painting correctness, Vertex ID and painted masks, smart-mask generators, mask layers/folders/filters, cache invalidation, incremental recomposition, and the CPU/GPU work directly involved. Validate realistic layer/mask/filter stacks and the representative production workload, not empty or nearly unpainted models.

Phase 1 requires relevant unit and GPU tests, correct rendered output, and runtime validation on the owner's actual machine. F6/performance measurements must be comparable. CI compilation alone is not proof. A GPU test marked ignored is not validated unless it was actually run on a GPU-capable setup; state any remaining gap.

**Phase 2: optimize the rest of the production foundation only after Phase 1 is validated.** Then audit memory, CPU allocations, GPU resource lifetime/reuse, viewport, baking, import, project I/O, export, background work, and architecture inside the defined production workload envelope.

Do not jump to broad cleanup while core painting/masking remains unresolved. Do not let hypothetical huge projects or cosmetic HUD work displace the foundation.

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


## Ten-branch policy — hard limit

The remote repository must contain **exactly ten long-lived branches**. All agents (GPT, Claude, Kimi, Copilot or others) must respect this fixed layout. Never create an eleventh branch, personal agent branch, ad-hoc feature branch, build branch or temporary test branch.

### Four code/recovery branches

1. Active development: `perf/vulkan-fill-color-fbm-cache-20261120`. Main integration branch. The only branch for production-ready source changes.
2. Backup: `perf/vulkan-cache-claude-ui-20261119`. Last-known recovery point immediately before the active line.
3. Backup-2: `feat/composite-uv-diagnostics-20261118`. Older stable recovery point, one generation behind Backup.
4. Original/fork baseline: `master`. Preserve as the reference baseline; do not use it for feature development.

### Three diagnostics/report branches

5. Owner's real-machine tests: `diagnostics/user-benchmarks`. Store sanitized snapshots and reports from the owner's actual tests: F5/F6 readings, FPS/frame time, layer and texture-set counts, texture resolutions, undo/readback observations, bake results, workload description, app version, GPU/backend/driver and comparable before/after runs.
6. Application diagnostics: `diagnostics/app-runtime`. Store application-generated diagnostics and reproducible technical evidence: profiler stage timings, GPU pass counts, cache hits/misses, resource-allocation estimates, crash/trace excerpts, regression reproductions and diagnostic-system findings. Clearly distinguish app estimates from physical GPU/VRAM telemetry.
7. Engineering reports: `reports/engineering`. Store consolidated audits, analysis, conclusions, validation matrices, regression summaries, agent handoffs and decisions. Link reports to commits/PRs and diagnostic snapshots.

### Three test/recovery branches

8. Test lane A: `test/agent-lab-a`. Isolated experiments and regression validation by one assigned agent.
9. Test lane B: `test/agent-lab-b`. A second independent test lane for another agent or a separate experiment. Assign one owner per lane; do not have two agents edit the same lane simultaneously.
10. Test backup: `backup/test-lab`. Preserve the last known-good test-lane state before a risky experiment, reset or replacement. It is a recovery snapshot, not a second active implementation lane.

### Diagnostics and test branch rules

- Branches 5–7 are evidence/report-only. No production source-code implementation there.
- Branches 8–9 may contain experimental code, prototypes and regression tests. They are not automatically production-approved. A successful result must be reviewed and ported/integrated into branch 1 with relevant tests.
- Branch 10 stores a recovery snapshot of useful test work. Update it only before replacing/resetting a test lane or when the owner explicitly requests a checkpoint.
- Test lanes must not become permanent agent-owned feature branches. Use them for bounded experiments and then port, discard, or archive the result within the ten-branch limit.
- No new branches for individual agents, PRs, temporary builds or diagnostics.

### Privacy and evidence quality

- Sanitize every diagnostic report before pushing. The fork is public: exclude local file paths, personal identifiers, credentials, project file contents, private assets and other machine-local information.
- Keep full unredacted history, local paths and personal project details only in the owner's ignored `LOCAL_CONTEXT.md` or local files.
- Use dated filenames and identify app version, test scenario, workload, capture mode and units.
- Mark evidence as CI, software Vulkan, other hardware, or owner's Vega 8 runtime. CI compilation is not real-GPU validation.
- Comparisons must disclose differences in layer count, resolution, texture sets, viewport mode, F6 state and other workload factors. Do not claim causation from non-equivalent runs.
- Treat application allocation estimates separately from measured physical VRAM.

### Branch maintenance and agent workflow

- All production-ready implementation goes to branch 1. Test-only implementation goes to branch 8 or 9, according to explicit task ownership.
- GPT, Claude and Kimi may use the two test lanes, but only one agent owns each lane/subsystem at a time.
- Backups, diagnostics/report branches and `master` are read-only except for their defined checkpoint/report purpose.
- Never delete, reset, force-push or rewrite any branch. Only the owner performs branch cleanup.
- Before deleting surplus branches, preserve useful unique commits or explicitly classify them as disposable. After owner cleanup, verify exactly the ten canonical branch names remain.

## Owner-local data boundary

LOCAL_CONTEXT.md is intentionally ignored by Git. It is the only place for personal project details, exact computer specifications, local paths, project files, and machine-specific performance baselines. Keep confirmed and unknown hardware fields distinct. Never invent CPU, RAM, GPU memory, storage, or driver details. Do not put personal project names, private workflow notes, local paths, or machine specifications in tracked documentation. The repository is public; all tracked content, commits, PRs, and CI logs must be treated as public.
