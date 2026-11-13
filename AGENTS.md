# Guide for AI coding agents

Read [PROJECT_SCOPE.md](PROJECT_SCOPE.md) first. It is the source of truth for the production objective, Air Fox workload ceiling, mandatory phase order, engineering latitude, shared GPT/Claude coordination, and repository boundaries. All instructions here are subordinate to that scope. Humans: see [CONTRIBUTING.md](CONTRIBUTING.md) too.

## Current priority — establish the production foundation

**Phase 1: finish and validate the core painting and mask systems.** Focus on 3D brush latency and spikes, UV/2D painting correctness, Vertex ID and painted masks, smart-mask generators, mask layers/folders/filters, cache invalidation, incremental recomposition, and the CPU/GPU work directly involved. Validate realistic layer/mask/filter stacks and the Air Fox production workload, not empty or nearly unpainted models.

Phase 1 requires relevant unit and GPU tests, correct rendered output, and runtime validation on the owner's actual machine. F6/performance measurements must be comparable. CI compilation alone is not proof. A GPU test marked ignored is not validated unless it was actually run on a GPU-capable setup; state any remaining gap.

**Phase 2: optimize the rest of the production foundation only after Phase 1 is validated.** Then audit memory, CPU allocations, GPU resource lifetime/reuse, viewport, baking, import, project I/O, export, background work, and architecture inside the Air Fox workload envelope.

Do not jump to broad cleanup while core painting/masking remains unresolved. Do not let hypothetical huge projects or cosmetic HUD work displace the foundation.

## Target workload

Personal production fork for Roblox asset texturing on the owner's machine:
- Air Fox normal target: 50,000–100,000 triangles, 6 MeshParts, 6 UV maps.
- Absolute ceiling: 200,000 triangles, up to 6 MeshParts and 12 UV maps.
- Realistic production layer stacks, masks, and filters within that envelope.

Projects around 100 items are a very distant possibility, not a current target. A major HUD redesign is deferred for months or until the owner explicitly reprioritizes it. Do not optimize beyond the stated envelope without approval.

The owner explicitly permits major, risky, or breaking internal refactors when they materially improve production reliability and performance. Do not avoid a needed rewrite just to keep a diff small. Protect user data, add tests, preserve project compatibility deliberately, inspect rendered results, and be explicit about unvalidated runtime behavior.

## Shared work with GPT and Claude

Both agents work in the same repository and must read [PROJECT_SCOPE.md](PROJECT_SCOPE.md) and this file. Claude must also read [CLAUDE.md](CLAUDE.md).

At the start of each task, inspect current branch, HEAD SHA, open PRs, changed files/diff, CI, and the other agent's recent commits/comments. Re-fetch files and current SHAs before writing. Never assume previous chat context equals current repository state.

Do not concurrently modify the same files/subsystem. Establish task ownership in the active PR or a concise handoff comment. Never overwrite the other agent's work. Commit coherent units and leave a handoff stating changes, invariants, tests, gaps, and next steps. Keep code understandable for the other agent; no agent-specific abstractions.

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
