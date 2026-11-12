# Guide for AI coding agents

Read `PROJECT_SCOPE.md` first. It defines the actual product objective, target workload, non-goals, and the owner's authorization for substantial refactors. Those priorities govern decisions in this fork. Humans: see [CONTRIBUTING.md](CONTRIBUTING.md) too.

## Active fork objective

This is a personal production fork of Material Painter, primarily for texturing Roblox assets on the owner's own machine. The representative target is an asset with 6 MeshParts, 6 UV maps, and 50,000–100,000 triangles. The maximum planned target is 12 UV maps and 200,000 triangles. The expected workload includes realistic layer stacks, masks, and filters.

Prioritize correctness, data integrity, stability, and responsive work inside that envelope. Do not spend time optimizing hypothetical 100-item projects, extreme asset sizes, universal hardware coverage, public distribution, or cosmetic HUD redesign ahead of the core. The owner explicitly allows substantial or risky refactors when they are the best way to make the program work well; validate them carefully rather than avoiding them for the sake of small diffs.

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
| `src/app/{properties_ui,layers_ui,shelf_ui,view_ui,windows_ui,mask_ui,bake_jobs,color_ui}.rs` | `impl PainterApp` split by area: Properties panel, Layers tab, Shelf, 3D/2D viewports, About/Crash/Help windows, mask Properties (generators, mask layers, filters, vertex-ID plumbing), bake jobs (`start_bake_*`, `poll_bake`; a "vertex colors only" bake keeps every other map via `Engine::replace_id_map`), and the colour pickers (`color_button`, the Color HUD, HSV helpers). Methods are `pub(super)`; private items of `app.rs` are visible to them |
| `src/app/vertex_id.rs` | Vertex Color ID mask: palette of the mesh's vertex colors (`Mesh::tri_vertex_color`, per corner, saved in projects), up to `MAX_ID_COLORS` ID colors per smart mask, the model overlay of the mask (`OverlaySlot::VertexColors`) and re-importing colors from the FBX/OBJ for old projects |
| `src/engine.rs` | One texture set: layer stack (paint, fill, folder), masks and mask filters, GPU painting and compositing, undo hooks. Eight paint channels (`CHANNELS`): opacity (`OPACITY`) composites into the final base color's alpha; normal (`NORMAL`) holds tangent-space detail that each layer adds to the stack below (fill normal-map images projected in world space go through the baked tangent map); specular (`SPECULAR`) is the specular level of non-metals (0.5: 4%) |
| `src/app/texture_sets.rs` | Texture sets: one `Engine` per material of the mesh (`Mesh::sets`); the edited one is `PainterApp::engine`, the others are parked. Views, bakes, export and projects cover every set. Hiding a set rebuilds the picking BVH without it; highlights ride in the view globals (`SetLook`) |
| `src/cage.rs`, `src/app/cage_paint.rs` | The bake cage: projection directions (averaged over split vertices), the probe behind the live cage preview, Auto-fix, and the cage brush (`BakeSettings::cage_paint`, a front/rear multiplier per vertex) |
| `src/polyfill.rs` | Polygon fill selection: triangles, quads, face loops, flat areas, parts, objects, UV islands, texture sets; lasso test |
| `src/bake.rs` | Mesh-map baking: CPU scene prep on a worker thread, then GPU passes pumped a few tiles per frame |
| `src/bake_cache.rs` | On-disk cache of baked maps, keyed by mesh, settings and the bake code itself |
| `src/viewport.rs`, `src/post.rs`, `src/raytrace.rs` | 3D and 2D view rendering, post-processing, path-traced preview |
| `src/graph/` | Material graph: node definitions (`def.rs`, WGSL files with a comment header), GPU runtime, presets; built-in nodes live in `src/graph/nodes/*.wgsl` |
| `src/shaders/*.wgsl` | Engine shaders (compositing, painting stamps, baking, views) |
| `src/presets.rs` | Smart materials, materials and mask generators on the shelf |
| `src/mesh.rs`, `src/import.rs` | Meshes, procedural primitives, OBJ/FBX import, MikkTSpace tangents |
| `src/graph/shading.rs`, `src/graph/nodes/shading/`, `src/app/preview_shader.rs` | Preview shaders: shading graphs compiled into one `custom_shade(c: Shade)` for the 3D view (`Shade` and the hook are in `view.wgsl`); parameters are a uniform array |
| `src/app/path_tool.rs` | Path tool: editable curves on the surface or UV map, stamped through the normal stroke pipeline |
| `src/library.rs`, `src/app/library_ui.rs` | Asset library: Poly Haven / ambientCG catalogs and downloads, the community library (`alla2001/material-painter-community`), sharing graphs; network work on worker threads |
| `src/export.rs` | Texture export templates (engine presets) and file writing |
| `src/undo.rs` | Undo/redo of every document edit |
| `src/mcp.rs`, `src/app/mcp_tools.rs` | Optional MCP server so AI agents can drive the app (off by default) |
| `src/theme.rs`, `src/icons.rs` | UI theme (sharp corners, grey palette, blue accent) and Phosphor icon glyphs |

## Conventions (follow these)

- **No timing hacks.** Never fix ordering problems with sleeps, frame counting or timeouts. Use the existing patterns: worker threads that send results over `mpsc` channels, `queue.on_submitted_work_done` callbacks, `ctx.request_repaint()`, and the `Action` queue in `app.rs`.
- **Never block the UI thread** on disk, decoding or GPU readback. Heavy work goes to a worker thread; GPU work is split so a frame stays fast.
- **Every document edit is undoable.** Go through the engine's existing edit functions (they record undo entries) rather than mutating layers directly.
- **Partial updates.** Painting and compositing redraw only dirty regions; use `mark_dirty()` / stroke regions as the surrounding code does instead of forcing full recomposites.
- **Projects must stay loadable.** If you change what a project saves, keep reading old files (see the version handling in `project_io.rs`).
- **WGSL must compile on every backend** (DX12 via FXC/DXC, Vulkan, Metal). FXC rejects some things Naga accepts, for example writing to a vector with a dynamic index; use `select` or a local array. Test with the GPU tests.
- **egui 0.37 API.** Many examples online are for older egui, and several names changed (for example `ctx.global_style()` instead of `ctx.style()`, and `egui::Panel` for side and top panels). When in doubt, copy the pattern from this codebase and let the compiler's deprecation warnings guide you.
- **UI text** is short, plain and specific: say what a control does, not how it works. Tooltips explain the non-obvious.
- **Comments explain why**, not what. Doc comments on modules and public items, like the existing ones. No commented-out code.
- **Warning-free** under `-D warnings`. No `#[allow(dead_code)]` to silence unfinished work.
- **No new dependencies** without a clear need; say why in the PR.
- **Cross-platform:** Windows, Linux and macOS. Don't hardcode paths; data lives in the settings folder (`crash::data_dir()`), assets are found with `environment::asset_dir`.

## Common agent mistakes here

- Avoid unrelated rewrites and formatting churn. However, do not reject a major refactor solely because it creates a large diff: this owner authorizes invasive changes when they materially improve reliability, architecture, or performance inside the target workload.
- Adding a new abstraction layer for a one-off. Prefer a small function next to the code that uses it unless a real subsystem boundary justifies the abstraction.
- Re-implementing something that exists: check `gpu.rs` helpers, `theme.rs` widgets (`theme::selectable`, `theme::pill`, `theme::card`), and the engine's edit functions first.
- Assuming a shader change works because it compiles: run the GPU tests and look at the output images.
- Forgetting the 2D (UV) view or the path tracer when changing how the 3D view draws something.

## Before you finish

1. `cargo build` with no warnings, and `cargo test` passes.
2. GPU tests pass if you touched rendering, baking, painting, nodes or export.
3. You tried the change in the running app, or clearly state when runtime validation requires the owner.
4. The diff contains only changes that serve the stated task.
