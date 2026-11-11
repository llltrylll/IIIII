# Contributing to this fork

This repository is a personal-use fork of [Material Painter](https://github.com/alla2001/material-painter). Its custom changes are maintained separately from upstream. Keep the upstream credits, license, third-party notices and SubCraft attribution intact.

## Build and test

Install stable Rust from [rustup.rs](https://rustup.rs), then clone this fork:

```sh
git clone https://github.com/llltrylll/IIIII.git
cd IIIII
cargo build --locked
cargo test --locked
```

On Linux, the windowing development packages may be needed:

```sh
sudo apt install pkg-config libxkbcommon-dev libwayland-dev libxcb-render0-dev \
    libxcb-shape0-dev libxcb-xfixes0-dev libgtk-3-dev
```

GPU tests render offscreen and require a compatible GPU:

```sh
cargo test --locked --bin material_painter -- --ignored --test-threads 1
```

They write images to `target/bake_test/`. For rendering, baking, painting, node or export changes, inspect those images; a passing test alone does not confirm the visual result. CI builds with warnings denied on Windows, Linux and macOS.

## Change guidelines

- Keep changes small, focused and easy to review. Avoid unrelated formatting or refactors, especially in the large UI and rendering modules.
- Preserve the fork's personal changes unless the task specifically asks to revise them: brush color controls, layer copy/paste and shortcut behavior.
- Follow [AGENTS.md](AGENTS.md) for repository architecture, conventions and commands.
- Every document edit should use the existing undo-aware engine operations.
- Keep saved projects backward-compatible when changing project data; review `src/app/project_io.rs` and its version handling.
- Avoid blocking the UI thread on disk, decoding or GPU readback. Follow the worker-thread and frame-pumping patterns already in the code.
- Do not add dependencies without a concrete reason. Keep builds warning-free.
- Say what was built and tested. For visible changes, include a screenshot or short clip when sharing the change.

AI-assisted changes are welcome, but review the diff, run the app and tests, and understand the behavior before relying on a change. The upstream project and its policies live at [alla2001/material-painter](https://github.com/alla2001/material-painter).

## License

Contributions to this fork remain under [GPL-3.0-or-later](LICENSE), subject to the upstream project's license and notices.
