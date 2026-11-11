# Material Painter — personal fork

This is a personal-use fork of [Material Painter](https://github.com/alla2001/material-painter), an open-source PBR texture painter written in Rust. It keeps the upstream application and adds personal changes, including brush color controls, layer copy/paste across texture sets, and shortcut adjustments.

The fork is maintained separately. Upstream releases and support links refer to the original project and do **not** contain this fork's changes. Please retain the upstream attribution and bundled third-party license notices when modifying or sharing the code.

![Smart materials painted in Material Painter](assets/ui/welcome.jpg)

## What it does

Paint PBR materials onto 3D models with layers, smart materials and masks; bake mesh maps; build materials in a node graph; and write custom node shaders.

> **Early access.** The upstream application is under active development. This fork is for personal use; changes here may differ from upstream and are not part of its official releases.

## Features

- **Material Paint**: paint, fill and folder layers; blend modes and opacity; brushes with tips, grain, pen pressure, stabilizer, symmetry, and screen, surface or UV projection; polygon fill by triangle, quad, face loop, flat area, loose part, object, UV island or texture set.
- **Channels**: base color, roughness, metallic, height, opacity, emission, normal and specular level.
- **Texture sets**: separate layers, resolution, baked maps and exports for each material on a mesh.
- **Smart materials and masks**: ready-made materials, mesh-map-driven mask generators, mask layers and filters.
- **Mesh map baking**: ambient occlusion, curvature, cavity, thickness, normal, height, ID and other maps; supports high-to-low baking, cages and custom maps.
- **Material Designer and Shader Maker**: create tileable materials with a node graph and author custom WGSL nodes.
- **Export**: templates for common engines and tools, custom channel packing, and PNG, TIFF, TGA, JPEG and EXR output.
- **Path tool**: place points on the mesh or UV map and paint evenly along the curve.
- **Asset library**: free CC0 materials from Poly Haven and ambientCG, plus community-shared graphs.
- **Substance 3D Painter import**: imports the model, texture sets, flattened channel results and supported parts of the layer structure. See [the import notes](docs/substance-painter-import.md) for limitations.
- **Docking workspaces**, path-traced preview, HDRI lighting, and an optional MCP server for AI tools.

## Build and run this fork

Install stable Rust from [rustup.rs](https://rustup.rs), then:

```sh
git clone https://github.com/llltrylll/IIIII.git
cd IIIII
cargo run --release
```

On Linux, install the windowing development packages first:

```sh
sudo apt install pkg-config libxkbcommon-dev libwayland-dev libxcb-render0-dev \
    libxcb-shape0-dev libxcb-xfixes0-dev libgtk-3-dev
```

**Requirements:** 64-bit Windows 10/11 or Linux x64 (glibc 2.35+), and a GPU supporting DirectX 12 or Vulkan. Baking at high resolution can use several gigabytes of GPU memory; start with a lower texture resolution if memory is limited.

Run the unit tests with:

```sh
cargo test --locked
```

GPU tests are opt-in and need a compatible GPU:

```sh
cargo test --locked --bin material_painter -- --ignored --test-threads 1
```

They write rendered images to `target/bake_test/`; inspect the images as well as the test results after changing rendering code.

## Upstream project and credits

- **Original project, official releases, community and upstream issue tracker:** [alla2001/material-painter](https://github.com/alla2001/material-painter).
- **This fork and its history:** [llltrylll/IIIII](https://github.com/llltrylll/IIIII).
- **Support for the original developer:** [Ko-fi](https://ko-fi.com/allaeddinerakik).

This fork is not affiliated with or endorsed by Adobe. Substance 3D Painter is an Adobe trademark. The application is licensed under [GPL-3.0-or-later](LICENSE); third-party components are listed in [THIRD_PARTY_NOTICES.html](THIRD_PARTY_NOTICES.html). The `.spp` reader is adapted from [SubCraft](https://github.com/Aalzard/subcraft), under its MIT license; bundled HDRIs and PBR textures are from Poly Haven (CC0), and Phosphor icons are MIT-licensed.
