//! Offscreen previews: bake a mesh, render the real 3D viewport from several
//! cameras in several view modes and write the frames to
//! `target/bake_test/preview/*.png`. Nothing touches a window. Without
//! `MP_PREVIEW_MESH` the rounded cube is used.
//!
//! `MP_PREVIEW_MESH=path.fbx cargo test --bin material_painter preview -- --ignored --nocapture`

use crate::bake::BakeSettings;
use crate::bake::tests::{gpu_with_adapter, run_bake};
use crate::engine::{Engine, MAPS};
use crate::gpu::{GpuMesh, read_texture};
use crate::mesh::Mesh;
use crate::presets;
use crate::viewport::{OrbitCam, ViewMode, ViewRenderer, ViewSettings};
use eframe::{egui, egui_wgpu, wgpu};
use glam::Vec3;
use std::path::Path;
use std::sync::Arc;

fn render_state() -> egui_wgpu::RenderState {
    let (instance, adapter, device, queue) = gpu_with_adapter();
    let target_format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let renderer = egui_wgpu::Renderer::new(&device, target_format, Default::default());
    egui_wgpu::RenderState {
        available_adapters: vec![adapter.clone()],
        adapter,
        instance,
        device,
        queue,
        target_format,
        renderer: Arc::new(egui::mutex::RwLock::new(renderer)),
    }
}

fn map_mode(name: &str) -> ViewMode {
    ViewMode::Map(MAPS.iter().position(|m| m.name == name).unwrap_or_else(|| panic!("no map {name}")))
}

#[test]
#[ignore]
fn preview_mesh() {
    let mesh = match std::env::var("MP_PREVIEW_MESH").as_deref() {
        Ok("ball") => crate::mesh::Primitive::MaterialBall.build(),
        Ok(path) => Mesh::load_file(Path::new(path)).unwrap_or_else(|e| panic!("{path}: {e}")),
        Err(_) => crate::mesh::Primitive::RoundedCube.build(),
    };
    let rs = render_state();
    let gpu_mesh = GpuMesh::new(&rs.device, &mesh);
    let res = 2048;
    let mut engine = Engine::new(&rs, &gpu_mesh, (mesh.bounds_min, mesh.bounds_max), res);
    let (maps, report, _) = run_bake(&rs.device, &rs.queue, &mesh, &gpu_mesh, &BakeSettings::default(), res);
    engine.set_maps(maps, report);

    // A smart material that leans on the edge, cavity and AO maps.
    let smart_name = std::env::var("MP_PREVIEW_SMART").unwrap_or_else(|_| "Painted Steel Worn".into());
    let smart = presets::SMART_MATERIALS.iter().find(|s| s.name == smart_name).unwrap_or_else(|| panic!("no smart material {smart_name}"));
    crate::app::add_smart_material(&mut engine, &mut crate::undo::UndoStack::default(), smart, 0, None);
    engine.composite();

    // Cameras: the whole mesh, then close-ups of both ends of its long axis.
    let (lo, hi) = (mesh.bounds_min, mesh.bounds_max);
    let ext = hi - lo;
    let axis = if ext.x >= ext.y && ext.x >= ext.z {
        Vec3::X
    } else if ext.y >= ext.z {
        Vec3::Y
    } else {
        Vec3::Z
    };
    let center = (lo + hi) * 0.5;
    let half = ext.dot(axis) * 0.5;
    let side_yaw = if axis == Vec3::X { 0.0 } else { std::f32::consts::FRAC_PI_2 };
    let cams = [
        ("full", OrbitCam { target: center, dist: 3.4, yaw: side_yaw + 0.3, pitch: 0.25, ..Default::default() }),
        ("end_a", OrbitCam { target: center + axis * half * 0.75, dist: 1.1, yaw: side_yaw + 0.6, pitch: 0.35, ..Default::default() }),
        ("end_b", OrbitCam { target: center - axis * half * 0.75, dist: 1.1, yaw: side_yaw - 0.5, pitch: 0.3, ..Default::default() }),
    ];
    let modes = [
        ("material", ViewMode::Material),
        ("ao", map_mode("Ambient Occlusion")),
        ("curvature", map_mode("Curvature")),
        ("edge", map_mode("Edge Distance")),
        ("cavity", map_mode("Cavity")),
        ("normal", map_mode("Normal")),
    ];

    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test/preview");
    std::fs::create_dir_all(&dir).unwrap();
    let mut views = ViewRenderer::new(&rs);
    let env = crate::environment::EnvBaker::new(&rs.device)
        .upload(&rs, &crate::environment::decode(&crate::environment::EnvSource::Builtin(0)).unwrap());
    let size = [960, 640];
    for (cname, cam) in &cams {
        for (mname, mode) in &modes {
            let settings = ViewSettings { mode: *mode, ..Default::default() };
            views.render_3d(&rs, size, &[&engine], &env, &gpu_mesh, cam, &settings, None);
            let img = views.view3d.image();
            let bytes = read_texture(&rs.device, &rs.queue, img);
            let file = dir.join(format!("{cname}_{mname}.png"));
            image::save_buffer(&file, &bytes, img.width, img.height, image::ExtendedColorType::Rgba8).unwrap();
        }
    }
    println!("  previews in {}", dir.display());
}


/// Every kind of edit, undone back to the start and redone to the end, must
/// reproduce the exact layer stack at each step.
#[test]
#[ignore]
fn undo_redo_roundtrip() {
    use crate::engine::{Blend, Content, FillParams, GenKind};
    use crate::undo::UndoStack;

    let rs = render_state();
    let mesh = crate::mesh::Primitive::RoundedCube.build();
    let gpu_mesh = GpuMesh::new(&rs.device, &mesh);
    let mut engine = Engine::new(&rs, &gpu_mesh, (mesh.bounds_min, mesh.bounds_max), 256);
    let mut undo = UndoStack::default();

    // (ids, names, opacity, blend, has mask, has generator, is fill)
    let snapshot = |e: &Engine| -> Vec<(u64, String, u32, Blend, bool, bool, bool)> {
        e.layers
            .iter()
            .map(|l| {
                let m = l.mask.as_ref();
                (l.id, l.name.clone(), l.opacity.to_bits(), l.blend, m.is_some(), m.is_some_and(|m| m.generator.is_some()), matches!(l.content, Content::Fill(_)))
            })
            .collect()
    };
    let mut states = vec![snapshot(&engine)];
    let mut step = |e: &mut Engine, u: &mut UndoStack, f: &dyn Fn(&mut Engine, &mut UndoStack)| {
        f(e, u);
        e.commit_props(u);
        states.push(snapshot(e));
    };
    step(&mut engine, &mut undo, &|e, u| {
        e.add_paint_layer(0, None, None, u);
    });
    step(&mut engine, &mut undo, &|e, u| {
        e.add_fill_layer(1, None, "Fill".into(), FillParams::default(), u);
    });
    let smart = &presets::SMART_MATERIALS[0];
    step(&mut engine, &mut undo, &|e, u| {
        crate::app::add_smart_material(e, u, smart, 2, None);
    });
    step(&mut engine, &mut undo, &|e, _| {
        e.layers[0].opacity = 0.25;
        e.layers[0].blend = Blend::Multiply;
        e.layers[1].name = "Renamed".into();
    });
    step(&mut engine, &mut undo, &|e, u| {
        e.move_layer(0, 3, u);
    });
    step(&mut engine, &mut undo, &|e, u| e.add_mask(0, true, None, u));
    step(&mut engine, &mut undo, &|e, u| e.add_mask(1, true, Some(GenKind::ALL[0].default_generator()), u));
    step(&mut engine, &mut undo, &|e, u| e.remove_mask(0, u));
    step(&mut engine, &mut undo, &|e, u| {
        e.duplicate_layer(2, u);
    });
    step(&mut engine, &mut undo, &|e, u| e.clear_layer(4, u));
    step(&mut engine, &mut undo, &|e, u| e.delete_layer(1, u));
    step(&mut engine, &mut undo, &|e, u| e.delete_layer(0, u));

    let n = states.len() - 1;
    println!("  labels: {:?}", undo.undo.iter().map(|e| e.label.as_str()).collect::<Vec<_>>());
    assert_eq!(undo.undo.len(), n, "one history entry per step: {:?}", undo.undo.iter().map(|e| &e.label).collect::<Vec<_>>());
    for i in (0..n).rev() {
        engine.undo(&mut undo);
        assert_eq!(snapshot(&engine), states[i], "undo back to step {i}");
    }
    for (i, state) in states.iter().enumerate().skip(1) {
        engine.redo(&mut undo);
        assert_eq!(&snapshot(&engine), state, "redo to step {i}");
    }
    println!("  {} steps undone and redone: {:?}", n, undo.undo.iter().map(|e| e.label.as_str()).collect::<Vec<_>>());
}

/// Every built-in export template, in every format, from a baked and
/// painted rounded cube: files exist, decode, have the right size and
/// channel count, and padding fills the space between UV islands.
#[test]
#[ignore]
fn export_templates() {
    use crate::export::{self, Depth, Format, Padding, Settings};
    let mesh = Arc::new(crate::mesh::Primitive::RoundedCube.build());
    let rs = render_state();
    let gpu_mesh = GpuMesh::new(&rs.device, &mesh);
    let res = 512;
    let mut engine = Engine::new(&rs, &gpu_mesh, (mesh.bounds_min, mesh.bounds_max), res);
    let (maps, report, _) = run_bake(&rs.device, &rs.queue, &mesh, &gpu_mesh, &BakeSettings::default(), res);
    engine.set_maps(maps, report);
    let smart = presets::SMART_MATERIALS.iter().find(|s| s.name == "Painted Steel Worn").unwrap();
    crate::app::add_smart_material(&mut engine, &mut crate::undo::UndoStack::default(), smart, 0, None);
    engine.composite();
    let snap = export::snapshot(&engine, &rs.device, &rs.queue, mesh.clone());

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test/export");
    let _ = std::fs::remove_dir_all(&root);
    for (fi, format) in Format::ALL.into_iter().enumerate() {
        for (ti, t) in export::builtin_templates().iter().enumerate() {
            // Every template in PNG; the other formats with two templates.
            if format != Format::Png && ti > 1 {
                continue;
            }
            let settings = Settings {
                folder: root.join(format.ext()).join(format!("{ti}")),
                mesh_name: "Cube".into(),
                project_name: "Test".into(),
                set_name: "Cube".into(),
                set_count: 1,
                format,
                depth: *format.depths().last().unwrap(),
                size: (fi % 2 == 1).then_some(256),
                padding: if ti == 1 { Padding::Transparent } else { Padding::Infinite },
                normal_strength: 1.0,
                use_baked_normal: true,
            };
            let files = export::export(&snap, t, &settings).unwrap_or_else(|e| panic!("{} {}: {e}", t.name, format.name()));
            // This cube emits nothing and keeps the default specular: its
            // emission and specular level maps are left out.
            let optional = |c: &export::Chan| matches!(c.src, export::Source::Emission | export::Source::SpecularLevel);
            let maps: Vec<_> = t.maps.iter().filter(|m| !m.chans.iter().any(optional)).collect();
            assert_eq!(files.len(), maps.len());
            for (f, m) in files.iter().zip(maps) {
                let img = image::open(f).unwrap_or_else(|e| panic!("{}: {e}", f.display()));
                let size = settings.size.unwrap_or(res);
                assert_eq!((img.width(), img.height()), (size, size), "{}", f.display());
                // Opacity in alpha is left out on this opaque cube.
                let opacity_alpha = m.layout == export::Layout::Rgba && m.chans[3].src == export::Source::Opacity;
                let want = match (m.layout, format) {
                    (export::Layout::Rgba, _) if opacity_alpha => 3,
                    (export::Layout::Rgba, Format::Jpeg) => 3,
                    (export::Layout::Gray, Format::Exr) => 3,
                    (l, _) => l.channels(),
                };
                assert_eq!(img.color().channel_count() as usize, want, "{}", f.display());
            }
        }
    }
    // Infinite padding leaves no untouched black between islands of the
    // base color (the cube's islands are surrounded by empty UV space).
    let base = image::open(root.join("png/0/T_Cube_BaseColor.png")).unwrap().to_rgb8();
    let black = base.pixels().filter(|p| p.0 == [0, 0, 0]).count();
    assert!(black < (res * res / 200) as usize, "{black} black texels after padding");
    let _ = Depth::U8;
}

/// A stroke across the rounded cube with each brush projection (and once
/// with grain), then every fill projection: each must put paint on the mesh
/// and comparable amounts of it. Frames go to `target/bake_test/paint/`.
#[test]
#[ignore]
fn paint_projections() {
    use crate::app::{BrushProjection, resolve_stamps};
    use crate::engine::{BrushPaint, FillParams, MatValues, Projection, StampUniform};
    use crate::undo::{StrokeTarget, UndoStack};
    use glam::Vec2;

    let mesh = crate::mesh::Primitive::RoundedCube.build();
    let bvh = crate::bvh::Bvh::from_mesh(&mesh);
    let rs = render_state();
    let gpu_mesh = GpuMesh::new(&rs.device, &mesh);
    let res = 512;
    let mut engine = Engine::new(&rs, &gpu_mesh, (mesh.bounds_min, mesh.bounds_max), res);
    let mut undo = UndoStack::default();
    engine.add_paint_layer(0, None, None, &mut undo);

    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test/paint");
    std::fs::create_dir_all(&dir).unwrap();
    let mut views = ViewRenderer::new(&rs);
    let env = crate::environment::EnvBaker::new(&rs.device)
        .upload(&rs, &crate::environment::decode(&crate::environment::EnvSource::Builtin(0)).unwrap());
    let cam = OrbitCam { target: (mesh.bounds_min + mesh.bounds_max) * 0.5, ..Default::default() };
    let size = [640u32, 480u32];
    let settings = ViewSettings::default();
    let save = |views: &ViewRenderer, name: &str| {
        let img = views.view3d.image();
        let bytes = read_texture(&rs.device, &rs.queue, img);
        image::save_buffer(dir.join(format!("{name}.png")), &bytes, img.width, img.height, image::ExtendedColorType::Rgba8).unwrap();
    };
    let red_texels = |engine: &Engine| {
        let bytes = read_texture(&rs.device, &rs.queue, engine.final_tex(0));
        bytes.chunks(4).filter(|p| p[0] > 150 && p[1] < 90).count()
    };
    let extent = (mesh.bounds_max - mesh.bounds_min).max_element();

    let mut counts = Vec::new();
    for (name, projection, grain) in [
        ("screen", BrushProjection::Screen, false),
        ("surface", BrushProjection::Surface, false),
        ("uv", BrushProjection::Uv, false),
        ("surface_grain", BrushProjection::Surface, true),
    ] {
        engine.clear_layer(0, &mut undo);
        if grain {
            let px = crate::textures::TEXTURES[0].render(256);
            engine.set_brush_grain(Some((&px, 256)));
        }
        views.render_3d(&rs, size, &[&engine], &env, &gpu_mesh, &cam, &settings, None);
        assert!(engine.begin_stroke(0, StrokeTarget::Content, [true, false, false, false, false, false, false, false], false));
        let radius = 28.0;
        let raw: Vec<([f32; 4], f32)> = (0..60).map(|i| ([180.0 + i as f32 * 5.0, 250.0, 1.0, 0.0], 1.0)).collect();
        let fsize = Vec2::new(size[0] as f32, size[1] as f32);
        let stamps = resolve_stamps(&mesh, &bvh, &cam, fsize, projection, 1, radius, &raw);
        assert!(!stamps.is_empty(), "{name}: no stamps hit the mesh");
        let su = StampUniform {
            view_proj: cam.view_proj(fsize.x / fsize.y).to_cols_array_2d(),
            cam_pos: cam.eye().extend(1.0).to_array(),
            view: [fsize.x, fsize.y, 0.0, 0.0],
            view2d: [0.0; 4],
            brush: [radius, 0.6, 1.0, 1.0],
            clip: [crate::viewport::NEAR, crate::viewport::FAR, 0.0, 0.0],
            proj: [projection as u32 as f32, 0.0, 85f32.to_radians().cos(), 0.0],
        };
        let paint = BrushPaint {
            mat: MatValues { enabled: [true, false, false, false, false, false, false, false], color: [0.9, 0.1, 0.05], ..Default::default() },
            mask_value: 1.0,
            opacity: 1.0,
            erase: false,
            grain: if grain { [1.0, 3.0 / extent, 0.0, 0.0] } else { [0.0; 4] },
        };
        engine.stroke_step(&stamps, su, views.view3d.depth_view(), &paint, None);
        engine.end_stroke();
        engine.composite();
        views.render_3d(&rs, size, &[&engine], &env, &gpu_mesh, &cam, &settings, None);
        save(&views, name);
        let n = red_texels(&engine);
        println!("  {name}: {} stamps, {n} painted texels", stamps.len());
        counts.push(n);
    }
    let (screen, surface, uv, grained) = (counts[0], counts[1], counts[2], counts[3]);
    assert!(screen > 500, "screen projection painted {screen} texels");
    // Same brush circle on screen: the surface and UV discs cover a
    // similar area (the surface one more, as it wraps the bevels).
    for (name, n) in [("surface", surface), ("uv", uv)] {
        assert!(n * 3 > screen && n < screen * 4, "{name} painted {n} texels vs screen {screen}");
    }
    assert!(grained < surface, "grain should hold back some paint ({grained} vs {surface})");

    // Fill projections with an image.
    engine.delete_layer(0, &mut undo);
    let png = {
        use image::ImageEncoder;
        let mut png = Vec::new();
        image::codecs::png::PngEncoder::new(&mut png)
            .write_image(&crate::textures::TEXTURES[0].render(256), 256, 256, image::ExtendedColorType::L8)
            .unwrap();
        png
    };
    let img = engine.decode_fill_image("grain.png".into(), Arc::new(png)).unwrap();
    for p in Projection::ALL {
        let fill = FillParams { projection: p, ..FillParams::default() };
        let idx = engine.add_fill_layer(0, None, p.name().into(), fill, &mut undo);
        engine.set_fill_image(idx, 0, Some(img.clone()), &mut undo);
        engine.composite();
        views.render_3d(&rs, size, &[&engine], &env, &gpu_mesh, &cam, &settings, None);
        save(&views, &format!("fill_{}", p.key()));
        let bytes = read_texture(&rs.device, &rs.queue, engine.final_tex(0));
        let lit = bytes.chunks(4).filter(|px| px[0] > 20).count();
        assert!(lit > 1000, "{}: image didn't show ({lit} texels)", p.name());
        engine.delete_layer(idx, &mut undo);
    }
    println!("  frames in {}", dir.display());
}

/// GPU time of one painting frame at 4K: stamping + applying a stroke step,
/// then recompositing a smart material stack with a paint layer on top.
/// `cargo test --bin material_painter perf_4k -- --ignored --nocapture`
#[test]
#[ignore]
fn perf_4k() {
    use crate::app::{BrushProjection, resolve_stamps};
    use crate::engine::{BrushPaint, MatValues, StampUniform};
    use crate::undo::{StrokeTarget, UndoStack};
    use glam::Vec2;
    use std::time::Instant;

    let mesh = crate::mesh::Primitive::RoundedCube.build();
    let bvh = crate::bvh::Bvh::from_mesh(&mesh);
    let rs = render_state();
    let gpu_mesh = GpuMesh::new(&rs.device, &mesh);
    let res = 4096;
    let mut engine = Engine::new(&rs, &gpu_mesh, (mesh.bounds_min, mesh.bounds_max), res);
    let mut undo = UndoStack::default();
    let smart = presets::SMART_MATERIALS.iter().find(|s| s.name == "Painted Steel Worn").unwrap();
    crate::app::add_smart_material(&mut engine, &mut undo, smart, 0, None);
    let top = engine.layers.len();
    engine.add_paint_layer(top, None, None, &mut undo);
    // A second paint layer under the smart material: painting it keeps the
    // stack below in a cache, as more passes follow it.
    engine.add_paint_layer(0, None, None, &mut undo);
    let top = top + 1;
    println!("  {} layers", engine.layers.len());
    let wait = |rs: &egui_wgpu::RenderState| {
        rs.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    };
    engine.composite();
    wait(&rs);

    let mut views = ViewRenderer::new(&rs);
    let env = crate::environment::EnvBaker::new(&rs.device)
        .upload(&rs, &crate::environment::decode(&crate::environment::EnvSource::Builtin(0)).unwrap());
    let cam = OrbitCam { target: (mesh.bounds_min + mesh.bounds_max) * 0.5, ..Default::default() };
    let size = [1600u32, 1000u32];
    let settings = ViewSettings::default();
    views.render_3d(&rs, size, &[&engine], &env, &gpu_mesh, &cam, &settings, None);
    wait(&rs);

    let fsize = Vec2::new(size[0] as f32, size[1] as f32);
    let su = StampUniform {
        view_proj: cam.view_proj(fsize.x / fsize.y).to_cols_array_2d(),
        cam_pos: cam.eye().extend(1.0).to_array(),
        view: [fsize.x, fsize.y, 0.0, 0.0],
        view2d: [0.0; 4],
        brush: [35.0, 0.6, 1.0, 0.0],
        clip: [crate::viewport::NEAR, crate::viewport::FAR, 0.0, 0.0],
        proj: [0.0, 0.0, 85f32.to_radians().cos(), 0.0],
    };
    let paint = BrushPaint {
        mat: MatValues { enabled: [true, true, true, false, false, false, false, false], color: [0.9, 0.1, 0.05], ..Default::default() },
        mask_value: 1.0,
        opacity: 1.0,
        erase: false,
        grain: [0.0; 4],
    };
    for (name, layer) in [("top layer", top), ("bottom layer", 0)] {
    assert!(engine.begin_stroke(layer, StrokeTarget::Content, [true, true, true, false, false, false, false, false], false));
    let frames = 30;
    let (mut t_stroke, mut t_comp, mut t_view) = (0.0, 0.0, 0.0);
    for f in 0..frames {
        let raw: Vec<([f32; 4], f32)> = (0..6).map(|i| ([500.0 + (f * 6 + i) as f32 * 3.0, 500.0, 1.0, 0.0], 1.0)).collect();
        let stamps = resolve_stamps(&mesh, &bvh, &cam, fsize, BrushProjection::Screen, 0, 35.0, &raw);
        let t = Instant::now();
        let region = crate::app::stroke_region(&mesh, &bvh, &cam, fsize, BrushProjection::Screen, 0, 35.0, &stamps);
        engine.stroke_step(&stamps, su, views.view3d.depth_view(), &paint, region);
        wait(&rs);
        t_stroke += t.elapsed().as_secs_f64();
        let t = Instant::now();
        engine.composite();
        wait(&rs);
        t_comp += t.elapsed().as_secs_f64();
        let t = Instant::now();
        views.render_3d(&rs, size, &[&engine], &env, &gpu_mesh, &cam, &settings, None);
        wait(&rs);
        t_view += t.elapsed().as_secs_f64();
    }
    // Stroke frames recomposite from the painted layer up: the result must
    // match a full composite exactly.
    let partial: Vec<Vec<u8>> = (0..4).map(|c| read_texture(&rs.device, &rs.queue, engine.final_tex(c))).collect();
    engine.mark_dirty();
    engine.composite();
    for c in 0..4 {
        assert!(read_texture(&rs.device, &rs.queue, engine.final_tex(c)) == partial[c], "{name}: channel {c} differs from a full composite");
    }
    engine.end_stroke();
    let ms = |t: f64| t * 1000.0 / frames as f64;
    println!(
        "  {name}, per frame at {res}: stroke {:.2} ms, composite {:.2} ms, 3D view {:.2} ms",
        ms(t_stroke),
        ms(t_comp),
        ms(t_view)
    );
    }
}

/// Every built-in graph node compiles and renders, fed by noise where it has
/// inputs; the starter graph produces its outputs. Thumbnails go to
/// `target/bake_test/graph/`.
#[test]
#[ignore]
fn graph_nodes_render() {
    use crate::graph::{Graph, Library, Runtime, Value};

    let rs = render_state();
    let mesh = crate::mesh::Primitive::RoundedCube.build();
    let gpu_mesh = GpuMesh::new(&rs.device, &mesh);
    let engine = Engine::new(&rs, &gpu_mesh, (mesh.bounds_min, mesh.bounds_max), 256);
    let lib = Library::load(None);
    let mut rt = Runtime::new(&rs.device, &rs.queue);
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test/graph");
    std::fs::create_dir_all(&dir).unwrap();
    let picture = dir.join("input.png");
    image::save_buffer(&picture, &crate::textures::TEXTURES[0].render(64), 64, 64, image::ExtendedColorType::L8).unwrap();
    let maps = |m: usize| engine.mesh_map(m);

    let half = |b: &[u8]| {
        let h = u16::from_le_bytes([b[0], b[1]]);
        let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 31) as i32, (h & 1023) as f32);
        let v = if e == 0 { m / 1024.0 * 2f32.powi(-14) } else { (1.0 + m / 1024.0) * 2f32.powi(e - 15) };
        if s == 1 { -v } else { v }
    };
    let texture_defs: Vec<_> = lib.defs.iter().filter(|d| !d.shading).collect();
    let mut sheet = image::RgbaImage::new(128 * 6, 128 * texture_defs.len().div_ceil(6) as u32);
    for (k, def) in texture_defs.into_iter().enumerate() {
        let mut g = Graph::new(&def.name);
        g.res = 128;
        let noise = g.add("noise", [0.0; 2]);
        let n = g.add(&def.id, [0.0; 2]);
        for inp in &def.inputs {
            g.connect(noise, n, &inp.name);
        }
        if def.params.iter().any(|p| p.kind == crate::graph::ParamKind::Image) {
            g.node_mut(n).unwrap().params.insert("image".into(), Value::Image(Some(picture.clone())));
        }
        let surface = engine.graph_surface();
        rt.evaluate(&rs.device, &rs.queue, &g, &lib, &maps, Some(&surface), 0, false);
        // Images decode on a worker: render again once they're in.
        rt.wait_images(&rs.device, &rs.queue);
        rt.evaluate(&rs.device, &rs.queue, &g, &lib, &maps, Some(&surface), 0, false);
        assert!(rt.errors.is_empty(), "{}: {:?}", def.id, rt.errors);
        let tex = rt.output(n).unwrap_or_else(|| panic!("{} rendered nothing", def.id));
        let bytes = read_texture(&rs.device, &rs.queue, tex);
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        for (i, px) in bytes.chunks(8).enumerate() {
            let c = [0, 1, 2, 3].map(|j| half(&px[j * 2..]));
            assert!(c.iter().all(|v| v.is_finite()), "{}: non-finite output", def.id);
            lo = lo.min(c[0]);
            hi = hi.max(c[0]);
            let (x, y) = ((i % 128) as u32, (i / 128) as u32);
            let u8c = c.map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8);
            sheet.put_pixel((k % 6) as u32 * 128 + x, (k / 6) as u32 * 128 + y, image::Rgba([u8c[0], u8c[1], u8c[2], 255]));
        }
        println!("  {:<18} red range {lo:.3}..{hi:.3}", def.id);
    }
    sheet.save(dir.join("nodes.png")).unwrap();

    // The starter graph feeds base color and roughness.
    let g = Graph::starter("Starter");
    rt.evaluate(&rs.device, &rs.queue, &g, &lib, &maps, None, 0, false);
    assert!(rt.errors.is_empty(), "{:?}", rt.errors);
    let outs = g.outputs(&lib);
    assert!(rt.output(outs[0].unwrap()).is_some() && rt.output(outs[1].unwrap()).is_some());
    // Nothing changed: nothing re-renders.
    assert!(!rt.evaluate(&rs.device, &rs.queue, &g, &lib, &maps, None, 0, false));

    // Through a fill layer: the composite shows the graph's base color, and
    // a graph's output written into a mask varies across it.
    let mut engine = engine;
    let mut undo = crate::undo::UndoStack::default();
    let fill = crate::engine::FillParams { graph: Some(1), ..Default::default() };
    let idx = engine.add_fill_layer(0, None, "Graph".into(), fill, &mut undo);
    let out = |u: usize| outs[u].and_then(|n| rt.output(n).cloned());
    engine.set_graph_images(idx, [out(0), out(1), out(2), out(3), None, None, None, None], "Starter");
    engine.composite();
    let color = read_texture(&rs.device, &rs.queue, engine.final_tex(0));
    let distinct: std::collections::HashSet<[u8; 3]> = color.chunks(4).map(|p| [p[0], p[1], p[2]]).collect();
    assert!(distinct.len() > 50, "base color shows {} distinct colors", distinct.len());
    engine.add_mask(idx, true, None, &mut undo);
    engine.set_graph_mask(idx, rt.output(outs[1].unwrap()).unwrap());
    let mask = read_texture(&rs.device, &rs.queue, engine.layers[idx].mask.as_ref().unwrap().paint.as_ref().unwrap());
    let (lo, hi) = mask.iter().fold((255u8, 0u8), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    assert!(hi > lo + 40, "mask range {lo}..{hi}");
    // Custom nodes: a valid one compiles; a broken one reports the
    // compiler's message instead of failing the device.
    let user = dir.join("user_nodes");
    let _ = std::fs::remove_dir_all(&user);
    let mut ulib = Library::load(Some(user.clone()));
    let id = ulib.save_user("tint", &crate::graph::template("Tint")).unwrap();
    assert!(rt.compile(&rs.device, ulib.find(&id).unwrap()).is_ok());
    let broken = crate::graph::template("Broken").replace("c.rgb * params.amount", "c.rgb * undefined_thing");
    let id = ulib.save_user("broken", &broken).unwrap();
    let err = rt.compile(&rs.device, ulib.find(&id).unwrap()).unwrap_err();
    assert!(err.contains("undefined_thing"), "{err}");
    // The location reads as a line of the user's file.
    let def = ulib.find(&id).unwrap();
    let want = broken.lines().position(|l| l.contains("undefined_thing")).unwrap() + 1;
    let mapped = def.map_error(&err);
    assert!(mapped.contains(&format!("line {want}:")), "expected line {want} in: {mapped}");
    println!("  broken node reports: {}", err.lines().find(|l| l.contains("undefined_thing")).unwrap_or(""));
    let img = engine.final_tex(0);
    image::save_buffer(dir.join("composited.png"), &color, img.width, img.height, image::ExtendedColorType::Rgba8).unwrap();
}

/// The ready-made material graphs render cleanly; their base colors go to
/// `target/bake_test/graph/presets.png`.
#[test]
#[ignore]
fn graph_presets_render() {
    use crate::graph::{Library, Runtime, presets::PRESETS};
    let rs = render_state();
    let mesh = crate::mesh::Primitive::RoundedCube.build();
    let gpu_mesh = GpuMesh::new(&rs.device, &mesh);
    let engine = Engine::new(&rs, &gpu_mesh, (mesh.bounds_min, mesh.bounds_max), 256);
    let lib = Library::load(None);
    let maps = |m: usize| engine.mesh_map(m);
    let mut sheet = image::RgbaImage::new(256 * PRESETS.len() as u32, 256);
    for (k, p) in PRESETS.iter().enumerate() {
        let mut g = (p.build)();
        g.res = 256;
        let mut rt = Runtime::new(&rs.device, &rs.queue);
        rt.evaluate(&rs.device, &rs.queue, &g, &lib, &maps, None, 0, false);
        assert!(rt.errors.is_empty(), "{}: {:?}", p.name, rt.errors);
        let base = g.outputs(&lib)[0].unwrap();
        let thumb = rt.thumb(base).unwrap();
        let bytes = read_texture(&rs.device, &rs.queue, thumb);
        let img = image::RgbaImage::from_raw(thumb.width, thumb.height, bytes).unwrap();
        image::imageops::overlay(&mut sheet, &img, k as i64 * 256, 0);
    }
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test/graph");
    std::fs::create_dir_all(&dir).unwrap();
    sheet.save(dir.join("presets.png")).unwrap();
}

/// Screenshots of the real UI, rendered offscreen by egui_kittest (no
/// window): every workspace, written to `target/bake_test/ui/`.
#[test]
#[ignore]
fn ui_screenshots() {
    use crate::app::{PainterApp, Workspace};
    // A hung frame would otherwise leave the test (and its GPU memory) running.
    std::thread::spawn(|| {
        std::thread::sleep(std::time::Duration::from_secs(600));
        eprintln!("ui_screenshots: still running after 10 minutes, giving up");
        std::process::exit(101);
    });
    use egui_kittest::kittest::Queryable;
    // The same device request as the app: the adapter's full limits.
    let mut setup = egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    setup.device_descriptor = Arc::new(|adapter| wgpu::DeviceDescriptor {
        label: Some("ui test device"),
        required_limits: adapter.limits(),
        ..Default::default()
    });
    let mut harness = egui_kittest::Harness::builder()
        .with_size([1600.0, 1300.0])
        .with_pixels_per_point(1.0)
        // A screenshot test, not an accessibility audit.
        .with_accessibility_check(false)
        .wgpu_setup(egui_wgpu::WgpuSetup::CreateNew(setup))
        .build_eframe(|cc| PainterApp::new(cc).expect("app starts"));
    harness.run_steps(8);
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test/ui");
    std::fs::create_dir_all(&dir).unwrap();
    // The welcome window opens at startup.
    harness.render().expect("renders").save(dir.join("welcome.png")).unwrap();
    // Its entries light up under the pointer.
    harness.get_by_label_contains("Open project").hover();
    harness.run_steps(2);
    harness.render().expect("renders").save(dir.join("welcome_hover.png")).unwrap();
    harness.state_mut().test_close_welcome();
    harness.run_steps(2);
    harness.state_mut().test_show_seams();
    harness.run_steps(3);
    harness.render().expect("renders").save(dir.join("seams.png")).unwrap();
    std::fs::create_dir_all(&dir).unwrap();
    for (name, ws) in [("paint", Workspace::Paint), ("designer", Workspace::Designer), ("shader", Workspace::ShaderMaker)] {
        harness.state_mut().set_workspace(ws);
        harness.run_steps(6);
        harness.render().expect("renders").save(dir.join(format!("{name}.png"))).unwrap();
    }
    // The shelf's plain materials, grouped by type.
    harness.state_mut().set_workspace(Workspace::Paint);
    harness.state_mut().test_shelf_materials();
    harness.run_steps(6);
    harness.render().expect("renders").save(dir.join("shelf_types.png")).unwrap();
    // A graph open on the canvas, and the paint workspace showing its layer.
    harness.state_mut().set_workspace(Workspace::Designer);
    harness.state_mut().test_add_preset(1);
    harness.run_steps(10);
    harness.render().expect("renders").save(dir.join("designer_graph.png")).unwrap();
    // The export window keeps its size over many frames (it used to grow
    // to the full width of the screen).
    harness.state_mut().set_workspace(Workspace::Paint);
    harness.state_mut().test_open_export();
    harness.run_steps(30);
    harness.render().expect("renders").save(dir.join("export.png")).unwrap();
    // HDRI pictures in the Preview tab.
    let ctx = harness.ctx.clone();
    harness.state_mut().test_show_hdris(&ctx);
    harness.run_steps(4);
    harness.render().expect("renders").save(dir.join("hdri.png")).unwrap();
    // The asset library (network: waits for the catalog, then thumbnails).
    let start = std::time::Instant::now();
    while !harness.state_mut().test_open_library() && start.elapsed().as_secs() < 60 {
        harness.run_steps(1);
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    let start = std::time::Instant::now();
    while start.elapsed().as_secs() < 8 {
        harness.run_steps(1);
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    harness.render().expect("renders").save(dir.join("library.png")).unwrap();
    // A downloaded material dropped on the mesh.
    harness.state_mut().test_add_downloaded();
    harness.run_steps(10);
    harness.render().expect("renders").save(dir.join("library_layer.png")).unwrap();
    // A smart material saved from the stack and applied again keeps its structure.
    let ctx = harness.ctx.clone();
    let [before, after] = harness.state_mut().test_smart_roundtrip(&ctx);
    println!("  smart material (layers, masks, generators, filters, images, folders): saved {before:?}, applied {after:?}");
    assert_eq!(before, after, "the smart material round-trips");
    harness.run_steps(10);
    harness.render().expect("renders").save(dir.join("smart_roundtrip.png")).unwrap();
    // The path tool: a stitch path in each view, previewed, then applied.
    for (name, in_3d) in [("path_3d", true), ("path_2d", false)] {
        harness.state_mut().test_path_setup(in_3d);
        harness.run_steps(4);
        harness.render().expect("renders").save(dir.join(format!("{name}_preview.png"))).unwrap();
        let status = harness.state_mut().test_path_apply();
        println!("  {name}: {status}");
        assert!(status.starts_with("Painted"), "{status}");
        harness.run_steps(6);
        harness.render().expect("renders").save(dir.join(format!("{name}.png"))).unwrap();
    }
    // A preview shader: Toon on the 3D view, and its graph in the Material Designer.
    harness.state_mut().set_workspace(Workspace::Paint);
    let toon = harness.state_mut().test_preview_preset(0);
    harness.run_steps(6);
    assert_eq!(harness.state_mut().test_preview_error(), None, "the Toon preview shader compiles in the app");
    harness.render().expect("renders").save(dir.join("preview_shader.png")).unwrap();
    harness.state_mut().test_open_graph(toon);
    harness.run_steps(8);
    harness.render().expect("renders").save(dir.join("preview_shader_graph.png")).unwrap();
    // A bake that leaves out AO and edge distance: shelf badges and mask notes.
    harness.state_mut().set_workspace(Workspace::Paint);
    let missing = (1 << 0) | (1 << 5);
    let start = std::time::Instant::now();
    while !harness.state_mut().test_bake_without(&ctx, missing) && start.elapsed().as_secs() < 120 {
        harness.run_steps(1);
    }
    harness.state_mut().test_show_missing();
    harness.run_steps(8);
    harness.render().expect("renders").save(dir.join("missing_maps.png")).unwrap();
    harness.state_mut().test_show_custom();
    harness.run_steps(8);
    harness.render().expect("renders").save(dir.join("bake_custom.png")).unwrap();
    // A mask reading a custom map: the map picker and the short inputs list.
    harness.state_mut().test_custom_mask();
    harness.run_steps(8);
    harness.render().expect("renders").save(dir.join("mask_custom.png")).unwrap();
}

/// Mask filters change what a layer covers: a white mask shows the layer
/// everywhere, Invert hides it, noise breakup eats into it, blur on a flat
/// mask changes nothing.
#[test]
#[ignore]
fn mask_filters_apply() {
    use crate::engine::{FillParams, FilterKind, MaskFilter, MatValues};
    let rs = render_state();
    let mesh = crate::mesh::Primitive::RoundedCube.build();
    let gpu_mesh = GpuMesh::new(&rs.device, &mesh);
    let mut engine = Engine::new(&rs, &gpu_mesh, (mesh.bounds_min, mesh.bounds_max), 256);
    let mut undo = crate::undo::UndoStack::default();
    let fill = FillParams { mat: MatValues { enabled: [true, false, false, false, false, false, false, false], color: [1.0, 0.0, 0.0], ..Default::default() }, ..Default::default() };
    let idx = engine.add_fill_layer(0, None, "Red".into(), fill, &mut undo);
    engine.add_mask(idx, true, None, &mut undo);
    let red = |e: &mut Engine| {
        e.composite();
        let px = read_texture(&rs.device, &rs.queue, e.final_tex(0));
        let red = px.chunks(4).filter(|p| p[0] > 200 && p[1] < 60).count();
        red as f32 / (px.len() / 4) as f32
    };
    let full = red(&mut engine);
    assert!(full > 0.95, "white mask: {full}");
    let set = |e: &mut Engine, filters: Vec<MaskFilter>| {
        e.layers[idx].mask.as_mut().unwrap().filters = filters;
        e.mark_dirty();
    };
    set(&mut engine, vec![MaskFilter::new(FilterKind::Blur)]);
    let blurred = red(&mut engine);
    assert!(blurred > 0.95, "blurred white mask: {blurred}");
    set(&mut engine, vec![MaskFilter::new(FilterKind::Invert)]);
    let inverted = red(&mut engine);
    assert!(inverted < 0.01, "inverted: {inverted}");
    let mut noise = MaskFilter::new(FilterKind::Noise);
    noise.a = 1.0;
    set(&mut engine, vec![noise]);
    let broken = red(&mut engine);
    assert!(broken > 0.05 && broken < 0.9, "noise breakup: {broken}");
    // Switched off, the filter does nothing.
    let mut off = MaskFilter::new(FilterKind::Invert);
    off.enabled = false;
    set(&mut engine, vec![off]);
    assert!(red(&mut engine) > 0.95);
    println!("  red coverage: white {full:.2}, blur {blurred:.2}, invert {inverted:.2}, noise {broken:.2}");

    // Mask layers blend over the base mask.
    use crate::engine::{GenKind, MaskBlend, MaskLayer, MaskSource};
    let stack = |e: &mut Engine, layers: Vec<MaskLayer>| {
        let m = e.layers[idx].mask.as_mut().unwrap();
        m.filters.clear();
        m.layers = layers;
        e.mark_dirty();
    };
    let ml = |source, blend| MaskLayer { source, blend, opacity: 1.0, enabled: true };
    stack(&mut engine, vec![ml(MaskSource::Fill(0.0), MaskBlend::Multiply)]);
    assert!(red(&mut engine) < 0.01, "multiplied by black");
    stack(&mut engine, vec![ml(MaskSource::Fill(1.0), MaskBlend::Subtract)]);
    assert!(red(&mut engine) < 0.01, "white subtracted");
    stack(&mut engine, vec![ml(MaskSource::Fill(0.0), MaskBlend::Normal), ml(MaskSource::Generator(GenKind::Clouds.default_generator()), MaskBlend::Add)]);
    let clouds = red(&mut engine);
    assert!(clouds > 0.05 && clouds < 0.95, "clouds added over black: {clouds}");
    let mut off = ml(MaskSource::Fill(0.0), MaskBlend::Multiply);
    off.enabled = false;
    stack(&mut engine, vec![off]);
    assert!(red(&mut engine) > 0.95, "a switched-off mask layer does nothing");
    println!("  mask layers: clouds over black covers {clouds:.2}");
}

/// Every built-in shading node compiles into a preview shader on the GPU
/// (one small graph each), and every preset renders: images in
/// `target/bake_test/shading/`.
#[test]
#[ignore]
fn preview_shaders_compile_and_render() {
    use crate::graph::{Graph, Library, shading};
    let rs = render_state();
    let mesh = crate::mesh::Primitive::MaterialBall.build();
    let gpu_mesh = GpuMesh::new(&rs.device, &mesh);
    let res = 1024;
    let mut engine = Engine::new(&rs, &gpu_mesh, (mesh.bounds_min, mesh.bounds_max), res);
    let (maps, report, _) = run_bake(&rs.device, &rs.queue, &mesh, &gpu_mesh, &BakeSettings::default(), res);
    engine.set_maps(maps, report);
    let smart = presets::SMART_MATERIALS.iter().find(|s| s.name == "Painted Steel Worn").expect("smart material");
    crate::app::add_smart_material(&mut engine, &mut crate::undo::UndoStack::default(), smart, 0, None);
    engine.composite();
    let lib = Library::load(None);
    let mut views = ViewRenderer::new(&rs);

    // Each node alone, its output into the Shader Output.
    let mut failures = Vec::new();
    for def in lib.defs.iter().filter(|d| d.shading && d.id != shading::OUTPUT) {
        let mut g = Graph::new(&def.name);
        let n = g.add(&def.id, [0.0, 0.0]);
        let o = g.add(shading::OUTPUT, [200.0, 0.0]);
        g.connect(n, o, "color");
        let compiled = shading::compile(&g, &lib).unwrap_or_else(|e| panic!("{}: {e}", def.name));
        if let Err(e) = views.set_custom_shader(&rs.device, Some(&compiled.code)) {
            failures.push(format!("{}: {e}", def.name));
        }
    }
    assert!(failures.is_empty(), "shading nodes that don't compile:\n{}", failures.join("\n"));

    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test/shading");
    std::fs::create_dir_all(&dir).unwrap();
    let env = crate::environment::EnvBaker::new(&rs.device)
        .upload(&rs, &crate::environment::decode(&crate::environment::EnvSource::Builtin(0)).unwrap());
    let cam = OrbitCam { dist: 3.2, yaw: 0.5, pitch: 0.3, ..Default::default() };
    let settings = ViewSettings::default();
    let mut shots: Vec<(String, Option<Graph>)> = vec![("Default".into(), None)];
    shots.extend(shading::presets().into_iter().map(|g| (g.name.clone(), Some(g))));
    for (name, graph) in &shots {
        match graph {
            Some(g) => {
                let compiled = shading::compile(g, &lib).unwrap();
                views.set_custom_shader(&rs.device, Some(&compiled.code)).unwrap_or_else(|e| panic!("{name}: {e}"));
                views.set_custom_params(&rs.queue, &shading::param_values(&compiled, g, &lib));
            }
            None => views.set_custom_shader(&rs.device, None).unwrap(),
        }
        views.render_3d(&rs, [640, 640], &[&engine], &env, &gpu_mesh, &cam, &settings, None);
        let img = views.view3d.image();
        let bytes = read_texture(&rs.device, &rs.queue, img);
        let file = dir.join(format!("{}.png", name.replace(' ', "_").replace('+', "and")));
        image::save_buffer(&file, &bytes, img.width, img.height, image::ExtendedColorType::Rgba8).unwrap();
        println!("  {name} -> {}", file.display());
    }
}

/// Reads a project's mesh (`MP_PROJECT=path.mpp`), reports what's wrong with
/// it (degenerate and flipped UV triangles, overlapping UVs, ...), bakes it
/// with the project's bake settings and writes every map to
/// `target/bake_test/project/`. For diagnosing bake artifacts.
#[test]
#[ignore]
fn project_bake() {
    use glam::Vec2;
    let Ok(path) = std::env::var("MP_PROJECT") else {
        println!("set MP_PROJECT to a .mpp file");
        return;
    };
    let (mesh, manifest) = load_project_mesh(&path);

    // ---- mesh health
    let tris = mesh.indices.len() / 3;
    let (mut degenerate, mut uv_degenerate, mut outside) = (0, 0, 0);
    let mut uv_area = 0.0f64;
    let mut signs = [0usize; 2];
    let res = 2048usize;
    let mut cover = vec![0u16; res * res];
    for t in mesh.indices.chunks_exact(3) {
        let v = [t[0], t[1], t[2]].map(|i| &mesh.vertices[i as usize]);
        let p = v.map(|v| Vec3::from_array(v.pos));
        let uv = v.map(|v| Vec2::from_array(v.uv));
        if (p[1] - p[0]).cross(p[2] - p[0]).length() < 1e-10 {
            degenerate += 1;
        }
        let a = (uv[1] - uv[0]).perp_dot(uv[2] - uv[0]);
        uv_area += (a.abs() * 0.5) as f64;
        if a.abs() < 1e-12 {
            uv_degenerate += 1;
        } else {
            signs[(a < 0.0) as usize] += 1;
        }
        if uv.iter().any(|u| u.x < 0.0 || u.y < 0.0 || u.x > 1.0 || u.y > 1.0) {
            outside += 1;
        }
        // Rasterize to count texels covered more than once (overlapping UVs).
        let px = uv.map(|u| u * res as f32);
        let lo = px[0].min(px[1]).min(px[2]).max(Vec2::ZERO);
        let hi = px[0].max(px[1]).max(px[2]).min(Vec2::splat(res as f32 - 1.0));
        if lo.x > hi.x || lo.y > hi.y {
            continue;
        }
        for y in lo.y as usize..=hi.y as usize {
            for x in lo.x as usize..=hi.x as usize {
                let q = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                let e = |a: Vec2, b: Vec2| (b - a).perp_dot(q - a);
                let (w0, w1, w2) = (e(px[1], px[2]), e(px[2], px[0]), e(px[0], px[1]));
                if (w0 > 0.0 && w1 > 0.0 && w2 > 0.0) || (w0 < 0.0 && w1 < 0.0 && w2 < 0.0) {
                    cover[y * res + x] = cover[y * res + x].saturating_add(1);
                }
            }
        }
    }
    let flipped = signs[0].min(signs[1]);
    let covered = cover.iter().filter(|&&c| c > 0).count();
    let overlapped = cover.iter().filter(|&&c| c > 1).count();
    // Seams: vertices sharing a position but not a UV.
    let mut by_pos: std::collections::HashMap<[u32; 3], Vec<[u32; 2]>> = Default::default();
    for v in &mesh.vertices {
        by_pos.entry(v.pos.map(f32::to_bits)).or_default().push(v.uv.map(f32::to_bits));
    }
    let seam_verts = by_pos.values().filter(|uvs| uvs.iter().any(|u| *u != uvs[0])).count();
    println!("  mesh: {tris} tris, {} verts ({} unique positions, {seam_verts} on UV seams)", mesh.vertices.len(), by_pos.len());
    println!("  degenerate tris: {degenerate} (3D), {uv_degenerate} (UV); flipped UV tris: {flipped} of {}; tris with UVs outside 0-1: {outside}", signs[0] + signs[1]);
    println!(
        "  UV area used: {:.1}%; texels covered {:.1}%, of which overlapping {:.2}% ({overlapped} texels at {res}^2)",
        uv_area * 100.0,
        covered as f64 / (res * res) as f64 * 100.0,
        overlapped as f64 / covered.max(1) as f64 * 100.0
    );
    let ov: Vec<u8> = cover.iter().map(|&c| match c {
        0 => 0,
        1 => 90,
        _ => 255,
    }).collect();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test/project");
    std::fs::create_dir_all(&dir).unwrap();
    image::save_buffer(dir.join("uv_coverage.png"), &ov, res as u32, res as u32, image::ExtendedColorType::L8).unwrap();

    // ---- bake with the project's settings
    let bv = &manifest["bake"];
    let mut settings = BakeSettings::default();
    if let Some(q) = bv["quality"].as_str().and_then(|q| crate::bake::Quality::ALL.into_iter().find(|x| format!("{x:?}") == q)) {
        settings.apply_quality(q);
    }
    let fl = |k: &str, d: f32| bv[k].as_f64().map_or(d, |v| v as f32);
    settings.ao_distance = fl("ao_distance", settings.ao_distance);
    settings.cavity_distance = fl("cavity_distance", settings.cavity_distance);
    settings.thickness_distance = fl("thickness_distance", settings.thickness_distance);
    let rs = render_state();
    let gpu_mesh = GpuMesh::new(&rs.device, &mesh);
    let bake_res: u32 = std::env::var("MP_BAKE_RES").ok().and_then(|r| r.parse().ok()).unwrap_or(2048);
    let (maps, report, _) = run_bake(&rs.device, &rs.queue, &mesh, &gpu_mesh, &settings, bake_res);
    let packed: Vec<Vec<u8>> = maps.maps.iter().map(|t| read_texture(&rs.device, &rs.queue, t)).collect();
    let fmt: Vec<_> = maps.maps.iter().map(|t| t.format).collect();
    for mi in MAPS.iter() {
        let (bytes, f) = (&packed[mi.tex], fmt[mi.tex]);
        let px = |i: usize| -> [f32; 4] {
            match f {
                wgpu::TextureFormat::Rgba16Float => {
                    let b = &bytes[i * 8..i * 8 + 8];
                    [0, 1, 2, 3].map(|k| crate::gpu::f16_to_f32(u16::from_le_bytes([b[k * 2], b[k * 2 + 1]])))
                }
                _ => [0, 1, 2, 3].map(|k| bytes[i * 4 + k] as f32 / 255.0),
            }
        };
        let n = (bake_res * bake_res) as usize;
        let img: Vec<u8> = (0..n).map(|i| (px(i)[mi.comp].clamp(0.0, 1.0) * 255.0) as u8).collect();
        let name = mi.name.to_lowercase().replace(' ', "_");
        image::save_buffer(dir.join(format!("map_{name}.png")), &img, bake_res, bake_res, image::ExtendedColorType::L8).unwrap();
    }
    let mut engine = Engine::new(&rs, &gpu_mesh, (mesh.bounds_min, mesh.bounds_max), bake_res);
    engine.set_maps(maps, report);
    engine.composite();
    let mut views = ViewRenderer::new(&rs);
    let env = crate::environment::EnvBaker::new(&rs.device)
        .upload(&rs, &crate::environment::decode(&crate::environment::EnvSource::Builtin(0)).unwrap());
    let center = (mesh.bounds_min + mesh.bounds_max) * 0.5;
    for (cname, yaw) in [("front", 0.0f32), ("back", std::f32::consts::PI), ("side", 1.3)] {
        let cam = OrbitCam { target: center, dist: 3.0, yaw, pitch: 0.15, ..Default::default() };
        for mname in ["Ambient Occlusion", "Curvature", "Normal", "Cavity", "Edge Distance", "Thickness"] {
            let mode = ViewMode::all().into_iter().find(|v| v.name() == mname).unwrap_or(ViewMode::Material);
            views.render_3d(&rs, [900, 900], &[&engine], &env, &gpu_mesh, &cam, &ViewSettings { mode, ..Default::default() }, None);
            let img = views.view3d.image();
            let bytes = read_texture(&rs.device, &rs.queue, img);
            let file = dir.join(format!("view_{cname}_{}.png", mname.to_lowercase().replace(' ', "_")));
            image::save_buffer(&file, &bytes, img.width, img.height, image::ExtendedColorType::Rgba8).unwrap();
        }
    }
    println!("  wrote {}", dir.display());
}

/// Custom bake maps, as many as a project wants: eight fluid flows from
/// different directions next to one of each other kind (twelve maps, three
/// textures). Each bakes something plausible, flows from opposite
/// directions differ, a mask reads one of the later maps by id, and the views
/// show them by id. Images in `target/bake_test/custom/`.
/// `MP_PREVIEW_MESH=path` bakes that mesh instead of the material ball.
#[test]
#[ignore]
fn custom_maps_bake() {
    use crate::bake::CustomKind;
    use crate::engine::{CustomInput, FillParams, GenKind, MatValues};
    let mesh = match std::env::var("MP_PREVIEW_MESH") {
        Ok(path) => Mesh::load_file(Path::new(&path)).unwrap_or_else(|e| panic!("{path}: {e}")),
        Err(_) => crate::mesh::Primitive::MaterialBall.build(),
    };
    let rs = render_state();
    let gpu_mesh = GpuMesh::new(&rs.device, &mesh);
    let res = 512;
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test/custom");
    std::fs::create_dir_all(&dir).unwrap();
    let mut views = ViewRenderer::new(&rs);
    let env = crate::environment::EnvBaker::new(&rs.device)
        .upload(&rs, &crate::environment::decode(&crate::environment::EnvSource::Builtin(0)).unwrap());
    let center = (mesh.bounds_min + mesh.bounds_max) * 0.5;

    let mut settings = BakeSettings::default();
    settings.apply_quality(crate::bake::Quality::Draft);
    // Eight fluid flows: from above, then from seven directions around and below.
    for (k, (az, el)) in [(0.0, 90.0), (0.0, 30.0), (90.0, 30.0), (180.0, 30.0), (270.0, 30.0), (45.0, 0.0), (225.0, 0.0), (0.0, -90.0)].into_iter().enumerate() {
        let i = settings.add_custom(CustomKind::Flow);
        let c = &mut settings.custom[i];
        (c.azimuth, c.elevation) = (az, el);
        c.name = format!("Rain {}", k + 1);
    }
    for kind in [CustomKind::Exposure, CustomKind::Deposit, CustomKind::Contact, CustomKind::Point] {
        let i = settings.add_custom(kind);
        if kind == CustomKind::Point {
            settings.custom[i].point = [mesh.bounds_max.x, center.y, mesh.bounds_max.z];
        }
    }
    let ids: Vec<u32> = settings.custom.iter().map(|c| c.id).collect();
    assert_eq!(ids.len(), 12);
    assert_eq!(ids.iter().collect::<std::collections::HashSet<_>>().len(), 12, "ids are unique");

    let start = std::time::Instant::now();
    let (baked, report, _) = run_bake(&rs.device, &rs.queue, &mesh, &gpu_mesh, &settings, res);
    println!("  12 custom maps at {res}²: {:.2}s", start.elapsed().as_secs_f32());
    assert_eq!(report.custom, ids, "every map baked, in order");
    assert_eq!(baked.custom.len(), 3, "four maps a texture");

    // Every map, from its texture channel.
    let textures: Vec<Vec<[f32; 4]>> = baked
        .custom
        .iter()
        .map(|t| {
            read_texture(&rs.device, &rs.queue, t)
                .chunks_exact(8)
                .map(|b| [0, 1, 2, 3].map(|k| crate::gpu::f16_to_f32(u16::from_le_bytes([b[k * 2], b[k * 2 + 1]]))))
                .collect()
        })
        .collect();
    let values = |i: usize| -> Vec<f32> { textures[i / 4].iter().map(|t| t[i % 4]).collect() };
    for (i, c) in settings.custom.iter().enumerate() {
        let v = values(i);
        let max = v.iter().copied().fold(0.0f32, f32::max);
        let lit = v.iter().filter(|&&x| x > 0.02).count() as f32 / v.len() as f32;
        println!("    {:>2} {:<22} max {max:.2}, {:.1}% lit", i + 1, c.name, lit * 100.0);
        assert!(max > 0.3, "{} reaches bright values (max {max})", c.name);
        assert!(lit < 0.98, "{} isn't uniformly lit ({lit})", c.name);
    }
    // Rain from above and fluid from below wet different places.
    let (above, below) = (values(0), values(7));
    let differ = above.iter().zip(&below).filter(|(a, b)| (*a - *b).abs() > 0.2).count() as f32 / above.len() as f32;
    println!("    above vs below differ on {:.0}%", differ * 100.0);
    assert!(differ > 0.05, "flows from opposite directions differ");

    let mut engine = Engine::new(&rs, &gpu_mesh, (mesh.bounds_min, mesh.bounds_max), res);
    engine.set_maps(baked, report);
    // A red fill masked by the eleventh map (Contact, in the third texture).
    let contact = ids[10];
    let mut undo = crate::undo::UndoStack::default();
    let fill = FillParams { mat: MatValues { color: [0.8, 0.1, 0.05], ..Default::default() }, ..Default::default() };
    let idx = engine.add_fill_layer(0, None, "Custom".into(), fill, &mut undo);
    let mut g = GenKind::CustomMap.default_generator();
    g.custom[0] = CustomInput { id: contact, weight: 1.0 };
    engine.add_mask(idx, true, Some(g), &mut undo);
    assert!(engine.missing_custom(&g).is_empty(), "the mask's map was baked");
    let mut gone = g;
    gone.custom[0].id = 999;
    assert_eq!(engine.missing_custom(&gone), vec![999], "a removed map is reported missing");
    engine.composite();
    let px = read_texture(&rs.device, &rs.queue, engine.final_tex(0));
    let red = px.chunks(4).filter(|p| p[0] > 150 && p[1] < 90).count() as f32 / (px.len() / 4) as f32;
    println!("    contact mask covers {:.1}% in red", red * 100.0);
    assert!(red > 0.002 && red < 0.6, "the mask reads the eleventh map ({red})");

    let mut modes: Vec<(String, ViewMode)> = vec![("material".into(), ViewMode::Material)];
    for c in &settings.custom {
        modes.push((c.name.to_lowercase().replace(' ', "_"), ViewMode::Custom(c.id)));
    }
    let cam = OrbitCam { target: center, dist: 3.2, yaw: 0.35, pitch: 0.25, ..Default::default() };
    let mut sheet = image::RgbaImage::new(4 * 300, 4 * 300);
    for (k, (mname, mode)) in modes.iter().enumerate() {
        views.render_3d(&rs, [600, 600], &[&engine], &env, &gpu_mesh, &cam, &ViewSettings { mode: *mode, ..Default::default() }, None);
        let img = views.view3d.image();
        let px = read_texture(&rs.device, &rs.queue, img);
        image::save_buffer(dir.join(format!("many_{k:02}_{mname}.png")), &px, img.width, img.height, image::ExtendedColorType::Rgba8).unwrap();
        let small = image::imageops::resize(&image::RgbaImage::from_raw(img.width, img.height, px).unwrap(), 300, 300, image::imageops::FilterType::Triangle);
        image::imageops::replace(&mut sheet, &small, (k as i64 % 4) * 300, (k as i64 / 4) * 300);
    }
    sheet.save(dir.join("many_sheet.png")).unwrap();
    println!("  wrote {}", dir.display());
}

/// The live bake preview: adding custom maps bakes a quick preview, then a
/// sharper one (no restarts while nothing changes), a settings change bakes
/// again, and
/// closing the bake window ends it. Waits are time-bounded so a regression fails
/// instead of hanging.
#[test]
#[ignore]
fn live_bake_preview() {
    use crate::app::PainterApp;
    let mut setup = egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    setup.device_descriptor = Arc::new(|adapter| wgpu::DeviceDescriptor {
        label: Some("ui test device"),
        required_limits: adapter.limits(),
        ..Default::default()
    });
    let mut harness = egui_kittest::Harness::builder()
        .with_size([1600.0, 1300.0])
        .with_pixels_per_point(1.0)
        .with_accessibility_check(false)
        .wgpu_setup(egui_wgpu::WgpuSetup::CreateNew(setup))
        .build_eframe(|cc| PainterApp::new(cc).expect("app starts"));
    harness.run_steps(4);
    harness.state_mut().test_close_welcome();
    // A full bake first (leaving AO out), as when tuning an existing bake.
    let ctx = harness.ctx.clone();
    let mut frames = 0;
    while !harness.state_mut().test_bake_without(&ctx, 1) {
        harness.run_steps(1);
        frames += 1;
        assert!(frames < 2000, "the full bake finishes");
    }
    println!("  full bake after {frames} frames");
    harness.state_mut().test_show_custom();
    let wait = |harness: &mut egui_kittest::Harness<'_, PainterApp>, what: &str| {
        // Bounded by time: the sharpest preview step takes seconds, however
        // fast offscreen frames go.
        let start = std::time::Instant::now();
        let mut frames = 0;
        while start.elapsed().as_secs() < 60 {
            if harness.state().test_preview_shown() {
                println!("  {what}: preview after {frames} frames ({:.1}s), {} starts", start.elapsed().as_secs_f32(), harness.state().test_preview_starts());
                return;
            }
            harness.run_steps(1);
            frames += 1;
        }
        panic!("{what}: no preview after 60 s ({} starts)", harness.state().test_preview_starts());
    };
    wait(&mut harness, "custom maps added");
    let starts = harness.state().test_preview_starts();
    assert!(starts <= 4, "one preparation, a quick preview and two sharper ones ({starts} starts)");
    harness.run_steps(20);
    assert_eq!(harness.state().test_preview_starts(), starts, "no restarts while nothing changes");
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test/ui");
    std::fs::create_dir_all(&dir).unwrap();
    harness.render().expect("renders").save(dir.join("bake_live_preview.png")).unwrap();

    // Dragging the gizmo's handle sideways tips the fluid's direction away
    // from straight above, and the preview follows.
    let before = harness.state().test_custom(0).unwrap();
    let from = harness.state().test_gizmo_handle().expect("the flow gizmo is on screen");
    let to = from + egui::vec2(160.0, 40.0);
    let pointer = |harness: &mut egui_kittest::Harness<'_, PainterApp>, event: egui::Event| {
        harness.input_mut().events.push(event);
        harness.run_steps(1);
    };
    let button = |pos, pressed| egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: Default::default() };
    pointer(&mut harness, egui::Event::PointerMoved(from));
    pointer(&mut harness, button(from, true));
    for i in 1..=8 {
        pointer(&mut harness, egui::Event::PointerMoved(from + (to - from) * (i as f32 / 8.0)));
    }
    pointer(&mut harness, button(to, false));
    let after = harness.state().test_custom(0).unwrap();
    println!("  gizmo drag: elevation {:.1} -> {:.1}, compass {:.1} -> {:.1}", before.elevation, after.elevation, before.azimuth, after.azimuth);
    assert!(after.elevation < before.elevation - 5.0, "dragging the handle tipped the direction");
    wait(&mut harness, "direction dragged");
    harness.render().expect("renders").save(dir.join("bake_live_gizmo.png")).unwrap();

    harness.state_mut().test_set_bake_window(false);
    harness.run_steps(2);
    assert!(harness.state().test_preview_ended(), "closing the bake window ends the preview");
}

/// Seams in custom maps: the live preview's quick (512²) and refined
/// (1024²) bakes and a full bake of the same Wind Deposits setup, rendered close up on the material ball's
/// base. Images in `target/bake_test/custom/seams_*.png`.
#[test]
#[ignore]
fn custom_map_seams() {
    use crate::bake::CustomKind;
    // MP_PROJECT=path.mpp: a real mesh, usually cut into many UV islands.
    let mesh = match std::env::var("MP_PROJECT") {
        Ok(path) => load_project_mesh(&path).0,
        Err(_) => crate::mesh::Primitive::MaterialBall.build(),
    };
    let rs = render_state();
    let gpu_mesh = GpuMesh::new(&rs.device, &mesh);
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test/custom");
    std::fs::create_dir_all(&dir).unwrap();
    let mut views = ViewRenderer::new(&rs);
    let env = crate::environment::EnvBaker::new(&rs.device)
        .upload(&rs, &crate::environment::decode(&crate::environment::EnvSource::Builtin(0)).unwrap());
    let mut full = BakeSettings::default();
    full.apply_quality(crate::bake::Quality::Ultra);
    let i = full.add_custom(CustomKind::Deposit);
    let c = &mut full.custom[i];
    (c.azimuth, c.spread, c.distance) = (60.0, 50.0, 0.6);
    full.add_custom(CustomKind::Flow);
    let ids: Vec<u32> = full.custom.iter().map(|c| c.id).collect();
    let center = (mesh.bounds_min + mesh.bounds_max) * 0.5;
    for (name, settings, res) in [("preview", full.preview(), 512), ("refined", full.preview(), 1024), ("full", full.clone(), 2048)] {
        let (maps, report, _) = run_bake(&rs.device, &rs.queue, &mesh, &gpu_mesh, &settings, res);
        seam_report(&rs, &mesh, &maps.maps, res, name);
        let mut engine = Engine::new(&rs, &gpu_mesh, (mesh.bounds_min, mesh.bounds_max), 2048);
        engine.set_maps(maps, report);
        engine.composite();
        for (cname, yaw, pitch) in [("right", 0.9f32, 0.35f32), ("left", -0.6, 0.35)] {
            let cam = OrbitCam { target: center - Vec3::Y * 0.4, dist: 2.2, yaw, pitch, ..Default::default() };
            for (slot, map) in [(0, "deposit"), (1, "flow")] {
                views.render_3d(&rs, [900, 700], &[&engine], &env, &gpu_mesh, &cam, &ViewSettings { mode: ViewMode::Custom(ids[slot]), ..Default::default() }, None);
                let img = views.view3d.image();
                let px = read_texture(&rs.device, &rs.queue, img);
                image::save_buffer(dir.join(format!("seams_{map}_{name}_{cname}.png")), &px, img.width, img.height, image::ExtendedColorType::Rgba8).unwrap();
            }
        }
    }
}

/// A project file's mesh and its manifest (for tests that bake a real project).
fn load_project_mesh(path: &str) -> (Mesh, serde_json::Value) {
    let data = std::fs::read(path).expect("project file");
    let u64_at = |at: usize| u64::from_le_bytes(data[at..at + 8].try_into().unwrap()) as usize;
    let len = u64_at(8);
    let manifest: serde_json::Value = serde_json::from_slice(&data[16..16 + len]).unwrap();
    let mut at = 16 + len;
    let count = u64_at(at);
    at += 8;
    let mut blobs = Vec::new();
    for _ in 0..count {
        let n = u64_at(at);
        blobs.push(lz4_flex::decompress_size_prepended(&data[at + 8..at + 8 + n]).unwrap());
        at += 8 + n;
    }
    let m = &manifest["mesh"];
    let blob = |k: &str| &blobs[m[k].as_u64().unwrap() as usize];
    let vertices: Vec<crate::mesh::Vertex> = bytemuck::pod_collect_to_vec(blob("vertices"));
    let indices: Vec<u32> = bytemuck::pod_collect_to_vec(blob("indices"));
    let c: Vec<f32> = m["center"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
    let mesh = Mesh::from_saved("project".into(), vertices, indices, Vec3::new(c[0], c[1], c[2]), m["scale"].as_f64().unwrap() as f32);

    (mesh, manifest)
}

/// How much each scalar map jumps across UV seams: along every seam edge,
/// the value just inside the island on one side against the other side
/// (the same surface point), next to how much it changes between two texels
/// inside one island. A map without seams has both about equal.
fn seam_report(rs: &egui_wgpu::RenderState, mesh: &Mesh, maps: &[crate::gpu::Tex], res: u32, name: &str) {
    use crate::engine::MapKind;
    use glam::Vec2;
    let planes: Vec<Vec<[f32; 4]>> = maps
        .iter()
        .map(|t| {
            let bytes = read_texture(&rs.device, &rs.queue, t);
            match t.format {
                wgpu::TextureFormat::Rgba16Float => bytes
                    .chunks_exact(8)
                    .map(|b| [0, 1, 2, 3].map(|k| crate::gpu::f16_to_f32(u16::from_le_bytes([b[k * 2], b[k * 2 + 1]]))))
                    .collect(),
                _ => bytes.chunks_exact(4).map(|b| [0, 1, 2, 3].map(|k| b[k] as f32 / 255.0)).collect(),
            }
        })
        .collect();
    let at = |plane: &[[f32; 4]], comp: usize, uv: Vec2| -> f32 {
        let x = ((uv.x * res as f32) as i32).clamp(0, res as i32 - 1) as usize;
        let y = ((uv.y * res as f32) as i32).clamp(0, res as i32 - 1) as usize;
        plane[y * res as usize + x][comp]
    };
    // Edges by their two (quantized) end positions, with the UVs each
    // triangle gives them and the triangle's UV centroid.
    let key = |p: [f32; 3]| (Vec3::from_array(p) * 1e4).round().as_ivec3().to_array();
    let mut edges: std::collections::HashMap<([i32; 3], [i32; 3]), Vec<(Vec2, Vec2, Vec2)>> = Default::default();
    for t in mesh.indices.chunks_exact(3) {
        let v = [t[0], t[1], t[2]].map(|i| &mesh.vertices[i as usize]);
        let centroid = (Vec2::from_array(v[0].uv) + Vec2::from_array(v[1].uv) + Vec2::from_array(v[2].uv)) / 3.0;
        for k in 0..3 {
            let (a, b) = (v[k], v[(k + 1) % 3]);
            let (ka, kb) = (key(a.pos), key(b.pos));
            let (ua, ub) = (Vec2::from_array(a.uv), Vec2::from_array(b.uv));
            let entry = if ka <= kb { ((ka, kb), (ua, ub)) } else { ((kb, ka), (ub, ua)) };
            edges.entry(entry.0).or_default().push((entry.1.0, entry.1.1, centroid));
        }
    }
    let texel = 1.0 / res as f32;
    let mut samples: Vec<(Vec2, Vec2, Vec2)> = Vec::new(); // side A, side B, A further in
    for list in edges.values() {
        if list.len() != 2 {
            continue;
        }
        let ((a0, a1, ca), (b0, b1, cb)) = (list[0], list[1]);
        if a0.distance(b0) < texel && a1.distance(b1) < texel {
            continue; // not a seam
        }
        if a0.distance(a1) < 4.0 * texel || b0.distance(b1) < 4.0 * texel {
            continue; // too short to sample
        }
        for t in [0.3f32, 0.5, 0.7] {
            let (pa, pb) = (a0.lerp(a1, t), b0.lerp(b1, t));
            let ina = (ca - pa).normalize_or_zero();
            let inb = (cb - pb).normalize_or_zero();
            samples.push((pa + ina * 2.0 * texel, pb + inb * 2.0 * texel, pa + ina * 4.0 * texel));
        }
    }
    println!("  {name}: {} seam samples at {res}^2; mean |jump| across seams vs between texels inside an island:", samples.len());
    for m in MAPS.iter().filter(|m| m.kind == MapKind::Scalar) {
        let plane = &planes[m.tex];
        let (mut across, mut inside) = (0.0f64, 0.0f64);
        for &(a, b, a2) in &samples {
            across += (at(plane, m.comp, a) - at(plane, m.comp, b)).abs() as f64;
            inside += (at(plane, m.comp, a) - at(plane, m.comp, a2)).abs() as f64;
        }
        let n = samples.len().max(1) as f64;
        println!("    {:<18} across {:.4}  inside {:.4}  ratio {:.1}", m.name, across / n, inside / n, across / inside.max(1e-9));
    }
}

/// Close-ups of the material ball's base with UV seams drawn over the
/// custom maps and AO, to see whether lines in a map sit on the seams.
/// Images in `target/bake_test/custom/closeup_*.png`.
#[test]
#[ignore]
fn custom_map_closeups() {
    use crate::bake::CustomKind;
    let mesh = crate::mesh::Primitive::MaterialBall.build();
    let rs = render_state();
    let gpu_mesh = GpuMesh::new(&rs.device, &mesh);
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test/custom");
    std::fs::create_dir_all(&dir).unwrap();
    let mut views = ViewRenderer::new(&rs);
    let env = crate::environment::EnvBaker::new(&rs.device)
        .upload(&rs, &crate::environment::decode(&crate::environment::EnvSource::Builtin(0)).unwrap());
    let mut s = BakeSettings::default();
    s.apply_quality(crate::bake::Quality::Standard);
    let i = s.add_custom(CustomKind::Deposit);
    let d = &mut s.custom[i];
    (d.azimuth, d.spread, d.distance) = (270.0, 50.0, 0.6);
    s.add_custom(CustomKind::Flow);
    s.add_custom(CustomKind::Exposure);
    let ids: Vec<u32> = s.custom.iter().map(|c| c.id).collect();
    let (maps, report, low_pos) = run_bake(&rs.device, &rs.queue, &mesh, &gpu_mesh, &s, 2048);
    // The flow map (second channel of the first custom texture) and the island mask, in UV space.
    let f = read_texture(&rs.device, &rs.queue, &maps.custom[0]);
    let flow: Vec<u8> = f.chunks_exact(8).map(|b| (crate::gpu::f16_to_f32(u16::from_le_bytes([b[2], b[3]])).clamp(0.0, 1.0) * 255.0) as u8).collect();
    image::save_buffer(dir.join("uv_flow.png"), &flow, 2048, 2048, image::ExtendedColorType::L8).unwrap();
    let lp = read_texture(&rs.device, &rs.queue, &low_pos);
    let mask: Vec<u8> = lp.chunks_exact(16).map(|b| (f32::from_le_bytes([b[12], b[13], b[14], b[15]]).clamp(0.0, 1.0) * 255.0) as u8).collect();
    image::save_buffer(dir.join("uv_islands.png"), &mask, 2048, 2048, image::ExtendedColorType::L8).unwrap();
    let mut engine = Engine::new(&rs, &gpu_mesh, (mesh.bounds_min, mesh.bounds_max), 2048);
    engine.set_maps(maps, report);
    engine.composite();
    let base = Vec3::new(mesh.bounds_max.x * 0.8, mesh.bounds_min.y + (mesh.bounds_max.y - mesh.bounds_min.y) * 0.2, mesh.bounds_max.z * 0.8);
    let cam = OrbitCam { target: base, dist: 1.1, yaw: 0.75, pitch: 0.45, ..Default::default() };
    for (name, mode) in [
        ("deposit", ViewMode::Custom(ids[0])),
        ("flow", ViewMode::Custom(ids[1])),
        ("exposure", ViewMode::Custom(ids[2])),
        ("ao", map_mode("Ambient Occlusion")),
        ("sky", map_mode("Sky Exposure")),
    ] {
        for seams in [false, true] {
            let settings = ViewSettings { mode, show_seams_3d: seams, ..Default::default() };
            views.render_3d(&rs, [800, 800], &[&engine], &env, &gpu_mesh, &cam, &settings, None);
            let img = views.view3d.image();
            let px = read_texture(&rs.device, &rs.queue, img);
            let file = dir.join(format!("closeup_{name}{}.png", if seams { "_seams" } else { "" }));
            image::save_buffer(&file, &px, img.width, img.height, image::ExtendedColorType::Rgba8).unwrap();
        }
    }
}

/// Layer thumbnails show what the layer and its mask really do: a fill's
/// color, a mask that inverts or breaks up with noise, a painted mask, and
/// they follow changes. Images in `target/bake_test/thumbs/`.
#[test]
#[ignore]
fn layer_thumbnails() {
    use crate::engine::{FillParams, FilterKind, GenKind, MaskFilter, MatValues};
    let rs = render_state();
    let mesh = crate::mesh::Primitive::MaterialBall.build();
    let gpu_mesh = GpuMesh::new(&rs.device, &mesh);
    let mut engine = Engine::new(&rs, &gpu_mesh, (mesh.bounds_min, mesh.bounds_max), 512);
    let (maps, report, _) = run_bake(&rs.device, &rs.queue, &mesh, &gpu_mesh, &BakeSettings::default(), 512);
    engine.set_maps(maps, report);
    let mut undo = crate::undo::UndoStack::default();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test/thumbs");
    std::fs::create_dir_all(&dir).unwrap();
    let stats = |engine: &Engine, tex: &crate::gpu::Tex, name: &str| -> (f32, f32, [f32; 3]) {
        let px = read_texture(&rs.device, &rs.queue, tex);
        image::save_buffer(dir.join(format!("{name}.png")), &px, tex.width, tex.height, image::ExtendedColorType::Rgba8).unwrap();
        let n = (px.len() / 4) as f32;
        let lum: Vec<f32> = px.chunks(4).map(|p| (p[0] as f32 + p[1] as f32 + p[2] as f32) / (3.0 * 255.0)).collect();
        let mean = lum.iter().sum::<f32>() / n;
        let var = lum.iter().map(|l| (l - mean) * (l - mean)).sum::<f32>() / n;
        let rgb = [0, 1, 2].map(|k| px.chunks(4).map(|p| p[k] as f32 / 255.0).sum::<f32>() / n);
        let _ = engine;
        (mean, var.sqrt(), rgb)
    };

    // A red fill with color variation, masked by edge wear.
    let fill = FillParams { mat: MatValues { color: [0.8, 0.1, 0.05], ..Default::default() }, color_var: 0.4, ..Default::default() };
    let idx = engine.add_fill_layer(0, None, "Red".into(), fill, &mut undo);
    engine.add_mask(idx, true, Some(GenKind::EdgeWear.default_generator()), &mut undo);
    let id = engine.layers[idx].id;
    engine.composite();
    let (content, mask) = engine.thumb_textures(id);
    let (_, spread, rgb) = stats(&engine, content.expect("fill thumbnail"), "fill_content");
    println!("  fill: rgb {rgb:?}, spread {spread:.3}");
    assert!(rgb[0] > rgb[1] * 2.0 && rgb[0] > rgb[2] * 2.0, "the fill thumbnail is red: {rgb:?}");
    assert!(spread > 0.01, "color variation shows ({spread})");
    let (wear, wear_spread, _) = stats(&engine, mask.expect("mask thumbnail"), "mask_edge_wear");
    println!("  edge wear mask: mean {wear:.3}, spread {wear_spread:.3}");
    assert!(wear > 0.01 && wear < 0.6 && wear_spread > 0.05, "edge wear covers the edges, not everything");

    // An Invert filter flips it.
    engine.layers[idx].mask.as_mut().unwrap().filters = vec![MaskFilter::new(FilterKind::Invert)];
    engine.mark_dirty();
    engine.composite();
    let (inverted, _, _) = stats(&engine, engine.thumb_textures(id).1.unwrap(), "mask_inverted");
    // Off the islands both are the same dark gray (0.16); on them they sum to
    // white. The fill thumbnail's alpha says how much is on them.
    let covered = {
        let px = read_texture(&rs.device, &rs.queue, engine.thumb_textures(id).0.unwrap());
        px.chunks(4).map(|p| p[3] as f32 / 255.0).sum::<f32>() / (px.len() / 4) as f32
    };
    let expected = covered + 2.0 * 0.16 * (1.0 - covered);
    println!("  inverted: mean {inverted:.3}; islands cover {:.0}%", covered * 100.0);
    assert!((wear + inverted - expected).abs() < 0.05, "inverting the mask inverts its thumbnail ({wear:.3} + {inverted:.3}, expected {expected:.3})");

    // A new fill color shows.
    if let crate::engine::Content::Fill(f) = &mut engine.layers[idx].content {
        f.mat.color = [0.05, 0.2, 0.9];
    }
    engine.mark_dirty();
    engine.composite();
    let (_, _, blue) = stats(&engine, engine.thumb_textures(id).0.unwrap(), "fill_blue");
    assert!(blue[2] > blue[0] * 2.0, "the thumbnail follows the fill's color: {blue:?}");
}

/// Painting a fill's mask rebuilds no thumbnail while the stroke is held, and
/// on release only the mask's: the fill's content thumbnail is a full 512²
/// composite, which once stalled every paint frame by over 100 ms.
#[test]
#[ignore]
fn mask_stroke_redoes_only_the_mask_thumbnail() {
    use crate::app::{BrushProjection, resolve_stamps};
    use crate::engine::{BrushPaint, CHANNELS, FillParams, MatValues, StampUniform};
    use crate::undo::{StrokeTarget, UndoStack};
    use glam::Vec2;

    let mesh = Arc::new(crate::mesh::Primitive::Plane.build());
    let bvh = crate::bvh::Bvh::from_mesh(&mesh);
    let rs = render_state();
    let gpu_mesh = GpuMesh::new(&rs.device, &mesh);
    let mut engine = Engine::new(&rs, &gpu_mesh, (mesh.bounds_min, mesh.bounds_max), 256);
    let mut undo = UndoStack::default();
    let idx = engine.add_fill_layer(0, None, "Red".into(), FillParams::default(), &mut undo);
    engine.add_mask(idx, true, None, &mut undo);
    engine.composite();
    let settled = engine.thumb_renders();
    assert_eq!(settled, 2, "a fill with a mask has a content and a mask thumbnail");

    assert!(engine.begin_stroke(idx, StrokeTarget::Mask, [false; CHANNELS], false));
    let cam = OrbitCam { pitch: 1.5, dist: 3.2, ..Default::default() };
    let size = [512u32, 512u32];
    let fsize = Vec2::new(size[0] as f32, size[1] as f32);
    let mut views = ViewRenderer::new(&rs);
    let env = crate::environment::EnvBaker::new(&rs.device)
        .upload(&rs, &crate::environment::decode(&crate::environment::EnvSource::Builtin(0)).unwrap());
    views.render_3d(&rs, size, &[&engine], &env, &gpu_mesh, &cam, &ViewSettings::default(), None);
    let radius = 30.0;
    let su = StampUniform {
        view_proj: cam.view_proj(fsize.x / fsize.y).to_cols_array_2d(),
        cam_pos: cam.eye().extend(1.0).to_array(),
        view: [fsize.x, fsize.y, 0.0, 0.0],
        view2d: [0.0; 4],
        brush: [radius, 0.6, 1.0, 0.0],
        clip: [crate::viewport::NEAR, crate::viewport::FAR, 0.0, 0.0],
        proj: [BrushProjection::Screen as u32 as f32, 0.0, 85f32.to_radians().cos(), 0.0],
    };
    let paint = BrushPaint { mat: MatValues::default(), mask_value: 0.0, opacity: 1.0, erase: false, grain: [0.0; 4] };
    // Several paint frames: stamps, then the recomposition each one triggers.
    for frame in 0..4 {
        let raw: Vec<([f32; 4], f32)> = (0..10).map(|i| ([100.0 + (frame * 10 + i) as f32 * 8.0, 256.0, 1.0, 0.0], 1.0)).collect();
        let stamps = resolve_stamps(&mesh, &bvh, &cam, fsize, BrushProjection::Screen, 1, radius, &raw);
        engine.stroke_step(&stamps, su, views.view3d.depth_view(), &paint, None);
        engine.composite();
        assert_eq!(engine.thumb_renders(), settled, "no thumbnail is rebuilt while the stroke is held (frame {frame})");
    }
    engine.end_stroke();
    assert_eq!(engine.thumb_renders(), settled + 1, "on release only the mask's thumbnail is redone");
    engine.composite();
    assert_eq!(engine.thumb_renders(), settled + 1, "and the next recomposition doesn't redo it");
}

/// What darkens crevices in the 3D view: the same gray material ball with
/// the baked AO at full and at the default view strength, without AO, and
/// without the key light.
/// Images in `target/bake_test/viewport/`.
#[test]
#[ignore]
fn viewport_crevices() {
    use crate::engine::{FillParams, MatValues};
    let mesh = crate::mesh::Primitive::MaterialBall.build();
    let rs = render_state();
    let gpu_mesh = GpuMesh::new(&rs.device, &mesh);
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test/viewport");
    std::fs::create_dir_all(&dir).unwrap();
    let env = crate::environment::EnvBaker::new(&rs.device)
        .upload(&rs, &crate::environment::decode(&crate::environment::EnvSource::Builtin(0)).unwrap());
    let mut views = ViewRenderer::new(&rs);
    let center = (mesh.bounds_min + mesh.bounds_max) * 0.5;
    let cam = OrbitCam { target: center - Vec3::Y * 0.3, dist: 2.4, yaw: 0.5, pitch: 0.3, ..Default::default() };
    let make = |baked: bool| {
        let mut engine = Engine::new(&rs, &gpu_mesh, (mesh.bounds_min, mesh.bounds_max), 1024);
        if baked {
            let (maps, report, _) = run_bake(&rs.device, &rs.queue, &mesh, &gpu_mesh, &BakeSettings::default(), 1024);
            engine.set_maps(maps, report);
        }
        let fill = FillParams { mat: MatValues { color: [0.7, 0.7, 0.7], rough: 0.5, ..Default::default() }, ..Default::default() };
        engine.add_fill_layer(0, None, "Gray".into(), fill, &mut crate::undo::UndoStack::default());
        engine.composite();
        engine
    };
    let (baked, plain) = (make(true), make(false));
    for (name, engine, key, ao) in [
        ("baked_full", &baked, true, 1.0),
        ("baked_default", &baked, true, ViewSettings::default().ao_strength),
        ("noao", &plain, true, 1.0),
        ("baked_nokey", &baked, false, 1.0),
    ] {
        let mut s = ViewSettings::default();
        s.key.on = key;
        s.ao_strength = ao;
        views.render_3d(&rs, [900, 900], &[engine], &env, &gpu_mesh, &cam, &s, None);
        let img = views.view3d.image();
        let px = read_texture(&rs.device, &rs.queue, img);
        let mean = px.chunks(4).map(|p| p[0] as f32).sum::<f32>() / (px.len() / 4) as f32;
        println!("  {name}: mean {mean:.1}");
        image::save_buffer(dir.join(format!("{name}.png")), &px, img.width, img.height, image::ExtendedColorType::Rgba8).unwrap();
    }
}

/// Opening Substance 3D Painter projects of your own (`MP_SPP=a.spp;b.spp`): the flattened
/// result as a visible paint layer, the rebuilt stack hidden next to it, the report up.
/// Screenshots in `target/bake_test/spp/`. Nothing runs without the variable (no Painter file
/// ships with the tests; `spp_import`'s unit tests use a synthetic one).
#[test]
#[ignore]
fn open_spp_projects() {
    use crate::app::PainterApp;
    let Ok(paths) = std::env::var("MP_SPP") else {
        println!("set MP_SPP to .spp files of your own");
        return;
    };
    let mut setup = egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    setup.device_descriptor = Arc::new(|adapter| wgpu::DeviceDescriptor {
        label: Some("ui test device"),
        required_limits: adapter.limits(),
        ..Default::default()
    });
    let mut harness = egui_kittest::Harness::builder()
        .with_size([1600.0, 1000.0])
        .with_pixels_per_point(1.0)
        .with_accessibility_check(false)
        .wgpu_setup(egui_wgpu::WgpuSetup::CreateNew(setup))
        .build_eframe(|cc| PainterApp::new(cc).expect("app starts"));
    harness.run_steps(4);
    harness.state_mut().test_close_welcome();
    // MP_PREFS: a preferences.json to run with (bake quality, default size) instead of the defaults.
    if let Some(json) = std::env::var("MP_PREFS").ok().and_then(|p| std::fs::read_to_string(p).ok()) {
        harness.state_mut().test_user_prefs(&json);
    }
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test/spp");
    std::fs::create_dir_all(&dir).unwrap();
    for (k, path) in paths.split(';').filter(|p| !p.is_empty()).enumerate() {
        let ctx = harness.ctx.clone();
        harness.state_mut().test_open_spp(Path::new(path), &ctx);
        let start = std::time::Instant::now();
        while !harness.state().test_spp_done() && start.elapsed().as_secs() < 120 {
            harness.run_steps(1);
        }
        assert!(harness.state().test_spp_done(), "{path}: the import finished");
        let layers = harness.state().test_layers();
        println!("  {path}:");
        for (name, kind, visible) in &layers {
            println!("    {kind:<12} {} {name}", if *visible { "shown " } else { "hidden" });
        }
        let flat = layers.iter().find(|(n, ..)| n == "Painter result (flattened)").expect("the flattened layer");
        assert_eq!((flat.1, flat.2), ("paint", true), "the flattened result is a visible paint layer");
        if let Some(folder) = layers.iter().find(|(n, ..)| n == "Painter layers (rebuilt)") {
            assert!(!folder.2, "the rebuilt stack starts hidden");
        }
        // Every texture set came in, each with its flattened result.
        let sets = harness.state().test_sets();
        for (name, res, layers) in &sets {
            println!("    set {name}: {res}², {} layers", layers.len());
            assert!(layers.iter().any(|l| l == "Painter result (flattened)"), "{name} has its flattened layer");
        }
        // Let the bakes land (every set's, one after another), then picture it with the report.
        let mut steps = 0;
        while harness.state().test_baking() && steps < 20_000 {
            harness.run_steps(1);
            steps += 1;
        }
        assert!(!harness.state().test_baking(), "{path}: the bakes finished");
        harness.run_steps(2);
        for (name, parts) in harness.state().test_vram() {
            let total: u64 = parts.iter().map(|p| p.1).sum();
            println!("    vram {name}: {total} MB {parts:?}");
        }
        harness.render().expect("renders").save(dir.join(format!("spp_{k}.png"))).unwrap();
        // The path tracer reads every set.
        harness.state_mut().test_ray_trace(true);
        harness.run_steps(8);
        harness.render().expect("renders").save(dir.join(format!("spp_{k}_raytraced.png"))).unwrap();
        harness.state_mut().test_ray_trace(false);
        // Saved and opened again, every set keeps its size and layers.
        let ctx = harness.ctx.clone();
        let project = dir.join(format!("spp_{k}.mpp"));
        harness.state_mut().test_save_and_reopen(&project, &ctx).expect("saves and reopens");
        harness.run_steps(2);
        assert_eq!(harness.state().test_sets(), sets, "{path}: the project round-trips");
        let _ = std::fs::remove_file(&project);
    }
}

/// Two texture sets on one plane (left and right halves of the UV square):
/// a stroke across the whole plane paints only the edited set's half, and
/// the 3D view draws each half with its own set. The frame goes to
/// `target/bake_test/texture_sets.png`.
#[test]
#[ignore]
fn texture_sets_paint_apart() {
    use crate::app::{BrushProjection, resolve_stamps};
    use crate::engine::{BrushPaint, FillParams, MatValues, StampUniform};
    use crate::undo::{StrokeTarget, UndoStack};
    use glam::Vec2;

    let mesh = crate::mesh::set_tests::split_plane();
    let bvh = crate::bvh::Bvh::from_mesh(&mesh);
    let rs = render_state();
    let gpu_mesh = GpuMesh::new(&rs.device, &mesh);
    let res = 256;
    let mut left = Engine::new(&rs, &gpu_mesh, (mesh.bounds_min, mesh.bounds_max), res);
    left.set_triangles(mesh.sets[0].index_range(), &gpu_mesh);
    let mut right = left.sibling(&gpu_mesh, mesh.sets[1].index_range(), res);
    let mut undo = UndoStack::default();
    left.add_paint_layer(0, None, None, &mut undo);
    let blue = MatValues { color: [0.05, 0.1, 0.9], ..Default::default() };
    right.add_fill_layer(0, None, "Blue".into(), FillParams { mat: blue, ..Default::default() }, &mut undo);

    let mut views = ViewRenderer::new(&rs);
    let env = crate::environment::EnvBaker::new(&rs.device)
        .upload(&rs, &crate::environment::decode(&crate::environment::EnvSource::Builtin(0)).unwrap());
    // Looking straight down at the plane.
    let cam = OrbitCam { pitch: 1.5, dist: 3.2, ..Default::default() };
    let size = [512u32, 512u32];
    let settings = ViewSettings::default();
    views.render_3d(&rs, size, &[&left, &right], &env, &gpu_mesh, &cam, &settings, None);

    // A stroke across the middle of the view, over both halves.
    assert!(left.begin_stroke(0, StrokeTarget::Content, [true, false, false, false, false, false, false, false], false));
    let radius = 40.0;
    let raw: Vec<([f32; 4], f32)> = (0..90).map(|i| ([30.0 + i as f32 * 5.0, 256.0, 1.0, 0.0], 1.0)).collect();
    let fsize = Vec2::new(size[0] as f32, size[1] as f32);
    let stamps = resolve_stamps(&mesh, &bvh, &cam, fsize, BrushProjection::Screen, 0, radius, &raw);
    assert!(!stamps.is_empty(), "the stroke hits the plane");
    let su = StampUniform {
        view_proj: cam.view_proj(fsize.x / fsize.y).to_cols_array_2d(),
        cam_pos: cam.eye().extend(1.0).to_array(),
        view: [fsize.x, fsize.y, 0.0, 0.0],
        view2d: [0.0; 4],
        // No mirroring (brush.w).
        brush: [radius, 0.6, 1.0, 0.0],
        clip: [crate::viewport::NEAR, crate::viewport::FAR, 0.0, 0.0],
        proj: [BrushProjection::Screen as u32 as f32, 0.0, 85f32.to_radians().cos(), 0.0],
    };
    let paint = BrushPaint {
        mat: MatValues { enabled: [true, false, false, false, false, false, false, false], color: [0.9, 0.1, 0.05], ..Default::default() },
        mask_value: 1.0,
        opacity: 1.0,
        erase: false,
        grain: [0.0; 4],
    };
    left.stroke_step(&stamps, su, views.view3d.depth_view(), &paint, None);
    left.end_stroke();
    left.composite();
    right.composite();

    // Red texels of the left set, by side of the UV square (past the
    // dilation margin on the right).
    let bytes = read_texture(&rs.device, &rs.queue, left.final_tex(0));
    let (mut on_left, mut on_right) = (0, 0);
    for (i, p) in bytes.chunks(4).enumerate() {
        let x = (i as u32 % res) as f32 / res as f32;
        if p[0] > 150 && p[1] < 90 {
            if x < 0.5 {
                on_left += 1;
            } else if x > 0.5 + 16.0 / res as f32 {
                on_right += 1;
            }
        }
    }
    println!("  painted texels: {on_left} in the left set's half, {on_right} past it");
    assert!(on_left > 500, "the left set took paint ({on_left})");
    assert_eq!(on_right, 0, "paint stays in the edited set");

    views.render_3d(&rs, size, &[&left, &right], &env, &gpu_mesh, &cam, &settings, None);
    let img = views.view3d.image();
    let px = read_texture(&rs.device, &rs.queue, img);
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test");
    std::fs::create_dir_all(&dir).unwrap();
    image::save_buffer(dir.join("texture_sets.png"), &px, img.width, img.height, image::ExtendedColorType::Rgba8).unwrap();
    let rgb = |p: &[u8]| (p[0] as i32, p[1] as i32, p[2] as i32);
    let bluish = px.chunks(4).map(rgb).filter(|&(r, g, b)| b > r + 40 && b > g + 40).count();
    let reddish = px.chunks(4).map(rgb).filter(|&(r, g, b)| r > g + 60 && r > b + 60).count();
    println!("  view: {bluish} blue pixels (right set), {reddish} red (the stroke)");
    assert!(bluish > 5000, "the right set draws with its own textures ({bluish})");
    assert!(reddish > 500, "the stroke shows ({reddish})");

    // Hidden, the right set isn't drawn; highlighted, the left one turns toward the accent.
    let looks = |views: &mut ViewRenderer, l: [crate::viewport::SetLook; 2]| {
        views.set_looks = l.to_vec();
        views.render_3d(&rs, size, &[&left, &right], &env, &gpu_mesh, &cam, &settings, None);
        read_texture(&rs.device, &rs.queue, views.view3d.image())
    };
    use crate::viewport::SetLook;
    let hidden = looks(&mut views, [SetLook::default(), SetLook { hidden: true, highlight: 0.0 }]);
    let bluish_hidden = hidden.chunks(4).filter(|p| p[2] as i32 > p[0] as i32 + 40 && p[2] as i32 > p[1] as i32 + 40).count();
    println!("  right set hidden: {bluish_hidden} blue pixels");
    assert!(bluish_hidden < bluish / 20, "a hidden set isn't drawn ({bluish_hidden})");
    let lit = looks(&mut views, [SetLook { hidden: false, highlight: 1.0 }, SetLook { hidden: true, highlight: 0.0 }]);
    image::save_buffer(dir.join("texture_sets_highlight.png"), &lit, size[0], size[1], image::ExtendedColorType::Rgba8).unwrap();
    // How far the image leans toward the accent (blue over red), summed.
    let lean = |px: &[u8]| px.chunks(4).map(|p| (p[2] as i64 - p[0] as i64).max(0)).sum::<i64>();
    println!("  left set highlighted: lean {} (before {})", lean(&lit), lean(&hidden));
    assert!(lean(&lit) > lean(&hidden) + 1_000_000, "the highlight tints the set");
}

/// A smart material dropped as a layer folder and edited through its live
/// graph: the graph shows the layers, layer edits show in it, graph edits
/// reach the layers (as undoable edits), and the artist's own nodes wired
/// into a layer's mask become its mask graph. Frames go to
/// `target/bake_test/ui/folder_graph*.png`.
#[test]
#[ignore]
fn smart_folder_live_graph() {
    use crate::app::PainterApp;
    use crate::engine::Content;
    use crate::graph::smart::NODE_CHANNEL;
    use crate::graph::{Role, Value};
    let mut setup = egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    setup.device_descriptor = Arc::new(|adapter| wgpu::DeviceDescriptor {
        label: Some("ui test device"),
        required_limits: adapter.limits(),
        ..Default::default()
    });
    let mut harness = egui_kittest::Harness::builder()
        .with_size([1600.0, 1000.0])
        .with_pixels_per_point(1.0)
        .with_accessibility_check(false)
        .wgpu_setup(egui_wgpu::WgpuSetup::CreateNew(setup))
        .build_eframe(|cc| PainterApp::new(cc).expect("app starts"));
    harness.run_steps(4);
    harness.state_mut().test_close_dialogs();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test/ui");
    std::fs::create_dir_all(&dir).unwrap();

    let k = presets::SMART_MATERIALS.iter().position(|s| s.layers.iter().filter(|l| l.generator().is_some()).count() >= 2).unwrap();
    let folder = harness.state_mut().test_add_smart_folder(k);
    let doc = harness.state_mut().test_open_folder_graph(folder).expect("a live graph");
    harness.run_steps(3);
    let folder_id = harness.state_mut().test_engine_mut().layers[folder].id;
    // (index, id, enabled channels, smart mask) of the fill layers inside.
    let kids = |h: &mut egui_kittest::Harness<'_, PainterApp>| -> Vec<(usize, u64, [bool; crate::engine::CHANNELS], bool)> {
        let e = h.state_mut().test_engine_mut();
        e.layers
            .iter()
            .enumerate()
            .filter(|(_, l)| l.parent == Some(folder_id))
            .filter_map(|(i, l)| match &l.content {
                Content::Fill(f) => Some((i, l.id, f.mat.enabled, l.mask.as_ref().is_some_and(|m| m.generator.is_some()))),
                _ => None,
            })
            .collect()
    };
    let layers = kids(&mut harness);
    {
        let g = harness.state_mut().test_graph_mut(doc).unwrap().clone();
        for (_, id, enabled, masked) in &layers {
            for (nk, ch) in NODE_CHANNEL.iter().enumerate() {
                assert_eq!(g.tagged(*id, Role::Channel(nk)).is_some(), enabled[*ch], "a node per channel the layer writes");
            }
            assert_eq!(g.tagged(*id, Role::Mask).is_some(), *masked, "a Smart Mask node per smart mask");
        }
    }
    harness.render().expect("renders").save(dir.join("folder_graph.png")).unwrap();

    // A layer edited in the layer stack shows in the graph.
    let (top_idx, top, ..) = *layers.iter().rev().find(|l| l.2[0]).expect("a layer with base color");
    harness.state_mut().test_engine_mut().layers[top_idx].opacity = 0.4;
    harness.run_steps(2);
    {
        let g = harness.state_mut().test_graph_mut(doc).unwrap();
        let n = g.tagged(top, Role::Channel(0)).expect("base color node");
        assert_eq!(g.node(n).unwrap().params.get("opacity"), Some(&Value::Float(0.4)), "the graph follows the layer");
    }

    // An edit in the graph reaches the layer, as an undoable edit.
    let serial = harness.state().test_undo_serial();
    {
        let g = harness.state_mut().test_graph_mut(doc).unwrap();
        let n = g.tagged(top, Role::Channel(0)).unwrap();
        g.node_mut(n).unwrap().params.insert("color".into(), Value::Color([0.9, 0.1, 0.1]));
    }
    harness.run_steps(2);
    match &harness.state_mut().test_engine_mut().layers[top_idx].content {
        Content::Fill(f) => assert_eq!(f.mat.color, [0.9, 0.1, 0.1], "the layer follows the graph"),
        _ => panic!("still a fill layer"),
    }
    assert_ne!(harness.state().test_undo_serial(), serial, "recorded for undo");

    // A blur of the smart mask wired into the layer's mask: a mask graph.
    let (m_idx, m_id, ..) = *layers.iter().find(|l| l.3).unwrap();
    let blur = {
        let g = harness.state_mut().test_graph_mut(doc).unwrap();
        let smart = g.tagged(m_id, Role::Mask).unwrap();
        let blur = g.add("blur", [0.0, -600.0]);
        assert!(g.connect(smart, blur, "input"));
        let first = (0..5).find_map(|nk| g.tagged(m_id, Role::Channel(nk))).unwrap();
        assert!(g.connect(blur, first, "mask"));
        blur
    };
    let mut steps = 0;
    while harness.state_mut().test_engine_mut().layers[m_idx].mask.as_ref().is_none_or(|m| m.paint.is_none()) && steps < 200 {
        harness.run_steps(1);
        steps += 1;
    }
    {
        let mask = harness.state_mut().test_engine_mut().layers[m_idx].mask.as_ref().unwrap();
        assert!(mask.graph.is_some() && mask.generator.is_none(), "the mask is a graph now");
        assert!(mask.paint.is_some(), "and it rendered");
        let g = harness.state_mut().test_graph_mut(doc).unwrap();
        for nk in 0..5 {
            if let Some(n) = g.tagged(m_id, Role::Channel(nk)) {
                assert_eq!(g.node(n).unwrap().links.get("mask"), Some(&blur), "every channel of the layer reads it");
            }
        }
    }
    harness.render().expect("renders").save(dir.join("folder_graph_custom_mask.png")).unwrap();

    // Undo brings the smart mask back, in the layer and in the graph.
    harness.state_mut().test_undo_step();
    harness.run_steps(2);
    let generator = harness.state_mut().test_engine_mut().layers[m_idx].mask.as_ref().and_then(|m| m.generator);
    assert!(generator.is_some(), "the smart mask is back");
    let g = harness.state_mut().test_graph_mut(doc).unwrap();
    let smart = g.tagged(m_id, Role::Mask).expect("a Smart Mask node again");
    let first = (0..5).find_map(|nk| g.tagged(m_id, Role::Channel(nk))).unwrap();
    assert_eq!(g.node(first).unwrap().links.get("mask"), Some(&smart));
    let tags = g.tags.len();

    // Saved with the project: the folder keeps its graph, tags and all.
    let file = dir.join("folder_graph.mpp");
    let ctx = harness.ctx.clone();
    harness.state_mut().test_save_and_reopen(&file, &ctx).expect("saves and opens");
    harness.run_steps(2);
    let e = harness.state_mut().test_engine_mut();
    let fi = e.layers.iter().position(|l| l.id == folder_id).expect("the folder");
    let Content::Folder(f) = &e.layers[fi].content else { panic!("still a folder") };
    let doc = f.graph.expect("its graph");
    assert_eq!(harness.state_mut().test_graph_mut(doc).expect("loaded").tags.len(), tags);
}

/// The polygon fill toolbar (every selection mode, box and lasso) and the
/// texture set list (hide, highlight, flash on click), offscreen. Frames go to
/// `target/bake_test/ui/poly_tool.png` and `texture_set_list.png`.
#[test]
#[ignore]
fn texture_set_list_and_poly_tool() {
    use crate::app::PainterApp;
    use egui_kittest::kittest::Queryable;
    let mut setup = egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    setup.device_descriptor = Arc::new(|adapter| wgpu::DeviceDescriptor {
        label: Some("ui test device"),
        required_limits: adapter.limits(),
        ..Default::default()
    });
    let mut harness = egui_kittest::Harness::builder()
        .with_size([1600.0, 1000.0])
        .with_pixels_per_point(1.0)
        .with_accessibility_check(false)
        .wgpu_setup(egui_wgpu::WgpuSetup::CreateNew(setup))
        .build_eframe(|cc| PainterApp::new(cc).expect("app starts"));
    harness.run_steps(4);
    harness.state_mut().test_close_dialogs();
    let ctx = harness.ctx.clone();
    harness.state_mut().test_load_mesh(crate::mesh::set_tests::split_plane(), &ctx);
    harness.state_mut().test_poly_tool(crate::polyfill::FillMode::Loop, true);
    harness.run_steps(3);
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test/ui");
    std::fs::create_dir_all(&dir).unwrap();
    harness.render().expect("renders").save(dir.join("poly_tool.png")).unwrap();
    assert!(harness.query_all_by_label_contains("Face loop").count() > 0, "the active mode is named");

    // Shift+drag a box over the 3D view fills the faces inside on release,
    // with the 2D view open too (each view runs its input every frame).
    harness.state_mut().test_poly_tool(crate::polyfill::FillMode::Triangle, false);
    harness.run_steps(2);
    let rect = harness.state().test_view_rect().expect("the 3D view");
    let before = harness.state().test_undo_serial();
    let shift = egui::Modifiers::SHIFT;
    let event = |h: &mut egui_kittest::Harness<'_, PainterApp>, e: egui::Event| h.input_mut().events.push(e);
    let button = |pos, pressed| egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: shift };
    let (from, to) = (rect.center() - egui::vec2(60.0, 60.0), rect.center() + egui::vec2(60.0, 60.0));
    event(&mut harness, egui::Event::ModifiersChanged(shift));
    event(&mut harness, egui::Event::PointerMoved(from));
    harness.run_steps(1);
    event(&mut harness, button(from, true));
    for i in 1..=10 {
        event(&mut harness, egui::Event::PointerMoved(from + (to - from) * (i as f32 / 10.0)));
        harness.run_steps(1);
    }
    harness.render().expect("renders").save(dir.join("poly_box_drag.png")).unwrap();
    event(&mut harness, button(to, false));
    harness.run_steps(2);
    event(&mut harness, egui::Event::ModifiersChanged(egui::Modifiers::NONE));
    harness.run_steps(1);
    assert_ne!(harness.state().test_undo_serial(), before, "Shift+drag box filled faces");

    // The texture set list: hiding a set takes it out of picking.
    harness.get_by_label_contains("Texture Set").click();
    harness.run_steps(2);
    let all = harness.state().test_pickable();
    assert!(harness.query_all_by_label_contains("Left").count() > 0, "the sets are listed");
    harness.get_by_label_contains("Right").click();
    harness.run_steps(1);
    assert_eq!(harness.state().test_set_state(), [(false, false), (false, true)], "the picked set flashes");
    harness.render().expect("renders").save(dir.join("texture_set_list.png")).unwrap();

    // A material dragged from the shelf lands in the set under the pointer,
    // not the edited one (Right): here, Left.
    assert_eq!(harness.state().test_active_set(), 1);
    let left = (0..40)
        .map(|i| egui::pos2(rect.left() + rect.width() * (i as f32 + 0.5) / 40.0, rect.center().y))
        .find(|p| harness.state_mut().test_set_at(*p) == Some(0))
        .expect("Left shows in the 3D view");
    let layers = |h: &egui_kittest::Harness<'_, PainterApp>| h.state().test_sets().iter().map(|s| s.2.len()).collect::<Vec<_>>();
    let before = layers(&harness);
    let plain = egui::Modifiers::NONE;
    let button = |pos, pressed| egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: plain };
    // Pressed on a tile (outside the view), dragged over the view, released.
    let tile = egui::pos2(rect.left() - 40.0, rect.top() + 10.0);
    event(&mut harness, egui::Event::PointerMoved(tile));
    harness.run_steps(1);
    event(&mut harness, button(tile, true));
    harness.run_steps(1);
    harness.state().test_drag_material(&ctx);
    for i in 1..=5 {
        event(&mut harness, egui::Event::PointerMoved(tile + (left - tile) * (i as f32 / 5.0)));
        harness.run_steps(1);
    }
    harness.render().expect("renders").save(dir.join("drop_on_set.png")).unwrap();
    event(&mut harness, button(left, false));
    harness.run_steps(3);
    assert_eq!(harness.state().test_active_set(), 0, "the set under the pointer is edited");
    let after = layers(&harness);
    assert_eq!(after[0], before[0] + 1, "the material went into Left");
    assert_eq!(after[1], before[1], "Right kept its layers");
    harness.state_mut().set_hidden_for_test(1, true);
    harness.run_steps(2);
    assert!(harness.state().test_pickable() < all, "a hidden set isn't pickable");
    harness.render().expect("renders").save(dir.join("texture_set_hidden.png")).unwrap();
}

/// Shelf items dragged onto the layer list and the fill layer's Images rows,
/// offscreen: a generator masks the row's layer, a smart material lands as a
/// folder beside it, a material as a fill, and a texture feeds the channel
/// row it's dropped on. Frames go to `target/bake_test/ui/drop_*.png`.
#[test]
#[ignore]
fn shelf_drops_on_layers_and_images() {
    use crate::app::{PainterApp, ShelfItem};
    use egui_kittest::kittest::{NodeT, Queryable};
    let mut setup = egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    setup.device_descriptor = Arc::new(|adapter| wgpu::DeviceDescriptor {
        label: Some("ui test device"),
        required_limits: adapter.limits(),
        ..Default::default()
    });
    let mut harness = egui_kittest::Harness::builder()
        .with_size([1600.0, 1000.0])
        .with_pixels_per_point(1.0)
        .with_accessibility_check(false)
        .wgpu_setup(egui_wgpu::WgpuSetup::CreateNew(setup))
        .build_eframe(|cc| PainterApp::new(cc).expect("app starts"));
    harness.run_steps(4);
    harness.state_mut().test_close_dialogs();
    harness.run_steps(2);
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test/ui");
    std::fs::create_dir_all(&dir).unwrap();
    let ctx = harness.ctx.clone();

    // A drag as the shelf starts one: pressed on a tile, moved, released on `to`.
    let drag = |h: &mut egui_kittest::Harness<'_, PainterApp>, item: ShelfItem, to: egui::Pos2, shot: &str| {
        let button = |pos, pressed| egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE };
        // An empty spot of the menu bar: no widget there starts a drag of its own.
        let tile = egui::pos2(700.0, 18.0);
        h.input_mut().events.push(egui::Event::PointerMoved(tile));
        h.run_steps(1);
        h.input_mut().events.push(button(tile, true));
        h.run_steps(1);
        h.state().test_drag(&ctx, item);
        for i in 1..=5 {
            h.input_mut().events.push(egui::Event::PointerMoved(tile + (to - tile) * (i as f32 / 5.0)));
            h.run_steps(1);
        }
        h.render().expect("renders").save(dir.join(shot)).unwrap();
        h.input_mut().events.push(button(to, false));
        h.run_steps(3);
    };
    let row = |h: &egui_kittest::Harness<'_, PainterApp>, name: &str| h.get_by_label(name).rect();
    // The dock tab, by its icon and title (not the shelf hint naming Layers).
    harness
        .query_all_by_label_contains("Layers")
        .find(|n| n.accesskit_node().label().is_some_and(|l| l.ends_with(" Layers")))
        .expect("the Layers tab")
        .click();
    harness.run_steps(2);
    let layers = harness.state().test_layers();
    let first = layers.last().expect("a layer").0.clone();

    // A generator on a row masks that layer.
    let r = row(&harness, &first);
    drag(&mut harness, ShelfItem::Generator(crate::engine::GenKind::EdgeWear), r.center(), "drop_generator.png");
    let top = harness.state().test_layers().len() - 1;
    assert!(harness.state().test_layer_sources()[top].0, "the generator masks the row's layer");

    // A smart material above a row lands as a folder.
    let n = harness.state().test_layers().len();
    let r = row(&harness, &first);
    drag(&mut harness, ShelfItem::Smart(0), egui::pos2(r.center().x, r.top() + 4.0), "drop_smart.png");
    let after = harness.state().test_layers();
    assert!(after.len() > n, "the smart material's layers were added");
    assert!(after.iter().any(|l| l.1 == "folder"), "as a folder");

    // A material dropped on the top row becomes a fill (selected), and a
    // texture on its Color row feeds the color.
    let name = after.last().unwrap().0.clone();
    let r = row(&harness, &name);
    drag(&mut harness, ShelfItem::Material(0), egui::pos2(r.center().x, r.top() + 4.0), "drop_material.png");
    let top = harness.state().test_layers().len() - 1;
    assert_eq!(harness.state().test_layers()[top].1, "fill", "the material is a fill on top");
    harness.run_steps(2);
    let load = harness.query_all_by_label_contains("Load image").next().expect("the Images rows show").rect();
    drag(&mut harness, ShelfItem::Texture(0), load.center(), "drop_texture.png");
    assert_eq!(harness.state().test_layer_sources()[top].1, [0], "the texture feeds the color");
    // The fill's channels include normal and specular, with image rows of their own.
    assert!(harness.query_by_label("specular").is_some() && harness.query_by_label("normal").is_some(), "channel toggles");
    assert_eq!(harness.query_all_by_label_contains("Load image").count(), 7, "one image row per channel without an image");
    harness.render().expect("renders").save(dir.join("drop_fill_channels.png")).unwrap();
}

/// The bake window's cage tools, offscreen: the probe colors the cage,
/// Auto-fix fits it so nothing misses, and the cage brush pushes it out under
/// a drag over the model. Frames go to `target/bake_test/ui/cage_*.png`.
#[test]
#[ignore]
fn cage_paint_and_auto_fix() {
    use crate::app::PainterApp;
    use egui_kittest::kittest::Queryable;
    let mut setup = egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    setup.device_descriptor = Arc::new(|adapter| wgpu::DeviceDescriptor {
        label: Some("ui test device"),
        required_limits: adapter.limits(),
        ..Default::default()
    });
    let mut harness = egui_kittest::Harness::builder()
        .with_size([1600.0, 1000.0])
        .with_pixels_per_point(1.0)
        .with_accessibility_check(false)
        .wgpu_setup(egui_wgpu::WgpuSetup::CreateNew(setup))
        .build_eframe(|cc| PainterApp::new(cc).expect("app starts"));
    harness.run_steps(4);
    harness.state_mut().test_close_dialogs();
    // A sphere, and a high poly bulging past the front distance where x > 0.5.
    let low = crate::mesh::Primitive::Sphere.build();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test/ui");
    std::fs::create_dir_all(&dir).unwrap();
    let obj = dir.join("cage_bulge_high.obj");
    let mut s = String::new();
    for v in &low.vertices {
        let p = glam::Vec3::from_array(v.pos);
        let p = p * if p.x > 0.5 { 1.3 } else { 1.03 } / low.scale + low.center;
        s.push_str(&format!("v {} {} {}\n", p.x, p.y, p.z));
    }
    for t in low.indices.chunks_exact(3) {
        s.push_str(&format!("f {} {} {}\n", t[0] + 1, t[1] + 1, t[2] + 1));
    }
    std::fs::write(&obj, s).unwrap();
    let ctx = harness.ctx.clone();
    harness.state_mut().test_load_mesh(low, &ctx);
    harness.state_mut().test_cage_setup(obj, 0.08, false);
    // The probe runs on a worker thread; it repaints when it lands.
    let mut steps = 0;
    while harness.state().test_cage_missed().is_none() && steps < 20_000 {
        harness.run_steps(1);
        steps += 1;
    }
    let before = harness.state().test_cage_missed().expect("probed");
    harness.render().expect("renders").save(dir.join("cage_probe.png")).unwrap();
    println!("  cage: {:.1}% missed before the fix", before * 100.0);
    assert!(before > 0.05, "the bulge misses");

    harness.get_by_label_contains("Auto-fix").click();
    harness.run_steps(3);
    let after = harness.state().test_cage_missed().expect("still probed");
    harness.render().expect("renders").save(dir.join("cage_fixed.png")).unwrap();
    println!("  cage: {:.1}% missed after Auto-fix, multipliers {:?}", after * 100.0, harness.state().test_cage_paint_range());
    assert!(after < before * 0.2, "Auto-fix reaches the bulge ({after})");

    // Reset, then paint the cage out over the middle of the model.
    harness.get_by_label("Reset").click();
    harness.run_steps(2);
    assert!(harness.state().test_cage_paint_range().is_none(), "reset");
    harness.get_by_label_contains("Paint cage").click();
    harness.run_steps(2);
    let rect = harness.state().test_view_rect().expect("the 3D view");
    let at = rect.center();
    let event = |h: &mut egui_kittest::Harness<'_, PainterApp>, e: egui::Event| h.input_mut().events.push(e);
    let button = |pos, pressed| egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: Default::default() };
    event(&mut harness, egui::Event::PointerMoved(at));
    harness.run_steps(1);
    event(&mut harness, button(at, true));
    for i in 0..20 {
        event(&mut harness, egui::Event::PointerMoved(at + egui::vec2(i as f32, 0.0)));
        harness.run_steps(1);
    }
    harness.render().expect("renders").save(dir.join("cage_painting.png")).unwrap();
    event(&mut harness, button(at + egui::vec2(20.0, 0.0), false));
    harness.run_steps(2);
    harness.render().expect("renders").save(dir.join("cage_painted.png")).unwrap();
    let (_, hi) = harness.state().test_cage_paint_range().expect("painted");
    println!("  cage brush: most raised multiplier {hi:.2}");
    assert!(hi > 1.05, "dragging pushed the cage out");

    // Shift+drag over the same spot puts the cage back to the front distance.
    let shift = egui::Modifiers::SHIFT;
    let shift_button = |pos, pressed| egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: shift };
    event(&mut harness, egui::Event::ModifiersChanged(shift));
    event(&mut harness, egui::Event::PointerMoved(at));
    harness.run_steps(1);
    event(&mut harness, shift_button(at, true));
    for i in 0..60 {
        event(&mut harness, egui::Event::ModifiersChanged(shift));
        event(&mut harness, egui::Event::PointerMoved(at + egui::vec2((i % 20) as f32, 0.0)));
        harness.run_steps(1);
    }
    event(&mut harness, shift_button(at, false));
    harness.run_steps(2);
    event(&mut harness, egui::Event::ModifiersChanged(egui::Modifiers::NONE));
    let (_, hi_after) = harness.state().test_cage_paint_range().expect("still painted");
    println!("  cage brush: most raised multiplier after Shift+drag {hi_after:.2}");
    assert!(hi_after < 1.0 + (hi - 1.0) * 0.3, "Shift+drag puts the cage back ({hi:.2} -> {hi_after:.2})");
}

/// Opacity: a fill at opacity 0 makes its texture set see-through (the view
/// shows what a hidden set shows: the backdrop), a brush stroke on the
/// opacity channel paints the opaque set see-through along the stroke, and
/// exports carry opacity as the base color's alpha. Frames go to
/// `target/bake_test/transparency_*.png`.
#[test]
#[ignore]
fn transparency_paints_shows_and_exports() {
    use crate::app::{BrushProjection, resolve_stamps};
    use crate::engine::{BrushPaint, CHANNELS, FillParams, MatValues, OPACITY, StampUniform};
    use crate::undo::{StrokeTarget, UndoStack};
    use crate::viewport::SetLook;
    use glam::Vec2;

    let mesh = Arc::new(crate::mesh::set_tests::split_plane());
    let bvh = crate::bvh::Bvh::from_mesh(&mesh);
    let rs = render_state();
    let gpu_mesh = GpuMesh::new(&rs.device, &mesh);
    let res = 256;
    let mut left = Engine::new(&rs, &gpu_mesh, (mesh.bounds_min, mesh.bounds_max), res);
    left.set_triangles(mesh.sets[0].index_range(), &gpu_mesh);
    let mut right = left.sibling(&gpu_mesh, mesh.sets[1].index_range(), res);
    let mut undo = UndoStack::default();
    // Left: opaque red, with a paint layer on top. Right: blue at opacity 0.
    let red = MatValues { color: [0.9, 0.1, 0.05], ..Default::default() };
    left.add_fill_layer(0, None, "Red".into(), FillParams { mat: red, ..Default::default() }, &mut undo);
    left.add_paint_layer(1, None, None, &mut undo);
    let mut glass = MatValues { color: [0.05, 0.1, 0.9], opacity: 0.0, ..Default::default() };
    glass.enabled[OPACITY] = true;
    right.add_fill_layer(0, None, "Glass".into(), FillParams { mat: glass, ..Default::default() }, &mut undo);
    assert!(right.uses_opacity() && !left.uses_opacity());
    left.composite();
    right.composite();
    // The final color's alpha is the opacity.
    let alpha = |e: &Engine| read_texture(&rs.device, &rs.queue, e.final_tex(0)).chunks(4).map(|p| p[3]).collect::<Vec<u8>>();
    assert!(alpha(&right).iter().all(|&a| a == 0), "the glass fill is see-through");
    assert!(alpha(&left).iter().all(|&a| a == 255), "the red fill is opaque");

    let mut views = ViewRenderer::new(&rs);
    let env = crate::environment::EnvBaker::new(&rs.device)
        .upload(&rs, &crate::environment::decode(&crate::environment::EnvSource::Builtin(0)).unwrap());
    let cam = OrbitCam { pitch: 1.5, dist: 3.2, ..Default::default() };
    let size = [512u32, 512u32];
    let settings = ViewSettings::default();
    let frame = |views: &mut ViewRenderer, left: &Engine, right: &Engine, looks: Vec<SetLook>| {
        views.set_looks = looks;
        views.render_3d(&rs, size, &[left, right], &env, &gpu_mesh, &cam, &settings, None);
        read_texture(&rs.device, &rs.queue, views.view3d.image())
    };
    let see_through = frame(&mut views, &left, &right, Vec::new());
    let hidden = frame(&mut views, &left, &right, vec![SetLook::default(), SetLook { hidden: true, highlight: 0.0 }]);
    let diff = see_through.iter().zip(&hidden).map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs()).max().unwrap_or(0);
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test");
    std::fs::create_dir_all(&dir).unwrap();
    image::save_buffer(dir.join("transparency_glass.png"), &see_through, size[0], size[1], image::ExtendedColorType::Rgba8).unwrap();
    println!("  see-through set vs hidden set: largest difference {diff}");
    assert!(diff <= 2, "opacity 0 shows what's behind, as if hidden ({diff})");

    // A stroke on the left set's opacity channel only: see-through along it.
    let mut only_opacity = [false; CHANNELS];
    only_opacity[OPACITY] = true;
    assert!(left.begin_stroke(1, StrokeTarget::Content, only_opacity, false));
    let radius = 30.0;
    let raw: Vec<([f32; 4], f32)> = (0..40).map(|i| ([60.0 + i as f32 * 4.0, 256.0, 1.0, 0.0], 1.0)).collect();
    let fsize = Vec2::new(size[0] as f32, size[1] as f32);
    let stamps = resolve_stamps(&mesh, &bvh, &cam, fsize, BrushProjection::Screen, 1, radius, &raw);
    let su = StampUniform {
        view_proj: cam.view_proj(fsize.x / fsize.y).to_cols_array_2d(),
        cam_pos: cam.eye().extend(1.0).to_array(),
        view: [fsize.x, fsize.y, 0.0, 0.0],
        view2d: [0.0; 4],
        brush: [radius, 0.6, 1.0, 0.0],
        clip: [crate::viewport::NEAR, crate::viewport::FAR, 0.0, 0.0],
        proj: [BrushProjection::Screen as u32 as f32, 0.0, 85f32.to_radians().cos(), 0.0],
    };
    let mut clear = MatValues { opacity: 0.0, ..Default::default() };
    clear.enabled = only_opacity;
    let paint = BrushPaint { mat: clear, mask_value: 1.0, opacity: 1.0, erase: false, grain: [0.0; 4] };
    left.stroke_step(&stamps, su, views.view3d.depth_view(), &paint, None);
    left.end_stroke();
    left.composite();
    let a = alpha(&left);
    let cleared = a.iter().filter(|&&v| v < 20).count();
    println!("  stroke on opacity: {cleared} see-through texels");
    assert!(cleared > 500, "the stroke made the red see-through ({cleared})");
    assert!(a.iter().filter(|&&v| v > 235).count() > cleared, "the rest stays opaque");
    // The color under the stroke is untouched: only opacity was painted.
    let color = read_texture(&rs.device, &rs.queue, left.final_tex(0));
    let i = a.iter().position(|&v| v < 20).unwrap();
    assert!(color[i * 4] > 200 && color[i * 4 + 2] < 60, "still red under the stroke");
    let stroked = frame(&mut views, &left, &right, Vec::new());
    image::save_buffer(dir.join("transparency_stroke.png"), &stroked, size[0], size[1], image::ExtendedColorType::Rgba8).unwrap();
    let reds = |px: &[u8]| px.chunks(4).filter(|p| p[0] as i32 > p[1] as i32 + 80 && p[0] as i32 > p[2] as i32 + 80).count();
    println!("  red pixels: {} before the stroke, {} after", reds(&see_through), reds(&stroked));
    assert!(reds(&stroked) + 2000 < reds(&see_through), "the backdrop shows through the stroke");

    // Export: base color as RGBA with the opacity in alpha (an opaque set stays RGB).
    use crate::export::{self, Format, Padding, Settings};
    let folder = dir.join("transparency_export");
    let _ = std::fs::remove_dir_all(&folder);
    let glb = export::builtin_templates().into_iter().find(|t| t.name.starts_with("glTF")).unwrap();
    for (name, engine, want) in [("Glass", &right, 4), ("Red", &left, 4)] {
        let snap = export::snapshot(engine, &rs.device, &rs.queue, mesh.clone());
        let settings = Settings {
            folder: folder.clone(),
            mesh_name: "Plane".into(),
            project_name: "Test".into(),
            set_name: name.into(),
            set_count: 2,
            format: Format::Png,
            depth: export::Depth::U8,
            size: None,
            padding: Padding::Infinite,
            normal_strength: 1.0,
            use_baked_normal: false,
        };
        let files = export::export(&snap, &glb, &settings).expect("exports");
        let base = files.iter().find(|f| f.to_string_lossy().contains("baseColor")).expect("a base color file");
        let img = image::open(base).unwrap();
        assert_eq!(img.color().channel_count(), want, "{}", base.display());
        let min_alpha = img.to_rgba8().pixels().map(|p| p[3]).min().unwrap();
        println!("  {}: {} channels, least alpha {min_alpha}", base.display(), img.color().channel_count());
    }
    let mut opaque = Engine::new(&rs, &gpu_mesh, (mesh.bounds_min, mesh.bounds_max), res);
    opaque.add_fill_layer(0, None, "Red".into(), FillParams { mat: red, ..Default::default() }, &mut undo);
    opaque.composite();
    let snap = export::snapshot(&opaque, &rs.device, &rs.queue, mesh.clone());
    let settings = Settings {
        folder: folder.clone(),
        mesh_name: "Opaque".into(),
        project_name: "Test".into(),
        set_name: "Opaque".into(),
        set_count: 1,
        format: Format::Png,
        depth: export::Depth::U8,
        size: None,
        padding: Padding::Infinite,
        normal_strength: 1.0,
        use_baked_normal: false,
    };
    let files = export::export(&snap, &glb, &settings).expect("exports");
    let base = files.iter().find(|f| f.to_string_lossy().contains("baseColor")).unwrap();
    assert_eq!(image::open(base).unwrap().color().channel_count(), 3, "an opaque set's base color stays RGB");
}

/// A smart material as a graph (`graph::smart`) looks the same as the
/// folder of fill layers with smart masks it was: same masks, variation and
/// blending, compared texel by texel. Side-by-side pictures go to
/// `target/bake_test/smart_graph/`.
#[test]
#[ignore]
fn smart_material_graphs_match_their_layers() {
    use crate::engine::{CHANNELS, FillParams};
    use crate::graph::{self, Library, Runtime};
    use crate::undo::UndoStack;

    let rs = render_state();
    let mesh = crate::mesh::Primitive::RoundedCube.build();
    let gpu_mesh = GpuMesh::new(&rs.device, &mesh);
    let res = 512;
    let mut engine = Engine::new(&rs, &gpu_mesh, (mesh.bounds_min, mesh.bounds_max), res);
    let (maps, report, _) = run_bake(&rs.device, &rs.queue, &mesh, &gpu_mesh, &BakeSettings::default(), res);
    engine.set_maps(maps, report);
    let lib = Library::load(None);
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test/smart_graph");
    std::fs::create_dir_all(&dir).unwrap();
    let read = |e: &Engine| [0, 1, 2, 3].map(|c| read_texture(&rs.device, &rs.queue, e.final_tex(c)));
    // Texels on the mesh (the rest is padding either way).
    let pos = read_texture(&rs.device, &rs.queue, engine.surface_maps().0);
    let covered: Vec<bool> = pos.chunks(8).map(|p| p[6] != 0 || p[7] != 0).collect();

    let mut names: Vec<&str> = vec!["Painted Steel Worn", "Oak Aged", "Rusty Iron", "Brass Aged", "Concrete Weathered"];
    let scanned = presets::SMART_MATERIALS.iter().find(|s| s.is_scanned() && crate::environment::asset_dir("pbr").is_some());
    names.extend(scanned.map(|s| s.name));
    for name in names {
        let smart = presets::SMART_MATERIALS.iter().find(|s| s.name == name).unwrap();
        let mut undo = UndoStack::default();
        engine.layers.clear();
        let folder = crate::app::add_smart_material(&mut engine, &mut undo, smart, 0, None);
        engine.composite();
        let layers = read(&engine);

        let mut g = graph::smart::graph(smart, &lib);
        g.res = res;
        let mut rt = Runtime::new(&rs.device, &rs.queue);
        let maps = |m: usize| engine.mesh_map(m);
        let surface = engine.graph_surface();
        rt.evaluate(&rs.device, &rs.queue, &g, &lib, &maps, Some(&surface), engine.maps_version, true);
        rt.wait_images(&rs.device, &rs.queue);
        rt.evaluate(&rs.device, &rs.queue, &g, &lib, &maps, Some(&surface), engine.maps_version, true);
        assert!(rt.errors.is_empty(), "{name}: {:?}", rt.errors);
        let outs = g.outputs(&lib).map(|o| o.and_then(|n| rt.output(n).cloned()));
        let chans: [Option<std::sync::Arc<crate::gpu::Tex>>; CHANNELS] = graph::channel_outputs(&outs);
        let mut fill = FillParams { graph: Some(1), ..FillParams::default() };
        fill.mat.enabled = chans.clone().map(|o| o.is_some());
        engine.layers[folder].visible = false;
        let top = engine.layers.len();
        let idx = engine.add_fill_layer(top, None, "Graph".into(), fill, &mut undo);
        engine.set_graph_images(idx, chans, name);
        engine.composite();
        let graph = read(&engine);

        // Color: layers left, graph right.
        let mut sheet = image::RgbaImage::new(res * 2, res);
        for (k, src) in [&layers[0], &graph[0]].into_iter().enumerate() {
            for (i, px) in src.chunks(4).enumerate() {
                let (x, y) = ((i as u32 % res) + k as u32 * res, i as u32 / res);
                sheet.put_pixel(x, y, image::Rgba([px[0], px[1], px[2], 255]));
            }
        }
        sheet.save(dir.join(format!("{}.png", name.replace(' ', "_")))).unwrap();
        let mut report = Vec::new();
        for (c, cname) in [(0, "color"), (1, "roughness"), (2, "metallic")] {
            let (a, b) = (&layers[c], &graph[c]);
            let step = if c == 0 { 4 } else { 1 };
            let mut diffs: Vec<i32> = Vec::new();
            for (i, _) in covered.iter().enumerate().filter(|(_, c)| **c) {
                let k = i * step;
                let d = (0..if c == 0 { 3 } else { 1 }).map(|j| (a[k + j] as i32 - b[k + j] as i32).abs()).max().unwrap();
                diffs.push(d);
            }
            diffs.sort_unstable();
            let mean = diffs.iter().sum::<i32>() as f32 / diffs.len().max(1) as f32;
            let p99 = diffs[(diffs.len() as f32 * 0.99) as usize];
            report.push(format!("{cname} mean {mean:.2} p99 {p99}"));
            assert!(mean < 2.5 && p99 < 24, "{name} {cname}: layers and graph differ (mean {mean:.2}, 99th percentile {p99})");
        }
        println!("  {name}: {}", report.join(", "));
    }
}

/// The skin, organic and eye smart materials, each as the graph layer the
/// shelf adds, on a sphere seen from its front (+Z): a contact sheet in
/// `target/bake_test/smart_graph/organic.png` (names in the test output,
/// left to right, top to bottom).
#[test]
#[ignore]
fn organic_smart_materials_sheet() {
    use crate::engine::{CHANNELS, FillParams};
    use crate::graph::{self, Library, Runtime};
    use crate::undo::UndoStack;

    let rs = render_state();
    let mesh = crate::mesh::Primitive::Sphere.build();
    let gpu_mesh = GpuMesh::new(&rs.device, &mesh);
    let res = 1024;
    let mut engine = Engine::new(&rs, &gpu_mesh, (mesh.bounds_min, mesh.bounds_max), res);
    let (maps, report, _) = run_bake(&rs.device, &rs.queue, &mesh, &gpu_mesh, &BakeSettings::default(), res);
    engine.set_maps(maps, report);
    let lib = Library::load(None);
    let names = [
        "Skin Realistic",
        "Skin Dark",
        "Skin Stylized",
        "Flesh Raw",
        "Reptile Scales",
        "Fungus Spotted",
        "Insect Chitin",
        "Eyeball Blue",
        "Eyeball Brown",
    ];
    let mut views = ViewRenderer::new(&rs);
    let env = crate::environment::EnvBaker::new(&rs.device)
        .upload(&rs, &crate::environment::decode(&crate::environment::EnvSource::Builtin(0)).unwrap());
    let cam = OrbitCam { yaw: 0.35, pitch: 0.2, dist: 2.9, ..Default::default() };
    let tile = 320u32;
    let cols = 3u32;
    let mut sheet = image::RgbaImage::new(tile * cols, tile * names.len().div_ceil(cols as usize) as u32);
    for (k, name) in names.iter().enumerate() {
        let smart = presets::SMART_MATERIALS.iter().find(|s| s.name == *name).unwrap_or_else(|| panic!("no smart material {name}"));
        assert_eq!(smart.kind(), presets::MaterialType::Organic, "{name} sorts under skin & organic");
        let mut g = graph::smart::graph(smart, &lib);
        g.res = res;
        let mut rt = Runtime::new(&rs.device, &rs.queue);
        let maps = |m: usize| engine.mesh_map(m);
        let surface = engine.graph_surface();
        rt.evaluate(&rs.device, &rs.queue, &g, &lib, &maps, Some(&surface), engine.maps_version, true);
        assert!(rt.errors.is_empty(), "{name}: {:?}", rt.errors);
        let outs = g.outputs(&lib).map(|o| o.and_then(|n| rt.output(n).cloned()));
        let chans: [Option<std::sync::Arc<crate::gpu::Tex>>; CHANNELS] = graph::channel_outputs(&outs);
        let mut fill = FillParams { graph: Some(1), ..FillParams::default() };
        fill.mat.enabled = chans.clone().map(|o| o.is_some());
        engine.layers.clear();
        let idx = engine.add_fill_layer(0, None, "Graph".into(), fill, &mut UndoStack::default());
        engine.set_graph_images(idx, chans, name);
        engine.composite();
        let settings = ViewSettings::default();
        views.render_3d(&rs, [tile, tile], &[&engine], &env, &gpu_mesh, &cam, &settings, None);
        let px = read_texture(&rs.device, &rs.queue, views.view3d.image());
        let (x0, y0) = ((k as u32 % cols) * tile, (k as u32 / cols) * tile);
        for (i, p) in px.chunks(4).enumerate() {
            sheet.put_pixel(x0 + i as u32 % tile, y0 + i as u32 / tile, image::Rgba([p[0], p[1], p[2], 255]));
        }
        println!("  {k}: {name}, {} nodes, {} parameters", g.nodes.len(), g.params.len());
    }
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test/smart_graph");
    std::fs::create_dir_all(&dir).unwrap();
    sheet.save(dir.join("organic.png")).unwrap();
}

/// Normal and specular channels: the final normal is flat and specular 0.5
/// until a layer sets them; a fill's normal-map image comes out as
/// authored in tangent space whether it's laid by UV or tri-planar (the
/// tangent map turns world slopes back); layers add their normal detail;
/// specular raises the reflection of non-metals in the view; exports carry
/// the painted normal and write a specular map only when one is set.
/// Frames go to `target/bake_test/normal_spec_*.png`.
#[test]
#[ignore]
fn normal_and_specular_channels() {
    use crate::engine::{FINAL_NORMAL, FINAL_SPECULAR, FillParams, MatValues, NORMAL, Projection, SPECULAR};
    use crate::undo::UndoStack;

    let mesh = Arc::new(crate::mesh::Primitive::Plane.build());
    let rs = render_state();
    let gpu_mesh = GpuMesh::new(&rs.device, &mesh);
    let res = 256;
    let mut engine = Engine::new(&rs, &gpu_mesh, (mesh.bounds_min, mesh.bounds_max), res);
    let mut undo = UndoStack::default();
    let grey = MatValues { color: [0.5, 0.5, 0.5], rough: 0.35, ..Default::default() };
    engine.add_fill_layer(0, None, "Base".into(), FillParams { mat: grey, ..Default::default() }, &mut undo);
    engine.composite();
    let texel = |e: &Engine, ch: usize| {
        let px = read_texture(&rs.device, &rs.queue, e.final_tex(ch));
        let bpp = px.len() / (res * res) as usize;
        let i = ((res / 2) * res + res / 2) as usize * bpp;
        px[i..i + bpp].to_vec()
    };
    let decode = |t: &[u8]| Vec3::new(t[0] as f32, t[1] as f32, t[2] as f32) / 127.5 - Vec3::ONE;
    assert!(decode(&texel(&engine, FINAL_NORMAL)).distance(Vec3::Z) < 0.02, "flat until a layer sets normals");
    assert!((texel(&engine, FINAL_SPECULAR)[0] as i32 - 128).abs() <= 1, "specular 0.5 until a layer sets it");
    assert!(!engine.uses_channel(NORMAL) && !engine.uses_channel(SPECULAR));

    // A normal-map image tilted one way, everywhere.
    let tilted = |n: Vec3| {
        let n = n.normalize();
        let c = ((n * 0.5 + Vec3::splat(0.5)) * 255.0).round();
        let img = image::RgbImage::from_pixel(16, 16, image::Rgb([c.x as u8, c.y as u8, c.z as u8]));
        let mut bytes = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Png).unwrap();
        bytes
    };
    let layer_with = |engine: &mut Engine, undo: &mut UndoStack, n: Vec3, projection: Projection| {
        let mut mat = MatValues { enabled: [false; crate::engine::CHANNELS], ..Default::default() };
        mat.enabled[NORMAL] = true;
        let at = engine.layers.len();
        let idx = engine.add_fill_layer(at, None, "Normal".into(), FillParams { mat, projection, ..Default::default() }, undo);
        let img = engine.decode_channel_image(NORMAL, "tilt.png".into(), Arc::new(tilted(n))).unwrap();
        engine.set_fill_image(idx, NORMAL, Some(img), undo);
        engine.composite();
        idx
    };
    for (what, tilt) in [("toward +u", Vec3::new(0.5, 0.0, 0.866)), ("toward +v (up the image)", Vec3::new(0.0, 0.5, 0.866))] {
        let idx = layer_with(&mut engine, &mut undo, tilt, Projection::Uv);
        let uv = decode(&texel(&engine, FINAL_NORMAL));
        engine.delete_layer(idx, &mut undo);
        let idx = layer_with(&mut engine, &mut undo, tilt, Projection::Triplanar);
        let tri = decode(&texel(&engine, FINAL_NORMAL));
        engine.delete_layer(idx, &mut undo);
        println!("  tilt {what}: by UV {uv:.3}, tri-planar {tri:.3}");
        assert!(uv.distance(tilt.normalize()) < 0.03, "a UV normal image comes out as authored ({what}: {uv})");
        assert!(tri.distance(uv) < 0.05, "tri-planar matches UV on a plane laid out along its axes ({what}: {tri} vs {uv})");
    }
    assert!(!engine.uses_channel(NORMAL), "deleted layers leave no normals");

    // Two layers add their detail.
    let one = layer_with(&mut engine, &mut undo, Vec3::new(0.3, 0.0, 0.95), Projection::Uv);
    let single = decode(&texel(&engine, FINAL_NORMAL));
    layer_with(&mut engine, &mut undo, Vec3::new(0.3, 0.0, 0.95), Projection::Uv);
    let both = decode(&texel(&engine, FINAL_NORMAL));
    println!("  one layer x {:.3}, two layers x {:.3}", single.x, both.x);
    assert!(both.x > single.x + 0.1, "layers add normal detail");
    assert!(engine.uses_channel(NORMAL));
    let _ = one;

    // Specular: a fill sets it, and non-metals reflect more for it.
    let mut views = ViewRenderer::new(&rs);
    let env = crate::environment::EnvBaker::new(&rs.device)
        .upload(&rs, &crate::environment::decode(&crate::environment::EnvSource::Builtin(0)).unwrap());
    let cam = OrbitCam { pitch: 0.5, dist: 3.0, ..Default::default() };
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test");
    std::fs::create_dir_all(&dir).unwrap();
    let mut frame = |engine: &Engine, name: &str| {
        // A black backdrop: only the surface counts.
        let s = ViewSettings { background: crate::viewport::Background::Color, bg_color: [0.0; 3], ..ViewSettings::default() };
        views.render_3d(&rs, [400, 400], &[engine], &env, &gpu_mesh, &cam, &s, None);
        let px = read_texture(&rs.device, &rs.queue, views.view3d.image());
        image::save_buffer(dir.join(name), &px, 400, 400, image::ExtendedColorType::Rgba8).unwrap();
        px.chunks(4).map(|p| p[0] as u64 + p[1] as u64 + p[2] as u64).sum::<u64>()
    };
    let spec_layer = |engine: &mut Engine, undo: &mut UndoStack, v: f32| {
        // A dark, glossy non-metal: what it shows is mostly its reflection.
        let mut mat = MatValues { color: [0.02; 3], rough: 0.2, specular: v, ..Default::default() };
        mat.enabled[SPECULAR] = true;
        let at = engine.layers.len();
        let idx = engine.add_fill_layer(at, None, "Specular".into(), FillParams { mat, ..Default::default() }, undo);
        engine.composite();
        idx
    };
    let none = spec_layer(&mut engine, &mut undo, 0.0);
    assert!(texel(&engine, FINAL_SPECULAR)[0] <= 1, "a fill sets specular");
    let dull = frame(&engine, "normal_spec_dull.png");
    engine.delete_layer(none, &mut undo);
    spec_layer(&mut engine, &mut undo, 1.0);
    assert!(texel(&engine, FINAL_SPECULAR)[0] >= 254);
    let shiny = frame(&engine, "normal_spec_shiny.png");
    println!("  view brightness: specular 0 {dull}, specular 1 {shiny}");
    assert!(shiny > dull * 3 / 2, "more specular reflects more");

    // Export: the normal map carries the painted normal; a specular map only when set.
    use crate::export::{self, Format, Padding, Settings};
    let folder = dir.join("normal_spec_export");
    let _ = std::fs::remove_dir_all(&folder);
    let template = export::builtin_templates().into_iter().find(|t| t.name.starts_with("PBR Metallic Roughness")).unwrap();
    let settings = |name: &str| Settings {
        folder: folder.clone(),
        mesh_name: name.into(),
        project_name: "Test".into(),
        set_name: name.into(),
        set_count: 1,
        format: Format::Png,
        depth: export::Depth::U8,
        size: None,
        padding: Padding::Infinite,
        normal_strength: 1.0,
        use_baked_normal: false,
    };
    let files = export::export(&export::snapshot(&engine, &rs.device, &rs.queue, mesh.clone()), &template, &settings("Set")).unwrap();
    let normal = files.iter().find(|f| f.to_string_lossy().contains("Normal")).expect("a normal map");
    let n = image::open(normal).unwrap().to_rgb8();
    let c = n.get_pixel(res / 2, res / 2);
    println!("  exported normal at the center: {c:?}");
    assert!(c[0] > 160, "the exported normal leans the painted way");
    let spec = files.iter().find(|f| f.to_string_lossy().contains("Specular")).expect("a specular map, since one is set");
    assert!(image::open(spec).unwrap().to_luma8().get_pixel(res / 2, res / 2)[0] >= 250);

    let mut plain = Engine::new(&rs, &gpu_mesh, (mesh.bounds_min, mesh.bounds_max), res);
    plain.add_fill_layer(0, None, "Base".into(), FillParams { mat: grey, ..Default::default() }, &mut undo);
    plain.composite();
    let files = export::export(&export::snapshot(&plain, &rs.device, &rs.queue, mesh.clone()), &template, &settings("Plain")).unwrap();
    assert!(!files.iter().any(|f| f.to_string_lossy().contains("Specular")), "no specular map when none is set");
}

/// The shelf in a narrow window: categories as icons, a header that packs
/// tight (no hint, - / + for the size) without overlapping, and small
/// thumbnails without names. Frames go to `target/bake_test/ui/shelf_*.png`.
#[test]
#[ignore]
fn shelf_compact_when_narrow() {
    use crate::app::PainterApp;
    use egui_kittest::kittest::Queryable;
    let build = |w: f32| {
        let mut setup = egui_wgpu::WgpuSetupCreateNew::without_display_handle();
        setup.device_descriptor = Arc::new(|adapter| wgpu::DeviceDescriptor {
            label: Some("ui test device"),
            required_limits: adapter.limits(),
            ..Default::default()
        });
        let mut h = egui_kittest::Harness::builder()
            .with_size([w, 800.0])
            .with_pixels_per_point(1.0)
            .with_accessibility_check(false)
            .wgpu_setup(egui_wgpu::WgpuSetup::CreateNew(setup))
            .build_eframe(|cc| PainterApp::new(cc).expect("app starts"));
        h.run_steps(4);
        h.state_mut().test_close_dialogs();
        h.run_steps(2);
        h
    };
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test/ui");
    std::fs::create_dir_all(&dir).unwrap();

    let mut wide = build(1600.0);
    wide.render().expect("renders").save(dir.join("shelf_wide.png")).unwrap();
    assert!(wide.query_all_by_label_contains("Get more assets").count() > 0, "a wide shelf names the button");
    assert!(wide.query_by_label("+").is_none(), "and sizes with the slider");

    let mut narrow = build(1000.0);
    narrow.render().expect("renders").save(dir.join("shelf_narrow.png")).unwrap();
    assert!(narrow.query_by_label("+").is_some() && narrow.query_by_label("-").is_some(), "a narrow shelf sizes with - / +");
    for _ in 0..3 {
        narrow.get_by_label("-").click();
        narrow.run_steps(2);
    }
    narrow.render().expect("renders").save(dir.join("shelf_narrow_small.png")).unwrap();
}

/// Polygon fill padding in a tight UV layout: two islands a texel apart,
/// one filled with wide padding. The padding fills the empty gutter around
/// the filled island but never the texels of its neighbor.
#[test]
#[ignore]
fn poly_fill_padding_stays_off_other_islands() {
    use crate::engine::{BrushPaint, CHANNELS, FillParams, MatValues};
    use crate::undo::{StrokeTarget, UndoStack};
    use glam::Vec2;

    // Two quads apart in space; in UV, A spans u 0.10..0.495 and B
    // 0.505..0.90 (about a texel apart at 128²), both v 0.1..0.9.
    let quad = |x0: f32, x1: f32, u0: f32, u1: f32| {
        let p = [Vec3::new(x0, 0.0, -1.0), Vec3::new(x1, 0.0, -1.0), Vec3::new(x1, 0.0, 1.0), Vec3::new(x0, 0.0, 1.0)];
        let uv = [Vec2::new(u0, 0.1), Vec2::new(u1, 0.1), Vec2::new(u1, 0.9), Vec2::new(u0, 0.9)];
        (p, uv)
    };
    let (pa, ua) = quad(-1.0, -0.1, 0.10, 0.495);
    let (pb, ub) = quad(0.1, 1.0, 0.505, 0.90);
    let positions: Vec<Vec3> = pa.into_iter().chain(pb).collect();
    let uvs: Vec<Vec2> = ua.into_iter().chain(ub).collect();
    let indices = vec![0, 2, 1, 0, 3, 2, 4, 6, 5, 4, 7, 6];
    let mesh = Mesh::from_parts("Two islands".into(), positions, Some(vec![Vec3::Y; 8]), uvs, indices);
    let topo = crate::polyfill::Topology::build(&mesh);

    let rs = render_state();
    let gpu_mesh = GpuMesh::new(&rs.device, &mesh);
    let res = 128;
    let mut engine = Engine::new(&rs, &gpu_mesh, (mesh.bounds_min, mesh.bounds_max), res);
    let mut undo = UndoStack::default();
    let black = MatValues { color: [0.0; 3], ..Default::default() };
    engine.add_fill_layer(0, None, "Black".into(), FillParams { mat: black, ..Default::default() }, &mut undo);
    engine.add_paint_layer(1, None, None, &mut undo);
    let mut only = [false; CHANNELS];
    only[0] = true;
    assert!(engine.begin_stroke(1, StrokeTarget::Content, only, false));
    // Island A's triangles (the first two), padded 8 texels.
    let a: Vec<u32> = (0..topo.triangle_count() as u32).filter(|&t| topo.uv_centroid(t).x < 0.5).collect();
    assert_eq!(a.len(), 2);
    let geometry = topo.uv_geometry(&a, 8.0 / res as f32);
    assert!(!geometry.padding.is_empty(), "A's outline is padded");
    let red = MatValues { color: [1.0, 0.0, 0.0], enabled: only, ..Default::default() };
    engine.stroke_polys(&geometry, &BrushPaint { mat: red, mask_value: 1.0, opacity: 1.0, erase: false, grain: [0.0; 4] });
    engine.end_stroke();
    engine.composite();

    let color = read_texture(&rs.device, &rs.queue, engine.final_tex(0));
    let at = |u: f32, v: f32| {
        let (x, y) = ((u * res as f32) as usize, (v * res as f32) as usize);
        color[(y * res as usize + x) * 4]
    };
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test");
    std::fs::create_dir_all(&dir).unwrap();
    image::save_buffer(dir.join("poly_padding_islands.png"), &color, res, res, image::ExtendedColorType::Rgba8).unwrap();
    assert!(at(0.3, 0.5) > 200, "island A is filled");
    assert!(at(0.07, 0.5) > 200 && at(0.3, 0.06) > 200, "the empty gutter around A is padded");
    // Every texel of B, including the ones right at its edge facing A.
    let spill: Vec<f32> = (0..20).map(|i| 0.51 + i as f32 * 0.02).filter(|&u| u < 0.9 && at(u, 0.5) > 40).collect();
    assert!(spill.is_empty(), "padding spilled onto island B at u = {spill:?}");
}

/// Bloom around emission: with Bloom on at its default threshold, a plane
/// giving off pure red light (the hardest case for a luminance threshold:
/// 0.21 of white's) spreads a red glow onto the background around it, at
/// the default emission strength. Frames go to `target/bake_test/bloom_*.png`.
#[test]
#[ignore]
fn bloom_glows_around_red_emission() {
    use crate::engine::{CHANNELS, EMISSION, FillParams, MatValues};
    use crate::undo::UndoStack;

    let mesh = Arc::new(crate::mesh::Primitive::Plane.build());
    let rs = render_state();
    let gpu_mesh = GpuMesh::new(&rs.device, &mesh);
    let mut engine = Engine::new(&rs, &gpu_mesh, (mesh.bounds_min, mesh.bounds_max), 256);
    let mut undo = UndoStack::default();
    let mut enabled = [false; CHANNELS];
    enabled[0] = true;
    enabled[EMISSION] = true;
    let mat = MatValues { color: [0.05, 0.05, 0.05], emission: [1.0, 0.0, 0.0], enabled, ..Default::default() };
    engine.add_fill_layer(0, None, "Red light".into(), FillParams { mat, ..Default::default() }, &mut undo);
    engine.composite();
    assert!(engine.uses_emission());

    // The plane fills the middle of the frame; background all around.
    let cam = OrbitCam { pitch: 1.2, dist: 5.0, ..Default::default() };
    let size = [512u32, 512u32];
    let mut views = ViewRenderer::new(&rs);
    let env = crate::environment::EnvBaker::new(&rs.device)
        .upload(&rs, &crate::environment::decode(&crate::environment::EnvSource::Builtin(0)).unwrap());
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test");
    std::fs::create_dir_all(&dir).unwrap();
    let mut frame = |bloom: f32, emission: f32, name: &str| {
        let mut s = ViewSettings::default();
        s.post.bloom = bloom;
        s.emission = emission;
        views.render_3d(&rs, size, &[&engine], &env, &gpu_mesh, &cam, &s, None);
        let px = read_texture(&rs.device, &rs.queue, views.view3d.image());
        image::save_buffer(dir.join(name), &px, size[0], size[1], image::ExtendedColorType::Rgba8).unwrap();
        px
    };
    let strength = ViewSettings::default().emission;
    let plain = frame(0.0, strength, "bloom_off.png");
    // Background: pixels that aren't red without bloom.
    let redness = |p: &[u8]| p[0] as i32 - (p[1] as i32 + p[2] as i32) / 2;
    let background: Vec<usize> = plain.chunks(4).enumerate().filter(|(_, p)| redness(p) < 20).map(|(i, _)| i).collect();
    assert!(background.len() > 20_000 && background.len() < plain.len() / 4 - 20_000, "the plane and the background both show");
    let halo = |px: &[u8]| background.iter().filter(|&&i| redness(&px[i * 4..i * 4 + 4]) - redness(&plain[i * 4..i * 4 + 4]) > 12).count();
    for s in [1.0, 2.0, 4.0] {
        let px = frame(1.0, s, &format!("bloom_strength_{s}.png"));
        println!("  emission strength {s}: {} background pixels glow red", halo(&px));
    }
    let glow = frame(1.0, strength, "bloom_on.png");
    let n = halo(&glow);
    println!("  default strength {strength}: {n} background pixels glow red");
    assert!(n > 3000, "red emission blooms onto the background at the default settings ({n})");
}

/// Emission: a fill that emits fills the final emission texture, a stroke
/// on the emission channel paints only where it goes, the 3D view glows
/// there (and brighter with the view's emission strength), and exports write
/// an emission map only for a set that emits. Frames go to
/// `target/bake_test/emission_*.png`.
#[test]
#[ignore]
fn emission_paints_glows_and_exports() {
    use crate::app::{BrushProjection, resolve_stamps};
    use crate::engine::{BrushPaint, CHANNELS, EMISSION, FINAL_EMISSION, FillParams, MatValues, StampUniform};
    use crate::undo::{StrokeTarget, UndoStack};
    use glam::Vec2;

    let mesh = Arc::new(crate::mesh::Primitive::Plane.build());
    let bvh = crate::bvh::Bvh::from_mesh(&mesh);
    let rs = render_state();
    let gpu_mesh = GpuMesh::new(&rs.device, &mesh);
    let mut engine = Engine::new(&rs, &gpu_mesh, (mesh.bounds_min, mesh.bounds_max), 256);
    let mut undo = UndoStack::default();
    let dark = MatValues { color: [0.05, 0.05, 0.05], rough: 0.8, ..Default::default() };
    engine.add_fill_layer(0, None, "Dark".into(), FillParams { mat: dark, ..Default::default() }, &mut undo);
    engine.add_paint_layer(1, None, None, &mut undo);
    engine.composite();
    assert!(!engine.uses_emission());
    let emission = |e: &Engine| read_texture(&rs.device, &rs.queue, e.final_tex(FINAL_EMISSION));
    assert!(emission(&engine).chunks(4).all(|p| p[0] == 0 && p[1] == 0 && p[2] == 0), "nothing emits yet");

    // A stroke of green light across the middle.
    let mut only = [false; CHANNELS];
    only[EMISSION] = true;
    assert!(engine.begin_stroke(1, StrokeTarget::Content, only, false));
    let cam = OrbitCam { pitch: 1.5, dist: 3.2, ..Default::default() };
    let size = [512u32, 512u32];
    let fsize = Vec2::new(size[0] as f32, size[1] as f32);
    let mut views = ViewRenderer::new(&rs);
    let env = crate::environment::EnvBaker::new(&rs.device)
        .upload(&rs, &crate::environment::decode(&crate::environment::EnvSource::Builtin(0)).unwrap());
    let settings = ViewSettings::default();
    views.render_3d(&rs, size, &[&engine], &env, &gpu_mesh, &cam, &settings, None);
    let dark_frame = read_texture(&rs.device, &rs.queue, views.view3d.image());
    let radius = 30.0;
    let raw: Vec<([f32; 4], f32)> = (0..40).map(|i| ([100.0 + i as f32 * 8.0, 256.0, 1.0, 0.0], 1.0)).collect();
    let stamps = resolve_stamps(&mesh, &bvh, &cam, fsize, BrushProjection::Screen, 1, radius, &raw);
    let su = StampUniform {
        view_proj: cam.view_proj(fsize.x / fsize.y).to_cols_array_2d(),
        cam_pos: cam.eye().extend(1.0).to_array(),
        view: [fsize.x, fsize.y, 0.0, 0.0],
        view2d: [0.0; 4],
        brush: [radius, 0.6, 1.0, 0.0],
        clip: [crate::viewport::NEAR, crate::viewport::FAR, 0.0, 0.0],
        proj: [BrushProjection::Screen as u32 as f32, 0.0, 85f32.to_radians().cos(), 0.0],
    };
    let mut green = MatValues { emission: [0.1, 1.0, 0.2], ..Default::default() };
    green.enabled = only;
    let paint = BrushPaint { mat: green, mask_value: 1.0, opacity: 1.0, erase: false, grain: [0.0; 4] };
    engine.stroke_step(&stamps, su, views.view3d.depth_view(), &paint, None);
    engine.end_stroke();
    engine.composite();
    assert!(engine.uses_emission());
    let e = emission(&engine);
    let lit = e.chunks(4).filter(|p| p[1] > 200 && p[0] < 80).count();
    println!("  emission stroke: {lit} glowing texels of {}", e.len() / 4);
    assert!(lit > 500 && lit < e.len() / 4 / 2, "the stroke emits, the rest doesn't ({lit})");
    // Only emission was painted: the color under it is still dark.
    let color = read_texture(&rs.device, &rs.queue, engine.final_tex(0));
    let i = e.chunks(4).position(|p| p[1] > 200).unwrap();
    assert!(color[i * 4 + 1] < 40, "base color untouched under the stroke");

    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test");
    std::fs::create_dir_all(&dir).unwrap();
    let mut frame = |strength: f32| {
        let s = ViewSettings { emission: strength, ..ViewSettings::default() };
        views.render_3d(&rs, size, &[&engine], &env, &gpu_mesh, &cam, &s, None);
        read_texture(&rs.device, &rs.queue, views.view3d.image())
    };
    let glow = frame(1.0);
    let bright = frame(4.0);
    image::save_buffer(dir.join("emission_glow.png"), &glow, size[0], size[1], image::ExtendedColorType::Rgba8).unwrap();
    image::save_buffer(dir.join("emission_bright.png"), &bright, size[0], size[1], image::ExtendedColorType::Rgba8).unwrap();
    let green_px = |px: &[u8]| px.chunks(4).filter(|p| p[1] as i32 > p[0] as i32 + 60 && p[1] as i32 > p[2] as i32 + 40).count();
    let sum_g = |px: &[u8]| px.chunks(4).map(|p| p[1] as u64).sum::<u64>();
    println!("  green pixels: {} dark, {} glowing; green sum x1 {} x4 {}", green_px(&dark_frame), green_px(&glow), sum_g(&glow), sum_g(&bright));
    assert!(green_px(&glow) > green_px(&dark_frame) + 2000, "the stroke glows in the 3D view");
    assert!(sum_g(&bright) > sum_g(&glow), "a stronger emission setting glows brighter");

    // Export: an emission map for this set; none for a set that emits nothing.
    use crate::export::{self, Format, Padding, Settings};
    let folder = dir.join("emission_export");
    let _ = std::fs::remove_dir_all(&folder);
    let gltf = export::builtin_templates().into_iter().find(|t| t.name.starts_with("glTF")).unwrap();
    let settings = |name: &str| Settings {
        folder: folder.clone(),
        mesh_name: name.into(),
        project_name: "Test".into(),
        set_name: name.into(),
        set_count: 1,
        format: Format::Png,
        depth: export::Depth::U8,
        size: None,
        padding: Padding::Infinite,
        normal_strength: 1.0,
        use_baked_normal: false,
    };
    let snap = export::snapshot(&engine, &rs.device, &rs.queue, mesh.clone());
    let files = export::export(&snap, &gltf, &settings("Glow")).expect("exports");
    let file = files.iter().find(|f| f.to_string_lossy().contains("emissive")).expect("an emission map");
    let img = image::open(file).unwrap().to_rgb8();
    let max_g = img.pixels().map(|p| p[1]).max().unwrap();
    println!("  {}: brightest green {max_g}", file.display());
    assert!(max_g > 200, "the emission map carries the stroke");

    let mut plain = Engine::new(&rs, &gpu_mesh, (mesh.bounds_min, mesh.bounds_max), 64);
    plain.add_fill_layer(0, None, "Dark".into(), FillParams { mat: dark, ..Default::default() }, &mut undo);
    plain.composite();
    let snap = export::snapshot(&plain, &rs.device, &rs.queue, mesh.clone());
    let files = export::export(&snap, &gltf, &settings("Plain")).expect("exports");
    assert!(!files.iter().any(|f| f.to_string_lossy().contains("emissive")), "no emission map for a set that emits nothing: {files:?}");
}

/// A clean picture of a Painter project opened in Material Painter (`MP_SPP_SHOT=a.spp`): no
/// dialogs, the bakes done, rasterized and fully path traced, in `target/bake_test/spp/shot*.png`.
#[test]
#[ignore]
fn spp_screenshot() {
    use crate::app::PainterApp;
    let Ok(path) = std::env::var("MP_SPP_SHOT") else {
        println!("set MP_SPP_SHOT to a .spp file of your own");
        return;
    };
    let mut setup = egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    setup.device_descriptor = Arc::new(|adapter| wgpu::DeviceDescriptor {
        label: Some("ui test device"),
        required_limits: adapter.limits(),
        ..Default::default()
    });
    let mut harness = egui_kittest::Harness::builder()
        .with_size([1600.0, 1000.0])
        .with_pixels_per_point(1.0)
        .with_accessibility_check(false)
        .wgpu_setup(egui_wgpu::WgpuSetup::CreateNew(setup))
        .build_eframe(|cc| PainterApp::new(cc).expect("app starts"));
    harness.run_steps(4);
    let ctx = harness.ctx.clone();
    harness.state_mut().test_open_spp(Path::new(&path), &ctx);
    let mut steps = 0;
    while !(harness.state().test_spp_done() && !harness.state().test_baking()) && steps < 50_000 {
        harness.run_steps(1);
        steps += 1;
    }
    assert!(harness.state().test_spp_done() && !harness.state().test_baking(), "imported and baked");
    harness.state_mut().test_close_dialogs();
    harness.run_steps(3);
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test/spp");
    std::fs::create_dir_all(&dir).unwrap();
    harness.render().expect("renders").save(dir.join("shot.png")).unwrap();
    harness.state_mut().test_ray_trace(true);
    let mut steps = 0;
    while !harness.state().test_ray_trace_done() && steps < 20_000 {
        harness.run_steps(1);
        steps += 1;
    }
    harness.run_steps(1);
    harness.render().expect("renders").save(dir.join("shot_raytraced.png")).unwrap();
}
