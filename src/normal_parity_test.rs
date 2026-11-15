//! Shading-normal parity with Blender.
//!
//! The reference data in `src/test_data/` was produced by real Blender
//! (Cycles) with `scripts/blender_normal_reference.py`; it is not computed from
//! this program's formulas. Two kinds of checks:
//!
//! * `mikk_*`: the per-corner MikkTSpace tangent and handedness this program
//!   builds against Blender's, on meshes with narrow tips, polar fans,
//!   mirrored UV islands and two-sided blades.
//! * `raster_*` (GPU, ignored): the real 3D view shader, read back through its
//!   diagnostic view modes, against the world-space normal Blender's Normal Map
//!   node produces for the same mesh, UVs, normals and tangent-space color.
//!
//! What "matching Blender" means here. Blender is the reference wherever the
//! imported normals agree with the triangle winding: front faces and back
//! faces both. For an imported normal that opposes the winding, Blender shades
//! the surface inside out; this viewport deliberately does not (see the
//! face-forward in `mesh_shade_inputs`), so the expectation there is
//! invariance: the viewport must show what Blender shows for the same mesh
//! with winding-consistent normals.

use crate::engine::{Engine, FillParams, MatValues, NORMAL};
use crate::gpu::{GpuMesh, read_texture};
use crate::mesh::Mesh;
use crate::preview_test::render_state;
use crate::undo::UndoStack;
use crate::viewport::{OrbitCam, ViewMode, ViewRenderer, ViewSettings};
use glam::{Vec2, Vec3};
use serde::Deserialize;
use std::sync::Arc;

// ---------------------------------------------------------------- data

#[derive(Deserialize)]
struct MikkCorner {
    t: [f32; 3],
    sign: f32,
    n: [f32; 3],
}

#[derive(Deserialize)]
struct QuadCase {
    uv: String,
    view: String,
    vertex_normal_z: i32,
    tilt: String,
    tilt_ts: [f32; 3],
    blender_world_normal: [f32; 3],
    mikk_corner0: MikkCorner,
}

#[derive(Deserialize)]
struct QuadData {
    cases: Vec<QuadCase>,
}

#[derive(Deserialize)]
struct TangentFixture {
    name: String,
    /// Per triangle, per corner: position (3), normal (3), UV (2, V up),
    /// tangent (3), bitangent sign (1).
    tris: Vec<[[f32; 12]; 3]>,
}

#[derive(Deserialize)]
struct TangentData {
    fixtures: Vec<TangentFixture>,
}

fn quad_cases() -> Vec<QuadCase> {
    serde_json::from_str::<QuadData>(include_str!("test_data/blender_quad_normals.json"))
        .expect("blender_quad_normals.json")
        .cases
}

/// The quad Blender rendered: XY plane, counter-clockwise seen from +Z, with
/// the UV orientation and imported normal of the case. UVs are given to
/// `Mesh::from_parts` in this program's V-down convention (v = 1 - Blender v).
fn quad_mesh(uv_mode: &str, vertex_normal_z: i32) -> Mesh {
    let positions = vec![
        Vec3::new(-1.0, -1.0, 0.0),
        Vec3::new(1.0, -1.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(-1.0, 1.0, 0.0),
    ];
    let (sx, sy) = match uv_mode {
        "normal" => (1.0, 1.0),
        "mirror_u" => (-1.0, 1.0),
        "flip_v" => (1.0, -1.0),
        "both" => (-1.0, -1.0),
        other => panic!("unknown UV mode {other}"),
    };
    let uvs = positions
        .iter()
        .map(|p| Vec2::new(0.5 + 0.4 * sx * p.x, 1.0 - (0.5 + 0.4 * sy * p.y)))
        .collect();
    Mesh::from_parts(
        format!("quad {uv_mode} {vertex_normal_z}"),
        positions,
        Some(vec![Vec3::new(0.0, 0.0, vertex_normal_z as f32); 4]),
        uvs,
        vec![0, 1, 2, 0, 2, 3],
    )
}

// ---------------------------------------------------------------- MikkTSpace

/// Steps 2 and 3 of the chain: the tangent and bitangent sign this program
/// generates (MikkTSpace with V flipped to up, handedness negated back to the
/// V-down convention the shaders read) must equal what Blender computes for
/// the same corner, including at narrow tips, polar fans and mirrored islands.
#[test]
fn mikk_tangents_match_blender_per_corner() {
    let data: TangentData = serde_json::from_str(include_str!("test_data/blender_tangents.json")).expect("blender_tangents.json");
    let mut report = Vec::new();
    for fx in &data.fixtures {
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut reference = Vec::new();
        for tri in &fx.tris {
            for c in tri {
                positions.push(Vec3::new(c[0], c[1], c[2]));
                normals.push(Vec3::new(c[3], c[4], c[5]));
                uvs.push(Vec2::new(c[6], 1.0 - c[7]));
                reference.push((Vec3::new(c[8], c[9], c[10]), c[11]));
            }
        }
        let indices: Vec<u32> = (0..positions.len() as u32).collect();
        let mesh = Mesh::from_parts(fx.name.clone(), positions, Some(normals), uvs, indices);
        assert_eq!(mesh.indices.len(), reference.len(), "{}: triangle soup must survive tangent generation", fx.name);
        let mut wrong_direction = 0usize;
        let mut wrong_sign = 0usize;
        let mut worst: f32 = 0.0;
        let mut skipped = 0usize;
        for (corner, (blender_t, blender_sign)) in reference.iter().enumerate() {
            if blender_t.length_squared() < 0.25 {
                skipped += 1;
                continue;
            }
            let v = &mesh.vertices[mesh.indices[corner] as usize];
            let t = Vec3::new(v.tangent[0], v.tangent[1], v.tangent[2]);
            let err = (t - *blender_t).length();
            worst = worst.max(err);
            if err > 5e-3 {
                wrong_direction += 1;
            }
            // Blender's sign is for V up; ours is for V down, so it is negated.
            if (v.tangent[3] - (-blender_sign)).abs() > 0.5 {
                wrong_sign += 1;
            }
        }
        println!(
            "  {:<22} {:>4} corners: {wrong_direction} tangents differ, {wrong_sign} handedness differ, worst tangent error {worst:.5} ({skipped} degenerate skipped)",
            fx.name,
            reference.len()
        );
        if wrong_direction > 0 || wrong_sign > 0 {
            report.push(format!("{}: {wrong_direction} tangents, {wrong_sign} handedness", fx.name));
        }
    }
    assert!(report.is_empty(), "MikkTSpace differs from Blender: {report:?}");
}

/// The quad fixtures' tangent and handedness for every UV orientation and
/// normal direction (the controlled surface the GPU test uses).
#[test]
fn mikk_tangents_match_blender_on_controlled_quad() {
    for case in quad_cases().iter().filter(|c| c.tilt == "flat" && c.view == "front") {
        let mesh = quad_mesh(&case.uv, case.vertex_normal_z);
        let v = &mesh.vertices[mesh.indices[0] as usize];
        let t = Vec3::new(v.tangent[0], v.tangent[1], v.tangent[2]);
        let blender_t = Vec3::from_array(case.mikk_corner0.t);
        assert!(
            (t - blender_t).length() < 1e-3,
            "uv {} normal {:+}: tangent {t:?} vs Blender {blender_t:?}",
            case.uv,
            case.vertex_normal_z
        );
        assert_eq!(
            v.tangent[3],
            -case.mikk_corner0.sign,
            "uv {} normal {:+}: handedness",
            case.uv,
            case.vertex_normal_z
        );
    }
}

// ---------------------------------------------------------------- GPU

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

/// Reads a vector back from a diagnostic view: the shader writes
/// `(v * 0.5 + 0.5)^2` and the target stores it sRGB-encoded.
fn decode_vector(px: &[u8]) -> Vec3 {
    Vec3::from_array(std::array::from_fn(|i| 2.0 * srgb_to_linear(px[i] as f32 / 255.0).sqrt() - 1.0))
}

fn tilted_normal_png(ts: [f32; 3]) -> Arc<Vec<u8>> {
    let color = ts.map(|c| ((c * 0.5 + 0.5) * 255.0).round() as u8);
    let image = image::RgbImage::from_pixel(32, 32, image::Rgb(color));
    let mut encoded = Vec::new();
    image.write_to(&mut std::io::Cursor::new(&mut encoded), image::ImageFormat::Png).unwrap();
    Arc::new(encoded)
}

fn engine_with_normal(rs: &eframe::egui_wgpu::RenderState, gpu_mesh: &GpuMesh, mesh: &Mesh, ts: [f32; 3]) -> Engine {
    let mut engine = Engine::new(rs, gpu_mesh, (mesh.bounds_min, mesh.bounds_max), 256);
    let mut undo = UndoStack::default();
    engine.add_fill_layer(
        0,
        None,
        "Neutral base".into(),
        FillParams { mat: MatValues { color: [0.65, 0.65, 0.65], rough: 0.55, ..Default::default() }, ..Default::default() },
        &mut undo,
    );
    let mut normal_mat = MatValues { enabled: [false; crate::engine::CHANNELS], ..Default::default() };
    normal_mat.enabled[NORMAL] = true;
    let layer = engine.add_fill_layer(1, None, "Tilted normal".into(), FillParams { mat: normal_mat, ..Default::default() }, &mut undo);
    let img = engine.decode_channel_image(NORMAL, "tilt.png".into(), tilted_normal_png(ts)).unwrap();
    engine.set_fill_image(layer, NORMAL, Some(img), &mut undo);
    engine.composite();
    engine
}

struct Mismatch {
    case: String,
    mode: &'static str,
    expected: Vec3,
    got: Vec3,
}

/// Steps 4 to 6 of the chain on the real 3D view shader, against Blender.
///
/// For every UV orientation, imported-normal direction, side seen and
/// tangent-space tilt Blender rendered, the diagnostic views must show:
/// * the decoded normal map (the "Normal (final)" view) equal to the tilt;
/// * the geometric normal facing the camera;
/// * the shading frame (normal, tangent, bitangent) equal to Blender's frame,
///   negated as a whole for back-facing triangles as Blender does;
/// * the mapped world-space normal equal to Blender's Normal Map node output.
///   Imported normals that oppose the winding must give the same result.
#[test]
#[ignore]
fn raster_normal_map_frame_matches_blender() {
    let cases = quad_cases();
    let rs = render_state();
    let env = crate::environment::EnvBaker::new(&rs.device)
        .upload(&rs, &crate::environment::decode(&crate::environment::EnvSource::Builtin(0)).unwrap());
    let size = [64u32, 64u32];
    let centre = ((size[1] / 2) as usize * size[0] as usize + (size[0] / 2) as usize) * 4;
    let mut mismatches: Vec<Mismatch> = Vec::new();
    let mut compared = 0usize;

    for uv in ["normal", "mirror_u", "flip_v", "both"] {
        for vn in [1, -1] {
            let mesh = quad_mesh(uv, vn);
            let gpu_mesh = GpuMesh::new(&rs.device, &mesh);
            let mut views = ViewRenderer::new(&rs);
            for tilt in ["flat", "x", "y"] {
                let ts = cases.iter().find(|c| c.tilt == tilt).unwrap().tilt_ts;
                let engine = engine_with_normal(&rs, &gpu_mesh, &mesh, ts);
                for view in ["front", "back"] {
                    let case = cases
                        .iter()
                        .find(|c| c.uv == uv && c.view == view && c.vertex_normal_z == vn && c.tilt == tilt)
                        .expect("Blender case");
                    let yaw = if view == "front" { 0.0 } else { std::f32::consts::PI };
                    let camera = OrbitCam { yaw, pitch: 0.0, dist: 4.0, target: Vec3::ZERO, fov_y: 38f32.to_radians() };
                    let to_camera = if view == "front" { Vec3::Z } else { -Vec3::Z };

                    // The reference is Blender's own case with winding-consistent
                    // normals (+1) for the same UVs, side and tilt. Its frame is
                    // negated as a whole for a back-facing triangle, as is its
                    // mapped normal (Blender's Normal Map node output).
                    let reference = cases
                        .iter()
                        .find(|c| c.uv == uv && c.view == view && c.vertex_normal_z == 1 && c.tilt == tilt)
                        .expect("Blender reference case");
                    let t0 = Vec3::from_array(reference.mikk_corner0.t);
                    let n0 = Vec3::from_array(reference.mikk_corner0.n).normalize();
                    let b0 = reference.mikk_corner0.sign * n0.cross(t0);
                    let frame_sign = if view == "back" { -1.0 } else { 1.0 };
                    let blender_normal = Vec3::from_array(reference.blender_world_normal);
                    let id = format!("uv={uv} imported normal {vn:+} view={view} tilt={tilt}");

                    let mut render = |mode: ViewMode| {
                        views.render_3d(
                            &rs,
                            size,
                            &[&engine],
                            &env,
                            &gpu_mesh,
                            &camera,
                            &ViewSettings { mode, ..Default::default() },
                            None,
                        );
                        decode_vector(&read_texture(&rs.device, &rs.queue, views.view3d.image())[centre..centre + 3])
                    };
                    let mut check = |mode: ViewMode, label: &'static str, expected: Vec3, tolerance: f32| {
                        let got = render(mode);
                        compared += 1;
                        if (got - expected).length() > tolerance {
                            mismatches.push(Mismatch { case: id.clone(), mode: label, expected, got });
                        }
                    };
                    check(ViewMode::Normal, "decoded normal map", Vec3::from_array(case.tilt_ts).normalize(), 0.06);
                    check(ViewMode::DiagGeometricNormal, "geometric normal", to_camera, 0.06);
                    check(ViewMode::DiagShadingNormal, "shading normal", frame_sign * n0, 0.06);
                    check(ViewMode::DiagTangent, "tangent", frame_sign * t0, 0.06);
                    check(ViewMode::DiagBitangent, "bitangent", frame_sign * b0, 0.06);
                    check(ViewMode::DiagMappedNormal, "mapped normal", blender_normal, 0.08);
                }
            }
        }
    }

    println!("  {} of {compared} diagnostic readings differ from Blender", mismatches.len());
    for m in &mismatches {
        println!(
            "    {:<52} {:<18} expected ({:+.2}, {:+.2}, {:+.2}) got ({:+.2}, {:+.2}, {:+.2})",
            m.case, m.mode, m.expected.x, m.expected.y, m.expected.z, m.got.x, m.got.y, m.got.z
        );
    }
    assert!(mismatches.is_empty(), "{} of {compared} shader readings differ from Blender", mismatches.len());
}

// ---------------------------------------------------------------- lighting

fn lit_value(px: &[u8]) -> f32 {
    (0..3).map(|i| srgb_to_linear(px[i] as f32 / 255.0)).sum::<f32>() / 3.0
}

/// Steps 6 and 7 of the chain, raster against path tracer against Blender.
///
/// A directional light from the side makes the lit brightness depend on which
/// way the mapped normal tilts. For every UV orientation, imported-normal
/// direction and side seen, a tilted normal map must get brighter or darker
/// than the flat one in the direction Blender's mapped normal predicts
/// (`N . L` from the Cycles render), in the raster view and in the path tracer
/// alike. Both lighting paths run for real; nothing here reproduces a formula.
///
/// NOT runnable on Lavapipe: the path-tracer compute shader segfaults the
/// software driver (even on a stock rounded cube), so this has only been
/// compiled, never executed, in the cloud. Run it on a real GPU:
/// `cargo test --bin material_painter normal_parity_test -- --include-ignored`.
#[test]
#[ignore]
fn raster_and_path_tracer_lighting_follow_blender_normals() {
    use crate::raytrace::{RayTracer, RtSettings};
    use crate::viewport::Background;
    let cases = quad_cases();
    let rs = render_state();
    let env = crate::environment::EnvBaker::new(&rs.device)
        .upload(&rs, &crate::environment::decode(&crate::environment::EnvSource::Builtin(0)).unwrap());
    let size = [64u32, 64u32];
    let centre = ((size[1] / 2) as usize * size[0] as usize + (size[0] / 2) as usize) * 4;
    let mut raytracer = RayTracer::new(&rs);
    let rt_settings = RtSettings { enabled: true, max_samples: 8, bounces: 1, ground: false };
    let mut rows = Vec::new();
    let mut failures = Vec::new();

    for uv in ["normal", "mirror_u"] {
        for vn in [1, -1] {
            let mesh = quad_mesh(uv, vn);
            let gpu_mesh = GpuMesh::new(&rs.device, &mesh);
            raytracer.set_mesh(&rs.device, &mesh, &crate::bvh::Bvh::from_mesh(&mesh));
            let mut views = ViewRenderer::new(&rs);
            for view in ["front", "back"] {
                // A light from the +X side and above, on the side the camera sees.
                let zs = if view == "front" { 0.55 } else { -0.55 };
                let to_light = Vec3::new(0.8, 0.25, zs).normalize();
                let mut settings = ViewSettings {
                    env_intensity: 0.0,
                    background: Background::Color,
                    bg_color: [0.0; 3],
                    bg_intensity: 0.0,
                    ..Default::default()
                };
                settings.key.on = true;
                settings.key.intensity = 3.0;
                settings.key.azimuth = to_light.x.atan2(to_light.z).to_degrees();
                settings.key.elevation = to_light.y.asin().to_degrees();
                let yaw = if view == "front" { 0.0 } else { std::f32::consts::PI };
                let camera = OrbitCam { yaw, pitch: 0.0, dist: 4.0, target: Vec3::ZERO, fov_y: 38f32.to_radians() };

                let mut brightness = |tilt: &str| -> (f32, f32) {
                    let ts = cases.iter().find(|c| c.tilt == tilt).unwrap().tilt_ts;
                    let engine = engine_with_normal(&rs, &gpu_mesh, &mesh, ts);
                    views.render_3d(&rs, size, &[&engine], &env, &gpu_mesh, &camera, &settings, None);
                    let raster = lit_value(&read_texture(&rs.device, &rs.queue, views.view3d.image())[centre..centre + 3]);
                    raytracer.samples = 0;
                    while !raytracer.converged(&rt_settings) {
                        raytracer.render(&rs, size, &[&engine], &env, &camera, &settings, &rt_settings);
                    }
                    let traced = lit_value(&read_texture(&rs.device, &rs.queue, raytracer.display_image().unwrap())[centre..centre + 3]);
                    (raster, traced)
                };
                let (flat_r, flat_p) = brightness("flat");
                for tilt in ["x", "y"] {
                    let (tilt_r, tilt_p) = brightness(tilt);
                    // Blender, with winding-consistent normals, as in the frame test.
                    let reference = |t: &str| {
                        Vec3::from_array(
                            cases.iter().find(|c| c.uv == uv && c.view == view && c.vertex_normal_z == 1 && c.tilt == t).unwrap().blender_world_normal,
                        )
                    };
                    let predicted = reference(tilt).dot(to_light) - reference("flat").dot(to_light);
                    let (dr, dp) = (tilt_r - flat_r, tilt_p - flat_p);
                    let id = format!("uv={uv} imported normal {vn:+} view={view} tilt={tilt}");
                    rows.push(format!("    {id:<46} Blender N.L {predicted:+.2}   raster {dr:+.3}   path tracer {dp:+.3}"));
                    // Only judge directions Blender predicts with a clear margin.
                    if predicted.abs() > 0.15 {
                        for (name, d) in [("raster", dr), ("path tracer", dp)] {
                            if d.signum() != predicted.signum() || d.abs() < 0.01 {
                                failures.push(format!("{id}: {name} brightness change {d:+.3}, Blender predicts {predicted:+.2}"));
                            }
                        }
                    }
                }
            }
        }
    }
    for r in &rows {
        println!("{r}");
    }
    assert!(failures.is_empty(), "lighting disagrees with Blender's normals:\n{}", failures.join("\n"));
}

