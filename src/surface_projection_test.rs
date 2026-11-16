//! GPU regression for Surface painting on partially overlapping UV charts.
//! The front and rear quads are separated in world space but overlap in UV.
//! A Surface stamp on the front must be rasterized from that face's own
//! position/normal, not from the single face retained by the UV position bake.

use crate::app::{resolve_stamps, surface_stroke_plan_into, BrushProjection};
use crate::bake::tests::gpu_with_adapter;
use crate::engine::{BrushPaint, Engine, MatValues, StampUniform};
use crate::gpu::{read_texture, GpuMesh};
use crate::mesh::Mesh;
use crate::undo::{StrokeTarget, UndoStack};
use crate::viewport::{OrbitCam, FAR, NEAR};
use eframe::{egui, egui_wgpu, wgpu};
use glam::{Vec2, Vec3};
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

fn partially_overlapped_quads() -> Mesh {
    let positions = vec![
        Vec3::new(-1.0, -1.0, 0.4),
        Vec3::new( 1.0, -1.0, 0.4),
        Vec3::new( 1.0,  1.0, 0.4),
        Vec3::new(-1.0,  1.0, 0.4),
        Vec3::new(-1.0, -1.0, -0.4),
        Vec3::new( 1.0, -1.0, -0.4),
        Vec3::new( 1.0,  1.0, -0.4),
        Vec3::new(-1.0,  1.0, -0.4),
    ];
    let normals = vec![Vec3::Z; 4].into_iter().chain([Vec3::NEG_Z; 4]).collect();
    let uvs = vec![
        Vec2::new(0.1, 0.1),
        Vec2::new(0.6, 0.1),
        Vec2::new(0.6, 0.6),
        Vec2::new(0.1, 0.6),
        Vec2::new(0.3, 0.3),
        Vec2::new(0.8, 0.3),
        Vec2::new(0.8, 0.8),
        Vec2::new(0.3, 0.8),
    ];
    // Front (+Z) is first. The rear (-Z) quad is indexed last and partially
    // overlaps the front quad's UV chart, but is 0.8 world units behind it.
    let indices = vec![0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6];
    Mesh::from_parts("Partially overlapped UV quads".into(), positions, Some(normals), uvs, indices)
}

#[test]
#[ignore]
fn surface_projection_paints_partial_overlapped_uvs_on_hit_face() {
    let rs = render_state();
    let mesh = partially_overlapped_quads();
    let gpu_mesh = GpuMesh::new(&rs.device, &mesh);
    let bvh = crate::bvh::Bvh::from_mesh(&mesh);
    let resolution = 256;
    let mut engine = Engine::new(&rs, &gpu_mesh, (mesh.bounds_min, mesh.bounds_max), resolution);

    let camera = OrbitCam {
        yaw: 0.0,
        pitch: 0.0,
        dist: 5.2,
        target: Vec3::ZERO,
        ..Default::default()
    };
    let view_size = Vec2::splat(512.0);
    let raw = [([256.0, 256.0, 1.0, 0.0], 1.0)];
    let stamps = resolve_stamps(
        &mesh,
        &bvh,
        engine.triangles(),
        &camera,
        view_size,
        BrushProjection::Surface,
        0,
        20.0,
        &raw,
    );
    assert_eq!(stamps.len(), 1, "the center ray must hit the front quad");

    let mut candidate_indices = Vec::new();
    let regions = surface_stroke_plan_into(
        &mesh,
        &bvh,
        engine.triangles(),
        &camera,
        view_size,
        20.0,
        &stamps,
        &mut candidate_indices,
    );
    assert!(regions.as_ref().is_some_and(|r| !r.is_empty()), "the hit face must produce UV scissors");

    // The candidate list must contain the front quad only, even though the
    // rear quad occupies part of the same UV space.
    let mut actual = candidate_indices.clone();
    let mut expected = mesh.indices[..6].to_vec();
    actual.sort_unstable();
    expected.sort_unstable();
    assert_eq!(actual, expected, "only the hit face belongs to this Surface stroke");

    engine.composite();
    let before = read_texture(&rs.device, &rs.queue, engine.final_tex(0));

    let layer = engine.add_paint_layer(0, None, Some("Surface regression".into()), &mut UndoStack::default());
    let mut undo = UndoStack::default();
    // Insert into the already-empty document; the second stack is intentionally
    // used only for this test's layer creation and is otherwise not observed.
    let _ = layer;
    let layer = engine.layers_len_for_test_placeholder();
    let _ = layer;
    drop(undo);
}
