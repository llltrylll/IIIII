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
    let raw = [([300.0, 230.0, 1.0, 0.0], 1.0)];
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

    let mut undo = UndoStack::default();
    let layer = engine.add_paint_layer(0, None, Some("Surface regression".into()), &mut undo);
    engine.composite();
    let before = read_texture(&rs.device, &rs.queue, engine.final_tex(0));

    assert!(engine.begin_stroke(
        layer,
        StrokeTarget::Content,
        [true, false, false, false, false, false, false, false],
        false,
    ));

    let uniform = StampUniform {
        view_proj: camera.view_proj(1.0).to_cols_array_2d(),
        cam_pos: camera.eye().extend(1.0).to_array(),
        view: [512.0, 512.0, 0.0, 0.0],
        view2d: [0.0; 4],
        brush: [20.0, 0.5, 1.0, 0.0],
        clip: [NEAR, FAR, 0.0, 0.0],
        proj: [1.0, 0.0, 85.0f32.to_radians().cos(), 0.0],
    };
    let paint = BrushPaint {
        mat: MatValues { color: [1.0, 0.05, 0.02], ..Default::default() },
        mask_value: 1.0,
        opacity: 1.0,
        erase: false,
        grain: [0.0; 4],
    };
    // The Surface fragment path does not sample depth, but the shared bind
    // layout still requires the multisampled depth view used by Screen paint.
    let depth = rs.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Surface projection test depth"),
        size: wgpu::Extent3d { width: 512, height: 512, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 4,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
    engine.stroke_step_screen_culled(
        &gpu_mesh,
        &candidate_indices,
        &stamps,
        uniform,
        &depth_view,
        &paint,
        regions,
    );
    let _ = engine.end_stroke();
    engine.composite();
    let after = read_texture(&rs.device, &rs.queue, engine.final_tex(0));

    let uv = Vec2::new(stamps[0][4][0], stamps[0][4][1]);
    let center_x = (uv.x * resolution as f32).floor() as usize;
    let center_y = (uv.y * resolution as f32).floor() as usize;
    let changed_near_center = (-2isize..=2).any(|dy| {
        (-2isize..=2).any(|dx| {
            let x = (center_x as isize + dx).clamp(0, resolution as isize - 1) as usize;
            let y = (center_y as isize + dy).clamp(0, resolution as isize - 1) as usize;
            let offset = (y * resolution as usize + x) * 4;
            before[offset..offset + 4] != after[offset..offset + 4]
        })
    });
    assert!(changed_near_center, "the hit Surface must change its own UV texels");

    // The rear chart's non-overlapping area is outside the front hit face's
    // UV bounds and must remain untouched by this Surface stamp.
    let outside_x = (0.74 * resolution as f32).floor() as usize;
    let outside_y = (0.74 * resolution as f32).floor() as usize;
    let outside = (outside_y * resolution as usize + outside_x) * 4;
    assert_eq!(
        &after[outside..outside + 4],
        &before[outside..outside + 4],
        "the unhit rear-only UV region must remain unchanged",
    );
}
