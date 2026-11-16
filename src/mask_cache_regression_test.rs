//! GPU regression: material-only edits must reuse filtered mask results,
//! while edits to mask inputs must rebuild the result.

use crate::bake::tests::gpu_with_adapter;
use crate::engine::{Content, Engine, FilterKind, FillParams, GenKind, MaskFilter};
use crate::gpu::GpuMesh;
use crate::mesh::Primitive;
use eframe::{egui, egui_wgpu, wgpu};
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

#[test]
#[ignore]
fn material_edits_reuse_filtered_id_mask_but_mask_edits_rebuild_it() {
    let rs = render_state();
    let mesh = Primitive::RoundedCube.build();
    let gpu_mesh = GpuMesh::new(&rs.device, &mesh);
    let mut engine = Engine::new(&rs, &gpu_mesh, (mesh.bounds_min, mesh.bounds_max), 128);
    let mut undo = crate::undo::UndoStack::default();

    let layer = engine.add_fill_layer(0, None, "ID-masked material".into(), FillParams::default(), &mut undo);
    engine.add_mask(layer, true, Some(GenKind::IdMask.default_generator()), &mut undo);
    let mask = engine.layers[layer].mask.as_mut().expect("ID mask exists");
    mask.filters = vec![MaskFilter::new(FilterKind::Blur), MaskFilter::new(FilterKind::Warp)];
    engine.mark_dirty();
    engine.commit_props(&mut undo);
    engine.composite();

    assert!(engine.last_mask_cache_misses > 0, "the initial composite must build the filtered mask");
    assert!(engine.last_mask_gpu_passes > 0, "the initial mask build must submit GPU passes");

    // These parameters change the material output but are not inputs to the
    // filtered mask. The mask texture must remain cached across the composite.
    if let Content::Fill(fill) = &mut engine.layers[layer].content {
        fill.mat.color = [0.92, 0.12, 0.04];
        fill.mat.rough = 0.27;
        fill.mat.metal = 0.68;
        fill.color_var = 0.31;
    } else {
        panic!("the test layer must remain a Fill layer");
    }
    engine.mark_dirty();
    engine.commit_props(&mut undo);
    engine.composite();

    assert!(engine.last_mask_cache_hits > 0, "material-only changes must reuse the filtered ID mask");
    assert_eq!(engine.last_mask_cache_misses, 0, "material-only changes must not rebuild the ID mask");
    assert_eq!(engine.last_mask_gpu_passes, 0, "a mask cache hit must submit no mask-generation/filter passes");

    // A generator edit is a real mask-input change and must invalidate the
    // previous cached result for this layer.
    engine.layers[layer].mask.as_mut().unwrap().generator.as_mut().unwrap().level += 0.1;
    engine.mark_dirty();
    engine.commit_props(&mut undo);
    engine.composite();
    assert!(engine.last_mask_cache_misses > 0, "changing generator parameters must invalidate the mask cache");
    assert!(engine.last_mask_gpu_passes > 0, "an invalidated generator must rebuild its GPU result");

    // Filter edits must independently invalidate the same mask result.
    engine.layers[layer].mask.as_mut().unwrap().filters[0].a += 2.0;
    engine.mark_dirty();
    engine.commit_props(&mut undo);
    engine.composite();
    assert!(engine.last_mask_cache_misses > 0, "changing filter parameters must invalidate the mask cache");
    assert!(engine.last_mask_gpu_passes > 0, "an invalidated filter stack must rebuild its GPU result");
}
