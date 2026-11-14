//! Shelf thumbnails rendered by the real engine on a baked material ball, so
//! a smart material's swatch shows how its masks read edges, cavities, creases
//! and occlusion. Until the ball is baked the shelf shows the CPU spheres.

use super::{Shelf, add_smart_material_with_images};
use crate::bake::{self, BakeRun, BakeSettings, GpuBaker, PreparedScene, Quality};
use crate::engine::{Engine, FillImage, FillParams, GenKind, MatValues, PreparedFillImage};
use crate::environment::{self, EnvBaker, EnvSource, Environment};
use crate::gpu::GpuMesh;
use crate::mesh::Primitive;
use crate::presets;
use crate::undo::UndoStack;
use crate::viewport::{Background, OrbitCam, ViewRenderer, ViewSettings};
use eframe::{egui, egui_wgpu};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::mpsc;

/// Texture resolution of the ball; thumbnails are small.
const RES: u32 = 512;
/// Rendered thumbnail size in pixels: sharp up to the largest shelf size,
/// and for the 2x hover preview at the default size.
const THUMB: u32 = 192;

#[derive(Clone, Copy)]
enum Item {
    Smart(usize),
    Material(usize),
    Generator(usize),
}

enum Stage {
    Preparing(mpsc::Receiver<Result<PreparedScene, String>>),
    Baking(BakeRun),
    /// Rendering one thumbnail per frame, so the UI never stalls.
    Rendering(Vec<Item>),
}

pub(super) struct ShelfPreviews {
    mesh: GpuMesh,
    bounds: (glam::Vec3, glam::Vec3),
    baker: GpuBaker,
    engine: Option<Engine>,
    views: ViewRenderer,
    /// Fixed studio lighting, so thumbnails don't change with the viewport HDRI.
    env: Environment,
    /// Requests one smart material's scan images from the CPU worker.
    scan_tx: mpsc::Sender<Vec<PathBuf>>,
    /// Completed CPU decode/mipmap batches; at most one request is in flight.
    scan_rx: mpsc::Receiver<HashMap<PathBuf, Result<PreparedFillImage, String>>>,
    pending_scans: Option<HashSet<PathBuf>>,
    prepared_scans: HashMap<PathBuf, PreparedFillImage>,
    scan_failures: HashSet<PathBuf>,
    /// Strong references only for scans reused by a later shelf thumbnail.
    scan_images: HashMap<PathBuf, FillImage>,
    stage: Stage,
}

impl ShelfPreviews {
    pub fn start(rs: &egui_wgpu::RenderState, ctx: &egui::Context, env_baker: &EnvBaker) -> ShelfPreviews {
        let mesh = { let _s = crate::startup_trace::span("shelf: MaterialBall.build"); Primitive::MaterialBall.build() };
        let gpu_mesh = { let _s = crate::startup_trace::span("shelf: GpuMesh::new"); GpuMesh::new(&rs.device, &mesh) };
        let bounds = (mesh.bounds_min, mesh.bounds_max);
        let (tx, rx) = mpsc::channel();
        let bake_ctx = ctx.clone();
        {
            let _s = crate::startup_trace::span("shelf: spawn bake preparation worker");
            std::thread::spawn(move || {
                let settings = BakeSettings { quality: Quality::Draft, ..Default::default() };
                let prepared = {
                    let _s = crate::startup_trace::span("shelf: CPU bake::prepare (worker)");
                    bake::prepare(&mesh, &settings, 0)
                };
                let _ = tx.send(prepared);
                bake_ctx.request_repaint();
            });
        }

        let (scan_request_tx, scan_request_rx) = mpsc::channel::<Vec<PathBuf>>();
        let (scan_result_tx, scan_result_rx) = mpsc::channel::<HashMap<PathBuf, Result<PreparedFillImage, String>>>();
        let scan_ctx = ctx.clone();
        {
            let _s = crate::startup_trace::span("shelf: spawn scan decoder worker");
            std::thread::Builder::new()
                .name("shelf-scan-decode".into())
                .spawn(move || {
                    while let Ok(paths) = scan_request_rx.recv() {
                        let _batch = crate::startup_trace::span_d("scan: worker batch (read+decode+mips, all files)", crate::startup_trace::scan_label(&paths));
                        let mut decoded = HashMap::with_capacity(paths.len());
                        for path in paths {
                            let result = if path.is_file() {
                                Engine::prepare_fill_image_file(&path)
                            } else {
                                Err(format!("scan image is missing: {}", path.display()))
                            };
                            decoded.insert(path, result);
                        }
                        if scan_result_tx.send(decoded).is_err() {
                            break;
                        }
                        scan_ctx.request_repaint();
                    }
                })
                .expect("could not start shelf scan decoder");
        }

        ShelfPreviews {
            mesh: gpu_mesh,
            bounds,
            baker: { let _s = crate::startup_trace::span("shelf: GpuBaker::new"); GpuBaker::new(&rs.device) },
            engine: None,
            views: { let _s = crate::startup_trace::span("shelf: ViewRenderer::new"); ViewRenderer::new(rs) },
            env: {
                let _s = crate::startup_trace::span("shelf: environment decode+upload");
                env_baker.upload(rs, &environment::decode(&EnvSource::Builtin(0)).expect("the procedural studio always decodes"))
            },
            scan_tx: scan_request_tx,
            scan_rx: scan_result_rx,
            pending_scans: None,
            prepared_scans: HashMap::new(),
            scan_failures: HashSet::new(),
            scan_images: HashMap::new(),
            stage: Stage::Preparing(rx),
        }
    }

    /// Advances the bake or renders the next thumbnail. Returns false once
    /// every thumbnail is in place (or the bake failed) and this can be dropped.
    pub fn pump(&mut self, rs: &egui_wgpu::RenderState, ctx: &egui::Context, shelf: &mut Shelf) -> bool {
        match self.scan_rx.try_recv() {
            Ok(images) => {
                let received: Vec<&PathBuf> = images.keys().collect();
                crate::startup_trace::scan_received(&received[..]);
                self.pending_scans = None;
                for (path, result) in images {
                    match result {
                        Ok(image) => {
                            self.prepared_scans.insert(path, image);
                        }
                        Err(error) => {
                            eprintln!("shelf scan decode failed: {error}");
                            self.scan_failures.insert(path);
                        }
                    }
                }
                ctx.request_repaint();
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                if let Some(paths) = self.pending_scans.take() {
                    self.scan_failures.extend(paths);
                }
            }
            Err(mpsc::TryRecvError::Empty) => {}
        }
        match &mut self.stage {
            Stage::Preparing(rx) => match rx.try_recv() {
                Ok(Ok(scene)) => {
                    let settings = BakeSettings { quality: Quality::Draft, ..Default::default() };
                    let run = self.baker.start(&rs.device, &rs.queue, &self.mesh, &scene, &settings, RES);
                    self.stage = Stage::Baking(run);
                    ctx.request_repaint();
                    true
                }
                Ok(Err(e)) => {
                    eprintln!("shelf preview bake failed: {e}");
                    false
                }
                Err(mpsc::TryRecvError::Empty) => true,
                Err(mpsc::TryRecvError::Disconnected) => false,
            },
            Stage::Baking(run) => {
                if let Some(maps) = self.baker.pump(run, &rs.device, &rs.queue, ctx) {
                    let report = run.report.clone();
                    crate::startup_trace::event("shelf: GPU bake done");
                    let _s = crate::startup_trace::span("shelf: Engine::new (UI thread)");
                    let mut engine = Engine::new(rs, &self.mesh, self.bounds, RES);
                    engine.set_maps(maps, report);
                    self.engine = Some(engine);
                    drop(_s);
                    let mut items: Vec<Item> = (0..presets::SMART_MATERIALS.len()).map(Item::Smart).collect();
                    items.extend((0..presets::MATERIALS.len()).map(Item::Material));
                    items.extend((0..GenKind::ALL.len()).map(Item::Generator));
                    items.reverse();
                    self.stage = Stage::Rendering(items);
                    ctx.request_repaint();
                }
                true
            }
            Stage::Rendering(items) => {
                let Some(item) = items.last().copied() else { return false };
                if let Item::Smart(i) = item {
                    let required = smart_scan_paths(&presets::SMART_MATERIALS[i]);
                    if crate::startup_trace::enabled() {
                        let keys: Vec<String> = required.iter().map(|p| crate::startup_trace::file_key(p)).collect();
                        crate::startup_trace::smart_files(presets::SMART_MATERIALS[i].name, keys);
                    }
                    let missing: Vec<PathBuf> = required
                        .iter()
                        .filter(|path| {
                            !self.scan_images.contains_key(*path)
                                && !self.prepared_scans.contains_key(*path)
                                && !self.scan_failures.contains(*path)
                        })
                        .cloned()
                        .collect();
                    if !missing.is_empty() {
                        crate::startup_trace::count("scan: pump frames that waited on the worker (no thumbnail rendered)", 1);
                        if self.pending_scans.is_none() {
                            let pending: HashSet<PathBuf> = missing.iter().cloned().collect();
                            crate::startup_trace::scan_request(presets::SMART_MATERIALS[i].name, &missing);
                            if self.scan_tx.send(missing).is_ok() {
                                self.pending_scans = Some(pending);
                            } else {
                                self.scan_failures.extend(pending);
                            }
                        }
                        return true;
                    }

                    let engine = self.engine.as_ref().expect("engine exists while rendering");
                    let _upload = crate::startup_trace::span_d("scan: upload batch (UI thread)", presets::SMART_MATERIALS[i].name.to_string());
                    for path in required {
                        if self.scan_images.contains_key(&path) {
                            crate::startup_trace::count("scan: files reused (already a GPU texture in scan_images)", 1);
                        }
                        if !self.scan_images.contains_key(&path)
                            && let Some(prepared) = self.prepared_scans.remove(&path)
                        {
                            let image = engine.upload_prepared_fill_image(prepared);
                            self.scan_images.insert(path, image);
                        }
                    }
                    drop(_upload);

                    // Decode the next smart material while this one is being rendered.
                    if self.pending_scans.is_none()
                        && let Some(Item::Smart(next)) = items.iter().rev().nth(1).copied()
                    {
                        let next_paths = smart_scan_paths(&presets::SMART_MATERIALS[next]);
                        let missing_next: Vec<PathBuf> = next_paths
                            .iter()
                            .filter(|path| {
                                !self.scan_images.contains_key(*path)
                                    && !self.prepared_scans.contains_key(*path)
                                    && !self.scan_failures.contains(*path)
                            })
                            .cloned()
                            .collect();
                        if !missing_next.is_empty() {
                            let pending: HashSet<PathBuf> = missing_next.iter().cloned().collect();
                            crate::startup_trace::scan_request(presets::SMART_MATERIALS[next].name, &missing_next);
                            if self.scan_tx.send(missing_next).is_ok() {
                                self.pending_scans = Some(pending);
                            } else {
                                self.scan_failures.extend(pending);
                            }
                        }
                    }
                }
                let item = items.pop().expect("item was checked above");
                let engine = self.engine.as_mut().expect("engine exists while rendering");
                let image = render(rs, engine, &self.env, &mut self.views, &self.mesh, item, &self.scan_images);
                let handle = match item {
                    Item::Smart(i) => &mut shelf.smart[i],
                    Item::Material(i) => &mut shelf.materials[i],
                    Item::Generator(i) => &mut shelf.generators[i],
                };
                {
                    let _s = crate::startup_trace::span("shelf: TextureHandle::set (UI thread)");
                    handle.set(image, egui::TextureOptions::LINEAR);
                }
                crate::startup_trace::event("thumbnail ready");

                // Keep a scan texture only when a later smart material will reuse it.
                let future_paths: HashSet<PathBuf> = items
                    .iter()
                    .filter_map(|item| match item {
                        Item::Smart(i) => Some(smart_scan_paths(&presets::SMART_MATERIALS[*i])),
                        _ => None,
                    })
                    .flatten()
                    .collect();
                self.scan_images.retain(|path, _| future_paths.contains(path));
                self.prepared_scans.retain(|path, _| future_paths.contains(path));
                self.scan_failures.retain(|path| future_paths.contains(path));
                ctx.request_repaint();
                !items.is_empty()
            }
        }
    }
}

fn smart_scan_paths(smart: &presets::SmartMaterial) -> Vec<PathBuf> {
    let Some(root) = environment::asset_dir("pbr") else { return Vec::new() };
    let mut paths = HashSet::new();
    for layer in smart.layers.iter().chain(&presets::MICRO_DETAIL) {
        if let Some(scan) = layer.scan {
            for (_, file) in presets::Scan::FILES {
                paths.insert(root.join(scan.id).join(file));
            }
        }
    }
    let mut paths: Vec<_> = paths.into_iter().collect();
    paths.sort();
    paths
}

fn render(
    rs: &egui_wgpu::RenderState,
    engine: &mut Engine,
    env: &Environment,
    views: &mut ViewRenderer,
    mesh: &GpuMesh,
    item: Item,
    scan_images: &HashMap<PathBuf, FillImage>,
) -> egui::ColorImage {
    let trace_start = std::time::Instant::now();
    let (trace_kind, trace_label) = match item {
        Item::Smart(i) => ("smart", presets::SMART_MATERIALS[i].name.to_string()),
        Item::Material(i) => ("material", presets::MATERIALS[i].name.to_string()),
        Item::Generator(i) => ("generator", GenKind::ALL[i].name().to_string()),
    };
    let mut undo = UndoStack::default();
    while !engine.layers.is_empty() {
        engine.delete_layer(0, &mut undo);
    }
    match item {
        Item::Smart(i) => {
            add_smart_material_with_images(engine, &mut undo, &presets::SMART_MATERIALS[i], 0, None, scan_images);
        }
        Item::Material(i) => {
            let m = &presets::MATERIALS[i];
            engine.add_fill_layer(0, None, m.name.to_string(), m.fill(), &mut undo);
        }
        Item::Generator(i) => {
            // The mask in white over black, like the generator's own preview.
            let flat = |c: f32| FillParams {
                mat: MatValues { color: [c; 3], rough: 0.6, ..MatValues::default() },
                ..Default::default()
            };
            engine.add_fill_layer(0, None, "Black".into(), flat(0.06), &mut undo);
            let top = engine.add_fill_layer(1, None, "Mask".into(), flat(0.92), &mut undo);
            engine.add_mask(top, true, Some(GenKind::ALL[i].default_generator()), &mut undo);
        }
    }
    let t_layers = trace_start.elapsed();
    engine.composite();
    let t_composite = trace_start.elapsed();
    let cam = OrbitCam { yaw: 0.5, pitch: 0.32, dist: 4.05, target: glam::Vec3::new(0.0, -0.2, 0.0), ..OrbitCam::default() };
    let mut settings = ViewSettings { show_wire: false, background: Background::Transparent, ..ViewSettings::default() };
    settings.post.vignette = 0.0;
    settings.post.bloom = 0.0;
    views.render_3d(rs, [THUMB, THUMB], &[engine], env, mesh, &cam, &settings, None);
    let t_render = trace_start.elapsed();
    let img = views.view3d.image();
    // Only when tracing: wait for the queue first so GPU execution of this
    // thumbnail is timed apart from the copy + map read_texture then does. The
    // result is the same; read_texture waits for the same work itself.
    let t_wait_start = trace_start.elapsed();
    if crate::startup_trace::enabled() {
        let _ = rs.device.poll(eframe::wgpu::PollType::wait_indefinitely());
    }
    let t_wait_end = trace_start.elapsed();
    let rgba = crate::gpu::read_texture(&rs.device, &rs.queue, img);
    let t_read = trace_start.elapsed();
    // MSAA resolve averages edge samples with the transparent clear, so the
    // silhouette comes out premultiplied.
    let image = egui::ColorImage::from_rgba_premultiplied([img.width as usize, img.height as usize], &rgba);
    if crate::startup_trace::enabled() {
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
        crate::startup_trace::thumb(crate::startup_trace::Thumb {
            label: trace_label,
            kind: trace_kind,
            at: crate::startup_trace::now_ms(),
            layers: ms(t_layers),
            composite: ms(t_composite - t_layers),
            render: ms(t_render - t_composite),
            gpu_wait: ms(t_wait_end - t_wait_start),
            copy_map: ms(t_read - t_wait_end),
            total: ms(trace_start.elapsed()),
        });
    }
    image
}
