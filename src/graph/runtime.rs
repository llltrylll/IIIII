//! Renders a graph on the GPU: one pipeline per node definition, one
//! texture per node. A node only re-renders when its definition, its
//! parameters or anything upstream changed (tracked by a hash).

use super::def::{self, MAX_IMAGES, MAX_INPUTS, NodeDef, ParamKind, Value};
use super::{Graph, Library};
use crate::gpu::{self, LAYER_USAGE, Tex};
use eframe::wgpu;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::{Arc, mpsc};

/// Every node renders in this format: colors and grays (in all of rgb)
/// alike, so any output can feed any input.
pub const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct NodeUniform {
    p: [[f32; 4]; def::MAX_PARAMS],
    defaults: [[f32; 4]; MAX_INPUTS],
    info: [f32; 4],
}

struct Compiled {
    /// Hash of the source it was compiled from.
    source: u64,
    pipeline: Result<wgpu::RenderPipeline, String>,
}

struct Rendered {
    hash: u64,
    /// Dropped after a lean evaluation (see [`Runtime::evaluate`]).
    tex: Option<Arc<Tex>>,
    /// 8-bit copy for the editor (egui shows Rgba8Unorm textures).
    thumb: Option<Arc<Tex>>,
}

/// Thumbnail edge in texels.
pub const THUMB: u32 = 256;

const BLIT: &str = "
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
@fragment
fn fs_blit(in: FsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(clamp(textureSampleLevel(src, smp, in.uv, 0.0).rgb, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
";

/// The mesh surface `@surface` nodes read: world position and normal
/// (`Engine::surface_maps`), the baked maps A, B, C and E (`Engine::map_tex`),
/// the bounds and the baked edge distance's range.
pub struct Surface<'a> {
    pub pos: &'a Tex,
    pub nrm: &'a Tex,
    pub maps: [&'a Tex; 4],
    pub bmin: [f32; 3],
    pub bsize: [f32; 3],
    pub edge_max: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SurfaceUniform {
    bmin: [f32; 4],
    bsize: [f32; 4],
    info: [f32; 4],
}

/// A path inside the app's assets folder ("pbr/rock/color.jpg"): its
/// first part names the asset folder, found wherever the app runs from.
pub(crate) fn asset_path(rel: &std::path::Path) -> PathBuf {
    let mut parts = rel.components();
    let first = parts.next().map(|c| c.as_os_str().to_string_lossy().into_owned()).unwrap_or_default();
    crate::environment::asset_dir(&first).map_or_else(|| rel.to_path_buf(), |d| d.join(parts.as_path()))
}

/// An image file a node uses: decoding on a worker thread, or done.
enum ImageState {
    Loading,
    Ready(Arc<Tex>),
    Failed(String),
}

/// Images decoded on worker threads, waiting to be uploaded.
type Decoded = (PathBuf, Result<image::RgbaImage, String>);

pub struct Runtime {
    blit: wgpu::RenderPipeline,
    blit_bgl: wgpu::BindGroupLayout,
    bgl: wgpu::BindGroupLayout,
    layout: wgpu::PipelineLayout,
    surface_bgl: wgpu::BindGroupLayout,
    surface_layout: wgpu::PipelineLayout,
    sampler: wgpu::Sampler,
    clamp_sampler: wgpu::Sampler,
    dummy: Tex,
    pipelines: HashMap<String, Compiled>,
    rendered: HashMap<u64, Rendered>,
    images: HashMap<PathBuf, ImageState>,
    decoded: (mpsc::Sender<Decoded>, mpsc::Receiver<Decoded>),
    /// Called from the worker when an image is decoded (a repaint).
    waker: Option<Arc<dyn Fn() + Send + Sync>>,
    /// Node id → why it didn't render.
    pub errors: HashMap<u64, String>,
}

fn hash_of(v: impl Hash) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    v.hash(&mut h);
    h.finish()
}

/// Runs `f` and returns the first validation error it caused, as text.
pub(crate) fn validation_scope<T>(device: &wgpu::Device, f: impl FnOnce() -> T) -> (T, Option<String>) {
    use std::future::Future;
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let value = f();
    let mut pop = std::pin::pin!(scope.pop());
    let err = match pop.as_mut().poll(&mut std::task::Context::from_waker(std::task::Waker::noop())) {
        std::task::Poll::Ready(Some(e)) => Some(e.to_string()),
        _ => None,
    };
    (value, err)
}

/// The mesh maps a graph can read: a Mesh Map value's texture and channel
/// (see `Engine::mesh_map`), `None` when it isn't baked.
pub type MeshMaps<'a> = &'a dyn Fn(usize) -> Option<(&'a Tex, usize)>;

impl Runtime {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Runtime {
        let mut entries = vec![gpu::uniform_entry(0, false), gpu::sampler_entry(1)];
        entries.extend((0..(MAX_INPUTS + MAX_IMAGES) as u32).map(|i| gpu::tex_entry(2 + i, true)));
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("graph node"), entries: &entries });
        let layout = gpu::layout(device, &bgl);
        let mut surface_entries = vec![gpu::uniform_entry(0, false)];
        surface_entries.extend((1..=6).map(|b| gpu::tex_entry(b, true)));
        surface_entries.push(gpu::sampler_entry(7));
        let surface_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("graph surface"), entries: &surface_entries });
        let surface_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("graph surface node"),
            bind_group_layouts: &[Some(&bgl), Some(&surface_bgl)],
            immediate_size: 0,
        });
        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("graph thumb"),
            entries: &[gpu::tex_entry(0, true), gpu::sampler_entry(1)],
        });
        let blit_module = gpu::shader(device, "graph thumb", BLIT);
        let blit = gpu::fullscreen_pipeline(
            device,
            "graph thumb",
            &blit_module,
            "fs_blit",
            &gpu::layout(device, &blit_bgl),
            &[gpu::target(wgpu::TextureFormat::Rgba8Unorm)],
        );
        let dummy = Tex::new(device, "graph dummy", 1, 1, FORMAT, LAYER_USAGE);
        queue.write_texture(dummy.copy_info(), bytemuck::cast_slice(&[0u16; 4]), Default::default(), dummy.extent());
        Runtime {
            blit,
            blit_bgl,
            bgl,
            layout,
            surface_bgl,
            surface_layout,
            sampler: gpu::linear_sampler(device, wgpu::AddressMode::Repeat),
            clamp_sampler: gpu::linear_sampler(device, wgpu::AddressMode::ClampToEdge),
            dummy,
            pipelines: HashMap::new(),
            rendered: HashMap::new(),
            images: HashMap::new(),
            decoded: mpsc::channel(),
            waker: None,
            errors: HashMap::new(),
        }
    }

    /// Called (from a worker thread) whenever an image a node needs has
    /// been decoded: the app repaints, and [`Runtime::take_images`] then
    /// says the graph has to render again.
    pub fn set_waker(&mut self, waker: Arc<dyn Fn() + Send + Sync>) {
        self.waker = Some(waker);
    }

    /// Uploads images decoded since the last call; true when any arrived
    /// (the nodes waiting for them can render now).
    pub fn take_images(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
        let mut any = false;
        while let Ok(d) = self.decoded.1.try_recv() {
            self.accept(device, queue, d);
            any = true;
        }
        any
    }

    fn accept(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, (path, result): Decoded) {
        let state = match result {
            Ok(img) => ImageState::Ready(Arc::new(gpu::upload_mips(device, queue, img, wgpu::TextureFormat::Rgba8Unorm, "graph image"))),
            Err(e) => ImageState::Failed(e),
        };
        self.images.insert(path, state);
    }

    /// Blocks until every image being decoded has arrived (tests).
    #[cfg(test)]
    pub fn wait_images(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        while self.images.values().any(|s| matches!(s, ImageState::Loading)) {
            let Ok(d) = self.decoded.1.recv() else { return };
            self.accept(device, queue, d);
        }
    }

    /// The last render of a node.
    pub fn output(&self, node: u64) -> Option<&Arc<Tex>> {
        self.rendered.get(&node).and_then(|r| r.tex.as_ref())
    }

    /// A node's 8-bit thumbnail.
    pub fn thumb(&self, node: u64) -> Option<&Arc<Tex>> {
        self.rendered.get(&node).and_then(|r| r.thumb.as_ref())
    }

    /// Checks a definition compiles, without rendering. `Err` holds the
    /// compiler's message.
    #[cfg(test)]
    pub fn compile(&mut self, device: &wgpu::Device, def: &NodeDef) -> Result<(), String> {
        self.pipeline(device, def).map(|_| ())
    }

    fn pipeline(&mut self, device: &wgpu::Device, def: &NodeDef) -> Result<&wgpu::RenderPipeline, String> {
        let source = hash_of(&def.source);
        let fresh = self.pipelines.get(&def.id).is_some_and(|c| c.source == source);
        if !fresh {
            let layout = if def.surface { &self.surface_layout } else { &self.layout };
            let (pipeline, err) = validation_scope(device, || {
                let module = gpu::shader(device, &def.id, &def.shader());
                gpu::fullscreen_pipeline(device, &def.id, &module, "fs_node", layout, &[gpu::target(FORMAT)])
            });
            let pipeline = match err {
                Some(e) => Err(e),
                None => Ok(pipeline),
            };
            self.pipelines.insert(def.id.clone(), Compiled { source, pipeline });
        }
        self.pipelines[&def.id].pipeline.as_ref().map_err(Clone::clone)
    }

    /// A node's image: `Ok(None)` while it decodes on a worker thread.
    /// Relative paths are inside the app's assets folder (built-in scans).
    fn image(&mut self, path: &PathBuf) -> Result<Option<Arc<Tex>>, String> {
        match self.images.get(path) {
            Some(ImageState::Ready(t)) => return Ok(Some(t.clone())),
            Some(ImageState::Failed(e)) => return Err(e.clone()),
            Some(ImageState::Loading) => return Ok(None),
            None => {}
        }
        self.images.insert(path.clone(), ImageState::Loading);
        let file = if path.is_relative() { asset_path(path) } else { path.clone() };
        let (tx, waker, key) = (self.decoded.0.clone(), self.waker.clone(), path.clone());
        std::thread::spawn(move || {
            let img = image::open(&file).map(|i| i.into_rgba8()).map_err(|e| format!("{}: {e}", file.display()));
            if tx.send((key, img)).is_ok()
                && let Some(w) = waker {
                    w();
                }
        });
        Ok(None)
    }

    /// Renders the nodes whose inputs changed. `maps_key` changes whenever
    /// the mesh maps do. With `lean`, only the Output nodes keep their
    /// textures afterwards (a graph that isn't open in the editor: a big
    /// smart material would otherwise hold a texture per node); a later
    /// change renders again what it needs. Returns whether anything
    /// re-rendered.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        graph: &Graph,
        lib: &Library,
        maps: MeshMaps,
        surface: Option<&Surface>,
        maps_key: u64,
        lean: bool,
    ) -> bool {
        let res = graph.res.clamp(16, 4096);
        self.rendered.retain(|id, _| graph.node(*id).is_some());
        self.errors.retain(|id, _| graph.node(*id).is_some());
        let order = graph.order();

        // What each node would render now: its hash, from its definition,
        // values and what feeds it.
        struct Plan<'a> {
            def: &'a NodeDef,
            values: Vec<Value>,
            inputs: Vec<Option<u64>>,
            hash: u64,
        }
        let mut plans: HashMap<u64, Plan> = HashMap::new();
        for &id in &order {
            let node = graph.node(id).expect("ordered from the graph");
            let Some(def) = lib.find(&node.def) else {
                self.errors.insert(id, format!("unknown node '{}'", node.def));
                self.rendered.remove(&id);
                continue;
            };
            let values: Vec<Value> = def.params.iter().map(|p| lib.value(node, &p.name).unwrap_or_else(|| p.default.clone())).collect();
            let inputs: Vec<Option<u64>> =
                def.inputs.iter().map(|i| node.links.get(&i.name).copied().filter(|from| plans.contains_key(from))).collect();
            let uses_maps = def.surface || def.params.iter().any(|p| p.kind == ParamKind::MeshMap);
            let hash = hash_of((
                &def.source,
                format!("{values:?}"),
                res,
                inputs.iter().map(|i| i.map(|f| plans[&f].hash)).collect::<Vec<_>>(),
                if uses_maps { maps_key } else { 0 },
            ));
            plans.insert(id, Plan { def, values, inputs, hash });
        }

        // Nodes to render: those that changed or failed, the outputs a lean
        // graph lacks, and (going upstream) whatever they read that holds
        // no texture right now.
        let outputs: HashSet<u64> = graph.outputs(lib).into_iter().flatten().collect();
        let stale = |id: &u64, p: &Plan| match self.rendered.get(id) {
            Some(r) => r.hash != p.hash || self.errors.contains_key(id),
            None => true,
        };
        let mut needed: HashSet<u64> = HashSet::new();
        for &id in order.iter().rev() {
            let Some(p) = plans.get(&id) else { continue };
            let has_tex = self.rendered.get(&id).is_some_and(|r| r.tex.is_some());
            let wanted = !lean || outputs.contains(&id) || needed.contains(&id);
            if wanted && (stale(&id, p) || !has_tex) {
                needed.insert(id);
            }
            if needed.contains(&id) {
                for from in p.inputs.iter().flatten() {
                    if !self.rendered.get(from).is_some_and(|r| r.tex.is_some()) {
                        needed.insert(*from);
                    }
                }
            }
        }
        let mut changed = false;
        // Nodes that didn't render this time (waiting for an image, failed).
        let mut missing: HashSet<u64> = HashSet::new();
        // Lean: how many nodes still to render read each node; at zero its
        // texture goes back to `pool` for the next node to render into, so
        // a big graph holds a few textures at a time rather than one per node.
        let mut readers: HashMap<u64, usize> = HashMap::new();
        for id in order.iter().filter(|id| needed.contains(*id)) {
            for f in plans[id].inputs.iter().flatten() {
                *readers.entry(*f).or_default() += 1;
            }
        }
        let mut pool: Vec<Arc<Tex>> = Vec::new();
        let mut encoder = device.create_command_encoder(&Default::default());
        for &id in &order {
            if !needed.contains(&id) {
                continue;
            }
            let p = &plans[&id];
            let reuse = if lean && !outputs.contains(&id) { pool.pop() } else { None };
            // Rendered without one of its inputs: stays stale, so it renders
            // again once that input is there.
            let partial = p.inputs.iter().flatten().any(|f| missing.contains(f));
            let was = self.rendered.get(&id).map(|r| r.hash);
            let result = self.render(device, &mut encoder, p.def, &p.values, &p.inputs, res, maps, surface, reuse);
            if lean {
                for f in p.inputs.iter().flatten() {
                    let left = readers.get_mut(f).map(|n| {
                        *n -= 1;
                        *n
                    });
                    if left == Some(0) && !outputs.contains(f)
                        && let Some(t) = self.rendered.get_mut(f).and_then(|r| r.tex.take()) {
                            // Only this runtime holds it: free to render into.
                            if Arc::strong_count(&t) == 1 {
                                pool.push(t);
                            }
                        }
                }
            }
            match result {
                Ok(None) => {
                    // Waits for an image: renders once it has arrived
                    // (`take_images` says when).
                    self.rendered.remove(&id);
                    missing.insert(id);
                }
                Ok(Some(tex)) if lean => {
                    // Nobody looks at thumbnails of a graph the editor doesn't show.
                    self.errors.remove(&id);
                    changed |= was != Some(p.hash);
                    let hash = if partial { !p.hash } else { p.hash };
                    if partial {
                        missing.insert(id);
                    }
                    self.rendered.insert(id, Rendered { hash, tex: Some(tex), thumb: None });
                }
                Ok(Some(tex)) => {
                    let thumb = Arc::new(Tex::new(device, "graph thumb", THUMB, THUMB, wgpu::TextureFormat::Rgba8Unorm, LAYER_USAGE));
                    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("graph thumb"),
                        layout: &self.blit_bgl,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&tex.view) },
                            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                        ],
                    });
                    {
                        let mut pass =
                            gpu::pass(&mut encoder, "graph thumb", &[gpu::color_attachment(&thumb.view, wgpu::LoadOp::Clear(wgpu::Color::BLACK))]);
                        pass.set_pipeline(&self.blit);
                        pass.set_bind_group(0, &bg, &[]);
                        pass.draw(0..3, 0..1);
                    }
                    self.errors.remove(&id);
                    // Re-rendering a dropped texture with the same hash isn't a change.
                    changed |= was != Some(p.hash);
                    let hash = if partial { !p.hash } else { p.hash };
                    if partial {
                        missing.insert(id);
                    }
                    self.rendered.insert(id, Rendered { hash, tex: Some(tex), thumb: Some(thumb) });
                }
                Err(e) => {
                    self.errors.insert(id, e);
                    self.rendered.remove(&id);
                    missing.insert(id);
                    changed = true;
                }
            }
        }
        queue.submit([encoder.finish()]);
        if lean {
            // Only the outputs stay; the hashes remember what was rendered.
            for (id, r) in &mut self.rendered {
                if !outputs.contains(id) {
                    r.tex = None;
                    r.thumb = None;
                }
            }
        }
        changed
    }

    #[allow(clippy::too_many_arguments)]
    fn render(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        def: &NodeDef,
        values: &[Value],
        inputs: &[Option<u64>],
        res: u32,
        maps: MeshMaps,
        surface: Option<&Surface>,
        reuse: Option<Arc<Tex>>,
    ) -> Result<Option<Arc<Tex>>, String> {
        if def.surface && surface.is_none() {
            return Err("needs a mesh: this node reads the model's surface".into());
        }
        // Image and mesh map slots.
        let mut slots: Vec<Option<Arc<Tex>>> = vec![None; MAX_IMAGES];
        let mut map_slots: Vec<Option<usize>> = vec![None; MAX_IMAGES];
        let mut comp_bits = 0u32;
        for (pi, v) in values.iter().enumerate() {
            let Some(slot) = def.image_slot(pi) else { continue };
            match v {
                Value::Image(Some(path)) => match self.image(path)? {
                    Some(t) => slots[slot] = Some(t),
                    None => return Ok(None),
                },
                Value::MeshMap(m) => {
                    if let Some((_, comp)) = maps(*m) {
                        map_slots[slot] = Some(*m);
                        comp_bits |= (comp as u32) << (slot * 2);
                    }
                }
                _ => {}
            }
        }
        let mut u = NodeUniform { p: [[0.0; 4]; def::MAX_PARAMS], defaults: [[0.0; 4]; MAX_INPUTS], info: [0.0; 4] };
        for (i, v) in values.iter().enumerate() {
            u.p[i] = def::pack(v);
        }
        let mut bits = 0u32;
        for (i, inp) in def.inputs.iter().enumerate() {
            u.defaults[i] = inp.default;
            if inputs[i].is_some() {
                bits |= 1 << i;
            }
        }
        u.info = [res as f32, bits as f32, comp_bits as f32, 0.0];
        let pipeline = self.pipeline(device, def)?.clone();

        let ubuf = {
            use wgpu::util::DeviceExt;
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("graph node"),
                contents: bytemuck::bytes_of(&u),
                usage: wgpu::BufferUsages::UNIFORM,
            })
        };
        let input_tex: Vec<Option<Arc<Tex>>> = inputs.iter().map(|i| i.and_then(|f| self.rendered.get(&f).and_then(|r| r.tex.clone()))).collect();
        // A fresh texture (layers may still hold the previous one), or one
        // a lean evaluation is done with.
        let out = reuse.unwrap_or_else(|| Arc::new(Tex::new(device, &def.id, res, res, FORMAT, LAYER_USAGE)));
        {
            let mut entries = vec![
                wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ];
            for i in 0..MAX_INPUTS {
                let view = input_tex.get(i).and_then(|t| t.as_ref()).map_or(&self.dummy.view, |t| &t.view);
                entries.push(wgpu::BindGroupEntry { binding: 2 + i as u32, resource: wgpu::BindingResource::TextureView(view) });
            }
            for i in 0..MAX_IMAGES {
                let view = match (&slots[i], map_slots[i]) {
                    (Some(t), _) => &t.view,
                    (None, Some(m)) => maps(m).map_or(&self.dummy.view, |(t, _)| &t.view),
                    _ => &self.dummy.view,
                };
                entries.push(wgpu::BindGroupEntry {
                    binding: (2 + MAX_INPUTS + i) as u32,
                    resource: wgpu::BindingResource::TextureView(view),
                });
            }
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("graph node"), layout: &self.bgl, entries: &entries });
            let surface_bg = surface.filter(|_| def.surface).map(|s| {
                use wgpu::util::DeviceExt;
                let su = SurfaceUniform {
                    bmin: [s.bmin[0], s.bmin[1], s.bmin[2], 0.0],
                    bsize: [s.bsize[0], s.bsize[1], s.bsize[2], 0.0],
                    info: [s.edge_max, 0.0, 0.0, 0.0],
                };
                let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("graph surface"),
                    contents: bytemuck::bytes_of(&su),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                fn tex(t: &Tex) -> wgpu::BindingResource<'_> {
                    wgpu::BindingResource::TextureView(&t.view)
                }
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("graph surface"),
                    layout: &self.surface_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: tex(s.pos) },
                        wgpu::BindGroupEntry { binding: 2, resource: tex(s.nrm) },
                        wgpu::BindGroupEntry { binding: 3, resource: tex(s.maps[0]) },
                        wgpu::BindGroupEntry { binding: 4, resource: tex(s.maps[1]) },
                        wgpu::BindGroupEntry { binding: 5, resource: tex(s.maps[2]) },
                        wgpu::BindGroupEntry { binding: 6, resource: tex(s.maps[3]) },
                        wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Sampler(&self.clamp_sampler) },
                    ],
                })
            });
            let mut pass = gpu::pass(encoder, "graph node", &[gpu::color_attachment(&out.view, wgpu::LoadOp::Clear(wgpu::Color::BLACK))]);
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            if let Some(sbg) = &surface_bg {
                pass.set_bind_group(1, sbg, &[]);
            }
            pass.draw(0..3, 0..1);
        }
        Ok(Some(out))
    }
}
