//! Layer thumbnails for the Layers panel: what each fill layer's content
//! and each mask really look like, in UV space.
//!
//! A thumbnail runs the same composite shader as the texture set, on the
//! layer alone, at thumbnail size. The composite reads the position and
//! normal maps (and a painted mask) texel by texel, so it gets copies of
//! them shrunk to that size: everything else it reads (mesh maps, fill
//! images) by UV, which works at any size. A mask with filters or mask
//! layers is already rendered at full size for the composite; its thumbnail
//! is that result, shrunk.
//!
//! A layer's thumbnails are rendered again only when what they show
//! changed: its settings (the composite uniform), images, mask stack, the
//! baked maps, or a painting stroke on it.

use super::*;
use std::hash::{Hash, Hasher};

/// Thumbnail size in texels.
pub const THUMB: u32 = 128;
/// Size the layer is composited at for its thumbnail, then averaged down:
/// one sample per thumbnail texel turns thin features (worn edges, scratches)
/// into scattered dots.
const RENDER: u32 = 512;
/// Samples per axis when shrinking a full-size texture.
const SHRINK_TAPS: f32 = 6.0;

#[derive(Default)]
pub(super) struct Thumbs {
    gpu: Option<ThumbGpu>,
    /// Position and normal maps at thumbnail size, for this `uv_version`.
    small: Option<(u64, Tex, Tex)>,
    entries: HashMap<u64, Entry>,
    /// Thumbnail images rendered so far, content and mask counted apart (the F6 meter shows how many a stroke cost).
    renders: u32,
}

struct ThumbGpu {
    bgl: wgpu::BindGroupLayout,
    nearest_pos: wgpu::RenderPipeline,
    nearest_nrm: wgpu::RenderPipeline,
    shrink_mask: wgpu::RenderPipeline,
    show: wgpu::RenderPipeline,
    /// Thumbnail-sized composite targets: a cleared stack below, and the result.
    below: [Tex; ACCUM],
    out: [Tex; ACCUM],
    /// A painted mask, shrunk; a mask's value.
    mask_paint: Tex,
    mask_value: Tex,
}

struct Entry {
    /// What the thumbnails were rendered from (0: render again).
    key: u64,
    /// A stroke changed this layer's mask and nothing else: the content
    /// thumbnail is still right (a fill's costs a full composite), so only
    /// the mask's is rendered again.
    mask_stale: bool,
    content: Option<Shown>,
    mask: Option<Shown>,
}

struct Shown {
    tex: Tex,
    id: egui::TextureId,
}

impl ThumbGpu {
    fn new(device: &wgpu::Device) -> ThumbGpu {
        let module = gpu::shader(device, "layer thumbnails", include_str!("../shaders/thumbs.wgsl"));
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("layer thumbnails"),
            entries: &[gpu::tex_entry(0, true), gpu::sampler_entry(1), gpu::uniform_entry(2, false), gpu::tex_entry(3, false)],
        });
        let layout = gpu::layout(device, &bgl);
        let pipe = |entry: &str, format: F| gpu::fullscreen_pipeline(device, "layer thumbnail", &module, entry, &layout, &[gpu::target(format)]);
        let t = |label: &str, f: F| Tex::new(device, label, RENDER, RENDER, f, LAYER_USAGE);
        ThumbGpu {
            nearest_pos: pipe("fs_nearest", POS_FORMAT),
            nearest_nrm: pipe("fs_nearest", NRM_FORMAT),
            shrink_mask: pipe("fs_box", MASK_FORMAT),
            show: pipe("fs_show", F::Rgba8Unorm),
            below: ACCUM_FORMATS.map(|f| t("thumb below", f)),
            out: ACCUM_FORMATS.map(|f| t("thumb layer", f)),
            mask_paint: t("thumb mask paint", MASK_FORMAT),
            mask_value: t("thumb mask", MASK_FILTER_FORMAT),
            bgl,
        }
    }
}

impl Thumbs {
    /// GPU texture memory owned by the thumbnail system.
    pub(super) fn vram_bytes(&self) -> u64 {
        let gpu = self.gpu.as_ref().map_or(0, |g| {
            g.below.iter().chain(g.out.iter()).map(|t| t.bytes()).sum::<u64>() + g.mask_paint.bytes() + g.mask_value.bytes()
        });
        let small = self.small.as_ref().map_or(0, |(_, pos, nrm)| pos.bytes() + nrm.bytes());
        let shown = self
            .entries
            .values()
            .flat_map(|entry| entry.content.iter().chain(entry.mask.iter()))
            .map(|shown| shown.tex.bytes())
            .sum::<u64>();
        gpu + small + shown
    }
}

impl Engine {
    /// The Layers panel image of a fill layer's content.
    pub fn content_thumb(&self, layer_id: u64) -> Option<egui::TextureId> {
        self.thumbs.entries.get(&layer_id)?.content.as_ref().map(|s| s.id)
    }

    /// The Layers panel image of a layer's mask.
    pub fn mask_thumb(&self, layer_id: u64) -> Option<egui::TextureId> {
        self.thumbs.entries.get(&layer_id)?.mask.as_ref().map(|s| s.id)
    }

    /// How many thumbnail images (content and mask counted apart) have been rendered so far.
    pub fn thumb_renders(&self) -> u32 {
        self.thumbs.renders
    }

    /// A layer's thumbnail textures: content and mask (tests read them back).
    #[cfg(test)]
    pub fn thumb_textures(&self, layer_id: u64) -> (Option<&Tex>, Option<&Tex>) {
        let e = self.thumbs.entries.get(&layer_id);
        (e.and_then(|e| e.content.as_ref()).map(|s| &s.tex), e.and_then(|e| e.mask.as_ref()).map(|s| &s.tex))
    }

    /// Renders every thumbnail again on the next update (after changes the
    /// keys can't see: undo and redo, finished strokes).
    pub(super) fn invalidate_thumbs(&mut self) {
        for e in self.thumbs.entries.values_mut() {
            e.key = 0;
        }
    }

    /// A finished stroke on a layer's mask: its mask thumbnail is rendered
    /// again on the next update, the content thumbnail is kept.
    pub(super) fn invalidate_mask_thumb(&mut self, layer_id: u64) {
        if let Some(e) = self.thumbs.entries.get_mut(&layer_id) {
            e.mask_stale = true;
        }
    }

    /// What a layer's thumbnails show depends on (see the module docs).
    fn thumb_key(&self, li: usize) -> u64 {
        let layer = &self.layers[li];
        let mut h = std::collections::hash_map::DefaultHasher::new();
        let mut u = self.layer_uniform(layer);
        // Opacity and blend don't change what the layer itself looks like.
        u.p0[0] = 1.0;
        u.p0[1] = 0.0;
        bytemuck::bytes_of(&u).hash(&mut h);
        for im in &layer.images {
            im.as_ref().map(|im| Arc::as_ptr(&im.tex) as usize).hash(&mut h);
        }
        if let Some(m) = &layer.mask {
            format!("{:?}{:?}", m.filters, m.layers).hash(&mut h);
            m.paint.is_some().hash(&mut h);
        }
        (self.maps_version, self.uv_version, self.has_mask_filters(li), self.mask_out.contains_key(&layer.id)).hash(&mut h);
        self.custom_ids.hash(&mut h);
        h.finish().max(1)
    }

    /// Brings the thumbnails up to date with the stack; call after compositing.
    pub(super) fn update_thumbs(&mut self) {
        // Forget layers that are gone.
        let ids: std::collections::HashSet<u64> = self.layers.iter().map(|l| l.id).collect();
        let gone: Vec<u64> = self.thumbs.entries.keys().copied().filter(|id| !ids.contains(id)).collect();
        for id in gone {
            if let Some(e) = self.thumbs.entries.remove(&id) {
                self.free_shown(e.content);
                self.free_shown(e.mask);
            }
        }
        // Nothing a thumbnail shows changes mid-stroke, and rebuilding them
        // every paint frame (a fill's is a full 512² composite) stalls the
        // GPU. end_stroke() marks what the stroke touched; it renders once on release.
        if self.active.is_some() {
            return;
        }
        let todo: Vec<(usize, u64, bool)> = (0..self.layers.len())
            .filter_map(|li| {
                let layer = &self.layers[li];
                let wants = matches!(layer.content, Content::Fill(_)) || layer.mask.is_some();
                if !wants {
                    return None;
                }
                let key = self.thumb_key(li);
                let entry = self.thumbs.entries.get(&layer.id);
                let current = entry.is_some_and(|e| e.key == key);
                let mask_only = current && entry.is_some_and(|e| e.mask_stale);
                (!current || mask_only).then_some((li, key, mask_only))
            })
            .collect();
        if todo.is_empty() {
            return;
        }
        if self.thumbs.gpu.is_none() {
            self.thumbs.gpu = Some(ThumbGpu::new(&self.device));
        }
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.shrink_surface(&mut encoder);
        for (li, key, mask_only) in todo {
            self.render_thumbs(&mut encoder, li, key, mask_only);
        }
        self.queue.submit([encoder.finish()]);
    }

    fn free_shown(&self, shown: Option<Shown>) {
        if let Some(s) = shown {
            self.rs.renderer.write().free_texture(&s.id);
        }
    }

    /// A pass of one of the thumbnail shaders from `src` into `dst`.
    fn thumb_pass(&self, encoder: &mut wgpu::CommandEncoder, pipeline: &wgpu::RenderPipeline, src: &Tex, dst: &Tex, show: [f32; 4]) {
        // Island coverage for `fs_show`: the shrunk position map once there is one.
        let cover = self.thumbs.small.as_ref().map_or(&self.pos, |(_, pos, _)| pos);
        let g = self.thumbs.gpu.as_ref().expect("thumbnail pipelines");
        let ubuf = wgpu::util::DeviceExt::create_buffer_init(
            &self.device,
            &wgpu::util::BufferInitDescriptor { label: Some("thumbnail"), contents: bytemuck::bytes_of(&show), usage: wgpu::BufferUsages::UNIFORM },
        );
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("layer thumbnail"),
            layout: &g.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src.view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&cover.view) },
            ],
        });
        let mut pass = gpu::pass(encoder, "layer thumbnail", &[gpu::color_attachment(&dst.view, wgpu::LoadOp::Clear(wgpu::Color::BLACK))]);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }

    /// The position and normal maps at the size thumbnails render at, made once per UV layout.
    fn shrink_surface(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if self.thumbs.small.as_ref().is_some_and(|(v, _, _)| *v == self.uv_version) {
            return;
        }
        let pos = Tex::new(&self.device, "thumb position", RENDER, RENDER, POS_FORMAT, LAYER_USAGE);
        let nrm = Tex::new(&self.device, "thumb normal", RENDER, RENDER, NRM_FORMAT, LAYER_USAGE);
        let g = self.thumbs.gpu.as_ref().expect("thumbnail pipelines");
        self.thumb_pass(encoder, &g.nearest_pos, &self.pos, &pos, [0.0; 4]);
        self.thumb_pass(encoder, &g.nearest_nrm, &self.nrm, &nrm, [0.0; 4]);
        self.thumbs.small = Some((self.uv_version, pos, nrm));
    }

    /// A display texture for a thumbnail, registered with egui once.
    fn shown_tex(&mut self, layer_id: u64, mask: bool) -> &Tex {
        let entry = self.thumbs.entries.entry(layer_id).or_insert(Entry { key: 0, mask_stale: false, content: None, mask: None });
        let slot = if mask { &mut entry.mask } else { &mut entry.content };
        if slot.is_none() {
            let tex = Tex::new(&self.device, "layer thumbnail", THUMB, THUMB, F::Rgba8Unorm, LAYER_USAGE);
            let id = self.rs.renderer.write().register_native_texture(&self.device, tex.srgb(), wgpu::FilterMode::Linear);
            *slot = Some(Shown { tex, id });
        }
        &slot.as_ref().expect("just made").tex
    }

    /// `mask_only`: the content thumbnail is current, redo just the mask's.
    fn render_thumbs(&mut self, encoder: &mut wgpu::CommandEncoder, li: usize, key: u64, mask_only: bool) {
        let mut rendered = 0u32;
        let layer_id = self.layers[li].id;
        let is_fill = matches!(self.layers[li].content, Content::Fill(_));
        let has_mask = self.layers[li].mask.is_some();
        // Make the display textures first (they borrow the engine mutably).
        if is_fill {
            self.shown_tex(layer_id, false);
        }
        if has_mask {
            self.shown_tex(layer_id, true);
        }
        let layer = &self.layers[li];
        let g = self.thumbs.gpu.as_ref().expect("thumbnail pipelines");
        let (_, small_pos, small_nrm) = self.thumbs.small.as_ref().expect("shrunk above");
        let entry = &self.thumbs.entries[&layer_id];

        // One uniform slot for the content, one for the mask.
        let mut content_u = self.layer_uniform(layer);
        content_u.p0 = [1.0, 0.0, content_u.p0[2], 0.0];
        content_u.gp1[2] = 0.0;
        let mask_u = self.layer_uniform(layer);
        let mut data = vec![0u8; UNIFORM_STRIDE as usize * 2];
        data[..size_of::<LayerUniform>()].copy_from_slice(bytemuck::bytes_of(&content_u));
        let off = UNIFORM_STRIDE as usize;
        data[off..off + size_of::<LayerUniform>()].copy_from_slice(bytemuck::bytes_of(&mask_u));
        let ubuf = wgpu::util::DeviceExt::create_buffer_init(
            &self.device,
            &wgpu::util::BufferInitDescriptor { label: Some("thumbnail layer"), contents: &data, usage: wgpu::BufferUsages::UNIFORM },
        );
        fn or_dummy<'a>(t: Option<&'a Tex>, dummy: &'a wgpu::TextureView) -> wgpu::BindingResource<'a> {
            wgpu::BindingResource::TextureView(t.map_or(dummy, |t| &t.view))
        }
        let dummy = &self.dummy.view;
        let view = |t| or_dummy(t, dummy);
        let images: [Option<&Tex>; super::CHANNELS] = std::array::from_fn(|i| layer.images[i].as_ref().map(|im| im.tex.as_ref()));
        let painted = layer.mask.as_ref().and_then(|m| m.paint.as_ref());
        if painted.is_some() {
            self.thumb_pass(encoder, &g.shrink_mask, painted.unwrap(), &g.mask_paint, [0.0, SHRINK_TAPS, RENDER as f32, 0.0]);
        }
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("thumbnail composite"),
            layout: &self.p.composite_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &ubuf,
                        offset: 0,
                        size: NonZeroU64::new(size_of::<LayerUniform>() as u64),
                    }),
                },
                wgpu::BindGroupEntry { binding: 1, resource: view(Some(&g.below[0])) },
                wgpu::BindGroupEntry { binding: 2, resource: view(Some(&g.below[1])) },
                wgpu::BindGroupEntry { binding: 3, resource: view(Some(&g.below[2])) },
                wgpu::BindGroupEntry { binding: 4, resource: view(Some(&g.below[3])) },
                wgpu::BindGroupEntry { binding: 5, resource: view(images[0]) },
                wgpu::BindGroupEntry { binding: 6, resource: view(images[1]) },
                wgpu::BindGroupEntry { binding: 7, resource: view(images[2]) },
                wgpu::BindGroupEntry { binding: 8, resource: view(images[3]) },
                wgpu::BindGroupEntry { binding: 9, resource: view(painted.map(|_| &g.mask_paint)) },
                wgpu::BindGroupEntry { binding: 10, resource: view(Some(small_pos)) },
                wgpu::BindGroupEntry { binding: 11, resource: view(Some(small_nrm)) },
                wgpu::BindGroupEntry { binding: 12, resource: view(Some(self.map_tex(0))) },
                wgpu::BindGroupEntry { binding: 13, resource: view(Some(self.map_tex(1))) },
                wgpu::BindGroupEntry { binding: 14, resource: view(Some(self.map_tex(2))) },
                wgpu::BindGroupEntry { binding: 15, resource: wgpu::BindingResource::TextureView(&self.custom_array.1) },
                wgpu::BindGroupEntry { binding: 16, resource: view(Some(self.map_tex(4))) },
                wgpu::BindGroupEntry { binding: 17, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 18, resource: wgpu::BindingResource::Sampler(&self.repeat_sampler) },
                wgpu::BindGroupEntry { binding: 19, resource: view(images[super::OPACITY]) },
                wgpu::BindGroupEntry { binding: 20, resource: view(Some(&g.below[super::FINAL_EMISSION])) },
                wgpu::BindGroupEntry { binding: 21, resource: view(images[super::EMISSION]) },
                wgpu::BindGroupEntry { binding: 22, resource: view(Some(&g.below[super::FINAL_NORMAL])) },
                wgpu::BindGroupEntry { binding: 23, resource: view(images[super::NORMAL]) },
                wgpu::BindGroupEntry { binding: 24, resource: view(Some(&g.below[super::FINAL_SPECULAR])) },
                wgpu::BindGroupEntry { binding: 25, resource: view(images[super::SPECULAR]) },
                // Full size, unlike the thumbnail's position and normal: it
                // only turns normal images, which thumbnails don't show.
                wgpu::BindGroupEntry { binding: 26, resource: view(Some(&self.tan)) },
            ],
        });

        // Content: the fill alone over a neutral stack, showing its first
        // channel that's on (color, else roughness, metallic or height).
        if let (false, Content::Fill(f), Some(shown)) = (mask_only, &layer.content, entry.content.as_ref()) {
            {
                let c = |v: f64| wgpu::LoadOp::Clear(wgpu::Color { r: v, g: v, b: v, a: 1.0 });
                let _clear = gpu::pass(
                    encoder,
                    "thumb clear",
                    &[
                        gpu::color_attachment(&g.below[0].view, c(0.5)),
                        gpu::color_attachment(&g.below[1].view, c(0.5)),
                        gpu::color_attachment(&g.below[2].view, c(0.0)),
                        gpu::color_attachment(&g.below[3].view, c(0.0)),
                        gpu::color_attachment(&g.below[4].view, c(0.0)),
                        gpu::color_attachment(&g.below[5].view, wgpu::LoadOp::Clear(wgpu::Color { r: 0.5, g: 0.5, b: 1.0, a: 1.0 })),
                        gpu::color_attachment(&g.below[6].view, c(0.5)),
                    ],
                );
            }
            {
                let clear = wgpu::LoadOp::Clear(wgpu::Color::BLACK);
                let att: Vec<_> = g.out.iter().map(|t| gpu::color_attachment(&t.view, clear)).collect();
                let mut pass = gpu::pass(encoder, "thumb layer", &att);
                pass.set_pipeline(&self.p.composite);
                pass.set_bind_group(0, &bg, &[0]);
                pass.draw(0..3, 0..1);
            }
            let channel = (0..4).find(|&c| f.mat.enabled[c] || layer.images[c].is_some()).unwrap_or(0);
            let what = match channel {
                0 => 0.0,
                3 => 2.0,
                _ => 1.0,
            };
            self.thumb_pass(encoder, &g.show, &g.out[channel], &shown.tex, [what, (RENDER / THUMB) as f32, THUMB as f32, 0.0]);
            rendered += 1;
        }

        // Mask: the filtered result the composite made, shrunk, or else the
        // mask (painting times smart mask) rendered at thumbnail size.
        if let Some(shown) = entry.mask.as_ref() {
            let filtered = self.mask_final.get(&layer_id).and_then(|&i| self.mask_out.get(&layer_id).map(|t| &t[i]));
            match filtered {
                Some(full) => self.thumb_pass(encoder, &g.show, full, &shown.tex, [1.0, SHRINK_TAPS, THUMB as f32, 0.0]),
                None => {
                    {
                        let mut pass = gpu::pass(encoder, "thumb mask", &[gpu::color_attachment(&g.mask_value.view, wgpu::LoadOp::Clear(wgpu::Color::BLACK))]);
                        pass.set_pipeline(&self.p.mask_source);
                        pass.set_bind_group(0, &bg, &[UNIFORM_STRIDE as u32]);
                        pass.draw(0..3, 0..1);
                    }
                    self.thumb_pass(encoder, &g.show, &g.mask_value, &shown.tex, [1.0, (RENDER / THUMB) as f32, THUMB as f32, 0.0]);
                }
            }
            rendered += 1;
        }
        self.thumbs.renders = self.thumbs.renders.wrapping_add(rendered);
        let entry = self.thumbs.entries.get_mut(&layer_id).expect("made above");
        entry.key = key;
        entry.mask_stale = false;
    }
}
