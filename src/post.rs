//! Viewport post-processing: bloom, then one pass that applies exposure,
//! white balance, tone mapping, grading and lens effects to an HDR image.

use crate::gpu::{self, Tex};
use eframe::{egui_wgpu, wgpu};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tonemap {
    Aces,
    Agx,
    Neutral,
    Filmic,
    Linear,
}

impl Tonemap {
    pub const ALL: [Tonemap; 5] = [Tonemap::Aces, Tonemap::Agx, Tonemap::Neutral, Tonemap::Filmic, Tonemap::Linear];

    pub fn name(self) -> &'static str {
        match self {
            Tonemap::Aces => "ACES",
            Tonemap::Agx => "AgX-style",
            Tonemap::Neutral => "Neutral",
            Tonemap::Filmic => "Filmic",
            Tonemap::Linear => "Linear (clip)",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            Tonemap::Aces => "Punchy filmic contrast; saturated highlights shift hue",
            Tonemap::Agx => "Soft highlights that fade to white; good for bright scenes",
            Tonemap::Neutral => "Keeps base colors true to the texture; best for judging albedo",
            Tonemap::Filmic => "Gentle shoulder, lifted shadows",
            Tonemap::Linear => "No curve; anything above 1 clips",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PostSettings {
    pub tonemap: Tonemap,
    pub contrast: f32,
    pub saturation: f32,
    /// -1 (cool) .. 1 (warm).
    pub temperature: f32,
    /// -1 (green) .. 1 (magenta).
    pub tint: f32,
    pub vignette: f32,
    pub bloom: f32,
    /// Scene brightness (its strongest channel) where bloom starts.
    pub bloom_threshold: f32,
    pub chromatic: f32,
    pub grain: f32,
    pub sharpen: f32,
}

impl Default for PostSettings {
    fn default() -> Self {
        Self {
            tonemap: Tonemap::Aces,
            contrast: 1.0,
            saturation: 1.0,
            temperature: 0.0,
            tint: 0.0,
            // Effects start off: the view shows the material as it is.
            vignette: 0.0,
            bloom: 0.0,
            bloom_threshold: 1.0,
            chromatic: 0.0,
            grain: 0.0,
            sharpen: 0.0,
        }
    }
}

/// Scene-referred format the viewport renders into before post.
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const MAX_BLOOM_LEVELS: usize = 6;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PostU {
    a: [f32; 4],
    b: [f32; 4],
    c: [f32; 4],
    d: [f32; 4],
}

struct BloomChain {
    size: [u32; 2],
    levels: Vec<Tex>,
    /// One per down pass, then one per up pass.
    uniforms: Vec<wgpu::Buffer>,
    /// These resources are stable until the chain is resized.
    down_groups: Vec<wgpu::BindGroup>,
    up_groups: Vec<Option<wgpu::BindGroup>>,
    uniforms_initialized: bool,
    threshold_bits: Option<u32>,
}

pub struct PostFx {
    post: wgpu::RenderPipeline,
    down: wgpu::RenderPipeline,
    up: wgpu::RenderPipeline,
    post_bgl: wgpu::BindGroupLayout,
    bloom_bgl: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    sampler: wgpu::Sampler,
    black: Tex,
    chain: Option<BloomChain>,
    post_bind_group: Option<wgpu::BindGroup>,
    post_bind_group_key: Option<([u32; 2], bool)>,
}

impl PostFx {
    pub fn new(rs: &egui_wgpu::RenderState, output: wgpu::TextureFormat) -> PostFx {
        let device = &rs.device;
        let post_module = gpu::shader(device, "post", include_str!("shaders/post.wgsl"));
        let bloom_module = gpu::shader(device, "bloom", include_str!("shaders/bloom.wgsl"));
        let post_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post"),
            entries: &[gpu::uniform_entry(0, false), gpu::tex_entry(1, true), gpu::tex_entry(2, true), gpu::sampler_entry(3)],
        });
        let bloom_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom"),
            entries: &[gpu::uniform_entry(0, false), gpu::tex_entry(1, true), gpu::sampler_entry(2)],
        });
        let post = gpu::fullscreen_pipeline(device, "post", &post_module, "fs_post", &gpu::layout(device, &post_bgl), &[gpu::target(output)]);
        let bloom_layout = gpu::layout(device, &bloom_bgl);
        let down = gpu::fullscreen_pipeline(device, "bloom down", &bloom_module, "fs_down", &bloom_layout, &[gpu::target(HDR_FORMAT)]);
        let additive = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add };
        let up = gpu::fullscreen_pipeline(
            device,
            "bloom up",
            &bloom_module,
            "fs_up",
            &bloom_layout,
            &[Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                blend: Some(wgpu::BlendState { color: additive, alpha: additive }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        );
        let black = Tex::new(device, "post black", 1, 1, HDR_FORMAT, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        rs.queue.write_texture(black.copy_info(), &[0u8; 8], wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(8), rows_per_image: None }, black.extent());
        PostFx {
            post,
            down,
            up,
            post_bgl,
            bloom_bgl,
            uniform: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("post uniform"),
                size: size_of::<PostU>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            sampler: gpu::linear_sampler(device, wgpu::AddressMode::ClampToEdge),
            black,
            chain: None,
            post_bind_group: None,
            post_bind_group_key: None,
        }
    }

    fn ensure_chain(&mut self, device: &wgpu::Device, size: [u32; 2], input: &wgpu::TextureView) {
        if self.chain.as_ref().is_some_and(|c| c.size == size) {
            return;
        }
        let usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT;
        let mut levels = Vec::new();
        let (mut w, mut h) = (size[0] / 2, size[1] / 2);
        while levels.len() < MAX_BLOOM_LEVELS && w >= 8 && h >= 8 {
            levels.push(Tex::new(device, "bloom", w, h, HDR_FORMAT, usage));
            w /= 2;
            h /= 2;
        }
        let uniforms: Vec<_> = (0..levels.len() * 2)
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("bloom uniform"),
                    size: 16,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();

        // All source views and uniform buffers are stable until a resize.
        // Build these bind groups once instead of once per bloom pass per frame.
        let mut down_groups = Vec::with_capacity(levels.len());
        for i in 0..levels.len() {
            let src = if i == 0 { input } else { &levels[i - 1].view };
            down_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bloom down"),
                layout: &self.bloom_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: uniforms[i].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(src) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            }));
        }
        let mut up_groups: Vec<Option<wgpu::BindGroup>> = (0..levels.len()).map(|_| None).collect();
        for i in 1..levels.len() {
            up_groups[i] = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bloom up"),
                layout: &self.bloom_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: uniforms[levels.len() + i].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&levels[i].view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            }));
        }
        self.chain = Some(BloomChain {
            size, levels, uniforms, down_groups, up_groups,
            uniforms_initialized: false,
            threshold_bits: None,
        });
        self.post_bind_group = None;
        self.post_bind_group_key = None;
    }

    /// The viewport replaces its input/output textures when its size changes.
    pub fn invalidate_targets(&mut self) {
        self.chain = None;
        self.post_bind_group = None;
        self.post_bind_group_key = None;
    }

    /// Records bloom and the final pass from `input` (HDR, premultiplied) into
    /// `output`. `raw` passes the image through untouched (data view modes).
    pub fn record(
        &mut self,
        rs: &egui_wgpu::RenderState,
        encoder: &mut wgpu::CommandEncoder,
        input: &Tex,
        output: &wgpu::TextureView,
        settings: &PostSettings,
        exposure: f32,
        raw: bool,
    ) {
        let device = &rs.device;
        let size = [input.width, input.height];
        let bloom_on = !raw && settings.bloom > 0.0;
        if bloom_on {
            self.ensure_chain(device, size, &input.view);
        }
        let bloom_view = match self.chain.as_mut() {
            Some(chain) if bloom_on && !chain.levels.is_empty() => {
                let n = chain.levels.len();
                let threshold_bits = settings.bloom_threshold.to_bits();
                if !chain.uniforms_initialized {
                    // Texel sizes and first-pass flags are fixed for this chain.
                    // Only the threshold can change while the chain remains valid.
                    for i in 0..n {
                        let src = if i == 0 { input } else { &chain.levels[i - 1] };
                        let u = [
                            1.0 / src.width as f32,
                            1.0 / src.height as f32,
                            if i == 0 { settings.bloom_threshold } else { 0.0 },
                            if i == 0 { 1.0 } else { 0.0 },
                        ];
                        rs.queue.write_buffer(&chain.uniforms[i], 0, bytemuck::cast_slice(&u));
                    }
                    for i in 1..n {
                        let src = &chain.levels[i];
                        let u = [1.0 / src.width as f32, 1.0 / src.height as f32, 0.0, 0.0];
                        rs.queue.write_buffer(&chain.uniforms[n + i], 0, bytemuck::cast_slice(&u));
                    }
                    chain.uniforms_initialized = true;
                    chain.threshold_bits = Some(threshold_bits);
                } else if chain.threshold_bits != Some(threshold_bits) {
                    let u = [
                        1.0 / input.width as f32,
                        1.0 / input.height as f32,
                        settings.bloom_threshold,
                        1.0,
                    ];
                    rs.queue.write_buffer(&chain.uniforms[0], 0, bytemuck::cast_slice(&u));
                    chain.threshold_bits = Some(threshold_bits);
                }

                for i in 0..n {
                    let mut pass = gpu::pass(encoder, "bloom down", &[gpu::color_attachment(&chain.levels[i].view, wgpu::LoadOp::Clear(wgpu::Color::BLACK))]);
                    pass.set_pipeline(&self.down);
                    pass.set_bind_group(0, &chain.down_groups[i], &[]);
                    pass.draw(0..3, 0..1);
                }
                for i in (1..n).rev() {
                    let bg = chain.up_groups[i].as_ref().expect("bloom up bind group");
                    let mut pass = gpu::pass(encoder, "bloom up", &[gpu::color_attachment(&chain.levels[i - 1].view, wgpu::LoadOp::Load)]);
                    pass.set_pipeline(&self.up);
                    pass.set_bind_group(0, bg, &[]);
                    pass.draw(0..3, 0..1);
                }
                Some((&chain.levels[0].view, n as f32))
            }
            _ => None,
        };
        let (bloom_tex, levels) = bloom_view.unwrap_or((&self.black.view, 1.0));

        let u = PostU {
            a: [exposure, settings.tonemap as u32 as f32, settings.contrast, settings.saturation],
            b: [settings.temperature, settings.tint, settings.vignette, settings.bloom / levels],
            c: [settings.chromatic, settings.grain, settings.sharpen, if raw { 1.0 } else { 0.0 }],
            d: [1.0 / size[0] as f32, 1.0 / size[1] as f32, size[0] as f32 / size[1] as f32, 0.0],
        };
        rs.queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
        let post_key = (size, bloom_view.is_some());
        if self.post_bind_group_key != Some(post_key) || self.post_bind_group.is_none() {
            self.post_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("post"),
                layout: &self.post_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&input.view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(bloom_tex) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            }));
            self.post_bind_group_key = Some(post_key);
        }
        let bg = self.post_bind_group.as_ref().expect("post bind group");
        let mut pass = gpu::pass(encoder, "post", &[gpu::color_attachment(output, wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT))]);
        pass.set_pipeline(&self.post);
        pass.set_bind_group(0, bg, &[]);
        pass.draw(0..3, 0..1);
    }
}
