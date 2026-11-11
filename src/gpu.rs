//! Small wgpu helpers shared by the engine and the viewport renderer.

use eframe::wgpu;

/// A texture and its views. Clones share the GPU texture.
#[derive(Clone)]
pub struct Tex {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    /// sRGB-decoding view, only for Rgba8Unorm textures holding sRGB-encoded color.
    pub srgb_view: Option<wgpu::TextureView>,
    pub format: wgpu::TextureFormat,
    pub width: u32,
    pub height: u32,
    pub samples: u32,
    /// Number of allocated mip levels (1 for ordinary render targets).
    pub mip_levels: u32,
}

pub const LAYER_USAGE: wgpu::TextureUsages = wgpu::TextureUsages::RENDER_ATTACHMENT
    .union(wgpu::TextureUsages::TEXTURE_BINDING)
    .union(wgpu::TextureUsages::COPY_SRC)
    .union(wgpu::TextureUsages::COPY_DST);

impl Tex {
    pub fn new(
        device: &wgpu::Device,
        label: &str,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> Tex {
        Self::new_ms(device, label, width, height, format, usage, 1)
    }

    pub fn new_ms(
        device: &wgpu::Device,
        label: &str,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
        samples: u32,
    ) -> Tex {
        let srgb = format == wgpu::TextureFormat::Rgba8Unorm;
        let view_formats: &[wgpu::TextureFormat] =
            if srgb { &[wgpu::TextureFormat::Rgba8UnormSrgb] } else { &[] };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats,
        });
        let view = texture.create_view(&Default::default());
        let srgb_view = srgb.then(|| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
                // sRGB formats can't be storage-bound; the linear view covers that.
                usage: Some(usage - wgpu::TextureUsages::STORAGE_BINDING),
                ..Default::default()
            })
        });
        Tex { texture, view, srgb_view, format, width, height, samples, mip_levels: 1 }
    }

    pub fn bytes(&self) -> u64 {
        texture_allocation_bytes(
            self.width,
            self.height,
            bytes_per_texel(self.format),
            self.mip_levels,
            self.samples,
        )
    }

    pub fn copy_info(&self) -> wgpu::TexelCopyTextureInfo<'_> {
        wgpu::TexelCopyTextureInfo {
            texture: &self.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        }
    }

    pub fn extent(&self) -> wgpu::Extent3d {
        wgpu::Extent3d { width: self.width, height: self.height, depth_or_array_layers: 1 }
    }

    pub fn srgb(&self) -> &wgpu::TextureView {
        self.srgb_view.as_ref().unwrap_or(&self.view)
    }
}

pub fn bytes_per_texel(format: wgpu::TextureFormat) -> u32 {
    use wgpu::TextureFormat as F;
    match format {
        F::R8Unorm => 1,
        F::Rg8Unorm | F::R16Float => 2,
        F::Rgba8Unorm | F::Rgba8UnormSrgb | F::Rg16Float | F::R32Float | F::Depth32Float => 4,
        F::Rgba16Float => 8,
        F::Rgba32Float => 16,
        _ => 4,
    }
}

fn texture_allocation_bytes(width: u32, height: u32, bytes_per_pixel: u32, mip_levels: u32, samples: u32) -> u64 {
    let mut width = width as u64;
    let mut height = height as u64;
    let bpp = bytes_per_pixel as u64;
    let mut total = 0u64;
    for _ in 0..mip_levels {
        total = total.saturating_add(width.saturating_mul(height).saturating_mul(bpp));
        width = (width / 2).max(1);
        height = (height / 2).max(1);
    }
    total.saturating_mul(samples as u64)
}

pub fn clear_texture(encoder: &mut wgpu::CommandEncoder, tex: &Tex, color: wgpu::Color) {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("clear"),
        color_attachments: &[color_attachment(&tex.view, wgpu::LoadOp::Clear(color))],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
}

pub fn color_attachment(
    view: &wgpu::TextureView,
    load: wgpu::LoadOp<wgpu::Color>,
) -> Option<wgpu::RenderPassColorAttachment<'_>> {
    Some(wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
    })
}

pub fn pass<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    label: &str,
    attachments: &[Option<wgpu::RenderPassColorAttachment<'_>>],
) -> wgpu::RenderPass<'a> {
    encoder
        .begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
        .forget_lifetime()
}

pub fn tex_entry(binding: u32, filterable: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

pub fn uniform_entry(binding: u32, dynamic: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: dynamic,
            min_binding_size: None,
        },
        count: None,
    }
}

pub fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

pub fn shader(device: &wgpu::Device, label: &str, src: &str) -> wgpu::ShaderModule {
    let full = format!("{}\n{}", include_str!("shaders/common.wgsl"), src);
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(full.into()),
    })
}

/// An image as a sampled texture with a full mip chain (each level a
/// box-filtered half of the one above), for tiling images seen from afar.
pub fn upload_mips<P: image::Pixel<Subpixel = u8> + 'static>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    img: image::ImageBuffer<P, Vec<u8>>,
    format: wgpu::TextureFormat,
    label: &str,
) -> Tex {
    use image::imageops::FilterType;
    let (w, h) = img.dimensions();
    let mips = w.max(h).max(1).ilog2() + 1;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: mips,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let channels = P::CHANNEL_COUNT as u32;
    let mut level = img;
    for mip in 0..mips {
        let (lw, lh) = level.dimensions();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &texture, mip_level: mip, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &level,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(lw * channels), rows_per_image: None },
            wgpu::Extent3d { width: lw, height: lh, depth_or_array_layers: 1 },
        );
        if mip + 1 < mips {
            level = image::imageops::resize(&level, (lw / 2).max(1), (lh / 2).max(1), FilterType::Triangle);
        }
    }
    let view = texture.create_view(&Default::default());
    Tex { texture, view, srgb_view: None, format, width: w, height: h, samples: 1, mip_levels: mips }
}

pub fn layout(device: &wgpu::Device, bgl: &wgpu::BindGroupLayout) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(bgl)],
        immediate_size: 0,
    })
}

/// Fullscreen-triangle pipeline writing to the given targets.
pub fn fullscreen_pipeline(
    device: &wgpu::Device,
    label: &str,
    module: &wgpu::ShaderModule,
    fs_entry: &str,
    layout: &wgpu::PipelineLayout,
    targets: &[Option<wgpu::ColorTargetState>],
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs_fullscreen"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(fs_entry),
            compilation_options: Default::default(),
            targets,
        }),
        multiview_mask: None,
        cache: None,
    })
}

pub fn target(format: wgpu::TextureFormat) -> Option<wgpu::ColorTargetState> {
    Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })
}

pub fn linear_sampler(device: &wgpu::Device, address: wgpu::AddressMode) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("linear"),
        address_mode_u: address,
        address_mode_v: address,
        address_mode_w: address,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    })
}

pub fn vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRS: [wgpu::VertexAttribute; 4] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x4];
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<crate::mesh::Vertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTRS,
    }
}

/// GPU copy of the current mesh, shared by the engine (UV bakes) and the viewport.
pub struct GpuMesh {
    pub vertices: wgpu::Buffer,
    pub indices: wgpu::Buffer,
    pub index_count: u32,
    pub edges: wgpu::Buffer,
    pub edge_count: u32,
    pub seams: wgpu::Buffer,
    pub seam_count: u32,
    /// The texture sets' ranges (see [`crate::mesh::MeshSet`]).
    pub sets: Vec<crate::mesh::MeshSet>,
}

impl GpuMesh {
    pub fn new(device: &wgpu::Device, mesh: &crate::mesh::Mesh) -> GpuMesh {
        use wgpu::util::DeviceExt;
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh vertices"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh indices"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let edges = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh uv edges"),
            contents: bytemuck::cast_slice(&mesh.uv_edges),
            usage: wgpu::BufferUsages::INDEX,
        });
        // (Never empty: a zero-size index buffer can't be bound.)
        let seam_data: &[u32] = if mesh.seam_edges.is_empty() { &[0, 0] } else { &mesh.seam_edges };
        let seams = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh uv seams"),
            contents: bytemuck::cast_slice(seam_data),
            usage: wgpu::BufferUsages::INDEX,
        });
        GpuMesh {
            vertices,
            indices,
            index_count: mesh.indices.len() as u32,
            edges,
            edge_count: mesh.uv_edges.len() as u32,
            seams,
            seam_count: mesh.seam_edges.len() as u32,
            sets: mesh.sets.clone(),
        }
    }
}

/// Read several textures in one submission and poll, decoding each mapped
/// buffer before moving to the next one. This avoids a GPU wait per texture
/// without retaining an extra raw-byte copy of every map at once.
pub fn read_wgpu_textures_with<T>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    textures: &[&wgpu::Texture],
    mut decode: impl FnMut(&wgpu::Texture, &[u8]) -> Result<T, String>,
) -> Result<Vec<T>, String> {
    struct Staging {
        buffer: wgpu::Buffer,
        row_bytes: u32,
        padded_row_bytes: u32,
        height: u32,
    }

    if textures.is_empty() {
        return Ok(Vec::new());
    }

    let mut encoder = device.create_command_encoder(&Default::default());
    let mut staging = Vec::with_capacity(textures.len());
    for texture in textures {
        let width = texture.width();
        let height = texture.height();
        let bpp = bytes_per_texel(texture.format());
        let row_bytes = width.checked_mul(bpp)
            .ok_or_else(|| "GPU texture readback row size overflow.".to_string())?;
        let padded_row_bytes = row_bytes.div_ceil(256).checked_mul(256)
            .ok_or_else(|| "GPU texture readback alignment overflow.".to_string())?;
        let buffer_size = (padded_row_bytes as u64).checked_mul(height as u64)
            .ok_or_else(|| "GPU texture readback buffer size overflow.".to_string())?;
        if width == 0 || height == 0 {
            return Err("Cannot read an empty GPU texture.".into());
        }
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("batched texture readback"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row_bytes),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        staging.push(Staging { buffer, row_bytes, padded_row_bytes, height });
    }

    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    for (index, item) in staging.iter().enumerate() {
        let tx = tx.clone();
        item.buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            let result = result.map_err(|error| format!("{error:?}"));
            let _ = tx.send((index, result));
        });
    }
    drop(tx);

    if let Err(error) = device.poll(wgpu::PollType::wait_indefinitely()) {
        for item in &staging {
            item.buffer.unmap();
        }
        return Err(format!("GPU texture readback did not complete: {error:?}"));
    }

    let mut mapped = vec![false; staging.len()];
    let mut failure = None;
    for _ in 0..staging.len() {
        match rx.recv() {
            Ok((index, Ok(()))) if index < mapped.len() => mapped[index] = true,
            Ok((_, Err(error))) => {
                if failure.is_none() {
                    failure = Some(format!("GPU texture mapping failed: {error}"));
                }
            }
            Ok(_) => {
                if failure.is_none() {
                    failure = Some("GPU texture mapping returned an invalid result.".into());
                }
            }
            Err(error) => {
                if failure.is_none() {
                    failure = Some(format!("GPU texture mapping callback stopped: {error}"));
                }
                break;
            }
        }
    }
    if let Some(error) = failure {
        for (item, is_mapped) in staging.iter().zip(mapped) {
            if is_mapped {
                item.buffer.unmap();
            }
        }
        return Err(error);
    }

    let mut output = Vec::with_capacity(textures.len());
    let mut decode_error = None;
    for (texture, item) in textures.iter().zip(&staging) {
        let view = match item.buffer.slice(..).get_mapped_range() {
            Ok(view) => view,
            Err(error) => {
                item.buffer.unmap();
                if decode_error.is_none() {
                    decode_error = Some(format!("GPU texture mapping could not be read: {error:?}"));
                }
                continue;
            }
        };
        let raw_size = (item.row_bytes as usize).checked_mul(item.height as usize);
        let mut raw = Vec::new();
        if let Some(size) = raw_size {
            if raw.try_reserve_exact(size).is_ok() {
                for y in 0..item.height as usize {
                    let start = y * item.padded_row_bytes as usize;
                    let end = start + item.row_bytes as usize;
                    raw.extend_from_slice(&view[start..end]);
                }
            } else if decode_error.is_none() {
                decode_error = Some("Cannot allocate the GPU texture readback buffer.".into());
            }
        } else if decode_error.is_none() {
            decode_error = Some("GPU texture readback size overflow.".into());
        }
        drop(view);
        item.buffer.unmap();

        if raw_size.is_some_and(|size| raw.len() == size) {
            match decode(texture, &raw) {
                Ok(value) => output.push(value),
                Err(error) if decode_error.is_none() => decode_error = Some(error),
                Err(_) => {}
            }
        }
    }

    if let Some(error) = decode_error {
        Err(error)
    } else if output.len() == textures.len() {
        Ok(output)
    } else {
        Err("GPU texture readback returned an incomplete result.".into())
    }
}

/// Blocking texture readback, used for export (an explicit user action).
pub fn read_texture(device: &wgpu::Device, queue: &wgpu::Queue, tex: &Tex) -> Vec<u8> {
    read_wgpu_texture(device, queue, &tex.texture)
}

/// Blocking readback of a 2D texture's first mip, tightly packed. Waits for
/// all submitted GPU work, so off the UI thread it suits background saves.
pub fn read_wgpu_texture(device: &wgpu::Device, queue: &wgpu::Queue, tex: &wgpu::Texture) -> Vec<u8> {
    let (width, height) = (tex.width(), tex.height());
    let bpp = bytes_per_texel(tex.format());
    let row = width * bpp;
    let padded = row.div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (padded * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        tex.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );
    queue.submit([encoder.finish()]);
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let data = slice.get_mapped_range().expect("readback buffer is mapped");
    let mut out = Vec::with_capacity((row * height) as usize);
    for y in 0..height as usize {
        let start = y * padded as usize;
        out.extend_from_slice(&data[start..start + row as usize]);
    }
    drop(data);
    buffer.unmap();
    out
}

/// A half float's bits for `v` (round to nearest; out of range clamps to infinity).
pub fn f32_to_f16(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = (man | 0x80_0000) >> (1 - e);
        return sign | ((m + 0x1000) >> 13) as u16;
    }
    let h = ((e as u32) << 10) | (man >> 13);
    // Round to nearest (a carry into the exponent is still right).
    sign | (h + ((man >> 12) & 1)) as u16
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let man = (h & 0x3ff) as f32;
    match exp {
        0 => sign * man * 2f32.powi(-24),
        31 => sign * f32::INFINITY,
        _ => sign * (1.0 + man / 1024.0) * 2f32.powi(exp - 15),
    }
}

/// Shaders are checked here without a GPU: a WGSL mistake would otherwise only
/// show when the app starts.
#[cfg(test)]
mod texture_memory_tests {
    use super::texture_allocation_bytes;

    #[test]
    fn counts_every_mip_level() {
        // 4x2 RGBA8: 32 + 8 + 4 bytes across 3 levels.
        assert_eq!(texture_allocation_bytes(4, 2, 4, 3, 1), 44);
        assert_eq!(texture_allocation_bytes(4, 2, 4, 3, 4), 176);
    }

    #[test]
    fn single_level_render_targets_keep_their_original_size() {
        assert_eq!(texture_allocation_bytes(2048, 1024, 8, 1, 1), 16 * 1024 * 1024);
    }
}

#[cfg(test)]
mod shader_tests {
    use super::wgpu;

    fn validate(label: &str, src: &str) {
        let full = format!("{}\n{}", include_str!("shaders/common.wgsl"), src);
        let module = wgpu::naga::front::wgsl::parse_str(&full).unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(&full)));
        wgpu::naga::valid::Validator::new(wgpu::naga::valid::ValidationFlags::all(), wgpu::naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{label}: {e:?}"));
    }

    #[test]
    fn composite_shader_is_valid() {
        validate("composite", concat!(include_str!("shaders/wear.wgsl"), include_str!("shaders/composite.wgsl")));
    }

    #[test]
    fn engine_shaders_are_valid() {
        validate("thumbs", include_str!("shaders/thumbs.wgsl"));
        validate("mask filter", include_str!("shaders/mask_filter.wgsl"));
        validate("apply", include_str!("shaders/apply.wgsl"));
        validate("maps resolve", include_str!("shaders/maps_resolve.wgsl"));
    }
}
