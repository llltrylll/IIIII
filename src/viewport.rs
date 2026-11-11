//! Offscreen viewport rendering (3D PBR view and 2D UV view) into textures
//! that egui displays.

use crate::engine::Engine;
use crate::environment::{Environment, RADIANCE_MIPS, env_sampler};
use crate::gpu::{self, GpuMesh, Tex};
use crate::post::{HDR_FORMAT, PostFx, PostSettings};
use eframe::{egui, egui_wgpu, wgpu};
use glam::{Mat4, Vec2, Vec3, Vec4};

pub const NEAR: f32 = 0.02;
pub const FAR: f32 = 100.0;
const MSAA: u32 = 4;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// What the viewports display. Discriminants match `shade()` in view.wgsl;
/// `Map(i)` shows baked map `engine::MAPS[i]`, `Custom(id)` a custom bake map.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ViewMode {
    Material,
    BaseColor,
    Roughness,
    Metallic,
    Height,
    Opacity,
    Emission,
    Specular,
    Normal,
    Map(usize),
    Custom(u32),
    WorldNormal,
    Position,
    MissedRays,
}

impl ViewMode {
    pub const CHANNELS: [ViewMode; 9] = [
        ViewMode::Material,
        ViewMode::BaseColor,
        ViewMode::Roughness,
        ViewMode::Metallic,
        ViewMode::Height,
        ViewMode::Opacity,
        ViewMode::Emission,
        ViewMode::Specular,
        ViewMode::Normal,
    ];

    pub fn all() -> Vec<ViewMode> {
        let mut v = Self::CHANNELS.to_vec();
        v.extend((0..crate::engine::MAPS.len()).map(ViewMode::Map));
        v.extend([ViewMode::WorldNormal, ViewMode::Position, ViewMode::MissedRays]);
        v
    }

    pub fn index(self) -> u32 {
        match self {
            ViewMode::Material => 0,
            ViewMode::BaseColor => 1,
            ViewMode::Roughness => 2,
            ViewMode::Metallic => 3,
            ViewMode::Height => 4,
            ViewMode::Opacity => 23,
            ViewMode::Emission => 24,
            ViewMode::Specular => 25,
            ViewMode::Normal => 5,
            ViewMode::Map(i) => 6 + i as u32,
            // Which custom map rides in the globals (see `base_globals`).
            ViewMode::Custom(_) => 22,
            ViewMode::WorldNormal => 19,
            ViewMode::Position => 20,
            ViewMode::MissedRays => 21,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ViewMode::Material => "Material",
            ViewMode::BaseColor => "Base Color",
            ViewMode::Roughness => "Roughness",
            ViewMode::Metallic => "Metallic",
            ViewMode::Height => "Height",
            ViewMode::Opacity => "Opacity",
            ViewMode::Emission => "Emission",
            ViewMode::Specular => "Specular",
            ViewMode::Normal => "Normal (final)",
            ViewMode::Map(i) => crate::engine::MAPS[i].name,
            // The app names it after the map (see `PainterApp::mode_name`).
            ViewMode::Custom(_) => "Custom map",
            ViewMode::WorldNormal => "World Space Normal",
            ViewMode::Position => "Position",
            ViewMode::MissedRays => "Missed Rays",
        }
    }
}

/// Right-handed perspective with [0, 1] depth (wgpu convention).
fn perspective_rh(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fov_y * 0.5).tan();
    let r = far / (near - far);
    Mat4::from_cols(
        Vec4::new(f / aspect, 0.0, 0.0, 0.0),
        Vec4::new(0.0, f, 0.0, 0.0),
        Vec4::new(0.0, 0.0, r, -1.0),
        Vec4::new(0.0, 0.0, r * near, 0.0),
    )
}

fn look_at_rh(eye: Vec3, center: Vec3, up: Vec3) -> Mat4 {
    let f = (center - eye).normalize();
    let s = f.cross(up).normalize();
    let u = s.cross(f);
    Mat4::from_cols(
        Vec4::new(s.x, u.x, -f.x, 0.0),
        Vec4::new(s.y, u.y, -f.y, 0.0),
        Vec4::new(s.z, u.z, -f.z, 0.0),
        Vec4::new(-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0),
    )
}

#[derive(Clone, Copy)]
pub struct OrbitCam {
    pub yaw: f32,
    pub pitch: f32,
    pub dist: f32,
    pub target: Vec3,
    pub fov_y: f32,
}

impl Default for OrbitCam {
    fn default() -> Self {
        Self { yaw: 0.6, pitch: 0.35, dist: 5.2, target: Vec3::ZERO, fov_y: 38f32.to_radians() }
    }
}

impl OrbitCam {
    pub fn eye(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        self.target + Vec3::new(cp * sy, sp, cp * cy) * self.dist
    }
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        perspective_rh(self.fov_y, aspect.max(1e-3), NEAR, FAR) * look_at_rh(self.eye(), self.target, Vec3::Y)
    }
    pub fn basis(&self) -> (Vec3, Vec3) {
        let fwd = (self.target - self.eye()).normalize();
        let right = fwd.cross(Vec3::Y).normalize();
        (right, right.cross(fwd))
    }
    /// World-space ray through a pixel of a viewport of `size` pixels.
    pub fn ray(&self, px: Vec2, size: Vec2) -> (Vec3, Vec3) {
        let ndc = Vec2::new(px.x / size.x * 2.0 - 1.0, 1.0 - px.y / size.y * 2.0);
        let inv = self.view_proj(size.x / size.y).inverse();
        let far = inv.project_point3(ndc.extend(1.0));
        let eye = self.eye();
        (eye, (far - eye).normalize())
    }
}

#[derive(Clone, Copy)]
pub struct Cam2d {
    /// Pixels per UV unit.
    pub zoom: f32,
    /// Pixel offset of the UV square center from the view center.
    pub pan: Vec2,
}

impl Default for Cam2d {
    fn default() -> Self {
        Self { zoom: 0.0, pan: Vec2::ZERO }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Background {
    /// The environment HDRI, optionally blurred.
    Environment,
    Color,
    /// Nothing behind the model (thumbnails, renders for compositing).
    Transparent,
}

impl Background {
    pub const ALL: [Background; 3] = [Background::Environment, Background::Color, Background::Transparent];

    pub fn name(self) -> &'static str {
        match self {
            Background::Environment => "Environment",
            Background::Color => "Solid color",
            Background::Transparent => "Transparent",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ViewSettings {
    pub mode: ViewMode,
    pub exposure: f32,
    pub env_rotation: f32,
    /// Strength of the environment lighting.
    pub env_intensity: f32,
    pub background: Background,
    /// 0 sharp .. 1 fully blurred environment backdrop.
    pub bg_blur: f32,
    pub bg_intensity: f32,
    /// Linear RGB.
    pub bg_color: [f32; 3],
    pub height_strength: f32,
    pub show_wire: bool,
    /// Mesh wireframe over the 3D view.
    pub show_wire_3d: bool,
    /// UV seams over the 3D view.
    pub show_seams_3d: bool,
    /// UV seams (island borders) in the 2D view.
    pub show_seams_2d: bool,
    /// Opacity of the 2D view's polygon edges.
    pub uv_wire_opacity: f32,
    pub use_baked_normal: bool,
    /// How much the baked ambient occlusion darkens the 3D view's lighting
    /// (display only: bakes and exports keep the full map).
    pub ao_strength: f32,
    /// How bright emission shows in the 3D view and the path tracer
    /// (display only: exports keep the 0-1 emission color).
    pub emission: f32,
    pub post: PostSettings,
    pub key: KeyLight,
}

/// A directional key light with shadows, on top of the environment. Its
/// direction turns with the environment rotation.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct KeyLight {
    pub on: bool,
    /// Degrees around the up axis.
    pub azimuth: f32,
    /// Degrees above the horizon.
    pub elevation: f32,
    pub intensity: f32,
    /// Linear RGB.
    pub color: [f32; 3],
}

impl Default for KeyLight {
    fn default() -> Self {
        Self { on: true, azimuth: 40.0, elevation: 45.0, intensity: 2.2, color: [1.0, 0.96, 0.9] }
    }
}

impl KeyLight {
    /// Unit direction toward the light, with the environment's rotation.
    pub fn direction(&self, env_rotation: f32) -> Vec3 {
        let (az, el) = ((self.azimuth.to_radians() + env_rotation), self.elevation.to_radians());
        Vec3::new(el.cos() * az.sin(), el.sin(), el.cos() * az.cos()).normalize()
    }
}

impl Default for ViewSettings {
    fn default() -> Self {
        Self {
            mode: ViewMode::Material,
            exposure: 1.0,
            env_rotation: 0.0,
            env_intensity: 1.0,
            background: Background::Environment,
            bg_blur: 0.6,
            bg_intensity: 0.55,
            bg_color: [0.05, 0.05, 0.055],
            height_strength: 1.0,
            show_wire: true,
            show_wire_3d: false,
            show_seams_3d: false,
            show_seams_2d: true,
            uv_wire_opacity: 0.45,
            use_baked_normal: true,
            ao_strength: 0.6,
            emission: 2.0,
            post: PostSettings::default(),
            key: KeyLight::default(),
        }
    }
}

/// A vertex of a 3D view overlay: world position and sRGB color with alpha.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OverlayVertex {
    pub pos: [f32; 3],
    pub color: [u8; 4],
}

/// Independent overlays drawn over the 3D view.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum OverlaySlot {
    /// Bake cage preview.
    Cage = 0,
    /// Imported per-corner vertex colors for the ID-map preview.
    VertexColors = 1,
    /// Polygon fill: the faces under the cursor (drawn over the ID preview).
    Selection = 2,
}

/// Triangles and lines of one overlay slot.
#[derive(Default)]
struct OverlayBuffers {
    tris: Option<(wgpu::Buffer, u32)>,
    lines: Option<(wgpu::Buffer, u32)>,
}

/// Geometry drawn over the 3D view, owned by the app so every renderer
/// (viewport, MCP renders) shows the same.
#[derive(Default)]
pub struct Overlays {
    slots: [OverlayBuffers; 3],
}

impl Overlays {
    /// Replaces a slot's geometry: triangle-list and line-list vertices.
    pub fn set(&mut self, device: &wgpu::Device, slot: OverlaySlot, tris: &[OverlayVertex], lines: &[OverlayVertex]) {
        use wgpu::util::DeviceExt;
        let upload = |label, v: &[OverlayVertex]| {
            (!v.is_empty()).then(|| {
                let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(v),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                (buffer, v.len() as u32)
            })
        };
        self.slots[slot as usize] = OverlayBuffers { tris: upload("overlay tris", tris), lines: upload("overlay lines", lines) };
    }

    pub fn clear(&mut self, slot: OverlaySlot) {
        self.slots[slot as usize] = OverlayBuffers::default();
    }

    pub fn is_set(&self, slot: OverlaySlot) -> bool {
        let s = &self.slots[slot as usize];
        s.tris.is_some() || s.lines.is_some()
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Globals {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    light: [f32; 4],
    params: [f32; 4],
    view2d: [f32; 4],
    size: [f32; 4],
    extra: [f32; 4],
    bmin: [f32; 4],
    bsize: [f32; 4],
    bg: [f32; 4],
    bg2: [f32; 4],
    light_vp: [[f32; 4]; 4],
    sun: [f32; 4],
    sun_c: [f32; 4],
}

const SHADOW_SIZE: u32 = 2048;

/// The 3D view renders scene-linear HDR with MSAA, resolves it, and post
/// writes the displayed image; the 2D view draws straight to the display.
struct Targets {
    color: Tex,
    resolve: Option<Tex>,
    depth: Option<Tex>,
    display: Option<Tex>,
}

pub struct ViewTarget {
    color: Tex,
    resolve: Option<Tex>,
    depth: Option<Tex>,
    display: Option<Tex>,
    pub egui_id: egui::TextureId,
    pub size: [u32; 2],
}

impl ViewTarget {
    fn create(device: &wgpu::Device, size: [u32; 2], hdr: bool) -> Targets {
        let [w, h] = size;
        // COPY_SRC lets tests and MCP read rendered images back.
        let display = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC;
        if hdr {
            Targets {
                color: Tex::new_ms(device, "view msaa", w, h, HDR_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT, MSAA),
                resolve: Some(Tex::new(device, "view hdr", w, h, HDR_FORMAT, display)),
                depth: Some(Tex::new_ms(device, "view depth", w, h, DEPTH_FORMAT, display, MSAA)),
                display: Some(Tex::new(device, "view", w, h, COLOR_FORMAT, display)),
            }
        } else {
            Targets { color: Tex::new(device, "view 2d", w, h, COLOR_FORMAT, display), resolve: None, depth: None, display: None }
        }
    }

    fn shown(t: &Targets) -> &Tex {
        t.display.as_ref().unwrap_or(&t.color)
    }

    fn new(rs: &egui_wgpu::RenderState, hdr: bool) -> ViewTarget {
        let size = [64, 64];
        let t = Self::create(&rs.device, size, hdr);
        let egui_id = rs.renderer.write().register_native_texture(&rs.device, &Self::shown(&t).view, wgpu::FilterMode::Linear);
        ViewTarget { color: t.color, resolve: t.resolve, depth: t.depth, display: t.display, egui_id, size }
    }

    fn ensure_size(&mut self, rs: &egui_wgpu::RenderState, size: [u32; 2]) {
        let size = [size[0].clamp(16, 8192), size[1].clamp(16, 8192)];
        if size == self.size {
            return;
        }
        let t = Self::create(&rs.device, size, self.display.is_some());
        rs.renderer.write().update_egui_texture_from_wgpu_texture(
            &rs.device,
            &Self::shown(&t).view,
            wgpu::FilterMode::Linear,
            self.egui_id,
        );
        self.color = t.color;
        self.resolve = t.resolve;
        self.depth = t.depth;
        self.display = t.display;
        self.size = size;
    }

    /// The displayed image.
    pub fn image(&self) -> &Tex {
        self.display.as_ref().unwrap_or(&self.color)
    }

    pub fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth.as_ref().expect("3D view has depth").view
    }
}

/// Parameter slots (vec4 each) of a custom preview shader.
pub const CUSTOM_PARAMS: usize = 128;

/// How the 3D view draws one texture set.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct SetLook {
    /// Not drawn at all.
    pub hidden: bool,
    /// 0..1: tints the set to show which part of the mesh it is.
    pub highlight: f32,
}

pub struct ViewRenderer {
    /// Per texture set (missing entries: shown, no highlight).
    pub set_looks: Vec<SetLook>,
    layout: wgpu::PipelineLayout,
    /// The mesh drawn with a custom preview shader (a compiled shading graph).
    custom_mesh: Option<wgpu::RenderPipeline>,
    custom_params: wgpu::Buffer,
    post: PostFx,
    env_sampler: wgpu::Sampler,
    sky: wgpu::RenderPipeline,
    mesh: wgpu::RenderPipeline,
    /// The mesh's see-through texels, blended after the opaque ones.
    mesh_t: wgpu::RenderPipeline,
    quad: wgpu::RenderPipeline,
    wire: wgpu::RenderPipeline,
    wire3d: wgpu::RenderPipeline,
    seam2d: wgpu::RenderPipeline,
    seam3d: wgpu::RenderPipeline,
    overlay_tris: wgpu::RenderPipeline,
    overlay_lines: wgpu::RenderPipeline,
    shadow: wgpu::RenderPipeline,
    shadow_bgl: wgpu::BindGroupLayout,
    shadow_map: Tex,
    shadow_sampler: wgpu::Sampler,
    bgl: wgpu::BindGroupLayout,
    /// Per-draw uniforms: the 2D view's, then one per texture set in the 3D
    /// view (each set has its own resolution, bake and custom maps).
    globals: Vec<wgpu::Buffer>,
    sampler: wgpu::Sampler,
    pub view3d: ViewTarget,
    pub view2d: ViewTarget,
}

impl ViewRenderer {
    pub fn new(rs: &egui_wgpu::RenderState) -> ViewRenderer {
        let device = &rs.device;
        let module = gpu::shader(device, "view", include_str!("shaders/view.wgsl"));
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("view"),
            entries: &[
                gpu::uniform_entry(0, false),
                gpu::tex_entry(1, true),
                gpu::tex_entry(2, true),
                gpu::tex_entry(3, true),
                gpu::tex_entry(4, true),
                gpu::tex_entry(5, true),
                gpu::tex_entry(6, true),
                gpu::tex_entry(7, true),
                gpu::tex_entry(8, true),
                gpu::tex_entry(9, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 19,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                gpu::sampler_entry(10),
                gpu::tex_entry(11, true),
                gpu::tex_entry(12, true),
                gpu::tex_entry(13, true),
                gpu::sampler_entry(14),
                gpu::uniform_entry(15, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 16,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 17,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                gpu::uniform_entry(18, false),
                gpu::tex_entry(20, true),
                gpu::tex_entry(21, true),
                gpu::tex_entry(22, true),
            ],
        });
        let layout = gpu::layout(device, &bgl);
        let depth = |write: bool, compare| {
            Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(write),
                depth_compare: Some(compare),
                stencil: Default::default(),
                bias: Default::default(),
            })
        };
        let ms = |count| wgpu::MultisampleState { count, ..Default::default() };
        let color = Some(wgpu::ColorTargetState {
            format: COLOR_FORMAT,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        });
        let hdr = Some(wgpu::ColorTargetState { format: HDR_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL });
        let blended = Some(wgpu::ColorTargetState {
            format: COLOR_FORMAT,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        });
        let pipeline = |label: &str,
                        vs: &str,
                        fs: &str,
                        buffers: &[Option<wgpu::VertexBufferLayout>],
                        topology,
                        depth_stencil,
                        samples,
                        target: &Option<wgpu::ColorTargetState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers,
                },
                primitive: wgpu::PrimitiveState { topology, ..Default::default() },
                depth_stencil,
                multisample: ms(samples),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: std::slice::from_ref(target),
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let tri = wgpu::PrimitiveTopology::TriangleList;
        let vb = [Some(gpu::vertex_layout())];
        let sky = pipeline("sky", "vs_fullscreen", "fs_sky", &[], tri, depth(false, wgpu::CompareFunction::Always), MSAA, &hdr);
        let mesh = pipeline("mesh", "vs_mesh", "fs_mesh", &vb, tri, depth(true, wgpu::CompareFunction::Less), MSAA, &hdr);
        let hdr_over = Some(wgpu::ColorTargetState {
            format: HDR_FORMAT,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        });
        let mesh_t = pipeline("mesh (see-through)", "vs_mesh", "fs_mesh_t", &vb, tri, depth(true, wgpu::CompareFunction::Less), MSAA, &hdr_over);
        let quad = pipeline("quad", "vs_quad", "fs_quad", &[], tri, None, 1, &color);
        let wire = pipeline("wire", "vs_wire", "fs_wire", &vb, wgpu::PrimitiveTopology::LineList, None, 1, &blended);
        // Overlays test against the mesh depth but never write it.
        let hdr_blended = Some(wgpu::ColorTargetState {
            format: HDR_FORMAT,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        });
        let over = || depth(false, wgpu::CompareFunction::LessEqual);
        let line = wgpu::PrimitiveTopology::LineList;
        let wire3d = pipeline("wire 3d", "vs_wire3d", "fs_overlay", &vb, line, over(), MSAA, &hdr_blended);
        let seam2d = pipeline("seams 2d", "vs_wire", "fs_seam", &vb, line, None, 1, &blended);
        let seam3d = pipeline("seams 3d", "vs_seam3d", "fs_overlay", &vb, line, over(), MSAA, &hdr_blended);
        const OVERLAY_ATTRS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Unorm8x4];
        let ob = [Some(wgpu::VertexBufferLayout {
            array_stride: size_of::<OverlayVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &OVERLAY_ATTRS,
        })];
        let overlay_tris = pipeline("overlay tris", "vs_overlay", "fs_overlay", &ob, tri, over(), MSAA, &hdr_blended);
        let overlay_lines = pipeline("overlay lines", "vs_overlay", "fs_overlay", &ob, line, over(), MSAA, &hdr_blended);
        // Key light shadow map: depth only, both faces (meshes may be open),
        // with slope-scaled bias against acne. Its own layout holds just the
        // globals: the view's bind group samples the shadow map it writes.
        let shadow_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("key shadow"),
            entries: &[gpu::uniform_entry(0, false)],
        });
        let shadow = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("key shadow"),
            layout: Some(&gpu::layout(device, &shadow_bgl)),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_shadow"),
                compilation_options: Default::default(),
                buffers: &vb,
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 },
            }),
            multisample: Default::default(),
            fragment: None,
            multiview_mask: None,
            cache: None,
        });
        let shadow_map = Tex::new(
            device,
            "key shadow map",
            SHADOW_SIZE,
            SHADOW_SIZE,
            DEPTH_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("key shadow"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        let globals = vec![Self::globals_buffer(device), Self::globals_buffer(device)];
        let custom_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("preview shader params"),
            size: (CUSTOM_PARAMS * 16) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ViewRenderer {
            layout,
            set_looks: Vec::new(),
            custom_mesh: None,
            custom_params,
            post: PostFx::new(rs, COLOR_FORMAT),
            env_sampler: env_sampler(device),
            sky,
            mesh,
            mesh_t,
            quad,
            wire,
            wire3d,
            seam2d,
            seam3d,
            overlay_tris,
            overlay_lines,
            shadow,
            shadow_bgl,
            shadow_map,
            shadow_sampler,
            bgl,
            globals,
            sampler: gpu::linear_sampler(device, wgpu::AddressMode::ClampToEdge),
            view3d: ViewTarget::new(rs, true),
            view2d: ViewTarget::new(rs, false),
        }
    }

    /// Uses `code` (a generated `custom_shade(c: Shade) -> vec3<f32>`) to
    /// shade the mesh in Material mode, or the built-in look when `None`.
    /// On a compile error the previous shader stays and the error is returned.
    pub fn set_custom_shader(&mut self, device: &wgpu::Device, code: Option<&str>) -> Result<(), String> {
        let Some(code) = code else {
            self.custom_mesh = None;
            return Ok(());
        };
        let src = format!(
            "{}\n{code}\n@fragment\nfn fs_mesh_custom(in: MeshOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {{\n    let c = mesh_shade_inputs(in, front);\n    if (view_alpha(c.s) < 0.5) {{\n        discard;\n    }}\n    return vec4<f32>(highlighted(max(custom_shade(c), vec3<f32>(0.0))), 1.0);\n}}\n",
            include_str!("shaders/view.wgsl")
        );
        let (pipeline, err) = crate::graph::runtime::validation_scope(device, || {
            let module = gpu::shader(device, "preview shader", &src);
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("mesh (preview shader)"),
                layout: Some(&self.layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_mesh"),
                    compilation_options: Default::default(),
                    buffers: &[Some(gpu::vertex_layout())],
                },
                primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState { count: MSAA, ..Default::default() },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_mesh_custom"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format: HDR_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                }),
                multiview_mask: None,
                cache: None,
            })
        });
        if let Some(e) = err {
            return Err(e);
        }
        self.custom_mesh = Some(pipeline);
        Ok(())
    }

    /// The custom preview shader's parameter values (see `CUSTOM_PARAMS`).
    pub fn set_custom_params(&self, queue: &wgpu::Queue, values: &[[f32; 4]]) {
        let n = values.len().min(CUSTOM_PARAMS);
        if n > 0 {
            queue.write_buffer(&self.custom_params, 0, bytemuck::cast_slice(&values[..n]));
        }
    }

    pub fn has_custom_shader(&self) -> bool {
        self.custom_mesh.is_some()
    }

    fn globals_buffer(device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view globals"),
            size: size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn bind_group(&self, device: &wgpu::Device, engine: &Engine, env: &Environment, globals: &wgpu::Buffer) -> wgpu::BindGroup {
        fn v(view: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
            wgpu::BindingResource::TextureView(view)
        }
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("view"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: globals.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: v(engine.final_tex(0).srgb()) },
                wgpu::BindGroupEntry { binding: 2, resource: v(&engine.final_tex(1).view) },
                wgpu::BindGroupEntry { binding: 3, resource: v(&engine.final_tex(2).view) },
                wgpu::BindGroupEntry { binding: 4, resource: v(&engine.final_tex(3).view) },
                wgpu::BindGroupEntry { binding: 5, resource: v(&engine.map_tex(0).view) },
                wgpu::BindGroupEntry { binding: 6, resource: v(&engine.map_tex(1).view) },
                wgpu::BindGroupEntry { binding: 7, resource: v(&engine.map_tex(2).view) },
                wgpu::BindGroupEntry { binding: 8, resource: v(&engine.map_tex(3).view) },
                wgpu::BindGroupEntry { binding: 9, resource: v(&engine.map_tex(4).view) },
                wgpu::BindGroupEntry { binding: 19, resource: v(engine.custom_maps_view()) },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 11, resource: v(&engine.surface_maps().1.view) },
                wgpu::BindGroupEntry { binding: 12, resource: v(&engine.surface_maps().0.view) },
                wgpu::BindGroupEntry { binding: 13, resource: v(&env.radiance) },
                wgpu::BindGroupEntry { binding: 14, resource: wgpu::BindingResource::Sampler(&self.env_sampler) },
                wgpu::BindGroupEntry { binding: 15, resource: env.sh.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 16, resource: v(&self.shadow_map.view) },
                wgpu::BindGroupEntry { binding: 17, resource: wgpu::BindingResource::Sampler(&self.shadow_sampler) },
                wgpu::BindGroupEntry { binding: 18, resource: self.custom_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 20, resource: v(engine.final_tex(crate::engine::FINAL_EMISSION).srgb()) },
                wgpu::BindGroupEntry { binding: 21, resource: v(&engine.final_tex(crate::engine::FINAL_NORMAL).view) },
                wgpu::BindGroupEntry { binding: 22, resource: v(&engine.final_tex(crate::engine::FINAL_SPECULAR).view) },
            ],
        })
    }

    fn base_globals(engine: &Engine, s: &ViewSettings, size: [u32; 2]) -> Globals {
        let res = engine.res as f32;
        Globals {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            inv_view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            cam_pos: [0.0; 4],
            light: [s.env_rotation, s.exposure, s.env_intensity, (RADIANCE_MIPS - 1) as f32],
            params: [s.mode.index() as f32, s.height_strength, 1.0 / res, res],
            view2d: [0.0; 4],
            size: [size[0] as f32, size[1] as f32, s.ao_strength, 0.0],
            extra: [
                if s.use_baked_normal && engine.has_bake { 1.0 } else { 0.0 },
                if engine.has_bake { 1.0 } else { 0.0 },
                // The custom map shown, by index among the baked ones.
                match s.mode {
                    ViewMode::Custom(id) => engine.custom_index(id).map_or(-1.0, |i| i as f32),
                    _ => 0.0,
                },
                // 3D wireframe opacity.
                0.35,
            ],
            bmin: engine.bounds().0.extend(0.0).to_array(),
            bsize: (engine.bounds().1 - engine.bounds().0).extend(0.0).to_array(),
            bg: [s.bg_color[0], s.bg_color[1], s.bg_color[2], s.bg_intensity],
            bg2: [s.bg_blur, s.background as u32 as f32, s.emission, 0.0],
            light_vp: Self::light_view_proj(engine, s).to_cols_array_2d(),
            sun: s.key.direction(s.env_rotation).extend(s.key.intensity).to_array(),
            sun_c: [s.key.color[0], s.key.color[1], s.key.color[2], if s.key.on { 1.0 } else { 0.0 }],
        }
    }

    /// Orthographic projection from the key light covering the mesh's
    /// bounding sphere.
    fn light_view_proj(engine: &Engine, s: &ViewSettings) -> Mat4 {
        let (lo, hi) = engine.bounds();
        let center = (lo + hi) * 0.5;
        let r = ((hi - lo).length() * 0.5).max(1e-3) * 1.05;
        let l = s.key.direction(s.env_rotation);
        let up = if l.y.abs() > 0.95 { Vec3::Z } else { Vec3::Y };
        let view = look_at_rh(center + l * r * 2.0, center, up);
        // Right-handed ortho with [0, 1] depth over [near, far] = [r, 3r].
        let (near, far) = (r * 0.5, r * 3.5);
        let proj = Mat4::from_cols(
            Vec4::new(1.0 / r, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0 / r, 0.0, 0.0),
            Vec4::new(0.0, 0.0, -1.0 / (far - near), 0.0),
            Vec4::new(0.0, 0.0, -near / (far - near), 1.0),
        );
        proj * view
    }

    /// Draws the mesh: texture set `k` of `mesh.sets` with `engines[k]`.
    pub fn render_3d(
        &mut self,
        rs: &egui_wgpu::RenderState,
        size: [u32; 2],
        engines: &[&Engine],
        env: &Environment,
        mesh: &GpuMesh,
        cam: &OrbitCam,
        settings: &ViewSettings,
        overlays: Option<&Overlays>,
    ) {
        self.view3d.ensure_size(rs, size);
        let size = self.view3d.size;
        let vp = cam.view_proj(size[0] as f32 / size[1] as f32);
        while self.globals.len() < engines.len() + 1 {
            self.globals.push(Self::globals_buffer(&rs.device));
        }
        let bgs: Vec<wgpu::BindGroup> = engines
            .iter()
            .enumerate()
            .map(|(k, engine)| {
                let mut g = Self::base_globals(engine, settings, size);
                g.size[3] = self.set_looks.get(k).map_or(0.0, |l| l.highlight);
                g.view_proj = vp.to_cols_array_2d();
                g.inv_view_proj = vp.inverse().to_cols_array_2d();
                g.cam_pos = cam.eye().extend(1.0).to_array();
                rs.queue.write_buffer(&self.globals[k + 1], 0, bytemuck::bytes_of(&g));
                self.bind_group(&rs.device, engine, env, &self.globals[k + 1])
            })
            .collect();
        let bg = &bgs[0];
        let transparent = settings.background == Background::Transparent;
        let shown: Vec<usize> = (0..mesh.sets.len()).filter(|&k| !self.set_looks.get(k).is_some_and(|l| l.hidden)).collect();

        let t = &self.view3d;
        let mut encoder = rs.device.create_command_encoder(&Default::default());
        if settings.key.on {
            let shadow_bg = rs.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("key shadow"),
                layout: &self.shadow_bgl,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: self.globals[1].as_entire_binding() }],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("key shadow"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_map.view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &shadow_bg, &[]);
            pass.set_pipeline(&self.shadow);
            pass.set_vertex_buffer(0, mesh.vertices.slice(..));
            pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
            for &k in &shown {
                pass.draw_indexed(mesh.sets[k].index_range(), 0, 0..1);
            }
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("3d view"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &t.color.view,
                    depth_slice: None,
                    resolve_target: t.resolve.as_ref().map(|r| &r.view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(if transparent { wgpu::Color::TRANSPARENT } else { wgpu::Color::BLACK }),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: t.depth_view(),
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, bg, &[]);
            if !transparent {
                pass.set_pipeline(&self.sky);
                pass.draw(0..3, 0..1);
            }
            pass.set_pipeline(self.custom_mesh.as_ref().filter(|_| settings.mode == ViewMode::Material).unwrap_or(&self.mesh));
            pass.set_vertex_buffer(0, mesh.vertices.slice(..));
            pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
            for &k in &shown {
                if let Some(bg) = bgs.get(k) {
                    pass.set_bind_group(0, bg, &[]);
                    pass.draw_indexed(mesh.sets[k].index_range(), 0, 0..1);
                }
            }
            // See-through texels, over everything opaque (the built-in look
            // only; a preview shader cuts them out at half opacity).
            if self.custom_mesh.is_none() || settings.mode != ViewMode::Material {
                pass.set_pipeline(&self.mesh_t);
                for &k in &shown {
                    if let (Some(bg), Some(e)) = (bgs.get(k), engines.get(k))
                        && e.uses_opacity() {
                            pass.set_bind_group(0, bg, &[]);
                            pass.draw_indexed(mesh.sets[k].index_range(), 0, 0..1);
                        }
                }
            }
            pass.set_bind_group(0, bg, &[]);
            if settings.show_wire_3d {
                pass.set_pipeline(&self.wire3d);
                pass.set_index_buffer(mesh.edges.slice(..), wgpu::IndexFormat::Uint32);
                for &k in &shown {
                    pass.draw_indexed(mesh.sets[k].edges.clone(), 0, 0..1);
                }
            }
            if settings.show_seams_3d && mesh.seam_count > 0 {
                pass.set_pipeline(&self.seam3d);
                pass.set_index_buffer(mesh.seams.slice(..), wgpu::IndexFormat::Uint32);
                for &k in &shown {
                    pass.draw_indexed(mesh.sets[k].seams.clone(), 0, 0..1);
                }
            }
            if let Some(overlays) = overlays {
                for o in &overlays.slots {
                    if let Some((buffer, count)) = &o.tris {
                        pass.set_pipeline(&self.overlay_tris);
                        pass.set_vertex_buffer(0, buffer.slice(..));
                        pass.draw(0..*count, 0..1);
                    }
                    if let Some((buffer, count)) = &o.lines {
                        pass.set_pipeline(&self.overlay_lines);
                        pass.set_vertex_buffer(0, buffer.slice(..));
                        pass.draw(0..*count, 0..1);
                    }
                }
            }
        }
        let t = &self.view3d;
        let (hdr, display) = (t.resolve.as_ref().expect("3D view is HDR"), t.display.as_ref().expect("3D view has a display image"));
        let raw = settings.mode != ViewMode::Material;
        self.post.record(rs, &mut encoder, hdr, &display.view, &settings.post, settings.exposure, raw);
        rs.queue.submit([encoder.finish()]);
    }

    /// Draws texture set `set` of the mesh, painted by `engine`, flat.
    pub fn render_2d(
        &mut self,
        rs: &egui_wgpu::RenderState,
        size: [u32; 2],
        engine: &Engine,
        env: &Environment,
        mesh: &GpuMesh,
        set: usize,
        cam: &Cam2d,
        settings: &ViewSettings,
    ) {
        self.view2d.ensure_size(rs, size);
        let size = self.view2d.size;
        let mut g = Self::base_globals(engine, settings, size);
        g.view2d = [cam.zoom, cam.pan.x, cam.pan.y, 0.0];
        g.extra[3] = settings.uv_wire_opacity;
        rs.queue.write_buffer(&self.globals[0], 0, bytemuck::bytes_of(&g));
        let bg = self.bind_group(&rs.device, engine, env, &self.globals[0]);
        let (edges, seams) = mesh.sets.get(set).map_or((0..mesh.edge_count, 0..mesh.seam_count), |s| (s.edges.clone(), s.seams.clone()));

        let mut encoder = rs.device.create_command_encoder(&Default::default());
        {
            let bg_color = wgpu::Color { r: 0.018, g: 0.018, b: 0.02, a: 1.0 };
            let mut pass = gpu::pass(
                &mut encoder,
                "2d view",
                &[gpu::color_attachment(&self.view2d.color.view, wgpu::LoadOp::Clear(bg_color))],
            );
            pass.set_bind_group(0, &bg, &[]);
            pass.set_pipeline(&self.quad);
            pass.draw(0..6, 0..1);
            pass.set_vertex_buffer(0, mesh.vertices.slice(..));
            if settings.show_wire {
                pass.set_pipeline(&self.wire);
                pass.set_index_buffer(mesh.edges.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(edges, 0, 0..1);
            }
            if settings.show_seams_2d && !seams.is_empty() {
                pass.set_pipeline(&self.seam2d);
                pass.set_index_buffer(mesh.seams.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(seams, 0, 0..1);
            }
        }
        rs.queue.submit([encoder.finish()]);
    }
}
