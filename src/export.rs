//! Texture export. An output template lists the files to write and what
//! goes in each of their channels (any painted or baked map, inverted or
//! not, or a constant), so engine conventions like packed ORM or Unity's
//! mask map are data the user can copy and edit. File format, bit depth,
//! size and UV padding apply to the whole export.

use crate::engine::{Engine, MAPS, MapKind};
use crate::gpu::{self, Tex, f16_to_f32};
use crate::mesh::Mesh;
use eframe::wgpu;
use glam::{Vec2, Vec3};
use image::{ImageBuffer, Luma, Rgb, Rgba};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// ------------------------------------------------------------------ sources

/// Something a channel can be filled from.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Source {
    BaseColor,
    /// Specular/glossiness workflow: base color of non-metals.
    Diffuse,
    /// Specular/glossiness workflow: reflectance color.
    Specular,
    /// The painted specular level of non-metals (0.5: 4% reflectance), as
    /// Unreal's and Blender's Specular inputs take it.
    SpecularLevel,
    Roughness,
    /// 1 - roughness.
    Glossiness,
    Metallic,
    /// Painted height, mid-gray at rest.
    Height,
    /// 1 opaque, 0 see-through (the base color's alpha).
    Opacity,
    /// Emitted light color (black where nothing emits).
    Emission,
    /// Final tangent-space normal (baked, painted normal and height), OpenGL (Y+).
    NormalGl,
    /// The same normal with Y flipped (DirectX, Y-).
    NormalDx,
    /// A baked mesh map, by index into `MAPS`.
    Mesh(usize),
    WorldNormal,
    /// Object-space position, normalized to the mesh bounds.
    Position,
    Zero,
    One,
}

impl Source {
    pub fn all() -> Vec<Source> {
        let mut v = vec![
            Source::BaseColor,
            Source::Diffuse,
            Source::Specular,
            Source::Roughness,
            Source::Glossiness,
            Source::Metallic,
            Source::SpecularLevel,
            Source::Height,
            Source::Opacity,
            Source::Emission,
            Source::NormalGl,
            Source::NormalDx,
        ];
        v.extend((0..MAPS.len()).map(Source::Mesh));
        v.extend([Source::WorldNormal, Source::Position, Source::Zero, Source::One]);
        v
    }

    pub fn name(self) -> String {
        match self {
            Source::BaseColor => "Base color".into(),
            Source::Diffuse => "Diffuse".into(),
            Source::Specular => "Specular".into(),
            Source::Roughness => "Roughness".into(),
            Source::Glossiness => "Glossiness".into(),
            Source::Metallic => "Metallic".into(),
            Source::SpecularLevel => "Specular level".into(),
            Source::Height => "Height".into(),
            Source::Opacity => "Opacity".into(),
            Source::Emission => "Emission".into(),
            Source::NormalGl => "Normal (OpenGL)".into(),
            Source::NormalDx => "Normal (DirectX)".into(),
            Source::Mesh(i) => format!("Baked {}", MAPS[i].name),
            Source::WorldNormal => "World space normal".into(),
            Source::Position => "Position".into(),
            Source::Zero => "Black (0)".into(),
            Source::One => "White (1)".into(),
        }
    }

    fn key(self) -> String {
        match self {
            Source::Mesh(i) => format!("mesh:{}", MAPS[i].name.replace(' ', "")),
            s => format!("{s:?}"),
        }
    }

    fn from_key(k: &str) -> Option<Source> {
        if let Some(name) = k.strip_prefix("mesh:") {
            return MAPS.iter().position(|m| m.name.replace(' ', "") == name).map(Source::Mesh);
        }
        Source::all().into_iter().find(|s| s.key() == k)
    }

    /// Color-like sources have three components; the rest one.
    pub fn components(self) -> usize {
        match self {
            Source::BaseColor | Source::Diffuse | Source::Specular | Source::NormalGl | Source::NormalDx | Source::WorldNormal | Source::Position => 3,
            Source::Mesh(i) if MAPS[i].kind != MapKind::Scalar => 3,
            _ => 1,
        }
    }

    fn is_normal(self) -> bool {
        matches!(self, Source::NormalGl | Source::NormalDx) || matches!(self, Source::Mesh(i) if MAPS[i].kind == MapKind::Normal)
    }
}

/// One output channel: a source component, optionally inverted.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Chan {
    pub src: Source,
    pub comp: u8,
    pub invert: bool,
}

impl Chan {
    pub const fn of(src: Source, comp: u8) -> Chan {
        Chan { src, comp, invert: false }
    }
    pub const fn inv(src: Source) -> Chan {
        Chan { src, comp: 0, invert: true }
    }
    pub fn label(&self) -> String {
        let comp = if self.src.components() > 1 { ["R", "G", "B"][self.comp.min(2) as usize] } else { "" };
        let s = if comp.is_empty() { self.src.name() } else { format!("{}.{comp}", self.src.name()) };
        if self.invert { format!("1 - {s}") } else { s }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Layout {
    Gray,
    Rgb,
    Rgba,
}

impl Layout {
    pub const ALL: [Layout; 3] = [Layout::Gray, Layout::Rgb, Layout::Rgba];
    pub fn name(self) -> &'static str {
        match self {
            Layout::Gray => "Grayscale",
            Layout::Rgb => "RGB",
            Layout::Rgba => "RGBA",
        }
    }
    pub fn channels(self) -> usize {
        match self {
            Layout::Gray => 1,
            Layout::Rgb => 3,
            Layout::Rgba => 4,
        }
    }
}

/// One file of a template.
#[derive(Clone, PartialEq, Debug)]
pub struct OutputMap {
    /// File name without extension; `$mesh`, `$set` and `$project` are replaced.
    pub name: String,
    pub layout: Layout,
    /// R, G, B, A (only the first `layout.channels()` are used; Gray uses R).
    pub chans: [Chan; 4],
    /// Writes 16 bits per channel when the format allows, whatever the
    /// export's bit depth (height maps band in 8 bits).
    pub force_16: bool,
}

impl OutputMap {
    fn rgb(name: &str, src: Source) -> OutputMap {
        OutputMap { name: name.into(), layout: Layout::Rgb, chans: [Chan::of(src, 0), Chan::of(src, 1), Chan::of(src, 2), Chan::of(Source::One, 0)], force_16: false }
    }
    /// A color with opacity in alpha (written RGB when the texture set is opaque).
    fn rgb_opacity(name: &str, src: Source) -> OutputMap {
        OutputMap { layout: Layout::Rgba, chans: [Chan::of(src, 0), Chan::of(src, 1), Chan::of(src, 2), Chan::of(Source::Opacity, 0)], ..OutputMap::rgb(name, src) }
    }

    /// Its alpha is only opacity: nothing to keep when everything is opaque.
    fn alpha_is_opacity(&self) -> bool {
        self.layout == Layout::Rgba && self.chans[3] == Chan::of(Source::Opacity, 0)
    }
    fn gray(name: &str, c: Chan) -> OutputMap {
        OutputMap { name: name.into(), layout: Layout::Gray, chans: [c, Chan::of(Source::Zero, 0), Chan::of(Source::Zero, 0), Chan::of(Source::One, 0)], force_16: false }
    }
    fn packed(name: &str, layout: Layout, chans: [Chan; 4]) -> OutputMap {
        OutputMap { name: name.into(), layout, chans, force_16: false }
    }
    fn height(name: &str) -> OutputMap {
        OutputMap { force_16: true, ..OutputMap::gray(name, Chan::of(Source::Height, 0)) }
    }

    /// Only emission or specular level, where the texture set has none of
    /// it (nothing emits, or specular is the default everywhere): not written.
    fn left_out(&self, s: &Snapshot) -> bool {
        let chans = &self.chans[..self.layout.channels()];
        let optional = |src: Source| match src {
            Source::Emission => Some(s.emissive),
            Source::SpecularLevel => Some(s.specular),
            Source::Zero | Source::One => Some(false),
            _ => None,
        };
        chans.iter().all(|c| optional(c.src).is_some())
            && chans.iter().any(|c| matches!(c.src, Source::Emission | Source::SpecularLevel))
            && !chans.iter().any(|c| optional(c.src) == Some(true))
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct Template {
    pub name: String,
    pub maps: Vec<OutputMap>,
    /// Shipped with the app: copy it to edit.
    pub builtin: bool,
}

const C0: Chan = Chan::of(Source::Zero, 0);
const C1: Chan = Chan::of(Source::One, 0);

pub fn builtin_templates() -> Vec<Template> {
    use Source::*;
    let ao = Chan::of(Mesh(0), 0);
    let rough = Chan::of(Roughness, 0);
    let metal = Chan::of(Metallic, 0);
    let smooth = Chan::inv(Roughness);
    let t = |name: &str, maps: Vec<OutputMap>| Template { name: name.into(), maps, builtin: true };
    let mesh_maps = {
        let mut maps: Vec<OutputMap> = MAPS
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let name = format!("$mesh_{}", m.name.replace(' ', ""));
                match m.kind {
                    MapKind::Scalar => OutputMap::gray(&name, Chan::of(Mesh(i), 0)),
                    _ => OutputMap::rgb(&name, Mesh(i)),
                }
            })
            .collect();
        maps.push(OutputMap::rgb("$mesh_WorldSpaceNormal", WorldNormal));
        maps.push(OutputMap::rgb("$mesh_Position", Position));
        maps
    };
    vec![
        t(
            "Unreal Engine (packed ORM)",
            vec![
                OutputMap::rgb_opacity("T_$mesh_BaseColor", BaseColor),
                OutputMap::rgb("T_$mesh_Normal", NormalDx),
                OutputMap::packed("T_$mesh_OcclusionRoughnessMetallic", Layout::Rgb, [ao, rough, metal, C1]),
                OutputMap::rgb("T_$mesh_Emissive", Emission),
                OutputMap::gray("T_$mesh_Specular", Chan::of(SpecularLevel, 0)),
            ],
        ),
        t(
            "Unity URP (metallic)",
            vec![
                OutputMap::rgb_opacity("$mesh_BaseMap", BaseColor),
                OutputMap::rgb("$mesh_Normal", NormalGl),
                OutputMap::packed("$mesh_MaskMap", Layout::Rgba, [metal, ao, C0, smooth]),
                OutputMap::height("$mesh_Height"),
                OutputMap::rgb("$mesh_Emission", Emission),
            ],
        ),
        t(
            "Unity HDRP",
            vec![
                OutputMap::rgb_opacity("$mesh_BaseMap", BaseColor),
                OutputMap::rgb("$mesh_Normal", NormalGl),
                OutputMap::packed("$mesh_MaskMap", Layout::Rgba, [metal, ao, C1, smooth]),
                OutputMap::height("$mesh_Height"),
                OutputMap::rgb("$mesh_Emissive", Emission),
            ],
        ),
        t(
            "Unity Built-in (Standard)",
            vec![
                OutputMap::rgb_opacity("$mesh_Albedo", BaseColor),
                OutputMap::packed("$mesh_MetallicSmoothness", Layout::Rgba, [metal, metal, metal, smooth]),
                OutputMap::rgb("$mesh_Normal", NormalGl),
                OutputMap::gray("$mesh_Occlusion", ao),
                OutputMap::height("$mesh_Height"),
                OutputMap::rgb("$mesh_Emission", Emission),
            ],
        ),
        t(
            "Godot 4 (ORM)",
            vec![
                OutputMap::rgb_opacity("$mesh_Albedo", BaseColor),
                OutputMap::packed("$mesh_ORM", Layout::Rgb, [ao, rough, metal, C1]),
                OutputMap::rgb("$mesh_Normal", NormalGl),
                OutputMap::height("$mesh_Height"),
                OutputMap::rgb("$mesh_Emission", Emission),
            ],
        ),
        t(
            "glTF PBR (metal/rough)",
            vec![
                OutputMap::rgb_opacity("$mesh_baseColor", BaseColor),
                OutputMap::packed("$mesh_occlusionRoughnessMetallic", Layout::Rgb, [ao, rough, metal, C1]),
                OutputMap::rgb("$mesh_normal", NormalGl),
                OutputMap::rgb("$mesh_emissive", Emission),
            ],
        ),
        t(
            "Roblox SurfaceAppearance",
            vec![
                OutputMap::rgb_opacity("$mesh_Color", BaseColor),
                OutputMap::rgb("$mesh_Normal", NormalGl),
                OutputMap::gray("$mesh_Roughness", rough),
                OutputMap::gray("$mesh_Metalness", metal),
            ],
        ),
        t(
            "Blender (Principled BSDF)",
            vec![
                OutputMap::rgb_opacity("$mesh_BaseColor", BaseColor),
                OutputMap::gray("$mesh_Roughness", rough),
                OutputMap::gray("$mesh_Metallic", metal),
                OutputMap::rgb("$mesh_Normal", NormalGl),
                OutputMap::height("$mesh_Height"),
                OutputMap::rgb("$mesh_Emission", Emission),
                OutputMap::gray("$mesh_Specular", Chan::of(SpecularLevel, 0)),
            ],
        ),
        t(
            "PBR Metallic Roughness (separate)",
            vec![
                OutputMap::rgb_opacity("$mesh_BaseColor", BaseColor),
                OutputMap::rgb("$mesh_Normal", NormalGl),
                OutputMap::gray("$mesh_Roughness", rough),
                OutputMap::gray("$mesh_Metallic", metal),
                OutputMap::gray("$mesh_AmbientOcclusion", ao),
                OutputMap::height("$mesh_Height"),
                OutputMap::rgb("$mesh_Emissive", Emission),
                OutputMap::gray("$mesh_Specular", Chan::of(SpecularLevel, 0)),
            ],
        ),
        t(
            "PBR Specular Glossiness",
            vec![
                OutputMap::rgb_opacity("$mesh_Diffuse", Diffuse),
                OutputMap::rgb("$mesh_Specular", Specular),
                OutputMap::gray("$mesh_Glossiness", Chan::of(Glossiness, 0)),
                OutputMap::rgb("$mesh_Normal", NormalGl),
                OutputMap::gray("$mesh_AmbientOcclusion", ao),
                OutputMap::height("$mesh_Height"),
                OutputMap::rgb("$mesh_Emissive", Emission),
            ],
        ),
        t("Baked mesh maps (all)", mesh_maps),
    ]
}

// ------------------------------------------------------------------ settings

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Format {
    Png,
    Tga,
    Jpeg,
    Tiff,
    Exr,
}

impl Format {
    pub const ALL: [Format; 5] = [Format::Png, Format::Tga, Format::Jpeg, Format::Tiff, Format::Exr];
    pub fn name(self) -> &'static str {
        match self {
            Format::Png => "PNG",
            Format::Tga => "TGA",
            Format::Jpeg => "JPEG",
            Format::Tiff => "TIFF",
            Format::Exr => "OpenEXR",
        }
    }
    pub fn ext(self) -> &'static str {
        match self {
            Format::Png => "png",
            Format::Tga => "tga",
            Format::Jpeg => "jpg",
            Format::Tiff => "tif",
            Format::Exr => "exr",
        }
    }
    /// Bit depths the format can store.
    pub fn depths(self) -> &'static [Depth] {
        match self {
            Format::Png | Format::Tiff => &[Depth::U8, Depth::U16],
            Format::Tga | Format::Jpeg => &[Depth::U8],
            Format::Exr => &[Depth::F32],
        }
    }
    pub fn from_name(s: &str) -> Option<Format> {
        Format::ALL.into_iter().find(|f| f.name().eq_ignore_ascii_case(s) || f.ext() == s.to_ascii_lowercase())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Depth {
    U8,
    U16,
    F32,
}

impl Depth {
    pub fn name(self) -> &'static str {
        match self {
            Depth::U8 => "8-bit",
            Depth::U16 => "16-bit",
            Depth::F32 => "32-bit float",
        }
    }
}

/// What happens to texels outside the UV islands.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Padding {
    /// Whatever the layer stack produced there.
    None,
    /// Island edges extended outward by this many pixels (at the document
    /// resolution), the rest left as is.
    Dilate(u32),
    /// Island edges extended over the whole empty space: no seams at any mip.
    Infinite,
    /// Dilated, with alpha 0 outside the islands (RGBA maps).
    Transparent,
}

impl Padding {
    pub fn name(self) -> String {
        match self {
            Padding::None => "None".into(),
            Padding::Dilate(n) => format!("Dilation {n} px"),
            Padding::Infinite => "Dilation infinite".into(),
            Padding::Transparent => "Dilation + transparent".into(),
        }
    }
    pub fn key(self) -> String {
        match self {
            Padding::None => "none".into(),
            Padding::Dilate(n) => format!("dilate:{n}"),
            Padding::Infinite => "infinite".into(),
            Padding::Transparent => "transparent".into(),
        }
    }
    pub fn from_key(s: &str) -> Option<Padding> {
        Some(match s {
            "none" => Padding::None,
            "infinite" => Padding::Infinite,
            "transparent" => Padding::Transparent,
            _ => Padding::Dilate(s.strip_prefix("dilate:")?.parse().ok()?),
        })
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct Settings {
    pub folder: PathBuf,
    /// `$mesh` in file names.
    pub mesh_name: String,
    /// `$project` in file names.
    pub project_name: String,
    /// `$set` in file names: the texture set exported.
    pub set_name: String,
    /// Texture sets in the document. With more than one, names without
    /// `$set` get it after `$mesh`, so sets don't overwrite each other.
    pub set_count: usize,
    pub format: Format,
    pub depth: Depth,
    /// Output size; `None` is the document resolution.
    pub size: Option<u32>,
    pub padding: Padding,
    pub normal_strength: f32,
    pub use_baked_normal: bool,
}

impl Settings {
    /// The bit depth a map is written at.
    fn depth_for(&self, map: &OutputMap) -> Depth {
        let ok = self.format.depths();
        let want = if map.force_16 && self.depth == Depth::U8 { Depth::U16 } else { self.depth };
        if ok.contains(&want) { want } else { ok[ok.len() - 1].min_with(want) }
    }

    pub fn file_name(&self, map: &OutputMap) -> String {
        let mut name = map.name.clone();
        if self.set_count > 1 && !name.contains("$set") {
            name = if name.contains("$mesh") { name.replace("$mesh", "$mesh_$set") } else { format!("$set_{name}") };
        }
        let stem = name.replace("$mesh", &self.mesh_name).replace("$set", &self.set_name).replace("$project", &self.project_name);
        let stem: String = stem.chars().map(|c| if r#"\/:*?"<>|"#.contains(c) { '_' } else { c }).collect();
        format!("{stem}.{}", self.format.ext())
    }
}

impl Depth {
    /// `self` (the format's deepest) or less when less was asked for.
    fn min_with(self, want: Depth) -> Depth {
        match (self, want) {
            (Depth::F32, _) => Depth::F32,
            (Depth::U16, Depth::U8) => Depth::U8,
            (d, _) => d,
        }
    }
}

// ------------------------------------------------------------------ JSON (custom templates)

fn chan_json(c: &Chan) -> Value {
    json!({"src": c.src.key(), "comp": c.comp, "invert": c.invert})
}

fn chan_from(v: &Value) -> Option<Chan> {
    Some(Chan {
        src: Source::from_key(v["src"].as_str()?)?,
        comp: v["comp"].as_u64().unwrap_or(0).min(2) as u8,
        invert: v["invert"].as_bool().unwrap_or(false),
    })
}

pub fn template_json(t: &Template) -> Value {
    json!({
        "name": t.name,
        "maps": t.maps.iter().map(|m| json!({
            "name": m.name,
            "layout": m.layout.name(),
            "channels": m.chans.iter().map(chan_json).collect::<Vec<_>>(),
            "force_16": m.force_16,
        })).collect::<Vec<_>>(),
    })
}

pub fn template_from(v: &Value) -> Option<Template> {
    let maps = v["maps"]
        .as_array()?
        .iter()
        .filter_map(|m| {
            let layout = Layout::ALL.into_iter().find(|l| Some(l.name()) == m["layout"].as_str())?;
            let ch = m["channels"].as_array()?;
            let chans = [0, 1, 2, 3].map(|k| ch.get(k).and_then(chan_from).unwrap_or(if k == 3 { C1 } else { C0 }));
            Some(OutputMap { name: m["name"].as_str()?.to_string(), layout, chans, force_16: m["force_16"].as_bool().unwrap_or(false) })
        })
        .collect();
    Some(Template { name: v["name"].as_str()?.to_string(), maps, builtin: false })
}

// ------------------------------------------------------------------ readback

/// A texture read back as f32 RGBA, whatever its format.
struct Plane {
    data: Vec<[f32; 4]>,
}

fn read_plane(device: &wgpu::Device, queue: &wgpu::Queue, tex: &Tex) -> Plane {
    let raw = gpu::read_texture(device, queue, tex);
    let data = match tex.format {
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => {
            raw.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]].map(|v| v as f32 / 255.0)).collect()
        }
        wgpu::TextureFormat::R8Unorm => raw.iter().map(|&v| [v as f32 / 255.0, 0.0, 0.0, 1.0]).collect(),
        wgpu::TextureFormat::R16Float => raw.chunks_exact(2).map(|b| [f16_to_f32(u16::from_le_bytes([b[0], b[1]])), 0.0, 0.0, 1.0]).collect(),
        wgpu::TextureFormat::Rgba16Float => raw
            .chunks_exact(8)
            .map(|b| [0, 1, 2, 3].map(|k| f16_to_f32(u16::from_le_bytes([b[k * 2], b[k * 2 + 1]]))))
            .collect(),
        wgpu::TextureFormat::Rgba32Float => raw
            .chunks_exact(16)
            .map(|b| [0, 1, 2, 3].map(|k| f32::from_le_bytes([b[k * 4], b[k * 4 + 1], b[k * 4 + 2], b[k * 4 + 3]])))
            .collect(),
        other => panic!("unsupported export format {other:?}"),
    };
    Plane { data }
}

/// Bilinear resample of a square plane to `res` (mesh maps can differ from
/// the document size until they are re-baked).
fn resample(p: Plane, w: u32, res: u32) -> Plane {
    if w == res {
        return p;
    }
    let (w, r) = (w as usize, res as usize);
    let data = (0..r * r)
        .into_par_iter()
        .map(|i| {
            let (x, y) = (i % r, i / r);
            let fx = ((x as f32 + 0.5) / r as f32 * w as f32 - 0.5).clamp(0.0, (w - 1) as f32);
            let fy = ((y as f32 + 0.5) / r as f32 * w as f32 - 0.5).clamp(0.0, (w - 1) as f32);
            let (x0, y0) = (fx as usize, fy as usize);
            let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(w - 1));
            let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
            [0, 1, 2, 3].map(|c| {
                let s = |x: usize, y: usize| p.data[y * w + x][c];
                let top = s(x0, y0) + (s(x1, y0) - s(x0, y0)) * tx;
                let bot = s(x0, y1) + (s(x1, y1) - s(x0, y1)) * tx;
                top + (bot - top) * ty
            })
        })
        .collect();
    Plane { data }
}

/// Everything an export reads from the GPU, owned so the rest can run on a
/// worker thread.
pub struct Snapshot {
    res: u32,
    /// Base color, with opacity in alpha.
    color: Plane,
    /// No layer sets opacity: color maps with opacity alpha are written RGB.
    opaque: bool,
    /// Emission, sRGB-encoded (black where nothing emits).
    emission: Plane,
    /// Some layer emits: only then are emission maps written.
    emissive: bool,
    /// Final tangent-space painted normal, encoded 0..1 (flat where no layer sets it).
    normal: Plane,
    /// Specular level (0.5 where no layer sets it).
    spec: Plane,
    /// Some layer sets specular: only then are specular level maps written.
    specular: bool,
    rough: Plane,
    metal: Plane,
    height: Plane,
    mesh: [Plane; crate::bake::MAP_TEXTURES],
    world_normal: Plane,
    position: Plane,
    bounds: (Vec3, Vec3),
    has_bake: bool,
    /// UV triangles, for the island coverage that padding needs.
    uvs: Arc<Mesh>,
    /// The texture set's range in `uvs.indices`.
    tris: std::ops::Range<u32>,
}

/// Reads the document's maps back from the GPU (call after compositing).
pub fn snapshot(engine: &Engine, device: &wgpu::Device, queue: &wgpu::Queue, mesh: Arc<Mesh>) -> Snapshot {
    let res = engine.res;
    let mesh_maps = std::array::from_fn(|t| {
        let tex = engine.map_tex(t);
        resample(read_plane(device, queue, tex), tex.width, res)
    });
    let (pos, nrm) = engine.surface_maps();
    Snapshot {
        res,
        color: read_plane(device, queue, engine.final_tex(0)),
        opaque: !engine.uses_opacity(),
        emission: read_plane(device, queue, engine.final_tex(crate::engine::FINAL_EMISSION)),
        emissive: engine.uses_emission(),
        normal: read_plane(device, queue, engine.final_tex(crate::engine::FINAL_NORMAL)),
        spec: read_plane(device, queue, engine.final_tex(crate::engine::FINAL_SPECULAR)),
        specular: engine.uses_channel(crate::engine::SPECULAR),
        rough: read_plane(device, queue, engine.final_tex(1)),
        metal: read_plane(device, queue, engine.final_tex(2)),
        height: read_plane(device, queue, engine.final_tex(3)),
        mesh: mesh_maps,
        world_normal: resample(read_plane(device, queue, nrm), nrm.width, res),
        position: resample(read_plane(device, queue, pos), pos.width, res),
        bounds: engine.bounds(),
        has_bake: engine.has_bake,
        uvs: mesh,
        tris: engine.triangles(),
    }
}

// ------------------------------------------------------------------ evaluation

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

/// Source values per texel, computed on first use.
struct Sources<'a> {
    s: &'a Snapshot,
    settings: &'a Settings,
    /// Float files hold linear color; integer files sRGB.
    linear: bool,
    cache: HashMap<Source, Arc<Vec<[f32; 3]>>>,
}

impl Sources<'_> {
    fn get(&mut self, src: Source) -> Arc<Vec<[f32; 3]>> {
        if let Some(p) = self.cache.get(&src) {
            return p.clone();
        }
        let p = Arc::new(self.compute(src));
        self.cache.insert(src, p.clone());
        p
    }

    fn compute(&mut self, src: Source) -> Vec<[f32; 3]> {
        let s = self.s;
        let n = (s.res * s.res) as usize;
        let enc = |lin: [f32; 3], linear: bool| if linear { lin } else { lin.map(linear_to_srgb) };
        let base_lin = |i: usize| {
            let c = s.color.data[i];
            [c[0], c[1], c[2]].map(srgb_to_linear)
        };
        let linear = self.linear;
        match src {
            Source::BaseColor => (0..n).into_par_iter().map(|i| enc(base_lin(i), linear)).collect(),
            Source::Diffuse => (0..n)
                .into_par_iter()
                .map(|i| {
                    let m = s.metal.data[i][0];
                    enc(base_lin(i).map(|c| c * (1.0 - m)), linear)
                })
                .collect(),
            Source::Specular => (0..n)
                .into_par_iter()
                .map(|i| {
                    let m = s.metal.data[i][0];
                    // Non-metals reflect 8% times the specular level (4% at 0.5).
                    let f0 = 0.08 * s.spec.data[i][0];
                    enc(base_lin(i).map(|c| f0 + (c - f0) * m), linear)
                })
                .collect(),
            Source::SpecularLevel => (0..n).map(|i| [s.spec.data[i][0]; 3]).collect(),
            Source::Roughness => (0..n).map(|i| [s.rough.data[i][0]; 3]).collect(),
            Source::Glossiness => (0..n).map(|i| [1.0 - s.rough.data[i][0]; 3]).collect(),
            Source::Metallic => (0..n).map(|i| [s.metal.data[i][0]; 3]).collect(),
            Source::Height => (0..n).map(|i| [s.height.data[i][0] * 0.5 + 0.5; 3]).collect(),
            Source::Opacity => (0..n).map(|i| [s.color.data[i][3]; 3]).collect(),
            Source::Emission => (0..n)
                .into_par_iter()
                .map(|i| {
                    let c = s.emission.data[i];
                    enc([c[0], c[1], c[2]].map(srgb_to_linear), linear)
                })
                .collect(),
            Source::NormalGl | Source::NormalDx => {
                let use_baked = self.settings.use_baked_normal && s.has_bake;
                normal_map(s, self.settings.normal_strength, use_baked, true, src == Source::NormalDx)
            }
            Source::Mesh(i) => {
                let info = &MAPS[i];
                if info.kind == MapKind::Normal {
                    return normal_map(s, 0.0, true, false, false);
                }
                let p = &s.mesh[info.tex];
                if info.kind == MapKind::Scalar {
                    (0..n).map(|t| [p.data[t][info.comp]; 3]).collect()
                } else {
                    (0..n).map(|t| [p.data[t][0], p.data[t][1], p.data[t][2]]).collect()
                }
            }
            // Stored encoded (0..1) already.
            Source::WorldNormal => (0..n).map(|i| {
                let v = s.world_normal.data[i];
                [v[0], v[1], v[2]]
            }).collect(),
            Source::Position => {
                let (lo, hi) = s.bounds;
                let size = (hi - lo).max(Vec3::splat(1e-4));
                (0..n)
                    .map(|i| {
                        let v = s.position.data[i];
                        [(v[0] - lo.x) / size.x, (v[1] - lo.y) / size.y, (v[2] - lo.z) / size.z]
                    })
                    .collect()
            }
            Source::Zero => vec![[0.0; 3]; n],
            Source::One => vec![[1.0; 3]; n],
        }
    }
}

/// Final tangent-space normal, encoded 0..1: baked normal plus painted
/// height detail, matching the viewport.
fn normal_map(m: &Snapshot, strength: f32, use_baked: bool, painted: bool, directx: bool) -> Vec<[f32; 3]> {
    let r = m.res as i64;
    let h = |x: i64, y: i64| m.height.data[(y.clamp(0, r - 1) * r + x.clamp(0, r - 1)) as usize][0];
    let k = strength * m.res as f32 / 256.0;
    (0..(r * r) as usize)
        .into_par_iter()
        .map(|i| {
            let (x, y) = ((i as i64) % r, (i as i64) / r);
            let hu = (h(x + 1, y) - h(x - 1, y)) * 0.5 * k;
            let ht = (h(x, y + 1) - h(x, y - 1)) * 0.5 * k;
            let d = m.mesh[3].data[i];
            let mut base = if use_baked { Vec3::new(d[0], d[1], (1.0 - d[0] * d[0] - d[1] * d[1]).max(0.0).sqrt()) } else { Vec3::Z };
            // The painted normal on the baked one (whiteout), as the view does.
            if painted {
                let p = m.normal.data[i];
                let p = Vec3::new(p[0] * 2.0 - 1.0, p[1] * 2.0 - 1.0, p[2] * 2.0 - 1.0).normalize_or(Vec3::Z);
                base = Vec3::new(base.x + p.x, base.y + p.y, base.z * p.z).normalize_or(Vec3::Z);
            }
            let mut n = Vec3::new(base.x - hu, base.y + ht, base.z).normalize();
            if directx {
                n.y = -n.y;
            }
            [n.x, n.y, n.z].map(|v| v * 0.5 + 0.5)
        })
        .collect()
}

// ------------------------------------------------------------------ padding

/// Texels inside a UV triangle (sampled at texel centers, both windings).
fn coverage(mesh: &Mesh, tris: std::ops::Range<u32>, res: u32) -> Vec<bool> {
    let r = res as usize;
    let mut out = vec![false; r * r];
    let indices = mesh.indices.get(tris.start as usize..tris.end as usize).unwrap_or(&mesh.indices);
    for tri in indices.chunks_exact(3) {
        let p = [0, 1, 2].map(|k| Vec2::from_array(mesh.vertices[tri[k] as usize].uv) * res as f32);
        let lo = p[0].min(p[1]).min(p[2]).floor().max(Vec2::ZERO);
        let hi = p[0].max(p[1]).max(p[2]).ceil().min(Vec2::splat(res as f32));
        let area = (p[1] - p[0]).perp_dot(p[2] - p[0]);
        if area.abs() < 1e-12 {
            continue;
        }
        for y in lo.y as usize..hi.y as usize {
            for x in lo.x as usize..hi.x as usize {
                let c = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                let e = |a: Vec2, b: Vec2| (b - a).perp_dot(c - a) * area.signum();
                if e(p[0], p[1]) >= 0.0 && e(p[1], p[2]) >= 0.0 && e(p[2], p[0]) >= 0.0 {
                    out[y * r + x] = true;
                }
            }
        }
    }
    out
}

const NO_SOURCE: u32 = u32::MAX;

/// For each texel, the island texel its value is copied from: itself inside
/// the islands, the nearest island texel (grown ring by ring) outside, up to
/// `limit` rings.
fn nearest_covered(cover: &[bool], res: u32, limit: u32) -> Vec<u32> {
    let r = res as i64;
    let mut src: Vec<u32> = cover.iter().enumerate().map(|(i, &c)| if c { i as u32 } else { NO_SOURCE }).collect();
    let mut frontier: Vec<u32> = (0..cover.len() as u32).filter(|&i| cover[i as usize]).collect();
    let mut ring = 0;
    while !frontier.is_empty() && ring < limit {
        let mut next = Vec::new();
        for &i in &frontier {
            let (x, y) = (i as i64 % r, i as i64 / r);
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)] {
                let (nx, ny) = (x + dx, y + dy);
                if nx < 0 || ny < 0 || nx >= r || ny >= r {
                    continue;
                }
                let j = (ny * r + nx) as usize;
                if src[j] == NO_SOURCE {
                    src[j] = src[i as usize];
                    next.push(j as u32);
                }
            }
        }
        frontier = next;
        ring += 1;
    }
    src
}
/// Copies dilated texels in place. Each source index points at an original
/// covered texel, never another padded texel, so source pixels remain stable
/// while destinations are filled. This avoids cloning the full RGBA image.
fn apply_padding(pixels: &mut [f32], near: &[u32], cover: &[bool], transparent: bool) {
    debug_assert_eq!(pixels.len(), cover.len() * 4);
    debug_assert_eq!(near.len(), cover.len());
    for (i, &source) in near.iter().enumerate() {
        let dst = i * 4;
        if source != NO_SOURCE {
            let src = source as usize * 4;
            let sample = [pixels[src], pixels[src + 1], pixels[src + 2], pixels[src + 3]];
            pixels[dst..dst + 4].copy_from_slice(&sample);
        }
        if transparent && !cover[i] {
            pixels[dst + 3] = 0.0;
        }
    }
}


// ------------------------------------------------------------------ writing

pub fn export(s: &Snapshot, template: &Template, settings: &Settings) -> Result<Vec<PathBuf>, String> {
    if template.maps.is_empty() {
        return Err("The template has no output maps.".into());
    }
    // A template of nothing but baked maps (the mesh maps preset) needs a bake.
    let only_baked = template.maps.iter().all(|m| m.chans[..m.layout.channels()].iter().all(|c| matches!(c.src, Source::Mesh(_))));
    if only_baked && !s.has_bake {
        return Err("Bake mesh maps first.".into());
    }
    std::fs::create_dir_all(&settings.folder).map_err(|e| format!("Cannot create folder: {e}"))?;
    let res = s.res;
    let side = usize::try_from(res).map_err(|_| "Export resolution is too large.".to_string())?;
    let n = side.checked_mul(side).ok_or_else(|| "Export pixel count overflow.".to_string())?;
    let components = n.checked_mul(4).ok_or_else(|| "Export buffer size overflow.".to_string())?;
    let padding = match settings.padding {
        Padding::None => None,
        Padding::Dilate(px) => Some(px),
        Padding::Infinite | Padding::Transparent => Some(u32::MAX),
    };
    let cover = padding.map(|_| coverage(&s.uvs, s.tris.clone(), res));
    let nearest = padding.zip(cover.as_ref()).map(|(limit, c)| nearest_covered(c, res, limit));

    // Emission maps of a texture set that emits nothing are left out, and
    // specular level maps of one that keeps the default.
    let maps: Vec<&OutputMap> = template.maps.iter().filter(|m| !m.left_out(s)).collect();
    // Evaluate every needed source once, then build the maps in parallel.
    let linear = settings.format == Format::Exr;
    let mut sources = Sources { s, settings, linear, cache: HashMap::new() };
    for m in &maps {
        for c in &m.chans[..m.layout.channels()] {
            sources.get(c.src);
        }
    }
    let cache = sources.cache;
    let out_size = settings.size.unwrap_or(res);
    let opaque = s.opaque;

    // Keep the source maps resident, but build one output map at a time.
    // Exporting several 4K/8K maps concurrently multiplied large temporary
    // RGBA buffers and could exhaust system RAM even when each map fit alone.
    maps.iter()
        .map(|&map| {
            // A color whose alpha is only opacity stays RGB on an opaque texture set.
            let rgb = (map.alpha_is_opacity() && opaque).then(|| OutputMap { layout: Layout::Rgb, ..map.clone() });
            let map = rgb.as_ref().unwrap_or(map);
            let chans = map.layout.channels();

            // This is the exact flat storage ImageBuffer consumes below. Keeping
            // it flat avoids allocating Vec<[f32; 4]> and then flattening into
            // a second equally large Vec<f32>.
            let mut px = vec![0.0f32; components];
            px.par_chunks_exact_mut(4).for_each(|p| p[3] = 1.0);
            for (k, c) in map.chans[..chans].iter().enumerate() {
                let plane = &cache[&c.src];
                let comp = (c.comp as usize).min(2);
                px.par_chunks_exact_mut(4).enumerate().for_each(|(i, p)| {
                    let v = plane[i][comp];
                    p[k] = if c.invert { 1.0 - v } else { v };
                });
            }
            if map.layout == Layout::Gray {
                px.par_chunks_exact_mut(4).for_each(|p| {
                    p[1] = p[0];
                    p[2] = p[0];
                    p[3] = 1.0;
                });
            }
            if let Some(near) = &nearest {
                let cover = cover.as_ref().expect("with nearest");
                apply_padding(&mut px, near, cover, settings.padding == Padding::Transparent && map.layout == Layout::Rgba);
            }
            let mut img: ImageBuffer<Rgba<f32>, Vec<f32>> =
                ImageBuffer::from_vec(res, res, px).expect("sized");
            if out_size != res {
                let filter = if out_size < res { image::imageops::FilterType::Triangle } else { image::imageops::FilterType::CatmullRom };
                img = image::imageops::resize(&img, out_size, out_size, filter);
                if map.chans[..chans.min(3)].iter().any(|c| c.src.is_normal()) {
                    for p in img.pixels_mut() {
                        let v = Vec3::new(p[0], p[1], p[2]) * 2.0 - Vec3::ONE;
                        let v = v.normalize_or(Vec3::Z) * 0.5 + Vec3::splat(0.5);
                        (p[0], p[1], p[2]) = (v.x, v.y, v.z);
                    }
                }
            }
            let path = settings.folder.join(settings.file_name(map));
            write(&img, map, settings, &path).map_err(|e| format!("Failed to write {}: {e}", path.display()))?;
            Ok(path)
        })
        .collect()
}

fn write(img: &ImageBuffer<Rgba<f32>, Vec<f32>>, map: &OutputMap, settings: &Settings, path: &Path) -> image::ImageResult<()> {
    let (w, h) = img.dimensions();
    let u8c = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    let u16c = |v: f32| (v.clamp(0.0, 1.0) * 65535.0).round() as u16;
    // JPEG has no alpha.
    let layout = if settings.format == Format::Jpeg && map.layout == Layout::Rgba { Layout::Rgb } else { map.layout };
    match settings.depth_for(map) {
        Depth::U8 => {
            let save = |dyn_img: image::DynamicImage| -> image::ImageResult<()> {
                if settings.format == Format::Jpeg {
                    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
                    // `encode_image` would expand grayscale to RGB.
                    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(file, 95);
                    enc.encode(dyn_img.as_bytes(), dyn_img.width(), dyn_img.height(), dyn_img.color().into())
                } else {
                    dyn_img.save(path)
                }
            };
            match layout {
                Layout::Gray => save(ImageBuffer::from_fn(w, h, |x, y| Luma([u8c(img.get_pixel(x, y)[0])])).into()),
                Layout::Rgb => save(ImageBuffer::from_fn(w, h, |x, y| {
                    let p = img.get_pixel(x, y);
                    Rgb([u8c(p[0]), u8c(p[1]), u8c(p[2])])
                }).into()),
                Layout::Rgba => save(ImageBuffer::from_fn(w, h, |x, y| Rgba(img.get_pixel(x, y).0.map(u8c))).into()),
            }
        }
        Depth::U16 => match layout {
            Layout::Gray => ImageBuffer::from_fn(w, h, |x, y| Luma([u16c(img.get_pixel(x, y)[0])])).save(path),
            Layout::Rgb => ImageBuffer::from_fn(w, h, |x, y| {
                let p = img.get_pixel(x, y);
                Rgb([u16c(p[0]), u16c(p[1]), u16c(p[2])])
            })
            .save(path),
            Layout::Rgba => ImageBuffer::from_fn(w, h, |x, y| Rgba(img.get_pixel(x, y).0.map(u16c))).save(path),
        },
        // EXR stores RGB or RGBA floats; grayscale goes in all three.
        Depth::F32 => match layout {
            Layout::Gray | Layout::Rgb => ImageBuffer::from_fn(w, h, |x, y| {
                let p = img.get_pixel(x, y);
                Rgb([p[0], p[1], p[2]])
            })
            .save(path),
            Layout::Rgba => img.save(path),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn templates_round_trip_through_json() {
        for t in builtin_templates() {
            let back = template_from(&template_json(&t)).expect("parses");
            assert_eq!(back.maps, t.maps, "{}", t.name);
        }
    }

    #[test]
    fn dilation_reaches_the_whole_image() {
        let res = 16;
        let mut cover = vec![false; 256];
        cover[5 * 16 + 5] = true;
        let near = nearest_covered(&cover, res, u32::MAX);
        assert!(near.iter().all(|&s| s == 5 * 16 + 5));
        let near = nearest_covered(&cover, res, 2);
        assert_eq!(near.iter().filter(|&&s| s != NO_SOURCE).count(), 25);
    }

    #[test]
    fn in_place_padding_matches_copy_based_padding() {
        let res = 8;
        let mut cover = vec![false; (res * res) as usize];
        cover[1 * res as usize + 1] = true;
        cover[1 * res as usize + 6] = true;
        cover[6 * res as usize + 3] = true;
        for limit in [0, 1, 2, u32::MAX] {
            let near = nearest_covered(&cover, res, limit);
            let original: Vec<f32> = (0..cover.len() * 4).map(|i| (i as f32 + 1.0) / 257.0).collect();

            let mut expected = original.clone();
            for i in 0..cover.len() {
                if near[i] != NO_SOURCE {
                    let src = near[i] as usize * 4;
                    let dst = i * 4;
                    let sample = [original[src], original[src + 1], original[src + 2], original[src + 3]];
                    expected[dst..dst + 4].copy_from_slice(&sample);
                }
                if !cover[i] {
                    expected[i * 4 + 3] = 0.0;
                }
            }

            let mut actual = original;
            apply_padding(&mut actual, &near, &cover, true);
            assert_eq!(actual, expected, "limit {limit}");
        }
    }

    #[test]
    fn depth_follows_the_format() {
        let mut s = Settings {
            folder: PathBuf::new(),
            mesh_name: "M".into(),
            project_name: "P".into(),
            set_name: "S".into(),
            set_count: 1,
            format: Format::Png,
            depth: Depth::U8,
            size: None,
            padding: Padding::Infinite,
            normal_strength: 1.0,
            use_baked_normal: true,
        };
        let h = OutputMap::height("$mesh_Height");
        assert_eq!(s.depth_for(&h), Depth::U16);
        s.format = Format::Tga;
        assert_eq!(s.depth_for(&h), Depth::U8);
        s.format = Format::Exr;
        assert_eq!(s.depth_for(&h), Depth::F32);
        assert_eq!(s.file_name(&h), "M_Height.exr");
    }
}
