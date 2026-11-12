//! The texture-set engine: layer stack data, GPU painting and compositing.
//!
//! Memory model:
//! * Paint layers allocate a channel texture only once that channel is painted,
//!   and each channel uses its own compact format (RGBA8 color, RG8 roughness,
//!   metallic and opacity, RG16F height; the second component is the channel's
//!   coverage).
//! * Opacity composites into the base color's alpha: the final color texture
//!   is RGB plus opacity (1, opaque, until a layer sets it).
//! * Emission is an RGBA8 channel like base color (sRGB rgb, coverage); its
//!   final texture is black until a layer emits.
//! * Normal is an RGBA8 channel of tangent-space normals (encoded 0..1,
//!   coverage); each layer's normals add detail to the ones below rather
//!   than replacing them. Its final texture is flat until a layer sets it.
//! * Specular is an RG8 channel like roughness: the specular level of
//!   non-metals, 0.5 for the usual 4% reflectance.
//! * Fill layers and mask generators are procedural: zero texture memory.
//! * Stroke history lives compressed in system RAM (see `undo.rs`).

use crate::bake::{BakeReport, BakedMaps, CUSTOM_FORMAT, CUSTOM_PER_TEXTURE, MAP_FORMATS, MAP_TEXTURES};
use crate::gpu::{self, GpuMesh, LAYER_USAGE, Tex};
use crate::undo::{DocProps, Edit, LayerProps, Snapshot, StrokeTarget, UndoEntry, UndoStack};
use eframe::{egui, egui_wgpu, wgpu};
use glam::Vec3;
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::{mpsc, Arc};
use wgpu::TextureFormat as F;

/// Paint channels: base color, roughness, metallic, height, opacity,
/// emission, normal, specular.
pub const CHANNELS: usize = 8;
/// The opacity channel (composited into the final base color's alpha).
pub const OPACITY: usize = 4;
/// The emission channel: light the surface gives off (sRGB color).
pub const EMISSION: usize = 5;
/// The normal channel: tangent-space normal detail (OpenGL convention, encoded 0..1).
pub const NORMAL: usize = 6;
/// The specular channel: specular level of non-metals (0.5: 4% reflectance).
pub const SPECULAR: usize = 7;
pub const CH_NAMES: [&str; CHANNELS] = ["Base Color", "Roughness", "Metallic", "Height", "Opacity", "Emission", "Normal", "Specular"];
const PAINT_FORMATS: [F; CHANNELS] =
    [F::Rgba8Unorm, F::Rg8Unorm, F::Rg8Unorm, F::Rg16Float, F::Rg8Unorm, F::Rgba8Unorm, F::Rgba8Unorm, F::Rg8Unorm];
/// Final channels: color (+ opacity), roughness, metallic, height, emission, normal, specular.
pub const ACCUM: usize = 7;
/// The final emission texture's index (see [`Engine::final_tex`]).
pub const FINAL_EMISSION: usize = 4;
/// The final normal texture's index: tangent-space, encoded 0..1 (flat: 0.5, 0.5, 1).
pub const FINAL_NORMAL: usize = 5;
/// The final specular level's index (0.5 where no layer sets it).
pub const FINAL_SPECULAR: usize = 6;
const ACCUM_FORMATS: [F; ACCUM] = [F::Rgba8Unorm, F::R8Unorm, F::R8Unorm, F::R16Float, F::Rgba8Unorm, F::Rgba8Unorm, F::R8Unorm];
const POS_FORMAT: F = F::Rgba16Float;
const NRM_FORMAT: F = F::Rgba8Unorm;
/// Tangent and bitangent sign per texel (see `uvbake.wgsl`).
const TAN_FORMAT: F = F::Rgba8Unorm;
const MASK_FORMAT: F = F::R8Unorm;
/// Filtered masks keep 16 bits, so blur and levels don't band.
const MASK_FILTER_FORMAT: F = F::R16Float;
/// Uniform slots for mask filter steps per composite (dynamic offsets).
const MAX_MASK_FILTER_STEPS: u64 = 256;
const MASK_FILTER_STRIDE: u64 = 256;
const STROKE_FORMAT: F = F::R16Float;
pub const MAX_STAMPS: usize = 512;

/// One brush stamp, as `stamp.wgsl` reads it: screen xy, radius scale,
/// rotation; world center, world radius (0: off the mesh); normal, uv radius;
/// surface x axis; uv, flow scale.
pub type Stamp = [[f32; 4]; 5];
/// Per-layer slot in the dynamic-offset uniform buffer: the uniform rounded up
/// to the 256-byte offset alignment wgpu guarantees.
const UNIFORM_STRIDE: u64 = (size_of::<LayerUniform>() as u64).next_multiple_of(256);
const DILATE_PASSES: u32 = 8;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Blend {
    Normal,
    Multiply,
    Screen,
    Overlay,
    Add,
    Subtract,
    Darken,
    Lighten,
}

impl Blend {
    pub const ALL: [Blend; 8] = [
        Blend::Normal,
        Blend::Multiply,
        Blend::Screen,
        Blend::Overlay,
        Blend::Add,
        Blend::Subtract,
        Blend::Darken,
        Blend::Lighten,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Blend::Normal => "Normal",
            Blend::Multiply => "Multiply",
            Blend::Screen => "Screen",
            Blend::Overlay => "Overlay",
            Blend::Add => "Linear Dodge (Add)",
            Blend::Subtract => "Subtract",
            Blend::Darken => "Darken",
            Blend::Lighten => "Lighten",
        }
    }
    pub fn short(self) -> &'static str {
        match self {
            Blend::Normal => "Norm",
            Blend::Multiply => "Mul",
            Blend::Screen => "Scrn",
            Blend::Overlay => "Ovr",
            Blend::Add => "LDdg",
            Blend::Subtract => "Sub",
            Blend::Darken => "Dark",
            Blend::Lighten => "Lght",
        }
    }
}

/// Material values shared by the brush and fill layers. Color is sRGB.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MatValues {
    pub enabled: [bool; CHANNELS],
    pub color: [f32; 3],
    pub rough: f32,
    pub metal: f32,
    pub height: f32,
    /// 1 opaque, 0 fully transparent.
    pub opacity: f32,
    /// Emitted light, sRGB (black: none).
    pub emission: [f32; 3],
    /// How strongly a normal-map image bends the surface (1: as authored).
    /// Without an image the normal channel is flat and changes nothing.
    pub normal: f32,
    /// Specular level of non-metals: 0.5 is the usual 4% reflectance, 0
    /// none, 1 twice as much.
    pub specular: f32,
}

impl Default for MatValues {
    fn default() -> Self {
        Self {
            enabled: [true, true, true, false, false, false, false, false],
            color: [0.8, 0.8, 0.8],
            rough: 0.5,
            metal: 0.0,
            height: 0.0,
            opacity: 1.0,
            emission: [0.0; 3],
            normal: 1.0,
            specular: 0.5,
        }
    }
}

/// Channel flags from the first four (base color, roughness, metallic,
/// height): the later channels off, as everything made before they existed.
pub fn channels4(c: [bool; 4]) -> [bool; CHANNELS] {
    [c[0], c[1], c[2], c[3], false, false, false, false]
}

/// Inputs a smart mask mixes, in the order `mask_input()` in wear.wgsl expects.
pub const GEN_INPUTS: [(&str, &str); GEN_INPUT_COUNT] = [
    ("Edges", "Within the edge width of a convex edge (baked edge distance)"),
    ("Convexity", "Convex curvature"),
    ("Concavity", "Concave curvature"),
    ("Occlusion", "Inverted ambient occlusion"),
    ("Cavity", "Small-scale occlusion: seams, gaps, engravings"),
    ("Sky exposure", "Open to the sky above: where dust, snow and rain land"),
    ("Facing up", "Surface normal pointing up"),
    ("Thin areas", "Inverted thickness: thin parts, translucency"),
    ("Part gradient", "Bottom to top within each mesh part"),
    ("World height", "Bottom to top over the whole mesh"),
    ("Part random", "One random value per mesh part"),
    ("Creases", "Within the edge width of a concave crease"),
    ("Streaks", "Vertical streak noise (rain marks, drips)"),
    ("Noise", "Fractal noise"),
    ("ID match", "Matches the ID color"),
    ("Overhangs", "Surface normal pointing down"),
    ("Wood grain", "Rings and fibers running along the planks (the long axis, vertical on end faces)"),
    ("Cells", "Distance to the nearest cell center: pits, hammer dents, leather pebbling, cracks (invert)"),
    ("Scratches", "Fine scratches in random directions"),
    ("Front distance", "Angle from the model's front (+Z) around its center: 0 straight ahead, 1 behind. Rings for eyes, lenses, dials"),
];

pub const GEN_INPUT_COUNT: usize = 20;

/// Custom bake maps one smart mask can mix (any of the project's, by id).
pub const MAX_CUSTOM_INPUTS: usize = 4;
/// Vertex colors one ID mask can select at once (`Generator::id_color` plus `id_more`).
pub const MAX_ID_COLORS: usize = 8;

/// A smart mask input reading a custom bake map.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct CustomInput {
    /// The map's `bake::CustomMap::id` (0: unused).
    pub id: u32,
    pub weight: f32,
}

/// Mesh Map values (graph nodes) from here on are custom bake maps: the
/// value minus this is the map's id. Below it they index [`MAPS`].
pub const CUSTOM_MAP_VALUE: usize = 1000;

/// Indices into [`GEN_INPUTS`].
mod thumbs;

pub mod gin {
    pub const EDGE: usize = 0;
    pub const CONVEX: usize = 1;
    pub const CONCAVE: usize = 2;
    pub const OCCLUSION: usize = 3;
    pub const CAVITY: usize = 4;
    pub const SKY: usize = 5;
    pub const UP: usize = 6;
    pub const THIN: usize = 7;
    pub const PART_GRADIENT: usize = 8;
    pub const WORLD_HEIGHT: usize = 9;
    pub const PART_RANDOM: usize = 10;
    pub const CREASE: usize = 11;
    pub const STREAKS: usize = 12;
    pub const NOISE: usize = 13;
    pub const ID: usize = 14;
    pub const OVERHANG: usize = 15;
    pub const GRAIN: usize = 16;
    pub const CELLS: usize = 17;
    pub const SCRATCHES: usize = 18;
    pub const FRONT: usize = 19;
}

/// Polygon fill triangles in UV space: the selected faces, and strips
/// past their island borders that pad the seams (drawn only outside every
/// island).
#[derive(Default)]
pub struct PolyGeometry {
    pub faces: Vec<[f32; 2]>,
    pub padding: Vec<[f32; 2]>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GenKind {
    EdgeWear,
    Dirt,
    Dust,
    RainStreaks,
    Drips,
    CavityGrime,
    CreaseGrime,
    ThinAreas,
    PartGradient,
    PartRandom,
    Clouds,
    PositionGradient,
    IdMask,
    CustomMap,
}

impl GenKind {
    pub const ALL: [GenKind; 14] = [
        GenKind::EdgeWear,
        GenKind::Dirt,
        GenKind::Dust,
        GenKind::RainStreaks,
        GenKind::Drips,
        GenKind::CavityGrime,
        GenKind::CreaseGrime,
        GenKind::ThinAreas,
        GenKind::PartGradient,
        GenKind::PartRandom,
        GenKind::Clouds,
        GenKind::PositionGradient,
        GenKind::IdMask,
        GenKind::CustomMap,
    ];
    pub fn name(self) -> &'static str {
        match self {
            GenKind::EdgeWear => "Edge Wear",
            GenKind::Dirt => "Dirt",
            GenKind::Dust => "Dust",
            GenKind::RainStreaks => "Rain Streaks",
            GenKind::Drips => "Drips & Bleeding",
            GenKind::CavityGrime => "Cavity Grime",
            GenKind::CreaseGrime => "Crease Grime",
            GenKind::ThinAreas => "Thin Areas",
            GenKind::PartGradient => "Part Gradient",
            GenKind::PartRandom => "Random Parts",
            GenKind::Clouds => "Clouds",
            GenKind::PositionGradient => "World Gradient",
            GenKind::IdMask => "Vertex Color ID Mask",
            GenKind::CustomMap => "Custom Bake Map",
        }
    }
    pub fn default_generator(self) -> Generator {
        use gin::*;
        let mut w = [0.0f32; GEN_INPUT_COUNT];
        let (level, contrast, scale, noise) = match self {
            GenKind::EdgeWear => {
                w[EDGE] = 1.0;
                w[CONVEX] = 0.5;
                (0.55, 0.1, 4.0, 0.6)
            }
            GenKind::Dirt => {
                w[OCCLUSION] = 1.0;
                w[CAVITY] = 0.8;
                w[CONCAVE] = 0.4;
                (0.4, 0.15, 3.0, 0.45)
            }
            GenKind::Dust => {
                w[SKY] = 0.8;
                w[UP] = 0.7;
                (0.95, 0.15, 3.0, 0.35)
            }
            GenKind::RainStreaks => {
                w[SKY] = 0.4;
                w[STREAKS] = 1.0;
                w[OVERHANG] = -0.4;
                (0.95, 0.12, 3.0, 0.15)
            }
            GenKind::Drips => {
                w[CREASE] = 0.6;
                w[STREAKS] = 0.7;
                w[CAVITY] = 0.4;
                w[OVERHANG] = 0.3;
                (0.75, 0.1, 3.0, 0.25)
            }
            GenKind::CavityGrime => {
                w[CAVITY] = 1.5;
                (0.25, 0.1, 4.0, 0.2)
            }
            GenKind::CreaseGrime => {
                w[CREASE] = 1.0;
                w[CONCAVE] = 0.4;
                (0.45, 0.15, 4.0, 0.35)
            }
            GenKind::ThinAreas => {
                w[THIN] = 1.0;
                (0.55, 0.2, 3.0, 0.0)
            }
            GenKind::PartGradient => {
                w[PART_GRADIENT] = 1.0;
                (0.5, 0.5, 3.0, 0.0)
            }
            GenKind::PartRandom => {
                w[PART_RANDOM] = 1.0;
                (0.5, 0.01, 3.0, 0.0)
            }
            GenKind::Clouds => {
                w[NOISE] = 1.0;
                (0.5, 0.1, 2.5, 0.0)
            }
            GenKind::PositionGradient => {
                w[WORLD_HEIGHT] = 1.0;
                (0.5, 0.3, 2.0, 0.2)
            }
            GenKind::IdMask => {
                w[ID] = 1.0;
                (0.5, 0.05, 3.0, 0.0)
            }
            // Reads a custom map the user picks (see `Generator::custom`).
            GenKind::CustomMap => (0.35, 0.15, 3.0, 0.3),
        };
        Generator {
            kind: self,
            weights: w,
            level,
            contrast,
            scale,
            noise,
            invert: false,
            seed: 0.0,
            edge_width: 0.04,
            id_color: [1.0, 0.0, 0.0],
            id_tolerance: 0.15,
            id_more: [[0.0; 3]; MAX_ID_COLORS - 1],
            id_count: 1,
            custom: [CustomInput::default(); MAX_CUSTOM_INPUTS],
        }
    }
}

/// A smart mask: a weighted mix of mesh-map inputs, then levels.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Generator {
    /// Preset this generator started from (see `needs_maps` for what it reads).
    pub kind: GenKind,
    pub weights: [f32; GEN_INPUT_COUNT],
    pub level: f32,
    pub contrast: f32,
    pub scale: f32,
    pub noise: f32,
    pub invert: bool,
    pub seed: f32,
    /// World-space width for the Edges and Creases inputs.
    pub edge_width: f32,
    /// The first ID color (the one old projects and tools know).
    pub id_color: [f32; 3],
    pub id_tolerance: f32,
    /// More ID colors; the first `id_count - 1` are in use.
    pub id_more: [[f32; 3]; MAX_ID_COLORS - 1],
    /// How many ID colors the mask selects (1 to `MAX_ID_COLORS`).
    pub id_count: u32,
    /// Custom bake maps mixed in, by id.
    pub custom: [CustomInput; MAX_CUSTOM_INPUTS],
}

impl Generator {
    /// The ID colors the mask selects.
    pub fn id_colors(&self) -> Vec<[f32; 3]> {
        let n = (self.id_count as usize).clamp(1, MAX_ID_COLORS);
        std::iter::once(self.id_color).chain(self.id_more.iter().copied()).take(n).collect()
    }

    /// Replaces the selected ID colors (at most `MAX_ID_COLORS`, at least one).
    pub fn set_id_colors(&mut self, colors: &[[f32; 3]]) {
        let colors = &colors[..colors.len().min(MAX_ID_COLORS)];
        let Some(first) = colors.first() else { return };
        self.id_color = *first;
        self.id_more = [[0.0; 3]; MAX_ID_COLORS - 1];
        for (slot, c) in self.id_more.iter_mut().zip(&colors[1..]) {
            *slot = *c;
        }
        self.id_count = colors.len() as u32;
    }

    /// The custom bake maps this generator reads (by id).
    pub fn custom_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.custom.iter().filter(|c| c.id != 0 && c.weight.abs() > 1e-4).map(|c| c.id)
    }

    /// The baked maps this generator reads (a bit per `MAPS` index).
    pub fn needs_maps(&self) -> u32 {
        // Generator input -> the map it comes from.
        const FROM: [(usize, usize); 10] = [
            (gin::EDGE, 5),
            (gin::CONVEX, 1),
            (gin::CONCAVE, 1),
            (gin::OCCLUSION, 0),
            (gin::CAVITY, 2),
            (gin::SKY, 4),
            (gin::THIN, 3),
            (gin::CREASE, 6),
            (gin::PART_GRADIENT, 8),
            (gin::PART_RANDOM, 7),
        ];
        FROM.iter().filter(|(i, _)| self.weights[*i].abs() > 1e-4).fold(0, |acc, (_, m)| acc | (1 << m))
    }
}

/// Names of the maps in a bit set, for messages.
pub fn map_names(bits: u32) -> String {
    MAPS.iter().enumerate().filter(|(i, _)| bits & (1 << i) != 0).map(|(_, m)| m.name).collect::<Vec<_>>().join(", ")
}

// ---------------------------------------------------------------- mesh maps

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MapKind {
    /// Single value; supports levels and painted fixes.
    Scalar,
    /// Tangent-space normal (x, y); a painted fix flattens it.
    Normal,
    /// Display and export only.
    Info,
}

pub struct MapInfo {
    pub name: &'static str,
    pub tex: usize,
    pub comp: usize,
    pub kind: MapKind,
    pub desc: &'static str,
}

const fn map(name: &'static str, tex: usize, comp: usize, kind: MapKind, desc: &'static str) -> MapInfo {
    MapInfo { name, tex, comp, kind, desc }
}

/// How many built-in baked maps there are (see [`MAPS`]).
pub const MAP_COUNT: usize = 13;

/// Every built-in baked map, packed into five textures
/// (`bake::MAP_TEXTURES`). View mode `6 + index` displays map `index`.
/// Custom bake maps come on top, any number (see `Engine::custom_index`).
pub const MAPS: [MapInfo; MAP_COUNT] = [
    map("Ambient Occlusion", 0, 0, MapKind::Scalar, "How much ambient light reaches each point."),
    map("Curvature", 0, 1, MapKind::Scalar, "Per-texel convex (bright) / concave (dark); includes high-poly detail."),
    map("Cavity", 0, 2, MapKind::Scalar, "Short-range occlusion: seams, panel gaps, engravings."),
    map("Thickness", 0, 3, MapKind::Scalar, "Distance through the mesh; dark = thin."),
    map("Sky Exposure", 1, 0, MapKind::Scalar, "Visible sky above each point: where dust, snow and rain settle."),
    map("Edge Distance", 1, 1, MapKind::Scalar, "World-space distance to the nearest convex edge."),
    map("Crease Distance", 1, 2, MapKind::Scalar, "World-space distance to the nearest concave crease."),
    map("Part Random", 1, 3, MapKind::Scalar, "One random value per connected mesh part."),
    map("Part Gradient", 2, 3, MapKind::Scalar, "Bottom-to-top gradient within each part."),
    map("Bent Normal", 2, 0, MapKind::Info, "Average unoccluded direction (world space)."),
    map("Normal", 3, 0, MapKind::Normal, "Tangent-space normal from the high poly."),
    map("Height", 3, 2, MapKind::Scalar, "High-poly displacement relative to the low poly."),
    map("ID", 4, 0, MapKind::Info, "Color per material or mesh part, for ID masks."),
];

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Levels {
    pub black: f32,
    pub white: f32,
    pub gamma: f32,
}

impl Default for Levels {
    fn default() -> Self {
        Self { black: 0.0, white: 1.0, gamma: 1.0 }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct FillParams {
    pub mat: MatValues,
    pub color_var: f32,
    pub var_scale: f32,
    pub rough_var: f32,
    pub height_var: f32,
    pub seed: f32,
    /// Per-part variation of brightness and roughness (uses the baked part random map).
    pub part_var: f32,
    /// How the layer's images map onto the mesh.
    pub projection: Projection,
    /// Image repeats across the UV square, or per world unit (tri-planar).
    pub tiling: f32,
    /// Image rotation in degrees.
    pub rotation: f32,
    pub offset: [f32; 2],
    /// Tri-planar blend sharpness between the three projections.
    pub tri_sharpness: f32,
    /// The material graph whose outputs fill the image slots (an app graph id).
    pub graph: Option<u64>,
}

/// The order is the shader's projection index (`composite.wgsl`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Projection {
    Uv,
    /// Projected along the three world axes and blended by the normal:
    /// seamless on any mesh, needs no UVs.
    Triplanar,
    /// Projected straight along one world axis, spanning the mesh's bounds:
    /// decals, labels, gradients that ignore the UV layout.
    PlanarX,
    PlanarY,
    PlanarZ,
    /// Wrapped around the mesh's center like a globe.
    Spherical,
    /// Wrapped around the vertical axis like a can label.
    Cylindrical,
}

impl Projection {
    pub const ALL: [Projection; 7] = [
        Projection::Uv,
        Projection::Triplanar,
        Projection::PlanarX,
        Projection::PlanarY,
        Projection::PlanarZ,
        Projection::Spherical,
        Projection::Cylindrical,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Projection::Uv => "UV",
            Projection::Triplanar => "Tri-planar",
            Projection::PlanarX => "Planar X",
            Projection::PlanarY => "Planar Y",
            Projection::PlanarZ => "Planar Z",
            Projection::Spherical => "Spherical",
            Projection::Cylindrical => "Cylindrical",
        }
    }
    /// Name in projects and MCP calls.
    pub fn key(self) -> &'static str {
        match self {
            Projection::Uv => "uv",
            Projection::Triplanar => "triplanar",
            Projection::PlanarX => "planar_x",
            Projection::PlanarY => "planar_y",
            Projection::PlanarZ => "planar_z",
            Projection::Spherical => "spherical",
            Projection::Cylindrical => "cylindrical",
        }
    }
    pub fn from_key(key: &str) -> Option<Projection> {
        Projection::ALL.into_iter().find(|p| p.key().eq_ignore_ascii_case(key) || format!("{p:?}") == key)
    }
    pub fn tip(self) -> &'static str {
        match self {
            Projection::Uv => "Follow the mesh's UV layout",
            Projection::Triplanar => "Project from the three world axes and blend: seamless on any mesh, no UVs needed",
            Projection::PlanarX | Projection::PlanarY | Projection::PlanarZ => {
                "Project straight along one axis across the whole mesh, like a decal or a gradient"
            }
            Projection::Spherical => "Wrap around the mesh's center like a globe",
            Projection::Cylindrical => "Wrap around the vertical axis like a can label",
        }
    }
}

impl Default for FillParams {
    fn default() -> Self {
        Self {
            mat: MatValues::default(),
            color_var: 0.0,
            var_scale: 4.0,
            rough_var: 0.0,
            height_var: 0.0,
            seed: 0.0,
            part_var: 0.0,
            projection: Projection::Uv,
            tiling: 1.0,
            rotation: 0.0,
            offset: [0.0; 2],
            tri_sharpness: 4.0,
            graph: None,
        }
    }
}

/// A bitmap feeding one channel of a fill layer. Shared, so duplicating a
/// layer or keeping it in the history costs no copy; `encoded` is the
/// original file, stored in projects.
#[derive(Clone)]
pub struct FillImage {
    pub tex: Arc<Tex>,
    pub name: String,
    pub encoded: Arc<Vec<u8>>,
}

/// Largest image edge kept on the GPU; bigger files are scaled down.
const MAX_IMAGE: u32 = 4096;

pub enum Content {
    Paint([Option<Tex>; CHANNELS]),
    Fill(FillParams),
    /// Groups the layers directly below it whose `parent` is this layer.
    Folder(Folder),
}

/// A folder's visibility hides everything inside. Its opacity and mask fade
/// the folder's combined result (the layers inside composited at full
/// strength) over what is below, like an isolated group; folders at full
/// opacity without a mask cost nothing extra. Folders have no blend mode.
#[derive(Clone, Copy, Debug)]
pub struct Folder {
    /// Expanded in the layer stack (UI only).
    pub open: bool,
    /// Its live graph (an app graph id): the fill layers inside as nodes.
    pub graph: Option<u64>,
}

/// One step of compositing.
#[derive(Clone, Copy, PartialEq)]
enum Pass {
    /// Saves the stack below a faded or masked folder.
    Begin,
    Layer(usize),
    /// Fades the folder's result over the saved stack.
    End(usize),
}

/// A composite's state just before the painted layer's pass: the pass list
/// it belongs to, that pass's index, the folder depth there and where the
/// input lives (`None`: in `prefix_cache`; else that accum set, untouched
/// because the painted layer's pass is the last one).
struct Prefix {
    layer_id: u64,
    passes: Vec<Pass>,
    at: usize,
    depth: usize,
    input: Option<usize>,
}

/// Where a moved layer goes, relative to another layer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Drop {
    Above(u64),
    Below(u64),
    /// At the top of a folder.
    Into(u64),
}

/// A step of a mask's filter stack, applied after painting, generator and graph.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FilterKind {
    Levels,
    Invert,
    Contrast,
    Blur,
    Sharpen,
    Warp,
    Noise,
}

impl FilterKind {
    pub const ALL: [FilterKind; 7] =
        [FilterKind::Levels, FilterKind::Invert, FilterKind::Contrast, FilterKind::Blur, FilterKind::Sharpen, FilterKind::Warp, FilterKind::Noise];
    pub fn name(self) -> &'static str {
        match self {
            FilterKind::Levels => "Levels",
            FilterKind::Invert => "Invert",
            FilterKind::Contrast => "Contrast",
            FilterKind::Blur => "Blur",
            FilterKind::Sharpen => "Sharpen",
            FilterKind::Warp => "Warp",
            FilterKind::Noise => "Noise breakup",
        }
    }
    pub fn tip(self) -> &'static str {
        match self {
            FilterKind::Levels => "Black point, white point and gamma",
            FilterKind::Invert => "Swaps where the layer shows",
            FilterKind::Contrast => "Pushes the mask toward black and white around a position",
            FilterKind::Blur => "Softens the mask's edges",
            FilterKind::Sharpen => "Crisps the mask's edges",
            FilterKind::Warp => "Pushes the mask around with noise: ragged, organic edges",
            FilterKind::Noise => "Eats into the mask with noise, like grunge",
        }
    }
    /// The parameters a filter shows: (name, min, max) for a, b, c.
    pub fn params(self) -> &'static [(&'static str, f32, f32)] {
        match self {
            FilterKind::Levels => &[("Black", 0.0, 1.0), ("White", 0.0, 1.0), ("Gamma", 0.1, 4.0)],
            FilterKind::Invert => &[],
            FilterKind::Contrast => &[("Position", 0.0, 1.0), ("Contrast", 0.0, 1.0)],
            FilterKind::Blur => &[("Radius", 0.0, 64.0)],
            FilterKind::Sharpen => &[("Amount", 0.0, 4.0), ("Radius", 0.5, 8.0)],
            FilterKind::Warp => &[("Intensity", 0.0, 1.0), ("Scale", 1.0, 64.0)],
            FilterKind::Noise => &[("Amount", 0.0, 1.0), ("Scale", 1.0, 64.0), ("Seed", 0.0, 100.0)],
        }
    }
    pub fn key(self) -> &'static str {
        match self {
            FilterKind::Levels => "levels",
            FilterKind::Invert => "invert",
            FilterKind::Contrast => "contrast",
            FilterKind::Blur => "blur",
            FilterKind::Sharpen => "sharpen",
            FilterKind::Warp => "warp",
            FilterKind::Noise => "noise",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MaskFilter {
    pub kind: FilterKind,
    pub enabled: bool,
    pub a: f32,
    pub b: f32,
    pub c: f32,
}

impl MaskFilter {
    pub fn new(kind: FilterKind) -> MaskFilter {
        let (a, b, c) = match kind {
            FilterKind::Levels => (0.0, 1.0, 1.0),
            FilterKind::Invert => (0.0, 0.0, 0.0),
            FilterKind::Contrast => (0.5, 0.8, 0.0),
            FilterKind::Blur => (6.0, 0.0, 0.0),
            FilterKind::Sharpen => (1.0, 1.5, 0.0),
            FilterKind::Warp => (0.15, 8.0, 0.0),
            FilterKind::Noise => (0.6, 12.0, 0.0),
        };
        MaskFilter { kind, enabled: true, a, b, c }
    }

    /// How far (texels at `res`) the filter reads from its neighbors.
    fn reach(&self, res: u32) -> f32 {
        let scale = res as f32 / FILTER_REF_RES;
        match self.kind {
            FilterKind::Blur => self.a * scale,
            FilterKind::Sharpen => self.b * scale,
            FilterKind::Warp => self.a * 0.2 * 0.5 * res as f32,
            _ => 0.0,
        }
    }
}

/// How a mask layer combines with the mask below it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MaskBlend {
    Normal,
    Multiply,
    Add,
    Subtract,
    Lighten,
    Darken,
}

impl MaskBlend {
    pub const ALL: [MaskBlend; 6] = [MaskBlend::Normal, MaskBlend::Multiply, MaskBlend::Add, MaskBlend::Subtract, MaskBlend::Lighten, MaskBlend::Darken];
    pub fn name(self) -> &'static str {
        match self {
            MaskBlend::Normal => "Normal",
            MaskBlend::Multiply => "Multiply",
            MaskBlend::Add => "Add",
            MaskBlend::Subtract => "Subtract",
            MaskBlend::Lighten => "Lighten",
            MaskBlend::Darken => "Darken",
        }
    }
    pub fn key(self) -> &'static str {
        match self {
            MaskBlend::Normal => "normal",
            MaskBlend::Multiply => "multiply",
            MaskBlend::Add => "add",
            MaskBlend::Subtract => "subtract",
            MaskBlend::Lighten => "lighten",
            MaskBlend::Darken => "darken",
        }
    }
}

/// What a mask layer contributes.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum MaskSource {
    /// A smart mask from the mesh maps.
    Generator(Generator),
    /// One value everywhere.
    Fill(f32),
}

/// One layer of a mask's stack, above the base (painting × smart mask).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MaskLayer {
    pub source: MaskSource,
    pub blend: MaskBlend,
    pub opacity: f32,
    pub enabled: bool,
}

/// One step of rendering a mask's stack (see `Engine::encode_mask_filters`).
#[derive(Clone, Copy)]
enum MaskOp {
    /// A mask layer: render it with this layer-uniform offset, then blend it
    /// in with this filter step.
    Layer { uniform: u32, combine: usize },
    /// A filter step.
    Filter(usize),
}

/// The filter shader's step kind that blends a mask layer in.
const COMBINE_STEP: f32 = 7.0;

/// Filter radii are in texels of a texture this size, so a filter looks
/// the same at any texture set resolution.
const FILTER_REF_RES: f32 = 2048.0;

pub struct Mask {
    /// Made as a black mask and not changed since: a mask layer added to it should
    /// reveal (Add), because Multiply with a black base shows nothing.
    pub black_base: bool,
    pub paint: Option<Tex>,
    pub generator: Option<Generator>,
    /// Filters on the mask, in order.
    pub filters: Vec<MaskFilter>,
    /// Mask layers combined over the base, bottom first (before the filters).
    pub layers: Vec<MaskLayer>,
    /// A material graph renders the paint texture (an app graph id).
    pub graph: Option<u64>,
}

pub struct Layer {
    pub id: u64,
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    pub blend: Blend,
    pub content: Content,
    pub mask: Option<Mask>,
    /// The folder this layer is in. A folder's layers sit directly below it.
    pub parent: Option<u64>,
    /// Fill layers: images replacing color, roughness, metallic, height, opacity.
    pub images: [Option<FillImage>; CHANNELS],
    /// egui handle of the color channel, used as the layer thumbnail.
    pub thumb: Option<egui::TextureId>,
}

impl Layer {
    pub fn is_fill(&self) -> bool {
        matches!(self.content, Content::Fill(_))
    }
    pub fn is_folder(&self) -> bool {
        matches!(self.content, Content::Folder(_))
    }
    pub fn is_paint(&self) -> bool {
        matches!(self.content, Content::Paint(_))
    }
    fn vram_bytes(&self) -> u64 {
        let content = match &self.content {
            Content::Paint(ch) => ch.iter().flatten().map(|t| t.bytes()).sum(),
            Content::Fill(_) | Content::Folder(_) => 0,
        };
        content + self.mask.as_ref().and_then(|m| m.paint.as_ref()).map_or(0, |t| t.bytes())
    }
}

/// The ID mask's colors as shader uniforms: the first color with the
/// tolerance, then the others.
fn id_uniform(g: &Generator) -> ([f32; 4], [[f32; 4]; MAX_ID_COLORS - 1]) {
    let colors = g.id_colors();
    let mut more = [[0.0; 4]; MAX_ID_COLORS - 1];
    for (slot, c) in more.iter_mut().zip(&colors[1..]) {
        *slot = [c[0], c[1], c[2], 1.0];
    }
    ([g.id_color[0], g.id_color[1], g.id_color[2], g.id_tolerance], more)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LayerUniform {
    fill_color: [f32; 4],
    fill_rmh: [f32; 4],
    ch: [f32; 4],
    p0: [f32; 4],
    var0: [f32; 4],
    var1: [f32; 4],
    bmin: [f32; 4],
    bsize: [f32; 4],
    gw: [[f32; 4]; 5],
    /// Custom map inputs: their index among the baked custom maps (-1 none), and weights.
    gc: [f32; 4],
    gcw: [f32; 4],
    gp0: [f32; 4],
    gp1: [f32; 4],
    gid: [f32; 4],
    gx: [f32; 4],
    img: [f32; 4],
    /// Fill emission (sRGB rgb) and the emission channel's flag.
    emis: [f32; 4],
    /// Normal image strength, normal channel flag, fill specular level,
    /// specular channel flag.
    ns: [f32; 4],
    /// The ID mask's further colors (rgb) with a 1 in w when in use.
    gids: [[f32; 4]; MAX_ID_COLORS - 1],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ResolveUniform {
    levels: [[f32; 4]; 4],
    kind: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct StampUniform {
    pub view_proj: [[f32; 4]; 4],
    pub cam_pos: [f32; 4],
    pub view: [f32; 4],
    pub view2d: [f32; 4],
    pub brush: [f32; 4],
    pub clip: [f32; 4],
    /// Projection (0 screen, 1 surface, 2 uv), paint through, cos of the angle limit, unused.
    pub proj: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ApplyUniform {
    color: [f32; 4],
    rmh: [f32; 4],
    ch: [f32; 4],
    p: [f32; 4],
    g: [f32; 4],
    /// Emission (sRGB rgb) and whether the stroke paints it.
    e: [f32; 4],
    /// Specular level, whether the stroke paints it, whether it paints the
    /// normal channel (flat), unused.
    s: [f32; 4],
}

/// What the brush deposits during a stroke.
pub struct BrushPaint {
    pub mat: MatValues,
    pub mask_value: f32,
    pub opacity: f32,
    pub erase: bool,
    /// Grain strength (0 off), tiling, in uv (else tri-planar world), invert; see `apply.wgsl`.
    pub grain: [f32; 4],
}

struct ActiveStroke {
    layer_id: u64,
    target: StrokeTarget,
    channels: Vec<usize>,
    before: Snapshot,
}

struct Pipelines {
    uvbake: wgpu::RenderPipeline,
    dilate: wgpu::RenderPipeline,
    dilate_bgl: wgpu::BindGroupLayout,
    stamp: wgpu::RenderPipeline,
    stamp_bgl: wgpu::BindGroupLayout,
    poly: wgpu::RenderPipeline,
    /// Polygon fill padding: only where no UV island is (reads the position map).
    poly_pad: wgpu::RenderPipeline,
    poly_pad_bgl: wgpu::BindGroupLayout,
    apply_module: wgpu::ShaderModule,
    apply_layout: wgpu::PipelineLayout,
    apply_bgl: wgpu::BindGroupLayout,
    /// Made on first use per channel combination; shared by every texture set.
    apply_paint: std::sync::Mutex<HashMap<u8, wgpu::RenderPipeline>>,
    apply_mask: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    folder_merge: wgpu::RenderPipeline,
    composite_bgl: wgpu::BindGroupLayout,
    mask_source: wgpu::RenderPipeline,
    mask_filter: wgpu::RenderPipeline,
    mask_filter_bgl: wgpu::BindGroupLayout,
    resample: HashMap<F, wgpu::RenderPipeline>,
    resample_bgl: wgpu::BindGroupLayout,
    resolve: HashMap<F, wgpu::RenderPipeline>,
    resolve_bgl: wgpu::BindGroupLayout,
}

#[derive(Clone)]
struct GpuStrokeProfileResources {
    query_set: Arc<wgpu::QuerySet>,
    resolve: Arc<wgpu::Buffer>,
    readback: Arc<wgpu::Buffer>,
}

struct GpuStrokeProfiler {
    resources: GpuStrokeProfileResources,
    tx: mpsc::Sender<bool>,
    rx: mpsc::Receiver<bool>,
    pending: bool,
    encoder_timestamps: bool,
}

impl GpuStrokeProfiler {
    fn new(device: &wgpu::Device) -> Self {
        let query_set = Arc::new(device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("stroke GPU timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 8,
        }));
        let resolve = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stroke timestamp resolve"),
            size: 64,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let readback = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stroke timestamp readback"),
            size: 64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
        let (tx, rx) = mpsc::channel();
        Self {
            resources: GpuStrokeProfileResources { query_set, resolve, readback },
            tx,
            rx,
            pending: false,
            encoder_timestamps: device.features().contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS),
        }
    }

    fn resources(&self) -> Option<GpuStrokeProfileResources> {
        (!self.pending).then(|| self.resources.clone())
    }

    fn request_map(&mut self) {
        self.pending = true;
        let tx = self.tx.clone();
        self.resources.readback.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result.is_ok());
        });
    }

    fn poll(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> Option<(f64, f64, f64, Option<f64>)> {
        if !self.pending {
            return None;
        }
        let _ = device.poll(wgpu::PollType::Poll);
        let mapped = match self.rx.try_recv() {
            Ok(mapped) => mapped,
            Err(mpsc::TryRecvError::Empty) => return None,
            Err(mpsc::TryRecvError::Disconnected) => {
                self.pending = false;
                return None;
            }
        };
        self.pending = false;
        if !mapped {
            return None;
        }
        let data = match self.resources.readback.slice(..).get_mapped_range() {
            Ok(data) => data,
            Err(_) => {
                self.resources.readback.unmap();
                return None;
            }
        };
        let read = |i: usize| u64::from_ne_bytes(data[i * 8..i * 8 + 8].try_into().expect("timestamp is 8 bytes"));
        let (s0, s1, a0, a1, c0, c1, t0, t1) =
            (read(0), read(1), read(2), read(3), read(4), read(5), read(6), read(7));
        drop(data);
        self.resources.readback.unmap();
        let period_ns = queue.get_timestamp_period();
        if period_ns <= 0.0 {
            return None;
        }
        let period_ms = f64::from(period_ns) / 1_000_000.0;
        Some((
            s1.wrapping_sub(s0) as f64 * period_ms,
            a1.wrapping_sub(a0) as f64 * period_ms,
            c1.wrapping_sub(c0) as f64 * period_ms,
            self.encoder_timestamps.then(|| t1.wrapping_sub(t0) as f64 * period_ms),
        ))
    }
}

pub struct Engine {
    /// Tiling, mipmapped sampler for fill images.
    repeat_sampler: wgpu::Sampler,
    /// The stack below each open faded/masked folder while its layers are
    /// composited, one set per nesting level; only allocated while needed.
    folder_below: Vec<[Tex; ACCUM]>,
    rs: egui_wgpu::RenderState,
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Optional non-blocking GPU timings when the adapter supports timestamps.
    gpu_profiler: Option<GpuStrokeProfiler>,
    /// Holds the current sample until its layer composite pass is recorded.
    active_gpu_profile: Option<GpuStrokeProfileResources>,
    /// Shared by the texture sets of a document.
    p: Arc<Pipelines>,
    pub res: u32,
    /// The texture set's range in the mesh's index buffer: only these
    /// triangles get position and normal maps, so only they take paint.
    tris: std::ops::Range<u32>,
    pos: Tex,
    nrm: Tex,
    /// Tangent frame per texel: fill normal-map images projected in world
    /// space (tri-planar, planar) are turned into tangent space with it.
    tan: Tex,
    accum: [[Tex; ACCUM]; 2],
    final_idx: usize,
    stroke: Tex,
    snap: [Option<Tex>; CHANNELS],
    snap_mask: Option<Tex>,
    /// Baked mesh maps as produced by the baker (see [`MAPS`]).
    maps: [Tex; MAP_TEXTURES],
    /// The baked custom map textures (four maps each), at the texture set's
    /// size, and the same copied into one array for the shaders.
    custom: Vec<Tex>,
    custom_array: (wgpu::Texture, wgpu::TextureView),
    /// The custom maps' ids, in baked order (map `i`: channel `i % 4` of layer `i / 4`).
    custom_ids: Vec<u32>,
    /// Maps with levels and painted fixes applied; `None` when unchanged.
    resolved: [Option<Tex>; MAP_TEXTURES],
    pub map_fix: [Option<Tex>; MAP_COUNT],
    pub map_levels: [Levels; MAP_COUNT],
    maps_dirty: bool,
    pub has_bake: bool,
    /// Bumped whenever the mesh maps (or their levels and fixes) change.
    pub maps_version: u64,
    /// Bumped whenever the position and normal maps are made again.
    uv_version: u64,
    /// Layer thumbnails (see `thumbs`).
    thumbs: thumbs::Thumbs,
    /// Composites that redid the whole texture instead of a stroke's patch (the F6 meter counts them).
    pub full_recomps: u32,
    /// Which of a filtered mask's textures (`mask_out`) holds its result.
    mask_final: HashMap<u64, usize>,
    pub bake_report: Option<Arc<BakeReport>>,
    edge_max: f32,
    dummy: Tex,
    sampler: wgpu::Sampler,
    layer_ubuf: wgpu::Buffer,
    layer_ubuf_slots: u64,
    stamp_ubuf: wgpu::Buffer,
    stamp_sbuf: wgpu::Buffer,
    apply_ubuf: wgpu::Buffer,
    /// Brush tip (alpha) the stamps sample when the brush uses one.
    brush_alpha: Tex,
    /// Brush grain texture (white when the brush has none).
    brush_grain: Tex,
    /// Image files already on the GPU, shared by every layer that uses them
    /// (scanned smart materials); entries die with their last user.
    image_cache: HashMap<std::path::PathBuf, (std::sync::Weak<Tex>, Arc<Vec<u8>>)>,
    pub layers: Vec<Layer>,
    /// Properties as of the last recorded edit; see [`Engine::commit_props`].
    committed: DocProps,
    next_id: u64,
    bounds: (Vec3, Vec3),
    active: Option<ActiveStroke>,
    /// Set whenever the layer stack changes; cleared by [`Engine::composite`].
    pub dirty: bool,
    /// Something besides the painted layer changed since the last composite.
    full_dirty: bool,
    /// The layer the current stroke paints (content or mask).
    stroke_layer: Option<u64>,
    /// The stack below the painted layer, kept so stroke frames only
    /// recomposite from that layer up.
    prefix: Option<Prefix>,
    /// Holds the stack below the painted layer when more passes follow it.
    prefix_cache: Option<[Tex; ACCUM]>,
    /// Filtered masks: per layer, the two textures its filter steps
    /// alternate between (the result is in one of them).
    mask_out: HashMap<u64, [Tex; 3]>,
    mask_filter_ubuf: wgpu::Buffer,
    /// Texels the stroke changed since the last composite (x0, y0, x1, y1,
    /// exclusive), so stroke frames recomposite only that patch; `None`
    /// with `region_unknown` when a step didn't say.
    stroke_region: Option<[u32; 4]>,
    region_unknown: bool,
    /// Incremented on every composite, so dependent views know the material changed.
    pub generation: u64,
}

fn accum_set(device: &wgpu::Device, res: u32, label: &str) -> [Tex; ACCUM] {
    ACCUM_FORMATS.map(|f| Tex::new(device, label, res, res, f, LAYER_USAGE))
}

impl Engine {
    /// A document of one texture set covering the whole mesh.
    pub fn new(rs: &egui_wgpu::RenderState, mesh: &GpuMesh, bounds: (Vec3, Vec3), res: u32) -> Engine {
        let p = Arc::new(Self::create_pipelines(&rs.device));
        Self::with_pipelines(rs, p, mesh, 0..mesh.index_count, bounds, res)
    }

    /// Another texture set of the same document: `tris` is its range in the
    /// mesh's index buffer. Shares this engine's pipelines.
    pub fn sibling(&self, mesh: &GpuMesh, tris: std::ops::Range<u32>, res: u32) -> Engine {
        Self::with_pipelines(&self.rs, self.p.clone(), mesh, tris, self.bounds, res)
    }

    fn with_pipelines(
        rs: &egui_wgpu::RenderState,
        p: Arc<Pipelines>,
        mesh: &GpuMesh,
        tris: std::ops::Range<u32>,
        bounds: (Vec3, Vec3),
        res: u32,
    ) -> Engine {
        let device = rs.device.clone();
        let queue = rs.queue.clone();
        let sampler = gpu::linear_sampler(&device, wgpu::AddressMode::ClampToEdge);
        let gpu_profiler = device.features().contains(wgpu::Features::TIMESTAMP_QUERY).then(|| GpuStrokeProfiler::new(&device));

        let dummy = Tex::new(&device, "dummy", 1, 1, F::Rgba8Unorm, LAYER_USAGE);
        queue.write_texture(dummy.copy_info(), &[0, 0, 0, 0], Default::default(), dummy.extent());
        let maps = Self::default_maps(&device, &queue);

        let buffer = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let layer_ubuf_slots = 16;
        let layer_ubuf = buffer("layer uniforms", UNIFORM_STRIDE * layer_ubuf_slots, wgpu::BufferUsages::UNIFORM);
        let stamp_ubuf = buffer("stamp uniform", size_of::<StampUniform>() as u64, wgpu::BufferUsages::UNIFORM);
        let stamp_sbuf = buffer("stamps", (MAX_STAMPS * size_of::<Stamp>()) as u64, wgpu::BufferUsages::STORAGE);
        let apply_ubuf = buffer("apply uniform", size_of::<ApplyUniform>() as u64, wgpu::BufferUsages::UNIFORM);

        let mut engine = Engine {
            rs: rs.clone(),
            gpu_profiler,
            active_gpu_profile: None,
            pos: Tex::new(&device, "position map", res, res, POS_FORMAT, LAYER_USAGE),
            nrm: Tex::new(&device, "normal map", res, res, NRM_FORMAT, LAYER_USAGE),
            tan: Tex::new(&device, "tangent map", res, res, TAN_FORMAT, LAYER_USAGE),
            accum: [accum_set(&device, res, "accum A"), accum_set(&device, res, "accum B")],
            final_idx: 0,
            stroke: Tex::new(&device, "stroke mask", res, res, STROKE_FORMAT, LAYER_USAGE),
            snap: Default::default(),
            snap_mask: None,
            maps,
            custom: Vec::new(),
            custom_array: Self::custom_array_texture(&device, 1, 1),
            custom_ids: Vec::new(),
            resolved: Default::default(),
            map_fix: Default::default(),
            map_levels: [Levels::default(); MAP_COUNT],
            maps_dirty: true,
            has_bake: false,
            maps_version: 0,
            uv_version: 0,
            thumbs: Default::default(),
            full_recomps: 0,
            mask_final: HashMap::new(),
            bake_report: None,
            edge_max: 0.2,
            dummy,
            sampler,
            layer_ubuf,
            layer_ubuf_slots,
            stamp_ubuf,
            stamp_sbuf,
            apply_ubuf,
            brush_alpha: Tex::new(&device, "brush alpha", 1, 1, F::R8Unorm, LAYER_USAGE),
            brush_grain: Tex::new(&device, "brush grain", 1, 1, F::R8Unorm, LAYER_USAGE),
            image_cache: HashMap::new(),
            layers: Vec::new(),
            folder_below: Vec::new(),
            repeat_sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("fill images"),
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                anisotropy_clamp: 8,
                ..Default::default()
            }),
            committed: DocProps::default(),
            next_id: 1,
            bounds,
            active: None,
            dirty: true,
            full_dirty: true,
            stroke_layer: None,
            prefix: None,
            prefix_cache: None,
            mask_out: HashMap::new(),
            mask_filter_ubuf: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mask filter uniforms"),
                size: MASK_FILTER_STRIDE * MAX_MASK_FILTER_STEPS,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            stroke_region: None,
            region_unknown: false,
            generation: 0,
            res,
            tris,
            p,
            device,
            queue,
        };
        engine.bake_uv(mesh);
        engine.committed = engine.doc_props();
        engine
    }

    fn create_pipelines(device: &wgpu::Device) -> Pipelines {
        // UV-space position/normal bake.
        let uv_module = gpu::shader(device, "uvbake", include_str!("shaders/uvbake.wgsl"));
        let empty_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        let uvbake = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("uv bake"),
            layout: Some(&empty_layout),
            vertex: wgpu::VertexState {
                module: &uv_module,
                entry_point: Some("vs_uv"),
                compilation_options: Default::default(),
                buffers: &[Some(gpu::vertex_layout())],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &uv_module,
                entry_point: Some("fs_uv"),
                compilation_options: Default::default(),
                targets: &[gpu::target(POS_FORMAT), gpu::target(NRM_FORMAT), gpu::target(TAN_FORMAT)],
            }),
            multiview_mask: None,
            cache: None,
        });
        let dilate_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dilate"),
            entries: &[gpu::tex_entry(0, false), gpu::tex_entry(1, false), gpu::tex_entry(2, false)],
        });
        let dilate = gpu::fullscreen_pipeline(
            device,
            "dilate",
            &uv_module,
            "fs_dilate",
            &gpu::layout(device, &dilate_bgl),
            &[gpu::target(POS_FORMAT), gpu::target(NRM_FORMAT), gpu::target(TAN_FORMAT)],
        );

        // Brush stamping into the stroke mask.
        let stamp_module = gpu::shader(device, "stamp", include_str!("shaders/stamp.wgsl"));
        let stamp_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stamp"),
            entries: &[
                gpu::uniform_entry(0, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                gpu::tex_entry(2, false),
                gpu::tex_entry(3, false),
                gpu::tex_entry(5, true),
                gpu::sampler_entry(6),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: true,
                    },
                    count: None,
                },
            ],
        });
        let flow_blend = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::OneMinusDst,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let stamp = gpu::fullscreen_pipeline(
            device,
            "stamp",
            &stamp_module,
            "fs_stamp",
            &gpu::layout(device, &stamp_bgl),
            &[Some(wgpu::ColorTargetState {
                format: STROKE_FORMAT,
                blend: Some(wgpu::BlendState { color: flow_blend, alpha: flow_blend }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        );

        // Polygon fill: selected triangles rasterized at their UVs into the stroke mask.
        let poly_module = gpu::shader(device, "poly fill", include_str!("shaders/poly.wgsl"));
        let poly_pad_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("poly padding"),
            entries: &[gpu::tex_entry(0, false)],
        });
        let poly_pipeline = |label: &str, fs: &str, bgls: &[Option<&wgpu::BindGroupLayout>]| device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: bgls,
                immediate_size: 0,
            })),
            vertex: wgpu::VertexState {
                module: &poly_module,
                entry_point: Some("vs_poly"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                })],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &poly_module,
                entry_point: Some(fs),
                compilation_options: Default::default(),
                targets: &[gpu::target(STROKE_FORMAT)],
            }),
            multiview_mask: None,
            cache: None,
        });
        let poly = poly_pipeline("poly fill", "fs_poly", &[]);
        let poly_pad = poly_pipeline("poly padding", "fs_poly_pad", &[Some(&poly_pad_bgl)]);

        // Applying the stroke to layer channels / masks.
        let apply_module = gpu::shader(device, "apply", include_str!("shaders/apply.wgsl"));
        let apply_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apply"),
            entries: &[
                gpu::uniform_entry(0, false),
                gpu::tex_entry(1, false),
                gpu::tex_entry(2, false),
                gpu::tex_entry(3, false),
                gpu::tex_entry(4, false),
                gpu::tex_entry(5, false),
                gpu::tex_entry(6, false),
                gpu::tex_entry(7, false),
                gpu::tex_entry(8, true),
                gpu::sampler_entry(9),
                gpu::tex_entry(10, false),
                gpu::tex_entry(11, false),
                gpu::tex_entry(12, false),
                gpu::tex_entry(13, false),
            ],
        });
        let apply_layout = gpu::layout(device, &apply_bgl);
        let apply_mask = gpu::fullscreen_pipeline(
            device,
            "apply mask",
            &apply_module,
            "fs_mask",
            &apply_layout,
            &[gpu::target(MASK_FORMAT)],
        );

        // Layer compositing.
        let composite_module =
            gpu::shader(device, "composite", concat!(include_str!("shaders/wear.wgsl"), include_str!("shaders/composite.wgsl")));
        let mut entries = vec![gpu::uniform_entry(0, true)];
        for b in 1..=11 {
            // Layer slots 5-8 also hold fill images, which are filtered.
            entries.push(gpu::tex_entry(b, (5..=8).contains(&b)));
        }
        for b in 12..=16 {
            // 15: the custom maps, an array.
            entries.push(if b == 15 {
                wgpu::BindGroupLayoutEntry {
                    binding: b,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                }
            } else {
                gpu::tex_entry(b, true)
            });
        }
        entries.push(gpu::sampler_entry(17));
        entries.push(gpu::sampler_entry(18));
        // The layer's opacity channel (paint) or image (fill).
        entries.push(gpu::tex_entry(19, true));
        // The stack's emission so far, and the layer's emission (paint) or image (fill).
        entries.push(gpu::tex_entry(20, false));
        entries.push(gpu::tex_entry(21, true));
        // Normal and specular: the stack's so far, and the layer's (paint or fill image).
        entries.push(gpu::tex_entry(22, false));
        entries.push(gpu::tex_entry(23, true));
        entries.push(gpu::tex_entry(24, false));
        entries.push(gpu::tex_entry(25, true));
        // The tangent frame, for normal images projected in world space.
        entries.push(gpu::tex_entry(26, false));
        let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite"),
            entries: &entries,
        });
        let composite = gpu::fullscreen_pipeline(
            device,
            "composite",
            &composite_module,
            "fs_layer",
            &gpu::layout(device, &composite_bgl),
            &ACCUM_FORMATS.map(gpu::target),
        );
        let folder_merge = gpu::fullscreen_pipeline(
            device,
            "folder merge",
            &composite_module,
            "fs_folder_merge",
            &gpu::layout(device, &composite_bgl),
            &ACCUM_FORMATS.map(gpu::target),
        );
        let mask_source = gpu::fullscreen_pipeline(
            device,
            "mask source",
            &composite_module,
            "fs_mask_source",
            &gpu::layout(device, &composite_bgl),
            &[gpu::target(MASK_FILTER_FORMAT)],
        );
        let mask_filter_module = gpu::shader(device, "mask filter", include_str!("shaders/mask_filter.wgsl"));
        let mask_filter_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mask filter"),
            entries: &[gpu::uniform_entry(0, true), gpu::tex_entry(1, true), gpu::sampler_entry(2), gpu::tex_entry(3, true)],
        });
        let mask_filter = gpu::fullscreen_pipeline(
            device,
            "mask filter",
            &mask_filter_module,
            "fs_filter",
            &gpu::layout(device, &mask_filter_bgl),
            &[gpu::target(MASK_FILTER_FORMAT)],
        );

        // Resolution changes.
        let resample_module = gpu::shader(device, "resample", include_str!("shaders/resample.wgsl"));
        let resample_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("resample"),
            entries: &[gpu::tex_entry(0, true), gpu::sampler_entry(1)],
        });
        let resample_layout = gpu::layout(device, &resample_bgl);
        let resample = [F::Rgba8Unorm, F::Rg8Unorm, F::Rg16Float, F::R8Unorm, F::R16Float, F::Rgba16Float]
            .into_iter()
            .map(|f| {
                let pl = gpu::fullscreen_pipeline(
                    device,
                    "resample",
                    &resample_module,
                    "fs_resample",
                    &resample_layout,
                    &[gpu::target(f)],
                );
                (f, pl)
            })
            .collect();

        // Mesh-map levels and painted fixes.
        let resolve_module = gpu::shader(device, "maps resolve", include_str!("shaders/maps_resolve.wgsl"));
        let resolve_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("maps resolve"),
            entries: &[
                gpu::uniform_entry(0, false),
                gpu::tex_entry(1, true),
                gpu::tex_entry(2, false),
                gpu::tex_entry(3, false),
                gpu::tex_entry(4, false),
                gpu::tex_entry(5, false),
                gpu::sampler_entry(6),
            ],
        });
        let resolve_layout = gpu::layout(device, &resolve_bgl);
        let resolve = [F::Rgba8Unorm, F::Rgba16Float]
            .into_iter()
            .map(|f| {
                let pl = gpu::fullscreen_pipeline(
                    device,
                    "maps resolve",
                    &resolve_module,
                    "fs_resolve",
                    &resolve_layout,
                    &[gpu::target(f)],
                );
                (f, pl)
            })
            .collect();

        Pipelines {
            uvbake,
            dilate,
            dilate_bgl,
            stamp,
            stamp_bgl,
            poly,
            poly_pad,
            poly_pad_bgl,
            apply_module,
            apply_layout,
            apply_bgl,
            apply_paint: Default::default(),
            apply_mask,
            composite,
            folder_merge,
            composite_bgl,
            mask_source,
            mask_filter,
            mask_filter_bgl,
            resample,
            resample_bgl,
            resolve,
            resolve_bgl,
        }
    }

    // ------------------------------------------------------------------ mesh

    /// Rasterizes position and normal maps for the current mesh at the
    /// current resolution, with dilation past UV island borders.
    pub fn bake_uv(&mut self, mesh: &GpuMesh) {
        // A mesh change can leave the range past the end; the app sets the
        // new range right after, and this draws nothing meanwhile.
        let tris = self.tris.start.min(mesh.index_count)..self.tris.end.min(mesh.index_count);
        let device = &self.device;
        let tmp_pos = Tex::new(device, "dilate pos", self.res, self.res, POS_FORMAT, LAYER_USAGE);
        let tmp_nrm = Tex::new(device, "dilate nrm", self.res, self.res, NRM_FORMAT, LAYER_USAGE);
        let tmp_tan = Tex::new(device, "dilate tan", self.res, self.res, TAN_FORMAT, LAYER_USAGE);
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let clear = wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT);
            let mut pass = gpu::pass(
                &mut encoder,
                "uv bake",
                &[
                    gpu::color_attachment(&self.pos.view, clear),
                    gpu::color_attachment(&self.nrm.view, clear),
                    gpu::color_attachment(&self.tan.view, clear),
                ],
            );
            pass.set_pipeline(&self.p.uvbake);
            pass.set_vertex_buffer(0, mesh.vertices.slice(..));
            pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(tris, 0, 0..1);
        }
        let bg = |src: [&Tex; 3]| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.p.dilate_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src[0].view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&src[1].view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&src[2].view) },
                ],
            })
        };
        let forward = bg([&self.pos, &self.nrm, &self.tan]);
        let backward = bg([&tmp_pos, &tmp_nrm, &tmp_tan]);
        for i in 0..DILATE_PASSES {
            let (group, dst) =
                if i % 2 == 0 { (&forward, [&tmp_pos, &tmp_nrm, &tmp_tan]) } else { (&backward, [&self.pos, &self.nrm, &self.tan]) };
            let load = wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT);
            let mut pass = gpu::pass(
                &mut encoder,
                "dilate",
                &[
                    gpu::color_attachment(&dst[0].view, load),
                    gpu::color_attachment(&dst[1].view, load),
                    gpu::color_attachment(&dst[2].view, load),
                ],
            );
            pass.set_pipeline(&self.p.dilate);
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..3, 0..1);
        }
        // DILATE_PASSES is even, so the result ends up back in pos/nrm.
        self.queue.submit([encoder.finish()]);
        self.uv_version += 1;
        self.mark_dirty();
    }

    pub fn bounds(&self) -> (Vec3, Vec3) {
        self.bounds
    }

    /// The texture set's range in the mesh's index buffer.
    pub fn triangles(&self) -> std::ops::Range<u32> {
        self.tris.clone()
    }

    /// Moves the texture set to another range of the (new) mesh and remakes
    /// its position and normal maps.
    pub fn set_triangles(&mut self, tris: std::ops::Range<u32>, mesh: &GpuMesh) {
        self.tris = tris;
        self.bake_uv(mesh);
    }

    pub fn set_bounds(&mut self, bounds: (Vec3, Vec3)) {
        self.bounds = bounds;
        self.mark_dirty();
    }

    // ------------------------------------------------------------------ mesh maps

    /// 1x1 neutral maps used until the first bake.
    fn default_maps(device: &wgpu::Device, queue: &wgpu::Queue) -> [Tex; MAP_TEXTURES] {
        let maps = std::array::from_fn(|i| Tex::new(device, "mesh map", 1, 1, MAP_FORMATS[i], LAYER_USAGE));
        // f16 bit patterns: 0.0 = 0x0000, 0.5 = 0x3800, 1.0 = 0x3C00.
        let map_b: [u16; 4] = [0x3C00, 0x3C00, 0x3C00, 0x3800];
        let half: [u16; 4] = [0, 0, 0x3800, 0];
        let data: [&[u8]; MAP_TEXTURES] = [
            &[255, 128, 255, 255],
            bytemuck::cast_slice(&map_b),
            &[128, 255, 128, 128],
            bytemuck::cast_slice(&half),
            &[128, 128, 128, 255],
        ];
        for (t, d) in maps.iter().zip(data) {
            queue.write_texture(t.copy_info(), d, Default::default(), t.extent());
        }
        maps
    }

    /// Maps the current bake left out (a bit per `MAPS` index); none before
    /// the first bake, when nothing is baked yet.
    pub fn missing_maps(&self) -> u32 {
        match (&self.bake_report, self.has_bake) {
            (Some(r), true) => crate::bake::ALL_MAPS & !r.baked,
            _ => 0,
        }
    }

    /// The baked maps on screen and their report (`None` before the first bake).
    pub fn current_maps(&self) -> Option<(BakedMaps, Arc<BakeReport>)> {
        let report = self.bake_report.clone().filter(|_| self.has_bake)?;
        Some((BakedMaps { maps: self.maps.clone(), custom: self.custom.clone() }, report))
    }

    /// A texture array of `layers` custom map textures (zeros: nothing baked).
    fn custom_array_texture(device: &wgpu::Device, res: u32, layers: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("custom maps"),
            size: wgpu::Extent3d { width: res, height: res, depth_or_array_layers: layers.max(1) },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: CUSTOM_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        (texture, view)
    }

    /// Copies the custom map textures into the array the shaders read.
    fn build_custom_array(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let res = self.custom.first().map_or(1, |t| t.width);
        self.custom_array = Self::custom_array_texture(&self.device, res, self.custom.len() as u32);
        for (layer, tex) in self.custom.iter().enumerate() {
            let dst = wgpu::TexelCopyTextureInfo {
                texture: &self.custom_array.0,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: layer as u32 },
                aspect: wgpu::TextureAspect::All,
            };
            encoder.copy_texture_to_texture(tex.copy_info(), dst, tex.extent());
        }
    }

    /// Where custom map `id` is among the baked ones (`None`: not baked).
    pub fn custom_index(&self, id: u32) -> Option<usize> {
        self.custom_ids.iter().position(|&c| c == id).filter(|_| self.has_bake)
    }

    /// The custom maps a generator reads that the bake didn't make (none
    /// before the first bake).
    pub fn missing_custom(&self, g: &Generator) -> Vec<u32> {
        if !self.has_bake {
            return Vec::new();
        }
        g.custom_ids().filter(|&id| self.custom_index(id).is_none()).collect()
    }

    /// The texture and channel a Mesh Map value reads (built-in maps by
    /// [`MAPS`] index, custom maps from [`CUSTOM_MAP_VALUE`]).
    pub fn mesh_map(&self, value: usize) -> Option<(&Tex, usize)> {
        if value < CUSTOM_MAP_VALUE {
            let m = MAPS.get(value)?;
            return Some((self.map_tex(m.tex), m.comp));
        }
        let i = self.custom_index((value - CUSTOM_MAP_VALUE) as u32)?;
        Some((&self.custom[i / CUSTOM_PER_TEXTURE], i % CUSTOM_PER_TEXTURE))
    }

    /// The custom maps' array for the shaders.
    pub fn custom_maps_view(&self) -> &wgpu::TextureView {
        &self.custom_array.1
    }

    /// A generator's custom inputs as the composite shader reads them: each
    /// map's index among the baked ones (-1: unused or not baked) and weight.
    fn custom_refs(&self, g: &Generator) -> ([f32; 4], [f32; 4]) {
        let mut idx = [-1.0; 4];
        let mut w = [0.0; 4];
        for (k, c) in g.custom.iter().enumerate() {
            if let Some(i) = (c.id != 0).then(|| self.custom_index(c.id)).flatten() {
                idx[k] = i as f32;
                w[k] = c.weight;
            }
        }
        (idx, w)
    }

    /// Back to the neutral maps of a document that was never baked.
    pub fn clear_maps(&mut self) {
        self.custom.clear();
        self.custom_ids.clear();
        self.custom_array = Self::custom_array_texture(&self.device, 1, 1);
        self.maps = Self::default_maps(&self.device, &self.queue);
        self.bake_report = None;
        self.has_bake = false;
        self.maps_dirty = true;
        self.mark_dirty();
    }

    /// Adopts freshly baked maps. Painted fixes and levels are kept, so artists
    /// can re-bake with new settings without losing their corrections.
    /// Swaps in only the ID map of a fresh bake (made from vertex colors); every other
    /// baked map stays. Without an earlier bake it is a normal `set_maps`.
    pub fn replace_id_map(&mut self, baked: BakedMaps, report: Arc<BakeReport>) {
        if !self.has_bake {
            return self.set_maps(baked, report);
        }
        let BakedMaps { maps, .. } = baked;
        let [_, _, _, _, id] = maps;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let id = if id.width != self.res { self.resample(&mut encoder, &id, self.res) } else { id };
        self.queue.submit([encoder.finish()]);
        self.maps[4] = id;
        self.map_fix = Default::default();
        self.maps_dirty = true;
        self.mark_dirty();
    }

    pub fn set_maps(&mut self, baked: BakedMaps, report: Arc<BakeReport>) {
        // Bakes stop at `bake::MAX_BAKE_RES`; bigger texture sets get them upscaled.
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let BakedMaps { maps, custom } = baked;
        let (maps, custom) = if maps[0].width < self.res {
            let up = maps.each_ref().map(|m| self.resample(&mut encoder, m, self.res));
            let custom = custom.iter().map(|m| self.resample(&mut encoder, m, self.res)).collect();
            (up, custom)
        } else {
            (maps, custom)
        };
        self.custom = custom;
        self.custom_ids = report.custom.clone();
        self.build_custom_array(&mut encoder);
        self.queue.submit([encoder.finish()]);
        if maps[0].width != self.res {
            self.map_fix = Default::default();
        }
        self.edge_max = report.edge_max_distance;
        self.maps = maps;
        self.bake_report = Some(report);
        self.has_bake = true;
        self.maps_dirty = true;
        self.mark_dirty();
    }

    /// Fills a fill layer's image slots from a material graph's outputs
    /// (derived data: no history entry).
    pub fn set_graph_images(&mut self, idx: usize, outputs: [Option<Arc<Tex>>; CHANNELS], name: &str) {
        let Some(layer) = self.layers.get_mut(idx) else { return };
        // A channel the graph has no output for keeps the layer's value.
        let mut images: [Option<FillImage>; CHANNELS] = Default::default();
        for (slot, o) in images.iter_mut().zip(outputs) {
            *slot = o.map(|tex| FillImage { tex, name: name.to_string(), encoded: Arc::new(Vec::new()) });
        }
        layer.images = images;
        self.mark_dirty();
    }

    /// Writes a material graph's output into a layer's mask (derived data:
    /// no history entry), resampled to the texture set.
    pub fn set_graph_mask(&mut self, idx: usize, src: &Tex) {
        let Some(layer) = self.layers.get(idx) else { return };
        if layer.mask.as_ref().is_some_and(|m| m.paint.as_ref().is_none_or(|t| t.width != self.res)) {
            let tex = self.new_mask_texture(false);
            self.layers[idx].mask.as_mut().unwrap().paint = Some(tex);
        }
        let Some(dst) = self.layers[idx].mask.as_ref().and_then(|m| m.paint.as_ref()) else { return };
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.resample_into(&mut encoder, src, dst);
        self.queue.submit([encoder.finish()]);
        self.invalidate_thumbs();
        self.mark_dirty();
    }

    pub fn map_levels_changed(&mut self) {
        self.maps_dirty = true;
        self.mark_dirty();
    }

    pub fn clear_map_fix(&mut self, map: usize, undo: &mut UndoStack) {
        let Some(tex) = self.map_fix[map].take() else { return };
        self.record(undo, format!("Clear fix: {}", MAPS[map].name), Edit::Fix { map, tex: Some(tex) });
        self.maps_dirty = true;
        self.mark_dirty();
    }

    /// Packed map texture `t` with levels and fixes applied.
    pub fn map_tex(&self, t: usize) -> &Tex {
        self.resolved[t].as_ref().unwrap_or(&self.maps[t])
    }

    fn map_needs_resolve(&self, t: usize) -> bool {
        MAPS.iter().enumerate().any(|(i, m)| {
            m.tex == t && (self.map_fix[i].is_some() || (m.kind == MapKind::Scalar && self.map_levels[i] != Levels::default()))
        })
    }

    /// Applies levels and painted fixes to every packed texture that has any.
    fn resolve_maps(&mut self) {
        self.maps_version += 1;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        for t in 0..MAP_TEXTURES {
            if !self.map_needs_resolve(t) {
                self.resolved[t] = None;
                continue;
            }
            let mut u: ResolveUniform = bytemuck::Zeroable::zeroed();
            let mut fixes: [Option<&Tex>; 4] = [None; 4];
            for c in 0..4 {
                u.levels[c] = [0.0, 1.0, 1.0, 0.0];
                u.kind[c] = 2.0;
            }
            for (i, m) in MAPS.iter().enumerate() {
                if m.tex != t {
                    continue;
                }
                let comps: &[usize] = match m.kind {
                    MapKind::Normal => &[m.comp, m.comp + 1],
                    _ => std::slice::from_ref(&m.comp),
                };
                for &c in comps {
                    let lv = self.map_levels[i];
                    let fix = self.map_fix[i].as_ref();
                    u.kind[c] = if m.kind == MapKind::Normal { 1.0 } else if m.kind == MapKind::Scalar { 0.0 } else { 2.0 };
                    u.levels[c] = [lv.black, lv.white, lv.gamma, if fix.is_some() { 1.0 } else { 0.0 }];
                    fixes[c] = fix;
                }
            }
            let ubuf = wgpu::util::DeviceExt::create_buffer_init(
                &self.device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some("resolve uniform"),
                    contents: bytemuck::bytes_of(&u),
                    usage: wgpu::BufferUsages::UNIFORM,
                },
            );
            let fix_view = |c: usize| wgpu::BindingResource::TextureView(fixes[c].map_or(&self.dummy.view, |t| &t.view));
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("maps resolve"),
                layout: &self.p.resolve_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.maps[t].view) },
                    wgpu::BindGroupEntry { binding: 2, resource: fix_view(0) },
                    wgpu::BindGroupEntry { binding: 3, resource: fix_view(1) },
                    wgpu::BindGroupEntry { binding: 4, resource: fix_view(2) },
                    wgpu::BindGroupEntry { binding: 5, resource: fix_view(3) },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            });
            let res = self.res;
            let reuse = self.resolved[t].as_ref().is_some_and(|r| r.width == res);
            if !reuse {
                self.resolved[t] = Some(Tex::new(&self.device, "resolved map", res, res, MAP_FORMATS[t], LAYER_USAGE));
            }
            let out = self.resolved[t].as_ref().unwrap();
            let mut pass = gpu::pass(&mut encoder, "maps resolve", &[gpu::color_attachment(&out.view, wgpu::LoadOp::Load)]);
            pass.set_pipeline(&self.p.resolve[&MAP_FORMATS[t]]);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        self.maps_dirty = false;
    }

    /// World-space position and normal maps of the texture set (for display/export).
    pub fn surface_maps(&self) -> (&Tex, &Tex) {
        (&self.pos, &self.nrm)
    }

    /// The mesh surface material graphs' surface nodes read.
    pub fn graph_surface(&self) -> crate::graph::runtime::Surface<'_> {
        let (lo, hi) = self.bounds;
        crate::graph::runtime::Surface {
            pos: &self.pos,
            nrm: &self.nrm,
            maps: [self.map_tex(0), self.map_tex(1), self.map_tex(2), self.map_tex(4)],
            bmin: lo.to_array(),
            bsize: (hi - lo).to_array(),
            edge_max: self.edge_max,
        }
    }

    // ------------------------------------------------------------------ layers

    // ------------------------------------------------------------------ projects

    /// A layer-sized texture holding `bytes` (tightly packed texels).
    pub fn texture_from_bytes(&self, label: &str, format: F, width: u32, height: u32, bytes: &[u8]) -> Tex {
        let tex = Tex::new(&self.device, label, width, height, format, LAYER_USAGE);
        let row = width * gpu::bytes_per_texel(format);
        self.queue.write_texture(
            tex.copy_info(),
            bytes,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: None },
            tex.extent(),
        );
        tex
    }

    pub fn read_tex(&self, tex: &Tex) -> Vec<u8> {
        gpu::read_texture(&self.device, &self.queue, tex)
    }

    /// Replaces the whole stack with loaded layers (no history).
    pub fn restore_layers(&mut self, mut layers: Vec<Layer>) {
        for mut l in self.layers.drain(..) {
            Self::release_thumb(&self.rs, &mut l);
        }
        for l in &mut layers {
            Self::attach_thumb(&self.rs, l);
        }
        self.next_id = layers.iter().map(|l| l.id).max().unwrap_or(0);
        self.layers = layers;
        self.folder_below.clear();
        self.committed = self.doc_props();
        self.maps_dirty = true;
        self.mark_dirty();
    }

    /// The final material at a UV position: sRGB color, roughness, metallic.
    pub fn sample_final(&self, uv: glam::Vec2) -> ([f32; 3], f32, f32) {
        let res = self.res as f32;
        let (x, y) = ((uv.x * res).clamp(0.0, res - 1.0) as u32, (uv.y * res).clamp(0.0, res - 1.0) as u32);
        // One texel of each channel, 256-byte aligned rows in one buffer.
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick"),
            size: 256 * 3,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        for ch in 0..3 {
            let tex = self.final_tex(ch);
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x, y, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout { offset: ch as u64 * 256, bytes_per_row: Some(256), rows_per_image: None },
                },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );
        }
        self.queue.submit([encoder.finish()]);
        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let data = slice.get_mapped_range().expect("pick buffer is mapped");
        let unorm = |b: u8| b as f32 / 255.0;
        let result = ([unorm(data[0]), unorm(data[1]), unorm(data[2])], unorm(data[256]), unorm(data[512]));
        drop(data);
        buffer.unmap();
        result
    }

    /// Decodes an image file into a mipmapped texture for a fill layer.
    /// A fill image from a file, sharing the GPU texture with other layers
    /// that use the same file.
    pub fn shared_fill_image(&mut self, path: &std::path::Path) -> Result<FillImage, String> {
        let dir = path.parent().and_then(|d| d.file_name()).map_or(String::new(), |d| format!("{}/", d.to_string_lossy()));
        let name = format!("{dir}{}", path.file_name().map_or("image".into(), |n| n.to_string_lossy()));
        if let Some((tex, encoded)) = self.image_cache.get(path)
            && let Some(tex) = tex.upgrade() {
                return Ok(FillImage { tex, name, encoded: encoded.clone() });
            }
        let encoded = Arc::new(std::fs::read(path).map_err(|e| format!("can't read {}: {e}", path.display()))?);
        let img = self.decode_fill_image(name, encoded.clone())?;
        self.image_cache.retain(|_, (t, _)| t.strong_count() > 0);
        self.image_cache.insert(path.to_path_buf(), (Arc::downgrade(&img.tex), encoded));
        Ok(img)
    }

    /// Decodes an image file into a mipmapped fill texture. Grayscale files
    /// (roughness, height, masks) stay single-channel: a quarter of the memory.
    pub fn decode_fill_image(&self, name: String, encoded: Arc<Vec<u8>>) -> Result<FillImage, String> {
        let img = image::load_from_memory(&encoded).map_err(|e| format!("can't read {name}: {e}"))?;
        let gray = matches!(img.color(), image::ColorType::L8 | image::ColorType::L16);
        let tex = if gray { self.upload_mips(img.into_luma8(), F::R8Unorm) } else { self.upload_mips(img.into_rgba8(), F::Rgba8Unorm) };
        Ok(FillImage { tex: Arc::new(tex), name, encoded })
    }

    /// An image for fill channel `ch`. Opacity takes the image's alpha when
    /// it has one (a cut-out: leaves, lace, decals), else its brightness.
    pub fn decode_channel_image(&self, ch: usize, name: String, encoded: Arc<Vec<u8>>) -> Result<FillImage, String> {
        if ch != OPACITY {
            return self.decode_fill_image(name, encoded);
        }
        let img = image::load_from_memory(&encoded).map_err(|e| format!("can't read {name}: {e}"))?;
        let gray = if img.color().has_alpha() {
            let rgba = img.into_rgba8();
            image::GrayImage::from_fn(rgba.width(), rgba.height(), |x, y| image::Luma([rgba.get_pixel(x, y)[3]]))
        } else {
            img.into_luma8()
        };
        Ok(FillImage { tex: Arc::new(self.upload_mips(gray, F::R8Unorm)), name, encoded })
    }

    fn upload_mips<P: image::Pixel<Subpixel = u8> + 'static>(&self, mut img: image::ImageBuffer<P, Vec<u8>>, format: F) -> Tex {
        if img.width().max(img.height()) > MAX_IMAGE {
            let k = MAX_IMAGE as f32 / img.width().max(img.height()) as f32;
            let (w, h) = (((img.width() as f32 * k) as u32).max(1), ((img.height() as f32 * k) as u32).max(1));
            img = image::imageops::resize(&img, w, h, image::imageops::FilterType::Triangle);
        }
        gpu::upload_mips(&self.device, &self.queue, img, format, "fill image")
    }

    /// Sets or clears the image on one channel of a fill layer, enabling the
    /// channel when an image is set; one undo step.
    pub fn set_fill_image(&mut self, idx: usize, ch: usize, image: Option<FillImage>, undo: &mut UndoStack) {
        let Some(layer) = self.layers.get(idx) else { return };
        if !layer.is_fill() || ch >= CHANNELS {
            return;
        }
        // Opacity reads the image's alpha (or brightness): decoded for that.
        let image = match image {
            Some(im) if ch == OPACITY && im.tex.format != F::R8Unorm && !im.encoded.is_empty() => {
                self.decode_channel_image(ch, im.name.clone(), im.encoded.clone()).ok().or(Some(im))
            }
            other => other,
        };
        let label = format!("Image on {}", layer.name);
        self.commit_props(undo);
        undo.begin_group(label.clone());
        let layer = &mut self.layers[idx];
        if let (Content::Fill(f), true) = (&mut layer.content, image.is_some()) {
            f.mat.enabled[ch] = true;
            // A height image is scaled by the height value: start it as is.
            if ch == 3 && f.mat.height == 0.0 && layer.images[3].is_none() {
                f.mat.height = 0.5;
            }
            // An emission image is tinted by the emission color: black would hide it.
            if ch == EMISSION && f.mat.emission == [0.0; 3] {
                f.mat.emission = [1.0; 3];
            }
        }
        let mut images = layer.images.clone();
        images[ch] = image;
        let old = std::mem::replace(&mut layer.images, images);
        let id = layer.id;
        self.commit_props(undo);
        self.record(undo, label, Edit::Images { id, images: old });
        undo.end_group();
        self.mark_dirty();
    }

    /// Property edits not yet recorded in the history.
    pub fn has_uncommitted_props(&self) -> bool {
        self.doc_props() != self.committed
    }

    fn new_id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    // ------------------------------------------------------------------ folders

    pub fn index_of(&self, id: u64) -> Option<usize> {
        self.layers.iter().position(|l| l.id == id)
    }

    /// Folders containing layer `idx`, innermost first.
    pub fn ancestors(&self, idx: usize) -> impl Iterator<Item = usize> + '_ {
        std::iter::successors(self.layers[idx].parent.and_then(|p| self.index_of(p)), |&i| {
            self.layers[i].parent.and_then(|p| self.index_of(p))
        })
    }

    pub fn depth(&self, idx: usize) -> usize {
        self.ancestors(idx).count()
    }

    fn is_inside(&self, idx: usize, folder: u64) -> bool {
        self.ancestors(idx).any(|a| self.layers[a].id == folder)
    }

    /// A layer with everything inside it: contiguous, ending at the layer.
    pub fn block(&self, idx: usize) -> std::ops::Range<usize> {
        let id = self.layers[idx].id;
        let mut start = idx;
        while start > 0 && self.is_inside(start - 1, id) {
            start -= 1;
        }
        start..idx + 1
    }

    /// The folder a layer inserted at `at` belongs in, keeping every folder's
    /// layers directly below it: right below a folder is inside it, right
    /// below any other layer is next to it.
    pub fn parent_for_insert(&self, at: usize) -> Option<u64> {
        let above = self.layers.get(at)?;
        if above.is_folder() { Some(above.id) } else { above.parent }
    }

    pub fn add_folder(&mut self, at: usize, parent: Option<u64>, name: String, undo: &mut UndoStack) -> usize {
        let layer = Layer {
            id: self.new_id(),
            name,
            visible: true,
            opacity: 1.0,
            blend: Blend::Normal,
            content: Content::Folder(Folder { open: true, graph: None }),
            mask: None,
            parent,
            images: Default::default(),
            thumb: None,
        };
        self.insert_layer(at, layer, undo, "Add folder".into())
    }

    /// Moves a layer, with everything inside it if it's a folder.
    pub fn move_to(&mut self, from: usize, drop: Drop, undo: &mut UndoStack) -> Option<usize> {
        if from >= self.layers.len() {
            return None;
        }
        let block = self.block(from);
        let anchor = match drop {
            Drop::Above(a) | Drop::Below(a) | Drop::Into(a) => a,
        };
        let anchor_idx = self.index_of(anchor)?;
        if block.contains(&anchor_idx) || (matches!(drop, Drop::Into(_)) && !self.layers[anchor_idx].is_folder()) {
            return None;
        }
        self.commit_props(undo);
        let moved = &self.layers[from];
        let (id, old_parent, label) = (moved.id, moved.parent, format!("Move {}", moved.name));
        undo.begin_group(label.clone());
        let order = self.layers.iter().map(|l| l.id).collect();
        let taken: Vec<Layer> = self.layers.drain(block).collect();
        let ai = self.index_of(anchor).expect("the anchor is outside the moved block");
        let (at, parent) = match drop {
            Drop::Above(_) => (ai + 1, self.layers[ai].parent),
            Drop::Below(_) => (self.block(ai).start, self.layers[ai].parent),
            Drop::Into(_) => (ai, Some(anchor)),
        };
        let top = at + taken.len() - 1;
        self.layers.splice(at..at, taken);
        self.record(undo, label.clone(), Edit::Order(order));
        if parent != old_parent {
            self.layers[top].parent = parent;
            self.record(undo, label, Edit::Parent { id, parent: old_parent });
        }
        undo.end_group();
        self.mark_dirty();
        Some(top)
    }

    fn insert_layer(&mut self, at: usize, layer: Layer, undo: &mut UndoStack, label: String) -> usize {
        self.commit_props(undo);
        let at = at.min(self.layers.len());
        let id = layer.id;
        self.layers.insert(at, layer);
        self.record(undo, label, Edit::Layer { index: at, id, layer: None });
        self.mark_dirty();
        at
    }

    pub fn add_paint_layer(&mut self, at: usize, parent: Option<u64>, name: Option<String>, undo: &mut UndoStack) -> usize {
        let id = self.new_id();
        let n = self.layers.iter().filter(|l| l.is_paint()).count() + 1;
        let layer = Layer {
            id,
            name: name.unwrap_or_else(|| format!("Layer {n}")),
            visible: true,
            opacity: 1.0,
            blend: Blend::Normal,
            content: Content::Paint(Default::default()),
            mask: None,
            parent,
            images: Default::default(),
            thumb: None,
        };
        self.insert_layer(at, layer, undo, "Add paint layer".into())
    }

    /// A paint layer holding given texels per channel (base color, roughness, metallic, height:
    /// `(size, texels in PAINT_FORMATS layout)`), resampled to the texture set's size.
    pub fn add_paint_layer_from(
        &mut self,
        at: usize,
        parent: Option<u64>,
        name: String,
        channels: [Option<(u32, Vec<u8>)>; CHANNELS],
        undo: &mut UndoStack,
    ) -> usize {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let mut chans: [Option<Tex>; CHANNELS] = Default::default();
        for (ch, data) in channels.into_iter().enumerate() {
            let Some((size, texels)) = data else { continue };
            let format = PAINT_FORMATS[ch];
            if texels.len() != (size * size * gpu::bytes_per_texel(format)) as usize {
                continue;
            }
            let src = Tex::new(&self.device, CH_NAMES[ch], size, size, format, LAYER_USAGE);
            let layout = wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(size * gpu::bytes_per_texel(format)), rows_per_image: Some(size) };
            self.queue.write_texture(src.copy_info(), &texels, layout, src.extent());
            chans[ch] = Some(if size == self.res { src } else { self.resample(&mut encoder, &src, self.res) });
        }
        self.queue.submit([encoder.finish()]);
        let thumb = chans[0].as_ref().map(|c| self.rs.renderer.write().register_native_texture(&self.device, c.srgb(), wgpu::FilterMode::Linear));
        let layer = Layer {
            id: self.new_id(),
            name,
            visible: true,
            opacity: 1.0,
            blend: Blend::Normal,
            content: Content::Paint(chans),
            mask: None,
            parent,
            images: Default::default(),
            thumb,
        };
        self.insert_layer(at, layer, undo, "Add paint layer".into())
    }

    pub fn add_fill_layer(
        &mut self,
        at: usize,
        parent: Option<u64>,
        name: String,
        fill: FillParams,
        undo: &mut UndoStack,
    ) -> usize {
        let id = self.new_id();
        let label = format!("Add {name}");
        let layer = Layer {
            id,
            name,
            visible: true,
            opacity: 1.0,
            blend: Blend::Normal,
            content: Content::Fill(fill),
            mask: None,
            parent,
            images: Default::default(),
            thumb: None,
        };
        self.insert_layer(at, layer, undo, label)
    }

    /// Removes a layer, with everything inside it if it's a folder; textures
    /// stay in the history until the entry expires.
    pub fn delete_layer(&mut self, idx: usize, undo: &mut UndoStack) {
        if idx >= self.layers.len() {
            return;
        }
        self.commit_props(undo);
        let label = format!("Delete {}", self.layers[idx].name);
        undo.begin_group(label.clone());
        for i in self.block(idx).rev() {
            let mut layer = self.layers.remove(i);
            Self::release_thumb(&self.rs, &mut layer);
            self.record(undo, label.clone(), Edit::Layer { index: i, id: layer.id, layer: Some(layer) });
        }
        undo.end_group();
        self.mark_dirty();
    }

    /// Duplicates a layer (a folder with everything inside) above itself;
    /// returns the copy's index.
    pub fn duplicate_layer(&mut self, idx: usize, undo: &mut UndoStack) -> Option<usize> {
        let src = self.layers.get(idx)?;
        let label = format!("Duplicate {}", src.name);
        let block = self.block(idx);
        self.commit_props(undo);
        undo.begin_group(label);
        let mut new_ids: HashMap<u64, u64> = HashMap::new();
        let mut at = idx + 1;
        for i in block {
            let copy = self.copy_layer(i);
            new_ids.insert(self.layers[i].id, copy.id);
            let mut copy = copy;
            copy.parent = self.layers[i].parent.map(|p| *new_ids.get(&p).unwrap_or(&p));
            if i == idx {
                copy.name = format!("{} copy", copy.name);
            }
            let label = format!("Duplicate {}", copy.name);
            at = self.insert_layer(at, copy, undo, label) + 1;
        }
        undo.end_group();
        Some(at - 1)
    }

    /// Deep-copies the selected layer and its folder contents for the app clipboard.
    /// Parent links inside the copied block are remapped, while an external parent
    /// is left for the destination stack to replace during insertion.
    pub fn copy_layer_block(&mut self, idx: usize) -> Option<Vec<Layer>> {
        if idx >= self.layers.len() {
            return None;
        }
        let indices: Vec<usize> = self.block(idx).collect();
        let mut copies: Vec<Layer> = indices.iter().map(|&i| self.copy_layer(i)).collect();
        let ids: HashMap<u64, u64> = indices
            .iter()
            .zip(&copies)
            .map(|(&i, copy)| (self.layers[i].id, copy.id))
            .collect();
        for (copy, &i) in copies.iter_mut().zip(&indices) {
            copy.parent = self.layers[i].parent.map(|p| ids.get(&p).copied().unwrap_or(p));
        }
        Some(copies)
    }

    /// Inserts clipboard layers into this texture set. Painted channels and masks
    /// are resampled to this set's resolution; fill images remain shared bitmaps.
    pub fn insert_layers_resampled(
        &mut self,
        at: usize,
        parent: Option<u64>,
        mut layers: Vec<Layer>,
        label: String,
        undo: &mut UndoStack,
    ) -> Vec<u64> {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let mut resampled = false;
        for layer in &mut layers {
            if let Content::Paint(channels) = &mut layer.content {
                for tex in channels.iter_mut().flatten() {
                    if tex.width != self.res || tex.height != self.res {
                        *tex = self.resample(&mut encoder, tex, self.res);
                        resampled = true;
                    }
                }
            }
            if let Some(mask) = &mut layer.mask
                && let Some(tex) = &mut mask.paint
                    && (tex.width != self.res || tex.height != self.res) {
                        *tex = self.resample(&mut encoder, tex, self.res);
                        resampled = true;
                    }
        }
        if resampled {
            self.queue.submit([encoder.finish()]);
        }
        self.insert_layers(at, parent, layers, label, undo)
    }

    /// Inserts layers (bottom first, a folder after its contents, as in the
    /// stack) at `at` as one undo step. They get new ids; parents among them
    /// are remapped, and the rest go into `parent`. Returns the new ids.
    pub fn insert_layers(&mut self, at: usize, parent: Option<u64>, mut layers: Vec<Layer>, label: String, undo: &mut UndoStack) -> Vec<u64> {
        self.commit_props(undo);
        undo.begin_group(label);
        let mut new_ids: HashMap<u64, u64> = HashMap::new();
        for l in &mut layers {
            let id = self.new_id();
            new_ids.insert(l.id, id);
            l.id = id;
        }
        let mut at = at.min(self.layers.len());
        let mut ids = Vec::with_capacity(layers.len());
        for mut l in layers {
            l.parent = l.parent.and_then(|p| new_ids.get(&p).copied()).or(parent);
            Self::attach_thumb(&self.rs, &mut l);
            ids.push(l.id);
            let label = format!("Add {}", l.name);
            at = self.insert_layer(at, l, undo, label) + 1;
        }
        undo.end_group();
        ids
    }

    /// A copy of one layer with new textures and id, not yet in the stack.
    fn copy_layer(&mut self, idx: usize) -> Layer {
        let src = &self.layers[idx];
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let copy = |encoder: &mut wgpu::CommandEncoder, t: &Tex| {
            let n = Tex::new(&self.device, "layer", t.width, t.height, t.format, LAYER_USAGE);
            encoder.copy_texture_to_texture(t.copy_info(), n.copy_info(), t.extent());
            n
        };
        let content = match &src.content {
            Content::Paint(ch) => Content::Paint(std::array::from_fn(|i| ch[i].as_ref().map(|t| copy(&mut encoder, t)))),
            Content::Fill(f) => Content::Fill(*f),
            // A copy gets a live graph of its own when one is asked for.
            Content::Folder(f) => Content::Folder(Folder { graph: None, ..*f }),
        };
        let mask = src.mask.as_ref().map(|m| Mask {
            black_base: m.black_base,
            paint: m.paint.as_ref().map(|t| copy(&mut encoder, t)),
            generator: m.generator,
            graph: m.graph,
            filters: m.filters.clone(),
            layers: m.layers.clone(),
        });
        let (name, visible, opacity, blend, parent) = (src.name.clone(), src.visible, src.opacity, src.blend, src.parent);
        let images = src.images.clone();
        self.queue.submit([encoder.finish()]);
        let id = self.new_id();
        let mut layer = Layer { id, name, visible, opacity, blend, content, mask, parent, images, thumb: None };
        Self::attach_thumb(&self.rs, &mut layer);
        layer
    }

    /// Moves a layer so it ends up where layer `to` is now (index-based, for
    /// MCP); it joins that layer's folder.
    pub fn move_layer(&mut self, from: usize, to: usize, undo: &mut UndoStack) -> Option<usize> {
        let target = self.layers.get(to.min(self.layers.len().checked_sub(1)?))?.id;
        let drop = if to > from { Drop::Above(target) } else { Drop::Below(target) };
        self.move_to(from, drop, undo)
    }

    /// Empties a paint layer; the painted data moves into the history.
    pub fn clear_layer(&mut self, idx: usize, undo: &mut UndoStack) {
        let Some(layer) = self.layers.get(idx) else { return };
        if !layer.is_paint() {
            return;
        }
        self.commit_props(undo);
        let layer = &mut self.layers[idx];
        Self::release_thumb(&self.rs, layer);
        let old = std::mem::replace(&mut layer.content, Content::Paint(Default::default()));
        let (id, label) = (layer.id, format!("Clear {}", layer.name));
        self.record(undo, label, Edit::Content { id, content: old });
        self.mark_dirty();
    }

    pub fn add_mask(&mut self, idx: usize, white: bool, generator: Option<Generator>, undo: &mut UndoStack) {
        if idx >= self.layers.len() {
            return;
        }
        self.commit_props(undo);
        let paint = generator.is_none().then(|| self.new_mask_texture(white));
        let black_base = generator.is_none() && !white;
        let layer = &mut self.layers[idx];
        let old = layer.mask.replace(Mask { black_base, paint, generator, graph: None, filters: Vec::new(), layers: Vec::new() });
        let (id, label) = (layer.id, format!("Add mask to {}", layer.name));
        self.record(undo, label, Edit::Mask { id, mask: old });
        self.mark_dirty();
    }

    /// Adds a paintable (white) layer to an existing generator mask. Not
    /// recorded: a white paint mask leaves the result unchanged.
    pub fn ensure_mask_paint(&mut self, idx: usize) {
        let needs = self.layers.get(idx).and_then(|l| l.mask.as_ref()).is_some_and(|m| m.paint.is_none());
        if needs {
            let tex = self.new_mask_texture(true);
            if let Some(m) = self.layers[idx].mask.as_mut() {
                m.paint = Some(tex);
            }
            self.mark_dirty();
        }
    }

    /// Writes a triangle selection into the layer's paint mask (adding a black
    /// mask if it has none), as one undo step.
    pub fn mask_selection(
        &mut self,
        idx: usize,
        mesh: &crate::mesh::Mesh,
        selected: &[bool],
        mode: crate::select::MaskMode,
        undo: &mut UndoStack,
    ) {
        let Some(layer) = self.layers.get(idx) else { return };
        let label = format!("Mask selection on {}", layer.name);
        self.commit_props(undo);
        undo.begin_group(label.clone());
        if self.layers[idx].mask.is_none() {
            self.add_mask(idx, false, None, undo);
        }
        self.ensure_mask_paint(idx);
        let layer_id = self.layers[idx].id;
        let tex = self.layers[idx].mask.as_ref().and_then(|m| m.paint.as_ref()).expect("mask paint exists");

        let mut texels = gpu::read_texture(&self.device, &self.queue, tex);
        crate::select::write_mask(mesh, selected, self.res, &mut texels, mode);

        let mut encoder = self.device.create_command_encoder(&Default::default());
        let before = Snapshot::encode(&self.device, &mut encoder, &[tex]);
        self.queue.submit([encoder.finish()]);
        before.request_map();
        // Queue writes run at the start of the next submission, after the "before" copy.
        self.queue.write_texture(
            tex.copy_info(),
            &texels,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(self.res), rows_per_image: Some(self.res) },
            tex.extent(),
        );
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let after = Snapshot::encode(&self.device, &mut encoder, &[tex]);
        self.queue.submit([encoder.finish()]);
        after.request_map();

        undo.push(
            label,
            Edit::Stroke(UndoEntry { layer_id, target: StrokeTarget::Mask, channels: vec![], before, after }),
        );
        undo.end_group();
        self.mark_dirty();
    }

    /// Points a layer's mask at a smart mask, a mask graph or neither (a
    /// white mask), keeping its filters and mask layers; one undo step.
    pub fn set_mask_source(&mut self, idx: usize, generator: Option<Generator>, graph: Option<u64>, undo: &mut UndoStack) {
        if idx >= self.layers.len() {
            return;
        }
        self.commit_props(undo);
        // A mask graph's output arrives as the paint (`set_graph_mask`).
        let paint = (generator.is_none() && graph.is_none()).then(|| self.new_mask_texture(true));
        let layer = &mut self.layers[idx];
        let (filters, layers) = layer.mask.as_ref().map_or_else(Default::default, |m| (m.filters.clone(), m.layers.clone()));
        let old = layer.mask.replace(Mask { black_base: false, paint, generator, graph, filters, layers });
        let (id, label) = (layer.id, format!("Change the mask of {}", layer.name));
        self.record(undo, label, Edit::Mask { id, mask: old });
        self.mark_dirty();
    }

    pub fn remove_mask(&mut self, idx: usize, undo: &mut UndoStack) {
        let Some(layer) = self.layers.get(idx) else { return };
        if layer.mask.is_none() {
            return;
        }
        self.commit_props(undo);
        let layer = &mut self.layers[idx];
        let old = layer.mask.take();
        let (id, label) = (layer.id, format!("Remove mask from {}", layer.name));
        self.record(undo, label, Edit::Mask { id, mask: old });
        self.mark_dirty();
    }

    // ------------------------------------------------------------------ history

    fn release_thumb(rs: &egui_wgpu::RenderState, layer: &mut Layer) {
        if let Some(t) = layer.thumb.take() {
            rs.renderer.write().free_texture(&t);
        }
    }

    fn attach_thumb(rs: &egui_wgpu::RenderState, layer: &mut Layer) {
        if layer.thumb.is_some() {
            return;
        }
        if let Content::Paint(ch) = &layer.content
            && let Some(color) = &ch[0] {
                layer.thumb = Some(rs.renderer.write().register_native_texture(
                    &rs.device,
                    color.srgb(),
                    wgpu::FilterMode::Linear,
                ));
            }
    }

    fn layer_props(layer: &Layer) -> LayerProps {
        LayerProps {
            name: layer.name.clone(),
            visible: layer.visible,
            opacity: layer.opacity,
            blend: layer.blend,
            fill: match &layer.content {
                Content::Fill(f) => Some(*f),
                Content::Paint(_) | Content::Folder(_) => None,
            },
            generator: layer.mask.as_ref().and_then(|m| m.generator),
            mask_filters: layer.mask.as_ref().map_or_else(Vec::new, |m| m.filters.clone()),
            mask_layers: layer.mask.as_ref().map_or_else(Vec::new, |m| m.layers.clone()),
        }
    }

    fn set_layer_props(layer: &mut Layer, p: &LayerProps) {
        layer.name.clone_from(&p.name);
        layer.visible = p.visible;
        layer.opacity = p.opacity;
        layer.blend = p.blend;
        if let (Content::Fill(f), Some(pf)) = (&mut layer.content, p.fill) {
            *f = pf;
        }
        if let Some(m) = &mut layer.mask {
            m.generator = p.generator;
            m.filters.clone_from(&p.mask_filters);
            m.layers.clone_from(&p.mask_layers);
        }
    }

    fn doc_props(&self) -> DocProps {
        DocProps {
            layers: self.layers.iter().map(|l| (l.id, Self::layer_props(l))).collect(),
            levels: self.map_levels.to_vec(),
        }
    }

    fn record(&mut self, undo: &mut UndoStack, label: String, edit: Edit) {
        undo.push(label, edit);
        self.committed = self.doc_props();
    }

    /// Records property edits (names, opacity, blend, fill and generator
    /// settings, map levels) made since the last recorded edit as one step.
    /// The app calls this once an interaction ends, so a slider drag becomes a
    /// single entry; every structural edit calls it first to keep the order.
    pub fn commit_props(&mut self, undo: &mut UndoStack) {
        let current = self.doc_props();
        if current == self.committed {
            return;
        }
        let same_layers = current.layers.len() == self.committed.layers.len()
            && current.layers.iter().zip(&self.committed.layers).all(|(a, b)| a.0 == b.0);
        if same_layers {
            let changed: Vec<&str> = current
                .layers
                .iter()
                .zip(&self.committed.layers)
                .filter(|(a, b)| a.1 != b.1)
                .map(|(a, _)| a.1.name.as_str())
                .collect();
            let label = match changed.as_slice() {
                [] => "Edit map levels".to_string(),
                [one] => format!("Edit {one}"),
                many => format!("Edit {} layers", many.len()),
            };
            let before = std::mem::replace(&mut self.committed, current);
            undo.push(label, Edit::Props(before));
        } else {
            self.committed = current;
        }
    }

    /// Applies a swap edit (see [`Edit`]): afterwards `edit` holds the state it replaced.
    fn apply_edit(&mut self, edit: &mut Edit, undoing: bool) {
        match edit {
            Edit::Stroke(e) => {
                if let Some(textures) = self.target_textures(e.layer_id, e.target, &e.channels) {
                    let snap = if undoing { &mut e.before } else { &mut e.after };
                    snap.restore(&self.device, &self.queue, &textures);
                }
            }
            Edit::Layer { index, id, layer } => match layer.take() {
                Some(mut l) => {
                    Self::attach_thumb(&self.rs, &mut l);
                    let at = (*index).min(self.layers.len());
                    self.layers.insert(at, l);
                }
                None => {
                    if let Some(pos) = self.layers.iter().position(|l| l.id == *id) {
                        let mut l = self.layers.remove(pos);
                        Self::release_thumb(&self.rs, &mut l);
                        *index = pos;
                        *layer = Some(l);
                    }
                }
            },
            Edit::Order(order) => {
                let current: Vec<u64> = self.layers.iter().map(|l| l.id).collect();
                let rank = |id: u64| order.iter().position(|&o| o == id).unwrap_or(usize::MAX);
                self.layers.sort_by_key(|l| rank(l.id));
                *order = current;
            }
            Edit::Content { id, content } => {
                if let Some(l) = self.layers.iter_mut().find(|l| l.id == *id) {
                    Self::release_thumb(&self.rs, l);
                    std::mem::swap(&mut l.content, content);
                    Self::attach_thumb(&self.rs, l);
                }
            }
            Edit::Mask { id, mask } => {
                if let Some(l) = self.layers.iter_mut().find(|l| l.id == *id) {
                    std::mem::swap(&mut l.mask, mask);
                }
            }
            Edit::Images { id, images } => {
                if let Some(l) = self.layers.iter_mut().find(|l| l.id == *id) {
                    std::mem::swap(&mut l.images, images);
                }
            }
            Edit::Parent { id, parent } => {
                if let Some(l) = self.layers.iter_mut().find(|l| l.id == *id) {
                    std::mem::swap(&mut l.parent, parent);
                }
            }
            Edit::Props(props) => {
                let current = self.doc_props();
                for (id, p) in &props.layers {
                    if let Some(l) = self.layers.iter_mut().find(|l| l.id == *id) {
                        Self::set_layer_props(l, p);
                    }
                }
                for (dst, src) in self.map_levels.iter_mut().zip(&props.levels) {
                    *dst = *src;
                }
                *props = current;
            }
            Edit::Fix { map, tex } => std::mem::swap(&mut self.map_fix[*map], tex),
            Edit::Group(edits) => {
                if undoing {
                    edits.iter_mut().rev().for_each(|e| self.apply_edit(e, true));
                } else {
                    edits.iter_mut().for_each(|e| self.apply_edit(e, false));
                }
            }
        }
    }

    fn new_mask_texture(&self, white: bool) -> Tex {
        let tex = Tex::new(&self.device, "mask", self.res, self.res, MASK_FORMAT, LAYER_USAGE);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let v = if white { 1.0 } else { 0.0 };
        gpu::clear_texture(&mut encoder, &tex, wgpu::Color { r: v, g: v, b: v, a: 1.0 });
        self.queue.submit([encoder.finish()]);
        tex
    }

    fn ensure_channel(&mut self, idx: usize, ch: usize) {
        let res = self.res;
        let layer = &mut self.layers[idx];
        let Content::Paint(chans) = &mut layer.content else { return };
        if chans[ch].is_some() {
            return;
        }
        let tex = Tex::new(&self.device, CH_NAMES[ch], res, res, PAINT_FORMATS[ch], LAYER_USAGE);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        gpu::clear_texture(&mut encoder, &tex, wgpu::Color::TRANSPARENT);
        self.queue.submit([encoder.finish()]);
        if ch == 0 {
            layer.thumb = Some(self.rs.renderer.write().register_native_texture(
                &self.device,
                tex.srgb(),
                wgpu::FilterMode::Linear,
            ));
        }
        chans[ch] = Some(tex);
    }

    // ------------------------------------------------------------------ strokes

    fn target_textures(&self, layer_id: u64, target: StrokeTarget, channels: &[usize]) -> Option<Vec<&Tex>> {
        if let StrokeTarget::BakeFix(m) = target {
            return Some(vec![self.map_fix.get(m)?.as_ref()?]);
        }
        let layer = self.layers.iter().find(|l| l.id == layer_id)?;
        match target {
            StrokeTarget::Content => {
                let Content::Paint(ch) = &layer.content else { return None };
                channels.iter().map(|&c| ch[c].as_ref()).collect()
            }
            StrokeTarget::Mask => Some(vec![layer.mask.as_ref()?.paint.as_ref()?]),
            StrokeTarget::BakeFix(_) => None,
        }
    }

    /// Bake fixes are (value, coverage) pairs, the same layout as a roughness
    /// paint channel, so they reuse that channel's stroke pipeline.
    const FIX_CHANNEL: usize = 1;

    fn begin_fix_stroke(&mut self, map: usize, erase: bool) -> bool {
        if !self.has_bake || map >= MAPS.len() || MAPS[map].kind == MapKind::Info {
            return false;
        }
        if self.map_fix[map].is_none() {
            if erase {
                return false;
            }
            let tex = Tex::new(&self.device, "map fix", self.res, self.res, PAINT_FORMATS[Self::FIX_CHANNEL], LAYER_USAGE);
            let mut encoder = self.device.create_command_encoder(&Default::default());
            gpu::clear_texture(&mut encoder, &tex, wgpu::Color::TRANSPARENT);
            self.queue.submit([encoder.finish()]);
            self.map_fix[map] = Some(tex);
        }
        true
    }

    /// Starts a stroke on the content or mask of layer `idx`. Returns false if
    /// there is nothing paintable there.
    pub fn begin_stroke(&mut self, idx: usize, target: StrokeTarget, enabled: [bool; CHANNELS], erase: bool) -> bool {
        if let StrokeTarget::BakeFix(map) = target {
            if !self.begin_fix_stroke(map, erase) {
                return false;
            }
            let c = Self::FIX_CHANNEL;
            if self.snap[c].is_none() {
                self.snap[c] = Some(Tex::new(&self.device, "snapshot", self.res, self.res, PAINT_FORMATS[c], LAYER_USAGE));
            }
            return self.encode_stroke_start(0, target, vec![c]);
        }
        let Some(layer) = self.layers.get(idx) else { return false };
        let layer_id = layer.id;
        let channels: Vec<usize> = match (target, &layer.content) {
            (StrokeTarget::Content, Content::Paint(ch)) => {
                (0..CHANNELS).filter(|&c| enabled[c] && (!erase || ch[c].is_some())).collect()
            }
            (StrokeTarget::Mask, _) => {
                if layer.mask.is_none() {
                    return false;
                }
                vec![]
            }
            _ => return false,
        };
        if target == StrokeTarget::Content && channels.is_empty() {
            return false;
        }
        match target {
            StrokeTarget::Content => {
                for &c in &channels {
                    self.ensure_channel(idx, c);
                    if self.snap[c].is_none() {
                        self.snap[c] =
                            Some(Tex::new(&self.device, "snapshot", self.res, self.res, PAINT_FORMATS[c], LAYER_USAGE));
                    }
                }
            }
            StrokeTarget::Mask => {
                self.ensure_mask_paint(idx);
                if self.snap_mask.is_none() {
                    self.snap_mask =
                        Some(Tex::new(&self.device, "mask snapshot", self.res, self.res, MASK_FORMAT, LAYER_USAGE));
                }
            }
            StrokeTarget::BakeFix(_) => unreachable!("handled above"),
        }
        self.encode_stroke_start(layer_id, target, channels)
    }

    /// Snapshots the stroke target (stroke base + undo "before") and clears the stroke mask.
    fn encode_stroke_start(&mut self, layer_id: u64, target: StrokeTarget, channels: Vec<usize>) -> bool {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        gpu::clear_texture(&mut encoder, &self.stroke, wgpu::Color::TRANSPARENT);
        let textures = self.target_textures(layer_id, target, &channels).expect("stroke target exists");
        let snaps: Vec<&Tex> = match target {
            StrokeTarget::Mask => vec![self.snap_mask.as_ref().unwrap()],
            _ => channels.iter().map(|&c| self.snap[c].as_ref().unwrap()).collect(),
        };
        for (src, dst) in textures.iter().zip(&snaps) {
            encoder.copy_texture_to_texture(src.copy_info(), dst.copy_info(), src.extent());
        }
        let before = Snapshot::encode(&self.device, &mut encoder, &textures);
        self.queue.submit([encoder.finish()]);
        before.request_map();
        self.active = Some(ActiveStroke { layer_id, target, channels, before });
        true
    }

    fn apply_pipeline(&self, mask: u8) -> wgpu::RenderPipeline {
        let p = &self.p;
        let mut cache = p.apply_paint.lock().unwrap_or_else(|e| e.into_inner());
        cache
            .entry(mask)
            .or_insert_with(|| {
                let targets: Vec<_> =
                    (0..CHANNELS).map(|c| if mask & (1 << c) != 0 { gpu::target(PAINT_FORMATS[c]) } else { None }).collect();
                gpu::fullscreen_pipeline(&self.device, "apply paint", &p.apply_module, "fs_paint", &p.apply_layout, &targets)
            })
            .clone()
    }

    /// Lays down this frame's stamps and writes the stroke into the target.
    /// Keep painting passes unscissored: composite bounds are not yet proven safe
    /// as raster bounds for every brush projection and texture-set layout.
    /// `region` bounds the texels the stamps can reach, in UV (u0, v0, u1,
    /// v1), or `None` if unknown.
    pub fn stroke_step(
        &mut self,
        stamps: &[Stamp],
        su: StampUniform,
        depth: &wgpu::TextureView,
        paint: &BrushPaint,
        region: Option<[f32; 4]>,
    ) {
        self.stroke_step_inner(stamps, su, depth, paint, region, false);
    }

    /// Same paint path, with optional GPU timestamps for F6.
    pub fn stroke_step_profiled(
        &mut self,
        stamps: &[Stamp],
        su: StampUniform,
        depth: &wgpu::TextureView,
        paint: &BrushPaint,
        region: Option<[f32; 4]>,
    ) {
        self.stroke_step_inner(stamps, su, depth, paint, region, true);
    }

    fn stroke_step_inner(
        &mut self,
        stamps: &[Stamp],
        mut su: StampUniform,
        depth: &wgpu::TextureView,
        paint: &BrushPaint,
        region: Option<[f32; 4]>,
        profile_gpu: bool,
    ) {
        if self.active.is_none() || stamps.is_empty() {
            return;
        }
        self.add_region(region);
        let stamps = &stamps[..stamps.len().min(MAX_STAMPS)];
        su.view[3] = stamps.len() as f32;
        self.queue.write_buffer(&self.stamp_ubuf, 0, bytemuck::bytes_of(&su));
        self.queue.write_buffer(&self.stamp_sbuf, 0, bytemuck::cast_slice(stamps));

        let device = &self.device;
        let stamp_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stamp"),
            layout: &self.p.stamp_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.stamp_ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.stamp_sbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.pos.view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.nrm.view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(depth) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.brush_alpha.view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        let profile = if profile_gpu && self.active_gpu_profile.is_none() {
            self.gpu_profiler.as_ref().and_then(GpuStrokeProfiler::resources)
        } else {
            None
        };
        if let Some(p) = &profile {
            self.active_gpu_profile = Some(p.clone());
        }
        let stamp_attachments = [gpu::color_attachment(&self.stroke.view, wgpu::LoadOp::Load)];
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stamp"),
                color_attachments: &stamp_attachments,
                depth_stencil_attachment: None,
                timestamp_writes: profile.as_ref().map(|p| wgpu::RenderPassTimestampWrites {
                    query_set: &p.query_set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }),
                occlusion_query_set: None,
                multiview_mask: None,
            }).forget_lifetime();
            pass.set_pipeline(&self.p.stamp);
            pass.set_bind_group(0, &stamp_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        self.encode_apply(&mut encoder, paint, profile.as_ref());
        self.queue.submit([encoder.finish()]);
    }

    /// Returns completed GPU timings for stamp and apply; does not wait for the GPU.
    pub fn poll_gpu_stroke_times(&mut self) -> Option<(f64, f64, f64, Option<f64>)> {
        self.gpu_profiler.as_mut()?.poll(&self.device, &self.queue)
    }

    pub fn gpu_stroke_profile_pending(&self) -> bool {
        self.gpu_profiler.as_ref().is_some_and(|p| p.pending)
    }

    /// Polygon fill: adds whole triangles to the stroke (UV-space vertices,
    /// three per triangle) and writes the stroke into the target.
    pub fn stroke_polys(&mut self, geometry: &PolyGeometry, paint: &BrushPaint) {
        let (faces, padding) = (&geometry.faces, &geometry.padding);
        if self.active.is_none() || faces.is_empty() {
            return;
        }
        let (lo, hi) = faces.iter().chain(padding).fold(([f32::MAX; 2], [f32::MIN; 2]), |(lo, hi), p| {
            ([lo[0].min(p[0]), lo[1].min(p[1])], [hi[0].max(p[0]), hi[1].max(p[1])])
        });
        self.add_region(Some([lo[0], lo[1], hi[0], hi[1]]));
        use wgpu::util::DeviceExt;
        let all: Vec<[f32; 2]> = faces.iter().chain(padding).copied().collect();
        let vertices = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("poly fill"),
            contents: bytemuck::cast_slice(&all),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let pos_map = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("poly padding"),
            layout: &self.p.poly_pad_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.pos.view) }],
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = gpu::pass(
                &mut encoder,
                "poly fill",
                &[gpu::color_attachment(&self.stroke.view, wgpu::LoadOp::Load)],
            );
            pass.set_vertex_buffer(0, vertices.slice(..));
            pass.set_pipeline(&self.p.poly);
            pass.draw(0..faces.len() as u32, 0..1);
            // Padding only where no UV island is: never onto a neighbor's texels.
            if !padding.is_empty() {
                pass.set_pipeline(&self.p.poly_pad);
                pass.set_bind_group(0, &pos_map, &[]);
                pass.draw(faces.len() as u32..all.len() as u32, 0..1);
            }
        }
        self.encode_apply(&mut encoder, paint, None);
        self.queue.submit([encoder.finish()]);
    }

    /// Brush grain (`size`² grayscale, tileable), or `None` for no grain.
    pub fn set_brush_grain(&mut self, grain: Option<(&[u8], u32)>) {
        let (pixels, size) = grain.unwrap_or((&[255], 1));
        if self.brush_grain.width != size {
            self.brush_grain = Tex::new(&self.device, "brush grain", size, size, F::R8Unorm, LAYER_USAGE);
        }
        self.queue.write_texture(
            self.brush_grain.copy_info(),
            pixels,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(size), rows_per_image: Some(size) },
            self.brush_grain.extent(),
        );
    }

    /// Brush tip for the stamps (`size`² grayscale), or `None` for round.
    pub fn set_brush_alpha(&mut self, alpha: Option<(&[u8], u32)>) {
        let (pixels, size) = alpha.unwrap_or((&[255], 1));
        if self.brush_alpha.width != size {
            self.brush_alpha = Tex::new(&self.device, "brush alpha", size, size, F::R8Unorm, LAYER_USAGE);
        }
        self.queue.write_texture(
            self.brush_alpha.copy_info(),
            pixels,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(size), rows_per_image: Some(size) },
            self.brush_alpha.extent(),
        );
    }

    /// Writes the stroke mask into the active stroke's target.
    fn encode_apply(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        paint: &BrushPaint,
        profile: Option<&GpuStrokeProfileResources>,
    ) {
        let Some(active) = &self.active else { return };
        let (target, channels, layer_id) = (active.target, active.channels.clone(), active.layer_id);
        let mut ch = [0.0f32; CHANNELS];
        for &c in &channels {
            ch[c] = 1.0;
        }
        let m = &paint.mat;
        let au = ApplyUniform {
            color: [m.color[0], m.color[1], m.color[2], m.opacity],
            rmh: [m.rough, m.metal, m.height, paint.mask_value],
            ch: [ch[0], ch[1], ch[2], ch[3]],
            p: [paint.opacity, if paint.erase { 1.0 } else { 0.0 }, ch[OPACITY], 0.0],
            // Bake fixes paint exact values; grain would only get in the way.
            g: if matches!(target, StrokeTarget::BakeFix(_)) { [0.0; 4] } else { paint.grain },
            e: [m.emission[0], m.emission[1], m.emission[2], ch[EMISSION]],
            s: [m.specular, ch[SPECULAR], ch[NORMAL], 0.0],
        };
        self.queue.write_buffer(&self.apply_ubuf, 0, bytemuck::bytes_of(&au));

        let mask_bits: u8 = channels.iter().map(|&c| 1u8 << c).sum();
        let paint_pipeline = (target != StrokeTarget::Mask).then(|| self.apply_pipeline(mask_bits));

        let device = &self.device;
        let snap_view = |c: usize| match target {
            StrokeTarget::Mask => {
                if c == 0 {
                    &self.snap_mask.as_ref().unwrap().view
                } else {
                    &self.dummy.view
                }
            }
            _ => self.snap[c].as_ref().filter(|_| ch[c] > 0.5).map_or(&self.dummy.view, |t| &t.view),
        };
        let apply_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apply"),
            layout: &self.p.apply_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.apply_ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.stroke.view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(snap_view(0)) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(snap_view(1)) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(snap_view(2)) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(snap_view(3)) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&self.pos.view) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&self.nrm.view) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&self.brush_grain.view) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::Sampler(&self.repeat_sampler) },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(snap_view(OPACITY)) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(snap_view(EMISSION)) },
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(snap_view(NORMAL)) },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(snap_view(SPECULAR)) },
            ],
        });

        {
            let textures = self.target_textures(layer_id, target, &channels).expect("stroke target exists");
            let (pipeline, attachments) = match target {
                StrokeTarget::Mask => {
                    (&self.p.apply_mask, vec![gpu::color_attachment(&textures[0].view, wgpu::LoadOp::Load)])
                }
                _ => {
                    let mut att = vec![None; CHANNELS];
                    for (tex, &c) in textures.iter().zip(&channels) {
                        att[c] = gpu::color_attachment(&tex.view, wgpu::LoadOp::Load);
                    }
                    (paint_pipeline.as_ref().expect("made above for paint targets"), att)
                }
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("apply"),
                color_attachments: &attachments,
                depth_stencil_attachment: None,
                timestamp_writes: profile.map(|p| wgpu::RenderPassTimestampWrites {
                    query_set: &p.query_set,
                    beginning_of_pass_write_index: Some(2),
                    end_of_pass_write_index: Some(3),
                }),
                occlusion_query_set: None,
                multiview_mask: None,
            }).forget_lifetime();
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &apply_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        if matches!(target, StrokeTarget::BakeFix(_)) {
            self.maps_dirty = true;
            self.mark_dirty();
        } else {
            // Only this layer changed: the next composite can start at its pass.
            if self.stroke_layer.is_some_and(|id| id != layer_id) {
                self.full_dirty = true;
            }
            self.stroke_layer = Some(layer_id);
            self.dirty = true;
        }
    }

    /// Grows the stroke region by a UV rectangle, plus the padding the
    /// position map is dilated into (stamps paint it too) and a texel for
    /// filtering.
    fn add_region(&mut self, region: Option<[f32; 4]>) {
        let Some([u0, v0, u1, v1]) = region else {
            self.region_unknown = true;
            return;
        };
        let res = self.res as f32;
        let margin = (DILATE_PASSES + 2) as f32;
        let lo = |v: f32| (v * res - margin).floor().clamp(0.0, res) as u32;
        let hi = |v: f32| (v * res + margin).ceil().clamp(0.0, res) as u32;
        let r = [lo(u0), lo(v0), hi(u1), hi(v1)];
        if r[2] <= r[0] || r[3] <= r[1] {
            return;
        }
        self.stroke_region = Some(match self.stroke_region {
            Some(o) => [o[0].min(r[0]), o[1].min(r[1]), o[2].max(r[2]), o[3].max(r[3])],
            None => r,
        });
    }

    /// The layer's mask has filters that are on.
    fn has_mask_filters(&self, li: usize) -> bool {
        let l = &self.layers[li];
        l.mask.as_ref().is_some_and(|m| m.filters.iter().any(|f| f.enabled) || m.layers.iter().any(|ml| ml.enabled))
    }

    /// Renders a layer's mask (painting times smart mask) and runs its
    /// filter steps over it; returns the texture holding the result.
    fn encode_mask_filters(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        layer_id: u64,
        source_bg: &wgpu::BindGroup,
        source_offset: u32,
        ops: &[MaskOp],
        scissor: Option<[u32; 4]>,
    ) -> &Tex {
        let pair = &self.mask_out[&layer_id];
        let load = if scissor.is_some() { wgpu::LoadOp::Load } else { wgpu::LoadOp::Clear(wgpu::Color::BLACK) };
        let clip = |pass: &mut wgpu::RenderPass| {
            if let Some([x0, y0, x1, y1]) = scissor {
                pass.set_scissor_rect(x0, y0, x1 - x0, y1 - y0);
            }
        };
        {
            let mut pass = gpu::pass(encoder, "mask source", &[gpu::color_attachment(&pair[0].view, load)]);
            clip(&mut pass);
            pass.set_pipeline(&self.p.mask_source);
            pass.set_bind_group(0, source_bg, &[source_offset]);
            pass.draw(0..3, 0..1);
        }
        let filter = |encoder: &mut wgpu::CommandEncoder, step: usize, input: &Tex, other: &Tex, out: &Tex| {
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mask filter"),
                layout: &self.p.mask_filter_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.mask_filter_ubuf,
                            offset: 0,
                            size: NonZeroU64::new(32),
                        }),
                    },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&input.view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.repeat_sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&other.view) },
                ],
            });
            let mut pass = gpu::pass(encoder, "mask filter", &[gpu::color_attachment(&out.view, load)]);
            clip(&mut pass);
            pass.set_pipeline(&self.p.mask_filter);
            pass.set_bind_group(0, &bg, &[(step.min(MAX_MASK_FILTER_STEPS as usize - 1) as u64 * MASK_FILTER_STRIDE) as u32]);
            pass.draw(0..3, 0..1);
        };
        let mut cur = 0;
        for op in ops {
            match *op {
                MaskOp::Filter(step) => filter(encoder, step, &pair[cur], &pair[2], &pair[1 - cur]),
                MaskOp::Layer { uniform, combine } => {
                    // The mask layer's own values, then combined over the mask so far.
                    {
                        let mut pass = gpu::pass(encoder, "mask layer", &[gpu::color_attachment(&pair[2].view, load)]);
                        clip(&mut pass);
                        pass.set_pipeline(&self.p.mask_source);
                        pass.set_bind_group(0, source_bg, &[uniform]);
                        pass.draw(0..3, 0..1);
                    }
                    filter(encoder, combine, &pair[cur], &pair[2], &pair[1 - cur]);
                }
            }
            cur = 1 - cur;
        }
        &pair[cur]
    }

    /// Flags the layer stack for a full composite.
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
        self.full_dirty = true;
    }

    /// Finishes the stroke and returns its undo record.
    pub fn end_stroke(&mut self) -> Option<UndoEntry> {
        self.stroke_layer = None;
        self.prefix = None;
        self.prefix_cache = None;
        self.stroke_region = None;
        self.region_unknown = false;
        let Some(active) = self.active.take() else {
            self.invalidate_thumbs();
            return None;
        };
        // A stroke on a mask changes only that mask's thumbnail; a fill's
        // content thumbnail (a full composite) stays as it was.
        match active.target {
            StrokeTarget::Mask => self.invalidate_mask_thumb(active.layer_id),
            _ => self.invalidate_thumbs(),
        }
        let textures = self.target_textures(active.layer_id, active.target, &active.channels)?;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let after = Snapshot::encode(&self.device, &mut encoder, &textures);
        self.queue.submit([encoder.finish()]);
        after.request_map();
        // Refresh invalidated mask/fill thumbnails once after painting, not
        // once per composite frame while the stroke is active.
        self.update_thumbs();
        Some(UndoEntry {
            layer_id: active.layer_id,
            target: active.target,
            channels: active.channels,
            before: active.before,
            after,
        })
    }

    pub fn undo(&mut self, stack: &mut UndoStack) {
        if self.active.is_some() {
            return;
        }
        self.commit_props(stack);
        if let Some(mut e) = stack.undo.pop() {
            self.apply_edit(&mut e.edit, true);
            stack.redo.push(e);
            self.after_history_step();
        }
    }

    pub fn redo(&mut self, stack: &mut UndoStack) {
        if self.active.is_some() {
            return;
        }
        if let Some(mut e) = stack.redo.pop() {
            self.apply_edit(&mut e.edit, false);
            stack.undo.push(e);
            self.after_history_step();
        }
    }

    fn after_history_step(&mut self) {
        self.invalidate_thumbs();
        self.committed = self.doc_props();
        self.maps_dirty = true;
        self.mark_dirty();
    }

    // ------------------------------------------------------------------ compositing

    fn layer_uniform(&self, layer: &Layer) -> LayerUniform {
        let mut u: LayerUniform = bytemuck::Zeroable::zeroed();
        let flags = |b: [bool; CHANNELS]| b.map(|v| if v { 1.0 } else { 0.0 });
        let (kind, ch) = match &layer.content {
            Content::Paint(ch) => (0.0, flags(std::array::from_fn(|i| ch[i].is_some()))),
            Content::Fill(f) => {
                let m = &f.mat;
                u.fill_color = [m.color[0], m.color[1], m.color[2], m.opacity];
                u.fill_rmh = [m.rough, m.metal, m.height, 0.0];
                u.var0 = [f.color_var, f.var_scale, f.rough_var, f.height_var];
                let mut bits = (0..4).filter(|&i| layer.images[i].is_some()).fold(0u32, |b, i| b | 1 << i);
                // A grayscale color image is single-channel: the shader spreads it to rgb.
                if layer.images[0].as_ref().is_some_and(|im| im.tex.format == F::R8Unorm) {
                    bits |= 16;
                }
                if layer.images[OPACITY].is_some() {
                    bits |= 32;
                }
                if layer.images[EMISSION].is_some() {
                    bits |= 64;
                }
                if layer.images[NORMAL].is_some() {
                    bits |= 128;
                }
                if layer.images[SPECULAR].is_some() {
                    bits |= 256;
                }
                u.emis = [m.emission[0], m.emission[1], m.emission[2], 0.0];
                u.ns = [m.normal, 0.0, m.specular, 0.0];
                u.var1 = [f.seed, f.part_var, f.projection as u32 as f32, bits as f32];
                u.img = [f.tiling, f.rotation, f.offset[0], f.offset[1]];
                u.gx[2] = f.tri_sharpness;
                (1.0, flags(m.enabled))
            }
            Content::Folder(_) => (2.0, [0.0; CHANNELS]),
        };
        u.ch = [ch[0], ch[1], ch[2], ch[3]];
        // Opacity's flag rides in the w of the fill values, emission's in its own.
        u.fill_rmh[3] = ch[OPACITY];
        u.emis[3] = ch[EMISSION];
        u.ns[1] = ch[NORMAL];
        u.ns[3] = ch[SPECULAR];
        let mut has_paint_mask = 0.0;
        if let Some(mask) = &layer.mask {
            if mask.paint.is_some() {
                has_paint_mask = 1.0;
            }
            if let Some(g) = &mask.generator {
                for (i, w) in g.weights.iter().enumerate() {
                    u.gw[i / 4][i % 4] = *w;
                }
                (u.gc, u.gcw) = self.custom_refs(g);
                u.gp0 = [g.level, g.contrast, g.scale, g.noise];
                u.gp1 = [if g.invert { 1.0 } else { 0.0 }, g.seed, 1.0, g.edge_width];
                (u.gid, u.gids) = id_uniform(g);
            }
        }
        u.p0 = [layer.opacity, layer.blend as u32 as f32, kind, has_paint_mask];
        let (lo, hi) = self.bounds;
        u.bmin = [lo.x, lo.y, lo.z, 0.0];
        u.bsize = [hi.x - lo.x, hi.y - lo.y, hi.z - lo.z, 0.0];
        u.gx = [self.edge_max, 0.0, 0.0, 0.0];
        u
    }

    /// Shown: the layer and every folder around it are visible.
    fn shown(&self, idx: usize) -> bool {
        self.layers[idx].visible && self.ancestors(idx).all(|a| self.layers[a].visible)
    }

    /// A folder that fades or masks its contents, so it composites them apart.
    fn isolates(&self, idx: usize) -> bool {
        let l = &self.layers[idx];
        l.is_folder() && (l.opacity < 1.0 || l.mask.is_some())
    }

    /// Re-composites the whole stack. Cheap enough to run every painting frame:
    /// one fullscreen pass per visible layer, plus one per faded or masked folder.
    /// Waits until the GPU has finished everything submitted so far.
    pub fn wait_idle(&self) {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    pub fn composite(&mut self) {
        if self.maps_dirty {
            self.resolve_maps();
        }
        let n = self.layers.len();
        let mut passes = Vec::new();
        let (mut depth, mut max_depth) = (0, 0);
        for i in 0..n {
            // Folders whose contents start here open, outermost (highest index) first.
            let mut opening: Vec<usize> =
                (i..n).filter(|&f| self.isolates(f) && self.shown(f) && self.block(f).start == i).collect();
            opening.sort_by(|a, b| b.cmp(a));
            for _ in opening {
                passes.push(Pass::Begin);
                depth += 1;
                max_depth = max_depth.max(depth);
            }
            let l = &self.layers[i];
            let has_content = match &l.content {
                Content::Paint(ch) => ch.iter().any(|c| c.is_some()),
                Content::Fill(f) => f.mat.enabled.iter().any(|&e| e),
                Content::Folder(_) => false,
            };
            if has_content && l.opacity > 0.0 && self.shown(i) {
                passes.push(Pass::Layer(i));
            }
            if self.isolates(i) && self.shown(i) {
                passes.push(Pass::End(i));
                depth -= 1;
            }
        }
        self.folder_below.retain(|set| set[0].width == self.res);
        self.folder_below.truncate(max_depth);
        while self.folder_below.len() < max_depth {
            let set = ACCUM_FORMATS.map(|f| Tex::new(&self.device, "folder below", self.res, self.res, f, LAYER_USAGE));
            self.folder_below.push(set);
        }

        if passes.len() as u64 > self.layer_ubuf_slots {
            self.layer_ubuf_slots = (passes.len() as u64).next_power_of_two();
            self.layer_ubuf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("layer uniforms"),
                size: UNIFORM_STRIDE * self.layer_ubuf_slots,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        // Layers with mask filters: their source pass reads the mask as usual
        // (an extra uniform slot after the passes'), their layer pass reads
        // the filtered texture instead (no smart mask on top).
        let mut source_slot: HashMap<usize, usize> = HashMap::new();
        for pass in &passes {
            if let Pass::Layer(li) | Pass::End(li) = *pass
                && self.has_mask_filters(li) {
                    source_slot.insert(li, passes.len() + source_slot.len());
                }
        }
        // Each enabled mask layer renders through its own uniform slot too: the
        // layer's uniform with that mask layer's generator (or none, for a fill).
        let mut item_slots: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut next_slot = passes.len() + source_slot.len();
        for &li in source_slot.keys() {
            let n = self.layers[li].mask.as_ref().map_or(0, |m| m.layers.iter().filter(|ml| ml.enabled).count());
            item_slots.insert(li, (next_slot..next_slot + n).collect());
            next_slot += n;
        }
        let slots = next_slot;
        if slots as u64 > self.layer_ubuf_slots {
            self.layer_ubuf_slots = (slots as u64).next_power_of_two();
            self.layer_ubuf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("layer uniforms"),
                size: UNIFORM_STRIDE * self.layer_ubuf_slots,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        let mut data = vec![0u8; (UNIFORM_STRIDE as usize) * slots.max(1)];
        for (k, pass) in passes.iter().enumerate() {
            let (Pass::Layer(li) | Pass::End(li)) = *pass else { continue };
            let mut u = self.layer_uniform(&self.layers[li]);
            if let Some(&slot) = source_slot.get(&li) {
                let off = slot * UNIFORM_STRIDE as usize;
                data[off..off + size_of::<LayerUniform>()].copy_from_slice(bytemuck::bytes_of(&u));
                u.p0[3] = 1.0;
                u.gp1[2] = 0.0;
            }
            let off = k * UNIFORM_STRIDE as usize;
            data[off..off + size_of::<LayerUniform>()].copy_from_slice(bytemuck::bytes_of(&u));
            if let Some(items) = item_slots.get(&li) {
                let mask = self.layers[li].mask.as_ref().expect("filtered layers have masks");
                for (ml, &slot) in mask.layers.iter().filter(|ml| ml.enabled).zip(items) {
                    let mut iu = u;
                    iu.p0[3] = 0.0;
                    match ml.source {
                        MaskSource::Generator(g) => {
                            for (i, w) in g.weights.iter().enumerate() {
                                iu.gw[i / 4][i % 4] = *w;
                            }
                            (iu.gc, iu.gcw) = self.custom_refs(&g);
                            iu.gp0 = [g.level, g.contrast, g.scale, g.noise];
                            iu.gp1 = [if g.invert { 1.0 } else { 0.0 }, g.seed, 1.0, g.edge_width];
                            (iu.gid, iu.gids) = id_uniform(&g);
                        }
                        MaskSource::Fill(_) => iu.gp1[2] = 0.0,
                    }
                    let off = slot * UNIFORM_STRIDE as usize;
                    data[off..off + size_of::<LayerUniform>()].copy_from_slice(bytemuck::bytes_of(&iu));
                }
            }
        }
        self.queue.write_buffer(&self.layer_ubuf, 0, &data);
        // Textures for the filtered masks; forget layers that lost theirs.
        let filtered_ids: Vec<u64> = source_slot.keys().map(|&li| self.layers[li].id).collect();
        self.mask_out.retain(|id, t| filtered_ids.contains(id) && t[0].width == self.res);
        for id in &filtered_ids {
            if !self.mask_out.contains_key(id) {
                let t = || Tex::new(&self.device, "filtered mask", self.res, self.res, MASK_FILTER_FORMAT, LAYER_USAGE);
                self.mask_out.insert(*id, [t(), t(), t()]);
            }
        }
        // Filter steps' uniforms, written up front: (kind, a, b, c), (dir x, dir y, scale).
        let mut steps: Vec<[f32; 8]> = Vec::new();
        let mut layer_ops: HashMap<usize, Vec<MaskOp>> = HashMap::new();
        for &li in source_slot.keys() {
            let mut ops = Vec::new();
            let scale = self.res as f32 / FILTER_REF_RES;
            let mask = self.layers[li].mask.as_ref().expect("filtered layers have masks");
            for (ml, &slot) in mask.layers.iter().filter(|ml| ml.enabled).zip(&item_slots[&li]) {
                let value = match ml.source {
                    MaskSource::Fill(v) => v,
                    MaskSource::Generator(_) => 1.0,
                };
                ops.push(MaskOp::Layer { uniform: (slot as u64 * UNIFORM_STRIDE) as u32, combine: steps.len() });
                steps.push([COMBINE_STEP, ml.blend as u32 as f32, ml.opacity, value, 0.0, 0.0, scale, 0.0]);
            }
            let begin = steps.len();
            for f in self.layers[li].mask.as_ref().map_or(&[][..], |m| &m.filters[..]).iter().filter(|f| f.enabled) {
                let kind = f.kind as u32 as f32;
                if f.kind == FilterKind::Blur {
                    steps.push([kind, f.a, f.b, f.c, 1.0, 0.0, scale, 0.0]);
                    steps.push([kind, f.a, f.b, f.c, 0.0, 1.0, scale, 0.0]);
                } else {
                    steps.push([kind, f.a, f.b, f.c, 0.0, 0.0, scale, 0.0]);
                }
            }
            ops.extend((begin..steps.len()).map(MaskOp::Filter));
            layer_ops.insert(li, ops);
        }
        steps.truncate(MAX_MASK_FILTER_STEPS as usize);
        let mut sdata = vec![0u8; (MASK_FILTER_STRIDE as usize) * steps.len().max(1)];
        for (i, s) in steps.iter().enumerate() {
            let off = i * MASK_FILTER_STRIDE as usize;
            sdata[off..off + 32].copy_from_slice(bytemuck::cast_slice(s));
        }
        self.queue.write_buffer(&self.mask_filter_ubuf, 0, &sdata);

        // While painting, everything below the painted layer is unchanged:
        // start at its pass from the kept prefix instead of the bottom.
        let resume = self
            .prefix
            .take()
            .filter(|p| !self.full_dirty && self.stroke_layer == Some(p.layer_id) && p.passes == passes);
        // A full composite while painting records the prefix for the next frames.
        let record = self.stroke_layer.filter(|_| resume.is_none()).and_then(|id| {
            passes.iter().position(|p| matches!(*p, Pass::Layer(li) | Pass::End(li) if self.layers[li].id == id))
        });
        if record.is_some_and(|k| k + 1 < passes.len()) && self.prefix_cache.as_ref().is_none_or(|c| c[0].width != self.res) {
            self.prefix_cache = Some(ACCUM_FORMATS.map(|f| Tex::new(&self.device, "composite prefix", self.res, self.res, f, LAYER_USAGE)));
        }

        let mut encoder = self.device.create_command_encoder(&Default::default());
        if self.gpu_profiler.as_ref().is_some_and(|p| p.encoder_timestamps)
            && let Some(p) = &self.active_gpu_profile {
                encoder.write_timestamp(&p.query_set, 6);
            }
        if resume.is_none() {
            let c = |v: f64| wgpu::LoadOp::Clear(wgpu::Color { r: v, g: v, b: v, a: 1.0 });
            let a = &self.accum[0];
            let _pass = gpu::pass(
                &mut encoder,
                "composite clear",
                &[
                    gpu::color_attachment(&a[0].view, c(0.75)),
                    gpu::color_attachment(&a[1].view, c(0.5)),
                    gpu::color_attachment(&a[2].view, c(0.0)),
                    gpu::color_attachment(&a[3].view, c(0.0)),
                    gpu::color_attachment(&a[4].view, c(0.0)),
                    // A flat normal.
                    gpu::color_attachment(&a[5].view, wgpu::LoadOp::Clear(wgpu::Color { r: 0.5, g: 0.5, b: 1.0, a: 1.0 })),
                    gpu::color_attachment(&a[6].view, c(0.5)),
                ],
            );
        }
        // Resumed frames only redo the patch the stroke touched. Each pass
        // there reads the texel it writes, so inside the patch the chain is
        // exact; outside it the final set keeps the last composite's result,
        // which the stroke didn't change. (Folder stacks saved after the
        // painted layer are only read inside the patch too.)
        let region = std::mem::take(&mut self.stroke_region);
        let unknown = std::mem::take(&mut self.region_unknown);
        let mut scissor = if resume.is_some() && !unknown { Some(region.unwrap_or([0, 0, 0, 0])) } else { None };
        // Filtered masks above the change read around it: widen the patch by
        // the farthest any of their filters reach.
        if let (Some(r), Some(at)) = (scissor.as_mut(), resume.as_ref().map(|p| p.at)) {
            let reach = passes[at..]
                .iter()
                .filter_map(|p| match *p {
                    Pass::Layer(li) => self.layers[li].mask.as_ref(),
                    _ => None,
                })
                .map(|m| m.filters.iter().filter(|f| f.enabled).map(|f| f.reach(self.res)).sum::<f32>())
                .fold(0.0f32, f32::max);
            if reach > 0.0 && r[2] > r[0] && r[3] > r[1] {
                let e = reach.ceil() as u32 + 2;
                *r = [r[0].saturating_sub(e), r[1].saturating_sub(e), (r[2] + e).min(self.res), (r[3] + e).min(self.res)];
            }
        }
        if scissor.is_some_and(|r| r[2] <= r[0] || r[3] <= r[1]) {
            // Nothing visible changed.
            let resume = resume.expect("scissor implies resume");
            self.prefix = Some(resume);
            self.dirty = false;
            self.queue.submit([encoder.finish()]);
            self.active_gpu_profile = None;
            return;
        }
        if scissor.is_none() {
            self.full_recomps = self.full_recomps.wrapping_add(1);
        }
        // `src` is the accum set holding the stack so far; `from_cache` reads
        // the first pass's input from the prefix cache instead.
        let (start, mut src, mut depth, mut from_cache) = match &resume {
            Some(p) => (p.at, p.input.map_or(1, |i| i), p.depth, p.input.is_none()),
            None => (0, 0, 0, false),
        };
        let mut finals: Vec<(u64, usize)> = Vec::new();
        let profile_layer = self.stroke_layer;
        let mut composite_profiled = false;
        for (k, pass) in passes.iter().enumerate().skip(start) {
            if record == Some(k) {
                let input = if k + 1 < passes.len() {
                    let cache = self.prefix_cache.as_ref().expect("allocated above");
                    for (from, to) in self.accum[src].iter().zip(cache) {
                        encoder.copy_texture_to_texture(from.copy_info(), to.copy_info(), from.extent());
                    }
                    None
                } else {
                    Some(src)
                };
                let layer_id = self.stroke_layer.expect("recording for a stroke");
                self.prefix = Some(Prefix { layer_id, passes: passes.clone(), at: k, depth, input });
            }
            let li = match *pass {
                Pass::Begin => {
                    // Keep the stack below the folder to fade its result over.
                    let [x0, y0, x1, y1] = scissor.unwrap_or([0, 0, self.res, self.res]);
                    fn at(t: &Tex, x: u32, y: u32) -> wgpu::TexelCopyTextureInfo<'_> {
                        wgpu::TexelCopyTextureInfo { origin: wgpu::Origin3d { x, y, z: 0 }, ..t.copy_info() }
                    }
                    let size = wgpu::Extent3d { width: x1 - x0, height: y1 - y0, depth_or_array_layers: 1 };
                    for (from, to) in self.accum[src].iter().zip(&self.folder_below[depth]) {
                        encoder.copy_texture_to_texture(at(from, x0, y0), at(to, x0, y0), size);
                    }
                    depth += 1;
                    continue;
                }
                Pass::Layer(li) => li,
                Pass::End(li) => {
                    depth -= 1;
                    li
                }
            };
            let dst = 1 - src;
            let layer = &self.layers[li];
            let dummy = &self.dummy.view;
            fn or_dummy<'a>(t: Option<&'a Tex>, dummy: &'a wgpu::TextureView) -> wgpu::BindingResource<'a> {
                wgpu::BindingResource::TextureView(t.map_or(dummy, |t| &t.view))
            }
            let view = |t| or_dummy(t, dummy);
            let chans: [Option<&Tex>; CHANNELS] = match &layer.content {
                Content::Paint(ch) => std::array::from_fn(|i| ch[i].as_ref()),
                // A folder merge reads the stack below it in the layer slots
                // (opacity is in its color's alpha).
                Content::Folder(_) => {
                    let below = &self.folder_below[depth];
                    let mut c: [Option<&Tex>; CHANNELS] = std::array::from_fn(|i| below.get(i).filter(|_| i < 4));
                    c[EMISSION] = below.get(FINAL_EMISSION);
                    c[NORMAL] = below.get(FINAL_NORMAL);
                    c[SPECULAR] = below.get(FINAL_SPECULAR);
                    c
                }
                Content::Fill(_) => std::array::from_fn(|i| layer.images[i].as_ref().map(|im| im.tex.as_ref())),
            };
            let mask = layer.mask.as_ref().and_then(|m| m.paint.as_ref());
            let prev = if from_cache { self.prefix_cache.as_ref().expect("prefix cache") } else { &self.accum[src] };
            let make_bg = |mask: Option<&Tex>| self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("composite"),
                layout: &self.p.composite_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.layer_ubuf,
                            offset: 0,
                            size: NonZeroU64::new(size_of::<LayerUniform>() as u64),
                        }),
                    },
                    wgpu::BindGroupEntry { binding: 1, resource: view(Some(&prev[0])) },
                    wgpu::BindGroupEntry { binding: 2, resource: view(Some(&prev[1])) },
                    wgpu::BindGroupEntry { binding: 3, resource: view(Some(&prev[2])) },
                    wgpu::BindGroupEntry { binding: 4, resource: view(Some(&prev[3])) },
                    wgpu::BindGroupEntry { binding: 5, resource: view(chans[0]) },
                    wgpu::BindGroupEntry { binding: 6, resource: view(chans[1]) },
                    wgpu::BindGroupEntry { binding: 7, resource: view(chans[2]) },
                    wgpu::BindGroupEntry { binding: 8, resource: view(chans[3]) },
                    wgpu::BindGroupEntry { binding: 9, resource: or_dummy(mask, dummy) },
                    wgpu::BindGroupEntry { binding: 10, resource: view(Some(&self.pos)) },
                    wgpu::BindGroupEntry { binding: 11, resource: view(Some(&self.nrm)) },
                    wgpu::BindGroupEntry { binding: 12, resource: view(Some(self.map_tex(0))) },
                    wgpu::BindGroupEntry { binding: 13, resource: view(Some(self.map_tex(1))) },
                    wgpu::BindGroupEntry { binding: 14, resource: view(Some(self.map_tex(2))) },
                    // Composite never reads map D; its slot holds the custom maps.
                    wgpu::BindGroupEntry { binding: 15, resource: wgpu::BindingResource::TextureView(&self.custom_array.1) },
                    wgpu::BindGroupEntry { binding: 16, resource: view(Some(self.map_tex(4))) },
                    wgpu::BindGroupEntry { binding: 17, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry { binding: 18, resource: wgpu::BindingResource::Sampler(&self.repeat_sampler) },
                    wgpu::BindGroupEntry { binding: 19, resource: view(chans[OPACITY]) },
                    wgpu::BindGroupEntry { binding: 20, resource: view(Some(&prev[FINAL_EMISSION])) },
                    wgpu::BindGroupEntry { binding: 21, resource: view(chans[EMISSION]) },
                    wgpu::BindGroupEntry { binding: 22, resource: view(Some(&prev[FINAL_NORMAL])) },
                    wgpu::BindGroupEntry { binding: 23, resource: view(chans[NORMAL]) },
                    wgpu::BindGroupEntry { binding: 24, resource: view(Some(&prev[FINAL_SPECULAR])) },
                    wgpu::BindGroupEntry { binding: 25, resource: view(chans[SPECULAR]) },
                    wgpu::BindGroupEntry { binding: 26, resource: view(Some(&self.tan)) },
                ],
            });
            // Mask filters: render the mask, run its filters, then read the result.
            let filtered = match (*pass, source_slot.get(&li)) {
                (Pass::Layer(_) | Pass::End(_), Some(&slot)) => Some(self.encode_mask_filters(
                    &mut encoder,
                    layer.id,
                    &make_bg(mask),
                    (slot as u64 * UNIFORM_STRIDE) as u32,
                    layer_ops.get(&li).map_or(&[][..], |o| &o[..]),
                    scissor,
                )),
                _ => None,
            };
            if let Some(t) = filtered
                && let Some(i) = self.mask_out[&layer.id].iter().position(|x| std::ptr::eq(x, t)) {
                    finals.push((layer.id, i));
                }
            let bg = make_bg(filtered.or(mask));
            let out = &self.accum[dst];
            // A patch keeps what is around it.
            let load = if scissor.is_some() { wgpu::LoadOp::Load } else { wgpu::LoadOp::Clear(wgpu::Color::BLACK) };
            let attachments = [
                gpu::color_attachment(&out[0].view, load),
                gpu::color_attachment(&out[1].view, load),
                gpu::color_attachment(&out[2].view, load),
                gpu::color_attachment(&out[3].view, load),
                gpu::color_attachment(&out[4].view, load),
                gpu::color_attachment(&out[5].view, load),
                gpu::color_attachment(&out[6].view, load),
            ];
            let profile = self.active_gpu_profile.as_ref().filter(|_| profile_layer == Some(layer.id));
            composite_profiled |= profile.is_some();
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("composite layer"),
                color_attachments: &attachments,
                depth_stencil_attachment: None,
                timestamp_writes: profile.map(|p| wgpu::RenderPassTimestampWrites {
                    query_set: &p.query_set,
                    beginning_of_pass_write_index: Some(4),
                    end_of_pass_write_index: Some(5),
                }),
                occlusion_query_set: None,
                multiview_mask: None,
            }).forget_lifetime();
            if let Some([x0, y0, x1, y1]) = scissor {
                pass.set_scissor_rect(x0, y0, x1 - x0, y1 - y0);
            }
            pass.set_pipeline(if layer.is_folder() { &self.p.folder_merge } else { &self.p.composite });
            pass.set_bind_group(0, &bg, &[(k as u64 * UNIFORM_STRIDE) as u32]);
            pass.draw(0..3, 0..1);
            drop(pass);
            src = dst;
            from_cache = false;
        }
        if composite_profiled {
            if let Some(p) = &self.active_gpu_profile {
                if self.gpu_profiler.as_ref().is_some_and(|profiler| profiler.encoder_timestamps) {
                    encoder.write_timestamp(&p.query_set, 7);
                }
                encoder.resolve_query_set(&p.query_set, 0..8, &p.resolve, 0);
                encoder.copy_buffer_to_buffer(&p.resolve, 0, &p.readback, 0, 64);
            }
        }
        self.queue.submit([encoder.finish()]);
        if composite_profiled {
            self.active_gpu_profile = None;
            if let Some(profiler) = &mut self.gpu_profiler {
                profiler.request_map();
            }
        } else {
            self.active_gpu_profile = None;
        }
        if let Some(p) = resume {
            self.prefix = Some(p);
        }
        self.final_idx = src;
        self.generation += 1;
        self.dirty = false;
        self.full_dirty = false;
        self.mask_final.retain(|id, _| self.mask_out.contains_key(id));
        self.mask_final.extend(finals);
        self.update_thumbs();
    }

    /// Whether any visible layer sets opacity: only then can the result be
    /// see-through (the 3D view skips its see-through pass otherwise).
    pub fn uses_opacity(&self) -> bool {
        self.layers.iter().any(|l| {
            l.visible
                && match &l.content {
                    Content::Paint(ch) => ch[OPACITY].is_some(),
                    Content::Fill(f) => f.mat.enabled[OPACITY] || l.images[OPACITY].is_some(),
                    Content::Folder(_) => false,
                }
        })
    }

    /// Whether any visible layer sets emission (exports skip the map otherwise).
    pub fn uses_emission(&self) -> bool {
        self.layers.iter().any(|l| {
            l.visible
                && match &l.content {
                    Content::Paint(ch) => ch[EMISSION].is_some(),
                    Content::Fill(f) => f.mat.enabled[EMISSION] || l.images[EMISSION].is_some(),
                    Content::Folder(_) => false,
                }
        })
    }

    /// Whether any visible layer sets normals or specular (exports skip
    /// the maps otherwise: flat normal, 0.5 specular).
    pub fn uses_channel(&self, ch: usize) -> bool {
        self.layers.iter().any(|l| {
            l.visible
                && match &l.content {
                    Content::Paint(c) => c[ch].is_some(),
                    Content::Fill(f) => f.mat.enabled[ch] || l.images[ch].is_some(),
                    Content::Folder(_) => false,
                }
        })
    }

    /// Final composited channels: base color (sRGB-encoded, with opacity in
    /// alpha), roughness, metallic, height, emission (sRGB-encoded), normal
    /// (tangent space, encoded 0..1), specular level.
    /// (A [`Pass`] list drives [`Engine::composite`].)
    pub fn final_tex(&self, ch: usize) -> &Tex {
        &self.accum[self.final_idx][ch]
    }

    // ------------------------------------------------------------------ resolution

    /// Draws `src` over all of `dst` (any filterable float format into one
    /// the resample pipelines target).
    fn resample_into(&self, encoder: &mut wgpu::CommandEncoder, src: &Tex, dst: &Tex) {
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.p.resample_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src.view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        let mut pass = gpu::pass(encoder, "resample", &[gpu::color_attachment(&dst.view, wgpu::LoadOp::Load)]);
        pass.set_pipeline(&self.p.resample[&dst.format]);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Draws `src` over all of `dst`, a view of a texture of format `format`
    /// (any of the layer, final and mesh-map formats); for copies at another size.
    pub fn resample_to_view(&self, encoder: &mut wgpu::CommandEncoder, src: &Tex, dst: &wgpu::TextureView, format: F) {
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.p.resample_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src.view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        let mut pass = gpu::pass(encoder, "resample", &[gpu::color_attachment(dst, wgpu::LoadOp::Load)]);
        pass.set_pipeline(&self.p.resample[&format]);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }

    fn resample(&self, encoder: &mut wgpu::CommandEncoder, src: &Tex, res: u32) -> Tex {
        let dst = Tex::new(&self.device, "resampled", res, res, src.format, LAYER_USAGE);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.p.resample_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src.view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        let mut pass = gpu::pass(encoder, "resample", &[gpu::color_attachment(&dst.view, wgpu::LoadOp::Load)]);
        pass.set_pipeline(&self.p.resample[&src.format]);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
        drop(pass);
        dst
    }

    /// Changes the texture set resolution, resampling every painted texture.
    pub fn set_resolution(&mut self, res: u32, mesh: &GpuMesh, undo: &mut UndoStack) {
        if res == self.res || self.active.is_some() {
            return;
        }
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let mut new_layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let content = match &layer.content {
                Content::Paint(ch) => Some(std::array::from_fn(|i| ch[i].as_ref().map(|t| self.resample(&mut encoder, t, res)))),
                Content::Fill(_) | Content::Folder(_) => None,
            };
            let mask = layer.mask.as_ref().and_then(|m| m.paint.as_ref()).map(|t| self.resample(&mut encoder, t, res));
            new_layers.push((content, mask));
        }
        if self.has_bake {
            let maps = std::array::from_fn(|t| self.resample(&mut encoder, &self.maps[t], res));
            self.maps = maps;
            self.custom = self.custom.iter().map(|t| self.resample(&mut encoder, t, res)).collect();
            self.build_custom_array(&mut encoder);
        }
        let fixes: Vec<Option<Tex>> =
            self.map_fix.iter().map(|f| f.as_ref().map(|t| self.resample(&mut encoder, t, res))).collect();
        for (dst, f) in self.map_fix.iter_mut().zip(fixes) {
            *dst = f;
        }
        self.resolved = Default::default();
        self.maps_dirty = true;
        self.queue.submit([encoder.finish()]);
        for (layer, (content, mask)) in self.layers.iter_mut().zip(new_layers) {
            if let Some(ch) = content {
                layer.content = Content::Paint(ch);
            }
            if let (Some(m), Some(t)) = (layer.mask.as_mut(), mask) {
                m.paint = Some(t);
            }
            if let (Some(id), Content::Paint(ch)) = (layer.thumb, &layer.content)
                && let Some(color) = &ch[0] {
                    self.rs.renderer.write().update_egui_texture_from_wgpu_texture(
                        &self.device,
                        color.srgb(),
                        wgpu::FilterMode::Linear,
                        id,
                    );
                }
        }
        self.res = res;
        self.uv_version += 1;
        self.pos = Tex::new(&self.device, "position map", res, res, POS_FORMAT, LAYER_USAGE);
        self.nrm = Tex::new(&self.device, "normal map", res, res, NRM_FORMAT, LAYER_USAGE);
        self.tan = Tex::new(&self.device, "tangent map", res, res, TAN_FORMAT, LAYER_USAGE);
        self.accum = [accum_set(&self.device, res, "accum A"), accum_set(&self.device, res, "accum B")];
        self.stroke = Tex::new(&self.device, "stroke mask", res, res, STROKE_FORMAT, LAYER_USAGE);
        self.snap = Default::default();
        self.snap_mask = None;
        undo.clear();
        self.bake_uv(mesh);
    }

    // ------------------------------------------------------------------ stats

    pub fn vram_bytes(&self) -> u64 {
        let fixed = self.pos.bytes()
            + self.nrm.bytes()
            + self.tan.bytes()
            + self.stroke.bytes()
            + self.maps.iter().map(|t| t.bytes()).sum::<u64>()
            + self.resolved.iter().flatten().map(|t| t.bytes()).sum::<u64>()
            + self.map_fix.iter().flatten().map(|t| t.bytes()).sum::<u64>()
            + self.accum.iter().flatten().map(|t| t.bytes()).sum::<u64>()
            + self.snap.iter().flatten().map(|t| t.bytes()).sum::<u64>()
            + self.snap_mask.as_ref().map_or(0, |t| t.bytes());
        fixed + self.layers.iter().map(|l| l.vram_bytes()).sum::<u64>()
    }

    pub fn layer_vram_bytes(&self, idx: usize) -> u64 {
        self.layers.get(idx).map_or(0, |l| l.vram_bytes())
    }
    /// Where the texture memory goes, in MB (offscreen tests).
    #[cfg(test)]
    pub fn vram_breakdown(&self) -> Vec<(&'static str, u64)> {
        let mb = |b: u64| b >> 20;
        let set = |s: &[Tex; ACCUM]| s.iter().map(|t| t.bytes()).sum::<u64>();
        // Layers can share one image texture: count each once.
        let unique_images: u64 = {
            let mut seen = Vec::new();
            let mut sum = 0;
            for i in self.layers.iter().flat_map(|l| l.images.iter().flatten()) {
                let p = Arc::as_ptr(&i.tex);
                if !seen.contains(&p) {
                    seen.push(p);
                    sum += i.tex.bytes();
                }
            }
            sum
        };
        vec![
            ("surface maps", mb(self.pos.bytes() + self.nrm.bytes() + self.tan.bytes() + self.stroke.bytes())),
            ("mesh maps", mb(self.maps.iter().map(|t| t.bytes()).sum::<u64>() + self.resolved.iter().flatten().map(|t| t.bytes()).sum::<u64>())),
            ("accum (2 sets)", mb(self.accum.iter().map(set).sum())),
            ("folder below", mb(self.folder_below.iter().map(set).sum())),
            ("prefix cache", mb(self.prefix_cache.as_ref().map_or(0, set))),
            ("filtered masks", mb(self.mask_out.values().flat_map(|t| t.iter()).map(|t| t.bytes()).sum())),
            ("paint layers", mb(self.layers.iter().map(|l| l.vram_bytes()).sum())),
            ("fill images", mb(unique_images)),
        ]
    }
}
