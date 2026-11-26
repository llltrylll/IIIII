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
use std::collections::{HashMap, HashSet};
use std::num::NonZeroU64;
use std::sync::{mpsc, Arc};
use wgpu::TextureFormat as F;

/// Paint channels: base color, roughness, metallic, height, opacity,
/// emission, normal, specular.
pub const CHANNELS: usize = 8;
/// The height channel (displacement data).
pub const HEIGHT: usize = 3;
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
const INITIAL_MASK_FILTER_STEPS: u64 = 256;
const MASK_FILTER_STRIDE: u64 = 256;

fn mask_filter_buffer_capacity(current: u64, required: usize) -> u64 {
    let required = (required.max(1) as u64).next_power_of_two().max(INITIAL_MASK_FILTER_STEPS);
    if required > current || (current > INITIAL_MASK_FILTER_STEPS && current >= required.saturating_mul(4)) {
        required
    } else {
        current
    }
}

fn layer_uniform_buffer_capacity(current: u64, required: usize) -> u64 {
    const MIN_SLOTS: u64 = 16;
    let required = (required.max(1) as u64).next_power_of_two().max(MIN_SLOTS);
    if required > current || (current > MIN_SLOTS && current >= required.saturating_mul(4)) {
        required
    } else {
        current
    }
}

fn retained_folder_depth(current: usize, required: usize) -> usize {
    if required > current || (current > 0 && current >= required.saturating_mul(4)) {
        required
    } else {
        current
    }
}

#[cfg(test)]
mod working_buffer_capacity_tests {
    use super::{layer_uniform_buffer_capacity, mask_filter_buffer_capacity, retained_folder_depth};

    #[test]
    fn filter_uniform_capacity_grows_without_clamping_steps() {
        assert_eq!(mask_filter_buffer_capacity(256, 0), 256);
        assert_eq!(mask_filter_buffer_capacity(256, 256), 256);
        assert_eq!(mask_filter_buffer_capacity(256, 257), 512);
        assert_eq!(mask_filter_buffer_capacity(512, 513), 1024);
        assert_eq!(mask_filter_buffer_capacity(1024, 300), 1024);
    }

    #[test]
    fn filter_uniform_capacity_shrinks_after_large_stack_reduction() {
        assert_eq!(mask_filter_buffer_capacity(4096, 300), 512);
        assert_eq!(mask_filter_buffer_capacity(4096, 0), 256);
        assert_eq!(mask_filter_buffer_capacity(1024, 300), 1024);
    }

    #[test]
    fn layer_uniform_capacity_has_hysteresis_when_stack_shrinks() {
        assert_eq!(layer_uniform_buffer_capacity(16, 0), 16);
        assert_eq!(layer_uniform_buffer_capacity(16, 17), 32);
        assert_eq!(layer_uniform_buffer_capacity(2048, 300), 512);
        assert_eq!(layer_uniform_buffer_capacity(1024, 300), 1024);
    }

    #[test]
    fn folder_cache_depth_only_shrinks_after_a_fourfold_reduction() {
        assert_eq!(retained_folder_depth(0, 3), 3);
        assert_eq!(retained_folder_depth(8, 3), 8);
        assert_eq!(retained_folder_depth(12, 3), 3);
        assert_eq!(retained_folder_depth(4, 0), 0);
    }
}
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
    map("ID", 4, 0, MapKind::Info, "Color per material, mesh part, or imported vertex color, for ID masks."),
];

/// Packed texture index of the categorical ID map. Unlike scalar/vector maps,
/// this channel stores discrete RGB labels and must never be linearly filtered.
const ID_MAP_TEXTURE: usize = MAPS[MAP_COUNT - 1].tex;

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

/// CPU-side image data prepared for a later upload on the wgpu thread.
pub(crate) struct PreparedFillImage {
    name: String,
    encoded: Arc<Vec<u8>>,
    mips: gpu::PreparedMipChain,
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
#[derive(Clone, Copy, PartialEq, Debug)]
enum Pass {
    /// Saves the stack below a faded or masked folder.
    Begin,
    Layer(usize),
    /// Fades the folder's result over the saved stack.
    End(usize),
}

/// Reused CPU-side scratch for building the layer/folder pass plan.
/// Keeps hash-table and nested-vector capacity across repeated recompositions.
#[derive(Default)]
struct CompositeScratch {
    index_by_id: HashMap<u64, usize>,
    parents: Vec<Option<usize>>,
    block_start: Vec<usize>,
    shown: Vec<bool>,
    isolates: Vec<bool>,
    opening_at: Vec<Vec<usize>>,
    passes: Vec<Pass>,
}

/// Builds the composite pass list in linear time.
///
/// Layers are stored with a folder after its contiguous descendants. That
/// ordering lets us resolve parent indices once, propagate each child's
/// block start upward in a forward scan, and resolve inherited visibility in
/// a reverse scan. Folder openings are bucketed by block start; iterating
/// folder indices backwards preserves the old outermost-first order without
/// allocating and sorting a temporary Vec for every layer.
#[cfg(test)]
fn composite_passes(layers: &[Layer]) -> (Vec<Pass>, usize) {
    composite_passes_with_scratch(layers, &mut CompositeScratch::default())
}

fn composite_passes_with_scratch(
    layers: &[Layer],
    scratch: &mut CompositeScratch,
) -> (Vec<Pass>, usize) {
    let n = layers.len();
    if n == 0 {
        return (Vec::new(), 0);
    }

    let index_by_id = &mut scratch.index_by_id;
    index_by_id.clear();
    index_by_id.extend(layers.iter().enumerate().map(|(i, layer)| (layer.id, i)));

    let parents = &mut scratch.parents;
    parents.clear();
    parents.extend(layers.iter().map(|layer| layer.parent.and_then(|id| index_by_id.get(&id).copied())));

    let block_start = &mut scratch.block_start;
    block_start.clear();
    block_start.extend(0..n);
    // A folder's block starts at the earliest descendant, including through
    // nested folders. Descendants precede their parent in the layer vector.
    for i in 0..n {
        if let Some(parent) = parents[i]
            && parent > i
        {
            block_start[parent] = block_start[parent].min(block_start[i]);
        }
    }

    // Resolve inherited visibility from parent to child. Valid layer stacks
    // place every parent after its descendants, so the parent's result is
    // already available when visiting a child.
    let shown = &mut scratch.shown;
    shown.clear();
    shown.resize(n, false);
    for i in (0..n).rev() {
        let parent_shown = parents[i].is_none_or(|parent| {
            if parent > i {
                shown[parent]
            } else {
                // Defensive fallback for malformed ordering; preserve the
                // previous ancestor walk rather than hiding a visible child.
                let mut at = Some(parent);
                while let Some(index) = at {
                    if !layers[index].visible {
                        return false;
                    }
                    at = parents[index];
                }
                true
            }
        });
        shown[i] = layers[i].visible && parent_shown;
    }

    let isolates = &mut scratch.isolates;
    isolates.clear();
    isolates.resize(n, false);
    let opening_at = &mut scratch.opening_at;
    opening_at.truncate(n);
    while opening_at.len() < n {
        opening_at.push(Vec::new());
    }
    for bucket in opening_at.iter_mut() {
        bucket.clear();
    }
    let mut folder_count = 0usize;
    for i in (0..n).rev() {
        let layer = &layers[i];
        let isolated = layer.is_folder() && (layer.opacity < 1.0 || layer.mask.is_some());
        isolates[i] = isolated;
        if isolated && shown[i] {
            opening_at[block_start[i]].push(i);
            folder_count += 1;
        }
    }

    let mut passes = std::mem::take(&mut scratch.passes);
    passes.clear();
    passes.reserve(n + folder_count * 2);
    let (mut depth, mut max_depth) = (0usize, 0usize);
    for (i, layer) in layers.iter().enumerate() {
        // Buckets were populated in reverse index order: outer folders first.
        for _ in &opening_at[i] {
            passes.push(Pass::Begin);
            depth += 1;
            max_depth = max_depth.max(depth);
        }

        let has_content = match &layer.content {
            Content::Paint(ch) => ch.iter().enumerate().any(|(ch, tex)| tex.is_some() && !layer.disabled_channels[ch]),
            Content::Fill(fill) => fill.mat.enabled.iter().enumerate().any(|(ch, enabled)| *enabled && !layer.disabled_channels[ch]),
            Content::Folder(_) => false,
        };
        if has_content && layer.opacity > 0.0 && shown[i] {
            passes.push(Pass::Layer(i));
        }
        if isolates[i] && shown[i] {
            passes.push(Pass::End(i));
            depth -= 1;
        }
    }

    debug_assert_eq!(depth, 0, "composite folder pass depth must balance");
    (passes, max_depth)
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
    /// Per-channel accumulation indices before the painted layer. `None`
    /// means the input is held in the shared prefix cache.
    input: Option<[usize; ACCUM]>,
}

const ACCUM_CHANNEL_MASK: u8 = ((1u16 << ACCUM) - 1) as u8;

/// Outputs touched by one layer. Base-color alpha carries opacity, so an
/// opacity-only layer must still write the base-color accumulation target.
fn composite_output_mask(u: &LayerUniform) -> u8 {
    let mut mask = 0u8;
    if u.ch[0] > 0.5 || u.fill_rmh[3] > 0.5 { mask |= 1 << 0; }
    if u.ch[1] > 0.5 { mask |= 1 << 1; }
    if u.ch[2] > 0.5 { mask |= 1 << 2; }
    if u.ch[3] > 0.5 { mask |= 1 << 3; }
    if u.emis[3] > 0.5 { mask |= 1 << 4; }
    // The original full-output shader normalized and re-encoded the previous
    // normal on every layer, even when the layer had no normal contribution.
    // Keep that exact channel path until a separately validated identity fast path exists.
    mask |= 1 << 5;
    if u.ns[3] > 0.5 { mask |= 1 << 6; }
    mask
}

fn toggle_accum_indices(mut indices: [usize; ACCUM], mask: u8) -> [usize; ACCUM] {
    for (channel, index) in indices.iter_mut().enumerate() {
        if mask & (1 << channel) != 0 { *index = 1 - *index; }
    }
    indices
}

/// The first resumed pass reads from the prefix cache and seeds all channels;
/// later passes only write channels changed by that layer.
fn resume_source_indices(final_indices: [usize; ACCUM], passes: &[Pass], output_masks: &[u8]) -> [usize; ACCUM] {
    let mut source = final_indices;
    for (pass, &mask) in passes.iter().zip(output_masks) {
        if !matches!(pass, Pass::Begin) {
            source = toggle_accum_indices(source, mask);
        }
    }
    source
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
    /// A filter step. `reach` is how many texels it reads around the one it
    /// writes; `wrap` is whether those reads wrap around the texture edge.
    Filter { step: usize, reach: f32, wrap: bool },
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
    /// Per-layer project-channel mask. Source maps and paint remain stored;
    /// masked outputs are excluded from compositing and export until re-enabled.
    pub disabled_channels: [bool; CHANNELS],
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
    /// Cached color/roughness FBM flags, transform selector and output-channel mask.
    varc: [f32; 4],
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
    /// Captured on the first valid paint pass, not when the pointer goes down.
    before: Option<Snapshot>,
}


/// OpenGL cannot lower textureLoad on a depth texture through Naga's GLSL backend.
/// For GL only, sample the same per-sample depth encoded into an RGBA8 MSAA target.
fn stamp_shader_source(gl_depth_color: bool) -> String {
    let mut source = include_str!("shaders/stamp.wgsl").to_owned();
    if !gl_depth_color {
        return source;
    }

    const DEPTH_BINDING: &str = "@group(0) @binding(4) var depth_map: texture_depth_multisampled_2d;";
    const COLOR_BINDING: &str = "@group(0) @binding(4) var depth_map: texture_2d<f32>;";
    const UNPACK: &str = r#"
fn unpack_depth24(packed: vec4<f32>) -> f32 {
    let bytes = vec3<u32>(round(clamp(packed.rgb, vec3<f32>(0.0), vec3<f32>(1.0)) * 255.0));
    let value = (bytes.x << 16u) | (bytes.y << 8u) | bytes.z;
    return f32(value) / 16777215.0;
}
"#;
    assert_eq!(source.matches(DEPTH_BINDING).count(), 1, "stamp depth binding must remain unique");
    source = source.replacen(DEPTH_BINDING, COLOR_BINDING, 1);

    const LINEAR_DEPTH: &str = "fn lin_depth(z: f32) -> f32 {";
    assert_eq!(source.matches(LINEAR_DEPTH).count(), 1, "stamp linear-depth helper must remain unique");
    source = source.replacen(LINEAR_DEPTH, &format!("{UNPACK}\n{LINEAR_DEPTH}"), 1);

    const POSITION_LOAD: &str = r#"let scene = textureLoad(depth_map, sp, 0);
        if (lin_depth(ndc.z) > lin_depth(scene) * 1.01 + 0.005) {"#;
    const POSITION_LOAD_GL: &str = r#"let scene = unpack_depth24(textureLoad(depth_map, sp, 0)) * u.clip.y;
        if (lin_depth(ndc.z) > scene * 1.01 + 0.005) {"#;
    assert_eq!(source.matches(POSITION_LOAD).count(), 1, "stamp position depth load must remain unique");
    source = source.replacen(POSITION_LOAD, POSITION_LOAD_GL, 1);

    const SAMPLE_START: &str = "        let sample_count = textureNumSamples(depth_map);";
    const SAMPLE_END: &str = "        visibility = f32(visible_samples) / f32(sample_count);";
    let start = source.find(SAMPLE_START).expect("stamp MSAA visibility loop must exist");
    let tail = &source[start..];
    let end = tail.find(SAMPLE_END).expect("stamp MSAA visibility loop terminator must exist") + SAMPLE_END.len();
    const SINGLE_SAMPLE: &str = r#"        let sample_depth = unpack_depth24(textureLoad(depth_map, sp, 0)) * u.clip.y;
        if (abs(surface_linear - sample_depth) > epsilon) {
            return vec3<f32>(0.0);
        }
        visibility = 1.0;"#;
    source.replace_range(start..start + end, SINGLE_SAMPLE);
    source
}

#[cfg(test)]
mod stamp_shader_variant_tests {
    use super::stamp_shader_source;
    use eframe::wgpu;

    #[test]
    fn gl_stamp_variant_uses_single_sample_color_depth_and_valid_wgsl() {
        let source = stamp_shader_source(true);
        assert!(source.contains("texture_2d<f32>"));
        assert!(!source.contains("texture_multisampled_2d"));
        assert!(!source.contains("texture_depth_multisampled_2d"));
        assert!(!source.contains("textureNumSamples(depth_map)"));
        assert_eq!(source.matches("unpack_depth24(textureLoad(").count(), 2);

        let full = format!("{}\n{}", include_str!("shaders/common.wgsl"), source);
        let module = wgpu::naga::front::wgsl::parse_str(&full)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&full)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{error:?}"));
    }
    #[test]
    fn vulkan_stamp_shader_remains_byte_identical() {
        assert_eq!(stamp_shader_source(false), include_str!("shaders/stamp.wgsl"));
    }
}

struct Pipelines {
    uvbake: wgpu::RenderPipeline,
    dilate: wgpu::RenderPipeline,
    dilate_bgl: wgpu::BindGroupLayout,
    stamp: wgpu::RenderPipeline,
    /// Fullscreen stamp pass restricted to dilated UV padding texels.
    stamp_pad: wgpu::RenderPipeline,
    /// Screen projection rasterized per mesh triangle in UV space.
    stamp_uv: wgpu::RenderPipeline,
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
    composite_layout: wgpu::PipelineLayout,
    composite_variants: std::sync::Mutex<HashMap<(bool, u8), wgpu::RenderPipeline>>,
    composite_bgl: wgpu::BindGroupLayout,
    /// OpenGL uses separate core/extra passes and a narrow, precomputed-mask layout.
    gl_composite: bool,
    gl_extra_layout: wgpu::PipelineLayout,
    gl_extra_bgl: wgpu::BindGroupLayout,
    gl_folder_bgl: wgpu::BindGroupLayout,
    gl_mask_bgl: wgpu::BindGroupLayout,
    gl_composite_variants: std::sync::Mutex<HashMap<u8, (Option<wgpu::RenderPipeline>, Option<wgpu::RenderPipeline>)>>,
    /// Built only when a Fill layer actually uses color or roughness variation.
    composite_module: wgpu::ShaderModule,
    fill_variation_noise: std::sync::OnceLock<wgpu::RenderPipeline>,
    fill_variation_noise_bgl: wgpu::BindGroupLayout,
    mask_source: wgpu::RenderPipeline,
    mask_filter: wgpu::RenderPipeline,
    mask_filter_bgl: wgpu::BindGroupLayout,
    resample: HashMap<F, wgpu::RenderPipeline>,
    /// The categorical ID labels use an integer texel load instead of bilinear filtering.
    resample_nearest_id: wgpu::RenderPipeline,
    resample_bgl: wgpu::BindGroupLayout,
    resolve: HashMap<F, wgpu::RenderPipeline>,
    resolve_bgl: wgpu::BindGroupLayout,
}

impl Pipelines {
    fn composite_pipeline(&self, device: &wgpu::Device, output_mask: u8, folder_merge: bool) -> wgpu::RenderPipeline {
        if output_mask == ACCUM_CHANNEL_MASK {
            return if folder_merge { self.folder_merge.clone() } else { self.composite.clone() };
        }
        let key = (folder_merge, output_mask);
        let mut cache = self.composite_variants.lock().unwrap_or_else(|e| e.into_inner());
        cache.entry(key).or_insert_with(|| {
            let targets = ACCUM_FORMATS.iter().enumerate().map(|(channel, &format)| {
                if output_mask & (1 << channel) != 0 { gpu::target(format) } else { None }
            }).collect::<Vec<_>>();
            let (label, entry) = if folder_merge { ("folder merge channels", "fs_folder_merge") } else { ("composite channels", "fs_layer") };
            gpu::fullscreen_pipeline(device, label, &self.composite_module, entry, &self.composite_layout, &targets)
        }).clone()
    }

    /// OpenGL's 16-texture floor is met by splitting the seven accumulation
    /// channels across two pipelines. Each pipeline binds only the resources
    /// its entry point reads; the mask has already been resolved to l_mask.
    fn gl_layer_pipelines(
        &self,
        device: &wgpu::Device,
        output_mask: u8,
    ) -> (Option<wgpu::RenderPipeline>, Option<wgpu::RenderPipeline>) {
        let mut cache = self.gl_composite_variants.lock().unwrap_or_else(|e| e.into_inner());
        cache.entry(output_mask).or_insert_with(|| {
            let core_mask = output_mask & 0x0f;
            let extra_mask = (output_mask >> 4) & 0x07;
            let core = if core_mask != 0 {
                let targets = (0..4).map(|channel| {
                    if core_mask & (1 << channel) != 0 { gpu::target(ACCUM_FORMATS[channel]) } else { None }
                }).collect::<Vec<_>>();
                Some(gpu::fullscreen_pipeline(
                    device, "OpenGL layer core", &self.composite_module, "fs_layer_gl_core",
                    &self.composite_layout, &targets,
                ))
            } else {
                None
            };
            let extra = if extra_mask != 0 {
                let targets = (0..3).map(|channel| {
                    if extra_mask & (1 << channel) != 0 { gpu::target(ACCUM_FORMATS[channel + 4]) } else { None }
                }).collect::<Vec<_>>();
                Some(gpu::fullscreen_pipeline(
                    device, "OpenGL layer extra", &self.composite_module, "fs_layer_gl_extra",
                    &self.gl_extra_layout, &targets,
                ))
            } else {
                None
            };
            (core, extra)
        }).clone()
    }
}

#[derive(Clone)]
struct GpuStrokeProfileResources {
    query_set: Arc<wgpu::QuerySet>,
    resolve: Arc<wgpu::Buffer>,
    readback: Arc<wgpu::Buffer>,
}

const GPU_PROFILE_BASE_QUERY_COUNT: u32 = 10;
const GPU_PROFILE_MAX_LAYER_PASSES: usize = 64;
const GPU_PROFILE_QUERY_COUNT: u32 = GPU_PROFILE_BASE_QUERY_COUNT + GPU_PROFILE_MAX_LAYER_PASSES as u32 * 2;
const GPU_PROFILE_READBACK_SIZE: u64 = GPU_PROFILE_QUERY_COUNT as u64 * 8;

fn begin_composite_profile_stage(
    encoder: &mut wgpu::CommandEncoder,
    query_set: Option<&wgpu::QuerySet>,
    stage_count: &mut usize,
) -> Option<usize> {
    let query_set = query_set?;
    if *stage_count >= GPU_PROFILE_MAX_LAYER_PASSES {
        return None;
    }
    let slot = *stage_count;
    *stage_count += 1;
    encoder.write_timestamp(query_set, GPU_PROFILE_BASE_QUERY_COUNT + (slot as u32 * 2));
    Some(slot)
}

fn end_composite_profile_stage(
    encoder: &mut wgpu::CommandEncoder,
    query_set: Option<&wgpu::QuerySet>,
    slot: Option<usize>,
    label: String,
    labels: &mut Vec<String>,
) {
    let (Some(query_set), Some(slot)) = (query_set, slot) else { return; };
    encoder.write_timestamp(query_set, GPU_PROFILE_BASE_QUERY_COUNT + (slot as u32 * 2) + 1);
    labels.push(label);
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
            count: GPU_PROFILE_QUERY_COUNT,
        }));
        let resolve = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stroke timestamp resolve"),
            size: GPU_PROFILE_READBACK_SIZE,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let readback = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stroke timestamp readback"),
            size: GPU_PROFILE_READBACK_SIZE,
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

    fn poll(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Option<(f64, f64, f64, Option<f64>, Option<f64>, Option<f64>, Vec<f64>)> {
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
        let (s0, s1, a0, a1, c0, c1) =
            (read(0), read(1), read(2), read(3), read(4), read(5));
        let (t0, t1, clear_end, body_end) = if self.encoder_timestamps {
            (read(6), read(7), read(8), read(9))
        } else {
            (0, 0, 0, 0)
        };
        let layer_pass_ms = if self.encoder_timestamps {
            (0..GPU_PROFILE_MAX_LAYER_PASSES)
                .map(|i| read(10 + i * 2 + 1).wrapping_sub(read(10 + i * 2)) as f64)
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
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
            self.encoder_timestamps.then(|| clear_end.wrapping_sub(t0) as f64 * period_ms),
            self.encoder_timestamps.then(|| body_end.wrapping_sub(clear_end) as f64 * period_ms),
            layer_pass_ms.into_iter().map(|ticks| ticks * period_ms).collect(),
        ))
    }
}

pub struct Engine {
    /// Stable identity used by renderer-side caches across Engine moves/swaps.
    id: u64,
    /// Tiling, mipmapped sampler for fill images.
    repeat_sampler: wgpu::Sampler,
    /// The stack below each open faded/masked folder while its layers are
    /// composited, one set per nesting level; only allocated while needed.
    folder_below: Vec<[Tex; ACCUM]>,
    rs: egui_wgpu::RenderState,
    /// Whether this adapter supports alternate sRGB views of UNORM textures.
    srgb_views_supported: bool,
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Optional non-blocking GPU timings when the adapter supports timestamps.
    gpu_profiler: Option<GpuStrokeProfiler>,
    /// Holds the current sample until its layer composite pass is recorded.
    active_gpu_profile: Option<GpuStrokeProfileResources>,
    /// F5 requests one complete-composite GPU timestamp capture.
    composite_profile_requested: bool,
    /// Distinguishes a forced F5 capture from the normal paint profiler readback.
    forced_composite_profile: bool,
    /// Shared by the texture sets of a document.
    p: Arc<Pipelines>,
    /// Bounded readback-buffer cache shared by this document's Texture Sets.
    snapshot_buffer_pool: Arc<crate::undo::SnapshotBufferPool>,
    /// Scratch maps/vectors reused to plan layer and folder composite passes.
    composite_scratch: CompositeScratch,
    /// Names paired with the detailed GPU timestamps from the most recent composite.
    last_profiled_pass_labels: Vec<String>,
    /// CPU milliseconds for the five phases of the most recent stroke end:
    /// cleanup, texture lookup, snapshot encoding, submit/map request, thumbnails.
    pub last_end_stages_ms: [f64; 5],
    pub res: u32,
    /// The texture set's range in the mesh's index buffer: only these
    /// triangles get position and normal maps, so only they take paint.
    tris: std::ops::Range<u32>,
    /// Reusable compact index buffer for the triangles near a screen stroke.
    /// Avoids submitting the entire mesh for every brush stamp.
    screen_index_buffer: wgpu::Buffer,
    screen_index_capacity: u32,
    pos: Tex,
    nrm: Tex,
    /// Tangent frame per texel: fill normal-map images projected in world
    /// space (tri-planar, planar) are turned into tangent space with it.
    tan: Tex,
    accum: [[Tex; ACCUM]; 2],
    /// The accumulation set currently holding each final material channel.
    final_indices: [usize; ACCUM],
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
    /// What the position/normal/tangent maps were last baked from, so
    /// `set_triangles` can skip an identical bake.
    uv_baked: Option<UvBakeKey>,
    /// Layer thumbnails (see `thumbs`).
    thumbs: thumbs::Thumbs,
    /// Composites that redid the whole texture instead of a stroke's patch (the F6 meter counts them).
    pub full_recomps: u32,
    /// Texel area used by the most recent stamp/apply scissor (full texture on fallback).
    pub last_paint_scissor_pixels: u64,
    /// Number of non-overlapping stamp/apply scissors in the most recent stroke step.
    pub last_paint_scissor_rects: u32,
    /// Texel area processed by the most recent composite scissor (full texture on full recomposites).
    pub last_scissor_pixels: u64,
    /// Bounding-box area of the last scissor set (for F6 comparison).
    pub last_scissor_bbox_pixels: u64,
    /// Number of non-overlapping scissors used by the last composite.
    pub last_scissor_rects: u32,
    /// Pass counts and mask-cache activity from the most recent composite (F6).
    pub last_composite_layer_passes: u32,
    pub last_composite_folder_passes: u32,
    pub last_mask_cache_hits: u32,
    pub last_mask_cache_misses: u32,
    pub last_mask_gpu_passes: u32,
    /// Procedural fill color/roughness FBM cache activity from the most recent composite.
    pub last_fill_variation_cache_hits: u32,
    pub last_fill_variation_cache_misses: u32,
    pub last_fill_variation_cache_gpu_passes: u32,
    /// Test-only switch to compare the cache against the original inline shader path.
    #[cfg(test)]
    pub(crate) force_inline_fill_variation: bool,
    /// Test-only reference path: every layer emits all seven accumulation channels.
    #[cfg(test)]
    pub(crate) force_full_composite_channel_outputs: bool,
    /// IDs whose final filtered mask texture currently holds valid contents.
    mask_final: HashSet<u64>,
    pub bake_report: Option<Arc<BakeReport>>,
    edge_max: f32,
    dummy: Tex,
    sampler: wgpu::Sampler,
    layer_ubuf: wgpu::Buffer,
    layer_ubuf_slots: u64,
    /// Reused CPU staging for per-composite layer uniforms.
    layer_uniform_scratch: Vec<u8>,
    stamp_ubuf: wgpu::Buffer,
    stamp_sbuf: wgpu::Buffer,
    apply_ubuf: wgpu::Buffer,
    /// Brush tip (alpha) the stamps sample when the brush uses one.
    brush_alpha: Tex,
    /// Brush grain texture (white when the brush has none).
    brush_grain: Tex,
    /// Image files already on the GPU, shared by every layer that uses them
    /// (scanned smart materials); entries die with their last user.
    /// Weak cache shared by every Texture Set Engine in the same document.
    image_cache: Arc<std::sync::Mutex<HashMap<std::path::PathBuf, (std::sync::Weak<Tex>, std::sync::Weak<Vec<u8>>)>>>,
    /// Reused pointer set for allocation-free repeated GPU memory estimates.
    fill_image_ids_scratch: std::cell::RefCell<HashSet<usize>>,
    pub layers: Vec<Layer>,
    /// Properties as of the last recorded edit; see [`Engine::commit_props`].
    committed: DocProps,
    next_id: u64,
    bounds: (Vec3, Vec3),
    active: Option<ActiveStroke>,
    /// Apply bindings are stable for a stroke; reuse them across its frames.
    apply_bg: Option<wgpu::BindGroup>,
    /// Stamp bindings are stable within a stroke. Screen/Surface and UV/2D
    /// use different depth views, so each path retains its own bind group.
    stamp_bg: [Option<Arc<wgpu::BindGroup>>; 2],
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
    /// One persistent result texture per filtered mask.
    mask_out: HashMap<u64, Tex>,
    /// Full-resolution, one-channel FBM fields for stable fill roughness variations.
    fill_variation_noise: HashMap<u64, FillVariationNoiseCache>,
    /// Three scratch textures shared by every filtered mask in this engine:
    /// two ping-pong buffers and one temporary mask-layer source.
    mask_scratch: Option<[Tex; 3]>,
    mask_filter_ubuf: wgpu::Buffer,
    mask_filter_ubuf_slots: u64,
    /// Reused CPU staging for per-composite mask-filter uniforms.
    mask_filter_uniform_scratch: Vec<u8>,
    /// Reused CPU list of filter/combine steps for the mask pass encoder.
    mask_filter_steps_scratch: Vec<[f32; 8]>,
    /// Disjoint UV-space rectangles changed by the active stroke, in texels.
    /// Multiple scissors keep distant UV islands from forcing one large bbox.
    stroke_regions: Vec<[u32; 4]>,
    region_unknown: bool,
    /// Incremented on every composite, so dependent views know the material changed.
    pub generation: u64,
}

/// The inputs of a UV bake: which GPU mesh, which range of its index buffer,
/// and the texture size. Two bakes with an equal key give identical maps.
#[derive(Clone, PartialEq, Eq, Debug)]
struct UvBakeKey {
    mesh_id: u64,
    tris: std::ops::Range<u32>,
    res: u32,
}

impl UvBakeKey {
    /// Whether baking `mesh_id`/`tris`/`res` would repeat the bake `baked` recorded.
    fn is_current(baked: Option<&UvBakeKey>, mesh_id: u64, tris: &std::ops::Range<u32>, res: u32) -> bool {
        baked.is_some_and(|k| k.mesh_id == mesh_id && k.tris == *tris && k.res == res)
    }
}

/// Inputs that change the color/roughness FBM fields. Variation amounts are
/// excluded because they scale the cached noise cheaply in fs_layer.
#[derive(Clone, Copy, PartialEq, Eq)]
struct FillVariationNoiseKey {
    res: u32,
    uv_version: u64,
    scale_bits: u32,
    seed_bits: u32,
    color_enabled: bool,
    roughness_enabled: bool,
}

struct FillVariationNoiseCache {
    key: FillVariationNoiseKey,
    color: Option<Tex>,
    roughness: Option<Tex>,
}

fn accum_set(device: &wgpu::Device, res: u32, label: &str, srgb_views_supported: bool) -> [Tex; ACCUM] {
    ACCUM_FORMATS.map(|f| {
        if srgb_views_supported && f == F::Rgba8Unorm {
            Tex::new(device, label, res, res, f, LAYER_USAGE)
        } else {
            Tex::new_without_srgb_view(device, label, res, res, f, LAYER_USAGE)
        }
    })
}

/// Maximum number of disjoint scissors per composite. Above this, merge
/// the pair with the least added area; this is conservative and bounds draw
/// submission cost even for highly fragmented UV layouts.
const MAX_COMPOSITE_SCISSORS: usize = 64;

fn scissor_area(r: [u32; 4]) -> u64 {
    u64::from(r[2].saturating_sub(r[0])) * u64::from(r[3].saturating_sub(r[1]))
}

fn scissor_union(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    [a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])]
}

fn scissor_overlap_area(a: [u32; 4], b: [u32; 4]) -> u64 {
    u64::from(a[2].min(b[2]).saturating_sub(a[0].max(b[0])))
        * u64::from(a[3].min(b[3]).saturating_sub(a[1].max(b[1])))
}

/// Expands one half-open interval, optionally wrapping across the texture seam.
fn expand_scissor_axis(start: u32, end: u32, margin: u32, limit: u32, wrap: bool) -> ([(u32, u32); 2], usize) {
    if limit == 0 {
        return ([(0, 0); 2], 0);
    }
    if !wrap {
        return ([(start.saturating_sub(margin), end.saturating_add(margin).min(limit)), (0, 0)], 1);
    }
    let length = u64::from(end.saturating_sub(start)) + 2 * u64::from(margin);
    if length >= u64::from(limit) {
        return ([(0, limit), (0, 0)], 1);
    }
    let start = (i64::from(start) - i64::from(margin)).rem_euclid(i64::from(limit)) as u32;
    let end = start + length as u32;
    if end <= limit {
        ([(start, end), (0, 0)], 1)
    } else {
        ([(start, limit), (0, end - limit)], 2)
    }
}

/// Expands patches for mask-filter neighborhoods. Warp uses a repeating
/// sampler, so its affected region can cross the UV seam and split in two
/// along either axis (at most four rectangles per input patch).
fn expand_scissors(rects: &mut Vec<[u32; 4]>, margin: u32, limit: u32, wrap: bool) {
    if wrap {
        let mut expanded = Vec::with_capacity(rects.len().saturating_mul(4));
        for &[x0, y0, x1, y1] in rects.iter() {
            let (xs, nx) = expand_scissor_axis(x0, x1, margin, limit, true);
            let (ys, ny) = expand_scissor_axis(y0, y1, margin, limit, true);
            for &(xa, xb) in &xs[..nx] {
                for &(ya, yb) in &ys[..ny] {
                    if xb > xa && yb > ya {
                        expanded.push([xa, ya, xb, yb]);
                    }
                }
            }
        }
        *rects = expanded;
    } else {
        for r in rects.iter_mut() {
            *r = [
                r[0].saturating_sub(margin),
                r[1].saturating_sub(margin),
                r[2].saturating_add(margin).min(limit),
                r[3].saturating_add(margin).min(limit),
            ];
        }
    }
    compact_scissors(rects);
}

/// Keep scissors disjoint. If the list exceeds the cap, merge the cheapest
/// pair by added texel area; a merge only expands coverage, never drops it.
/// Converts conservative normalized paint bounds to half-open pixel rectangles.
fn normalized_scissors(regions: &[[f32; 4]], res: u32) -> Option<Vec<[u32; 4]>> {
    if regions.iter().flatten().any(|v| !v.is_finite()) {
        return None;
    }
    let size = res.max(1) as f32;
    let low = |v: f32| (v * size).floor().clamp(0.0, size) as u32;
    let high = |v: f32| (v * size).ceil().clamp(0.0, size) as u32;
    // Position/normal maps are dilated around UV islands. Keep that halo in
    // the raster bounds too, or a brush near an island edge can lose texels.
    let margin = DILATE_PASSES + 2;
    let mut rects: Vec<[u32; 4]> = regions
        .iter()
        .map(|&[u0, v0, u1, v1]| {
            let [x0, y0, x1, y1] = [low(u0), low(v0), high(u1), high(v1)];
            [
                x0.saturating_sub(margin),
                y0.saturating_sub(margin),
                x1.saturating_add(margin).min(res),
                y1.saturating_add(margin).min(res),
            ]
        })
        .filter(|r| r[2] > r[0] && r[3] > r[1])
        .collect();
    compact_scissors(&mut rects);
    Some(rects)
}

fn compact_scissors(rects: &mut Vec<[u32; 4]>) {
    loop {
        let mut merge: Option<(usize, usize, [u32; 4])> = None;
        let mut best_extra = u64::MAX;
        for i in 0..rects.len() {
            for j in i + 1..rects.len() {
                let a = rects[i];
                let b = rects[j];
                let union = scissor_union(a, b);
                let covered = scissor_area(a) + scissor_area(b) - scissor_overlap_area(a, b);
                let extra = scissor_area(union).saturating_sub(covered);
                let overlaps = scissor_overlap_area(a, b) > 0;
                if (overlaps || extra == 0) && extra < best_extra {
                    best_extra = extra;
                    merge = Some((i, j, union));
                }
            }
        }
        if let Some((i, j, union)) = merge {
            rects[i] = union;
            rects.swap_remove(j);
            continue;
        }
        if rects.len() <= MAX_COMPOSITE_SCISSORS {
            break;
        }
        let mut best = (u64::MAX, 0usize, 1usize, [0u32; 4]);
        for i in 0..rects.len() {
            for j in i + 1..rects.len() {
                let a = rects[i];
                let b = rects[j];
                let union = scissor_union(a, b);
                let covered = scissor_area(a) + scissor_area(b) - scissor_overlap_area(a, b);
                let extra = scissor_area(union).saturating_sub(covered);
                if extra < best.0 {
                    best = (extra, i, j, union);
                }
            }
        }
        rects[best.1] = best.3;
        rects.swap_remove(best.2);
    }
    rects.sort_unstable_by_key(|r| (r[1], r[0], r[3], r[2]));
}

fn fit_fill_image<P: image::Pixel<Subpixel = u8> + 'static>(
    mut img: image::ImageBuffer<P, Vec<u8>>,
) -> image::ImageBuffer<P, Vec<u8>> {
    if img.width().max(img.height()) > MAX_IMAGE {
        let k = MAX_IMAGE as f32 / img.width().max(img.height()) as f32;
        let (w, h) = (((img.width() as f32 * k) as u32).max(1), ((img.height() as f32 * k) as u32).max(1));
        img = image::imageops::resize(&img, w, h, image::imageops::FilterType::Triangle);
    }
    img
}

impl Engine {
    /// Identity and texture-view generations used by the 3D viewport bind-group cache.
    /// Final composites swap ping-pong targets; UV bakes and map changes replace
    /// bound views. Keeping these generations explicit avoids rebuilding bind
    /// groups on every frame while still invalidating them when a view changes.
    pub(crate) fn viewport_binding_identity(&self) -> (u64, u32, [usize; ACCUM], u64, u64) {
        (self.id, self.res, self.final_indices, self.uv_version, self.maps_version)
    }

    /// A document of one texture set covering the whole mesh.
    pub fn new(rs: &egui_wgpu::RenderState, mesh: &GpuMesh, bounds: (Vec3, Vec3), res: u32) -> Engine {
        let gl_depth_color = rs.adapter.get_info().backend == wgpu::Backend::Gl;
        let p = {
            let _s = crate::startup_trace::span("Engine::create_pipelines");
            Arc::new(Self::create_pipelines(&rs.device, gl_depth_color))
        };
        Self::with_pipelines(
            rs,
            p,
            mesh,
            0..mesh.index_count,
            bounds,
            res,
            Arc::new(crate::undo::SnapshotBufferPool::default()),
        )
    }

    /// Another texture set of the same document: `tris` is its range in the
    /// mesh's index buffer. Shares this engine's pipelines and document bounds.
    pub fn sibling(&self, mesh: &GpuMesh, tris: std::ops::Range<u32>, res: u32) -> Engine {
        let mut sibling = Self::with_pipelines(
            &self.rs, self.p.clone(), mesh, tris, self.bounds, res, self.snapshot_buffer_pool.clone(),
        );
        sibling.image_cache = self.image_cache.clone();
        sibling
    }

    /// Remove expired weak entries from the document-shared Fill image cache.
    /// The cache is only an index: live layers/history own the actual textures and bytes.
    fn prune_fill_image_cache(&self) {
        let mut cache = self.image_cache.lock().unwrap_or_else(|poison| poison.into_inner());
        let before = cache.len();
        cache.retain(|_, (texture, bytes)| texture.strong_count() > 0 && bytes.strong_count() > 0);
        if cache.len() < before / 2 {
            cache.shrink_to_fit();
        }
    }

    /// Drops GPU resources used only while this Texture Set is being edited.
    /// The final composited result stays resident for the viewport. The next
    /// composite/paint recreates the released workspace lazily.
    pub(crate) fn release_inactive_working_set(&mut self) {
        if self.active.is_some() || self.generation == 0 {
            return;
        }
        // Drop expired path/Weak bookkeeping from the document-wide cache on
        // set switches, not on every frame or image lookup.
        self.prune_fill_image_cache();
        // A parked Engine cannot reuse an incremental stroke prefix until it is
        // active again. Drop it now so the workspace below can be reclaimed.
        self.prefix = None;
        self.full_dirty = true;
        for channel in 0..ACCUM {
            let scratch = 1 - self.final_indices[channel];
            if self.accum[scratch][channel].width != 1 {
                self.accum[scratch][channel] = self.new_tex("parked composite scratch", 1, 1, ACCUM_FORMATS[channel], LAYER_USAGE,
                );
            }
        }
        if self.stroke.width != 1 {
            self.stroke = self.new_tex("parked stroke mask", 1, 1, STROKE_FORMAT, LAYER_USAGE);
        }
        self.snap = Default::default();
        self.snap_mask = None;
        self.folder_below.clear();
        self.folder_below.shrink_to_fit();
        self.composite_scratch = CompositeScratch::default();
        self.stroke_regions.clear();
        self.stroke_regions.shrink_to_fit();
        self.last_profiled_pass_labels.clear();
        self.last_profiled_pass_labels.shrink_to_fit();
        self.fill_image_ids_scratch.get_mut().clear();
        self.fill_image_ids_scratch.get_mut().shrink_to_fit();
        self.prefix_cache = None;
        self.mask_scratch = None;
        self.mask_out.clear();
        self.fill_variation_noise.clear();
        self.mask_final.clear();
        self.layer_uniform_scratch.clear();
        self.layer_uniform_scratch.shrink_to_fit();
        self.mask_filter_uniform_scratch.clear();
        self.mask_filter_uniform_scratch.shrink_to_fit();
        self.mask_filter_steps_scratch.clear();
        self.mask_filter_steps_scratch.shrink_to_fit();
        self.apply_bg = None;
        // Cached stamp bind groups retain views such as the last viewport depth target.
        // A parked Engine won't paint; recreate these lazily on its next stroke.
        self.stamp_bg = [None, None];

        // Parked Engines never encode layer or mask-filter passes. Return their
        // dynamic uniform buffers to the small baseline; they grow lazily again
        // if this Engine is reactivated and needs the larger working set.
        let layer_slots = layer_uniform_buffer_capacity(self.layer_ubuf_slots, 0);
        if layer_slots != self.layer_ubuf_slots {
            self.layer_ubuf_slots = layer_slots;
            self.layer_ubuf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("parked layer uniforms"),
                size: UNIFORM_STRIDE * self.layer_ubuf_slots,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        let mask_slots = mask_filter_buffer_capacity(self.mask_filter_ubuf_slots, 0);
        if mask_slots != self.mask_filter_ubuf_slots {
            self.mask_filter_ubuf_slots = mask_slots;
            self.mask_filter_ubuf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("parked mask filter uniforms"),
                size: MASK_FILTER_STRIDE * self.mask_filter_ubuf_slots,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        // Keep the already-rendered UI thumbnails, but drop the 512²
        // render targets and reduced surface maps used only to regenerate them.
        self.thumbs.release_scratch();
    }

    /// Releases world-space UV maps while this Texture Set is parked.
    /// The built-in Material viewport does not sample them. Their bake key is
    /// retained so restoring the same mesh/range does not invalidate the material.
    pub(crate) fn release_surface_maps(&mut self) -> bool {
        if self.pos.width == 1 && self.nrm.width == 1 && self.tan.width == 1 {
            return false;
        }
        self.pos = self.new_tex("parked position map", 1, 1, POS_FORMAT, LAYER_USAGE);
        self.nrm = self.new_tex("parked normal map", 1, 1, NRM_FORMAT, LAYER_USAGE);
        self.tan = self.new_tex("parked tangent map", 1, 1, TAN_FORMAT, LAYER_USAGE);
        self.uv_version = self.uv_version.wrapping_add(1);
        true
    }

    /// Ensures world-space UV maps exist. An unchanged bake does not invalidate
    /// the already-correct material composite.
    pub(crate) fn ensure_surface_maps(&mut self, mesh: &GpuMesh) -> bool {
        let resident = self.pos.width == self.res
            && self.nrm.width == self.res
            && self.tan.width == self.res;
        let same_bake = UvBakeKey::is_current(self.uv_baked.as_ref(), mesh.mesh_id, &self.tris, self.res);
        if !resident {
            self.bake_uv_internal(mesh, !same_bake);
            true
        } else if !same_bake {
            self.bake_uv(mesh);
            true
        } else {
            false
        }
    }

    fn new_tex(
        &self,
        label: &str,
        width: u32,
        height: u32,
        format: F,
        usage: wgpu::TextureUsages,
    ) -> Tex {
        if self.srgb_views_supported {
            Tex::new(&self.device, label, width, height, format, usage)
        } else {
            Tex::new_without_srgb_view(&self.device, label, width, height, format, usage)
        }
    }

    fn ensure_composite_working_set(&mut self) {
        for set in 0..2 {
            for channel in 0..ACCUM {
                if self.accum[set][channel].width != self.res {
                    self.accum[set][channel] = self.new_tex(
                        if set == 0 { "accum A (restored)" } else { "accum B (restored)" },
                        self.res,
                        self.res,
                        ACCUM_FORMATS[channel],
                        LAYER_USAGE,
                    );
                }
            }
        }
    }

    fn ensure_stroke_target(&mut self) {
        if self.stroke.width != self.res {
            self.stroke = self.new_tex("stroke mask (restored)", self.res, self.res, STROKE_FORMAT, LAYER_USAGE);
        }
    }

    pub(crate) fn restore_active_working_set(&mut self, mesh: &GpuMesh) -> bool {
        self.ensure_composite_working_set();
        self.ensure_stroke_target();
        self.ensure_surface_maps(mesh)
    }

    /// Creates an engine for another mesh while reusing immutable GPU pipelines.
    /// Useful for secondary render targets that share the same device and formats.
    pub fn sibling_with_bounds(
        &self,
        mesh: &GpuMesh,
        tris: std::ops::Range<u32>,
        bounds: (Vec3, Vec3),
        res: u32,
    ) -> Engine {
        let mut sibling = Self::with_pipelines(
            &self.rs, self.p.clone(), mesh, tris, bounds, res, self.snapshot_buffer_pool.clone(),
        );
        sibling.image_cache = self.image_cache.clone();
        sibling
    }

    fn with_pipelines(
        rs: &egui_wgpu::RenderState,
        p: Arc<Pipelines>,
        mesh: &GpuMesh,
        tris: std::ops::Range<u32>,
        bounds: (Vec3, Vec3),
        res: u32,
        snapshot_buffer_pool: Arc<crate::undo::SnapshotBufferPool>,
    ) -> Engine {
        static NEXT_ENGINE_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let id = NEXT_ENGINE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let _allocation = crate::startup_trace::span("Engine::with_pipelines GPU allocations");
        let device = rs.device.clone();
        let queue = rs.queue.clone();
        let srgb_views_supported = rs
            .adapter
            .get_downlevel_capabilities()
            .flags
            .contains(wgpu::DownlevelFlags::VIEW_FORMATS);
        let sampler = gpu::linear_sampler(&device, wgpu::AddressMode::ClampToEdge);
        let gpu_profiler = device.features().contains(wgpu::Features::TIMESTAMP_QUERY).then(|| GpuStrokeProfiler::new(&device));

        let dummy = Tex::new_without_srgb_view(&device, "dummy", 1, 1, F::Rgba8Unorm, LAYER_USAGE);
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

        let screen_index_capacity = (tris.end - tris.start).max(3);
        let screen_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("screen stroke candidate indices"),
            size: u64::from(screen_index_capacity) * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut engine = {
            let _textures = crate::startup_trace::span("Engine::with_pipelines texture/buffer allocation");
            Engine {
            id,
            rs: rs.clone(),
            srgb_views_supported,
            gpu_profiler,
            active_gpu_profile: None,
            composite_profile_requested: false,
            forced_composite_profile: false,
            last_end_stages_ms: [0.0; 5],
            pos: Tex::new_without_srgb_view(&device, "position map", res, res, POS_FORMAT, LAYER_USAGE),
            nrm: Tex::new_without_srgb_view(&device, "normal map", res, res, NRM_FORMAT, LAYER_USAGE),
            tan: Tex::new_without_srgb_view(&device, "tangent map", res, res, TAN_FORMAT, LAYER_USAGE),
            accum: [
                accum_set(&device, res, "accum A", srgb_views_supported),
                accum_set(&device, res, "accum B", srgb_views_supported),
            ],
            final_indices: [0; ACCUM],
            stroke: Tex::new_without_srgb_view(&device, "stroke mask", res, res, STROKE_FORMAT, LAYER_USAGE),
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
            uv_baked: None,
            thumbs: Default::default(),
            full_recomps: 0,
            last_paint_scissor_pixels: 0,
            last_paint_scissor_rects: 0,
            last_scissor_pixels: 0,
            last_scissor_bbox_pixels: 0,
            last_scissor_rects: 0,
            last_composite_layer_passes: 0,
            last_composite_folder_passes: 0,
            last_mask_cache_hits: 0,
            last_mask_cache_misses: 0,
            last_mask_gpu_passes: 0,
            last_fill_variation_cache_hits: 0,
            last_fill_variation_cache_misses: 0,
            last_fill_variation_cache_gpu_passes: 0,
            #[cfg(test)]
            force_inline_fill_variation: false,
            #[cfg(test)]
            force_full_composite_channel_outputs: false,
            mask_final: HashSet::new(),
            bake_report: None,
            edge_max: 0.2,
            dummy,
            sampler,
            layer_ubuf,
            layer_ubuf_slots,
            layer_uniform_scratch: Vec::new(),
            stamp_ubuf,
            stamp_sbuf,
            apply_ubuf,
            brush_alpha: Tex::new_without_srgb_view(&device, "brush alpha", 1, 1, F::R8Unorm, LAYER_USAGE),
            brush_grain: Tex::new_without_srgb_view(&device, "brush grain", 1, 1, F::R8Unorm, LAYER_USAGE),
            image_cache: Arc::new(std::sync::Mutex::new(HashMap::new())),
            fill_image_ids_scratch: Default::default(),
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
            apply_bg: None,
            stamp_bg: [None, None],
            dirty: true,
            full_dirty: true,
            stroke_layer: None,
            prefix: None,
            prefix_cache: None,
            mask_out: HashMap::new(),
            fill_variation_noise: HashMap::new(),
            mask_scratch: None,
            mask_filter_ubuf: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mask filter uniforms"),
                size: MASK_FILTER_STRIDE * INITIAL_MASK_FILTER_STEPS,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            mask_filter_ubuf_slots: INITIAL_MASK_FILTER_STEPS,
            mask_filter_uniform_scratch: Vec::new(),
            mask_filter_steps_scratch: Vec::new(),
            stroke_regions: Vec::new(),
            region_unknown: false,
            generation: 0,
            res,
            tris,
            screen_index_buffer,
            screen_index_capacity,
            p,
            snapshot_buffer_pool,
            composite_scratch: CompositeScratch::default(),
            last_profiled_pass_labels: Vec::new(),
            device,
            queue,
            }
        };
        {
            let _s = crate::startup_trace::span("Engine::bake_uv");
            engine.bake_uv(mesh);
        }
        engine.committed = engine.doc_props();
        engine
    }

    fn create_pipelines(device: &wgpu::Device, gl_depth_color: bool) -> Pipelines {
        // UV-space position/normal bake.
        let uv_module = gpu::shader(device, "uvbake", include_str!("shaders/uvbake.wgsl"));
        let empty_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        let uvbake = {
            let _trace = crate::startup_trace::span("engine render pipeline: uv bake");
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
            })
        };
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
        let stamp_source = stamp_shader_source(gl_depth_color);
        let stamp_module = gpu::shader(device, "stamp", &stamp_source);
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
                        sample_type: if gl_depth_color { wgpu::TextureSampleType::Float { filterable: false } } else { wgpu::TextureSampleType::Depth },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: !gl_depth_color,
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
        let stamp_layout = gpu::layout(device, &stamp_bgl);
        let stamp_target = Some(wgpu::ColorTargetState {
            format: STROKE_FORMAT,
            blend: Some(wgpu::BlendState { color: flow_blend, alpha: flow_blend }),
            write_mask: wgpu::ColorWrites::ALL,
        });
        let stamp = gpu::fullscreen_pipeline(
            device,
            "stamp",
            &stamp_module,
            "fs_stamp",
            &stamp_layout,
            &[stamp_target.clone()],
        );
        let stamp_pad = gpu::fullscreen_pipeline(
            device,
            "stamp padding",
            &stamp_module,
            "fs_stamp_pad",
            &stamp_layout,
            &[stamp_target.clone()],
        );
        let stamp_uv = {
            let _trace = crate::startup_trace::span("engine render pipeline: screen stamp per UV triangle");
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("screen stamp per UV triangle"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("screen stamp per UV triangle layout"),
                bind_group_layouts: &[Some(&stamp_bgl)],
                immediate_size: 0,
            })),
            vertex: wgpu::VertexState {
                module: &stamp_module,
                entry_point: Some("vs_stamp_uv"),
                compilation_options: Default::default(),
                buffers: &[Some(gpu::vertex_layout())],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &stamp_module,
                entry_point: Some("fs_stamp_uv"),
                compilation_options: Default::default(),
                targets: &[stamp_target],
            }),
            multiview_mask: None,
            cache: None,
            })
        };

        // Polygon fill: selected triangles rasterized at their UVs into the stroke mask.
        let poly_module = gpu::shader(device, "poly fill", include_str!("shaders/poly.wgsl"));
        let poly_pad_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("poly padding"),
            entries: &[gpu::tex_entry(0, false)],
        });
        let poly_pipeline = |label: &str, fs: &str, bgls: &[Option<&wgpu::BindGroupLayout>]| {
            let _trace = crate::startup_trace::span_d("engine render pipeline", format!("{label} | fs={fs}"));
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
            })
        };
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
        let custom_maps_entry = wgpu::BindGroupLayoutEntry {
            binding: 15,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        };

        // A smart mask is evaluated in its own pass and cached in l_mask.
        // Its source pipeline therefore has a compact, independent layout.
        let gl_mask_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite mask source"),
            entries: &[
                gpu::uniform_entry(0, true),
                gpu::tex_entry(9, false),
                gpu::tex_entry(10, false),
                gpu::tex_entry(11, false),
                gpu::tex_entry(12, true),
                gpu::tex_entry(13, true),
                gpu::tex_entry(14, true),
                custom_maps_entry.clone(),
                gpu::tex_entry(16, true),
                gpu::sampler_entry(17),
            ],
        });

        // The OpenGL core pass composites color, roughness, metal, and height.
        // It binds 15 sampled textures at most, including the cached mask.
        let mut gl_core_entries = vec![gpu::uniform_entry(0, true)];
        for b in 1..=11 {
            gl_core_entries.push(gpu::tex_entry(b, (5..=8).contains(&b)));
        }
        gl_core_entries.push(gpu::tex_entry(13, true)); // part-random map
        gl_core_entries.push(gpu::sampler_entry(17));
        gl_core_entries.push(gpu::sampler_entry(18));
        gl_core_entries.push(gpu::tex_entry(19, true)); // opacity image/channel
        gl_core_entries.push(gpu::tex_entry(27, false)); // cached color variation
        gl_core_entries.push(gpu::tex_entry(28, false)); // cached roughness variation

        // The OpenGL extra pass handles emission, normal, and specular.
        let gl_extra_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("OpenGL composite extra"),
            entries: &[
                gpu::uniform_entry(0, true),
                gpu::tex_entry(9, false),
                gpu::tex_entry(10, false),
                gpu::tex_entry(11, false),
                gpu::sampler_entry(18),
                gpu::tex_entry(20, false),
                gpu::tex_entry(21, true),
                gpu::tex_entry(22, false),
                gpu::tex_entry(23, true),
                gpu::tex_entry(24, false),
                gpu::tex_entry(25, true),
                gpu::tex_entry(26, false),
            ],
        });

        // Folder merging reads seven prior channels, seven folder channels,
        // and the cached mask: exactly 15 sampled textures.
        let mut gl_folder_entries = vec![gpu::uniform_entry(0, true)];
        for b in 1..=9 {
            gl_folder_entries.push(gpu::tex_entry(b, false));
        }
        for b in 20..=25 {
            gl_folder_entries.push(gpu::tex_entry(b, false));
        }
        let gl_folder_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("OpenGL folder merge"),
            entries: &gl_folder_entries,
        });

        let composite_bgl = if gl_depth_color {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("OpenGL composite core"),
                entries: &gl_core_entries,
            })
        } else {
            let mut entries = vec![gpu::uniform_entry(0, true)];
            for b in 1..=11 {
                // Layer slots 5-8 also hold fill images, which are filtered.
                entries.push(gpu::tex_entry(b, (5..=8).contains(&b)));
            }
            for b in 12..=16 {
                entries.push(if b == 15 { custom_maps_entry } else { gpu::tex_entry(b, true) });
            }
            entries.push(gpu::sampler_entry(17));
            entries.push(gpu::sampler_entry(18));
            entries.push(gpu::tex_entry(19, true));
            entries.push(gpu::tex_entry(20, false));
            entries.push(gpu::tex_entry(21, true));
            entries.push(gpu::tex_entry(22, false));
            entries.push(gpu::tex_entry(23, true));
            entries.push(gpu::tex_entry(24, false));
            entries.push(gpu::tex_entry(25, true));
            entries.push(gpu::tex_entry(26, false));
            entries.push(gpu::tex_entry(27, false));
            entries.push(gpu::tex_entry(28, false));
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("composite"),
                entries: &entries,
            })
        };
        // The color/roughness FBM prepass needs only the layer uniform and position map.
        let fill_variation_noise_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fill variation noise"),
            entries: &[gpu::uniform_entry(0, true), gpu::tex_entry(10, false)],
        });
        // The variation prepass pipeline is intentionally lazy. Paint-only
        // layers and projects without Fill variation should not compile an
        // expensive FBM shader during Engine construction.
        let fill_variation_noise = std::sync::OnceLock::new();
        let composite_layout = gpu::layout(device, &composite_bgl);
        let gl_extra_layout = gpu::layout(device, &gl_extra_bgl);
        let gl_folder_layout = gpu::layout(device, &gl_folder_bgl);
        let gl_mask_layout = gpu::layout(device, &gl_mask_bgl);
        let composite = if gl_depth_color {
            // This shared core pipeline may be selected by the full seven-slot
            // accumulation pass. Keep all attachment slots in the pipeline
            // signature, while retaining the actual formats but disabling writes to the three outputs owned by the extra
            // OpenGL pass. The dedicated per-layer core pipelines below still
            // use four slots because their render passes have four attachments.
            let targets = ACCUM_FORMATS.iter().enumerate().map(|(channel, &format)| {
                gpu::target(format).map(|mut target| {
                    if channel >= 4 {
                        target.write_mask = wgpu::ColorWrites::empty();
                    }
                    target
                })
            }).collect::<Vec<_>>();
            gpu::fullscreen_pipeline(
                device, "OpenGL composite core", &composite_module, "fs_layer_gl_core",
                &composite_layout, &targets,
            )
        } else {
            gpu::fullscreen_pipeline(
                device, "composite", &composite_module, "fs_layer",
                &composite_layout, &ACCUM_FORMATS.map(gpu::target),
            )
        };
        let folder_merge = if gl_depth_color {
            gpu::fullscreen_pipeline(
                device, "OpenGL folder merge", &composite_module, "fs_folder_merge_gl",
                &gl_folder_layout, &ACCUM_FORMATS.map(gpu::target),
            )
        } else {
            gpu::fullscreen_pipeline(
                device, "folder merge", &composite_module, "fs_folder_merge",
                &composite_layout, &ACCUM_FORMATS.map(gpu::target),
            )
        };
        let mask_source = gpu::fullscreen_pipeline(
            device, "mask source", &composite_module, "fs_mask_source",
            &gl_mask_layout, &[gpu::target(MASK_FILTER_FORMAT)],
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
        let resample_nearest_id = gpu::fullscreen_pipeline(
            device,
            "ID map nearest resample",
            &resample_module,
            "fs_resample_nearest",
            &resample_layout,
            &[gpu::target(MAP_FORMATS[ID_MAP_TEXTURE])],
        );

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
            stamp_pad,
            stamp_uv,
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
            composite_layout,
            composite_variants: Default::default(),
            composite_bgl,
            gl_composite: gl_depth_color,
            gl_extra_layout,
            gl_extra_bgl,
            gl_folder_bgl,
            gl_mask_bgl,
            gl_composite_variants: Default::default(),
            composite_module,
            fill_variation_noise,
            fill_variation_noise_bgl,
            mask_source,
            mask_filter,
            mask_filter_bgl,
            resample,
            resample_nearest_id,
            resample_bgl,
            resolve,
            resolve_bgl,
        }
    }

    // ------------------------------------------------------------------ mesh

    /// Rasterizes position and normal maps for the current mesh at the
    /// current resolution, with dilation past UV island borders.
    pub fn bake_uv(&mut self, mesh: &GpuMesh) {
        self.bake_uv_internal(mesh, true);
    }

    fn bake_uv_internal(&mut self, mesh: &GpuMesh, invalidate_composite: bool) {
        self.apply_bg = None;
        if self.pos.width != self.res || self.nrm.width != self.res || self.tan.width != self.res {
            self.pos = self.new_tex("position map", self.res, self.res, POS_FORMAT, LAYER_USAGE);
            self.nrm = self.new_tex("normal map", self.res, self.res, NRM_FORMAT, LAYER_USAGE);
            self.tan = self.new_tex("tangent map", self.res, self.res, TAN_FORMAT, LAYER_USAGE);
        }
        // A mesh change can leave the range past the end; the app sets the
        // new range right after, and this draws nothing meanwhile.
        let tris = self.tris.start.min(mesh.index_count)..self.tris.end.min(mesh.index_count);
        let device = &self.device;
        let tmp_pos = Tex::new_without_srgb_view(device, "dilate pos", self.res, self.res, POS_FORMAT, LAYER_USAGE);
        let tmp_nrm = Tex::new_without_srgb_view(device, "dilate nrm", self.res, self.res, NRM_FORMAT, LAYER_USAGE);
        let tmp_tan = Tex::new_without_srgb_view(device, "dilate tan", self.res, self.res, TAN_FORMAT, LAYER_USAGE);
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
        self.uv_baked = Some(UvBakeKey { mesh_id: mesh.mesh_id, tris: self.tris.clone(), res: self.res });
        if invalidate_composite {
            self.mask_final.clear();
            self.mark_dirty();
        }
    }

    pub fn bounds(&self) -> (Vec3, Vec3) {
        self.bounds
    }

    /// The texture set's range in the mesh's index buffer.
    pub fn triangles(&self) -> std::ops::Range<u32> {
        self.tris.clone()
    }

    /// Moves the texture set to another range of the (new) mesh and remakes
    /// its position and normal maps. The bake is skipped when the maps were
    /// already baked from this same GPU mesh, range and size.
    pub fn set_triangles(&mut self, tris: std::ops::Range<u32>, mesh: &GpuMesh) {
        let needed = (tris.end - tris.start).max(3);
        if needed > self.screen_index_capacity {
            self.screen_index_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("screen stroke candidate indices"),
                size: u64::from(needed) * std::mem::size_of::<u32>() as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.screen_index_capacity = needed;
        }
        let repeat = UvBakeKey::is_current(self.uv_baked.as_ref(), mesh.mesh_id, &tris, self.res);
        self.tris = tris;
        if !repeat {
            self.bake_uv(mesh);
        } else if self.pos.width != self.res || self.nrm.width != self.res || self.tan.width != self.res {
            self.ensure_surface_maps(mesh);
        }
    }

    pub fn set_bounds(&mut self, bounds: (Vec3, Vec3)) {
        self.bounds = bounds;
        self.mask_final.clear();
        self.mark_dirty();
    }

    // ------------------------------------------------------------------ mesh maps

    /// 1x1 neutral maps used until the first bake.
    fn default_maps(device: &wgpu::Device, queue: &wgpu::Queue) -> [Tex; MAP_TEXTURES] {
        let maps = std::array::from_fn(|i| Tex::new_without_srgb_view(device, "mesh map", 1, 1, MAP_FORMATS[i], LAYER_USAGE));
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
        self.maps_version = self.maps_version.wrapping_add(1);
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
        let report = self.bake_report.as_ref()
            .map(|existing| Arc::new(existing.with_id_regions_from(&report)))
            .unwrap_or(report);
        let BakedMaps { maps, .. } = baked;
        let [_, _, _, _, id] = maps;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let id = if id.width != self.res { self.resample_id_map(&mut encoder, &id, self.res) } else { id };
        self.queue.submit([encoder.finish()]);
        self.maps[4] = id;
        self.maps_version = self.maps_version.wrapping_add(1);
        self.bake_report = Some(report);
        // The ID map is an Info map and has no paintable fix layer. Preserve
        // user-painted fixes on every other baked map during an ID-only rebake.
        self.maps_dirty = true;
        self.mark_dirty();
    }

    pub fn set_maps(&mut self, baked: BakedMaps, report: Arc<BakeReport>) {
        // Bakes stop at `bake::MAX_BAKE_RES`; bigger texture sets get them upscaled.
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let BakedMaps { maps, custom } = baked;
        let (maps, custom) = if maps[0].width < self.res {
            let up = std::array::from_fn(|index| {
                if index == ID_MAP_TEXTURE {
                    self.resample_id_map(&mut encoder, &maps[index], self.res)
                } else {
                    self.resample(&mut encoder, &maps[index], self.res)
                }
            });
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
        self.maps_version = self.maps_version.wrapping_add(1);
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
        // A newly connected output enables that channel's source material slot;
        // the project-specific mask decides whether it currently contributes.
        let mut images: [Option<FillImage>; CHANNELS] = Default::default();
        if let Content::Fill(fill) = &mut layer.content {
            for (ch, (slot, output)) in images.iter_mut().zip(outputs).enumerate() {
                if output.is_some() {
                    fill.mat.enabled[ch] = true;
                }
                *slot = output.map(|tex| FillImage {
                    tex,
                    name: name.to_string(),
                    encoded: Arc::new(Vec::new()),
                });
            }
        }
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
        let layer_id = self.layers[idx].id;
        self.invalidate_mask_cache(layer_id);
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
    fn resolve_maps(
        &mut self,
        profile_query: Option<&wgpu::QuerySet>,
        profile_stage_count: &mut usize,
        profile_labels: &mut Vec<String>,
    ) {
        self.maps_version += 1;
        let resolve_count = (0..MAP_TEXTURES).filter(|&t| self.map_needs_resolve(t)).count();
        if resolve_count == 0 {
            // No level/fix shader work is required. Drop old resolved outputs and
            // avoid submitting an otherwise empty command buffer on map invalidation.
            self.resolved.iter_mut().for_each(|resolved| *resolved = None);
            self.maps_dirty = false;
            return;
        }
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let profile_slot = if resolve_count > 0 {
            begin_composite_profile_stage(&mut encoder, profile_query, profile_stage_count)
        } else {
            None
        };
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
                self.resolved[t] = Some(self.new_tex("resolved map", res, res, MAP_FORMATS[t], LAYER_USAGE));
            }
            let out = self.resolved[t].as_ref().unwrap();
            let mut pass = gpu::pass(&mut encoder, "maps resolve", &[gpu::color_attachment(&out.view, wgpu::LoadOp::Load)]);
            pass.set_pipeline(&self.p.resolve[&MAP_FORMATS[t]]);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
        end_composite_profile_stage(
            &mut encoder,
            profile_query,
            profile_slot,
            format!("Mesh maps resolve ({resolve_count} packed textures)"),
            profile_labels,
        );
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
        let tex = self.new_tex( label, width, height, format, LAYER_USAGE);
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

    /// Reads and prepares a fill image without accessing the GPU. The returned
    /// mip pyramid can be built on a worker and uploaded later on the UI thread.
    pub(crate) fn prepare_fill_image_file(path: &std::path::Path) -> Result<PreparedFillImage, String> {
        let dir = path.parent().and_then(|d| d.file_name()).map_or(String::new(), |d| format!("{}/", d.to_string_lossy()));
        let name = format!("{dir}{}", path.file_name().map_or("image".into(), |n| n.to_string_lossy()));
        let _detail = crate::startup_trace::detail(name.clone());
        crate::startup_trace::note_decode(&name);
        let encoded = {
            let _s = crate::startup_trace::span("scan: read file");
            Arc::new(std::fs::read(path).map_err(|e| format!("can't read {}: {e}", path.display()))?)
        };
        Self::prepare_fill_image(name, encoded)
    }

    /// Decodes an image and builds all mip levels using CPU-only operations.
    pub(crate) fn prepare_fill_image(name: String, encoded: Arc<Vec<u8>>) -> Result<PreparedFillImage, String> {
        let img = {
            let _s = crate::startup_trace::span("scan: decode");
            image::load_from_memory(&encoded).map_err(|e| format!("can't read {name}: {e}"))?
        };
        let gray = matches!(img.color(), image::ColorType::L8 | image::ColorType::L16);
        let mips = if gray {
            let img = {
                let _s = crate::startup_trace::span("scan: convert");
                img.into_luma8()
            };
            let img = {
                let _s = crate::startup_trace::span("scan: resize");
                fit_fill_image(img)
            };
            let _s = crate::startup_trace::span("scan: mips");
            gpu::prepare_mips(img, F::R8Unorm)
        } else {
            let img = {
                let _s = crate::startup_trace::span("scan: convert");
                img.into_rgba8()
            };
            let img = {
                let _s = crate::startup_trace::span("scan: resize");
                fit_fill_image(img)
            };
            let _s = crate::startup_trace::span("scan: mips");
            gpu::prepare_mips(img, F::Rgba8Unorm)
        };
        Ok(PreparedFillImage { name, encoded, mips })
    }

    /// Creates the GPU texture for CPU-prepared image data. Call on the thread
    /// that owns this engine's device and queue.
    pub(crate) fn upload_prepared_fill_image(&self, prepared: PreparedFillImage) -> FillImage {
        let _detail = crate::startup_trace::detail(prepared.name.clone());
        let tex = gpu::upload_prepared_mips(&self.device, &self.queue, prepared.mips, "fill image");
        FillImage { tex: Arc::new(tex), name: prepared.name, encoded: prepared.encoded }
    }

    /// A fill image from a file, sharing the GPU texture with other layers
    /// that use the same file.
    pub fn shared_fill_image(&mut self, path: &std::path::Path) -> Result<FillImage, String> {
        let dir = path.parent().and_then(|d| d.file_name()).map_or(String::new(), |d| format!("{}/", d.to_string_lossy()));
        let name = format!("{dir}{}", path.file_name().map_or("image".into(), |n| n.to_string_lossy()));
        let cached = self
            .image_cache
            .lock()
            .map_err(|_| "fill image cache lock is poisoned")?
            .get(path)
            .and_then(|(texture, bytes)| Some((texture.upgrade()?, bytes.upgrade()?)));
        if let Some((tex, encoded)) = cached {
            return Ok(FillImage { tex, name, encoded });
        }
        let encoded = Arc::new(std::fs::read(path).map_err(|e| format!("can't read {}: {e}", path.display()))?);
        let img = self.decode_fill_image(name, encoded.clone())?;
        let mut cache = self.image_cache.lock().map_err(|_| "fill image cache lock is poisoned")?;
        cache.retain(|_, (texture, bytes)| texture.strong_count() > 0 && bytes.strong_count() > 0);
        // The cache is not an owner: once the last layer/history reference is
        // dropped, both the GPU texture and original CPU file bytes can die.
        cache.insert(path.to_path_buf(), (Arc::downgrade(&img.tex), Arc::downgrade(&img.encoded)));
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
            disabled_channels: [false; CHANNELS],
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
            disabled_channels: [false; CHANNELS],
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
            let src = self.new_tex( CH_NAMES[ch], size, size, format, LAYER_USAGE);
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
            disabled_channels: [false; CHANNELS],
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
            disabled_channels: [false; CHANNELS],
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
            let n = self.new_tex("layer", t.width, t.height, t.format, LAYER_USAGE);
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
        let disabled_channels = src.disabled_channels;
        let images = src.images.clone();
        self.queue.submit([encoder.finish()]);
        let id = self.new_id();
        let mut layer = Layer { id, name, visible, opacity, blend, content, mask, disabled_channels, parent, images, thumb: None };
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
        self.invalidate_mask_cache(id);
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
            let layer_id = self.layers[idx].id;
            self.invalidate_mask_cache(layer_id);
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
        let before = Snapshot::encode(&self.device, &mut encoder, &[tex], self.snapshot_buffer_pool.clone());
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
        let after = Snapshot::encode(&self.device, &mut encoder, &[tex], self.snapshot_buffer_pool.clone());
        self.queue.submit([encoder.finish()]);
        after.request_map();
        self.invalidate_mask_cache(layer_id);

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
        self.invalidate_mask_cache(id);
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
        self.invalidate_mask_cache(id);
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

    /// Checks property state without allocating the DocProps snapshot. This is
    /// used on idle frames; the owned snapshot is built only when a real edit
    /// must be compared or stored in Undo history.
    fn doc_props_match(&self, props: &DocProps) -> bool {
        if self.layers.len() != props.layers.len() || self.map_levels.as_slice() != props.levels.as_slice() {
            return false;
        }
        self.layers.iter().zip(&props.layers).all(|(layer, (id, saved))| {
            if layer.id != *id
                || layer.name != saved.name
                || layer.visible != saved.visible
                || layer.opacity != saved.opacity
                || layer.blend != saved.blend
            {
                return false;
            }
            let fill = match &layer.content {
                Content::Fill(fill) => Some(*fill),
                Content::Paint(_) | Content::Folder(_) => None,
            };
            if fill != saved.fill {
                return false;
            }
            let (generator, filters, mask_layers): (
                Option<Generator>,
                &[MaskFilter],
                &[MaskLayer],
            ) = match &layer.mask {
                Some(mask) => (mask.generator, &mask.filters, &mask.layers),
                None => (None, &[], &[]),
            };
            generator == saved.generator
                && filters == saved.mask_filters.as_slice()
                && mask_layers == saved.mask_layers.as_slice()
        })
    }

    /// Returns whether a property edit changes every mask input, and the layer
    /// IDs whose own mask source/filter configuration changed.
    // Kept only for the focused unit tests that assert the invalidation diff.
    // Production invalidation below removes IDs in place without allocating this Vec.
    #[cfg(test)]
    fn mask_cache_changes(before: &DocProps, after: &DocProps) -> (bool, Vec<u64>) {
        if before.levels != after.levels
            || before.layers.len() != after.layers.len()
            || before.layers.iter().zip(&after.layers).any(|(a, b)| a.0 != b.0) {
            return (true, Vec::new());
        }
        let changed = before.layers.iter().zip(&after.layers)
            .filter(|(a, b)| {
                a.1.generator != b.1.generator
                    || a.1.mask_filters != b.1.mask_filters
                    || a.1.mask_layers != b.1.mask_layers
            })
            .map(|(a, _)| a.0)
            .collect();
        (false, changed)
    }

    /// Invalidates mask results directly from the live Engine state. Unlike
    /// doc_props(), this needs no temporary copy of every layer's properties.
    fn apply_live_mask_cache_changes(
        mask_final: &mut HashSet<u64>,
        layers: &[Layer],
        levels: &[Levels],
        before: &DocProps,
    ) {
        if levels != before.levels.as_slice()
            || layers.len() != before.layers.len()
            || layers.iter().zip(&before.layers).any(|(layer, (id, _))| layer.id != *id)
        {
            mask_final.clear();
            return;
        }
        for (layer, (id, saved)) in layers.iter().zip(&before.layers) {
            let (generator, filters, mask_layers): (
                Option<Generator>,
                &[MaskFilter],
                &[MaskLayer],
            ) = match &layer.mask {
                Some(mask) => (mask.generator, &mask.filters, &mask.layers),
                None => (None, &[], &[]),
            };
            if generator != saved.generator
                || filters != saved.mask_filters.as_slice()
                || mask_layers != saved.mask_layers.as_slice()
            {
                mask_final.remove(id);
            }
        }
    }

    fn apply_mask_cache_changes(mask_final: &mut HashSet<u64>, before: &DocProps, after: &DocProps) {
        if before.levels != after.levels
            || before.layers.len() != after.layers.len()
            || before.layers.iter().zip(&after.layers).any(|(a, b)| a.0 != b.0) {
            mask_final.clear();
            return;
        }
        // Invalidate only affected IDs in place. A temporary Vec<u64> of all
        // changed masks is unnecessary even when a document has many layers.
        for ((before_id, before), (after_id, after)) in before.layers.iter().zip(&after.layers) {
            if before_id == after_id
                && (before.generator != after.generator
                    || before.mask_filters != after.mask_filters
                    || before.mask_layers != after.mask_layers)
            {
                mask_final.remove(before_id);
            }
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
        // This runs on idle frames. Avoid constructing a full cloned
        // DocProps (names, filters and mask layers) when nothing changed.
        if self.doc_props_match(&self.committed) {
            return;
        }
        let current = self.doc_props();
        Self::apply_mask_cache_changes(&mut self.mask_final, &self.committed, &current);
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
                if e.target == StrokeTarget::Mask { self.invalidate_mask_cache(e.layer_id); }
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
                self.invalidate_mask_cache(*id);
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
                let updated = self.doc_props();
                Self::apply_mask_cache_changes(&mut self.mask_final, &current, &updated);
                if current.levels != updated.levels { self.maps_dirty = true; }
                *props = current;
            }
            Edit::Fix { map, tex } => {
                std::mem::swap(&mut self.map_fix[*map], tex);
                self.maps_dirty = true;
                self.mask_final.clear();
            }
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
        let tex = self.new_tex("mask", self.res, self.res, MASK_FORMAT, LAYER_USAGE);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let v = if white { 1.0 } else { 0.0 };
        gpu::clear_texture(&mut encoder, &tex, wgpu::Color { r: v, g: v, b: v, a: 1.0 });
        self.queue.submit([encoder.finish()]);
        tex
    }

    fn ensure_channel(&mut self, idx: usize, ch: usize) {
        let res = self.res;
        match &self.layers[idx].content {
            Content::Paint(chans) if chans[ch].is_none() => {}
            _ => return,
        }
        let tex = self.new_tex(CH_NAMES[ch], res, res, PAINT_FORMATS[ch], LAYER_USAGE);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        gpu::clear_texture(&mut encoder, &tex, wgpu::Color::TRANSPARENT);
        self.queue.submit([encoder.finish()]);
        let layer = &mut self.layers[idx];
        let Content::Paint(chans) = &mut layer.content else { return };
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
            let tex = self.new_tex("map fix", self.res, self.res, PAINT_FORMATS[Self::FIX_CHANNEL], LAYER_USAGE);
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
            return self.encode_stroke_start(0, target, vec![Self::FIX_CHANNEL]);
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
        // Texture creation, stroke-mask clearing, and Undo readback are deferred
        // until the first paint pass has valid stamps and non-empty UV scissors.
        // A click/miss therefore performs no snapshot submit and allocates no
        // previously-unused channel or mask texture.
        self.encode_stroke_start(layer_id, target, channels)
    }

    /// Starts the logical stroke only. GPU resources and the before-snapshot
    /// are prepared lazily in the first valid paint encoder.
    fn encode_stroke_start(&mut self, layer_id: u64, target: StrokeTarget, channels: Vec<usize>) -> bool {
        self.stamp_bg = [None, None];
        self.apply_bg = None;
        self.active = Some(ActiveStroke { layer_id, target, channels, before: None });
        true
    }

    /// Encodes the initial stroke clear, target-to-snapshot copies and Undo
    /// readback into the same command buffer as the first real paint pass.
    /// Returns the pending before-snapshot, which must be installed and mapped
    /// only after that command buffer has been submitted.
    fn encode_lazy_stroke_start(&mut self, encoder: &mut wgpu::CommandEncoder) -> Option<Snapshot> {
        let (layer_id, target, channels) = self.active.as_ref()
            .filter(|active| active.before.is_none())
            .map(|active| (active.layer_id, active.target, active.channels.clone()))?;

        match target {
            StrokeTarget::Content => {
                let idx = self.layers.iter().position(|layer| layer.id == layer_id)
                    .expect("active paint layer still exists");
                for &channel in &channels {
                    self.ensure_channel(idx, channel);
                    if self.snap[channel].is_none() {
                        self.snap[channel] = Some(self.new_tex("snapshot", self.res, self.res, PAINT_FORMATS[channel], LAYER_USAGE,
                        ));
                    }
                }
            }
            StrokeTarget::Mask => {
                let idx = self.layers.iter().position(|layer| layer.id == layer_id)
                    .expect("active mask layer still exists");
                self.ensure_mask_paint(idx);
                if self.snap_mask.is_none() {
                    self.snap_mask = Some(self.new_tex("mask snapshot", self.res, self.res, MASK_FORMAT, LAYER_USAGE,
                    ));
                }
            }
            StrokeTarget::BakeFix(_) => {
                let channel = Self::FIX_CHANNEL;
                if self.snap[channel].is_none() {
                    self.snap[channel] = Some(self.new_tex("snapshot", self.res, self.res, PAINT_FORMATS[channel], LAYER_USAGE,
                    ));
                }
            }
        }

        self.ensure_stroke_target();
        gpu::clear_texture(encoder, &self.stroke, wgpu::Color::TRANSPARENT);
        let textures = self.target_textures(layer_id, target, &channels).expect("stroke target exists");
        let snaps: Vec<&Tex> = match target {
            StrokeTarget::Mask => vec![self.snap_mask.as_ref().expect("mask snapshot was allocated")],
            _ => channels.iter().map(|&channel| self.snap[channel].as_ref().expect("channel snapshot was allocated")).collect(),
        };
        for (source, snapshot) in textures.iter().zip(&snaps) {
            encoder.copy_texture_to_texture(source.copy_info(), snapshot.copy_info(), source.extent());
        }
        Some(Snapshot::encode(&self.device, encoder, &textures, self.snapshot_buffer_pool.clone()))
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
    /// A conservative UV `region` clips the stamp/apply passes to the texels
    /// the brush can reach, including the mesh-map dilation halo. `None`
    /// means the bounds are unknown and the pass must cover the full texture;
    /// an empty region means no texel can be affected this frame.
    pub fn stroke_step(
        &mut self,
        stamps: &[Stamp],
        su: StampUniform,
        depth: &wgpu::TextureView,
        paint: &BrushPaint,
        regions: Option<Vec<[f32; 4]>>,
    ) {
        self.stroke_step_inner(stamps, su, depth, paint, regions, false, None, None);
    }

    /// Same paint path, with optional GPU timestamps for F6.
    pub fn stroke_step_profiled(
        &mut self,
        stamps: &[Stamp],
        su: StampUniform,
        depth: &wgpu::TextureView,
        paint: &BrushPaint,
        regions: Option<Vec<[f32; 4]>>,
    ) {
        self.stroke_step_inner(stamps, su, depth, paint, regions, true, None, None);
    }

    /// Screen projection using only triangles that can overlap the stamp's
    /// screen-space bounds. Their vertex indices were collected from the mesh BVH.
    pub fn stroke_step_screen_culled(
        &mut self,
        mesh: &GpuMesh,
        candidate_indices: &[u32],
        stamps: &[Stamp],
        su: StampUniform,
        depth: &wgpu::TextureView,
        paint: &BrushPaint,
        regions: Option<Vec<[f32; 4]>>,
    ) {
        self.stroke_step_inner(stamps, su, depth, paint, regions, false, Some(mesh), Some(candidate_indices));
    }

    /// Profiled version of culling-aware per-triangle screen projection.
    pub fn stroke_step_screen_culled_profiled(
        &mut self,
        mesh: &GpuMesh,
        candidate_indices: &[u32],
        stamps: &[Stamp],
        su: StampUniform,
        depth: &wgpu::TextureView,
        paint: &BrushPaint,
        regions: Option<Vec<[f32; 4]>>,
    ) {
        self.stroke_step_inner(stamps, su, depth, paint, regions, true, Some(mesh), Some(candidate_indices));
    }

    fn stroke_step_inner(
        &mut self,
        stamps: &[Stamp],
        mut su: StampUniform,
        depth: &wgpu::TextureView,
        paint: &BrushPaint,
        regions: Option<Vec<[f32; 4]>>,
        profile_gpu: bool,
        screen_mesh: Option<&GpuMesh>,
        candidate_indices: Option<&[u32]>,
    ) {
        if self.active.is_none() || stamps.is_empty() {
            return;
        }
        if candidate_indices.is_some_and(|indices| indices.is_empty()) {
            return;
        }
        // The per-frame bounds are conservative paint raster bounds. Keep
        // them separate from the wider accumulated bounds used by compositing.
        let frame_scissors = regions.as_deref().and_then(|r| normalized_scissors(r, self.res));
        // Invalid normalized bounds are not safe to scissor; force the
        // accumulated composite path to full-frame too.
        let regions = frame_scissors.as_ref().and(regions);
        let full_area = u64::from(self.res) * u64::from(self.res);
        self.last_paint_scissor_pixels = frame_scissors.as_ref().map_or(full_area, |rects| rects.iter().map(|&r| scissor_area(r)).sum());
        self.last_paint_scissor_rects = frame_scissors.as_ref().map_or(1, |rects| rects.len() as u32);
        if frame_scissors.as_ref().is_some_and(|rects| rects.is_empty()) {
            // A known empty region means no texel can change; avoid both GPU
            // passes and a pointless full composite.
            return;
        }
        self.add_regions(regions);
        let stamps = &stamps[..stamps.len().min(MAX_STAMPS)];
        su.view[3] = stamps.len() as f32;
        if let Some(indices) = candidate_indices {
            debug_assert!(
                indices.len() <= self.screen_index_capacity as usize,
                "candidate index list cannot exceed the screen index buffer capacity",
            );
            self.queue.write_buffer(&self.screen_index_buffer, 0, bytemuck::cast_slice(indices));
        }
        self.queue.write_buffer(&self.stamp_ubuf, 0, bytemuck::bytes_of(&su));
        self.queue.write_buffer(&self.stamp_sbuf, 0, bytemuck::cast_slice(stamps));

        let device = &self.device;
        let stamp_bg_slot = usize::from(screen_mesh.is_some());
        if self.stamp_bg[stamp_bg_slot].is_none() {
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            self.stamp_bg[stamp_bg_slot] = Some(Arc::new(group));
        }
        let stamp_bg = self.stamp_bg[stamp_bg_slot].as_ref().expect("created above").clone();
        let profile = if profile_gpu && self.active_gpu_profile.is_none() {
            self.gpu_profiler.as_ref().and_then(GpuStrokeProfiler::resources)
        } else {
            None
        };
        if let Some(p) = &profile {
            self.active_gpu_profile = Some(p.clone());
        }
        // A parked Engine may still have its 1x1 placeholder stroke target.
        // Restore it before taking the attachment view for the first paint pass.
        let mut encoder = device.create_command_encoder(&Default::default());
        let before = self.encode_lazy_stroke_start(&mut encoder);
        self.ensure_stroke_target();
        let stamp_attachments = [gpu::color_attachment(&self.stroke.view, wgpu::LoadOp::Load)];
        if let Some(mesh) = screen_mesh {
            // Keep UV geometry and padding in one render pass. Both write the
            // same stroke target and padding excludes original island texels,
            // so switching pipelines avoids a second pass setup without changing
            // the order or overlap semantics.
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("screen stamp UV geometry and padding"),
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

            // First rasterize every face at its own UV coordinates.
            pass.set_pipeline(&self.p.stamp_uv);
            pass.set_bind_group(0, &*stamp_bg, &[]);
            pass.set_vertex_buffer(0, mesh.vertices.slice(..));
            let draw_range = if let Some(indices) = candidate_indices {
                pass.set_index_buffer(self.screen_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                0..indices.len() as u32
            } else {
                pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
                self.tris.clone()
            };
            if let Some(rects) = &frame_scissors {
                for &[x0, y0, x1, y1] in rects {
                    pass.set_scissor_rect(x0, y0, x1 - x0, y1 - y0);
                    pass.draw_indexed(draw_range.clone(), 0, 0..1);
                }
            } else {
                pass.draw_indexed(draw_range, 0, 0..1);
            }

            // Then fill only dilated padding texels, without touching island pixels.
            pass.set_pipeline(&self.p.stamp_pad);
            if let Some(rects) = &frame_scissors {
                for &[x0, y0, x1, y1] in rects {
                    pass.set_scissor_rect(x0, y0, x1 - x0, y1 - y0);
                    pass.draw(0..3, 0..1);
                }
            } else {
                pass.draw(0..3, 0..1);
            }
        } else {
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
            pass.set_bind_group(0, &*stamp_bg, &[]);
            if let Some(rects) = &frame_scissors {
                for &[x0, y0, x1, y1] in rects {
                    pass.set_scissor_rect(x0, y0, x1 - x0, y1 - y0);
                    pass.draw(0..3, 0..1);
                }
            } else {
                pass.draw(0..3, 0..1);
            }
        }
        self.encode_apply(&mut encoder, paint, profile.as_ref(), frame_scissors.as_deref());
        self.queue.submit([encoder.finish()]);
        if let Some(before) = before {
            before.request_map();
            if let Some(active) = &mut self.active {
                active.before = Some(before);
            }
        }
    }

    /// Returns completed GPU timings for stamp and apply; does not wait for the GPU.
    pub fn poll_gpu_stroke_times(
        &mut self,
    ) -> Option<(bool, f64, f64, f64, Option<f64>, Option<f64>, Option<f64>, Vec<(String, f64)>)> {
        let (profile_result, still_pending) = {
            let profiler = self.gpu_profiler.as_mut()?;
            let result = profiler.poll(&self.device, &self.queue);
            (result, profiler.pending)
        };
        let Some((stamp, apply, layer, total, clear, body, pass_ms)) = profile_result else {
            // A failed readback ends the request too. Do not let a stale F5
            // marker turn a later brush-stroke profile into an F5 capture.
            if !still_pending && self.active_gpu_profile.is_none() && !self.composite_profile_requested {
                self.forced_composite_profile = false;
                self.last_profiled_pass_labels.clear();
            }
            return None;
        };
        let forced_composite = std::mem::take(&mut self.forced_composite_profile);
        let pass_labels = std::mem::take(&mut self.last_profiled_pass_labels);
        let pass_times = pass_labels.into_iter().zip(pass_ms).collect();
        Some((forced_composite, stamp, apply, layer, total, clear, body, pass_times))
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
        self.last_paint_scissor_pixels = u64::from(self.res) * u64::from(self.res);
        self.last_paint_scissor_rects = 1;
        self.add_regions(Some(vec![[lo[0], lo[1], hi[0], hi[1]]]));
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
        let before = self.encode_lazy_stroke_start(&mut encoder);
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
        self.encode_apply(&mut encoder, paint, None, None);
        self.queue.submit([encoder.finish()]);
        if let Some(before) = before {
            before.request_map();
            if let Some(active) = &mut self.active {
                active.before = Some(before);
            }
        }
    }

    /// Brush grain (`size`² grayscale, tileable), or `None` for no grain.
    pub fn set_brush_grain(&mut self, grain: Option<(&[u8], u32)>) {
        // The cached apply bind group holds the current grain texture view.
        self.apply_bg = None;
        let (pixels, size) = grain.unwrap_or((&[255], 1));
        if self.brush_grain.width != size {
            self.brush_grain = self.new_tex("brush grain", size, size, F::R8Unorm, LAYER_USAGE);
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
            self.stamp_bg = [None, None];
            self.brush_alpha = self.new_tex("brush alpha", size, size, F::R8Unorm, LAYER_USAGE);
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
        scissors: Option<&[[u32; 4]]>,
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
        if self.apply_bg.is_none() {
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
            self.apply_bg = Some(apply_bg);
        }
        let apply_bg = self.apply_bg.as_ref().expect("created above");

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
            pass.set_bind_group(0, apply_bg, &[]);
            if let Some(rects) = scissors {
                for &[x0, y0, x1, y1] in rects {
                    pass.set_scissor_rect(x0, y0, x1 - x0, y1 - y0);
                    pass.draw(0..3, 0..1);
                }
            } else {
                pass.draw(0..3, 0..1);
            }
        }
        if matches!(target, StrokeTarget::BakeFix(_)) {
            self.maps_dirty = true;
            self.mark_dirty();
        } else {
            // Painting a mask changes its filtered result, but ordinary layer
            // painting does not change any mask source or filter inputs.
            if target == StrokeTarget::Mask {
                self.mask_final.remove(&layer_id);
            }
            // Only this layer changed: the next composite can start at its pass.
            if self.stroke_layer.is_some_and(|id| id != layer_id) {
                self.full_dirty = true;
            }
            self.stroke_layer = Some(layer_id);
            self.dirty = true;
        }
    }

    /// Grows the changed UV area by the position-map dilation margin.
    /// Rectangles stay separate unless they overlap or the draw-count cap
    /// requires a conservative merge.
    fn add_regions(&mut self, regions: Option<Vec<[f32; 4]>>) {
        let Some(regions) = regions else {
            self.region_unknown = true;
            return;
        };
        let res = self.res as f32;
        let margin = (DILATE_PASSES + 2) as f32;
        let lo = |v: f32| (v * res - margin).floor().clamp(0.0, res) as u32;
        let hi = |v: f32| (v * res + margin).ceil().clamp(0.0, res) as u32;
        for [u0, v0, u1, v1] in regions {
            let rect = [lo(u0), lo(v0), hi(u1), hi(v1)];
            if rect[2] > rect[0] && rect[3] > rect[1] {
                self.stroke_regions.push(rect);
            }
        }
        compact_scissors(&mut self.stroke_regions);
    }

    /// The layer's mask has filters or extra mask layers that need a cached result.
    fn has_mask_filters(&self, li: usize) -> bool {
        let l = &self.layers[li];
        l.mask.as_ref().is_some_and(|m| m.filters.iter().any(|f| f.enabled) || m.layers.iter().any(|ml| ml.enabled))
    }

    /// A smart generator is expensive per texel and independent of the material
    /// channels below it, so cache its full mask result between composites.
    fn needs_mask_cache(&self, li: usize) -> bool {
        self.has_mask_filters(li)
            || self.layers[li].mask.as_ref().is_some_and(|mask| mask.generator.is_some())
    }

    /// Renders a cached mask source with no post-source filters.
    fn encode_mask_source<'a>(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        layer_name: &str,
        source_bg: &wgpu::BindGroup,
        source_offset: u32,
        result: &'a Tex,
        scissors: Option<&[[u32; 4]]>,
        profile_query: Option<&wgpu::QuerySet>,
        profile_stage_count: &mut usize,
        profile_labels: &mut Vec<String>,
    ) -> &'a Tex {
        let load = if scissors.is_some() {
            wgpu::LoadOp::Load
        } else {
            wgpu::LoadOp::Clear(wgpu::Color::BLACK)
        };
        let profile_slot = begin_composite_profile_stage(encoder, profile_query, profile_stage_count);
        let mut pass = gpu::pass(
            encoder,
            "cached smart mask",
            &[gpu::color_attachment(&result.view, load)],
        );
        pass.set_pipeline(&self.p.mask_source);
        pass.set_bind_group(0, source_bg, &[source_offset]);
        if let Some(rects) = scissors {
            for &[x0, y0, x1, y1] in rects {
                pass.set_scissor_rect(x0, y0, x1 - x0, y1 - y0);
                pass.draw(0..3, 0..1);
            }
        } else {
            pass.draw(0..3, 0..1);
        }
        drop(pass);
        end_composite_profile_stage(
            encoder,
            profile_query,
            profile_slot,
            format!("Mask generator · {layer_name}"),
            profile_labels,
        );
        result
    }

    /// Renders a layer's mask (painting times smart mask) and runs its
    /// filter steps over it using engine-shared scratch textures.
    fn encode_mask_filters(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        layer_id: u64,
        layer_name: &str,
        source_bg: &wgpu::BindGroup,
        source_offset: u32,
        ops: &[MaskOp],
        op_labels: &[String],
        scissors: Option<&[[u32; 4]]>,
        profile_query: Option<&wgpu::QuerySet>,
        profile_stage_count: &mut usize,
        profile_labels: &mut Vec<String>,
    ) -> &Tex {
        // Scratch 0 and 1 ping-pong; scratch 2 holds a mask layer's own
        // values. The per-mask result texture is kept separate from all scratch.
        let scratch = self.mask_scratch.as_ref().expect("scratch allocated for filtered masks");
        let result = &self.mask_out[&layer_id];
        let load = if scissors.is_some() { wgpu::LoadOp::Load } else { wgpu::LoadOp::Clear(wgpu::Color::BLACK) };
        let steps = ops.len();
        // A patch can't use a ping-pong buffer's leftovers outside it: step k
        // reads its input around every texel it writes, and that input texture
        // is overwritten by later steps. So each step writes only as far as the
        // next one reads: the last over `scissors`, each earlier one wider by
        // the reach of the step after it. Every read then lands on texels
        // written in this same frame, and the result texture keeps its old
        // values outside `scissors`.
        let regions: Option<Vec<Vec<[u32; 4]>>> = scissors.map(|rects| {
            let mut per_step = vec![Vec::new(); steps + 1];
            let mut region = rects.to_vec();
            for k in (0..steps).rev() {
                per_step[k + 1] = region.clone();
                if let MaskOp::Filter { reach, wrap, .. } = ops[k]
                    && reach > 0.0 {
                        expand_scissors(&mut region, reach.ceil() as u32 + 2, self.res, wrap);
                    }
            }
            per_step[0] = region;
            per_step
        });
        let draw = |pass: &mut wgpu::RenderPass<'_>, rects: Option<&[[u32; 4]]>| {
            if let Some(rects) = rects {
                for &[x0, y0, x1, y1] in rects {
                    pass.set_scissor_rect(x0, y0, x1 - x0, y1 - y0);
                    pass.draw(0..3, 0..1);
                }
            } else {
                pass.draw(0..3, 0..1);
            }
        };
        let source_out = if steps == 0 { result } else { &scratch[0] };
        let source_label = if profile_query.is_some() { format!("Mask source · {layer_name}") } else { String::new() };
        {
            let profile_slot = begin_composite_profile_stage(encoder, profile_query, profile_stage_count);
            let mut pass = gpu::pass(encoder, "mask source", &[gpu::color_attachment(&source_out.view, load)]);
            pass.set_pipeline(&self.p.mask_source);
            pass.set_bind_group(0, source_bg, &[source_offset]);
            draw(&mut pass, regions.as_ref().map(|r| &r[0][..]));
            drop(pass);
            end_composite_profile_stage(encoder, profile_query, profile_slot, source_label, profile_labels);
        }
        // Filter inputs alternate between scratch 0 and 1. Reuse one bind group
        // per input for the whole stack instead of rebuilding one for every step.
        let mut filter_bgs: [Option<wgpu::BindGroup>; 2] = [None, None];
        let mut filter = |encoder: &mut wgpu::CommandEncoder, step: usize, input_slot: usize, out: &Tex, rects: Option<&[[u32; 4]]>, label: String, profile_stage_count: &mut usize, profile_labels: &mut Vec<String>| {
            if filter_bgs[input_slot].is_none() {
                filter_bgs[input_slot] = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&scratch[input_slot].view) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.repeat_sampler) },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&scratch[2].view) },
                    ],
                }));
            }
            let bg = filter_bgs[input_slot].as_ref().expect("mask filter bind group created");
            let profile_slot = begin_composite_profile_stage(encoder, profile_query, profile_stage_count);
            let mut pass = gpu::pass(encoder, "mask filter", &[gpu::color_attachment(&out.view, load)]);
            pass.set_pipeline(&self.p.mask_filter);
            let offset = u32::try_from(step as u64 * MASK_FILTER_STRIDE).expect("mask filter uniform offset fits in u32");
            pass.set_bind_group(0, bg, &[offset]);
            draw(&mut pass, rects);
            drop(pass);
            end_composite_profile_stage(encoder, profile_query, profile_slot, label, profile_labels);
        };
        let mut cur = 0;
        for (k, op) in ops.iter().enumerate() {
            let last = k + 1 == steps;
            let out = if last { result } else { &scratch[1 - cur] };
            let rects = regions.as_ref().map(|r| &r[k + 1][..]);
            let op_label = if profile_query.is_some() {
                op_labels.get(k).cloned().unwrap_or_else(|| format!("Mask op {}", k + 1))
            } else {
                String::new()
            };
            match *op {
                MaskOp::Filter { step, .. } => filter(encoder, step, cur, out, rects, op_label, profile_stage_count, profile_labels),
                MaskOp::Layer { uniform, combine } => {
                    // The mask layer's own values, then combined over the mask so far.
                    {
                        let profile_slot = begin_composite_profile_stage(encoder, profile_query, profile_stage_count);
                        let mut pass = gpu::pass(encoder, "mask layer", &[gpu::color_attachment(&scratch[2].view, load)]);
                        pass.set_pipeline(&self.p.mask_source);
                        pass.set_bind_group(0, source_bg, &[uniform]);
                        draw(&mut pass, rects);
                        drop(pass);
                        let label = if profile_query.is_some() { format!("Mask layer source · {op_label}") } else { String::new() };
                        end_composite_profile_stage(encoder, profile_query, profile_slot, label, profile_labels);
                    }
                    let label = if profile_query.is_some() { format!("Mask layer combine · {op_label}") } else { String::new() };
                    filter(encoder, combine, cur, out, rects, label, profile_stage_count, profile_labels);
                }
            }
            if !last { cur = 1 - cur; }
        }
        result
    }

    fn invalidate_mask_cache(&mut self, layer_id: u64) {
        self.mask_final.remove(&layer_id);
    }

    /// Flags the layer stack for a full composite. Keep filtered mask results
    /// when only material properties changed; invalidate the exact mask inputs
    /// that changed, or all masks when mesh-map levels changed.
    pub fn mark_dirty(&mut self) {
        // Most dirty marks are paint/composite changes, not property edits.
        // For property changes, invalidate masks from live fields directly;
        // do not build a temporary DocProps snapshot just to discard it.
        Self::apply_live_mask_cache_changes(
            &mut self.mask_final,
            &self.layers,
            &self.map_levels,
            &self.committed,
        );
        if self.maps_dirty { self.mask_final.clear(); }
        self.dirty = true;
        self.full_dirty = true;
    }

    /// Finishes the stroke and returns its undo record.
    pub fn end_stroke(&mut self) -> Option<UndoEntry> {
        self.last_end_stages_ms = [0.0; 5];
        let mut stages = [0.0; 5];
        let phase = std::time::Instant::now();

        self.apply_bg = None;
        self.stamp_bg = [None, None];
        // Keep the prior composite prefix and any uncomposited dirty regions
        // across release. A following stroke on the same layer can resume it;
        // composite rejects it when the layer, pass list, or dirty state differs.
        // Keep the full-resolution prefix textures reusable between strokes.
        // The next stroke rewrites their contents before reading them, so the
        // old prefix itself is discarded above. At 4K the retained seven-target
        // cache would cost roughly 272 MiB, so release it instead of pinning
        // that much VRAM in the common idle state.
        const MAX_RETAINED_PREFIX_CACHE_BYTES: u64 = 96 * 1024 * 1024;
        let prefix_cache_bytes = u64::from(self.res)
            * u64::from(self.res)
            * ACCUM_FORMATS.iter().map(|&format| u64::from(gpu::bytes_per_texel(format))).sum::<u64>();
        if prefix_cache_bytes > MAX_RETAINED_PREFIX_CACHE_BYTES {
            self.prefix_cache = None;
            self.prefix = None;
        }
        let Some(active) = self.active.take() else {
            self.invalidate_thumbs();
            stages[0] = phase.elapsed().as_secs_f64() * 1000.0;
            self.last_end_stages_ms = stages;
            return None;
        };
        // Empty/missed strokes never prepare a before-snapshot. They do not
        // generate an after readback or a no-op entry in Undo history.
        let Some(before) = active.before else {
            stages[0] = phase.elapsed().as_secs_f64() * 1000.0;
            self.last_end_stages_ms = stages;
            return None;
        };
        // Content strokes only target Paint layers; their contents are not
        // shown in layer thumbnails. Avoid scanning and hashing the whole stack
        // at release when the stroke cannot have invalidated a thumbnail.
        let refresh_thumbnails = match active.target {
            StrokeTarget::Mask => {
                self.invalidate_mask_thumb(active.layer_id);
                true
            }
            StrokeTarget::BakeFix(_) => {
                self.invalidate_thumbs();
                true
            }
            StrokeTarget::Content => false,
        };
        stages[0] = phase.elapsed().as_secs_f64() * 1000.0;

        let phase = std::time::Instant::now();
        let Some(textures) = self.target_textures(active.layer_id, active.target, &active.channels) else {
            stages[1] = phase.elapsed().as_secs_f64() * 1000.0;
            self.last_end_stages_ms = stages;
            return None;
        };
        stages[1] = phase.elapsed().as_secs_f64() * 1000.0;

        let phase = std::time::Instant::now();
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let after = Snapshot::encode(&self.device, &mut encoder, &textures, self.snapshot_buffer_pool.clone());
        stages[2] = phase.elapsed().as_secs_f64() * 1000.0;
        drop(textures);

        let phase = std::time::Instant::now();
        self.queue.submit([encoder.finish()]);
        after.request_map();
        stages[3] = phase.elapsed().as_secs_f64() * 1000.0;

        let phase = std::time::Instant::now();
        // Refresh only when this stroke actually invalidated a thumbnail.
        if refresh_thumbnails {
            self.update_thumbs();
        }
        stages[4] = phase.elapsed().as_secs_f64() * 1000.0;
        self.last_end_stages_ms = stages;

        Some(UndoEntry {
            layer_id: active.layer_id,
            target: active.target,
            channels: active.channels,
            before,
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
        self.mark_dirty();
    }

    // ------------------------------------------------------------------ compositing

    fn layer_uniform(&self, layer: &Layer) -> LayerUniform {
        let mut u: LayerUniform = bytemuck::Zeroable::zeroed();
        let flags = |b: [bool; CHANNELS]| b.map(|v| if v { 1.0 } else { 0.0 });
        let enabled = |ch: usize| !layer.disabled_channels[ch];
        let (kind, ch) = match &layer.content {
            Content::Paint(ch) => (
                0.0,
                flags(std::array::from_fn(|i| ch[i].is_some() && enabled(i))),
            ),
            Content::Fill(f) => {
                let m = &f.mat;
                u.fill_color = [m.color[0], m.color[1], m.color[2], m.opacity];
                u.fill_rmh = [m.rough, m.metal, m.height, 0.0];
                u.var0 = [f.color_var, f.var_scale, f.rough_var, f.height_var];
                let mut bits = (0..4)
                    .filter(|&i| enabled(i) && layer.images[i].is_some())
                    .fold(0u32, |b, i| b | 1 << i);
                // A grayscale color image is single-channel: the shader spreads it to rgb.
                if enabled(0) && layer.images[0].as_ref().is_some_and(|im| im.tex.format == F::R8Unorm) {
                    bits |= 16;
                }
                if enabled(OPACITY) && layer.images[OPACITY].is_some() {
                    bits |= 32;
                }
                if enabled(EMISSION) && layer.images[EMISSION].is_some() {
                    bits |= 64;
                }
                if enabled(NORMAL) && layer.images[NORMAL].is_some() {
                    bits |= 128;
                }
                if enabled(SPECULAR) && layer.images[SPECULAR].is_some() {
                    bits |= 256;
                }
                u.emis = [m.emission[0], m.emission[1], m.emission[2], 0.0];
                u.ns = [m.normal, 0.0, m.specular, 0.0];
                u.var1 = [f.seed, f.part_var, f.projection as u32 as f32, bits as f32];
                u.img = [f.tiling, f.rotation, f.offset[0], f.offset[1]];
                u.gx[2] = f.tri_sharpness;
                (1.0, flags(std::array::from_fn(|i| m.enabled[i] && enabled(i))))
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
        u.gx = [self.edge_max, self.res as f32, 0.0, 0.0];
        u
    }

    /// Arm one full-stack GPU timestamp capture for the next composite.
    /// Returns false if timestamp queries are unsupported or another capture is active.
    pub fn request_composite_profile(&mut self) -> bool {
        let Some(profiler) = self.gpu_profiler.as_ref() else { return false; };
        if !profiler.encoder_timestamps || profiler.pending || self.active_gpu_profile.is_some() || self.active.is_some() {
            return false;
        }
        self.composite_profile_requested = true;
        self.forced_composite_profile = true;
        true
    }

    /// Re-composites the whole stack. Cheap enough to run every painting frame:
    /// one fullscreen pass per visible layer, plus one per faded or masked folder.
    /// Waits until the GPU has finished everything submitted so far.
    pub fn wait_idle(&self) {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    pub fn composite(&mut self) {
        if self.composite_profile_requested {
            self.composite_profile_requested = false;
            if self.active_gpu_profile.is_none() {
                self.active_gpu_profile = self.gpu_profiler.as_ref()
                    .filter(|profiler| profiler.encoder_timestamps)
                    .and_then(GpuStrokeProfiler::resources);
            }
            if self.active_gpu_profile.is_none() {
                self.forced_composite_profile = false;
            }
        }
        let mut composite_profiled = self.active_gpu_profile.is_some();
        let profile_query = self.active_gpu_profile.as_ref()
            .filter(|_| self.gpu_profiler.as_ref().is_some_and(|profiler| profiler.encoder_timestamps))
            .map(|profile| profile.query_set.clone());
        let mut profile_stage_count = 0usize;
        let mut profile_labels = Vec::new();
        if composite_profiled {
            self.last_profiled_pass_labels.clear();
        }

        self.ensure_composite_working_set();
        if self.maps_dirty {
            self.resolve_maps(profile_query.as_deref(), &mut profile_stage_count, &mut profile_labels);
        }
        let (passes, max_depth) = composite_passes_with_scratch(&self.layers, &mut self.composite_scratch);
        self.last_composite_layer_passes = passes.iter().filter(|p| matches!(**p, Pass::Layer(_))).count() as u32;
        self.last_composite_folder_passes = passes.iter().filter(|p| matches!(**p, Pass::Begin | Pass::End(_))).count() as u32;
        let (mut mask_cache_hits, mut mask_cache_misses, mut mask_gpu_passes) = (0u32, 0u32, 0u32);
        self.last_mask_cache_hits = 0;
        self.last_mask_cache_misses = 0;
        self.last_mask_gpu_passes = 0;
        self.folder_below.retain(|set| set[0].width == self.res);
        let retained_depth = retained_folder_depth(self.folder_below.len(), max_depth);
        self.folder_below.truncate(retained_depth);
        while self.folder_below.len() < retained_depth {
            let set = ACCUM_FORMATS.map(|f| self.new_tex("folder below", self.res, self.res, f, LAYER_USAGE));
            self.folder_below.push(set);
        }

        // Cache color and roughness FBM together. Both depend only on the
        // position map, variation scale and seed; channel amounts and per-part
        // variation remain cheap controls in the main layer pass.
        let mut variation_noise_needed = HashSet::with_capacity(passes.len());
        let mut variation_noise_misses: Vec<(usize, usize, u64, bool)> = Vec::with_capacity(passes.len().saturating_mul(2));
        let mut variation_noise_hits = 0u32;
        for (slot, pass) in passes.iter().enumerate() {
            let Pass::Layer(li) = *pass else { continue };
            let layer = &self.layers[li];
            let Content::Fill(fill) = &layer.content else { continue };
            #[cfg(test)]
            if self.force_inline_fill_variation {
                continue;
            }
            let color_enabled = fill.color_var != 0.0 && fill.mat.enabled[0] && !layer.disabled_channels[0];
            let roughness_enabled = fill.rough_var != 0.0 && fill.mat.enabled[1] && !layer.disabled_channels[1];
            if !color_enabled && !roughness_enabled {
                continue;
            }
            let key = FillVariationNoiseKey {
                res: self.res,
                uv_version: self.uv_version,
                scale_bits: fill.var_scale.to_bits(),
                seed_bits: fill.seed.to_bits(),
                color_enabled,
                roughness_enabled,
            };
            variation_noise_needed.insert(layer.id);
            let current = self.fill_variation_noise.get(&layer.id)
                .is_some_and(|cache| {
                    cache.key == key
                        && cache.color.as_ref().is_some_and(|texture| texture.width == self.res) == color_enabled
                        && cache.roughness.as_ref().is_some_and(|texture| texture.width == self.res) == roughness_enabled
                });
            if current {
                variation_noise_hits += 1;
                continue;
            }
            let color = color_enabled.then(|| self.new_tex("fill color FBM cache", self.res, self.res, F::R32Float, LAYER_USAGE));
            let roughness = roughness_enabled.then(|| self.new_tex("fill roughness FBM cache", self.res, self.res, F::R32Float, LAYER_USAGE));
            self.fill_variation_noise.insert(layer.id, FillVariationNoiseCache { key, color, roughness });
            if color_enabled {
                variation_noise_misses.push((slot, li, layer.id, false));
            }
            if roughness_enabled {
                variation_noise_misses.push((slot, li, layer.id, true));
            }
        }
        self.fill_variation_noise.retain(|id, cache| {
            variation_noise_needed.contains(id)
                && cache.color.as_ref().is_none_or(|texture| texture.width == self.res)
                && cache.roughness.as_ref().is_none_or(|texture| texture.width == self.res)
        });
        self.last_fill_variation_cache_hits = variation_noise_hits;
        self.last_fill_variation_cache_misses = variation_noise_needed.len() as u32 - variation_noise_hits;
        self.last_fill_variation_cache_gpu_passes = variation_noise_misses.len() as u32;

        // Cached mask sources get an extra uniform slot after the layer passes.
        // Filtered masks run their filter stack after the source; generator-only
        // masks go directly into the persistent result texture.
        let mut source_slot: HashMap<usize, usize> = HashMap::with_capacity(passes.len());
        for pass in &passes {
            if let Pass::Layer(li) | Pass::End(li) = *pass
                && self.needs_mask_cache(li) {
                    source_slot.insert(li, passes.len() + source_slot.len());
                }
        }
        // Each enabled mask layer renders through its own uniform slot too: the
        // layer's uniform with that mask layer's generator (or none, for a fill).
        let mut item_slots: HashMap<usize, Vec<usize>> = HashMap::with_capacity(source_slot.len());
        let mut next_slot = passes.len() + source_slot.len();
        for &li in source_slot.keys() {
            let n = self.layers[li].mask.as_ref().map_or(0, |m| m.layers.iter().filter(|ml| ml.enabled).count());
            item_slots.insert(li, (next_slot..next_slot + n).collect());
            next_slot += n;
        }
        // Each single-channel FBM prepass gets its own dynamic uniform slot.
        let variation_noise_slots: Vec<(usize, usize, u64, bool)> = variation_noise_misses
            .iter()
            .map(|&(_layer_slot, layer_index, layer_id, roughness)| {
                let prepass_slot = next_slot;
                next_slot += 1;
                (prepass_slot, layer_index, layer_id, roughness)
            })
            .collect();
        let slots = next_slot;
        let required_layer_slots = layer_uniform_buffer_capacity(self.layer_ubuf_slots, slots);
        if required_layer_slots != self.layer_ubuf_slots {
            self.layer_ubuf_slots = required_layer_slots;
            self.layer_ubuf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("layer uniforms"),
                size: UNIFORM_STRIDE * self.layer_ubuf_slots,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

        }
        let mut data = std::mem::take(&mut self.layer_uniform_scratch);
        data.resize((UNIFORM_STRIDE as usize) * slots.max(1), 0);
        data.fill(0);
        let mut pass_output_masks = vec![0u8; passes.len()];
        for (k, pass) in passes.iter().enumerate() {
            let (Pass::Layer(li) | Pass::End(li)) = *pass else { continue };
            let mut u = self.layer_uniform(&self.layers[li]);
            let output_mask = {
                let mask = if matches!(*pass, Pass::End(_)) {
                    ACCUM_CHANNEL_MASK
                } else {
                    composite_output_mask(&u)
                };
                #[cfg(test)]
                {
                    if self.force_full_composite_channel_outputs { ACCUM_CHANNEL_MASK } else { mask }
                }
                #[cfg(not(test))]
                {
                    mask
                }
            };
            pass_output_masks[k] = output_mask;
            // The fragment shader avoids loading previous channels that the
            // selected composite pipeline does not write.
            u.varc[3] = output_mask as f32;
            if let Some(cache) = self.fill_variation_noise.get(&self.layers[li].id) {
                u.varc[0] = if cache.roughness.is_some() { 1.0 } else { 0.0 };
                u.varc[1] = if cache.color.is_some() { 1.0 } else { 0.0 };
            }
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
        for &(slot, layer_index, _layer_id, roughness) in &variation_noise_slots {
            let mut u = self.layer_uniform(&self.layers[layer_index]);
            u.varc[2] = if roughness { 1.0 } else { 0.0 };
            let offset = slot * UNIFORM_STRIDE as usize;
            data[offset..offset + size_of::<LayerUniform>()].copy_from_slice(bytemuck::bytes_of(&u));
        }
        if slots > 0 {
            self.queue.write_buffer(&self.layer_ubuf, 0, &data);
        }
        self.layer_uniform_scratch = data;
        // One persistent result per cached mask. Only masks with actual filter
        // operations need the three full-resolution ping-pong scratch targets.
        let cached_ids: HashSet<u64> = source_slot.keys().map(|&li| self.layers[li].id).collect();
        let has_filtered_masks = source_slot.keys().any(|&li| self.has_mask_filters(li));
        let res = self.res;
        let mask_final = &mut self.mask_final;
        self.mask_out.retain(|id, t| {
            let keep = cached_ids.contains(id) && t.width == res;
            if !keep { mask_final.remove(id); }
            keep
        });
        for id in &cached_ids {
            if !self.mask_out.contains_key(id) {
                let t = self.new_tex("cached mask result", res, res, MASK_FILTER_FORMAT, LAYER_USAGE);
                self.mask_out.insert(*id, t);
            }
        }
        if !has_filtered_masks {
            self.mask_scratch = None;
        } else if self.mask_scratch.as_ref().is_none_or(|set| set[0].width != res) {
            let t = || self.new_tex("filtered mask scratch", res, res, MASK_FILTER_FORMAT, LAYER_USAGE);
            self.mask_scratch = Some([t(), t(), t()]);
        }
        // Filter steps' uniforms, written up front: (kind, a, b, c), (dir x, dir y, scale).
        let mut steps = std::mem::take(&mut self.mask_filter_steps_scratch);
        steps.clear();
        let mut layer_ops: HashMap<usize, Vec<MaskOp>> = HashMap::with_capacity(source_slot.len());
        let mut layer_op_labels: HashMap<usize, Vec<String>> = HashMap::with_capacity(source_slot.len());
        for &li in source_slot.keys() {
            // A valid filtered-mask result bypasses its source and filter passes.
            // Avoid rebuilding its CPU operation plan and uniforms on cache hits.
            if self.mask_final.contains(&self.layers[li].id) || !self.has_mask_filters(li) {
                continue;
            }
            let scale = self.res as f32 / FILTER_REF_RES;
            let mask = self.layers[li].mask.as_ref().expect("filtered layers have masks");
            let enabled_layers = mask.layers.iter().filter(|ml| ml.enabled).count();
            let filter_steps = mask.filters.iter().filter(|filter| filter.enabled)
                .map(|filter| if filter.kind == FilterKind::Blur { 2 } else { 1 })
                .sum::<usize>();
            let operation_capacity = enabled_layers + filter_steps;
            let mut ops = Vec::with_capacity(operation_capacity);
            let mut op_labels = Vec::with_capacity(if profile_query.is_some() { operation_capacity } else { 0 });
            let mut mask_layer_number = 0usize;
            for (ml, &slot) in mask.layers.iter().filter(|ml| ml.enabled).zip(&item_slots[&li]) {
                mask_layer_number += 1;
                let value = match ml.source {
                    MaskSource::Fill(v) => v,
                    MaskSource::Generator(_) => 1.0,
                };
                ops.push(MaskOp::Layer { uniform: (slot as u64 * UNIFORM_STRIDE) as u32, combine: steps.len() });
                if profile_query.is_some() {
                    op_labels.push(format!("mask layer {mask_layer_number}"));
                }
                steps.push([COMBINE_STEP, ml.blend as u32 as f32, ml.opacity, value, 0.0, 0.0, scale, 0.0]);
            }
            for f in self.layers[li].mask.as_ref().map_or(&[][..], |m| &m.filters[..]).iter().filter(|f| f.enabled) {
                let kind = f.kind as u32 as f32;
                // Each step of a filter reads as far as the whole filter does.
                let (reach, wrap) = (f.reach(self.res), f.kind == FilterKind::Warp);
                if f.kind == FilterKind::Blur {
                    steps.push([kind, f.a, f.b, f.c, 1.0, 0.0, scale, 0.0]);
                    ops.push(MaskOp::Filter { step: steps.len() - 1, reach, wrap });
                    if profile_query.is_some() { op_labels.push("Blur horizontal".to_string()); }
                    steps.push([kind, f.a, f.b, f.c, 0.0, 1.0, scale, 0.0]);
                    ops.push(MaskOp::Filter { step: steps.len() - 1, reach, wrap });
                    if profile_query.is_some() { op_labels.push("Blur vertical".to_string()); }
                } else {
                    steps.push([kind, f.a, f.b, f.c, 0.0, 0.0, scale, 0.0]);
                    ops.push(MaskOp::Filter { step: steps.len() - 1, reach, wrap });
                    if profile_query.is_some() { op_labels.push(format!("{:?}", f.kind)); }
                }
            }
            layer_op_labels.insert(li, op_labels);
            layer_ops.insert(li, ops);
        }
        // Grow for the complete active stack. Truncating here would silently make
        // later filters reuse the final uniform slot and render the wrong mask.
        let required_capacity = mask_filter_buffer_capacity(self.mask_filter_ubuf_slots, steps.len());
        if required_capacity != self.mask_filter_ubuf_slots {
            self.mask_filter_ubuf_slots = required_capacity;
            self.mask_filter_ubuf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mask filter uniforms"),
                size: MASK_FILTER_STRIDE * self.mask_filter_ubuf_slots,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        let mut sdata = std::mem::take(&mut self.mask_filter_uniform_scratch);
        sdata.resize((MASK_FILTER_STRIDE as usize) * steps.len().max(1), 0);
        sdata.fill(0);
        for (i, s) in steps.iter().enumerate() {
            let off = i * MASK_FILTER_STRIDE as usize;
            sdata[off..off + 32].copy_from_slice(bytemuck::cast_slice(s));
        }
        if !steps.is_empty() {
            self.queue.write_buffer(&self.mask_filter_ubuf, 0, &sdata);
        }
        self.mask_filter_uniform_scratch = sdata;
        self.mask_filter_steps_scratch = steps;

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
            self.prefix_cache = Some(ACCUM_FORMATS.map(|f| self.new_tex("composite prefix", self.res, self.res, f, LAYER_USAGE)));
        }

        let mut encoder = self.device.create_command_encoder(&Default::default());
        if self.gpu_profiler.as_ref().is_some_and(|p| p.encoder_timestamps)
            && let Some(p) = &self.active_gpu_profile {
                encoder.write_timestamp(&p.query_set, 6);
            }
        // Compile the FBM prepass only when a cache miss actually needs it.
        let variation_pipeline = if variation_noise_slots.is_empty() {
            None
        } else {
            Some(self.p.fill_variation_noise.get_or_init(|| {
                gpu::fullscreen_pipeline(
                    &self.device,
                    "fill variation noise",
                    &self.p.composite_module,
                    "fs_fill_variation_noise",
                    &gpu::layout(&self.device, &self.p.fill_variation_noise_bgl),
                    // Preserve FBM precision: R16Float quantization changes final 8-bit pixels.
                    &[gpu::target(F::R32Float)],
                )
            }))
        };
        // The cached variation field is independent of the accumulation below.
        // Generate one FBM channel per pass on a key miss; later composites reuse it.
        for &(slot, layer_index, layer_id, roughness) in &variation_noise_slots {
            let layer = &self.layers[layer_index];
            let cache = &self.fill_variation_noise[&layer_id];
            let target = if roughness {
                cache.roughness.as_ref().expect("roughness cache texture")
            } else {
                cache.color.as_ref().expect("color cache texture")
            };
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fill variation noise"),
                layout: &self.p.fill_variation_noise_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.layer_ubuf,
                            offset: 0,
                            size: NonZeroU64::new(size_of::<LayerUniform>() as u64),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: wgpu::BindingResource::TextureView(&self.pos.view),
                    },
                ],
            });
            let profile_slot = begin_composite_profile_stage(
                &mut encoder, profile_query.as_deref(), &mut profile_stage_count,
            );
            let mut pass = gpu::pass(
                &mut encoder,
                "fill variation noise cache",
                &[gpu::color_attachment(&target.view, wgpu::LoadOp::Clear(wgpu::Color::BLACK))],
            );
            pass.set_pipeline(variation_pipeline.expect("variation pipeline created for cache misses"));
            pass.set_bind_group(0, &bg, &[(slot as u64 * UNIFORM_STRIDE) as u32]);
            pass.draw(0..3, 0..1);
            drop(pass);
            end_composite_profile_stage(
                &mut encoder,
                profile_query.as_deref(),
                profile_slot,
                if profile_query.is_some() {
                    format!("Fill {} FBM cache · {}", if roughness { "roughness" } else { "color" }, layer.name)
                } else {
                    String::new()
                },
                &mut profile_labels,
            );
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
        if let Some(p) = &self.active_gpu_profile {
            if self.gpu_profiler.as_ref().is_some_and(|profiler| profiler.encoder_timestamps) {
                encoder.write_timestamp(&p.query_set, 8);
            }
        }
        // Resumed frames only redo the patch the stroke touched. Each pass
        // there reads the texel it writes, so inside the patch the chain is
        // exact; outside it the final set keeps the last composite's result,
        // which the stroke didn't change. (Folder stacks saved after the
        // painted layer are only read inside the patch too.)
        let unknown = std::mem::take(&mut self.region_unknown);
        let mut scissors = if resume.is_some() && !unknown {
            Some(std::mem::take(&mut self.stroke_regions))
        } else {
            self.stroke_regions.clear();
            None
        };
        // Only a mask stroke changes a filtered mask's output neighborhood.
        // Content strokes never alter mask inputs, so expanding for every later
        // layer's cached filter chain wastes the dirty-region optimization.
        if let (Some(rects), Some(_)) = (scissors.as_mut(), resume.as_ref())
            && let Some(active) = self.active.as_ref().filter(|stroke| stroke.target == StrokeTarget::Mask)
            && let Some(mask) = self.layers.iter().find(|layer| layer.id == active.layer_id).and_then(|layer| layer.mask.as_ref()) {
                let reach = mask.filters.iter().filter(|f| f.enabled).map(|f| f.reach(self.res)).sum::<f32>();
                let wraps = mask.filters.iter().any(|f| f.enabled && f.kind == FilterKind::Warp);
                if reach > 0.0 {
                    expand_scissors(rects, reach.ceil() as u32 + 2, self.res, wraps);
                }
            }
        let full_area = u64::from(self.res) * u64::from(self.res);
        self.last_scissor_pixels = scissors.as_ref().map_or(full_area, |rects| rects.iter().map(|&r| scissor_area(r)).sum());
        self.last_scissor_bbox_pixels = scissors.as_ref().map_or(full_area, |rects| {
            let Some(first) = rects.first() else { return 0 };
            let bounds = rects.iter().skip(1).fold(*first, |a, &b| scissor_union(a, b));
            scissor_area(bounds)
        });
        self.last_scissor_rects = scissors.as_ref().map_or(1, |rects| rects.len() as u32);
        let (start, mut src_indices, mut depth, copy_prefix) = match &resume {
            Some(p) => {
                let sources = p.input.unwrap_or_else(|| {
                    resume_source_indices(self.final_indices, &passes[p.at..], &pass_output_masks[p.at..])
                });
                (p.at, sources, p.depth, p.input.is_none())
            }
            None => (0, [0; ACCUM], 0, false),
        };
        if scissors.as_ref().is_some_and(|rects| rects.is_empty()) {
            // Nothing visible changed.
            let resume = resume.expect("scissors imply resume");
            self.prefix = Some(resume);
            self.dirty = false;
            self.queue.submit([encoder.finish()]);
            self.active_gpu_profile = None;
            if let Some(rects) = scissors.take() {
                self.stroke_regions = rects;
                self.stroke_regions.clear();
            }
            self.composite_scratch.passes = passes;
            return;
        }
        if copy_prefix {
            let cache = self.prefix_cache.as_ref().expect("prefix cache holds the resumed input");
            if let Some(rects) = &scissors {
                for &[x0, y0, x1, y1] in rects {
                    let size = wgpu::Extent3d { width: x1 - x0, height: y1 - y0, depth_or_array_layers: 1 };
                    for channel in 0..ACCUM {
                        let from = &cache[channel];
                        let to = &self.accum[src_indices[channel]][channel];
                        let origin = wgpu::Origin3d { x: x0, y: y0, z: 0 };
                        encoder.copy_texture_to_texture(
                            wgpu::TexelCopyTextureInfo { origin, ..from.copy_info() },
                            wgpu::TexelCopyTextureInfo { origin, ..to.copy_info() },
                            size,
                        );
                    }
                }
            } else {
                for channel in 0..ACCUM {
                    let from = &cache[channel];
                    let to = &self.accum[src_indices[channel]][channel];
                    encoder.copy_texture_to_texture(from.copy_info(), to.copy_info(), from.extent());
                }
            }
        }
        if scissors.is_none() {
            self.full_recomps = self.full_recomps.wrapping_add(1);
        }
        // Each material channel has an independent ping-pong index. A cached
        // prefix is copied into the appropriate source textures only in the
        // changed region; subsequent passes can skip untouched channels.
        let mut finals: Vec<u64> = Vec::new();
        let profile_layer = self.stroke_layer;
        for (k, pass) in passes.iter().enumerate().skip(start) {
            if record == Some(k) {
                let input = if k + 1 < passes.len() {
                    let cache = self.prefix_cache.as_ref().expect("allocated above");
                    for channel in 0..ACCUM {
                        let from = &self.accum[src_indices[channel]][channel];
                        let to = &cache[channel];
                        encoder.copy_texture_to_texture(from.copy_info(), to.copy_info(), from.extent());
                    }
                    None
                } else {
                    Some(src_indices)
                };
                let layer_id = self.stroke_layer.expect("recording for a stroke");
                self.prefix = Some(Prefix { layer_id, passes: passes.clone(), at: k, depth, input });
            }
            let li = match *pass {
                Pass::Begin => {
                    let folder_snapshot_slot = begin_composite_profile_stage(
                        &mut encoder, profile_query.as_deref(), &mut profile_stage_count,
                    );
                    // Keep the stack below the folder to fade its result over.
                    fn at(t: &Tex, x: u32, y: u32) -> wgpu::TexelCopyTextureInfo<'_> {
                        wgpu::TexelCopyTextureInfo { origin: wgpu::Origin3d { x, y, z: 0 }, ..t.copy_info() }
                    }
                    if let Some(rects) = &scissors {
                        for &[x0, y0, x1, y1] in rects {
                            let size = wgpu::Extent3d { width: x1 - x0, height: y1 - y0, depth_or_array_layers: 1 };
                            for channel in 0..ACCUM {
                                let from = &self.accum[src_indices[channel]][channel];
                                let to = &self.folder_below[depth][channel];
                                encoder.copy_texture_to_texture(at(from, x0, y0), at(to, x0, y0), size);
                            }
                        }
                    } else {
                        for channel in 0..ACCUM {
                            let from = &self.accum[src_indices[channel]][channel];
                            let to = &self.folder_below[depth][channel];
                            encoder.copy_texture_to_texture(from.copy_info(), to.copy_info(), from.extent());
                        }
                    }
                    depth += 1;
                    end_composite_profile_stage(
                        &mut encoder,
                        profile_query.as_deref(),
                        folder_snapshot_slot,
                        format!("Folder stack snapshot depth {}", depth),
                        &mut profile_labels,
                    );
                    continue;
                }
                Pass::Layer(li) => li,
                Pass::End(li) => {
                    depth -= 1;
                    li
                }
            };
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
            let output_mask = pass_output_masks[k];
            if output_mask == 0 {
                continue;
            }
            let prev: [&Tex; ACCUM] = std::array::from_fn(|channel| {
                &self.accum[src_indices[channel]][channel]
            });
            let variation_cache = self.fill_variation_noise.get(&layer.id);
            let color_noise = variation_cache.and_then(|cache| cache.color.as_ref());
            let roughness_noise = variation_cache.and_then(|cache| cache.roughness.as_ref());
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
                    wgpu::BindGroupEntry { binding: 1, resource: view(Some(prev[0])) },
                    wgpu::BindGroupEntry { binding: 2, resource: view(Some(prev[1])) },
                    wgpu::BindGroupEntry { binding: 3, resource: view(Some(prev[2])) },
                    wgpu::BindGroupEntry { binding: 4, resource: view(Some(prev[3])) },
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
                    wgpu::BindGroupEntry { binding: 20, resource: view(Some(prev[FINAL_EMISSION])) },
                    wgpu::BindGroupEntry { binding: 21, resource: view(chans[EMISSION]) },
                    wgpu::BindGroupEntry { binding: 22, resource: view(Some(prev[FINAL_NORMAL])) },
                    wgpu::BindGroupEntry { binding: 23, resource: view(chans[NORMAL]) },
                    wgpu::BindGroupEntry { binding: 24, resource: view(Some(prev[FINAL_SPECULAR])) },
                    wgpu::BindGroupEntry { binding: 25, resource: view(chans[SPECULAR]) },
                    wgpu::BindGroupEntry { binding: 26, resource: view(Some(&self.tan)) },
                    wgpu::BindGroupEntry { binding: 27, resource: view(color_noise) },
                    wgpu::BindGroupEntry { binding: 28, resource: view(roughness_noise) },
                ],
            });
            let make_mask_bg = |mask: Option<&Tex>| self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("composite mask source"),
                layout: &self.p.gl_mask_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.layer_ubuf,
                            offset: 0,
                            size: NonZeroU64::new(size_of::<LayerUniform>() as u64),
                        }),
                    },
                    wgpu::BindGroupEntry { binding: 9, resource: or_dummy(mask, dummy) },
                    wgpu::BindGroupEntry { binding: 10, resource: view(Some(&self.pos)) },
                    wgpu::BindGroupEntry { binding: 11, resource: view(Some(&self.nrm)) },
                    wgpu::BindGroupEntry { binding: 12, resource: view(Some(self.map_tex(0))) },
                    wgpu::BindGroupEntry { binding: 13, resource: view(Some(self.map_tex(1))) },
                    wgpu::BindGroupEntry { binding: 14, resource: view(Some(self.map_tex(2))) },
                    wgpu::BindGroupEntry { binding: 15, resource: wgpu::BindingResource::TextureView(&self.custom_array.1) },
                    wgpu::BindGroupEntry { binding: 16, resource: view(Some(self.map_tex(4))) },
                    wgpu::BindGroupEntry { binding: 17, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            });
            let make_gl_core_bg = |mask: Option<&Tex>| self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("OpenGL composite core"),
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
                    wgpu::BindGroupEntry { binding: 1, resource: view(Some(prev[0])) },
                    wgpu::BindGroupEntry { binding: 2, resource: view(Some(prev[1])) },
                    wgpu::BindGroupEntry { binding: 3, resource: view(Some(prev[2])) },
                    wgpu::BindGroupEntry { binding: 4, resource: view(Some(prev[3])) },
                    wgpu::BindGroupEntry { binding: 5, resource: view(chans[0]) },
                    wgpu::BindGroupEntry { binding: 6, resource: view(chans[1]) },
                    wgpu::BindGroupEntry { binding: 7, resource: view(chans[2]) },
                    wgpu::BindGroupEntry { binding: 8, resource: view(chans[3]) },
                    wgpu::BindGroupEntry { binding: 9, resource: or_dummy(mask, dummy) },
                    wgpu::BindGroupEntry { binding: 10, resource: view(Some(&self.pos)) },
                    wgpu::BindGroupEntry { binding: 11, resource: view(Some(&self.nrm)) },
                    wgpu::BindGroupEntry { binding: 13, resource: view(Some(self.map_tex(1))) },
                    wgpu::BindGroupEntry { binding: 17, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry { binding: 18, resource: wgpu::BindingResource::Sampler(&self.repeat_sampler) },
                    wgpu::BindGroupEntry { binding: 19, resource: view(chans[OPACITY]) },
                    wgpu::BindGroupEntry { binding: 27, resource: view(color_noise) },
                    wgpu::BindGroupEntry { binding: 28, resource: view(roughness_noise) },
                ],
            });
            let make_gl_extra_bg = |mask: Option<&Tex>| self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("OpenGL composite extra"),
                layout: &self.p.gl_extra_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.layer_ubuf,
                            offset: 0,
                            size: NonZeroU64::new(size_of::<LayerUniform>() as u64),
                        }),
                    },
                    wgpu::BindGroupEntry { binding: 9, resource: or_dummy(mask, dummy) },
                    wgpu::BindGroupEntry { binding: 10, resource: view(Some(&self.pos)) },
                    wgpu::BindGroupEntry { binding: 11, resource: view(Some(&self.nrm)) },
                    wgpu::BindGroupEntry { binding: 18, resource: wgpu::BindingResource::Sampler(&self.repeat_sampler) },
                    wgpu::BindGroupEntry { binding: 20, resource: view(Some(prev[FINAL_EMISSION])) },
                    wgpu::BindGroupEntry { binding: 21, resource: view(chans[EMISSION]) },
                    wgpu::BindGroupEntry { binding: 22, resource: view(Some(prev[FINAL_NORMAL])) },
                    wgpu::BindGroupEntry { binding: 23, resource: view(chans[NORMAL]) },
                    wgpu::BindGroupEntry { binding: 24, resource: view(Some(prev[FINAL_SPECULAR])) },
                    wgpu::BindGroupEntry { binding: 25, resource: view(chans[SPECULAR]) },
                    wgpu::BindGroupEntry { binding: 26, resource: view(Some(&self.tan)) },
                ],
            });
            let make_gl_folder_bg = |mask: Option<&Tex>| self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("OpenGL folder merge"),
                layout: &self.p.gl_folder_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.layer_ubuf,
                            offset: 0,
                            size: NonZeroU64::new(size_of::<LayerUniform>() as u64),
                        }),
                    },
                    wgpu::BindGroupEntry { binding: 1, resource: view(Some(prev[0])) },
                    wgpu::BindGroupEntry { binding: 2, resource: view(Some(prev[1])) },
                    wgpu::BindGroupEntry { binding: 3, resource: view(Some(prev[2])) },
                    wgpu::BindGroupEntry { binding: 4, resource: view(Some(prev[3])) },
                    wgpu::BindGroupEntry { binding: 5, resource: view(chans[0]) },
                    wgpu::BindGroupEntry { binding: 6, resource: view(chans[1]) },
                    wgpu::BindGroupEntry { binding: 7, resource: view(chans[2]) },
                    wgpu::BindGroupEntry { binding: 8, resource: view(chans[3]) },
                    wgpu::BindGroupEntry { binding: 9, resource: or_dummy(mask, dummy) },
                    wgpu::BindGroupEntry { binding: 20, resource: view(Some(prev[FINAL_EMISSION])) },
                    wgpu::BindGroupEntry { binding: 21, resource: view(chans[EMISSION]) },
                    wgpu::BindGroupEntry { binding: 22, resource: view(Some(prev[FINAL_NORMAL])) },
                    wgpu::BindGroupEntry { binding: 23, resource: view(chans[NORMAL]) },
                    wgpu::BindGroupEntry { binding: 24, resource: view(Some(prev[FINAL_SPECULAR])) },
                    wgpu::BindGroupEntry { binding: 25, resource: view(chans[SPECULAR]) },
                ],
            });
            // A filtered mask depends on its own paint/generator/filter data,
            // not on the accumulated layers below it. Reuse the previous result
            // until that mask is edited; a mask-paint stroke invalidates only
            // its own cache and can refresh it with the current scissors.
            let filtered = match (*pass, source_slot.get(&li)) {
                (Pass::Layer(_) | Pass::End(_), Some(&slot)) => {
                    if self.mask_final.contains(&layer.id) {
                        mask_cache_hits += 1;
                        Some(&self.mask_out[&layer.id])
                    } else {
                        mask_cache_misses += 1;
                        let source_bg = make_mask_bg(mask);
                        if self.has_mask_filters(li) {
                            let ops = layer_ops.get(&li).map_or(&[][..], |o| &o[..]);
                            mask_gpu_passes += 1 + ops.iter().map(|op| match op {
                                MaskOp::Layer { .. } => 2,
                                MaskOp::Filter { .. } => 1,
                            }).sum::<u32>();
                            Some(self.encode_mask_filters(
                                &mut encoder,
                                layer.id,
                                &layer.name,
                                &source_bg,
                                (slot as u64 * UNIFORM_STRIDE) as u32,
                                ops,
                                layer_op_labels.get(&li).map_or(&[][..], |labels| labels.as_slice()),
                                scissors.as_deref(),
                                profile_query.as_deref(),
                                &mut profile_stage_count,
                                &mut profile_labels,
                            ))
                        } else {
                            mask_gpu_passes += 1;
                            Some(self.encode_mask_source(
                                &mut encoder,
                                &layer.name,
                                &source_bg,
                                (slot as u64 * UNIFORM_STRIDE) as u32,
                                &self.mask_out[&layer.id],
                                scissors.as_deref(),
                                profile_query.as_deref(),
                                &mut profile_stage_count,
                                &mut profile_labels,
                            ))
                        }
                    }
                }
                _ => None,
            };
            if filtered.is_some() && !self.mask_final.contains(&layer.id) {
                finals.push(layer.id);
            }
            // A patch keeps what is around it. Inactive channel attachments
            // are absent from the pass to avoid read/write traffic.
            let load = if scissors.is_some() { wgpu::LoadOp::Load } else { wgpu::LoadOp::Clear(wgpu::Color::BLACK) };
            let profile = self.active_gpu_profile.as_ref().filter(|_| profile_layer == Some(layer.id));
            composite_profiled |= profile.is_some();
            let detailed_profile_slot = if profile_query.is_some() && profile_stage_count < GPU_PROFILE_MAX_LAYER_PASSES {
                let slot = profile_stage_count;
                profile_stage_count += 1;
                encoder.write_timestamp(profile_query.as_deref().expect("query set checked"), GPU_PROFILE_BASE_QUERY_COUNT + (slot as u32 * 2));
                Some(slot)
            } else {
                None
            };
            let detailed_profile_label = detailed_profile_slot.map(|_| {
                let kind = if matches!(*pass, Pass::End(_)) { "Folder merge" } else { "Layer" };
                format!("{kind} #{} {}", li + 1, layer.name)
            });

            if self.p.gl_composite {
                let resolved_mask = filtered.or(mask);
                if layer.is_folder() {
                    let bg = make_gl_folder_bg(resolved_mask);
                    let attachments: [Option<wgpu::RenderPassColorAttachment<'_>>; ACCUM] =
                        std::array::from_fn(|channel| {
                            if output_mask & (1 << channel) == 0 {
                                None
                            } else {
                                let dst = 1 - src_indices[channel];
                                gpu::color_attachment(&self.accum[dst][channel].view, load)
                            }
                        });
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("OpenGL folder merge"),
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
                    pass.set_pipeline(&self.p.folder_merge);
                    pass.set_bind_group(0, &bg, &[(k as u64 * UNIFORM_STRIDE) as u32]);
                    if let Some(rects) = &scissors {
                        for &[x0, y0, x1, y1] in rects {
                            pass.set_scissor_rect(x0, y0, x1 - x0, y1 - y0);
                            pass.draw(0..3, 0..1);
                        }
                    } else {
                        pass.draw(0..3, 0..1);
                    }
                    drop(pass);
                } else {
                    let (core_pipeline, extra_pipeline) = self.p.gl_layer_pipelines(&self.device, output_mask);
                    let has_core = core_pipeline.is_some();
                    let has_extra = extra_pipeline.is_some();
                    if let Some(pipeline) = core_pipeline.as_ref() {
                        let bg = make_gl_core_bg(resolved_mask);
                        let core_mask = output_mask & 0x0f;
                        let attachments: [Option<wgpu::RenderPassColorAttachment<'_>>; 4] =
                            std::array::from_fn(|channel| {
                                if core_mask & (1 << channel) == 0 {
                                    None
                                } else {
                                    let dst = 1 - src_indices[channel];
                                    gpu::color_attachment(&self.accum[dst][channel].view, load)
                                }
                            });
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("OpenGL layer core"),
                            color_attachments: &attachments,
                            depth_stencil_attachment: None,
                            timestamp_writes: profile.map(|p| wgpu::RenderPassTimestampWrites {
                                query_set: &p.query_set,
                                beginning_of_pass_write_index: Some(4),
                                end_of_pass_write_index: if has_extra { None } else { Some(5) },
                            }),
                            occlusion_query_set: None,
                            multiview_mask: None,
                        }).forget_lifetime();
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, &bg, &[(k as u64 * UNIFORM_STRIDE) as u32]);
                        if let Some(rects) = &scissors {
                            for &[x0, y0, x1, y1] in rects {
                                pass.set_scissor_rect(x0, y0, x1 - x0, y1 - y0);
                                pass.draw(0..3, 0..1);
                            }
                        } else {
                            pass.draw(0..3, 0..1);
                        }
                        drop(pass);
                    }
                    if let Some(pipeline) = extra_pipeline.as_ref() {
                        let bg = make_gl_extra_bg(resolved_mask);
                        let extra_mask = (output_mask >> 4) & 0x07;
                        let attachments: [Option<wgpu::RenderPassColorAttachment<'_>>; 3] =
                            std::array::from_fn(|channel| {
                                if extra_mask & (1 << channel) == 0 {
                                    None
                                } else {
                                    let output_channel = channel + 4;
                                    let dst = 1 - src_indices[output_channel];
                                    gpu::color_attachment(&self.accum[dst][output_channel].view, load)
                                }
                            });
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("OpenGL layer extra"),
                            color_attachments: &attachments,
                            depth_stencil_attachment: None,
                            timestamp_writes: profile.map(|p| wgpu::RenderPassTimestampWrites {
                                query_set: &p.query_set,
                                beginning_of_pass_write_index: if has_core { None } else { Some(4) },
                                end_of_pass_write_index: Some(5),
                            }),
                            occlusion_query_set: None,
                            multiview_mask: None,
                        }).forget_lifetime();
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, &bg, &[(k as u64 * UNIFORM_STRIDE) as u32]);
                        if let Some(rects) = &scissors {
                            for &[x0, y0, x1, y1] in rects {
                                pass.set_scissor_rect(x0, y0, x1 - x0, y1 - y0);
                                pass.draw(0..3, 0..1);
                            }
                        } else {
                            pass.draw(0..3, 0..1);
                        }
                        drop(pass);
                    }
                }
            } else {
                let bg = make_bg(filtered.or(mask));
                let attachments: [Option<wgpu::RenderPassColorAttachment<'_>>; ACCUM] =
                    std::array::from_fn(|channel| {
                        if output_mask & (1 << channel) == 0 {
                            None
                        } else {
                            let dst = 1 - src_indices[channel];
                            gpu::color_attachment(&self.accum[dst][channel].view, load)
                        }
                    });
                let pipeline = self.p.composite_pipeline(&self.device, output_mask, layer.is_folder());
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
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bg, &[(k as u64 * UNIFORM_STRIDE) as u32]);
                if let Some(rects) = &scissors {
                    for &[x0, y0, x1, y1] in rects {
                        pass.set_scissor_rect(x0, y0, x1 - x0, y1 - y0);
                        pass.draw(0..3, 0..1);
                    }
                } else {
                    pass.draw(0..3, 0..1);
                }
                drop(pass);
            }
            if let Some(slot) = detailed_profile_slot {
                encoder.write_timestamp(profile_query.as_deref().expect("profile slot requires query set"), GPU_PROFILE_BASE_QUERY_COUNT + (slot as u32 * 2) + 1);
                if let Some(label) = detailed_profile_label {
                    profile_labels.push(label);
                }
            }
            src_indices = toggle_accum_indices(src_indices, output_mask);
        }
        if composite_profiled {
            if let Some(p) = &self.active_gpu_profile
                && self.gpu_profiler.as_ref().is_some_and(|profiler| profiler.encoder_timestamps)
            {
                encoder.write_timestamp(&p.query_set, 9);
                encoder.write_timestamp(&p.query_set, 7);
            }
        }
        self.queue.submit([encoder.finish()]);
        self.last_mask_cache_hits = mask_cache_hits;
        self.last_mask_cache_misses = mask_cache_misses;
        self.last_mask_gpu_passes = mask_gpu_passes;
        if let Some(p) = resume {
            self.prefix = Some(p);
        }
        self.final_indices = src_indices;
        self.generation += 1;
        self.dirty = false;
        self.full_dirty = false;
        self.mask_final.retain(|id| self.mask_out.contains_key(id));
        self.mask_final.extend(finals);
        self.update_thumbs_profiled(
            profile_query.as_deref(),
            &mut profile_stage_count,
            &mut profile_labels,
        );
        if composite_profiled {
            self.last_profiled_pass_labels = profile_labels;
            if let Some(profile) = self.active_gpu_profile.clone() {
                let query_count = if self.gpu_profiler.as_ref().is_some_and(|profiler| profiler.encoder_timestamps) {
                    GPU_PROFILE_QUERY_COUNT
                } else {
                    6
                };
                let resolve_bytes = u64::from(query_count) * 8;
                let mut resolve_encoder = self.device.create_command_encoder(&Default::default());
                resolve_encoder.resolve_query_set(&profile.query_set, 0..query_count, &profile.resolve, 0);
                resolve_encoder.copy_buffer_to_buffer(&profile.resolve, 0, &profile.readback, 0, resolve_bytes);
                self.queue.submit([resolve_encoder.finish()]);
            }
            self.active_gpu_profile = None;
            if let Some(profiler) = &mut self.gpu_profiler {
                profiler.request_map();
            }
        } else {
            self.active_gpu_profile = None;
        }
        if let Some(rects) = scissors.take() {
            self.stroke_regions = rects;
            self.stroke_regions.clear();
        }
        self.composite_scratch.passes = passes;
    }

    /// Whether any visible layer sets opacity: only then can the result be
    /// see-through (the 3D view skips its see-through pass otherwise).
    pub fn uses_opacity(&self) -> bool {
        self.layers.iter().any(|l| {
            l.visible
                && !l.disabled_channels[OPACITY]
                && match &l.content {
                    Content::Paint(ch) => ch[OPACITY].is_some(),
                    Content::Fill(fill) => fill.mat.enabled[OPACITY] || l.images[OPACITY].is_some(),
                    Content::Folder(_) => false,
                }
        })
    }

    /// Whether any visible layer sets emission (exports skip the map otherwise).
    pub fn uses_emission(&self) -> bool {
        self.layers.iter().any(|l| {
            l.visible
                && !l.disabled_channels[EMISSION]
                && match &l.content {
                    Content::Paint(ch) => ch[EMISSION].is_some(),
                    Content::Fill(fill) => fill.mat.enabled[EMISSION] || l.images[EMISSION].is_some(),
                    Content::Folder(_) => false,
                }
        })
    }

    /// Whether any visible layer sets normals or specular (exports skip
    /// the maps otherwise: flat normal, 0.5 specular).
    pub fn uses_channel(&self, ch: usize) -> bool {
        self.layers.iter().any(|l| {
            l.visible
                && !l.disabled_channels[ch]
                && match &l.content {
                    Content::Paint(c) => c[ch].is_some(),
                    Content::Fill(fill) => fill.mat.enabled[ch] || l.images[ch].is_some(),
                    Content::Folder(_) => false,
                }
        })
    }

    /// Final composited channels: base color (sRGB-encoded, with opacity in
    /// alpha), roughness, metallic, height, emission (sRGB-encoded), normal
    /// (tangent space, encoded 0..1), specular level.
    /// (A [`Pass`] list drives [`Engine::composite`].)
    pub fn final_tex(&self, ch: usize) -> &Tex {
        &self.accum[self.final_indices[ch]][ch]
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
        let dst = self.new_tex("resampled", res, res, src.format, LAYER_USAGE);
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

    /// Resamples the categorical ID map with nearest-neighbor sampling. A
    /// linear average of two different ID colors creates a label that did
    /// not exist in the source mesh and makes the mask miss both regions.
    fn resample_id_map(&self, encoder: &mut wgpu::CommandEncoder, src: &Tex, res: u32) -> Tex {
        debug_assert_eq!(src.format, MAP_FORMATS[ID_MAP_TEXTURE]);
        let dst = self.new_tex("resampled ID map", res, res, src.format, LAYER_USAGE);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nearest ID map resample"),
            layout: &self.p.resample_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src.view) },
                // The nearest entry point uses textureLoad, so the sampler is
                // present only to satisfy the shared bind-group layout.
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        let mut pass = gpu::pass(encoder, "resample ID map (nearest)", &[gpu::color_attachment(&dst.view, wgpu::LoadOp::Load)]);
        pass.set_pipeline(&self.p.resample_nearest_id);
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
            let maps = std::array::from_fn(|t| {
                if t == ID_MAP_TEXTURE {
                    self.resample_id_map(&mut encoder, &self.maps[t], res)
                } else {
                    self.resample(&mut encoder, &self.maps[t], res)
                }
            });
            self.maps = maps;
            self.custom = self.custom.iter().map(|t| self.resample(&mut encoder, t, res)).collect();
            self.build_custom_array(&mut encoder);
        }
        let fixes: Vec<Option<Tex>> = self.map_fix.iter().enumerate().map(|(map, fix)| {
            fix.as_ref().map(|tex| {
                if MAPS[map].tex == ID_MAP_TEXTURE {
                    self.resample_id_map(&mut encoder, tex, res)
                } else {
                    self.resample(&mut encoder, tex, res)
                }
            })
        }).collect();
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
        self.apply_bg = None;
        self.res = res;
        self.uv_version += 1;
        self.pos = self.new_tex("position map", res, res, POS_FORMAT, LAYER_USAGE);

        self.nrm = self.new_tex("normal map", res, res, NRM_FORMAT, LAYER_USAGE);
        self.tan = self.new_tex("tangent map", res, res, TAN_FORMAT, LAYER_USAGE);
        self.accum = [
            accum_set(&self.device, res, "accum A", self.srgb_views_supported),
            accum_set(&self.device, res, "accum B", self.srgb_views_supported),
        ];
        self.stroke = self.new_tex("stroke mask", res, res, STROKE_FORMAT, LAYER_USAGE);
        self.snap = Default::default();
        self.snap_mask = None;
        undo.clear();
        self.bake_uv(mesh);
    }

    // ------------------------------------------------------------------ stats


    fn fill_image_bytes(&self) -> u64 {
        let mut seen = self.fill_image_ids_scratch.borrow_mut();
        seen.clear();
        self.layers
            .iter()
            .flat_map(|layer| layer.images.iter().flatten())
            .filter(|image| seen.insert(Arc::as_ptr(&image.tex) as usize))
            .map(|image| image.tex.bytes())
            .sum()
    }

    fn custom_array_bytes(&self) -> u64 {
        let res = self.custom.first().map_or(1, |t| t.width) as u64;
        res * res * self.custom.len().max(1) as u64 * gpu::bytes_per_texel(CUSTOM_FORMAT) as u64
    }

    fn gpu_buffer_bytes(&self) -> u64 {
        let stroke = self.screen_index_buffer.size()
            + self.layer_ubuf.size()
            + self.stamp_ubuf.size()
            + self.stamp_sbuf.size()
            + self.apply_ubuf.size()
            + self.mask_filter_ubuf.size();
        let profile = self.gpu_profiler.as_ref().map_or(0, |p| p.resources.resolve.size() + p.resources.readback.size());
        stroke + profile
    }

    /// Application-side estimate of known GPU allocations by owned resource category.
    /// Driver padding, shared renderer resources and the viewport mesh are excluded.
    pub fn gpu_resource_breakdown(&self) -> [(&'static str, u64); 14] {
        [
            ("surface maps", self.pos.bytes() + self.nrm.bytes() + self.tan.bytes() + self.stroke.bytes()),
            ("mesh maps and fixes", self.maps.iter().map(|t| t.bytes()).sum::<u64>()
                + self.resolved.iter().flatten().map(|t| t.bytes()).sum::<u64>()
                + self.map_fix.iter().flatten().map(|t| t.bytes()).sum::<u64>()),
            ("custom maps", self.custom.iter().map(|t| t.bytes()).sum::<u64>() + self.custom_array_bytes()),
            ("accumulation", self.accum.iter().flatten().map(|t| t.bytes()).sum::<u64>()),
            ("folder cache", self.folder_below.iter().flatten().map(|t| t.bytes()).sum::<u64>()),
            ("prefix cache", self.prefix_cache.as_ref().map_or(0, |set| set.iter().map(|t| t.bytes()).sum())),
            ("filtered masks", self.mask_out.values().map(|t| t.bytes()).sum::<u64>()
                + self.mask_scratch.as_ref().map_or(0, |set| set.iter().map(|t| t.bytes()).sum::<u64>())),
            ("procedural variation caches", self.fill_variation_noise.values()
                .map(|cache| cache.color.as_ref().map_or(0, |texture| texture.bytes())
                    + cache.roughness.as_ref().map_or(0, |texture| texture.bytes()))
                .sum()),
            ("paint layers and masks", self.layers.iter().map(|l| l.vram_bytes()).sum::<u64>()),
            ("fill images", self.fill_image_bytes()),
            ("thumbnails", self.thumbs.vram_bytes()),
            ("brush and fallback textures", self.brush_alpha.bytes() + self.brush_grain.bytes() + self.dummy.bytes()),
            ("stroke snapshots", self.snap.iter().flatten().map(|t| t.bytes()).sum::<u64>()
                + self.snap_mask.as_ref().map_or(0, |t| t.bytes())),
            ("GPU buffers", self.gpu_buffer_bytes()),
        ]
    }

    /// Estimated GPU memory owned by this texture set: textures and known buffers.
    /// Driver padding, shared renderer resources and the viewport mesh are excluded.
    pub fn vram_bytes(&self) -> u64 {
        self.gpu_resource_breakdown().iter().map(|(_, bytes)| *bytes).sum()
    }

    /// Cached Undo staging buffers shared by all Texture Set Engines in this document.
    pub fn snapshot_buffer_pool_cached_bytes(&self) -> u64 {
        self.snapshot_buffer_pool.cached_bytes()
    }

    pub fn layer_vram_bytes(&self, idx: usize) -> u64 {
        self.layers.get(idx).map_or(0, |l| l.vram_bytes())
    }
    /// Where the texture memory goes, in MB (offscreen tests).
    #[cfg(test)]
    pub fn vram_breakdown(&self) -> Vec<(&'static str, u64)> {
        self.gpu_resource_breakdown()
            .into_iter()
            .map(|(category, bytes)| (category, bytes >> 20))
            .collect()
    }
}


#[cfg(test)]
mod composite_pass_tests {
    use super::*;

    fn layer(id: u64, parent: Option<u64>, visible: bool, opacity: f32, content: Content) -> Layer {
        Layer {
            id,
            name: format!("Layer {id}"),
            visible,
            opacity,
            blend: Blend::Normal,
            content,
            mask: None,
            disabled_channels: [false; CHANNELS],
            parent,
            images: std::array::from_fn(|_| None),
            thumb: None,
        }
    }

    fn fill() -> Content {
        Content::Fill(FillParams::default())
    }

    fn folder() -> Content {
        Content::Folder(Folder { open: true, graph: None })
    }

    fn legacy_passes(layers: &[Layer]) -> (Vec<Pass>, usize) {
        let ancestors = |idx: usize| {
            std::iter::successors(layers[idx].parent.and_then(|id| layers.iter().position(|l| l.id == id)), |&i| {
                layers[i].parent.and_then(|id| layers.iter().position(|l| l.id == id))
            })
        };
        let shown = |idx: usize| layers[idx].visible && ancestors(idx).all(|a| layers[a].visible);
        let isolates = |idx: usize| layers[idx].is_folder() && (layers[idx].opacity < 1.0 || layers[idx].mask.is_some());
        let block_start = |idx: usize| {
            let mut start = idx;
            while start > 0 && ancestors(start - 1).any(|a| layers[a].id == layers[idx].id) {
                start -= 1;
            }
            start
        };

        let n = layers.len();
        let mut passes = Vec::new();
        let (mut depth, mut max_depth) = (0usize, 0usize);
        for i in 0..n {
            let mut opening: Vec<usize> = (i..n).filter(|&f| isolates(f) && shown(f) && block_start(f) == i).collect();
            opening.sort_by(|a, b| b.cmp(a));
            for _ in opening {
                passes.push(Pass::Begin);
                depth += 1;
                max_depth = max_depth.max(depth);
            }
            let has_content = match &layers[i].content {
                Content::Paint(ch) => ch.iter().any(Option::is_some),
                Content::Fill(fill) => fill.mat.enabled.iter().any(|&enabled| enabled),
                Content::Folder(_) => false,
            };
            if has_content && layers[i].opacity > 0.0 && shown(i) {
                passes.push(Pass::Layer(i));
            }
            if isolates(i) && shown(i) {
                passes.push(Pass::End(i));
                depth -= 1;
            }
        }
        (passes, max_depth)
    }

    #[test]
    fn linear_pass_builder_matches_nested_folder_order_and_visibility() {
        let layers = vec![
            layer(10, Some(12), true, 1.0, fill()),
            layer(11, Some(12), true, 1.0, fill()),
            layer(12, Some(14), true, 0.5, folder()),
            layer(13, Some(14), true, 1.0, fill()),
            layer(14, None, true, 0.5, folder()),
            layer(15, None, true, 1.0, fill()),
            layer(16, Some(17), true, 1.0, fill()),
            layer(17, None, false, 0.5, folder()),
            layer(18, Some(19), true, 1.0, fill()),
            layer(19, None, true, 1.0, folder()),
        ];

        let expected = vec![
            Pass::Begin,
            Pass::Begin,
            Pass::Layer(0),
            Pass::Layer(1),
            Pass::End(2),
            Pass::Layer(3),
            Pass::End(4),
            Pass::Layer(5),
            Pass::Layer(8),
        ];
        assert_eq!(composite_passes(&layers), (expected.clone(), 2));
        assert_eq!(composite_passes(&layers), legacy_passes(&layers));

        // Reuse one scratch object while the stack shrinks, empties and grows
        // again. Retained nested buckets must not leak stale folder passes.
        let mut scratch = CompositeScratch::default();
        assert_eq!(composite_passes_with_scratch(&layers, &mut scratch), (expected.clone(), 2));
        let small = vec![
            layer(30, None, true, 1.0, fill()),
            layer(31, None, true, 1.0, fill()),
        ];
        assert_eq!(
            composite_passes_with_scratch(&small, &mut scratch),
            legacy_passes(&small),
        );
        assert_eq!(composite_passes_with_scratch(&[], &mut scratch), (Vec::new(), 0));
        assert_eq!(
            composite_passes_with_scratch(&layers, &mut scratch),
            legacy_passes(&layers),
        );
    }

    #[test]
    fn gpu_timestamp_readback_covers_every_profiled_query() {
        assert_eq!(
            GPU_PROFILE_READBACK_SIZE,
            u64::from(GPU_PROFILE_QUERY_COUNT) * 8,
        );
        assert_eq!(
            GPU_PROFILE_QUERY_COUNT,
            GPU_PROFILE_BASE_QUERY_COUNT + GPU_PROFILE_MAX_LAYER_PASSES as u32 * 2,
        );
        assert!(GPU_PROFILE_QUERY_COUNT > 10);
    }

    /// The shared GL core pipeline can be used with the full seven-slot
    /// accumulation render pass. Its fragment shader writes only core outputs;
    /// the trailing extra-channel attachments must remain disabled, not absent.
    #[test]
    #[ignore]
    fn gl_composite_core_matches_full_accum_attachment_slots() {
        let (_instance, adapter, device, queue) = crate::bake::tests::gpu_with_adapter();
        assert_eq!(
            adapter.get_info().backend,
            wgpu::Backend::Gl,
            "OpenGL compatibility test did not obtain a GL adapter",
        );

        let pipelines = Engine::create_pipelines(&device, true);
        let textures = ACCUM_FORMATS.iter().map(|&format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("OpenGL composite compatibility target"),
                size: wgpu::Extent3d { width: 4, height: 4, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
        }).collect::<Vec<_>>();
        assert_eq!(textures.len(), ACCUM, "test must bind every accumulation channel");
        let views = textures.iter().map(|texture| texture.create_view(&Default::default())).collect::<Vec<_>>();
        let attachments = views.iter().map(|view| {
            gpu::color_attachment(view, wgpu::LoadOp::Clear(wgpu::Color::BLACK))
        }).collect::<Vec<_>>();

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("OpenGL seven-slot composite compatibility"),
                color_attachments: &attachments,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            // This is the exact operation that failed while opening the user's project.
            pass.set_pipeline(&pipelines.composite);
        }
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).expect("OpenGL composite compatibility poll");
    }

    #[test]
    fn empty_layer_stack_has_no_composite_passes() {
        assert_eq!(composite_passes(&[]), (Vec::new(), 0));
    }
}

#[cfg(test)]
mod scissor_tests {
    use super::*;

    fn assert_covered(original: &[[u32; 4]], compacted: &[[u32; 4]]) {
        for &[x0, y0, x1, y1] in original {
            for y in y0..y1 {
                for x in x0..x1 {
                    assert!(
                        compacted.iter().any(|r| r[0] <= x && x < r[2] && r[1] <= y && y < r[3]),
                        "original texel ({x}, {y}) was dropped"
                    );
                }
            }
        }
    }

    fn assert_disjoint(rects: &[[u32; 4]]) {
        for i in 0..rects.len() {
            for j in i + 1..rects.len() {
                assert_eq!(scissor_overlap_area(rects[i], rects[j]), 0, "scissors overlap");
            }
        }
    }

    #[test]
    fn warp_filter_scissors_wrap_across_uv_seams() {
        let (x, nx) = expand_scissor_axis(2, 5, 4, 100, true);
        assert_eq!(nx, 2);
        assert_eq!(&x[..nx], &[(98, 100), (0, 9)]);

        let (y, ny) = expand_scissor_axis(20, 30, 5, 100, true);
        assert_eq!(ny, 1);
        assert_eq!(y[0], (15, 35));

        let (full, nf) = expand_scissor_axis(10, 20, 50, 100, true);
        assert_eq!(nf, 1);
        assert_eq!(full[0], (0, 100));

        let mut rects = vec![[2, 20, 5, 24]];
        expand_scissors(&mut rects, 4, 100, true);
        assert_eq!(rects, vec![[0, 16, 9, 28], [98, 16, 100, 28]]);
        assert_disjoint(&rects);

        // A patch touching both UV seams becomes the four texture corners.
        let mut corner = vec![[2, 3, 5, 6]];
        expand_scissors(&mut corner, 4, 100, true);
        assert_eq!(corner.len(), 4);
        assert_disjoint(&corner);
        for point in [[0, 0], [99, 0], [0, 99], [99, 99]] {
            assert!(
                corner.iter().any(|r| r[0] <= point[0] && point[0] < r[2] && r[1] <= point[1] && point[1] < r[3]),
                "wrapped corner texel {:?} wasn't included",
                point
            );
        }

        // Blur/sharpen clamp at the texture edge and must not wrap.
        let mut clamped = vec![[2, 20, 5, 24]];
        expand_scissors(&mut clamped, 4, 100, false);
        assert_eq!(clamped, vec![[0, 16, 9, 28]]);
    }

    #[test]
    fn resumed_composite_finishes_in_previous_final_accum_set_per_channel() {
        let sequences: &[(&[Pass], &[u8])] = &[
            (&[Pass::Layer(0)], &[0b000_0001]),
            (&[Pass::Layer(0), Pass::Layer(1)], &[0b000_0011, 0b000_0010]),
            (&[Pass::Layer(0), Pass::Layer(1), Pass::Layer(2)], &[0b000_0001, 0b000_0010, 0b000_0100]),
            (
                &[Pass::Begin, Pass::Layer(0), Pass::Begin, Pass::Layer(1), Pass::End(1), Pass::End(0)],
                &[0, 0b000_0011, 0, 0b000_0100, ACCUM_CHANNEL_MASK, ACCUM_CHANNEL_MASK],
            ),
            (
                &[Pass::Layer(0), Pass::Begin, Pass::Layer(1), Pass::End(1), Pass::Layer(2)],
                &[0b000_0001, 0, 0b000_0010, ACCUM_CHANNEL_MASK, 0b000_0100],
            ),
        ];
        for pattern in 0..2 {
            let final_indices = std::array::from_fn(|channel| (channel + pattern) & 1);
            for (passes, masks) in sequences {
                let mut source = resume_source_indices(final_indices, passes, masks);
                for (&pass, &mask) in passes.iter().zip(*masks) {
                    if !matches!(pass, Pass::Begin) {
                        source = toggle_accum_indices(source, mask);
                    }
                }
                assert_eq!(source, final_indices, "resumed pass sequence changed a channel's final index");
            }
        }
    }

    #[test]
    fn normalized_scissors_reject_non_finite_bounds() {
        assert!(normalized_scissors(&[[f32::NAN, 0.0, 1.0, 1.0]], 128).is_none());
        assert!(normalized_scissors(&[[0.0, 0.0, f32::INFINITY, 1.0]], 128).is_none());
    }

    #[test]
    fn normalized_scissors_round_outward_and_clip_to_texture() {
        let margin = DILATE_PASSES + 2;
        let rects = normalized_scissors(&[[-0.1, 0.2, 0.251, 1.2], [0.7, 0.7, 0.8, 0.8]], 100).expect("finite bounds");
        assert_eq!(
            rects,
            vec![
                [0, 20 - margin, 26 + margin, 100],
                [70 - margin, 70 - margin, 80 + margin, 80 + margin],
            ],
        );
    }

    #[test]
    fn compact_scissors_merges_overlaps_without_losing_coverage() {
        let original = vec![[0, 0, 10, 10], [5, 5, 15, 15], [20, 0, 25, 5], [25, 0, 30, 5]];
        let mut compacted = original.clone();
        compact_scissors(&mut compacted);
        assert_covered(&original, &compacted);
        assert_disjoint(&compacted);
        assert!(compacted.len() <= original.len());
    }

    #[test]
    fn compact_scissors_caps_draw_count_without_losing_coverage() {
        let original: Vec<[u32; 4]> = (0..65)
            .map(|i| {
                let x = (i % 13) * 4;
                let y = (i / 13) * 4;
                [x, y, x + 2, y + 2]
            })
            .collect();
        let mut compacted = original.clone();
        compact_scissors(&mut compacted);
        assert!(compacted.len() <= MAX_COMPOSITE_SCISSORS);
        assert_covered(&original, &compacted);
        assert_disjoint(&compacted);
    }
}

#[cfg(test)]
mod mask_cache_invalidation_tests {
    use super::{Blend, Content, Engine, FilterKind, Folder, Layer, Mask, MaskFilter};
    use crate::undo::{DocProps, LayerProps};
    use std::collections::HashSet;

    fn live_layer(id: u64, visible: bool, filters: Vec<MaskFilter>) -> Layer {
        Layer {
            id,
            name: "Layer".into(),
            visible,
            opacity: 1.0,
            blend: Blend::Normal,
            content: Content::Folder(Folder { open: true, graph: None }),
            mask: Some(Mask {
                black_base: false,
                paint: None,
                generator: None,
                filters,
                layers: Vec::new(),
                graph: None,
            }),
            disabled_channels: [false; super::CHANNELS],
            parent: None,
            images: Default::default(),
            thumb: None,
        }
    }

    fn props() -> LayerProps {
        LayerProps {
            name: "Layer".into(),
            visible: true,
            opacity: 1.0,
            blend: Blend::Normal,
            fill: None,
            generator: None,
            mask_filters: Vec::new(),
            mask_layers: Vec::new(),
        }
    }

    fn doc() -> DocProps {
        DocProps { layers: vec![(7, props())], levels: Vec::new() }
    }

    #[test]
    fn live_property_check_does_not_invalidate_for_unrelated_layer_properties() {
        let before = doc();
        let layer = live_layer(7, false, Vec::new());
        let mut cache = HashSet::from([7]);
        Engine::apply_live_mask_cache_changes(&mut cache, &[layer], &[], &before);
        assert!(cache.contains(&7), "visibility and mask absence must not invalidate unchanged mask inputs");
    }

    #[test]
    fn live_property_check_invalidates_only_the_changed_mask() {
        let before = doc();
        let layer = live_layer(7, true, vec![MaskFilter::new(FilterKind::Blur)]);
        let mut cache = HashSet::from([7, 8]);
        Engine::apply_live_mask_cache_changes(&mut cache, &[layer], &[], &before);
        assert_eq!(cache, HashSet::from([8]));
    }

    #[test]
    fn live_property_check_clears_every_mask_when_layer_identity_changes() {
        let before = doc();
        let layer = live_layer(8, true, Vec::new());
        let mut cache = HashSet::from([7, 8]);
        Engine::apply_live_mask_cache_changes(&mut cache, &[layer], &[], &before);
        assert!(cache.is_empty());
    }

    #[test]
    fn material_visibility_and_opacity_keep_filtered_mask_cache() {
        let before = doc();
        let mut after = before.clone();
        after.layers[0].1.opacity = 0.35;
        after.layers[0].1.visible = false;
        after.layers[0].1.blend = Blend::Screen;
        assert_eq!(Engine::mask_cache_changes(&before, &after), (false, Vec::new()));
    }

    #[test]
    fn filter_edit_invalidates_only_its_layer_mask() {
        let before = doc();
        let mut after = before.clone();
        after.layers[0].1.mask_filters.push(MaskFilter::new(FilterKind::Blur));
        assert_eq!(Engine::mask_cache_changes(&before, &after), (false, vec![7]));
    }

    #[test]
    fn map_level_or_layer_identity_change_invalidates_all_masks() {
        let before = doc();
        let mut levels = before.clone();
        levels.levels.push(Default::default());
        assert_eq!(Engine::mask_cache_changes(&before, &levels), (true, Vec::new()));

        let mut structure = before.clone();
        structure.layers[0].0 = 8;
        assert_eq!(Engine::mask_cache_changes(&before, &structure), (true, Vec::new()));
    }
}

#[cfg(test)]
mod uv_bake_key_tests {
    use super::UvBakeKey;

    fn key() -> UvBakeKey {
        UvBakeKey { mesh_id: 7, tris: 0..36, res: 2048 }
    }

    #[test]
    fn same_mesh_range_and_size_repeats_the_bake() {
        assert!(UvBakeKey::is_current(Some(&key()), 7, &(0..36), 2048));
    }

    #[test]
    fn never_baked_engine_must_bake() {
        assert!(!UvBakeKey::is_current(None, 7, &(0..36), 2048));
    }

    #[test]
    fn changed_range_must_bake() {
        assert!(!UvBakeKey::is_current(Some(&key()), 7, &(0..18), 2048));
        assert!(!UvBakeKey::is_current(Some(&key()), 7, &(18..36), 2048));
    }

    #[test]
    fn changed_mesh_identity_must_bake_even_with_the_same_range() {
        assert!(!UvBakeKey::is_current(Some(&key()), 8, &(0..36), 2048));
    }

    #[test]
    fn changed_resolution_must_bake() {
        assert!(!UvBakeKey::is_current(Some(&key()), 7, &(0..36), 1024));
    }
}


#[cfg(test)]
mod id_map_resample_tests {
    use super::Engine;
    use crate::bake::tests::gpu_with_adapter;
    use crate::gpu::{GpuMesh, LAYER_USAGE, Tex, read_texture};
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
    fn id_map_resample_nearest_preserves_categorical_labels() {
        let rs = render_state();
        let mesh = Primitive::Plane.build();
        let gpu_mesh = GpuMesh::new(&rs.device, &mesh);
        let engine = Engine::new(&rs, &gpu_mesh, (mesh.bounds_min, mesh.bounds_max), 64);

        let source = Tex::new_without_srgb_view(
            &rs.device,
            "categorical ID source",
            2,
            1,
            wgpu::TextureFormat::Rgba8Unorm,
            LAYER_USAGE,
        );
        let source_pixels = [
            255u8, 0, 0, 255, // red ID
            0, 255, 0, 255,   // green ID
        ];
        rs.queue.write_texture(
            source.copy_info(),
            &source_pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(8),
                rows_per_image: Some(1),
            },
            source.extent(),
        );

        let mut encoder = rs.device.create_command_encoder(&Default::default());
        let resized = engine.resample_id_map(&mut encoder, &source, 4);
        rs.queue.submit([encoder.finish()]);
        rs.device.poll(wgpu::PollType::wait_indefinitely()).expect("ID map nearest resample completion");

        let pixels = read_texture(&rs.device, &rs.queue, &resized);
        assert_eq!(pixels.len(), 4 * 4 * 4);
        for (index, pixel) in pixels.chunks_exact(4).enumerate() {
            let expected = if index % 4 < 2 { [255, 0, 0, 255] } else { [0, 255, 0, 255] };
            assert_eq!(pixel, expected, "resampled texel {index} must keep one of the original IDs");
        }
    }
}
