//! Mesh-map baking: CPU scene preparation (worker thread) and the GPU baker.
//!
//! The bake source is either the low poly itself or a high poly. Everything
//! is ray traced on the GPU with a software BVH. The expensive occlusion pass
//! runs in tiles, a few in flight at a time, so the UI stays responsive and no
//! single dispatch can trip the OS GPU watchdog.

use crate::bvh::Bvh;
use crate::gpu::{self, GpuMesh, Tex};
use crate::mesh::{HighPoly, Mesh, smooth_normals, weld_positions};
use eframe::{egui, wgpu};
use glam::Vec3;
use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::time::Instant;
use wgpu::TextureFormat as F;
use wgpu::util::DeviceExt;

/// Number of distinct vertex-color IDs exposed in the bake report. This
/// matches the top palette entries shown in the Vertex ID panel.
pub const MAX_VERTEX_COLOR_ID_REGIONS: usize = 48;

/// Largest resolution the baker works at. Its working textures take about
/// 240 bytes a texel (surface, low-poly and cage maps in 32-bit float, the
/// edge search, the ray-traced maps and their scratch copies): 4 GB at
/// 4096², 16 GB at 8192², more than most GPUs have next to an 8K texture
/// set. Bigger texture sets get the bake upscaled (see `Engine::set_maps`).
pub const MAX_BAKE_RES: u32 = 4096;

/// How many packed map textures a bake makes (see `engine::MAPS`).
pub const MAP_TEXTURES: usize = 5;

/// Formats of the packed map textures (see `engine::MAPS`).
/// Map B (sky, edge distance, crease distance, part random) is 16-bit: narrow
/// edge masks read only a small slice of the distance range, which 8 bits
/// would band into stripes running along every edge.
pub const MAP_FORMATS: [F; MAP_TEXTURES] = [F::Rgba8Unorm, F::Rgba16Float, F::Rgba8Unorm, F::Rgba16Float, F::Rgba8Unorm];

/// Custom maps bake four to a texture, one per channel, 16-bit for smooth
/// flow and exposure gradients: map `i` is channel `i % 4` of texture `i / 4`.
pub const CUSTOM_FORMAT: F = F::Rgba16Float;
pub const CUSTOM_PER_TEXTURE: usize = 4;
/// Most custom maps a bake makes (the views and masks read them from one
/// texture array; this keeps it within every GPU's layer limit).
pub const MAX_CUSTOM_MAPS: usize = 64;

/// A bake's maps: the packed built-in maps and the custom map textures.
#[derive(Clone)]
pub struct BakedMaps {
    pub maps: [Tex; MAP_TEXTURES],
    pub custom: Vec<Tex>,
}

/// What a custom map bakes (see `cs_custom` in bake.wgsl).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CustomKind {
    /// Run-off: fluid lands on surfaces facing its direction and runs along
    /// the surface away from it (rain streaks, rust runs, leaks).
    Flow,
    /// Facing a direction and open to it (wind and sand erosion, sun bleaching).
    Exposure,
    /// Up-facing ground sheltered from a wind (sand, snow and dust drifts).
    Deposit,
    /// Closeness to other mesh parts (grime and rust where parts meet).
    Contact,
    /// Distance from a point (impact damage, heat around an exhaust).
    Point,
}

impl CustomKind {
    pub const ALL: [CustomKind; 5] =
        [CustomKind::Flow, CustomKind::Exposure, CustomKind::Deposit, CustomKind::Contact, CustomKind::Point];

    pub fn name(self) -> &'static str {
        match self {
            CustomKind::Flow => "Fluid Flow",
            CustomKind::Exposure => "Directional Exposure",
            CustomKind::Deposit => "Wind Deposits",
            CustomKind::Contact => "Contact",
            CustomKind::Point => "Distance from Point",
        }
    }

    pub fn help(self) -> &'static str {
        match self {
            CustomKind::Flow => {
                "Fluid lands on the surfaces facing where it comes from and runs along the surface away from it. \
                 Bright where a lot of surface drains through: the streaks below ledges, bolts and windows. \
                 For rain streaks, rust runs, leaks and water stains."
            }
            CustomKind::Exposure => {
                "How much each point faces the direction and sees it without anything in the way. \
                 For wind and sand erosion, sandblasting, sun bleaching and weathered paint on the windward side."
            }
            CustomKind::Deposit => {
                "Up-facing ground sheltered from the wind by something upwind. \
                 For sand, snow and dust drifting in behind walls and into corners."
            }
            CustomKind::Contact => {
                "How close another mesh part comes: 1 where parts touch, 0 at the distance. \
                 For grime, rust and wear where bolts, plates and panels meet. Reads only separate parts."
            }
            CustomKind::Point => "Falls off from a point to the distance. For impact damage, scorching and heat tint.",
        }
    }

    /// Takes a direction (where the fluid or wind comes from).
    pub fn has_direction(self) -> bool {
        matches!(self, CustomKind::Flow | CustomKind::Exposure | CustomKind::Deposit)
    }

    /// A new map of this kind with its usual settings (`id` and `name` to
    /// fill in, see [`BakeSettings::add_custom`]).
    pub fn default_map(self) -> CustomMap {
        let base = CustomMap {
            id: 0,
            name: self.name().to_string(),
            kind: self,
            azimuth: 0.0,
            elevation: 90.0,
            spread: 25.0,
            distance: 0.5,
            strength: 2.0,
            wander: 25.0,
            point: [0.0; 3],
        };
        match self {
            CustomKind::Flow => CustomMap { spread: 12.0, ..base },
            CustomKind::Exposure => CustomMap { elevation: 15.0, spread: 30.0, distance: 4.0, ..base },
            CustomKind::Deposit => CustomMap { elevation: 5.0, spread: 25.0, distance: 0.4, ..base },
            CustomKind::Contact => CustomMap { distance: 0.08, ..base },
            CustomKind::Point => CustomMap { distance: 1.0, ..base },
        }
    }

    /// The kind's number in the shader (0 = empty slot).
    fn code(self) -> f32 {
        match self {
            CustomKind::Flow => 1.0,
            CustomKind::Exposure => 2.0,
            CustomKind::Deposit => 3.0,
            CustomKind::Contact => 4.0,
            CustomKind::Point => 5.0,
        }
    }
}

/// A custom map's settings. Distances are in mesh units (the mesh is
/// normalized to a 2-unit box).
#[derive(Clone, PartialEq)]
pub struct CustomMap {
    /// Masks refer to the map by this (never reused within a project), so
    /// renaming, reordering or removing maps doesn't mix them up.
    pub id: u32,
    pub name: String,
    pub kind: CustomKind,
    /// Where the fluid or wind comes from: compass angle around the up axis
    /// (0 = +Z, 90 = +X) and height above the horizon (90 = straight above),
    /// in degrees.
    pub azimuth: f32,
    pub elevation: f32,
    /// Half-angle of the cone of directions around it (degrees): how
    /// unevenly rain falls, how gusty the wind is.
    pub spread: f32,
    /// Flow length, how far a blocker shelters or blocks, contact reach, or
    /// the point's radius.
    pub distance: f32,
    /// Flow: how quickly run-off saturates to white.
    pub strength: f32,
    /// Flow: how much the paths meander (degrees).
    pub wander: f32,
    /// Distance from point: the point.
    pub point: [f32; 3],
}

/// Everything but the name, which doesn't change what's baked (the bake
/// cache and the live preview key on this).
impl std::fmt::Debug for CustomMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomMap")
            .field("id", &self.id)
            .field("kind", &self.kind)
            .field("azimuth", &self.azimuth)
            .field("elevation", &self.elevation)
            .field("spread", &self.spread)
            .field("distance", &self.distance)
            .field("strength", &self.strength)
            .field("wander", &self.wander)
            .field("point", &self.point)
            .finish()
    }
}

impl CustomMap {
    /// Unit vector toward where the fluid or wind comes from.
    pub fn direction(&self) -> Vec3 {
        let (az, el) = (self.azimuth.to_radians(), self.elevation.to_radians());
        Vec3::new(el.cos() * az.sin(), el.sin(), el.cos() * az.cos())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Quality {
    Draft,
    Standard,
    High,
    Ultra,
}

impl Quality {
    pub const ALL: [Quality; 4] = [Quality::Draft, Quality::Standard, Quality::High, Quality::Ultra];
    pub fn name(self) -> &'static str {
        match self {
            Quality::Draft => "Draft (fast preview)",
            Quality::Standard => "Standard",
            Quality::High => "High",
            Quality::Ultra => "Ultra (final)",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IdSource {
    MeshParts,
    Materials,
    /// The mesh's vertex colors, as painted in the modeling app.
    VertexColors,
}

#[derive(Clone, Debug)]
pub struct BakeSettings {
    pub quality: Quality,
    pub high_poly: Option<PathBuf>,
    pub id_source: IdSource,
    /// Cage: how far outside / inside the low poly rays search for the high poly.
    pub front: f32,
    pub rear: f32,
    pub aa: u32,
    pub ao_rays: u32,
    pub ao_distance: f32,
    pub cavity_distance: f32,
    pub thickness_rays: u32,
    pub thickness_distance: f32,
    pub sky_rays: u32,
    pub sky_cone_deg: f32,
    pub curvature_radius: f32,
    pub curvature_scale: f32,
    pub edge_threshold: f32,
    pub edge_max_distance: f32,
    /// Derive the edge search distance from the mesh's proportions instead of
    /// `edge_max_distance`.
    pub edge_auto: bool,
    /// Mesh edges whose faces meet at more than this angle count as hard edges.
    pub hard_edge_angle: f32,
    /// Project each low-poly object only onto the high-poly objects with the
    /// same name (suffixes like _low/_high ignored), so parts close together
    /// don't capture each other's detail.
    pub match_by_name: bool,
    /// A cage mesh (the low poly with pushed-out vertices, same topology):
    /// rays start on the cage and travel toward the low poly.
    pub cage: Option<PathBuf>,
    /// Without a cage file: rays search along averaged normals (every face at
    /// a vertex's position, so split copies share one direction). The cage
    /// they start from stays closed over hard edges, which leave no gaps or
    /// doubled detail. Off: along each vertex's own normal.
    pub average_normals: bool,
    /// Rays skip faces they reach from behind (from inside another part or
    /// the high poly's far side). A ray that then finds nothing tries again
    /// without, so high polys with flipped faces still bake.
    pub ignore_backfaces: bool,
    /// A texel whose ray finds no surface takes the nearest point of the high
    /// poly instead of staying flat: no holes, even where the cage misses.
    pub fill_misses: bool,
    /// Painted cage distance: a multiplier of front and rear per low-poly
    /// vertex, raised where rays miss and lowered where another part bleeds in.
    pub cage_paint: Option<std::sync::Arc<CagePaint>>,
    /// Which maps to bake: a bit per `engine::MAPS` index. Maps left out are
    /// written neutral and their passes skipped where they can be.
    pub maps: u32,
    /// The custom maps, baked in this order (at most [`MAX_CUSTOM_MAPS`]).
    pub custom: Vec<CustomMap>,
}

/// Every built-in map (a bit per `engine::MAPS` index).
pub const ALL_MAPS: u32 = (1 << 13) - 1;
/// The maps a bake can leave out (the rest are cheap and always baked):
/// ambient occlusion, curvature, cavity, thickness, sky exposure, edge and
/// crease distance.
pub const OPTIONAL_MAPS: [usize; 7] = [0, 1, 2, 3, 4, 5, 6];
/// Map bits always baked.
const ALWAYS_MAPS: u32 = ALL_MAPS & !0b111_1111;

impl Default for BakeSettings {
    fn default() -> Self {
        let mut s = Self {
            quality: Quality::Standard,
            high_poly: None,
            id_source: IdSource::MeshParts,
            front: 0.05,
            rear: 0.05,
            aa: 2,
            ao_rays: 48,
            ao_distance: 0.6,
            cavity_distance: 0.04,
            thickness_rays: 24,
            thickness_distance: 0.5,
            sky_rays: 24,
            sky_cone_deg: 40.0,
            curvature_radius: 2.0,
            curvature_scale: 0.25,
            edge_threshold: 0.35,
            edge_max_distance: 0.2,
            edge_auto: true,
            hard_edge_angle: 35.0,
            match_by_name: true,
            cage: None,
            average_normals: true,
            ignore_backfaces: true,
            fill_misses: true,
            cage_paint: None,
            maps: ALL_MAPS,
            custom: Vec::new(),
        };
        s.apply_quality(Quality::Standard);
        s
    }
}

/// Largest distance (in the normalized mesh's units) a painted cage reaches
/// out or in from the low poly.
pub const MAX_CAGE_REACH: f32 = 2.0;
/// Range of a painted cage distance multiplier.
pub const CAGE_SCALE: std::ops::RangeInclusive<f32> = 0.05..=8.0;

/// A painted cage: a front and rear distance multiplier per low-poly vertex.
/// Compared and hashed (bake keys) by content, printed as a short digest.
#[derive(Clone)]
pub struct CagePaint {
    scale: Vec<f32>,
    digest: u64,
}

impl CagePaint {
    pub fn new(mut scale: Vec<f32>) -> CagePaint {
        use std::hash::{Hash, Hasher};
        for s in &mut scale {
            *s = s.clamp(*CAGE_SCALE.start(), *CAGE_SCALE.end());
        }
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for s in &scale {
            s.to_bits().hash(&mut h);
        }
        CagePaint { digest: h.finish(), scale }
    }

    /// Vertex `v`'s multiplier (1 past the end: a mesh it wasn't painted for).
    pub fn at(&self, v: usize) -> f32 {
        self.scale.get(v).copied().unwrap_or(1.0)
    }

    pub fn scales(&self) -> &[f32] {
        &self.scale
    }

    /// Painted for a mesh with this many vertices.
    pub fn fits(&self, vertices: usize) -> bool {
        self.scale.len() == vertices
    }
}

impl PartialEq for CagePaint {
    fn eq(&self, o: &Self) -> bool {
        self.digest == o.digest && self.scale == o.scale
    }
}

impl std::fmt::Debug for CagePaint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CagePaint({} vertices, {:016x})", self.scale.len(), self.digest)
    }
}

impl BakeSettings {
    /// Whether the bake makes its own cage from the low poly (no cage file):
    /// pushed out along averaged normals, by the painted distances.
    pub fn builds_cage(&self) -> bool {
        self.high_poly.is_some() && self.cage.is_none() && (self.average_normals || self.cage_paint.is_some())
    }

    /// The maps this bake makes (the chosen ones plus the always-baked).
    pub fn baked_maps(&self) -> u32 {
        (self.maps & ALL_MAPS) | ALWAYS_MAPS
    }

    /// The custom maps this bake makes (the first [`MAX_CUSTOM_MAPS`]).
    pub fn baked_custom(&self) -> &[CustomMap] {
        &self.custom[..self.custom.len().min(MAX_CUSTOM_MAPS)]
    }

    /// Adds a custom map of this kind with a fresh id and a name not yet
    /// taken ("Fluid Flow 2"); returns its index.
    pub fn add_custom(&mut self, kind: CustomKind) -> usize {
        let mut c = kind.default_map();
        c.id = self.custom.iter().map(|c| c.id).max().unwrap_or(0) + 1;
        let taken = |n: &str| self.custom.iter().any(|c| c.name == n);
        if taken(&c.name) {
            let base = c.name.clone();
            c.name = (2..).map(|k| format!("{base} {k}")).find(|n| !taken(n)).expect("a free name");
        }
        self.custom.push(c);
        self.custom.len() - 1
    }


    /// Rays per texel a custom map traces (flow: upstream walks).
    fn custom_rays(&self, kind: CustomKind) -> u32 {
        match kind {
            CustomKind::Flow => (self.sky_rays / 8).clamp(2, 8),
            CustomKind::Point => 0,
            _ => self.sky_rays.max(8),
        }
    }

    /// Custom maps `4g..4g+4` as the shader reads them (see `BakeU.cust`).
    fn custom_uniform(&self, g: usize) -> [[f32; 4]; CUSTOM_PER_TEXTURE * 3] {
        let mut u = [[0.0; 4]; CUSTOM_PER_TEXTURE * 3];
        for (k, c) in self.baked_custom().iter().skip(g * CUSTOM_PER_TEXTURE).take(CUSTOM_PER_TEXTURE).enumerate() {
            let d = c.direction();
            u[k * 3] = [c.kind.code(), c.distance.max(1e-4), c.spread.clamp(0.0, 89.0).to_radians().cos(), self.custom_rays(c.kind) as f32];
            u[k * 3 + 1] = [d.x, d.y, d.z, c.strength.max(0.0)];
            u[k * 3 + 2] = [c.point[0], c.point[1], c.point[2], c.wander.clamp(0.0, 89.0).to_radians()];
        }
        u
    }

    fn bakes(&self, map: usize) -> bool {
        self.baked_maps() & (1 << map) != 0
    }

    /// The same bake at preview quality: one projection ray and a few
    /// occlusion rays a texel, so it finishes in a fraction of a second.
    pub fn preview(&self) -> BakeSettings {
        let mut s = self.clone();
        s.aa = 1;
        s.ao_rays = s.ao_rays.min(16);
        s.thickness_rays = s.thickness_rays.min(8);
        // Sky rays also set the custom maps' rays (see `custom_rays`); with
        // fewer, their noise shows on thin UV islands.
        s.sky_rays = s.sky_rays.min(24);
        s
    }

    /// Everything [`prepare`] reads: bakes whose keys match can share one
    /// prepared scene and only redo the GPU passes.
    pub fn prepare_key(&self) -> String {
        format!(
            "{:?}|{:?}|{}|{:?}|{}|{:?}|{}|{}|{}|{}|{}",
            self.high_poly,
            self.cage,
            self.average_normals,
            self.cage_paint,
            // A built cage reaches `front` out from the low poly.
            if self.builds_cage() { self.front } else { 0.0 },
            self.id_source,
            self.match_by_name,
            self.edge_auto,
            self.edge_max_distance,
            self.curvature_scale,
            self.hard_edge_angle,
        )
    }

    pub fn apply_quality(&mut self, q: Quality) {
        self.quality = q;
        let (aa, ao, th, sky) = match q {
            Quality::Draft => (1, 16, 8, 8),
            Quality::Standard => (2, 64, 32, 32),
            Quality::High => (3, 160, 64, 64),
            Quality::Ultra => (8, 384, 128, 128),
        };
        self.aa = aa;
        self.ao_rays = ao;
        self.thickness_rays = th;
        self.sky_rays = sky;
    }

    /// Edge search distance for a mesh. Auto uses a tenth of the mesh's
    /// smallest dimension (meshes are normalized to a 2-unit box), so a thin
    /// wand gets as fine an edge mask as a crate, capped at 0.2.
    pub fn edge_distance_for(&self, mesh: &Mesh) -> f32 {
        if self.edge_auto {
            ((mesh.bounds_max - mesh.bounds_min).min_element() * 0.1).clamp(0.005, 0.2)
        } else {
            self.edge_max_distance
        }
    }

    /// Curvature contrast for a mesh. With auto edges, curvature (1 / length)
    /// is measured relative to the same detail scale, so thin meshes don't
    /// saturate the curvature map.
    pub fn curvature_scale_for(&self, mesh: &Mesh) -> f32 {
        if self.edge_auto {
            self.curvature_scale * self.edge_distance_for(mesh) / 0.2
        } else {
            self.curvature_scale
        }
    }

    pub fn total_rays_per_texel(&self) -> u32 {
        // A low-poly-only bake traces a single projection ray per texel.
        let projection = if self.high_poly.is_some() { self.aa * self.aa } else { 1 };
        // A flow walk traces about two rays per step.
        let custom: u32 = self
            .baked_custom()
            .iter()
            .map(|c| self.custom_rays(c.kind) * if c.kind == CustomKind::Flow { 64 } else { 1 })
            .sum();
        projection + self.ao_rays + self.thickness_rays + self.sky_rays + custom
    }
}

/// A source vertex stored once for the indexed bake BVH (28 bytes).
/// The scalar-array layout avoids vec3's 16-byte storage alignment.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BakeVertex {
    position: [f32; 3],
    normal_curvature: [f32; 4],
}

/// One BVH-ordered triangle: three source vertex indices plus per-triangle
/// part, ID and name-match group metadata. 24 bytes instead of three expanded
/// position and normal pairs (96 bytes) for every triangle.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BakeTriangle {
    indices: [u32; 3],
    metadata: [u32; 3],
}

/// GPU-ready bake source.
pub struct PreparedScene {
    /// Baking the low poly alone: the curvature (1/length) from which its
    /// surface counts as an edge. 0 for high polys (curvature scale and edge
    /// threshold decide).
    edge_seed_curvature: f32,
    nodes: Vec<[f32; 4]>,
    bake_vertices: Vec<BakeVertex>,
    bake_triangles: Vec<BakeTriangle>,
    /// Per-corner sRGB colors for Vertex Colors mode; empty for flat ID modes.
    tri_vcolor: Vec<[f32; 4]>,
    parts: Vec<[f32; 4]>,
    ids: Vec<[f32; 4]>,
    /// Hard-edge segments (two entries each) and their BVH: convex edges, then concave creases.
    seg_nodes: Vec<[f32; 4]>,
    segs: Vec<[f32; 4]>,
    edge_roots: [f32; 2],
    pub hard_edges: usize,
    /// Edge search distance and curvature contrast in effect (see
    /// [`BakeSettings::edge_distance_for`]).
    pub edge_max_distance: f32,
    curvature_scale: f32,
    pub triangles: usize,
    pub part_count: usize,
    pub id_regions: Vec<IdRegion>,
    pub source: String,
    pub prep_seconds: f32,
    /// Low-poly triangles sorted by name group, with (first index, index
    /// count, group) draws; `None` draws the mesh as is (no matching).
    low_draws: Option<(Vec<u32>, Vec<(u32, u32, u32)>)>,
    /// Name groups matched between low and high poly.
    pub matched_names: usize,
    /// The low-poly triangles baked: one texture set's. Rays still see the
    /// whole mesh.
    raster: std::ops::Range<u32>,
    /// Low-poly-only bakes: one vertex per low-poly corner in BVH order, with
    /// the triangle's BVH slot + 2 in the tangent w (sign = handedness).
    low_corners: Option<Vec<crate::mesh::Vertex>>,
    /// The cage as one vertex per low-poly corner (cage position, low UV).
    cage: Option<Vec<crate::mesh::Vertex>>,
    /// Largest distance from a low-poly corner to its cage corner (0 without a cage).
    cage_reach: f32,
    /// World units per UV unit on the low poly (on average): a texel's size
    /// in the world is this over the resolution.
    uv_scale: f32,
}

/// World units per UV unit, averaged over triangles `tris` of the mesh: the
/// square root of their surface area over the UV area they cover.
fn uv_scale(mesh: &Mesh, tris: std::ops::Range<u32>) -> f32 {
    let (mut area, mut uv_area) = (0.0f64, 0.0f64);
    for t in mesh.indices[tris.start as usize * 3..tris.end as usize * 3].chunks_exact(3) {
        let v = [t[0], t[1], t[2]].map(|i| &mesh.vertices[i as usize]);
        let p = v.map(|v| Vec3::from_array(v.pos));
        let uv = v.map(|v| glam::Vec2::from_array(v.uv));
        area += ((p[1] - p[0]).cross(p[2] - p[0]).length() * 0.5) as f64;
        uv_area += ((uv[1] - uv[0]).perp_dot(uv[2] - uv[0]).abs() * 0.5) as f64;
    }
    if uv_area < 1e-9 { 2.0 } else { (area / uv_area).sqrt() as f32 }
}

/// A part's name with low/high-poly suffixes and duplicate numbers removed,
/// so "Lid_low" pairs with "Lid_high" and "lid_hp.001".
pub(crate) fn base_name(name: &str) -> String {
    let mut s = name.trim().to_lowercase();
    if let Some(dot) = s.rfind('.')
        && s[dot + 1..].chars().all(|c| c.is_ascii_digit()) && dot + 1 < s.len() {
            s.truncate(dot);
        }
    for suffix in ["lowpoly", "highpoly", "low", "high", "lp", "hp", "lo", "hi"] {
        if let Some(rest) = s.strip_suffix(suffix)
            && let Some(stem) = rest.strip_suffix(['_', '-', ' ', '.'])
                && !stem.is_empty() {
                    s = stem.to_string();
                    break;
                }
    }
    s
}

/// One source ID in the baked map: a mesh part, material, or imported
/// vertex color. Vertex-color regions may share triangles when a face has
/// different colors at its corners.
#[derive(Clone, Debug)]
pub struct IdRegion {
    /// sRGB, the value an ID mask matches.
    pub color: [f32; 3],
    /// Triangles containing this ID (regions may overlap in vertex-color mode).
    pub triangles: usize,
    /// Bounds in the normalized mesh space the viewport shows.
    pub min: [f32; 3],
    pub max: [f32; 3],
}

fn hash01(x: u32) -> f32 {
    let mut h = x.wrapping_mul(0x9e3779b1) ^ 0x85ebca6b;
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b3c6d);
    h ^= h >> 12;
    (h & 0xffffff) as f32 / 16777215.0
}

/// Evenly spread, saturated ID colors (golden-ratio hue walk).
fn id_color(i: u32) -> [f32; 4] {
    let h = (i as f32 * 0.618_034).fract() * 6.0;
    let s = 0.75;
    let v = 0.95 - 0.25 * ((i / 7) % 2) as f32;
    let c = v * s;
    let x = c * (1.0 - ((h % 2.0) - 1.0).abs());
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    [r + m, g + m, b + m, 1.0]
}

/// First categorical color used by Mesh Parts and Materials ID maps.
pub fn first_id_color() -> [f32; 3] {
    let color = id_color(0);
    [color[0], color[1], color[2]]
}

/// Connected components over welded positions; returns the part of each triangle.
pub(crate) fn components(positions: &[Vec3], indices: &[u32]) -> (Vec<u32>, u32) {
    components_cancellable(positions, indices, &AtomicBool::new(false))
        .expect("uncancelled component traversal")
}

fn components_cancellable(positions: &[Vec3], indices: &[u32], cancel: &AtomicBool) -> Result<(Vec<u32>, u32), String> {
    check_prepare_cancelled(cancel)?;
    let weld = weld_positions(positions);
    check_prepare_cancelled(cancel)?;
    let mut parent: Vec<u32> = (0..positions.len() as u32).collect();
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    for (i, tri) in indices.chunks_exact(3).enumerate() {
        if i.is_multiple_of(8192) { check_prepare_cancelled(cancel)?; }
        let a = find(&mut parent, weld[tri[0] as usize] as u32);
        for &v in &tri[1..] {
            let b = find(&mut parent, weld[v as usize] as u32);
            if a != b { parent[b as usize] = a; }
        }
    }
    let mut remap = std::collections::HashMap::new();
    let mut tri_part = Vec::with_capacity(indices.len() / 3);
    for (i, tri) in indices.chunks_exact(3).enumerate() {
        if i.is_multiple_of(8192) { check_prepare_cancelled(cancel)?; }
        let root = find(&mut parent, weld[tri[0] as usize] as u32);
        let next = remap.len() as u32;
        tri_part.push(*remap.entry(root).or_insert(next));
    }
    let count = remap.len() as u32;
    Ok((tri_part, count))
}
/// A feature for the edge-distance bake: a segment, or with `half_width > 0`
/// a band around it (a bevel facet), so every point inside reads as distance 0.
/// Hard edges have `half_width` [`HARD_EDGE`]; only they draw curvature lines.
#[derive(Clone, Copy, Debug)]
pub(crate) struct EdgeSeg {
    pub ends: [Vec3; 2],
    pub part: u32,
    pub half_width: f32,
}

pub(crate) const HARD_EDGE: f32 = -1.0;

/// Feature edges of a mesh, as convex edges and concave creases with the part
/// of each edge:
/// - hard edges: welded edges whose faces meet at more than `min_angle_deg`;
/// - with `bevel_curvature` set, edges of tight bevels: the faces fold at the
///   edge and the smooth surface bends faster than that curvature across it.
///   Convexity comes from the smooth surface, not the fold, so quad diagonals
///   folding against a doubly curved surface don't read as creases.
///   Each bevel facet (the quad between two neighboring bevel edges) is also
///   added as a band, so the whole bevel is "on the edge" instead of the
///   distance rising and falling across every facet.
///
/// With `shading_normals`, a fold only counts as a hard edge where the
/// normals are split along it (a smoothing-group edge); folds of a smooth
/// shaded surface are left to its curvature.
///
/// Inline storage for a triangle's at-most-three bevel edges.
/// A sentinel marks unused pairs, keeping the per-triangle record at 24 bytes
/// and avoiding one heap allocation per triangle with bevel edges.
#[derive(Clone, Copy)]
struct TriangleBevelEdges([u32; 6]);

impl Default for TriangleBevelEdges {
    fn default() -> Self {
        Self([u32::MAX; 6])
    }
}

impl TriangleBevelEdges {
    fn push(&mut self, edge: (u32, u32)) {
        let slot = self.0.chunks_exact_mut(2)
            .find(|pair| pair[0] == u32::MAX)
            .expect("a triangle cannot have more than three edges");
        slot[0] = edge.0;
        slot[1] = edge.1;
    }

    fn iter(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.0.chunks_exact(2)
            .take_while(|pair| pair[0] != u32::MAX)
            .map(|pair| (pair[0], pair[1]))
    }
}

/// Open borders and folded-back (double-sided) faces are skipped.
#[cfg(test)]
pub(crate) fn feature_edges(
    positions: &[Vec3],
    indices: &[u32],
    tri_part: &[u32],
    min_angle_deg: f32,
    bevel_curvature: Option<f32>,
    shading_normals: Option<&[Vec3]>,
) -> [Vec<EdgeSeg>; 2] {
    feature_edges_cancellable(positions, indices, tri_part, min_angle_deg, bevel_curvature, shading_normals, &AtomicBool::new(false))
        .expect("uncancelled feature-edge preparation")
}

fn feature_edges_cancellable(
    positions: &[Vec3],
    indices: &[u32],
    tri_part: &[u32],
    min_angle_deg: f32,
    bevel_curvature: Option<f32>,
    shading_normals: Option<&[Vec3]>,
    cancel: &AtomicBool,
) -> Result<[Vec<EdgeSeg>; 2], String> {
    check_prepare_cancelled(cancel)?;
    const NONE: u32 = u32::MAX;
    let weld = weld_positions(positions);
    check_prepare_cancelled(cancel)?;
    let smooth = if bevel_curvature.is_some() { smooth_normals(positions, indices) } else { Vec::new() };
    check_prepare_cancelled(cancel)?;
    let mut face_n = Vec::with_capacity(indices.len() / 3);
    let mut center = Vec::with_capacity(indices.len() / 3);
    for (i, t) in indices.chunks_exact(3).enumerate() {
        if i.is_multiple_of(8192) { check_prepare_cancelled(cancel)?; }
        let [a, b, c] = [t[0], t[1], t[2]].map(|i| positions[i as usize]);
        face_n.push((b - a).cross(c - a).normalize_or_zero());
        center.push((a + b + c) / 3.0);
    }
    // Per welded edge, the first two faces using it: face, whether it walks
    // the edge from the lower to the higher vertex, and its opposite vertex.
    #[derive(Clone, Copy)]
    struct Side {
        tri: u32,
        forward: bool,
        opposite: u32,
    }
    let empty = Side { tri: NONE, forward: false, opposite: 0 };
    // Connected triangle meshes usually have about 1.5 unique edges per face.
    // Reserve for 2 per face instead of 3; disconnected triangle soup can still grow.
    let mut edges: std::collections::HashMap<(u32, u32), [Side; 2]> =
        std::collections::HashMap::with_capacity(indices.len().saturating_mul(2) / 3);
    for (t, tri) in indices.chunks_exact(3).enumerate() {
        if t.is_multiple_of(8192) { check_prepare_cancelled(cancel)?; }
        for k in 0..3 {
            let a = weld[tri[k] as usize] as u32;
            let b = weld[tri[(k + 1) % 3] as usize] as u32;
            if a == b {
                continue;
            }
            let slot = edges.entry((a.min(b), a.max(b))).or_insert([empty; 2]);
            if let Some(s) = slot.iter_mut().find(|s| s.tri == NONE) {
                *s = Side { tri: t as u32, forward: a < b, opposite: tri[(k + 2) % 3] };
            }
        }
    }
    let cos_hard = min_angle_deg.to_radians().cos();
    let cos_fold = 170f32.to_radians().cos();
    // Edges folding less than this are flat or ruled-surface diagonals.
    let cos_bend = 2f32.to_radians().cos();
    let mut out = [Vec::new(), Vec::new()];
    // Bevel edges with their convexity, and the bevel edges of each triangle.
    let mut bevel: std::collections::HashMap<(u32, u32), bool> = std::collections::HashMap::new();
    // Every triangle has only three edges. Store them inline rather than making
    // a separate heap allocation for each triangle that contains a bevel edge.
    let mut tri_bevels = vec![TriangleBevelEdges::default(); face_n.len()];
    for (edge_index, (&(a, b), &[s0, s1])) in edges.iter().enumerate() {
        if edge_index.is_multiple_of(8192) { check_prepare_cancelled(cancel)?; }
        if s1.tri == NONE {
            continue;
        }
        let (t0, t1) = (s0.tri as usize, s1.tri as usize);
        let na = face_n[t0];
        // Consistently wound neighbors walk a shared edge in opposite directions.
        let nb = if s0.forward == s1.forward { -face_n[t1] } else { face_n[t1] };
        let cos = na.dot(nb);
        if na == Vec3::ZERO || nb == Vec3::ZERO || cos < cos_fold {
            continue;
        }
        let mut hard = cos <= cos_hard;
        if let (true, Some(ns)) = (hard, shading_normals) {
            // The two faces' own normals at an edge corner.
            let corner = |t: usize, w: u32| {
                indices[t * 3..t * 3 + 3].iter().find(|&&i| weld[i as usize] as u32 == w).map(|&i| ns[i as usize])
            };
            let split = |w: u32| match (corner(t0, w), corner(t1, w)) {
                (Some(x), Some(y)) => x.dot(y) < 0.999,
                _ => true,
            };
            if !split(a) && !split(b) {
                if bevel_curvature.is_none() {
                    continue;
                }
                hard = false;
            }
        }
        let convex = if hard {
            nb.dot(center[t0] - center[t1]) < 0.0
        } else {
            let Some(k_min) = bevel_curvature else { continue };
            if cos > cos_bend {
                continue;
            }
            // Normal curvature across the edge, between the two opposite
            // vertices; only their offset across the edge counts, since thin
            // triangles put them mostly along it.
            let (oa, ob) = (s0.opposite as usize, s1.opposite as usize);
            let e = (positions[b as usize] - positions[a as usize]).normalize_or_zero();
            let dp = positions[ob] - positions[oa];
            let across = dp - e * dp.dot(e);
            let k = (smooth[ob] - smooth[oa]).dot(across) / across.length_squared().max(1e-20);
            if k.abs() < k_min {
                continue;
            }
            k > 0.0
        };
        let half_width = if hard { HARD_EDGE } else { 0.0 };
        out[if convex { 0 } else { 1 }].push(EdgeSeg {
            ends: [positions[a as usize], positions[b as usize]],
            part: tri_part[t0],
            half_width,
        });
        if !hard {
            bevel.insert((a, b), convex);
            tri_bevels[t0].push((a, b));
            tri_bevels[t1].push((a, b));
        }
    }
    // Bevel facets: two triangles sharing a non-bevel edge (the quad
    // diagonal), each holding a different bevel edge, the two edges roughly
    // parallel and spanning the quad's four corners.
    for (edge_index, (&(a, b), &[s0, s1])) in edges.iter().enumerate() {
        if edge_index.is_multiple_of(8192) { check_prepare_cancelled(cancel)?; }
        if s1.tri == NONE || bevel.contains_key(&(a, b)) {
            continue;
        }
        let (ta, tb) = (s0.tri as usize, s1.tri as usize);
        for ea in tri_bevels[ta].iter() {
            for eb in tri_bevels[tb].iter() {
                let corners = [ea.0, ea.1, eb.0, eb.1];
                let distinct = (0..4).all(|i| (i + 1..4).all(|j| corners[i] != corners[j]));
                if !distinct || bevel[&ea] != bevel[&eb] {
                    continue;
                }
                let pa = [positions[ea.0 as usize], positions[ea.1 as usize]];
                let mut pb = [positions[eb.0 as usize], positions[eb.1 as usize]];
                let (da, db) = ((pa[1] - pa[0]).normalize_or_zero(), (pb[1] - pb[0]).normalize_or_zero());
                if da.dot(db).abs() < 0.8 {
                    continue;
                }
                if da.dot(db) < 0.0 {
                    pb.swap(0, 1);
                }
                let half_width = pa[0].distance(pb[0]).max(pa[1].distance(pb[1])) * 0.5;
                out[if bevel[&ea] { 0 } else { 1 }].push(EdgeSeg {
                    ends: [(pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5],
                    part: tri_part[ta],
                    half_width,
                });
            }
        }
    }
    check_prepare_cancelled(cancel)?;
    Ok(out)
}
/// Mean curvature at each vertex of a mesh, from how its shading normals
/// turn toward its neighbors: (n_j - n_i)·(p_j - p_i) / |p_j - p_i|²,
/// averaged. Vertices that share a position and (within a degree) a normal
/// are one sample: copies along UV seams weld; split normals (hard edges)
/// stay apart, so flat faces read as flat.
///
/// The result is then smoothed over the welded mesh (neighbors weighted by
/// distance), which removes the fan pattern a coarse mesh's triangles leave
/// when the values are interpolated, without ever seeing a UV seam: meshes
/// cut into many small UV islands smooth as one surface.
///
/// Interpolated across each triangle this gives smooth, continuous
/// curvature. Measuring it per texel from interpolated normals instead
/// gives one constant per triangle, which shows the triangles on a dense
/// mesh as faceted steps.
///
/// `limit` is the curvature the map saturates at: values past it look the
/// same in the map but would swamp their neighbors when smoothing (a thin
/// edge's curvature is huge), so they're clamped first.
pub(crate) fn vertex_curvature(positions: &[Vec3], normals: &[Vec3], indices: &[u32], limit: f32) -> Vec<f32> {
    let ids = weld_groups(positions, normals);
    let groups = ids.iter().map(|&g| g as usize + 1).max().unwrap_or(0);
    let mut sum = vec![0.0f32; groups];
    let mut count = vec![0u32; groups];
    // Welded edges (both directions) with their length, for smoothing.
    let mut edges: Vec<(u32, u32, f32)> = Vec::with_capacity(indices.len() * 2);
    for tri in indices.chunks_exact(3) {
        for k in 0..3 {
            let (i, j) = (tri[k] as usize, tri[(k + 1) % 3] as usize);
            let d = positions[j] - positions[i];
            let l2 = d.length_squared();
            if l2 < 1e-14 {
                continue;
            }
            // The edge seen from both ends.
            let k_ij = (normals[j] - normals[i]).dot(d) / l2;
            for v in [i, j] {
                sum[ids[v] as usize] += k_ij;
                count[ids[v] as usize] += 1;
            }
            let (gi, gj) = (ids[i], ids[j]);
            if gi != gj {
                let l = l2.sqrt();
                edges.push((gi, gj, l));
                edges.push((gj, gi, l));
            }
        }
    }
    let mut k: Vec<f32> = sum.iter().zip(&count).map(|(s, &c)| (s / c.max(1) as f32).clamp(-limit, limit)).collect();
    // Two diffusion passes; each neighbor weighs like a Gaussian of its
    // distance relative to the vertex's own edge length.
    let mut mean_len = vec![0.0f32; groups];
    let mut n_len = vec![0u32; groups];
    for &(a, _, l) in &edges {
        mean_len[a as usize] += l;
        n_len[a as usize] += 1;
    }
    for (m, &n) in mean_len.iter_mut().zip(&n_len) {
        *m /= n.max(1) as f32;
    }
    for _ in 0..2 {
        let mut acc: Vec<f32> = k.clone();
        let mut wsum = vec![1.0f32; groups];
        for &(a, b, l) in &edges {
            let s = mean_len[a as usize].max(1e-9);
            let w = (-0.5 * (l / s) * (l / s)).exp();
            acc[a as usize] += k[b as usize] * w;
            wsum[a as usize] += w;
        }
        k = acc.iter().zip(&wsum).map(|(a, w)| a / w).collect();
    }
    ids.iter().map(|&g| k[g as usize]).collect()
}

/// The first normal group for a welded position is inline. Most positions
/// have only one group, so this avoids one small heap allocation per vertex;
/// extra allocations occur only where split normals share the same position.
struct NormalGroupBucket {
    first: (Vec3, u32),
    additional: Vec<(Vec3, u32)>,
}

impl NormalGroupBucket {
    fn new(normal: Vec3, group: u32) -> Self {
        Self { first: (normal, group), additional: Vec::new() }
    }

    fn find_or_insert(&mut self, normal: Vec3, next: &mut u32) -> u32 {
        if self.first.0.dot(normal) > 0.99985 {
            return self.first.1;
        }
        if let Some(&(_, group)) = self.additional.iter().find(|(candidate, _)| candidate.dot(normal) > 0.99985) {
            return group;
        }
        let group = *next;
        *next += 1;
        self.additional.push((normal, group));
        group
    }
}

/// Welds vertices for curvature: same position (to 1e-4 of the unit-sized
/// mesh) and normals within about a degree share a group.
fn weld_groups(positions: &[Vec3], normals: &[Vec3]) -> Vec<u32> {
    let mut by_pos: HashMap<[i32; 3], NormalGroupBucket> = HashMap::with_capacity(positions.len());
    let mut next = 0u32;
    positions
        .iter()
        .zip(normals)
        .map(|(p, n)| {
            let key = (*p * 1.0e4).round().as_ivec3().to_array();
            let normal = n.normalize_or_zero();
            let bucket = by_pos.entry(key).or_insert_with(|| {
                let group = next;
                next += 1;
                NormalGroupBucket::new(normal, group)
            });
            bucket.find_or_insert(normal, &mut next)
        })
        .collect()
}

/// Packs edge-segment sets into one node and segment buffer: a BVH per set,
/// nodes and leaves offset so every set shares the buffers.
fn pack_segments(sets: [Vec<EdgeSeg>; 2]) -> (Vec<[f32; 4]>, Vec<[f32; 4]>, [f32; 2]) {
    let mut nodes = Vec::new();
    let mut segs = Vec::new();
    let mut roots = [-1.0; 2];
    for (k, set) in sets.into_iter().enumerate() {
        if set.is_empty() {
            continue;
        }
        // Boxes grow by the band half-width, so culling never skips a band.
        let bvh = Bvh::build(
            set.iter()
                .map(|s| {
                    let r = Vec3::splat(s.half_width.max(0.0));
                    let (lo, hi) = (s.ends[0].min(s.ends[1]) - r, s.ends[0].max(s.ends[1]) + r);
                    [lo, hi, lo]
                })
                .collect(),
        );
        let node_base = (nodes.len() / 2) as u32;
        let seg_base = (segs.len() / 2) as u32;
        roots[k] = node_base as f32;
        for n in &bvh.nodes {
            let first = if n.count > 0 { n.first + seg_base } else { n.first + node_base };
            nodes.push([n.min.x, n.min.y, n.min.z, f32::from_bits(first)]);
            nodes.push([n.max.x, n.max.y, n.max.z, f32::from_bits(n.count)]);
        }
        for &i in &bvh.order {
            let s = set[i as usize];
            let [a, b] = s.ends;
            segs.push([a.x, a.y, a.z, s.part as f32]);
            segs.push([b.x, b.y, b.z, s.half_width]);
        }
    }
    (nodes, segs, roots)
}

/// Low-poly indices of triangles `tris` sorted by group, plus one `(first
/// index, count, group)` draw per run: the UV raster draws each group as its
/// own instance.
fn group_draws(indices: &[u32], tris: std::ops::Range<usize>, group: impl Fn(usize) -> u32) -> (Vec<u32>, Vec<(u32, u32, u32)>) {
    let mut order: Vec<usize> = tris.collect();
    order.sort_by_key(|&t| group(t));
    let mut sorted = Vec::with_capacity(indices.len());
    let mut draws: Vec<(u32, u32, u32)> = Vec::new();
    for t in order {
        let g = group(t);
        match draws.last_mut() {
            Some(d) if d.2 == g => d.1 += 3,
            _ => draws.push((sorted.len() as u32, 3, g)),
        }
        sorted.extend_from_slice(&indices[t * 3..t * 3 + 3]);
    }
    (sorted, draws)
}

/// Builds the bake source on a worker thread: loads the high poly if any,
/// builds the BVH, finds mesh parts and assigns IDs.
/// A-trous denoise passes for an AO ray count: 4 at Draft (16 rays) down to
/// 1 at Ultra (384), keeping the leftover noise about the same.
fn denoise_passes(ao_rays: u32) -> u32 {
    match ao_rays {
        0..=32 => 4,
        33..=96 => 3,
        97..=256 => 2,
        _ => 1,
    }
}

#[inline]
fn check_prepare_cancelled(cancel: &AtomicBool) -> Result<(), String> {
    if cancel.load(Ordering::Relaxed) {
        Err("Bake preparation cancelled".into())
    } else {
        Ok(())
    }
}

/// Rejects source combinations that cannot produce IDs matching the visible
/// low-poly palette. This runs before both cache lookup and CPU preparation.
pub fn validate_settings(settings: &BakeSettings) -> Result<(), String> {
    if settings.id_source == IdSource::VertexColors && settings.high_poly.is_some() {
        return Err("Vertex Colors IDs come from the loaded low-poly mesh, while a high-poly projection uses a different triangle topology. Finish the high-poly bake with Mesh parts or Materials, then use the Vertex ID panel to rebake only the ID map from the low-poly colors.".into());
    }
    Ok(())
}

/// Counts the mesh's original per-corner vertex colors and returns the exact
/// top palette used by both the Vertex ID panel and the bake report. The common
/// path is optimized for authored palettes; high-cardinality meshes switch to
/// compact sorted RGB keys and retain only the top entries.
pub fn vertex_color_palette(colors: &[[[u8; 3]; 3]]) -> (Vec<([u8; 3], u32)>, usize) {
    vertex_color_palette_impl(colors, None).expect("uncancellable palette scan")
}

/// Cancellable palette scan used by CPU bake preparation.
fn vertex_color_palette_cancellable(
    colors: &[[[u8; 3]; 3]],
    cancel: &AtomicBool,
) -> Result<(Vec<([u8; 3], u32)>, usize), String> {
    vertex_color_palette_impl(colors, Some(cancel))
}

fn vertex_color_palette_impl(
    colors: &[[[u8; 3]; 3]],
    cancel: Option<&AtomicBool>,
) -> Result<(Vec<([u8; 3], u32)>, usize), String> {
    const HASH_DISTINCT_LIMIT: usize = MAX_VERTEX_COLOR_ID_REGIONS * 8;
    let check_cancelled = || cancel.map_or(Ok(()), check_prepare_cancelled);
    let order = |a: &([u8; 3], u32), b: &([u8; 3], u32)| b.1.cmp(&a.1).then(a.0.cmp(&b.0));
    let mut counts: HashMap<[u8; 3], u32> = HashMap::with_capacity(HASH_DISTINCT_LIMIT);
    for (triangle, corners) in colors.iter().enumerate() {
        if triangle.is_multiple_of(8192) {
            check_cancelled()?;
        }
        for color in corners {
            {
                let count = counts.entry(*color).or_default();
                *count = count.saturating_add(1);
            }
            if counts.len() > HASH_DISTINCT_LIMIT {
                return vertex_color_palette_sorted(colors, cancel);
            }
        }
    }

    let distinct = counts.len();
    let mut palette: Vec<_> = counts.into_iter().collect();
    if palette.len() > MAX_VERTEX_COLOR_ID_REGIONS {
        palette.select_nth_unstable_by(MAX_VERTEX_COLOR_ID_REGIONS, order);
        palette.truncate(MAX_VERTEX_COLOR_ID_REGIONS);
    }
    palette.sort_unstable_by(order);
    Ok((palette, distinct))
}

/// High-cardinality fallback. RGB is packed into one u32, sorted in place and
/// counted by runs; a bounded heap retains only the 48 most-used labels.
fn vertex_color_palette_sorted(
    colors: &[[[u8; 3]; 3]],
    cancel: Option<&AtomicBool>,
) -> Result<(Vec<([u8; 3], u32)>, usize), String> {
    let check_cancelled = || cancel.map_or(Ok(()), check_prepare_cancelled);
    let mut packed = Vec::with_capacity(colors.len().saturating_mul(3));
    for (triangle, corners) in colors.iter().enumerate() {
        if triangle.is_multiple_of(8192) {
            check_cancelled()?;
        }
        for rgb in corners {
            packed.push(((rgb[0] as u32) << 16) | ((rgb[1] as u32) << 8) | rgb[2] as u32);
        }
    }
    check_cancelled()?;
    packed.sort_unstable();
    check_cancelled()?;

    // Heap key is (frequency, inverted RGB). Higher frequency wins; for ties,
    // the smaller original RGB value wins deterministically. Reverse keeps the
    // worst retained entry at the root for O(log MAX_VERTEX_COLOR_ID_REGIONS) replacement.
    let mut top: BinaryHeap<Reverse<(u32, [u8; 3])>> =
        BinaryHeap::with_capacity(MAX_VERTEX_COLOR_ID_REGIONS + 1);
    let mut distinct = 0usize;
    let mut start = 0usize;
    while start < packed.len() {
        if distinct.is_multiple_of(8192) {
            check_cancelled()?;
        }
        let key = packed[start];
        let mut end = start + 1;
        while end < packed.len() && packed[end] == key {
            if end.is_multiple_of(8192) {
                check_cancelled()?;
            }
            end += 1;
        }
        distinct += 1;
        let rgb = [(key >> 16) as u8, (key >> 8) as u8, key as u8];
        let corners = u32::try_from(end - start).unwrap_or(u32::MAX);
        let inverse_rgb = rgb.map(|channel| 255 - channel);
        let candidate = (corners, inverse_rgb);
        if top.len() < MAX_VERTEX_COLOR_ID_REGIONS {
            top.push(Reverse(candidate));
        } else if top.peek().is_some_and(|Reverse(worst)| candidate > *worst) {
            top.pop();
            top.push(Reverse(candidate));
        }
        start = end;
    }

    let mut palette: Vec<_> = top
        .into_iter()
        .map(|Reverse((corners, inverse_rgb))| {
            (inverse_rgb.map(|channel| 255 - channel), corners)
        })
        .collect();
    palette.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    Ok((palette, distinct))
}

/// Keeps only the shared top palette entries and calculates their region bounds.
/// The baked texture still receives every raw corner color; only the report/MCP
/// index is bounded.
fn vertex_color_id_data(
    colors: &[[[u8; 3]; 3]],
    indices: &[u32],
    positions: &[Vec3],
    cancel: &AtomicBool,
) -> Result<(Vec<[f32; 4]>, Vec<IdRegion>), String> {
    let (palette, _) = vertex_color_palette_cancellable(colors, cancel)?;
    let ids: Vec<[f32; 4]> = palette.iter().map(|(rgb, _)| {
        [rgb[0] as f32 / 255.0, rgb[1] as f32 / 255.0, rgb[2] as f32 / 255.0, 1.0]
    }).collect();
    let lookup: HashMap<[u8; 3], usize> = palette.iter().enumerate().map(|(index, (rgb, _))| (*rgb, index)).collect();
    let mut regions: Vec<IdRegion> = ids.iter()
        .map(|color| IdRegion {
            color: [color[0], color[1], color[2]],
            triangles: 0,
            min: [f32::MAX; 3],
            max: [f32::MIN; 3],
        })
        .collect();

    for (triangle, (indices, corners)) in indices.chunks_exact(3).zip(colors).enumerate() {
        if triangle.is_multiple_of(8192) {
            check_prepare_cancelled(cancel)?;
        }
        let mut seen = [usize::MAX; 3];
        let mut seen_count = 0usize;
        for corner in 0..3 {
            let Some(&region) = lookup.get(&corners[corner]) else { continue };
            if !seen[..seen_count].contains(&region) {
                seen[seen_count] = region;
                seen_count += 1;
                regions[region].triangles += 1;
            }
            let position = positions[indices[corner] as usize].to_array();
            for axis in 0..3 {
                regions[region].min[axis] = regions[region].min[axis].min(position[axis]);
                regions[region].max[axis] = regions[region].max[axis].max(position[axis]);
            }
        }
    }
    Ok((ids, regions))
}

/// Prepares baking texture set `set` of `low` (its triangles get maps; the
/// whole mesh occludes).
pub fn prepare(low: &Mesh, settings: &BakeSettings, set: usize) -> Result<PreparedScene, String> {
    prepare_cancellable(low, settings, set, &AtomicBool::new(false))
}

/// CPU Bake preparation can be abandoned when a newer request supersedes it.
pub fn prepare_cancellable(
    low: &Mesh,
    settings: &BakeSettings,
    set: usize,
    cancel: &AtomicBool,
) -> Result<PreparedScene, String> {
    check_prepare_cancelled(cancel)?;
    if settings.id_source == IdSource::VertexColors && settings.high_poly.is_some() {
        return Err(
            "Vertex Colors IDs come from the loaded low-poly mesh, while a high-poly projection uses a different triangle topology. Finish the high-poly bake with Mesh parts or Materials, then use the Vertex ID panel to rebake only the ID map from the low-poly colors.".into()
        );
    }
    let start = Instant::now();
    let low_tris = low.indices.len() / 3;
    let raster = low.sets.get(set).map_or(0..low_tris as u32, |s| s.tris.clone());
    let (positions, normals, indices, tri_material, material_count, source, src_objects, tri_vertex_color) = match &settings.high_poly {
        Some(path) => {
            let mut hp = HighPoly::load(path, low.center, low.scale)?;
            check_prepare_cancelled(cancel)?;
            // Normalize in place instead of retaining a second normal array.
            for (i, normal) in hp.normals.iter_mut().enumerate() {
                if i.is_multiple_of(16384) {
                    check_prepare_cancelled(cancel)?;
                }
                *normal = normal.normalize_or_zero();
            }
            let objects = if settings.match_by_name && !low.objects.is_empty() && hp.tri_object.len() == hp.indices.len() / 3 {
                Some((std::mem::take(&mut hp.tri_object), std::mem::take(&mut hp.objects)))
            } else {
                None
            };
            let tri_material = if settings.id_source == IdSource::Materials {
                let triangle_count = hp.indices.len() / 3;
                if hp.tri_material.len() == triangle_count {
                    hp.tri_material
                } else if hp.material_count <= 1 {
                    vec![0; triangle_count]
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };
            let tri_vertex_color = if settings.id_source == IdSource::VertexColors {
                hp.tri_vertex_color
            } else {
                None
            };
            (
                hp.positions,
                hp.normals,
                Cow::Owned(hp.indices),
                tri_material,
                hp.material_count,
                format!("{} (high poly)", hp.name),
                objects,
                tri_vertex_color,
            )
        }
        None => {
            let positions: Vec<Vec3> = low.vertices.iter().map(|v| Vec3::from_array(v.pos)).collect();
            let normals: Vec<Vec3> = low.vertices.iter().map(|v| Vec3::from_array(v.normal)).collect();
            check_prepare_cancelled(cancel)?;
            // Material IDs are only needed when explicitly selected. Vertex
            // color payloads are copied only for Vertex Color ID mode.
            let count = low.sets.len().max(1) as u32;
            let mut tri_material = if settings.id_source == IdSource::Materials {
                vec![0; low_tris]
            } else {
                Vec::new()
            };
            if !tri_material.is_empty() {
                for (k, s) in low.sets.iter().enumerate() {
                    tri_material[s.tris.start as usize..s.tris.end as usize].fill(k as u32);
                }
            }
            let tri_vertex_color = if settings.id_source == IdSource::VertexColors {
                low.tri_vertex_color.clone()
            } else {
                None
            };
            (positions, normals, Cow::Borrowed(&low.indices), tri_material, count, format!("{} (low poly)", low.name), None, tri_vertex_color)
        }
    };
    check_prepare_cancelled(cancel)?;
    let triangle_count = indices.len() / 3;
    if settings.id_source == IdSource::VertexColors
        && tri_vertex_color.as_ref().is_none_or(|colors| colors.len() != triangle_count)
    {
        return Err(format!(
            "Vertex Colors was selected as the ID source, but the chosen bake source ({source}) has no complete vertex-color data. Add a color attribute to that mesh or choose Mesh parts/Materials."
        ));
    }

    // Name groups: 1-based per distinct low-poly base name; 0 = unmatched
    // (an unmatched low part sees everything, an unmatched high part is seen
    // by everyone).
    let mut src_group = Vec::new();
    let mut low_draws = None;
    let mut matched_names = 0;
    if let (true, Some((src_tri_object, src_names))) = (settings.match_by_name, &src_objects)
        && !low.objects.is_empty() && low.tri_object.len() == low_tris && src_tri_object.len() == indices.len() / 3 {
            let mut groups: HashMap<String, u32> = HashMap::new();
            for (i, name) in low.objects.iter().enumerate() {
                if i.is_multiple_of(1024) {
                    check_prepare_cancelled(cancel)?;
                }
                let n = groups.len() as u32 + 1;
                groups.entry(base_name(name)).or_insert(n);
            }
            let src_by_object: Vec<u32> = src_names.iter().map(|n| groups.get(&base_name(n)).copied().unwrap_or(0)).collect();
            let matched: std::collections::HashSet<u32> = src_by_object.iter().copied().filter(|&g| g > 0).collect();
            matched_names = matched.len();
            if matched_names > 0 {
                src_group = src_tri_object.iter().map(|&o| src_by_object[o as usize]).collect();
                let low_by_object: Vec<u32> =
                    low.objects.iter().map(|n| groups[&base_name(n)]).map(|g| if matched.contains(&g) { g } else { 0 }).collect();
                let tris = raster.start as usize..raster.end as usize;
                low_draws = Some(group_draws(&low.indices, tris, |t| low_by_object[low.tri_object[t] as usize]));
            }
        }
    // Object IDs and names are only preparation inputs; don't retain them
    // alongside the expanded per-corner GPU payloads.
    drop(src_objects);
    check_prepare_cancelled(cancel)?;

    let cage = match &settings.cage {
        Some(path) if settings.high_poly.is_some() => {
            let cage = HighPoly::load(path, low.center, low.scale)?;
            check_prepare_cancelled(cancel)?;
            if cage.indices.len() != low.indices.len() {
                return Err(format!(
                    "The cage has {} triangles but the low poly has {low_tris}; a cage must be the low poly with its vertices moved.",
                    cage.indices.len() / 3
                ));
            }
            let corners: Vec<crate::mesh::Vertex> = low
                .indices
                .iter()
                .zip(&cage.indices)
                .map(|(&l, &c)| crate::mesh::Vertex { pos: cage.positions[c as usize].to_array(), ..low.vertices[l as usize] })
                .collect();
            Some(corners)
        }
        _ if settings.builds_cage() => {
            // The bake's own cage, by direction and distance rather than as
            // a shell: rays search along the interpolated projection
            // direction (averaged: shared by split copies of a vertex, so
            // the cage stays closed over hard edges), starting front times
            // the painted multiplier out. Kept apart, a painted spot only
            // moves where its rays start, never which way they point. The
            // corners carry the direction in `pos` and the distance in the
            // tangent's x (see `cs_surface`).
            let dirs = crate::cage::projection_dirs(low, settings.average_normals);
            check_prepare_cancelled(cancel)?;
            let paint = settings.cage_paint.as_deref().filter(|p| p.fits(low.vertices.len()));
            let corners: Vec<crate::mesh::Vertex> = low
                .indices
                .iter()
                .map(|&l| {
                    let v = low.vertices[l as usize];
                    let reach = (settings.front * paint.map_or(1.0, |p| p.at(l as usize))).min(MAX_CAGE_REACH);
                    crate::mesh::Vertex { pos: dirs[l as usize].to_array(), tangent: [reach, 0.0, 0.0, v.tangent[3]], ..v }
                })
                .collect();
            Some(corners)
        }
        _ => None,
    };
    check_prepare_cancelled(cancel)?;
    let cage_reach = match &cage {
        Some(corners) if settings.builds_cage() => corners.iter().map(|c| c.tangent[0]).fold(0.0, f32::max),
        Some(corners) => corners
            .iter()
            .zip(&low.indices)
            .map(|(c, &l)| Vec3::from_array(c.pos).distance(Vec3::from_array(low.vertices[l as usize].pos)))
            .fold(0.0, f32::max),
        None => 0.0,
    };

    let (tri_part, part_count) = components_cancellable(&positions, &indices, cancel)?;
    check_prepare_cancelled(cancel)?;
    let mut part_lo = vec![f32::MAX; part_count as usize];
    let mut part_hi = vec![f32::MIN; part_count as usize];
    for (t, tri) in indices.chunks_exact(3).enumerate() {
        if t.is_multiple_of(8192) {
            check_prepare_cancelled(cancel)?;
        }
        let p = tri_part[t] as usize;
        for &v in tri {
            part_lo[p] = part_lo[p].min(positions[v as usize].y);
            part_hi[p] = part_hi[p].max(positions[v as usize].y);
        }
    }
    let parts: Vec<[f32; 4]> = (0..part_count as usize)
        .map(|p| [hash01(p as u32 + 1), part_lo[p], (part_hi[p] - part_lo[p]).max(1e-5), 0.0])
        .collect();
    let material_count = material_count.max(1);
    let use_materials = settings.id_source == IdSource::Materials && tri_material.len() == triangle_count;
    let (tri_id, ids, id_regions): (Cow<'_, [u32]>, Vec<[f32; 4]>, Vec<IdRegion>) =
        if settings.id_source == IdSource::VertexColors {
            let colors = tri_vertex_color.as_ref().expect("vertex-color source was validated above");
            let (ids, regions) = vertex_color_id_data(colors, &indices, &positions, cancel)?;
            // The shader reads the original per-corner colors directly. The
            // flat ID slot is unused in this mode, so borrow part IDs rather
            // than allocating one extra index per triangle.
            (Cow::Borrowed(&tri_part), ids, regions)
        } else {
            let (tri_id, ids): (Cow<'_, [u32]>, Vec<[f32; 4]>) = if use_materials {
                (Cow::Owned(tri_material), (0..material_count).map(id_color).collect())
            } else {
                (Cow::Borrowed(&tri_part), (0..part_count).map(id_color).collect())
            };
            let mut regions: Vec<IdRegion> = ids
                .iter()
                .map(|color| IdRegion {
                    color: [color[0], color[1], color[2]],
                    triangles: 0,
                    min: [f32::MAX; 3],
                    max: [f32::MIN; 3],
                })
                .collect();
            for (triangle, tri) in indices.chunks_exact(3).enumerate() {
                if triangle.is_multiple_of(8192) {
                    check_prepare_cancelled(cancel)?;
                }
                let region = &mut regions[tri_id[triangle] as usize];
                region.triangles += 1;
                for &vertex in tri {
                    let position = positions[vertex as usize].to_array();
                    for axis in 0..3 {
                        region.min[axis] = region.min[axis].min(position[axis]);
                        region.max[axis] = region.max[axis].max(position[axis]);
                    }
                }
            }
            (tri_id, ids, regions)
        };

    // Baking needs node bounds and triangle order, not a second CPU triangle
    // array. Build those directly from the source indices.
    check_prepare_cancelled(cancel)?;
    let (bvh_nodes, bvh_order) = Bvh::build_ordered(&positions, &indices);
    check_prepare_cancelled(cancel)?;

    // Curvature comes from the source mesh's vertices (see
    // `vertex_curvature`) and rides in the normal's w.
    let curvature = vertex_curvature(&positions, &normals, &indices, 1.0 / settings.curvature_scale_for(low).max(1e-6));
    check_prepare_cancelled(cancel)?;
    let mut bake_triangles = Vec::with_capacity(bvh_order.len());
    // Flat mesh-part/material IDs come from the compact palette. Only true
    // vertex-color mode retains a separate per-corner color payload.
    let has_vertex_colors = settings.id_source == IdSource::VertexColors && tri_vertex_color.as_ref().is_some_and(|colors| !colors.is_empty());
    let mut tri_vcolor = if has_vertex_colors { Vec::with_capacity(bvh_order.len() * 3) } else { Vec::new() };
    for (i, &t) in bvh_order.iter().enumerate() {
        if i.is_multiple_of(8192) {
            check_prepare_cancelled(cancel)?;
        }
        let t = t as usize;
        let tri = &indices[t * 3..t * 3 + 3];
        bake_triangles.push(BakeTriangle {
            indices: [tri[0], tri[1], tri[2]],
            metadata: [tri_part[t], tri_id[t], src_group.get(t).copied().unwrap_or(0)],
        });
        if has_vertex_colors {
            let flat_id = ids.get(tri_id[t] as usize).copied().unwrap_or([0.5, 0.5, 0.5, 1.0]);
            for k in 0..3 {
                let color = tri_vertex_color.as_ref().and_then(|colors| colors.get(t)).map(|tri| tri[k])
                    .map(|rgb| [rgb[0] as f32 / 255.0, rgb[1] as f32 / 255.0, rgb[2] as f32 / 255.0, 1.0])
                    .unwrap_or(flat_id);
                tri_vcolor.push(color);
            }
        }
    }
    // These source payloads have been reduced to the final GPU arrays/palette.
    // Release them before feature-edge construction allocates its own scratch.
    drop(tri_id);
    drop(tri_vertex_color);
    drop(src_group);

    // A low poly's interpolated normals give unreliable texel curvature, so
    // its bevels come from the mesh too. High polys keep texel curvature,
    // which follows their real surface detail.
    // A bevel is an edge when its radius is within reach of the edge search.
    let edge_max_distance = settings.edge_distance_for(low);
    let bevels = settings.high_poly.is_none().then(|| 1.0 / (1.25 * edge_max_distance));
    // Baking the low poly alone, its edges are traced from its (vertex)
    // curvature like a high poly's, at the curvature the old mesh-bevel test
    // used; only split-normal hard edges come from the mesh. Mesh bands
    // traced every bent triangle edge of a dense mesh, which showed as
    // staircases and scalloped mask borders.
    let low_only = settings.high_poly.is_none();
    check_prepare_cancelled(cancel)?;
    let [convex, concave] = feature_edges_cancellable(
        &positions, &indices, &tri_part, settings.hard_edge_angle, None,
        low_only.then_some(&normals[..]), cancel,
    )?;
    check_prepare_cancelled(cancel)?;
    // The remaining preparation stages use only packed data and the original low mesh.
    drop(tri_part);
    let edge_seed_curvature = bevels.unwrap_or(0.0);
    let hard_edge_count = convex.iter().chain(&concave).filter(|s| s.half_width <= 0.0).count();
    let (seg_nodes, segs, edge_roots) = pack_segments([convex, concave]);

    // Materialize one packed record per source vertex, not one per triangle
    // corner. Defer this until feature-edge preparation has released its
    // scratch arrays, then discard the source CPU arrays immediately.
    let mut bake_vertices = Vec::with_capacity(positions.len());
    for (i, p) in positions.iter().enumerate() {
        if i.is_multiple_of(16384) {
            check_prepare_cancelled(cancel)?;
        }
        let n = normals.get(i).copied().unwrap_or(Vec3::ZERO);
        bake_vertices.push(BakeVertex {
            position: [p.x, p.y, p.z],
            normal_curvature: [n.x, n.y, n.z, curvature.get(i).copied().unwrap_or(0.0)],
        });
    }
    drop(curvature);
    drop(positions);
    drop(normals);
    drop(indices);

    // Baking the low poly alone, every texel already lies on a known low-poly
    // triangle. The raster records it (as the BVH slot, in the tangent w,
    // whose sign stays the handedness) so the surface pass never has to search
    // for it: a search ray at a seam where parts intersect can land on the
    // other part. f32 holds slots exactly below 2^24.
    let low_corners = if settings.high_poly.is_none() && bvh_order.len() < (1 << 24) - 2 {
        let mut corners = Vec::with_capacity(bvh_order.len() * 3);
        for (slot, &t) in bvh_order.iter().enumerate() {
            if slot.is_multiple_of(8192) {
                check_prepare_cancelled(cancel)?;
            }
            if !raster.contains(&t) {
                continue;
            }
            for &i in &low.indices[t as usize * 3..t as usize * 3 + 3] {
                let mut v = low.vertices[i as usize];
                v.tangent[3] = (slot as f32 + 2.0).copysign(v.tangent[3]);
                corners.push(v);
            }
        }
        Some(corners)
    } else {
        None
    };

    check_prepare_cancelled(cancel)?;
    let mut nodes = Vec::with_capacity(bvh_nodes.len() * 2);
    for n in &bvh_nodes {
        nodes.push([n.min.x, n.min.y, n.min.z, f32::from_bits(n.first)]);
        nodes.push([n.max.x, n.max.y, n.max.z, f32::from_bits(n.count)]);
    }

    check_prepare_cancelled(cancel)?;
    Ok(PreparedScene {
        nodes,
        bake_vertices,
        bake_triangles,
        tri_vcolor,
        parts,
        ids,
        seg_nodes,
        segs,
        edge_roots,
        hard_edges: hard_edge_count,
        edge_max_distance,
        curvature_scale: settings.curvature_scale_for(low),
        triangles: triangle_count,
        part_count: part_count as usize,
        id_regions,
        source,
        prep_seconds: start.elapsed().as_secs_f32(),
        low_draws,
        low_corners,
        edge_seed_curvature,
        matched_names,
        cage,
        cage_reach,
        uv_scale: uv_scale(low, raster.clone()),
        raster,
    })
}

/// Half the span the height map encodes around mid-gray: the farthest a
/// projection ray can find the high poly above (the front distance, or the
/// cage's largest offset) or below (rear) the low poly.
fn height_range(front: f32, rear: f32, cage_reach: Option<f32>) -> f32 {
    cage_reach.unwrap_or(front).max(rear).max(1e-5)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BakeUniform {
    a: [f32; 4],
    b: [f32; 4],
    c: [f32; 4],
    d: [f32; 4],
    e: [f32; 4],
    f: [f32; 4],
    g: [f32; 4],
    cust: [[f32; 4]; CUSTOM_PER_TEXTURE * 3],
}

pub struct BakeReport {
    pub source: String,
    pub triangles: usize,
    pub parts: usize,
    pub hard_edges: usize,
    pub resolution: u32,
    pub rays_per_texel: u32,
    pub prep_seconds: f32,
    pub edge_max_distance: f32,
    pub id_regions: Vec<IdRegion>,
    /// Low/high-poly part names paired by match-by-name (0 = not matching).
    pub matched_names: usize,
    pub used_cage: bool,
    /// Which maps were baked (a bit per `engine::MAPS` index).
    pub baked: u32,
    /// The custom maps baked, by id, in texture order (map `i` is channel
    /// `i % 4` of custom texture `i / 4`).
    pub custom: Vec<u32>,
    /// Loaded from the bake cache instead of baked.
    pub from_cache: bool,
    started: Instant,
    gpu_seconds: AtomicU32,
    finished: AtomicBool,
    valid_texels: AtomicU32,
    missed_texels: AtomicU32,
    /// Texels whose rays found nothing, filled from the nearest high-poly point.
    filled_texels: AtomicU32,
}

impl BakeReport {
    /// Share of the islands' texels filled from the nearest surface (their rays missed).
    pub fn filled_percent(&self) -> Option<f32> {
        if !self.finished() {
            return None;
        }
        let v = self.valid_texels.load(Ordering::Relaxed).max(1) as f32;
        Some(self.filled_texels.load(Ordering::Relaxed) as f32 / v * 100.0)
    }

    pub fn finished(&self) -> bool {
        self.finished.load(Ordering::Acquire)
    }

    /// Keeps the full-bake report intact while replacing only the ID-region
    /// index with the metadata for a separately rebaked ID texture.
    pub fn with_id_regions_from(&self, id_report: &BakeReport) -> BakeReport {
        BakeReport {
            source: self.source.clone(),
            triangles: self.triangles,
            parts: self.parts,
            hard_edges: self.hard_edges,
            resolution: self.resolution,
            rays_per_texel: self.rays_per_texel,
            prep_seconds: self.prep_seconds,
            edge_max_distance: self.edge_max_distance,
            id_regions: id_report.id_regions.clone(),
            matched_names: self.matched_names,
            used_cage: self.used_cage,
            baked: self.baked,
            custom: self.custom.clone(),
            from_cache: self.from_cache,
            started: self.started.clone(),
            gpu_seconds: AtomicU32::new(self.gpu_seconds.load(Ordering::Relaxed)),
            finished: AtomicBool::new(self.finished.load(Ordering::Acquire)),
            valid_texels: AtomicU32::new(self.valid_texels.load(Ordering::Relaxed)),
            missed_texels: AtomicU32::new(self.missed_texels.load(Ordering::Relaxed)),
            filled_texels: AtomicU32::new(self.filled_texels.load(Ordering::Relaxed)),
        }
    }

    pub fn missed_percent(&self) -> Option<f32> {
        if !self.finished() {
            return None;
        }
        let v = self.valid_texels.load(Ordering::Relaxed).max(1) as f32;
        Some(self.missed_texels.load(Ordering::Relaxed) as f32 / v * 100.0)
    }
    pub fn seconds(&self) -> Option<f32> {
        self.finished().then(|| f32::from_bits(self.gpu_seconds.load(Ordering::Relaxed)))
    }

    /// The finished report, as stored with a cached bake.
    pub fn cache_json(&self) -> serde_json::Value {
        serde_json::json!({
            "source": self.source,
            "triangles": self.triangles,
            "parts": self.parts,
            "hard_edges": self.hard_edges,
            "resolution": self.resolution,
            "rays_per_texel": self.rays_per_texel,
            "prep_seconds": self.prep_seconds,
            "edge_max_distance": self.edge_max_distance,
            "id_regions": self.id_regions.iter().map(|r| serde_json::json!({
                "color": r.color, "triangles": r.triangles, "min": r.min, "max": r.max,
            })).collect::<Vec<_>>(),
            "matched_names": self.matched_names,
            "used_cage": self.used_cage,
            "baked": self.baked,
            "custom": self.custom,
            "gpu_seconds": self.seconds().unwrap_or(0.0),
            "valid_texels": self.valid_texels.load(Ordering::Relaxed),
            "missed_texels": self.missed_texels.load(Ordering::Relaxed),
            "filled_texels": self.filled_texels.load(Ordering::Relaxed),
        })
    }

    /// A finished report restored from [`Self::cache_json`].
    pub fn from_cache_json(v: &serde_json::Value) -> Option<Self> {
        let u = |k: &str| v[k].as_u64();
        let f = |k: &str| v[k].as_f64().map(|x| x as f32);
        let f3 = |r: &serde_json::Value, k: &str| -> Option<[f32; 3]> {
            let a = r[k].as_array()?;
            Some([a.first()?.as_f64()? as f32, a.get(1)?.as_f64()? as f32, a.get(2)?.as_f64()? as f32])
        };
        let id_regions = v["id_regions"]
            .as_array()?
            .iter()
            .map(|r| Some(IdRegion { color: f3(r, "color")?, triangles: r["triangles"].as_u64()? as usize, min: f3(r, "min")?, max: f3(r, "max")? }))
            .collect::<Option<Vec<_>>>()?;
        Some(BakeReport {
            source: v["source"].as_str()?.to_string(),
            triangles: u("triangles")? as usize,
            parts: u("parts")? as usize,
            hard_edges: u("hard_edges")? as usize,
            resolution: u("resolution")? as u32,
            rays_per_texel: u("rays_per_texel")? as u32,
            prep_seconds: f("prep_seconds")?,
            edge_max_distance: f("edge_max_distance")?,
            id_regions,
            matched_names: u("matched_names")? as usize,
            used_cage: v["used_cage"].as_bool()?,
            baked: v["baked"].as_u64().map_or(ALL_MAPS, |b| b as u32),
            custom: v["custom"].as_array().map_or(Vec::new(), |a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect()),
            from_cache: true,
            started: Instant::now(),
            gpu_seconds: AtomicU32::new(f("gpu_seconds")?.to_bits()),
            finished: AtomicBool::new(true),
            valid_texels: AtomicU32::new(u("valid_texels")? as u32),
            missed_texels: AtomicU32::new(u("missed_texels")? as u32),
            filled_texels: AtomicU32::new(u("filled_texels").unwrap_or(0) as u32),
        })
    }
}

fn v(t: &Tex) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::TextureView(&t.view)
}

#[derive(Clone, Copy)]
enum Entry {
    Uniform,
    Tex,
    TexUint,
    Storage(bool),
    StorageTex(F),
}

struct SceneUploads {
    nodes: wgpu::Buffer,
    bake_vertices: wgpu::Buffer,
    bake_triangles: wgpu::Buffer,
    tri_vcolor: wgpu::Buffer,
    parts: wgpu::Buffer,
    ids: wgpu::Buffer,
    seg_nodes: wgpu::Buffer,
    segs: wgpu::Buffer,
    low_corners: Option<(wgpu::Buffer, usize)>,
    low_draws: Option<(wgpu::Buffer, Vec<(u32, u32, u32)>)>,
    cage: Option<wgpu::Buffer>,
}

impl SceneUploads {
    fn vertex(device: &wgpu::Device, label: &str, vertices: &[crate::mesh::Vertex]) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        })
    }

    fn vertex_owned(device: &wgpu::Device, label: &str, vertices: Vec<crate::mesh::Vertex>) -> wgpu::Buffer {
        Self::vertex(device, label, &vertices)
    }

    fn indices(device: &wgpu::Device, label: &str, indices: &[u32]) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        })
    }

    fn indices_owned(device: &wgpu::Device, label: &str, indices: Vec<u32>) -> wgpu::Buffer {
        Self::indices(device, label, &indices)
    }

    fn borrowed(device: &wgpu::Device, scene: &PreparedScene) -> Self {
        let low_corners = scene.low_corners.as_ref().map(|vertices| {
            (Self::vertex(device, "bake low corners", vertices), vertices.len())
        });
        let low_draws = scene.low_draws.as_ref().map(|(sorted, draws)| {
            (Self::indices(device, "bake low by group", sorted), draws.clone())
        });
        let cage = scene.cage.as_ref().map(|vertices| Self::vertex(device, "cage", vertices));
        Self {
            nodes: GpuBaker::storage(device, "bvh nodes", &scene.nodes),
            bake_vertices: GpuBaker::storage(device, "bake source vertices", &scene.bake_vertices),
            bake_triangles: GpuBaker::storage(device, "bake triangle indices and metadata", &scene.bake_triangles),
            tri_vcolor: GpuBaker::storage(device, "tri vertex colors", &scene.tri_vcolor),
            parts: GpuBaker::storage(device, "parts", &scene.parts),
            ids: GpuBaker::storage(device, "ids", &scene.ids),
            seg_nodes: GpuBaker::storage(device, "edge bvh nodes", &scene.seg_nodes),
            segs: GpuBaker::storage(device, "edge segments", &scene.segs),
            low_corners,
            low_draws,
            cage,
        }
    }

    /// Moves each large CPU array into a short-lived upload call. The source
    /// vector is dropped as soon as its GPU buffer has been initialized, rather
    /// than keeping every prepared array alive until the entire start pass ends.
    fn owned(device: &wgpu::Device, scene: &mut PreparedScene) -> Self {
        let nodes = GpuBaker::storage(device, "bvh nodes", &std::mem::take(&mut scene.nodes));
        let bake_vertices = GpuBaker::storage(device, "bake source vertices", &std::mem::take(&mut scene.bake_vertices));
        let bake_triangles = GpuBaker::storage(device, "bake triangle indices and metadata", &std::mem::take(&mut scene.bake_triangles));
        let tri_vcolor = GpuBaker::storage(device, "tri vertex colors", &std::mem::take(&mut scene.tri_vcolor));
        let parts = GpuBaker::storage(device, "parts", &std::mem::take(&mut scene.parts));
        let ids = GpuBaker::storage(device, "ids", &std::mem::take(&mut scene.ids));
        let seg_nodes = GpuBaker::storage(device, "edge bvh nodes", &std::mem::take(&mut scene.seg_nodes));
        let segs = GpuBaker::storage(device, "edge segments", &std::mem::take(&mut scene.segs));
        let low_corners = scene.low_corners.take().map(|vertices| {
            let count = vertices.len();
            (Self::vertex_owned(device, "bake low corners", vertices), count)
        });
        let low_draws = scene.low_draws.take().map(|(sorted, draws)| {
            (Self::indices_owned(device, "bake low by group", sorted), draws)
        });
        let cage = scene.cage.take().map(|vertices| Self::vertex_owned(device, "cage", vertices));
        Self { nodes, bake_vertices, bake_triangles, tri_vcolor, parts, ids, seg_nodes, segs, low_corners, low_draws, cage }
    }
}

pub struct GpuBaker {
    /// Format of the rasterized low-poly position and tangent maps.
    uv_pos_format: F,
    uv_tan_format: F,
    /// Storage format for the per-texel curvature/triangle-size pair.
    surface_curv_format: F,
    uv: wgpu::RenderPipeline,
    dilate: wgpu::RenderPipeline,
    dilate_bgl: wgpu::BindGroupLayout,
    surface: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    occlusion: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    curvature: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    jfa_init: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    jfa_step: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    edges: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    denoise: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    curv_smooth: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    occ_smooth: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pack: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    custom: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    cust_smooth: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
}

const TILE: u32 = 256;
/// Custom-map tiles are smaller: a flow walk traces hundreds of rays a texel.
const CUSTOM_TILE: u32 = 128;
const MAX_IN_FLIGHT: usize = 3;
const DILATE: u32 = 16;

pub struct BakeRun {
    res: u32,
    uniform: BakeUniform,
    low: [Tex; 3],
    surf_pos: Tex,
    surf_nrm: Tex,
    surf_geo: Tex,
    /// Per texel: the source mesh's vertex curvature, interpolated, and the
    /// size of the source triangle there.
    surf_curv: Tex,
    occ: Tex,
    bent: Tex,
    maps: Option<[Tex; MAP_TEXTURES]>,
    /// The custom map textures, four maps each, and each one's settings.
    custom: Vec<(Tex, [[f32; 4]; CUSTOM_PER_TEXTURE * 3])>,
    nodes: wgpu::Buffer,
    bake_vertices: wgpu::Buffer,
    bake_triangles: wgpu::Buffer,
    seg_nodes: wgpu::Buffer,
    segs: wgpu::Buffer,
    /// Ray-traced tiles: x, y and the pass (0 occlusion, 1 custom maps).
    tiles: Vec<[u32; 3]>,
    next_tile: usize,
    done_tiles: Arc<AtomicUsize>,
    in_flight: Arc<AtomicUsize>,
    pub report: Arc<BakeReport>,
}

impl BakeRun {
    pub fn progress(&self) -> f32 {
        self.done_tiles.load(Ordering::Relaxed) as f32 / self.tiles.len().max(1) as f32
    }
}

impl GpuBaker {
    /// Test helper for paths that use the original floating-point UV targets.
    #[cfg(test)]
    pub fn new(device: &wgpu::Device) -> GpuBaker {
        Self::new_with_formats(device, F::Rgba32Float, F::Rgba32Float, false, F::Rg32Float)
    }

    /// Selects render-target and storage formats from the adapter's actual capabilities.
    pub fn new_for_adapter(device: &wgpu::Device, adapter: &wgpu::Adapter) -> GpuBaker {
        let required_uv_usages = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST;
        let float_uv = adapter.get_texture_format_features(F::Rgba32Float);
        let uint_uv = adapter.get_texture_format_features(F::Rgba32Uint);
        let (uv_pos_format, uv_tan_format, integer_uv_textures) =
            if float_uv.allowed_usages.contains(required_uv_usages) {
                (F::Rgba32Float, F::Rgba32Float, false)
            } else if uint_uv.allowed_usages.contains(required_uv_usages) {
                (F::Rgba32Uint, F::Rgba32Uint, true)
            } else {
                panic!(
                    "GPU adapter {:?} supports neither Rgba32Float nor Rgba32Uint for bake UV render/sample/storage usage",
                    adapter.get_info()
                );
            };

        let rg32 = adapter.get_texture_format_features(F::Rg32Float);
        let rgba32 = adapter.get_texture_format_features(F::Rgba32Float);
        let surface_curv_format = if rg32.allowed_usages.contains(wgpu::TextureUsages::STORAGE_BINDING)
            && rg32.flags.contains(wgpu::TextureFormatFeatureFlags::STORAGE_WRITE_ONLY)
        {
            F::Rg32Float
        } else if rgba32.allowed_usages.contains(wgpu::TextureUsages::STORAGE_BINDING)
            && rgba32.flags.contains(wgpu::TextureFormatFeatureFlags::STORAGE_WRITE_ONLY)
        {
            F::Rgba32Float
        } else {
            panic!(
                "GPU adapter {:?} supports neither Rg32Float nor Rgba32Float write-only storage textures for surface curvature",
                adapter.get_info()
            );
        };

        Self::new_with_formats(device, uv_pos_format, uv_tan_format, integer_uv_textures, surface_curv_format)
    }

    fn new_with_formats(
        device: &wgpu::Device,
        uv_pos_format: F,
        uv_tan_format: F,
        integer_uv_textures: bool,
        surface_curv_format: F,
    ) -> GpuBaker {
        let mut uv_source = include_str!("shaders/bake_uv.wgsl").to_owned();
        let mut bake_source = include_str!("shaders/bake.wgsl").to_owned();
        if integer_uv_textures {
            uv_source = uv_source
                .replace("@location(0) pos: vec4<f32>", "@location(0) pos: vec4<u32>")
                .replace("@location(2) tan: vec4<f32>", "@location(2) tan: vec4<u32>")
                .replace(
                    "out.pos = vec4<f32>(in.wpos, 1.0);",
                    "out.pos = bitcast<vec4<u32>>(vec4<f32>(in.wpos, 1.0));",
                )
                .replace(
                    "out.tan = vec4<f32>(in.t.xyz, in.tw);",
                    "out.tan = bitcast<vec4<u32>>(vec4<f32>(in.t.xyz, in.tw));",
                )
                .replace("var src_pos: texture_2d<f32>;", "var src_pos: texture_2d<u32>;")
                .replace("var src_tan: texture_2d<f32>;", "var src_tan: texture_2d<u32>;")
                .replace("if (p.a > 0.25)", "if (bitcast<f32>(p.a) > 0.25)")
                .replace(
                    "out.pos = vec4<f32>(s.xyz, 0.5);",
                    "out.pos = vec4<u32>(s.xyz, bitcast<u32>(0.5));",
                )
                .replace("if (s.a > 0.25)", "if (bitcast<f32>(s.a) > 0.25)")
                .replace("out.pos = vec4<f32>(0.0);", "out.pos = vec4<u32>(0u);")
                .replace("out.tan = vec4<f32>(0.0);", "out.tan = vec4<u32>(0u);");

            for (from, to) in [
                ("var low_pos: texture_2d<f32>;", "var low_pos: texture_2d<u32>;"),
                ("var low_tan: texture_2d<f32>;", "var low_tan: texture_2d<u32>;"),
                ("var cage_pos: texture_2d<f32>;", "var cage_pos: texture_2d<u32>;"),
                ("var cage_tan: texture_2d<f32>;", "var cage_tan: texture_2d<u32>;"),
                (
                    "textureLoad(low_pos, clamp(c + dir, vec2<i32>(0), vec2<i32>(r)), 0)",
                    "bitcast<vec4<f32>>(textureLoad(low_pos, clamp(c + dir, vec2<i32>(0), vec2<i32>(r)), 0))",
                ),
                (
                    "textureLoad(low_pos, clamp(c - dir, vec2<i32>(0), vec2<i32>(r)), 0)",
                    "bitcast<vec4<f32>>(textureLoad(low_pos, clamp(c - dir, vec2<i32>(0), vec2<i32>(r)), 0))",
                ),
                ("textureLoad(low_pos, q, 0)", "bitcast<vec4<f32>>(textureLoad(low_pos, q, 0))"),
                ("textureLoad(low_pos, c, 0)", "bitcast<vec4<f32>>(textureLoad(low_pos, c, 0))"),
                ("textureLoad(low_tan, q, 0)", "bitcast<vec4<f32>>(textureLoad(low_tan, q, 0))"),
                ("textureLoad(low_tan, c, 0)", "bitcast<vec4<f32>>(textureLoad(low_tan, c, 0))"),
                ("textureLoad(cage_pos, c, 0)", "bitcast<vec4<f32>>(textureLoad(cage_pos, c, 0))"),
                ("textureLoad(cage_tan, c, 0)", "bitcast<vec4<f32>>(textureLoad(cage_tan, c, 0))"),
            ] {
                bake_source = bake_source.replace(from, to);
            }
        }
        if surface_curv_format == F::Rgba32Float {
            bake_source = bake_source.replace(
                "texture_storage_2d<rg32float, write>",
                "texture_storage_2d<rgba32float, write>",
            );
        }
        let uv_module = gpu::shader(device, "bake uv", &uv_source);
        // Tangent is f32: its w carries a triangle slot (see `low_corners`).
        let surface_formats = [uv_pos_format, F::Rgba16Float, uv_tan_format];
        let empty = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        let uv = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("bake uv"),
            layout: Some(&empty),
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
                targets: &surface_formats.map(gpu::target),
            }),
            multiview_mask: None,
            cache: None,
        });
        let dilate_entries = if integer_uv_textures {
            vec![gpu::uint_tex_entry(0), gpu::tex_entry(1, false), gpu::uint_tex_entry(2)]
        } else {
            vec![gpu::tex_entry(0, false), gpu::tex_entry(1, false), gpu::tex_entry(2, false)]
        };
        let dilate_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bake dilate"),
            entries: &dilate_entries,
        });
        let dilate = gpu::fullscreen_pipeline(
            device,
            "bake dilate",
            &uv_module,
            "fs_dilate",
            &gpu::layout(device, &dilate_bgl),
            &surface_formats.map(gpu::target),
        );

        let module = gpu::shader(device, "bake", &bake_source);
        let compute = |entry: &str, bindings: &[(u32, Entry)]| {
            let entries: Vec<wgpu::BindGroupLayoutEntry> = bindings
                .iter()
                .map(|(b, e)| wgpu::BindGroupLayoutEntry {
                    binding: *b,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: match e {
                        Entry::Uniform => wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        Entry::Tex => wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        Entry::TexUint => wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Uint,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        Entry::Storage(rw) => wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: !rw },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        Entry::StorageTex(f) => wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: *f,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                    },
                    count: None,
                })
                .collect();
            let bgl =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(entry), entries: &entries });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&gpu::layout(device, &bgl)),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            });
            (pipeline, bgl)
        };
        use Entry::*;
        // Position/tangent UV targets are f32 on adapters with float render
        // support and bitcast u32 on adapters that need integer render targets.
        // Every sampled view of those maps must use the matching sample type.
        let uv_tex = if integer_uv_textures { TexUint } else { Tex };
        let surface = compute(
            "cs_surface",
            &[
                (0, Uniform),
                (1, uv_tex),
                (2, Tex),
                (3, uv_tex),
                (4, Storage(false)),
                (5, Storage(false)),
                (6, Storage(false)),
                (7, Storage(false)),
                (39, Storage(false)),
                (8, Storage(false)),
                (9, StorageTex(F::Rgba32Float)),
                (10, StorageTex(F::Rgba16Float)),
                (11, StorageTex(F::Rgba16Float)),
                (12, StorageTex(F::Rgba8Unorm)),
                (26, Storage(true)),
                (27, StorageTex(F::Rgba32Float)),
                (34, StorageTex(surface_curv_format)),
                (33, uv_tex),
                (38, uv_tex),
            ],
        );
        let occlusion = compute(
            "cs_occlusion",
            &[
                (0, Uniform),
                (1, uv_tex),
                (4, Storage(false)),
                (5, Storage(false)),
                (6, Storage(false)),
                (13, Tex),
                (14, Tex),
                (15, StorageTex(F::Rgba8Unorm)),
                (16, StorageTex(F::Rgba8Unorm)),
                (28, Tex),
                (35, Tex),
            ],
        );
        let curvature =
            compute("cs_curvature", &[(0, Uniform), (1, uv_tex), (13, Tex), (14, Tex), (17, StorageTex(F::Rgba16Float)), (35, Tex)]);
        let jfa_init =
            compute("cs_jfa_init", &[(0, Uniform), (1, uv_tex), (13, Tex), (18, Tex), (19, StorageTex(F::Rgba32Float))]);
        let jfa_step = compute("cs_jfa_step", &[(0, Uniform), (13, Tex), (20, Tex), (19, StorageTex(F::Rgba32Float)), (28, Tex)]);
        let edges = compute(
            "cs_edges",
            &[
                (0, Uniform),
                (13, Tex),
                (20, Tex),
                (28, Tex),
                (29, Storage(false)),
                (30, Storage(false)),
                (31, StorageTex(F::Rgba16Float)),
            ],
        );
        let denoise = compute(
            "cs_denoise",
            &[
                (0, Uniform),
                (1, uv_tex),
                (13, Tex),
                (14, Tex),
                (21, Tex),
                (22, Tex),
                (15, StorageTex(F::Rgba8Unorm)),
                (16, StorageTex(F::Rgba8Unorm)),
            ],
        );
        let smooth_inputs = |io: &[(u32, Entry)]| {
            let mut e = vec![(0, Uniform), (1, uv_tex), (13, Tex), (14, Tex), (28, Tex), (35, Tex)];
            e.extend(io.iter().map(|(b, k)| (*b, *k)));
            e
        };
        let curv_smooth = compute("cs_curv_smooth", &smooth_inputs(&[(18, Tex), (17, StorageTex(F::Rgba16Float))]));
        let occ_smooth = compute(
            "cs_occ_smooth",
            &smooth_inputs(&[(21, Tex), (22, Tex), (15, StorageTex(F::Rgba8Unorm)), (16, StorageTex(F::Rgba8Unorm))]),
        );
        let pack = compute(
            "cs_pack",
            &[
                (0, Uniform),
                (1, uv_tex),
                (13, Tex),
                (14, Tex),
                (18, Tex),
                (21, Tex),
                (22, Tex),
                (32, Tex),
                (23, StorageTex(F::Rgba8Unorm)),
                (24, StorageTex(F::Rgba16Float)),
                (25, StorageTex(F::Rgba8Unorm)),
            ],
        );
        let custom = compute(
            "cs_custom",
            &[
                (0, Uniform),
                (1, uv_tex),
                (4, Storage(false)),
                (5, Storage(false)),
                (6, Storage(false)),
                (13, Tex),
                (14, Tex),
                (28, Tex),
                (35, Tex),
                (36, StorageTex(F::Rgba16Float)),
            ],
        );
        let cust_smooth = compute("cs_cust_smooth", &smooth_inputs(&[(37, Tex), (36, StorageTex(F::Rgba16Float))]));
        GpuBaker {
            uv_pos_format,
            uv_tan_format,
            surface_curv_format,
            uv,
            dilate,
            dilate_bgl,
            surface,
            occlusion,
            curvature,
            jfa_init,
            jfa_step,
            edges,
            denoise,
            curv_smooth,
            occ_smooth,
            pack,
            custom,
            cust_smooth,
        }
    }

    fn uniform_buffer(device: &wgpu::Device, u: &BakeUniform) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bake uniform"),
            contents: bytemuck::bytes_of(u),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    fn storage<T: bytemuck::Pod>(device: &wgpu::Device, label: &str, data: &[T]) -> wgpu::Buffer {
        // Empty storage bindings are invalid; keep one zeroed element.
        let fallback = [T::zeroed()];
        let data = if data.is_empty() { &fallback[..] } else { data };
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        })
    }

    fn group(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        entries: Vec<(u32, wgpu::BindingResource<'_>)>,
    ) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> =
            entries.into_iter().map(|(binding, resource)| wgpu::BindGroupEntry { binding, resource }).collect();
        device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: bgl, entries: &entries })
    }

    /// Six a-trous passes (spacing 1 to 32) of a triangle-scale smoothing
    /// shader. Each map `(tex, scratch, read binding, write binding)`
    /// ping-pongs between its two textures and ends back in `tex`.
    fn smooth_passes(
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &(wgpu::ComputePipeline, wgpu::BindGroupLayout),
        run: &BakeRun,
        maps: &[(&Tex, &Tex, u32, u32)],
        passes: u32,
    ) {
        // An even count ends back in the source texture.
        debug_assert!(passes.is_multiple_of(2));
        for i in 0..passes {
            let mut u = run.uniform;
            u.e[2] = (1u32 << i) as f32;
            let buf = Self::uniform_buffer(device, &u);
            let mut entries = vec![
                (0, buf.as_entire_binding()),
                (1, v(&run.low[0])),
                (13, v(&run.surf_pos)),
                (14, v(&run.surf_nrm)),
                (28, v(&run.surf_geo)),
                (35, v(&run.surf_curv)),
            ];
            for &(tex, scratch, read, write) in maps {
                let (src, dst) = if i % 2 == 0 { (tex, scratch) } else { (scratch, tex) };
                entries.push((read, v(src)));
                entries.push((write, v(dst)));
            }
            let g = Self::group(device, &pipeline.1, entries);
            Self::dispatch(encoder, &pipeline.0, &g, run.res, run.res);
        }
    }

    fn dispatch(encoder: &mut wgpu::CommandEncoder, pipeline: &wgpu::ComputePipeline, group: &wgpu::BindGroup, w: u32, h: u32) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
    }

    /// Uploads the scene, renders the low-poly UV maps and runs the projection
    /// pass. Occlusion tiles are then fed by [`GpuBaker::pump`].
    pub fn start(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &GpuMesh,
        scene: &PreparedScene,
        settings: &BakeSettings,
        res: u32,
    ) -> BakeRun {
        let uploads = SceneUploads::borrowed(device, scene);
        self.start_uploaded(device, queue, mesh, scene, settings, res, uploads)
    }

    /// Bake a prepared scene that can release its CPU arrays after each upload.
    /// Live preview uses start because it keeps its prepared scene cached.
    pub fn start_owned(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &GpuMesh,
        scene: &mut PreparedScene,
        settings: &BakeSettings,
        res: u32,
    ) -> BakeRun {
        let uploads = SceneUploads::owned(device, scene);
        self.start_uploaded(device, queue, mesh, scene, settings, res, uploads)
    }

    fn start_uploaded(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &GpuMesh,
        scene: &PreparedScene,
        settings: &BakeSettings,
        res: u32,
        uploads: SceneUploads,
    ) -> BakeRun {
        let SceneUploads {
            nodes, bake_vertices, bake_triangles, tri_vcolor, parts, ids, seg_nodes, segs,
            low_corners, low_draws, cage: cage_vertices,
        } = uploads;
        let res = res.min(MAX_BAKE_RES);
        let storage_usage = wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::STORAGE_BINDING;
        let uv_usage = wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::RENDER_ATTACHMENT;
        let t = |label: &str, f: F| Tex::new_without_srgb_view(device, label, res, res, f, storage_usage);
        let rt = |label: &str, f: F| Tex::new_without_srgb_view(device, label, res, res, f, uv_usage);
        let low = [rt("bake pos", self.uv_pos_format), rt("bake nrm", F::Rgba16Float), rt("bake tan", self.uv_tan_format)];
        let tmp = [rt("bake pos 2", self.uv_pos_format), rt("bake nrm 2", F::Rgba16Float), rt("bake tan 2", self.uv_tan_format)];

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let clear = wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT);
            let att: Vec<_> = low.iter().map(|t| gpu::color_attachment(&t.view, clear)).collect();
            let mut pass = gpu::pass(&mut encoder, "bake uv", &att);
            pass.set_pipeline(&self.uv);
            pass.set_vertex_buffer(0, mesh.vertices.slice(..));
            match (&low_corners, &low_draws) {
                // The low poly alone: one vertex per corner, carrying its triangle.
                (Some((vertices, count)), _) => {
                    pass.set_vertex_buffer(0, vertices.slice(..));
                    pass.draw(0..*count as u32, 0..1);
                }
                // Each name group draws as instance group, which the shader
                // records per texel.
                (None, Some((index_buffer, draws))) => {
                    pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    for &(first, count, group) in draws {
                        pass.draw_indexed(first..first + count, 0, group..group + 1);
                    }
                }
                (None, None) => {
                    pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(scene.raster.start * 3..scene.raster.end * 3, 0, 0..1);
                }
            }
        }
        let dilate_group =
            |src: &[Tex; 3]| Self::group(device, &self.dilate_bgl, vec![(0, v(&src[0])), (1, v(&src[1])), (2, v(&src[2]))]);
        let forward = dilate_group(&low);
        let backward = dilate_group(&tmp);
        for i in 0..DILATE {
            // DILATE is even, so the result ends up back in `low`.
            let (group, dst) = if i % 2 == 0 { (&forward, &tmp) } else { (&backward, &low) };
            let clear = wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT);
            let att: Vec<_> = dst.iter().map(|t| gpu::color_attachment(&t.view, clear)).collect();
            let mut pass = gpu::pass(&mut encoder, "bake dilate", &att);
            pass.set_pipeline(&self.dilate);
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..3, 0..1);
        }

        // The cage, rasterized at the low poly's UVs and dilated the same way.
        let cage = cage_vertices.as_ref().map(|vertices| {
            let set = [rt("cage pos", self.uv_pos_format), rt("cage nrm", F::Rgba16Float), rt("cage tan", self.uv_tan_format)];
            {
                let clear = wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT);
                let att: Vec<_> = set.iter().map(|t| gpu::color_attachment(&t.view, clear)).collect();
                let mut pass = gpu::pass(&mut encoder, "bake cage", &att);
                pass.set_pipeline(&self.uv);
                pass.set_vertex_buffer(0, vertices.slice(..));
                // One corner per low-poly index: the set's corners.
                pass.draw(scene.raster.start * 3..scene.raster.end * 3, 0..1);
            }
            let forward = dilate_group(&set);
            let backward = dilate_group(&tmp);
            for i in 0..DILATE {
                let (group, dst) = if i % 2 == 0 { (&forward, &tmp) } else { (&backward, &set) };
                let clear = wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT);
                let att: Vec<_> = dst.iter().map(|t| gpu::color_attachment(&t.view, clear)).collect();
                let mut pass = gpu::pass(&mut encoder, "bake cage dilate", &att);
                pass.set_pipeline(&self.dilate);
                pass.set_bind_group(0, group, &[]);
                pass.draw(0..3, 0..1);
            }
            set
        });

        let s = settings;
        let uniform = BakeUniform {
            a: [res as f32, s.aa as f32, s.front, s.rear],
            b: [
                if s.bakes(0) || s.bakes(2) { s.ao_rays as f32 } else { 0.0 },
                s.ao_distance,
                s.cavity_distance,
                if s.bakes(3) { s.thickness_rays as f32 } else { 0.0 },
            ],
            c: [s.thickness_distance, if s.bakes(4) { s.sky_rays as f32 } else { 0.0 }, s.sky_cone_deg.to_radians().cos(), s.curvature_radius],
            d: [s.edge_threshold, scene.edge_max_distance, scene.curvature_scale, if s.high_poly.is_none() { 1.0 } else { 0.0 }],
            e: [0.0, 0.0, 0.0, 1.0],
            f: [scene.edge_roots[0], scene.edge_roots[1], if cage.is_some() { 1.0 } else { 0.0 }, scene.edge_seed_curvature],
            // Random ray sets are picked per world cell of about two texels
            // (see `world_seed` in bake.wgsl).
            g: [
                height_range(s.front, s.rear * if s.builds_cage() { scene.cage_reach / s.front.max(1e-6) } else { 1.0 }, cage.is_some().then_some(scene.cage_reach)),
                s.baked_maps() as f32,
                2.0 * scene.uv_scale / res as f32,
                (u32::from(s.builds_cage()) | u32::from(s.ignore_backfaces) << 1 | u32::from(s.fill_misses) << 2) as f32,
            ],
            cust: [[0.0; 4]; CUSTOM_PER_TEXTURE * 3],
        };
        let ubuf = Self::uniform_buffer(device, &uniform);
        let counters = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bake counters"),
            contents: bytemuck::cast_slice(&[0u32, 0u32, 0u32]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let surf_pos = t("surface pos", F::Rgba32Float);
        let surf_nrm = t("surface nrm", F::Rgba16Float);
        let surf_geo = t("surface geo", F::Rgba32Float);
        let surf_curv = t("surface curvature", self.surface_curv_format);
        let maps = [
            t("map A", MAP_FORMATS[0]),
            t("map B", MAP_FORMATS[1]),
            t("map C", MAP_FORMATS[2]),
            t("map D", MAP_FORMATS[3]),
            t("map E", MAP_FORMATS[4]),
        ];
        let groups = settings.baked_custom().len().div_ceil(CUSTOM_PER_TEXTURE);
        let custom: Vec<_> = (0..groups).map(|g| (t("custom maps", CUSTOM_FORMAT), settings.custom_uniform(g))).collect();
        let group = Self::group(
            device,
            &self.surface.1,
            vec![
                (0, ubuf.as_entire_binding()),
                (1, v(&low[0])),
                (2, v(&low[1])),
                (3, v(&low[2])),
                (4, nodes.as_entire_binding()),
                (5, bake_vertices.as_entire_binding()),
                (6, bake_triangles.as_entire_binding()),
                (7, parts.as_entire_binding()),
                (39, tri_vcolor.as_entire_binding()),
                (8, ids.as_entire_binding()),
                (9, v(&surf_pos)),
                (10, v(&surf_nrm)),
                (11, v(&maps[3])),
                (12, v(&maps[4])),
                (26, counters.as_entire_binding()),
                (27, v(&surf_geo)),
                (33, v(cage.as_ref().map_or(&low[0], |c| &c[0]))),
                (38, v(cage.as_ref().map_or(&low[2], |c| &c[2]))),
                (34, v(&surf_curv)),
            ],
        );
        Self::dispatch(&mut encoder, &self.surface.0, &group, res, res);

        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bake counters readback"),
            size: 12,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&counters, 0, &readback, 0, 12);
        queue.submit([encoder.finish()]);

        let report = Arc::new(BakeReport {
            source: scene.source.clone(),
            triangles: scene.triangles,
            parts: scene.part_count,
            hard_edges: scene.hard_edges,
            resolution: res,
            rays_per_texel: settings.total_rays_per_texel(),
            prep_seconds: scene.prep_seconds,
            edge_max_distance: scene.edge_max_distance,
            id_regions: scene.id_regions.clone(),
            matched_names: scene.matched_names,
            used_cage: cage_vertices.is_some(),
            baked: settings.baked_maps(),
            custom: settings.baked_custom().iter().map(|c| c.id).collect(),
            from_cache: false,
            started: Instant::now(),
            gpu_seconds: AtomicU32::new(0),
            finished: AtomicBool::new(false),
            valid_texels: AtomicU32::new(0),
            missed_texels: AtomicU32::new(0),
            filled_texels: AtomicU32::new(0),
        });
        {
            let report = report.clone();
            let rb = readback.clone();
            readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                if r.is_ok() {
                    if let Ok(data) = rb.slice(..).get_mapped_range() {
                        let c: &[u32] = bytemuck::cast_slice(&data);
                        report.valid_texels.store(c[0], Ordering::Relaxed);
                        report.missed_texels.store(c[1], Ordering::Relaxed);
                        report.filled_texels.store(c[2], Ordering::Relaxed);
                    }
                    rb.unmap();
                }
            });
        }

        let mut tiles = Vec::new();
        for y in (0..res).step_by(TILE as usize) {
            for x in (0..res).step_by(TILE as usize) {
                tiles.push([x, y, 0]);
            }
        }
        // Custom maps, a texture (four maps) at a time: pass 1 + texture index.
        for g in 0..custom.len() as u32 {
            for y in (0..res).step_by(CUSTOM_TILE as usize) {
                for x in (0..res).step_by(CUSTOM_TILE as usize) {
                    tiles.push([x, y, 1 + g]);
                }
            }
        }
        BakeRun {
            res,
            uniform,
            low,
            surf_pos,
            surf_nrm,
            surf_geo,
            surf_curv,
            occ: t("occlusion", F::Rgba8Unorm),
            bent: t("bent normal", F::Rgba8Unorm),
            maps: Some(maps),
            custom,
            nodes,
            bake_vertices,
            bake_triangles,
            seg_nodes,
            segs,
            tiles,
            next_tile: 0,
            done_tiles: Arc::new(AtomicUsize::new(0)),
            in_flight: Arc::new(AtomicUsize::new(0)),
            report,
        }
    }

    /// Feeds occlusion tiles to the GPU, keeping a bounded number in flight.
    /// Once every tile is submitted, encodes the final passes and returns the
    /// maps; the GPU runs them in submission order after the last tile.
    pub fn pump(
        &self,
        run: &mut BakeRun,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        ctx: &egui::Context,
    ) -> Option<BakedMaps> {
        while run.next_tile < run.tiles.len() && run.in_flight.load(Ordering::Acquire) < MAX_IN_FLIGHT {
            let [x, y, pass] = run.tiles[run.next_tile];
            run.next_tile += 1;
            let mut u = run.uniform;
            u.e = [x as f32, y as f32, 0.0, run.next_tile as f32];
            let ubuf = Self::uniform_buffer(device, &u);
            let mut encoder = device.create_command_encoder(&Default::default());
            if pass == 0 {
                let group = Self::group(
                    device,
                    &self.occlusion.1,
                    vec![
                        (0, ubuf.as_entire_binding()),
                        (1, v(&run.low[0])),
                        (4, run.nodes.as_entire_binding()),
                        (5, run.bake_vertices.as_entire_binding()),
                        (6, run.bake_triangles.as_entire_binding()),
                        (13, v(&run.surf_pos)),
                        (14, v(&run.surf_nrm)),
                        (15, v(&run.occ)),
                        (16, v(&run.bent)),
                        (28, v(&run.surf_geo)),
                        (35, v(&run.surf_curv)),
                    ],
                );
                Self::dispatch(&mut encoder, &self.occlusion.0, &group, TILE.min(run.res - x), TILE.min(run.res - y));
            } else {
                let (target, settings) = &run.custom[pass as usize - 1];
                u.cust = *settings;
                let ubuf = Self::uniform_buffer(device, &u);
                let group = Self::group(
                    device,
                    &self.custom.1,
                    vec![
                        (0, ubuf.as_entire_binding()),
                        (1, v(&run.low[0])),
                        (4, run.nodes.as_entire_binding()),
                        (5, run.bake_vertices.as_entire_binding()),
                        (6, run.bake_triangles.as_entire_binding()),
                        (13, v(&run.surf_pos)),
                        (14, v(&run.surf_nrm)),
                        (28, v(&run.surf_geo)),
                        (35, v(&run.surf_curv)),
                        (36, v(target)),
                    ],
                );
                Self::dispatch(&mut encoder, &self.custom.0, &group, CUSTOM_TILE.min(run.res - x), CUSTOM_TILE.min(run.res - y));
            }
            run.in_flight.fetch_add(1, Ordering::AcqRel);
            queue.submit([encoder.finish()]);
            let (in_flight, done, ctx) = (run.in_flight.clone(), run.done_tiles.clone(), ctx.clone());
            queue.on_submitted_work_done(move || {
                in_flight.fetch_sub(1, Ordering::AcqRel);
                done.fetch_add(1, Ordering::AcqRel);
                ctx.request_repaint();
            });
            Self::poll_until_done(device);
        }
        if run.next_tile < run.tiles.len() || run.maps.is_none() {
            return None;
        }

        // Every occlusion tile is queued: curvature, edge distance and packing.
        let res = run.res;
        let usage = wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::STORAGE_BINDING;
        let curv = Tex::new_without_srgb_view(device, "curvature", res, res, F::Rgba16Float, usage);
        let jfa = [
            Tex::new_without_srgb_view(device, "jfa a", res, res, F::Rgba32Float, usage),
            Tex::new_without_srgb_view(device, "jfa b", res, res, F::Rgba32Float, usage),
        ];
        let ubuf = Self::uniform_buffer(device, &run.uniform);
        let mut encoder = device.create_command_encoder(&Default::default());
        let edges = Tex::new_without_srgb_view(device, "edge distance", res, res, F::Rgba16Float, usage);
        // Curvature seeds the edge search, so the three are baked together.
        let baked = run.uniform.g[1] as u32;
        if baked & 0b110_0010 != 0 {
        let g = Self::group(
            device,
            &self.curvature.1,
            vec![
                (0, ubuf.as_entire_binding()),
                (1, v(&run.low[0])),
                (13, v(&run.surf_pos)),
                (14, v(&run.surf_nrm)),
                (17, v(&curv)),
                (35, v(&run.surf_curv)),
            ],
        );
        Self::dispatch(&mut encoder, &self.curvature.0, &g, res, res);
        // Triangle-scale smoothing (see `cs_curv_smooth`), before edges are
        // seeded from it. An even pass count ends back in `curv`.
        let curv_scratch = Tex::new_without_srgb_view(device, "curvature 2", res, res, F::Rgba16Float, usage);
        Self::smooth_passes(device, &mut encoder, &self.curv_smooth, run, &[(&curv, &curv_scratch, 18, 17)], 2);
        let g = Self::group(
            device,
            &self.jfa_init.1,
            vec![
                (0, ubuf.as_entire_binding()),
                (1, v(&run.low[0])),
                (13, v(&run.surf_pos)),
                (18, v(&curv)),
                (19, v(&jfa[0])),
            ],
        );
        // Low-poly-only bakes take edges from the mesh alone (see `prepare`).
        Self::dispatch(&mut encoder, &self.jfa_init.0, &g, res, res);
        let mut step = res.next_power_of_two() / 2;
        let mut src = 0;
        while step >= 1 {
            let mut u = run.uniform;
            u.e[2] = step as f32;
            let step_buf = Self::uniform_buffer(device, &u);
            let g = Self::group(
                device,
                &self.jfa_step.1,
                vec![
                    (0, step_buf.as_entire_binding()),
                    (13, v(&run.surf_pos)),
                    (20, v(&jfa[src])),
                    (19, v(&jfa[1 - src])),
                    (28, v(&run.surf_geo)),
                ],
            );
            Self::dispatch(&mut encoder, &self.jfa_step.0, &g, res, res);
            src = 1 - src;
            step /= 2;
        }
        let g = Self::group(
            device,
            &self.edges.1,
            vec![
                (0, ubuf.as_entire_binding()),
                (13, v(&run.surf_pos)),
                (20, v(&jfa[src])),
                (28, v(&run.surf_geo)),
                (29, run.seg_nodes.as_entire_binding()),
                (30, run.segs.as_entire_binding()),
                (31, v(&edges)),
            ],
        );
        Self::dispatch(&mut encoder, &self.edges.0, &g, res, res);
        }

        // A-trous iterations (spacing 1, 2, 4, 8) ping-pong through a scratch
        // pair, ending back in the maps; the edge-stopping weights keep real
        // detail. Noise falls as 1/sqrt(rays), so more rays need fewer, and
        // narrower, passes: each one blurs flat areas a little more and
        // softens contact shadows and cavity.
        let passes = denoise_passes(run.uniform.b[0] as u32);
        let scratch = [
            Tex::new_without_srgb_view(device, "occlusion 2", res, res, F::Rgba8Unorm, usage),
            Tex::new_without_srgb_view(device, "bent normal 2", res, res, F::Rgba8Unorm, usage),
        ];
        for i in 0..passes {
            let (src_occ, src_bent, dst_occ, dst_bent) = if i % 2 == 0 {
                (&run.occ, &run.bent, &scratch[0], &scratch[1])
            } else {
                (&scratch[0], &scratch[1], &run.occ, &run.bent)
            };
            let mut u = run.uniform;
            u.e[2] = (1u32 << i) as f32;
            let step_buf = Self::uniform_buffer(device, &u);
            let g = Self::group(
                device,
                &self.denoise.1,
                vec![
                    (0, step_buf.as_entire_binding()),
                    (1, v(&run.low[0])),
                    (13, v(&run.surf_pos)),
                    (14, v(&run.surf_nrm)),
                    (21, v(src_occ)),
                    (22, v(src_bent)),
                    (15, v(dst_occ)),
                    (16, v(dst_bent)),
                ],
            );
            Self::dispatch(&mut encoder, &self.denoise.0, &g, res, res);
        }
        if passes % 2 == 1 {
            encoder.copy_texture_to_texture(scratch[0].copy_info(), run.occ.copy_info(), run.occ.extent());
            encoder.copy_texture_to_texture(scratch[1].copy_info(), run.bent.copy_info(), run.bent.extent());
        }
        // Rays follow the facets of a coarse mesh; smooth at triangle scale.
        Self::smooth_passes(
            device,
            &mut encoder,
            &self.occ_smooth,
            run,
            &[(&run.occ, &scratch[0], 21, 15), (&run.bent, &scratch[1], 22, 16)],
            6,
        );
        if !run.custom.is_empty() {
            let scratch = Tex::new_without_srgb_view(device, "custom maps 2", res, res, CUSTOM_FORMAT, usage);
            for (target, _) in &run.custom {
                Self::smooth_passes(device, &mut encoder, &self.cust_smooth, run, &[(target, &scratch, 37, 36)], 4);
            }
        }

        let maps = run.maps.take().unwrap();
        let g = Self::group(
            device,
            &self.pack.1,
            vec![
                (0, ubuf.as_entire_binding()),
                (1, v(&run.low[0])),
                (13, v(&run.surf_pos)),
                (14, v(&run.surf_nrm)),
                (18, v(&curv)),
                (21, v(&run.occ)),
                (22, v(&run.bent)),
                (32, v(&edges)),
                (23, v(&maps[0])),
                (24, v(&maps[1])),
                (25, v(&maps[2])),
            ],
        );
        Self::dispatch(&mut encoder, &self.pack.0, &g, res, res);
        queue.submit([encoder.finish()]);
        let report = run.report.clone();
        let ctx = ctx.clone();
        queue.on_submitted_work_done(move || {
            report.gpu_seconds.store(report.started.elapsed().as_secs_f32().to_bits(), Ordering::Relaxed);
            report.finished.store(true, Ordering::Release);
            ctx.request_repaint();
        });
        Self::poll_until_done(device);
        Some(BakedMaps { maps, custom: run.custom.iter().map(|(t, _)| t.clone()).collect() })
    }

    /// wgpu runs completion callbacks only while the device is polled, which
    /// the UI does when it draws a frame. A hidden or minimized window draws
    /// none, so a worker waits on the GPU to deliver the callbacks that wake it.
    fn poll_until_done(device: &wgpu::Device) {
        let device = device.clone();
        std::thread::spawn(move || {
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        });
    }
}

#[cfg(test)]
pub(crate) mod tests {
    //! Offscreen bake tests: bake on the GPU without a window and write every
    //! map to `target/bake_test/*.png` for inspection.
    //!
    //! `cargo test bake -- --ignored --nocapture`; add meshes with
    //! `MP_BAKE_TEST="a.fbx;b.obj"`.

    use super::*;
    use crate::engine::{MAPS, MapKind};
    use crate::gpu::{GpuMesh, f16_to_f32, read_texture};
    use crate::mesh::Primitive;
    use glam::Vec2;
    use std::path::Path;

    #[test]
    fn compact_triangle_bevel_storage_preserves_edge_order() {
        let mut edges = TriangleBevelEdges::default();
        assert_eq!(edges.iter().count(), 0);
        edges.push((4, 9));
        edges.push((2, 7));
        edges.push((1, 3));
        assert_eq!(edges.iter().collect::<Vec<_>>(), vec![(4, 9), (2, 7), (1, 3)]);
    }

    #[test]
    fn weld_groups_keeps_position_and_split_normal_equivalence() {
        let positions = [Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, Vec3::X];
        let normals = [
            Vec3::Z,
            Vec3::new(0.0, 0.001, 1.0).normalize(),
            Vec3::Y,
            Vec3::Z,
        ];
        assert_eq!(weld_groups(&positions, &normals), vec![0, 0, 1, 2]);
    }

    /// Distance from p to the nearest convex feature, as the edge bake measures it.
    fn edge_distance(segs: &[EdgeSeg], p: Vec3) -> f32 {
        segs.iter()
            .map(|s| {
                let [a, b] = s.ends;
                let e = b - a;
                let t = ((p - a).dot(e) / e.length_squared().max(1e-20)).clamp(0.0, 1.0);
                (p.distance(a + e * t) - s.half_width.max(0.0)).max(0.0)
            })
            .fold(f32::MAX, f32::min)
    }

    #[test]
    fn cancelled_bake_preparation_exits_before_work() {
        let mesh = Primitive::RoundedCube.build();
        let settings = BakeSettings::default();
        let cancel = AtomicBool::new(true);
        let result = prepare_cancellable(&mesh, &settings, 0, &cancel);
        assert!(matches!(result, Err(message) if message == "Bake preparation cancelled"));
    }

    #[test]
    fn indexed_bake_payload_preserves_triangles_and_reduces_common_case_storage() {
        let mesh = Primitive::Sphere.build();
        let scene = prepare(&mesh, &BakeSettings::default(), 0).expect("prepare indexed source");
        assert_eq!(scene.bake_vertices.len(), mesh.vertices.len());
        assert_eq!(scene.bake_triangles.len(), mesh.indices.len() / 3);
        assert_eq!(std::mem::size_of::<BakeVertex>(), 28);
        assert_eq!(std::mem::size_of::<BakeTriangle>(), 24);

        let mut source: Vec<[u32; 3]> = mesh.indices.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
        let mut packed: Vec<[u32; 3]> = scene.bake_triangles.iter().map(|t| [t.indices[0], t.indices[1], t.indices[2]]).collect();
        source.sort_unstable();
        packed.sort_unstable();
        assert_eq!(packed, source, "BVH ordering must preserve each source triangle exactly");

        for (i, vertex) in scene.bake_vertices.iter().enumerate() {
            assert_eq!(&vertex.position[..3], &mesh.vertices[i].pos);
            assert_eq!(&vertex.normal_curvature[..3], &mesh.vertices[i].normal);
            assert!(vertex.normal_curvature[3].is_finite());
        }
        assert!(scene.bake_triangles.iter().all(|t| t.metadata[0] < scene.part_count as u32));

        // Old payload: 3 corners * 2 vec4 values per triangle = 96 bytes/T.
        // Indexed payload: one 28-byte record per vertex and 24 bytes per triangle.
        let expanded = scene.bake_triangles.len() * 96;
        let indexed = scene.bake_vertices.len() * 28 + scene.bake_triangles.len() * 24;
        println!(
            "  indexed Bake payload: {} vertices, {} triangles; {indexed} bytes vs {expanded} expanded bytes ({:.1}% less)",
            scene.bake_vertices.len(), scene.bake_triangles.len(),
            (1.0 - indexed as f64 / expanded.max(1) as f64) * 100.0,
        );
        assert!(indexed < expanded, "sphere source should benefit from shared vertex indexing ({indexed} >= {expanded})");
    }

    #[test]
    fn bevels_read_as_one_edge_band() {
        let mesh = Primitive::RoundedCube.build();
        let positions: Vec<Vec3> = mesh.vertices.iter().map(|v| Vec3::from_array(v.pos)).collect();
        let (tri_part, _) = components(&positions, &mesh.indices);
        let emax = BakeSettings::default().edge_distance_for(&mesh);
        let [convex, _] = feature_edges(&positions, &mesh.indices, &tri_part, 35.0, Some(1.0 / (1.25 * emax)), None);
        let (mut bevel, mut inside) = (0, 0);
        for t in mesh.indices.chunks_exact(3) {
            let [a, b, c] = [t[0], t[1], t[2]].map(|i| positions[i as usize]);
            let n = (b - a).cross(c - a).normalize_or_zero().abs();
            // Bevel strips bend around one axis: exactly one normal component is ~0.
            let zeros = [n.x, n.y, n.z].iter().filter(|&&v| v < 1e-3).count();
            if zeros != 1 || n.max_element() > 0.999 {
                continue;
            }
            bevel += 1;
            if edge_distance(&convex, (a + b + c) / 3.0) < 1e-4 {
                inside += 1;
            }
        }
        assert!(bevel > 0);
        assert_eq!(inside, bevel, "every bevel strip should read as distance 0");
        // The middle of a face is far from every edge.
        assert!(edge_distance(&convex, Vec3::new(0.0, 0.0, 1.0)) > 0.5);
    }

    /// The bake's tangent-space encoding (`cs_surface`) and the viewport's
    /// decoding (`fs_mesh`) are inverses, and green means +V up (OpenGL), on
    /// both a regular and a mirrored UV layout.
    #[test]
    fn tangent_space_round_trip_is_opengl() {
        // A quad in the XY plane facing +Z. V runs down the texture, so world
        // +Y is texture-up when v = 0.5 - y.
        let quad = |mirror: bool| {
            let p = [Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 0.0), Vec3::new(-1.0, 1.0, 0.0)];
            let uv = p.map(|p| glam::Vec2::new(if mirror { 0.5 - p.x * 0.5 } else { 0.5 + p.x * 0.5 }, 0.5 - p.y * 0.5));
            Mesh::from_parts("quad".into(), p.to_vec(), Some(vec![Vec3::Z; 4]), uv.to_vec(), vec![0, 1, 2, 0, 2, 3])
        };
        for mirror in [false, true] {
            let mesh = quad(mirror);
            let v = &mesh.vertices[0];
            let n = Vec3::from_array(v.normal);
            let tg = Vec3::from_array([v.tangent[0], v.tangent[1], v.tangent[2]]);
            let w = v.tangent[3];
            // Bake: frame as in `low_frame`, encoding as in `cs_surface`.
            let t = (tg - n * n.dot(tg)).normalize();
            let b = n.cross(t) * if w < 0.0 { -1.0 } else { 1.0 };
            let encode = |hn: Vec3| Vec3::new(hn.dot(t), -hn.dot(b), hn.dot(n));
            // Viewport: `fs_mesh`.
            let b_up = -n.cross(t) * w;
            let decode = |ts: Vec3| (t * ts.x + b_up * ts.y + n * ts.z).normalize();
            for hn in [Vec3::new(0.3, 0.2, 0.9), Vec3::new(-0.5, 0.4, 0.6), Vec3::new(0.1, -0.7, 0.5)] {
                let hn = hn.normalize();
                let back = decode(encode(hn));
                assert!(back.distance(hn) < 1e-5, "mirror {mirror}: {hn} -> {back}");
            }
            // A surface tilted toward texture-up (world +Y) reads green > 0.5.
            assert!(encode(Vec3::new(0.0, 0.5, 1.0).normalize()).y > 0.3, "mirror {mirror}");
            // Red follows +U: world +X, or -X on the mirrored island.
            let red = encode(Vec3::new(0.5, 0.0, 1.0).normalize()).x;
            assert!(if mirror { red < -0.3 } else { red > 0.3 }, "mirror {mirror}: red {red}");
        }
    }

    /// `encode_tangent` (bake) inverts MikkTSpace decoding (`fs_mesh`) exactly
    /// for the raw interpolated frame: unnormalized, non-orthogonal, mirrored.
    #[test]
    fn mikktspace_encoding_inverts_decoding() {
        let decode = |ts: Vec3, rn: Vec3, rt: Vec3, w: f32| {
            let b = -rn.cross(rt) * w;
            (rt * ts.x + b * ts.y + rn * ts.z).normalize()
        };
        let encode = |hn: Vec3, rn: Vec3, rt: Vec3, w: f32| {
            let b = -rn.cross(rt) * w;
            let bn = b.cross(rn);
            let det = rt.dot(bn);
            (Vec3::new(hn.dot(bn), hn.dot(rn.cross(rt)), hn.dot(rt.cross(b))) / det).normalize()
        };
        // A frame between two vertices of a coarse smooth-shaded low poly.
        let rn = Vec3::new(0.2, 0.1, 0.85);
        let rt = Vec3::new(0.9, 0.0, 0.3);
        for w in [1.0, -1.0] {
            for hn in [Vec3::new(0.3, 0.2, 0.9), Vec3::new(-0.5, 0.4, 0.6), Vec3::new(0.1, -0.7, 0.5)] {
                let hn = hn.normalize();
                let ts = encode(hn, rn, rt, w);
                assert!(ts.z > 0.0, "{hn}: tangent-space z {ts}");
                let back = decode(ts, rn, rt, w);
                assert!(back.distance(hn) < 1e-5, "w {w}: {hn} -> {back}");
            }
        }
        // The flat low-poly normal itself encodes as straight up.
        let up = encode(rn.normalize(), rn, rt, 1.0);
        assert!(up.distance(Vec3::Z) < 1e-5, "{up}");
    }

    /// The height map's range covers every hit a projection ray can report.
    #[test]
    fn height_range_covers_the_ray() {
        assert_eq!(height_range(0.05, 0.02, None), 0.05);
        assert_eq!(height_range(0.05, 0.08, None), 0.08);
        // A cage replaces the front distance, even when it reaches farther.
        assert_eq!(height_range(0.05, 0.02, Some(0.3)), 0.3);
        assert_eq!(height_range(0.05, 0.02, Some(0.01)), 0.02);
    }

    /// Curvature sign and scale: a sphere of radius r reads +1/r (convex
    /// bright), its inside-out copy -1/r.
    #[test]
    fn vertex_curvature_of_a_sphere() {
        let mesh = Primitive::Sphere.build();
        let p: Vec<Vec3> = mesh.vertices.iter().map(|v| Vec3::from_array(v.pos)).collect();
        let n: Vec<Vec3> = mesh.vertices.iter().map(|v| Vec3::from_array(v.normal)).collect();
        let r = p.iter().map(|p| p.length()).sum::<f32>() / p.len() as f32;
        let k = vertex_curvature(&p, &n, &mesh.indices, f32::INFINITY);
        let mean = k.iter().sum::<f32>() / k.len() as f32;
        assert!((mean * r - 1.0).abs() < 0.02, "mean curvature {mean}, radius {r}");
        let inward: Vec<Vec3> = n.iter().map(|n| -*n).collect();
        let k_in = vertex_curvature(&p, &inward, &mesh.indices, f32::INFINITY);
        assert!(k_in.iter().all(|&k| k < 0.0));
    }

    /// The AO estimator: `cosine_dir` (mirrored here from bake.wgsl) on a
    /// Hammersley set draws directions with pdf cos/pi, so E[cos] = 2/3 and
    /// the plain hit ratio is the cosine-weighted visibility. Checked against
    /// a wall blocking half the hemisphere (visibility exactly 1/2).
    #[test]
    fn cosine_hemisphere_sampling() {
        fn onb(n: Vec3) -> (Vec3, Vec3) {
            let s = if n.z >= 0.0 { 1.0 } else { -1.0 };
            let a = -1.0 / (s + n.z);
            let b = n.x * n.y * a;
            (Vec3::new(1.0 + s * n.x * n.x * a, s * b, -s * n.x), Vec3::new(b, s + n.y * n.y * a, -n.y))
        }
        fn radical_inverse(i: u32) -> f32 {
            i.reverse_bits() as f32 * 2.328_306_4e-10
        }
        let count = 4096;
        for n in [Vec3::Z, -Vec3::Z, Vec3::new(0.3, -0.8, 0.2).normalize(), Vec3::new(-0.6, 0.0, -0.8)] {
            let (t, b) = onb(n);
            assert!(t.dot(b).abs() < 1e-5 && t.dot(n).abs() < 1e-5 && b.dot(n).abs() < 1e-5);
            assert!((t.length() - 1.0).abs() < 1e-5 && (b.length() - 1.0).abs() < 1e-5);
            let (mut cos, mut open) = (0.0, 0.0);
            for i in 0..count {
                let r = glam::Vec2::new((i as f32 + 0.5) / count as f32, radical_inverse(i));
                let rad = r.x.sqrt();
                let phi = std::f32::consts::TAU * r.y;
                let d = (t * rad * phi.cos() + b * rad * phi.sin() + n * (1.0 - r.x).max(0.0).sqrt()).normalize();
                cos += d.dot(n);
                open += if d.dot(t) > 0.0 { 1.0 } else { 0.0 };
            }
            assert!((cos / count as f32 - 2.0 / 3.0).abs() < 2e-3, "{n}: E[cos] {}", cos / count as f32);
            assert!((open / count as f32 - 0.5).abs() < 2e-3, "{n}: half-blocked visibility {}", open / count as f32);
        }
    }

    fn ready<F: std::future::Future>(f: F) -> F::Output {
        let mut f = std::pin::pin!(f);
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        match f.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(v) => v,
            std::task::Poll::Pending => panic!("native wgpu requests resolve immediately"),
        }
    }

    /// Windowless GPU with the same limits the app requests.
    pub(crate) fn gpu_with_adapter() -> (wgpu::Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = ready(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("a GPU adapter");
        let (device, queue) = ready(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: adapter.features() & wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
            required_limits: adapter.limits(),
            ..Default::default()
        })).expect("a GPU device");
        (instance, adapter, device, queue)
    }

    pub(crate) fn gpu() -> (wgpu::Device, wgpu::Queue) {
        let (_, _, device, queue) = gpu_with_adapter();
        (device, queue)
    }

    /// Runs a whole bake to completion; returns the maps, the report and the
    /// low-poly position map (its alpha marks island interiors).
    pub(crate) fn run_bake(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &Mesh,
        gpu_mesh: &GpuMesh,
        settings: &BakeSettings,
        res: u32,
    ) -> (BakedMaps, Arc<BakeReport>, Tex) {
        let mut scene = prepare(mesh, settings, 0).expect("bake source");
        let baker = GpuBaker::new(device);
        let ctx = egui::Context::default();
        let mut run = baker.start_owned(device, queue, gpu_mesh, &mut scene, settings, res);
        // The owned upload path must release every large CPU payload as soon
        // as its corresponding GPU buffer has been created.
        assert!(scene.nodes.is_empty());
        assert!(scene.bake_vertices.is_empty());
        assert!(scene.bake_triangles.is_empty());
        assert!(scene.tri_vcolor.is_empty());
        assert!(scene.parts.is_empty());
        assert!(scene.ids.is_empty());
        assert!(scene.seg_nodes.is_empty());
        assert!(scene.segs.is_empty());
        assert!(scene.low_corners.is_none());
        assert!(scene.low_draws.is_none());
        assert!(scene.cage.is_none());
        let maps = loop {
            if let Some(maps) = baker.pump(&mut run, device, queue, &ctx) {
                break maps;
            }
            // Blocks until the in-flight tiles finish, which frees slots for the next ones.
            device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");
        };
        device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");
        let r = &run.report;
        println!(
            "  {}: {} tris, {} parts, {} feature edges, {:.2}s GPU, {:.3}% missed",
            r.source,
            r.triangles,
            r.parts,
            r.hard_edges,
            r.seconds().unwrap_or(0.0),
            r.missed_percent().unwrap_or(0.0)
        );
        let [low_pos, _, _] = run.low;
        (maps, run.report, low_pos)
    }

    /// Executes a tiny low-poly bake on the selected adapter. This covers the
    /// integer UV render targets, their bitcast reads in compute, and the
    /// write-only surface-curvature storage texture on OpenGL.
    #[test]
    #[ignore]
    fn gl_adapter_low_poly_bake_roundtrip() {
        let (_instance, adapter, device, queue) = gpu_with_adapter();
        assert_eq!(adapter.get_info().backend, wgpu::Backend::Gl, "expected an OpenGL adapter");

        let mesh = Primitive::RoundedCube.build();
        let gpu_mesh = GpuMesh::new(&device, &mesh);
        let mut settings = BakeSettings::default();
        settings.maps = 0;
        settings.aa = 1;
        let mut scene = prepare(&mesh, &settings, 0).expect("low-poly bake preparation");
        let baker = GpuBaker::new_for_adapter(&device, &adapter);
        let mut run = baker.start_owned(&device, &queue, &gpu_mesh, &mut scene, &settings, 64);
        let ctx = egui::Context::default();

        let _maps = loop {
            if let Some(maps) = baker.pump(&mut run, &device, &queue, &ctx) {
                break maps;
            }
            device.poll(wgpu::PollType::wait_indefinitely()).expect("OpenGL bake poll");
        };
        device.poll(wgpu::PollType::wait_indefinitely()).expect("OpenGL bake completion");

        if run.low[0].format == F::Rgba32Uint {
            assert_eq!(run.low[2].format, F::Rgba32Uint);
            assert_eq!(run.surf_curv.format, F::Rgba32Float);
        }
        let bytes = read_texture(&device, &queue, &run.low[0]);
        let inside = bytes
            .chunks_exact(16)
            .filter(|pixel| f32::from_le_bytes([pixel[12], pixel[13], pixel[14], pixel[15]]) > 0.75)
            .count();
        assert!(inside > 0, "the low-poly UV raster produced no interior texels");
    }

    struct Baked {
        res: u32,
        /// Texel lies inside a UV island (not padding or empty).
        inside: Vec<bool>,
        /// Packed maps as f32 RGBA per texel.
        maps: Vec<Vec<[f32; 4]>>,
    }

    impl Baked {
        fn value(&self, map: usize, i: usize) -> f32 {
            let m = &MAPS[map];
            self.maps[m.tex][i][m.comp]
        }
    }

    fn bake(device: &wgpu::Device, queue: &wgpu::Queue, mesh: &Mesh, settings: &BakeSettings, res: u32) -> Baked {
        let gpu_mesh = GpuMesh::new(device, mesh);
        let (baked, _, low_pos) = run_bake(device, queue, mesh, &gpu_mesh, settings, res);
        let maps = baked.maps;
        let inside = read_texture(device, queue, &low_pos)
            .chunks_exact(16)
            .map(|b| f32::from_le_bytes([b[12], b[13], b[14], b[15]]) > 0.75)
            .collect();
        let maps = maps
            .iter()
            .map(|t| {
                let bytes = read_texture(device, queue, t);
                match t.format {
                    F::Rgba16Float => bytes
                        .chunks_exact(8)
                        .map(|b| [0, 1, 2, 3].map(|k| f16_to_f32(u16::from_le_bytes([b[k * 2], b[k * 2 + 1]]))))
                        .collect(),
                    _ => bytes.chunks_exact(4).map(|b| [0, 1, 2, 3].map(|k| b[k] as f32 / 255.0)).collect(),
                }
            })
            .collect();
        Baked { res, inside, maps }
    }

    fn write_pngs(baked: &Baked, name: &str) {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test");
        std::fs::create_dir_all(&dir).unwrap();
        let px = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        for (mi, m) in MAPS.iter().enumerate() {
            let data = &baked.maps[m.tex];
            let rgb: Vec<u8> = (0..data.len())
                .flat_map(|i| match m.kind {
                    MapKind::Scalar => [px(baked.value(mi, i)); 3],
                    MapKind::Normal => {
                        let [x, y, ..] = data[i];
                        let z = (1.0 - x * x - y * y).max(0.0).sqrt();
                        [px(x * 0.5 + 0.5), px(y * 0.5 + 0.5), px(z * 0.5 + 0.5)]
                    }
                    MapKind::Info => [px(data[i][0]), px(data[i][1]), px(data[i][2])],
                })
                .collect();
            let file = dir.join(format!("{name}_{}.png", m.name.to_lowercase().replace(' ', "_")));
            image::save_buffer(&file, &rgb, baked.res, baked.res, image::ExtendedColorType::Rgb8).unwrap();
        }
        let islands: Vec<u8> = baked.inside.iter().map(|&i| if i { 255 } else { 0 }).collect();
        let file = dir.join(format!("{name}_islands.png"));
        image::save_buffer(&file, &islands, baked.res, baked.res, image::ExtendedColorType::L8).unwrap();
    }

    fn normal_map() -> usize {
        MAPS.iter().position(|m| m.kind == MapKind::Normal).unwrap()
    }

    /// Writes a mesh as an OBJ in its original (pre-normalization) space, so
    /// it can serve as its own high poly.
    fn write_obj(mesh: &Mesh, path: &Path) {
        use std::fmt::Write;
        let mut s = String::new();
        for v in &mesh.vertices {
            let p = Vec3::from_array(v.pos) / mesh.scale + mesh.center;
            writeln!(s, "v {} {} {}", p.x, p.y, p.z).unwrap();
            writeln!(s, "vn {} {} {}", v.normal[0], v.normal[1], v.normal[2]).unwrap();
        }
        for t in mesh.indices.chunks_exact(3) {
            writeln!(s, "f {0}//{0} {1}//{1} {2}//{2}", t[0] + 1, t[1] + 1, t[2] + 1).unwrap();
        }
        std::fs::write(path, s).unwrap();
    }

    /// Projecting a mesh onto itself must give a flat normal map: any tilt is
    /// an artifact of the projection (the old bake streaked near bevels).
    #[test]
    #[ignore]
    fn bake_self_projection_is_flat() {
        let (device, queue) = gpu();
        let mesh = Primitive::RoundedCube.build();
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test");
        std::fs::create_dir_all(&dir).unwrap();
        let obj = dir.join("rounded_cube_high.obj");
        write_obj(&mesh, &obj);
        let settings = BakeSettings { high_poly: Some(obj), ..Default::default() };
        let baked = bake(&device, &queue, &mesh, &settings, 1024);
        write_pngs(&baked, "self");
        let nm = normal_map();
        let mut tilt: Vec<f32> = (0..baked.maps[0].len())
            .filter(|&i| baked.inside[i])
            .map(|i| {
                let x = baked.value(nm, i);
                let y = baked.maps[MAPS[nm].tex][i][MAPS[nm].comp + 1];
                (x * x + y * y).sqrt()
            })
            .collect();
        tilt.sort_by(f32::total_cmp);
        let p999 = tilt[tilt.len() * 999 / 1000];
        let max = *tilt.last().unwrap();
        let over = tilt.iter().filter(|&&t| t > 0.05).count();
        println!("  normal tilt inside islands: p99.9 {p999:.4}, max {max:.4}, {over} texels over 0.05");
        assert!(p999 < 0.02, "self-projected normal map is tilted: p99.9 {p999}");
    }

    /// High-to-low bake of `MP_BAKE_PAIR="low.fbx;high.fbx"` (optional
    /// `MP_BAKE_RES`, default 2048), maps written as `pair_*.png`.
    #[test]
    #[ignore]
    fn bake_high_to_low_pair() {
        let Ok(pair) = std::env::var("MP_BAKE_PAIR") else { return };
        let (low, high) = pair.split_once(';').expect("MP_BAKE_PAIR=low;high");
        let res = std::env::var("MP_BAKE_RES").ok().and_then(|r| r.parse().ok()).unwrap_or(2048);
        let (device, queue) = gpu();
        let mesh = Mesh::load_file(Path::new(low)).unwrap_or_else(|e| panic!("{low}: {e}"));
        let reach = std::env::var("MP_BAKE_REACH").ok().and_then(|r| r.parse().ok()).unwrap_or(0.15);
        let settings = BakeSettings { high_poly: Some(high.into()), front: reach, rear: reach, ..Default::default() };
        let baked = bake(&device, &queue, &mesh, &settings, res);
        write_pngs(&baked, "pair");
    }

    /// Maps left out of the bake come out neutral and are reported missing.
    #[test]
    #[ignore]
    fn bake_skips_maps_left_out() {
        // Only curvature and AO: thickness, sky, cavity and edges come out
        // neutral and the report says they weren't baked.
        let (device, queue) = gpu();
        let mesh = Primitive::MaterialBall.build();
        let gpu_mesh = GpuMesh::new(&device, &mesh);
        let mut s = BakeSettings::default();
        s.apply_quality(Quality::Draft);
        s.maps = (1 << 0) | (1 << 1);
        let (baked, report, _) = run_bake(&device, &queue, &mesh, &gpu_mesh, &s, 128);
        let maps = baked.maps;
        assert_eq!(report.baked & 0b111_1111, 0b11, "AO and curvature baked");
        let a = read_texture(&device, &queue, &maps[0]);
        let b = read_texture(&device, &queue, &maps[1]);
        let half = |b: &[u8], i: usize, k: usize| f16_to_f32(u16::from_le_bytes([b[i * 8 + k * 2], b[i * 8 + k * 2 + 1]]));
        let texels = (128 * 128) as usize;
        let ao_min = (0..texels).map(|i| a[i * 4] as f32 / 255.0).fold(1.0f32, f32::min);
        assert!(ao_min < 0.6, "AO was baked (darkest {ao_min})");
        for i in 0..texels {
            assert_eq!(a[i * 4 + 2], 255, "cavity neutral");
            assert_eq!(a[i * 4 + 3], 255, "thickness neutral");
            assert!((half(&b, i, 0) - 1.0).abs() < 1e-3, "sky neutral");
            assert!((half(&b, i, 1) - 1.0).abs() < 1e-3 && (half(&b, i, 2) - 1.0).abs() < 1e-3, "edges neutral");
        }
    }

    /// The usual Mesh Parts mode must use the compact ID palette, not
    /// allocate one RGBA value for every triangle corner.
    #[test]
    #[ignore]
    fn mesh_part_ids_use_palette_without_corner_colors() {
        let (device, queue) = gpu();
        let mesh = Primitive::RoundedCube.build();
        let settings = BakeSettings::default();
        let scene = prepare(&mesh, &settings, 0).expect("prepare low-poly scene");
        assert!(scene.tri_vcolor.is_empty(), "flat IDs should not allocate per-corner colors");
        drop(scene);
        let gpu_mesh = GpuMesh::new(&device, &mesh);
        let _ = run_bake(&device, &queue, &mesh, &gpu_mesh, &settings, 16);
    }

    /// Low-poly-only bakes of the built-in meshes plus any in MP_BAKE_TEST.
    #[test]
    #[ignore]
    fn bake_low_poly_maps() {
        let (device, queue) = gpu();
        let mut meshes: Vec<Mesh> = vec![Primitive::RoundedCube.build(), Primitive::Torus.build(), Primitive::MaterialBall.build()];
        for path in std::env::var("MP_BAKE_TEST").unwrap_or_default().split(';').filter(|p| !p.is_empty()) {
            meshes.push(Mesh::load_file(Path::new(path)).unwrap_or_else(|e| panic!("{path}: {e}")));
        }
        for mesh in &meshes {
            let baked = bake(&device, &queue, mesh, &BakeSettings::default(), 1024);
            write_pngs(&baked, &mesh.name.to_lowercase().replace(' ', "_"));
        }
    }

    /// What one bake's projection found per texel: inside an island, the low
    /// poly's point, and the high-poly normal it took.
    struct Projected {
        report: Arc<BakeReport>,
        inside: Vec<bool>,
        low: Vec<Vec3>,
        high_n: Vec<Vec3>,
        height: Vec<f32>,
    }

    fn project(device: &wgpu::Device, queue: &wgpu::Queue, mesh: &Mesh, settings: &BakeSettings, res: u32) -> Projected {
        let gpu_mesh = GpuMesh::new(device, mesh);
        let mut scene = prepare(mesh, settings, 0).expect("bake source");
        let baker = GpuBaker::new(device);
        let ctx = egui::Context::default();
        let mut run = baker.start_owned(device, queue, &gpu_mesh, &mut scene, settings, res);
        let maps = loop {
            if let Some(maps) = baker.pump(&mut run, device, queue, &ctx) {
                break maps;
            }
            device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");
        };
        device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");
        let f32s = |t: &Tex| -> Vec<[f32; 4]> {
            read_texture(device, queue, t).chunks_exact(16).map(|b| [0, 1, 2, 3].map(|k| f32::from_le_bytes([b[k * 4], b[k * 4 + 1], b[k * 4 + 2], b[k * 4 + 3]]))).collect()
        };
        let low = f32s(&run.low[0]);
        let nrm: Vec<Vec3> = read_texture(device, queue, &run.surf_nrm)
            .chunks_exact(8)
            .map(|b| Vec3::new(f16_to_f32(u16::from_le_bytes([b[0], b[1]])), f16_to_f32(u16::from_le_bytes([b[2], b[3]])), f16_to_f32(u16::from_le_bytes([b[4], b[5]]))))
            .collect();
        let hm = &MAPS[MAPS.iter().position(|m| m.name == "Height").unwrap()];
        let height_tex = &maps.maps[hm.tex];
        let height: Vec<f32> = match height_tex.format {
            F::Rgba16Float => read_texture(device, queue, height_tex)
                .chunks_exact(8)
                .map(|b| f16_to_f32(u16::from_le_bytes([b[hm.comp * 2], b[hm.comp * 2 + 1]])))
                .collect(),
            _ => read_texture(device, queue, height_tex).chunks_exact(4).map(|b| b[hm.comp] as f32 / 255.0).collect(),
        };
        let r = &run.report;
        println!(
            "  {:.3}% missed, {:.3}% filled from the nearest surface",
            r.missed_percent().unwrap_or(0.0),
            r.filled_percent().unwrap_or(0.0)
        );
        Projected {
            report: run.report.clone(),
            inside: low.iter().map(|p| p[3] > 0.75).collect(),
            low: low.iter().map(|p| Vec3::new(p[0], p[1], p[2])).collect(),
            high_n: nrm,
            height,
        }
    }

    /// A box with sharp edges: six faces of `n`×`n` quads, each with its own
    /// normal (split at the edges), in a 3×2 UV atlas.
    fn hard_box(n: u32) -> Mesh {
        let faces = [
            (Vec3::Z, Vec3::X, -Vec3::Y),
            (Vec3::X, -Vec3::Z, -Vec3::Y),
            (-Vec3::Z, -Vec3::X, -Vec3::Y),
            (-Vec3::X, Vec3::Z, -Vec3::Y),
            (Vec3::Y, Vec3::X, Vec3::Z),
            (-Vec3::Y, Vec3::X, -Vec3::Z),
        ];
        let (mut pos, mut nrm, mut uv, mut idx) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (f, (nf, ua, va)) in faces.iter().enumerate() {
            let base = pos.len() as u32;
            let (col, row) = ((f % 3) as f32, (f / 3) as f32);
            for j in 0..=n {
                for i in 0..=n {
                    let (a, b) = (i as f32 / n as f32 * 2.0 - 1.0, j as f32 / n as f32 * 2.0 - 1.0);
                    pos.push(*nf + *ua * a + *va * b);
                    nrm.push(*nf);
                    uv.push(Vec2::new((col + 0.05 + (a + 1.0) * 0.45) / 3.0, (row + 0.05 + (b + 1.0) * 0.45) / 2.0));
                }
            }
            for j in 0..n {
                for i in 0..n {
                    let v = |i: u32, j: u32| base + j * (n + 1) + i;
                    for t in [[v(i, j), v(i + 1, j), v(i + 1, j + 1)], [v(i, j), v(i + 1, j + 1), v(i, j + 1)]] {
                        let p = t.map(|k| pos[k as usize]);
                        let outward = (p[1] - p[0]).cross(p[2] - p[0]).dot(*nf) > 0.0;
                        idx.extend(if outward { t } else { [t[0], t[2], t[1]] });
                    }
                }
            }
        }
        Mesh::from_parts("Hard Box".into(), pos, Some(nrm), uv, idx)
    }

    fn write_high(path: &Path, positions: &[Vec3], tris: &[[u32; 3]]) {
        use std::fmt::Write;
        let mut s = String::new();
        for p in positions {
            writeln!(s, "v {} {} {}", p.x, p.y, p.z).unwrap();
        }
        for t in tris {
            writeln!(s, "f {} {} {}", t[0] + 1, t[1] + 1, t[2] + 1).unwrap();
        }
        std::fs::write(path, s).unwrap();
    }

    /// A sharp-edged low poly baked from a beveled high poly: with averaged
    /// normals the texels meeting at a hard edge take the same high-poly
    /// point (the cage is closed), so the normal map shades continuously
    /// across the edge. Along each face's own normal they take points a
    /// quarter turn apart: a seam.
    #[test]
    #[ignore]
    fn bake_averaged_normals_close_hard_edges() {
        let (device, queue) = gpu();
        let low = hard_box(16);
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test");
        std::fs::create_dir_all(&dir).unwrap();
        let obj = dir.join("beveled_box_high.obj");
        write_obj(&Primitive::RoundedCube.build(), &obj);
        let res = 384;
        let base = BakeSettings { high_poly: Some(obj), front: 0.05, rear: 0.3, ..Default::default() };
        // Across each hard edge: texels of different faces at the same point.
        let agreement = |p: &Projected| {
            let face = |i: usize| ((i as u32 % res) * 3 / res, (i as u32 / res) * 2 / res);
            let mut grid: std::collections::HashMap<[i32; 3], Vec<usize>> = Default::default();
            let q = |v: Vec3| (v * 150.0).round().as_ivec3().to_array();
            let edge = |v: Vec3| v.abs().to_array().iter().filter(|c| **c > 0.985).count() >= 2;
            for i in (0..p.low.len()).filter(|&i| p.inside[i] && edge(p.low[i])) {
                grid.entry(q(p.low[i])).or_default().push(i);
            }
            let mut dots = Vec::new();
            for cell in grid.values() {
                for (a, &i) in cell.iter().enumerate() {
                    for &j in &cell[a + 1..] {
                        if face(i) != face(j) && p.low[i].distance(p.low[j]) < 0.01 {
                            dots.push(p.high_n[i].normalize_or_zero().dot(p.high_n[j].normalize_or_zero()));
                        }
                    }
                }
            }
            assert!(dots.len() > 50, "texel pairs across the edges ({})", dots.len());
            dots.iter().sum::<f32>() / dots.len() as f32
        };
        let split = project(&device, &queue, &low, &BakeSettings { average_normals: false, ..base.clone() }, res);
        let averaged = project(&device, &queue, &low, &base, res);
        let (s, a) = (agreement(&split), agreement(&averaged));
        println!("  high-poly normals across hard edges agree: {s:.3} along face normals, {a:.3} averaged");
        assert!(a > 0.97, "averaged: the texels at an edge take one point ({a})");
        assert!(a > s + 0.15, "averaging closes the seam ({s} → {a})");
        assert_eq!(averaged.report.missed_percent(), Some(0.0));
    }

    /// A ray reaching a face from behind (the underside of a lid over the
    /// surface) passes it and finds the surface.
    #[test]
    #[ignore]
    fn bake_ignores_backfaces() {
        let (device, queue) = gpu();
        // The low poly: a flat square at y = 0.
        let n = 16;
        let (mut pos, mut uv, mut idx) = (Vec::new(), Vec::new(), Vec::new());
        for j in 0..=n {
            for i in 0..=n {
                let (u, v) = (i as f32 / n as f32, j as f32 / n as f32);
                pos.push(Vec3::new(u * 2.0 - 1.0, 0.0, v * 2.0 - 1.0));
                uv.push(Vec2::new(u, v));
            }
        }
        for j in 0..n {
            for i in 0..n {
                let v = |i: u32, j: u32| j * (n + 1) + i;
                idx.extend([v(i, j), v(i, j + 1), v(i + 1, j + 1), v(i, j), v(i + 1, j + 1), v(i + 1, j)]);
            }
        }
        let low = Mesh::from_parts("Floor".into(), pos, Some(vec![Vec3::Y; (n as usize + 1).pow(2)]), uv, idx);
        // The high poly: the same square, and a lid 0.03 above it facing down.
        let corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
        let mut hp: Vec<Vec3> = corners.iter().map(|&(x, z)| Vec3::new(x, 0.0, z) / low.scale + low.center).collect();
        hp.extend(corners.iter().map(|&(x, z)| Vec3::new(x, 0.03, z) / low.scale + low.center));
        // Floor faces up (+Y), the lid down.
        let tris = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7]];
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test");
        std::fs::create_dir_all(&dir).unwrap();
        let obj = dir.join("floor_with_lid.obj");
        write_high(&obj, &hp, &tris);
        let base = BakeSettings { high_poly: Some(obj), front: 0.06, rear: 0.06, ..Default::default() };
        let mean_height = |p: &Projected| {
            let h: Vec<f32> = (0..p.inside.len()).filter(|&i| p.inside[i]).map(|i| p.height[i]).collect();
            h.iter().sum::<f32>() / h.len().max(1) as f32
        };
        let culled = project(&device, &queue, &low, &base, 128);
        let seen = project(&device, &queue, &low, &BakeSettings { ignore_backfaces: false, ..base.clone() }, 128);
        let (c, s) = (mean_height(&culled), mean_height(&seen));
        println!("  mean height: {c:.3} ignoring backfaces, {s:.3} without");
        assert!((c - 0.5).abs() < 0.02, "the floor itself (mid gray), not the lid ({c})");
        assert!(s > 0.6, "without, the lid's underside is taken ({s})");
    }

    /// A high poly that bulges past the front distance on one side: without
    /// help those texels miss; filling takes the nearest surface (no holes);
    /// a fitted cage reaches it properly (nothing even needs filling).
    #[test]
    #[ignore]
    fn bake_never_misses() {
        let (device, queue) = gpu();
        let low = Primitive::Sphere.build();
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bake_test");
        std::fs::create_dir_all(&dir).unwrap();
        let obj = dir.join("bulging_sphere_high.obj");
        let hp: Vec<Vec3> = low
            .vertices
            .iter()
            .map(|v| {
                let p = Vec3::from_array(v.pos);
                p * if p.x > 0.5 { 1.3 } else { 1.03 } / low.scale + low.center
            })
            .collect();
        let tris: Vec<[u32; 3]> = low.indices.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
        write_high(&obj, &hp, &tris);
        let base = BakeSettings { high_poly: Some(obj.clone()), front: 0.08, rear: 0.05, ..Default::default() };
        let res = 256;
        let holes = project(&device, &queue, &low, &BakeSettings { fill_misses: false, ..base.clone() }, res);
        assert!(holes.report.missed_percent().unwrap() > 5.0, "the bulge misses without help");
        let filled = project(&device, &queue, &low, &base, res);
        assert_eq!(filled.report.missed_percent(), Some(0.0), "no holes");
        assert!(filled.report.filled_percent().unwrap() > 5.0, "the bulge filled from the nearest surface");
        // Auto-fit the cage from the probe, then bake without filling.
        let key = crate::cage::ProbeKey::new(&low, &base);
        let probe = crate::cage::CageProbe::build(&low, key).expect("probes");
        let (scale, fix) = probe.auto_fix(&crate::cage::Reach::new(&base)).expect("fits");
        println!("  auto-fix: {fix:?}");
        let fitted = BakeSettings { cage_paint: Some(Arc::new(CagePaint::new(scale))), fill_misses: false, ..base.clone() };
        let exact = project(&device, &queue, &low, &fitted, res);
        let missed = exact.report.missed_percent().unwrap();
        assert!(missed < 0.5, "a fitted cage reaches the bulge ({missed}%)");
        // Where filled and fitted both found the bulge, they agree on its normal.
        let agree: Vec<f32> = (0..exact.inside.len())
            .filter(|&i| exact.inside[i] && exact.low[i].x > 0.7)
            .map(|i| exact.high_n[i].normalize_or_zero().dot(filled.high_n[i].normalize_or_zero()))
            .collect();
        let mean = agree.iter().sum::<f32>() / agree.len().max(1) as f32;
        println!("  bulge normals, filled vs fitted cage: {mean:.3}");
        assert!(mean > 0.9, "the nearest surface is the bulge itself ({mean})");
    }
}


#[cfg(test)]
mod vertex_color_id_source_tests {
    use super::{prepare, BakeSettings, IdSource, MAX_VERTEX_COLOR_ID_REGIONS};
    use crate::mesh::Primitive;
    use glam::Vec3;
    use std::collections::HashMap;

    #[test]
    fn bake_id_regions_use_the_shared_high_cardinality_vertex_palette() {
        let triangle_count = 150usize;
        let colors: Vec<[[u8; 3]; 3]> = (0..triangle_count)
            .map(|triangle| {
                std::array::from_fn(|corner| {
                    let value = (triangle * 3 + corner) as u32;
                    [
                        (value >> 16) as u8,
                        (value >> 8) as u8,
                        value as u8,
                    ]
                })
            })
            .collect();
        let indices: Vec<u32> = (0..triangle_count)
            .flat_map(|_| [0, 1, 2])
            .collect();
        let positions = [Vec3::ZERO, Vec3::X, Vec3::Y];
        let cancel = std::sync::atomic::AtomicBool::new(false);

        let (palette, distinct) = super::vertex_color_palette(&colors);
        let (ids, regions) = super::vertex_color_id_data(&colors, &indices, &positions, &cancel)
            .expect("high-cardinality ID regions");

        assert_eq!(distinct, triangle_count * 3);
        assert_eq!(palette.len(), MAX_VERTEX_COLOR_ID_REGIONS);
        assert_eq!(ids.len(), palette.len());
        assert_eq!(regions.len(), palette.len());
        let expected: Vec<[f32; 4]> = palette.iter().map(|(rgb, _)| [
            rgb[0] as f32 / 255.0,
            rgb[1] as f32 / 255.0,
            rgb[2] as f32 / 255.0,
            1.0,
        ]).collect();
        assert_eq!(ids, expected, "bake and panel must consume identical top-color ordering");
    }

    #[test]
    fn cancelled_vertex_palette_scan_stops_before_counting() {
        let colors = vec![[[255, 0, 0]; 3]; 4];
        let cancel = std::sync::atomic::AtomicBool::new(true);
        let error = super::vertex_color_palette_cancellable(&colors, &cancel)
            .expect_err("a cancelled palette scan must stop");
        assert!(error.contains("cancelled"), "{error}");
    }

    #[test]
    fn vertex_color_id_source_rejects_high_poly_projection_before_loading_it() {
        let mesh = Primitive::Plane.build();
        let settings = BakeSettings {
            id_source: IdSource::VertexColors,
            high_poly: Some(std::path::PathBuf::from("this-file-does-not-need-to-exist.fbx")),
            ..Default::default()
        };
        let error = match prepare(&mesh, &settings, 0) {
            Ok(_) => panic!("low-poly vertex colors must not index high-poly triangles"),
            Err(error) => error,
        };
        assert!(error.contains("different triangle topology"), "{error}");
        assert!(error.contains("rebake only the ID map"), "{error}");
    }

    #[test]
    fn vertex_color_id_source_rejects_missing_color_attributes() {
        let mesh = Primitive::Plane.build();
        let settings = BakeSettings { id_source: IdSource::VertexColors, ..Default::default() };
        let error = match prepare(&mesh, &settings, 0) {
            Ok(_) => panic!("Vertex Colors mode must not silently fall back to mesh-part IDs"),
            Err(error) => error,
        };
        assert!(error.contains("Vertex Colors was selected as the ID source"), "{error}");
    }

    #[test]
    fn vertex_color_id_source_keeps_corner_colors_aligned_after_bvh_reorder() {
        let mut mesh = Primitive::Sphere.build();
        let triangles = mesh.triangle_count();
        let colors: Vec<[[u8; 3]; 3]> = (0..triangles).map(|triangle| {
            std::array::from_fn(|corner| [
                (triangle % 251) as u8,
                (corner * 83) as u8,
                ((triangle * 7 + corner * 29) % 253) as u8,
            ])
        }).collect();
        mesh.tri_color = Some(colors.iter().map(|corners| {
            std::array::from_fn(|channel| {
                (corners.iter().map(|rgb| rgb[channel] as u32).sum::<u32>() / 3) as u8
            })
        }).collect());
        mesh.tri_vertex_color = Some(colors.clone());
        let settings = BakeSettings { id_source: IdSource::VertexColors, ..Default::default() };

        let prepared = prepare(&mesh, &settings, 0).expect("complete vertex-color source");

        assert_eq!(prepared.tri_vcolor.len(), mesh.indices.len());
        assert_eq!(prepared.bake_triangles.len(), triangles);
        for (bake_triangle, baked_corners) in prepared.bake_triangles.iter()
            .zip(prepared.tri_vcolor.chunks_exact(3))
        {
            let source_triangle = mesh.indices.chunks_exact(3).position(|source| {
                source[0] == bake_triangle.indices[0]
                    && source[1] == bake_triangle.indices[1]
                    && source[2] == bake_triangle.indices[2]
            }).expect("every BVH-ordered triangle must correspond to one source triangle");
            for corner in 0..3 {
                let expected = colors[source_triangle][corner].map(|channel| channel as f32 / 255.0);
                assert_eq!(&baked_corners[corner][..3], &expected);
                assert_eq!(baked_corners[corner][3], 1.0);
            }
        }
    }

    #[test]
    fn vertex_color_id_regions_match_the_bounded_ui_palette() {
        let mut mesh = Primitive::Sphere.build();
        let triangles = mesh.triangle_count();
        let colors = (0..triangles).map(|triangle| {
            std::array::from_fn(|corner| {
                let index = (triangle * 3 + corner) % 60;
                let red = index as u8;
                [red, 255 - red, red.wrapping_mul(7)]
            })
        }).collect::<Vec<_>>();
        mesh.tri_vertex_color = Some(colors);
        let settings = BakeSettings { id_source: IdSource::VertexColors, ..Default::default() };

        let prepared = prepare(&mesh, &settings, 0).expect("vertex-color palette");

        assert_eq!(prepared.id_regions.len(), MAX_VERTEX_COLOR_ID_REGIONS);
        assert!(prepared.ids.len() <= MAX_VERTEX_COLOR_ID_REGIONS);
        let mut ordered: Vec<_> = mesh.tri_vertex_color.as_ref().unwrap().iter()
            .flat_map(|corners| corners.iter().copied())
            .fold(HashMap::<[u8; 3], u32>::new(), |mut counts, color| {
                *counts.entry(color).or_default() += 1;
                counts
            })
            .into_iter()
            .collect();
        ordered.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        ordered.truncate(MAX_VERTEX_COLOR_ID_REGIONS);
        let expected: Vec<_> = ordered.iter().map(|(rgb, _)| rgb.map(|v| v as f32 / 255.0)).collect();
        let actual: Vec<_> = prepared.id_regions.iter().map(|region| region.color).collect();
        assert_eq!(actual, expected, "bake report order must match the Vertex ID panel palette");
    }

    #[test]
    fn vertex_color_id_regions_use_all_corner_colors_not_triangle_averages() {
        let mut mesh = Primitive::Plane.build();
        let triangles = mesh.triangle_count();
        let corners = [[255, 0, 0], [0, 255, 0], [0, 0, 255]];
        mesh.tri_color = Some(vec![[85, 85, 85]; triangles]);
        mesh.tri_vertex_color = Some(vec![corners; triangles]);
        let settings = BakeSettings { id_source: IdSource::VertexColors, ..Default::default() };

        let prepared = prepare(&mesh, &settings, 0).expect("complete vertex-color source");

        assert_eq!(prepared.id_regions.len(), 3, "every distinct vertex color needs its own region");
        let mut actual: Vec<[u8; 3]> = prepared.id_regions.iter().map(|region| {
            region.color.map(|channel| (channel * 255.0).round() as u8)
        }).collect();
        let mut expected = corners.to_vec();
        actual.sort_unstable();
        expected.sort_unstable();
        assert_eq!(actual, expected, "all original per-corner colors must have distinct regions, not one averaged ID");
        for expected in corners {
            let color = expected.map(|v| v as f32 / 255.0);
            let region = prepared.id_regions.iter().find(|region| region.color == color).expect("each corner color has a report region");
            assert_eq!(region.triangles, triangles, "each color appears in every test triangle");
        }
    }

    #[test]
    fn materials_id_source_keeps_one_label_for_disconnected_faces() {
        use crate::mesh::Mesh;
        use glam::{Vec2, Vec3};

        let positions = vec![
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(-0.2, -1.0, 0.0),
            Vec3::new(-1.0, -0.2, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.2, 1.0, 0.0),
            Vec3::new(1.0, 0.2, 0.0),
        ];
        let normals = Some(vec![Vec3::Z; positions.len()]);
        let uvs = vec![
            Vec2::new(0.0, 0.0), Vec2::new(0.4, 0.0), Vec2::new(0.0, 0.4),
            Vec2::new(0.6, 0.6), Vec2::new(1.0, 0.6), Vec2::new(0.6, 1.0),
        ];
        let mesh = Mesh::from_parts(
            "one material, disconnected parts".into(),
            positions,
            normals,
            uvs,
            vec![0, 1, 2, 3, 4, 5],
        );
        assert_eq!(mesh.sets.len(), 1);
        let settings = BakeSettings { id_source: IdSource::Materials, ..Default::default() };

        let prepared = prepare(&mesh, &settings, 0).expect("prepare one-material mesh");
        assert_eq!(prepared.id_regions.len(), 1, "all disconnected faces share the one material ID");
        assert_eq!(prepared.id_regions[0].triangles, 2);
    }

    #[test]
    fn merging_id_regions_preserves_the_full_bake_report() {
        use super::{BakeReport, IdRegion};
        use std::sync::atomic::{AtomicBool, AtomicU32};
        use std::time::Instant;

        let report = |source: &str, regions: Vec<IdRegion>, baked: u32| BakeReport {
            source: source.into(),
            triangles: 12,
            parts: 3,
            hard_edges: 9,
            resolution: 1024,
            rays_per_texel: 48,
            prep_seconds: 0.5,
            edge_max_distance: 0.25,
            id_regions: regions,
            matched_names: 2,
            used_cage: true,
            baked,
            custom: vec![4, 7],
            from_cache: false,
            started: Instant::now(),
            gpu_seconds: AtomicU32::new(1.25f32.to_bits()),
            finished: AtomicBool::new(true),
            valid_texels: AtomicU32::new(100),
            missed_texels: AtomicU32::new(3),
            filled_texels: AtomicU32::new(2),
        };
        let region = |color| IdRegion {
            color,
            triangles: 2,
            min: [0.0; 3],
            max: [1.0; 3],
        };
        let full = report("high poly", vec![region([1.0, 0.0, 0.0])], 0b1111);
        let id_only = report("low poly", vec![region([0.0, 1.0, 0.0]), region([0.0, 0.0, 1.0])], 1 << 4);

        let merged = full.with_id_regions_from(&id_only);

        assert_eq!(merged.source, "high poly");
        assert_eq!(merged.id_regions.len(), 2);
        assert_eq!(merged.id_regions[0].color, [0.0, 1.0, 0.0]);
        assert_eq!(merged.baked, 0b1111);
        assert_eq!(merged.custom, vec![4, 7]);
        assert_eq!(merged.triangles, 12);
        assert_eq!(merged.missed_percent(), Some(3.0));
    }
}
