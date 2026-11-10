//! Application shell: Substance-Painter-style layout, input handling and
//! the per-frame GPU flow (stamps -> composite -> viewport render).

use crate::bake::{self, BakeRun, BakeSettings, GpuBaker, IdSource, PreparedScene, Quality};
use crate::bake_cache;
use crate::bvh::Bvh;
use crate::cage::{CageProbe, CageStats, ProbeKey};
use crate::environment::{self, EnvBaker, EnvImage, EnvSource, Environment};
use crate::engine::{Drop, Folder, Layer, Projection};
use crate::engine::{
    Blend, BrushPaint, Content, Engine, FillParams, GEN_INPUTS, GenKind, Levels, MAPS, MapKind, MatValues, Stamp,
    StampUniform,
};
use crate::gpu::GpuMesh;
use crate::icons;
use crate::mesh::{Mesh, Primitive};
use crate::polyfill::{FillMode, SelectOpts, Topology};
use crate::post::Tonemap;
use crate::presets::{self, PreviewLayer};
use crate::raytrace::{RayTracer, RtSettings};
use crate::theme;
use crate::undo::{StrokeTarget, UndoStack};
use crate::viewport::{Background, Cam2d, FAR, NEAR, OrbitCam, OverlaySlot, OverlayVertex, Overlays, ViewMode, ViewRenderer, ViewSettings};
use eframe::{egui, egui_wgpu, wgpu};
use egui::{Color32, Rect, Sense, Stroke, pos2, vec2};
use glam::{Vec2, Vec3};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

mod export_ui;
mod dock;
mod gallery;
mod folder_graph;
mod graph_ui;
mod shader_ui;
mod library_ui;
mod path_tool;
mod bake_live;
mod spp_open;
mod preview_shader;
mod smart_io;
mod welcome;
mod mcp_tools;
pub(crate) mod prefs;
mod project_io;
mod shelf_previews;
mod texture_sets;
mod cage_paint;
mod color_picker;

use export_ui::{ExportDialog, open_in_file_browser};
use graph_ui::GraphStore;
use prefs::Preferences;

/// Preferences window state.
struct PrefsUi {
    open: bool,
    page: prefs::Page,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tool {
    Paint,
    Eraser,
    /// Picks the brush material from the surface under the cursor.
    Picker,
    /// Fills whole triangles, quads, mesh parts or UV islands.
    PolyFill,
    /// Lays the brush along an editable path (stitches, seams, panel lines).
    Path,
}

/// Bake cage preview (see `crate::cage`): probe results and what's drawn.
struct CagePreview {
    show: bool,
    probe: Option<Arc<CageProbe>>,
    job: Option<(ProbeKey, mpsc::Receiver<Result<CageProbe, String>>)>,
    /// A probe that failed, so it isn't retried until the inputs change.
    failed: Option<(ProbeKey, String)>,
    /// The distances (and painted cage) the overlay was built for.
    drawn: Option<crate::cage::Reach>,
    stats: Option<CageStats>,
}

/// Polygon fill tool state.
struct PolyFillState {
    mode: FillMode,
    /// Removes paint instead of adding it (Ctrl inverts while clicking).
    erase: bool,
    /// Connectivity of the current mesh, built on first use.
    topo: Option<Arc<Topology>>,
    /// Flat area: the sharpest edge it spreads across, in degrees.
    angle: f32,
    /// Shift+drag draws a lasso instead of a box.
    lasso: bool,
    /// Box and lasso also take faces hidden behind others.
    through: bool,
    /// The triangle under the cursor this frame and where on it, from whichever view has it.
    frame_hover: Option<(u32, Vec3)>,
    /// What the selection overlay currently shows.
    shown: Option<PolyShown>,
    /// Triangles filled during the current drag, so each fills once.
    filled: std::collections::HashSet<u32>,
    stroke_erase: bool,
    /// Shift+drag box or lasso selection in progress.
    box_drag: Option<BoxDrag>,
}

/// What the polygon fill highlight was built for: rebuilt when any of it changes.
#[derive(Clone, Copy, PartialEq)]
struct PolyShown {
    tri: u32,
    mode: FillMode,
    erase: bool,
    /// Face loop: the edge it runs through (the pick point decides).
    edge: Option<usize>,
    angle: u32,
}

/// A box or lasso drawn with Shift+drag; the faces inside fill on release.
#[derive(Clone)]
struct BoxDrag {
    view: ViewKind,
    start: egui::Pos2,
    end: egui::Pos2,
    /// The lasso's outline (empty for a box).
    lasso: Vec<egui::Pos2>,
    erase: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Layout {
    Split,
    View3d,
    View2d,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ViewKind {
    V3d,
    V2d,
}

/// The top-level workspaces, switched by the tabs in the menu bar.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Workspace {
    /// Painting and layers on the mesh.
    Paint,
    /// Material graphs: the node canvas next to the material on the mesh.
    Designer,
    /// Writing custom node shaders with a live preview.
    ShaderMaker,
}

impl Workspace {
    const ALL: [Workspace; 3] = [Workspace::Paint, Workspace::Designer, Workspace::ShaderMaker];
    fn name(self) -> &'static str {
        match self {
            Workspace::Paint => "Material Paint",
            Workspace::Designer => "Material Designer",
            Workspace::ShaderMaker => "Shader Maker",
        }
    }
    fn icon(self) -> &'static str {
        match self {
            Workspace::Paint => icons::PAINT_BRUSH,
            Workspace::Designer => icons::SPARKLE,
            Workspace::ShaderMaker => icons::MAGIC_WAND,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ShelfCat {
    SmartMaterials,
    Materials,
    Generators,
    Brushes,
    Textures,
    Graphs,
    /// Materials downloaded from the asset library.
    Downloaded,
}

struct Brush {
    /// Diameter in screen points.
    size: f32,
    hardness: f32,
    flow: f32,
    opacity: f32,
    /// Stamp spacing as a fraction of the diameter.
    spacing: f32,
    mat: MatValues,
    mask_value: f32,
    /// Value painted onto a baked map when fixing it.
    fix_value: f32,
    /// Mirror strokes across the mesh's center plane: 0 off, 1 X, 2 Y, 3 Z.
    symmetry: u8,
    /// Brush tip from `brushes::BRUSH_ALPHAS`; `None` is the round brush.
    tip: Option<usize>,
    /// Tip rotation in degrees.
    angle: f32,
    /// Turn the tip along the stroke direction.
    follow: bool,
    /// Random rotation per stamp, 0..1 of a full turn.
    angle_jitter: f32,
    /// Random shrink per stamp, 0..1.
    size_jitter: f32,
    /// Random offset per stamp, in brush radii.
    scatter: f32,
    /// Jitter random state.
    rng: u32,
    /// How stamps reach the surface in the 3D view.
    projection: BrushProjection,
    /// Screen projection: also paint surfaces hidden behind others.
    paint_through: bool,
    /// Paint fades out on surfaces turned further than this from the brush (degrees).
    angle_limit: f32,
    /// Lazy mouse: the brush trails the cursor by this many points, smoothing strokes.
    stabilize: f32,
    /// Pen pressure scales the stamp size / flow.
    pressure_size: bool,
    pressure_flow: bool,
    /// Grain texture from `textures::TEXTURES` that breaks up the paint.
    grain: Option<usize>,
    /// Grain repeats across the mesh (or the UV square).
    grain_scale: f32,
    grain_strength: f32,
    grain_invert: bool,
}

/// How brush stamps are projected onto the mesh.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum BrushProjection {
    /// From the camera: what you see under the brush gets paint.
    Screen,
    /// A disc lying on the surface under the cursor: no stretching on
    /// slopes, wraps around edges and onto hidden faces.
    Surface,
    /// A disc in texture space around the cursor's UV.
    Uv,
}

impl BrushProjection {
    pub const ALL: [BrushProjection; 3] = [BrushProjection::Screen, BrushProjection::Surface, BrushProjection::Uv];
    pub fn name(self) -> &'static str {
        match self {
            BrushProjection::Screen => "Screen",
            BrushProjection::Surface => "Surface",
            BrushProjection::Uv => "UV",
        }
    }
    pub fn key(self) -> &'static str {
        match self {
            BrushProjection::Screen => "screen",
            BrushProjection::Surface => "surface",
            BrushProjection::Uv => "uv",
        }
    }
    fn tip(self) -> &'static str {
        match self {
            BrushProjection::Screen => "Paint from the camera, like a spray can: only what you see gets paint",
            BrushProjection::Surface => "The brush lies on the surface: even dots on slopes, wraps around edges",
            BrushProjection::Uv => "The brush is a circle in texture space around the cursor",
        }
    }
}

impl Default for Brush {
    fn default() -> Self {
        Self {
            size: 70.0,
            hardness: 0.55,
            flow: 1.0,
            opacity: 1.0,
            spacing: 0.08,
            mat: MatValues { color: [0.78, 0.2, 0.12], rough: 0.4, height: 0.5, ..MatValues::default() },
            mask_value: 1.0,
            fix_value: 0.5,
            symmetry: 0,
            tip: None,
            angle: 0.0,
            follow: false,
            angle_jitter: 0.0,
            size_jitter: 0.0,
            scatter: 0.0,
            rng: 0x9e37_79b9,
            projection: BrushProjection::Screen,
            paint_through: false,
            angle_limit: 85.0,
            stabilize: 0.0,
            pressure_size: true,
            pressure_flow: false,
            grain: None,
            grain_scale: 4.0,
            grain_strength: 1.0,
            grain_invert: false,
        }
    }
}

struct StrokeInput {
    view: ViewKind,
    /// Where the last stamp went.
    last: Vec2,
    /// The stabilized brush position trailing the cursor.
    lazy: Vec2,
    /// Pen pressure at the last stamp.
    pressure: f32,
}

/// Where the last brush stroke ended, for Shift+click straight lines. Only
/// valid while the view hasn't moved.
#[derive(Clone, Copy)]
struct LineAnchor {
    view: ViewKind,
    at: Vec2,
    key: [f32; 8],
}

enum BakeJob {
    /// Worker thread is looking the bake up in the cache, or else loading the
    /// source and building the BVH.
    Preparing(mpsc::Receiver<Result<Prepared, String>>),
    /// GPU passes in progress; the result is cached under the key.
    Running(BakeRun, bake_cache::Key),
}

enum Prepared {
    Scene(PreparedScene, bake_cache::Key),
    Cached(bake_cache::Entry),
}

struct Shelf {
    cat: ShelfCat,
    search: String,
    /// Only materials of this type (smart, plain and downloaded materials); `None`: all,
    /// grouped by type.
    kind: Option<presets::MaterialType>,
    /// Thumbnail edge in points (Ctrl+scroll over the shelf, or the slider).
    icon_size: f32,
    smart: Vec<egui::TextureHandle>,
    materials: Vec<egui::TextureHandle>,
    generators: Vec<egui::TextureHandle>,
    brushes: Vec<egui::TextureHandle>,
    textures: Vec<egui::TextureHandle>,
}

enum Action {
    NewProject(Primitive),
    /// `None` asks for a file.
    OpenProject(Option<std::path::PathBuf>),
    /// Opens a Substance 3D Painter project as a new project.
    OpenSpp(Option<std::path::PathBuf>),
    SaveProject,
    SaveProjectAs,
    /// Saves, then runs the action if the save went through.
    SaveThen(Box<Action>),
    /// Runs an action that would discard unsaved changes, without asking.
    Confirmed(Box<Action>),
    Exit,
    ChangeMesh(Primitive),
    ImportMesh,
    Undo,
    Redo,
    AddPaint,
    AddFill(FillParams, String),
    AddMask { white: bool },
    AddGenerator(GenKind),
    RemoveMask,
    Duplicate,
    CopyLayer,
    PasteLayer,
    Delete,
    Clear,
    MoveTo { from: usize, drop: Drop },
    /// Loads (or with `None` clears) the image on a fill layer channel.
    SetFillImage { idx: usize, ch: usize, path: Option<std::path::PathBuf> },
    /// A shelf item dropped at a spot in the layer stack.
    ShelfDrop { item: ShelfItem, at: usize, parent: Option<u64> },
    /// A fill layer driven by a new material graph.
    AddGraphLayer,
    /// A graph fill layer from a ready-made graph (`graph::presets::PRESETS`).
    AddGraphPreset(usize),
    /// A mask on the selected layer driven by a new material graph.
    AddGraphMask,
    /// Opens the selected layer's graph in the Material Graph window.
    EditGraph,
    GraphUndo,
    GraphRedo,
    /// Opens a custom node in the code editor.
    GraphEditCode(String),
    /// Copies a node into the user folder as a custom node and opens it.
    GraphCopyNode(String),
    /// A graph renders again (a node it uses was saved).
    GraphRefresh(u64),
    /// Puts a node into the graph open in the Material Designer and shows it.
    GraphAddNode(String),
    AddOnTop(ShelfItem),
    /// Saves the selected layer (with its folder contents) as a smart material.
    SaveSmart,
    AddFolder,
    /// Puts the selected layer into a new folder.
    GroupIntoFolder,
    SetResolution(u32),
    /// Bakes the texture set being edited.
    Bake,
    /// Bakes every texture set.
    BakeAll,
    /// Selects a brush tip (`None`: round) and switches to painting.
    SetBrushTip(Option<usize>),
    /// Puts a shelf texture on a channel of the selected fill layer.
    TextureToChannel { texture: usize, ch: usize },
}

pub struct PainterApp {
    rs: egui_wgpu::RenderState,
    mesh: Arc<Mesh>,
    bvh: Arc<Bvh>,
    gpu_mesh: GpuMesh,
    /// The texture set being edited (see `texture_sets`).
    engine: Engine,
    sets: texture_sets::TextureSets,
    views: ViewRenderer,
    raytracer: RayTracer,
    rt: RtSettings,
    undo: UndoStack,
    selected: usize,
    target: StrokeTarget,
    tool: Tool,
    brush: Brush,
    other_brush: Brush,
    color2: [f32; 3],
    color_sliders: ColorSliders,
    /// Target mask generator while sampling a vertex ID from the 3D mesh.
    id_color_pick_target: Option<(usize, Option<usize>)>,
    /// Whether imported per-vertex colors are currently shown over the 3D mesh.
    vertex_color_preview_active: bool,
    /// Latest material edit; the expensive preview rebuild waits until this settles.
    preview_update_at: Option<Instant>,
    cam: OrbitCam,
    cam2d: Cam2d,
    layout: Layout,
    view: ViewSettings,
    /// Lighting environment for the viewports.
    env: Environment,
    env_baker: EnvBaker,
    /// An HDRI decoding on a worker thread.
    env_load: Option<mpsc::Receiver<Result<EnvImage, String>>>,
    /// Built-in HDRIs whose files are installed.
    env_builtins: Vec<usize>,
    /// HDRI preview pictures by environment name.
    hdri_thumbs: std::collections::HashMap<String, egui::TextureHandle>,
    /// Built-in HDRI pictures arriving from the worker that makes them.
    hdri_thumb_rx: Option<mpsc::Receiver<(String, Vec<u8>)>>,
    /// The picture of an environment that just loaded.
    hdri_new_thumb: Option<(String, Vec<u8>)>,
    stroke: Option<StrokeInput>,
    bake_settings: BakeSettings,
    bake_job: Option<BakeJob>,
    baker: GpuBaker,
    show_bake: bool,
    shelf: Shelf,
    /// Renders the shelf thumbnails on a baked material ball; gone once done.
    shelf_previews: Option<shelf_previews::ShelfPreviews>,
    export: ExportDialog,
    prefs: Preferences,
    /// The preferences in effect (and saved); `prefs` is what the UI edits.
    prefs_applied: Preferences,
    prefs_ui: PrefsUi,
    /// Material graphs: the node library, the graphs layers use, the editor.
    graphs: GraphStore,
    workspace: Workspace,
    /// The docked panel layouts of the workspaces.
    docks: dock::Docks,
    /// A 3D view drew this frame (else its depth is rendered offscreen).
    drew_3d: bool,
    pending: Vec<Action>,
    /// A deep copy of a layer block; survives texture-set switches.
    layer_clipboard: Option<Vec<Layer>>,
    status: String,
    show_help: bool,
    show_about: bool,
    /// The welcome window (startup, or Help → Welcome).
    show_welcome: bool,
    welcome_tex: Option<egui::TextureHandle>,
    library: library_ui::LibraryUi,
    user_smarts: Vec<smart_io::UserSmart>,
    path: path_tool::PathTool,
    bake_preview: bake_live::LivePreview,
    custom_ui: bake_live::CustomUi,
    spp: spp_open::SppOpen,
    preview: preview_shader::PreviewShader,
    /// A crash report the last run left, shown once.
    crash_report: Option<std::path::PathBuf>,
    frame_ms: f32,
    ppp: f32,
    mcp: Option<crate::mcp::McpServer>,
    /// MCP calls answered when the running bake completes.
    mcp_bake_waiters: Vec<mpsc::Sender<crate::mcp::ToolResult>>,
    /// The open project file, if it has been saved or opened.
    project_path: Option<std::path::PathBuf>,
    /// Undo state at the last save (see `UndoStack::state_serial`).
    saved_serial: u64,
    /// An action waiting for "save changes?" to be answered.
    confirm: Option<Action>,
    /// The window may close: changes were saved or discarded.
    allow_close: bool,
    recent: Vec<std::path::PathBuf>,
    /// Undo state at the last autosave.
    autosaved_serial: u64,
    /// An autosave from a session that didn't close cleanly, offered at startup.
    recovery: Option<std::path::PathBuf>,
    window_title: String,
    /// MCP calls answered once the loading HDRI is in use.
    mcp_env_waiters: Vec<mpsc::Sender<crate::mcp::ToolResult>>,
    /// Offscreen renderer for MCP previews, created on first use.
    mcp_views: Option<ViewRenderer>,
    /// Geometry drawn over the 3D view: bake cage, polygon fill selection.
    overlays: Overlays,
    cage_preview: CagePreview,
    cage_brush: cage_paint::CageBrush,
    poly: PolyFillState,
    /// The brush tip the engine holds (`None` until the first upload).
    uploaded_tip: Option<Option<usize>>,
    /// The brush grain the engine holds (`None` until the first upload).
    uploaded_grain: Option<Option<usize>>,
    /// Pen pressure, 0..1 (1 for mice).
    pressure: f32,
    line_anchor: Option<LineAnchor>,
}

const RESOLUTIONS: [u32; 4] = [512, 1024, 2048, 4096];
/// Where the source and releases live.
pub(crate) const REPO_URL: &str = "https://github.com/alla2001/material-painter";
/// `selected` when no layer is selected.
const NO_LAYER: usize = usize::MAX;
/// Wait until material controls have been idle before rebuilding the preview.
const MATERIAL_PREVIEW_DELAY: Duration = Duration::from_millis(1200);

fn srgb_u8(c: [f32; 3]) -> Color32 {
    Color32::from_rgb((c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8)
}

fn paint_checker(painter: &egui::Painter, rect: Rect) {
    painter.rect_filled(rect, 0.0, Color32::from_gray(0x55));
    let n = 4;
    let s = rect.width() / n as f32;
    for y in 0..n {
        for x in 0..n {
            if (x + y) % 2 == 0 {
                let r = Rect::from_min_size(rect.min + vec2(x as f32 * s, y as f32 * s), vec2(s, s));
                painter.rect_filled(r, 0.0, Color32::from_gray(0x77));
            }
        }
    }
}

/// Image files a fill layer can load.
const IMAGE_EXT: [&str; 9] = ["png", "jpg", "jpeg", "tga", "bmp", "tif", "tiff", "exr", "hdr"];

/// Shelf thumbnail size range, in points.
const ICON_MIN: f32 = 32.0;
/// Below this thumbnail size the shelf shows thumbnails alone; names show on hover.
const NAMED_MIN: f32 = 56.0;
/// Below this shelf width the categories show as icons and the header packs tight.
const SHELF_COMPACT: f32 = 720.0;
const ICON_MAX: f32 = 128.0;

/// A card-style section title: semibold white text over a hairline.
fn section(ui: &mut egui::Ui, title: &str) {
    ui.add_space(10.0);
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 22.0), Sense::hover());
    ui.painter().text(rect.left_center(), egui::Align2::LEFT_CENTER, title, egui::FontId::new(14.0, theme::heading_family()), Color32::WHITE);
    ui.painter().line_segment([rect.left_bottom(), rect.right_bottom()], Stroke::new(1.0, theme::OUTLINE));
    ui.add_space(6.0);
}

fn labeled<R>(ui: &mut egui::Ui, label: &str, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    ui.horizontal(|ui| {
        ui.add_sized([92.0, 18.0], egui::Label::new(egui::RichText::new(label).color(theme::TEXT_DIM)));
        add(ui)
    })
    .inner
}

/// Normal, uv and uv units per world unit at a point of a triangle.
fn surface_at(mesh: &Mesh, tri: u32, bary: Vec3) -> (Vec3, Vec2, f32) {
    let v = |k: usize| &mesh.vertices[mesh.indices[tri as usize * 3 + k] as usize];
    let vs = [v(0), v(1), v(2)];
    let p = vs.map(|v| Vec3::from(v.pos));
    let t = vs.map(|v| Vec2::from(v.uv));
    let face = (p[1] - p[0]).cross(p[2] - p[0]);
    let smooth = Vec3::from(vs[0].normal) * bary.x + Vec3::from(vs[1].normal) * bary.y + Vec3::from(vs[2].normal) * bary.z;
    let n = smooth.try_normalize().unwrap_or_else(|| face.normalize_or(Vec3::Y));
    let uv = t[0] * bary.x + t[1] * bary.y + t[2] * bary.z;
    let world_area = face.length();
    let uv_area = (t[1] - t[0]).perp_dot(t[2] - t[0]).abs();
    let scale = if world_area > 1e-12 { (uv_area / world_area).sqrt() } else { 0.0 };
    (n, uv, scale)
}

/// Turns screen stamps (with their flow) into what the stamp shader
/// reads. The surface and UV projections find the surface under each
/// stamp, sized so it covers what the brush circle covers on screen, plus
/// its mirror image when `symmetry` is on (the screen projection mirrors
/// in the shader instead).
pub(crate) fn resolve_stamps(
    mesh: &Mesh,
    bvh: &Bvh,
    cam: &OrbitCam,
    size: Vec2,
    projection: BrushProjection,
    symmetry: u8,
    radius: f32,
    raw: &[([f32; 4], f32)],
) -> Vec<Stamp> {
    let mut out = Vec::with_capacity(raw.len() * if symmetry > 0 { 2 } else { 1 });
    let eye = cam.eye();
    let fwd = (cam.target - eye).normalize();
    let (right, _) = cam.basis();
    // World size of one pixel at unit distance in front of the camera.
    let px_world = 2.0 * (cam.fov_y * 0.5).tan() / size.y.max(1.0);
    let tangent = |n: Vec3, r: Vec3| (r - n * r.dot(n)).try_normalize().unwrap_or_else(|| n.any_orthonormal_vector());
    for &(s, flow) in raw {
        let mut st: Stamp = [s, [0.0; 4], [0.0; 4], [0.0; 4], [0.0, 0.0, flow, 0.0]];
        if projection == BrushProjection::Screen {
            out.push(st);
            continue;
        }
        let (o, d) = cam.ray(Vec2::new(s[0], s[1]), size);
        let Some(hit) = bvh.raycast(o, d) else { continue };
        let r = radius * s[2] * (hit.pos - eye).dot(fwd).max(1e-4) * px_world;
        let (n, uv, uv_scale) = surface_at(mesh, hit.tri, hit.bary);
        let x = tangent(n, right);
        st[1] = hit.pos.extend(r).to_array();
        st[2] = n.extend(r * uv_scale).to_array();
        st[3] = x.extend(0.0).to_array();
        st[4] = [uv.x, uv.y, flow, 0.0];
        out.push(st);
        if symmetry > 0 {
            let mut flip = Vec3::ONE;
            flip[symmetry as usize - 1] = -1.0;
            let (c, n) = (hit.pos * flip, n * flip);
            let mut m = st;
            m[1] = c.extend(r).to_array();
            m[2] = n.extend(r * uv_scale).to_array();
            m[3] = (x * flip).extend(0.0).to_array();
            if projection == BrushProjection::Uv {
                // The mirror image's UV: the surface at the mirrored point.
                match bvh.raycast(c + n * r, -n) {
                    Some(h) if h.pos.distance(c) < 2.0 * r => {
                        let (_, uv, scale) = surface_at(mesh, h.tri, h.bary);
                        m[2][3] = r * scale;
                        m[4][0] = uv.x;
                        m[4][1] = uv.y;
                    }
                    _ => continue,
                }
            }
            out.push(m);
        }
    }
    out
}

fn grow(acc: Option<[f32; 4]>, lo: Vec2, hi: Vec2) -> [f32; 4] {
    match acc {
        Some(a) => [a[0].min(lo.x), a[1].min(lo.y), a[2].max(hi.x), a[3].max(hi.y)],
        None => [lo.x, lo.y, hi.x, hi.y],
    }
}

/// UV bounds of what a frame's stamps can paint in the 3D view: the UV
/// boxes of every triangle they can reach. Screen stamps reach whatever
/// projects into their circles at any depth (and, with symmetry, whatever
/// mirrors into them), surface stamps whatever lies near their discs, UV
/// stamps their own squares. `None` when nothing is reachable.
pub(crate) fn stroke_region(
    mesh: &Mesh,
    bvh: &Bvh,
    cam: &OrbitCam,
    size: Vec2,
    projection: BrushProjection,
    symmetry: u8,
    radius: f32,
    stamps: &[Stamp],
) -> Option<[f32; 4]> {
    if projection == BrushProjection::Uv {
        return stamps.iter().filter(|s| s[1][3] > 0.0).fold(None, |acc, s| {
            let (uv, r) = (Vec2::new(s[4][0], s[4][1]), s[2][3]);
            Some(grow(acc, uv - Vec2::splat(r), uv + Vec2::splat(r)))
        });
    }
    let mut flip = Vec3::ONE;
    if symmetry > 0 {
        flip[symmetry as usize - 1] = -1.0;
    }
    let overlaps: Box<dyn Fn(Vec3, Vec3) -> bool> = match projection {
        BrushProjection::Surface => {
            // Spheres around the discs (planar radius r, slab r).
            let spheres: Vec<(Vec3, f32)> =
                stamps.iter().filter(|s| s[1][3] > 0.0).map(|s| (Vec3::new(s[1][0], s[1][1], s[1][2]), s[1][3] * 1.5)).collect();
            Box::new(move |lo: Vec3, hi: Vec3| spheres.iter().any(|&(c, r)| c.clamp(lo, hi).distance_squared(c) <= r * r))
        }
        _ => {
            // Screen box of all the circles; boxes that reach behind the
            // camera count as overlapping.
            let (mut a, mut b) = (Vec2::MAX, Vec2::MIN);
            for s in stamps {
                let r = radius * s[0][2];
                a = a.min(Vec2::new(s[0][0] - r, s[0][1] - r));
                b = b.max(Vec2::new(s[0][0] + r, s[0][1] + r));
            }
            let vp = cam.view_proj(size.x / size.y.max(1.0));
            let on_screen = move |lo: Vec3, hi: Vec3| {
                let (mut smin, mut smax) = (Vec2::MAX, Vec2::MIN);
                for k in 0..8 {
                    let p = Vec3::new(
                        if k & 1 == 0 { lo.x } else { hi.x },
                        if k & 2 == 0 { lo.y } else { hi.y },
                        if k & 4 == 0 { lo.z } else { hi.z },
                    );
                    let c = vp * p.extend(1.0);
                    if c.w <= 1e-4 {
                        return true;
                    }
                    let q = (Vec2::new(c.x, -c.y) / c.w * 0.5 + 0.5) * size;
                    smin = smin.min(q);
                    smax = smax.max(q);
                }
                smin.x <= b.x && smax.x >= a.x && smin.y <= b.y && smax.y >= a.y
            };
            Box::new(move |lo: Vec3, hi: Vec3| {
                on_screen(lo, hi) || (symmetry > 0 && { let (m, n) = (lo * flip, hi * flip); on_screen(m.min(n), m.max(n)) })
            })
        }
    };
    let mut acc = None;
    bvh.query(overlaps, |t| {
        for k in 0..3 {
            let uv = Vec2::from(mesh.vertices[mesh.indices[t as usize * 3 + k] as usize].uv);
            acc = Some(grow(acc, uv, uv));
        }
    });
    acc
}

/// The third-party license notices shipped next to the executable (or in
/// the source tree when run from it).
fn notices_path() -> Option<std::path::PathBuf> {
    let name = "THIRD_PARTY_NOTICES.html";
    let exe_dir = std::env::current_exe().ok().and_then(|e| e.parent().map(|p| p.to_path_buf()));
    // A macOS app bundle keeps it in Contents/Resources, beside Contents/MacOS.
    let resources = exe_dir.as_ref().and_then(|d| d.parent()).map(|c| c.join("Resources"));
    exe_dir.into_iter().chain(resources).chain(std::env::current_dir().ok()).map(|d| d.join(name)).find(|p| p.is_file())
}

/// HDRI preview pictures: width, height.
const HDRI_THUMB: [usize; 2] = [256, 128];

/// Makes a picture of every built-in HDRI on a worker thread, reading them
/// from the on-disk cache when it has them (else decoding the HDRI and
/// caching the picture), and sends them as they're ready.
fn start_hdri_thumbs(ctx: &egui::Context, builtins: Vec<usize>) -> mpsc::Receiver<(String, Vec<u8>)> {
    let (tx, rx) = mpsc::channel();
    let ctx = ctx.clone();
    std::thread::spawn(move || {
        let dir = project_io::data_dir().map(|d| d.join("hdri_thumbs"));
        let [w, h] = HDRI_THUMB;
        for i in builtins {
            let name = environment::BUILTIN[i].0.to_string();
            let cached = dir.as_ref().map(|d| d.join(format!("{}.png", name.replace(' ', "_"))));
            let px = cached
                .as_ref()
                .and_then(|p| image::open(p).ok())
                .map(|img| img.into_rgba8().into_raw())
                .filter(|px| px.len() == w * h * 4)
                .or_else(|| {
                    let img = environment::decode(&EnvSource::Builtin(i)).ok()?;
                    let px = img.thumbnail(w, h);
                    if let Some(path) = &cached {
                        if let Some(d) = path.parent() {
                            let _ = std::fs::create_dir_all(d);
                        }
                        let _ = image::save_buffer(path, &px, w as u32, h as u32, image::ExtendedColorType::Rgba8);
                    }
                    Some(px)
                });
            if let Some(px) = px {
                if tx.send((name, px)).is_err() {
                    return;
                }
                ctx.request_repaint();
            }
        }
    });
    rx
}

/// Runs `f`, turning a GPU out-of-memory error it causes into a message
/// instead of wgpu's default (a panic). Native error scopes resolve at once.
fn out_of_memory_guard<T>(device: &wgpu::Device, f: impl FnOnce() -> T) -> Result<T, String> {
    use std::future::Future;
    let scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let value = f();
    let mut pop = std::pin::pin!(scope.pop());
    match pop.as_mut().poll(&mut std::task::Context::from_waker(std::task::Waker::noop())) {
        std::task::Poll::Ready(Some(_)) => {
            Err("Not enough GPU memory to bake. Lower the texture set resolution or close other GPU-heavy apps.".into())
        }
        _ => Ok(value),
    }
}

fn projection_combo(ui: &mut egui::Ui, id: &str, p: &mut BrushProjection) {
    egui::ComboBox::from_id_salt(id)
        .width(72.0)
        .selected_text(p.name())
        .show_ui(ui, |ui| {
            for m in BrushProjection::ALL {
                ui.selectable_value(p, m, m.name()).on_hover_text(m.tip());
            }
        })
        .response
        .on_hover_text(p.tip());
}

/// Smart mask inputs by what they read, for the Add input menu (custom
/// bake maps follow, by name).
const INPUT_GROUPS: [(&str, &[usize]); 4] = {
    use crate::engine::gin::*;
    [
        ("Shape", &[EDGE, CONVEX, CONCAVE, CREASE, THIN]),
        ("Light and exposure", &[OCCLUSION, CAVITY, SKY, UP, OVERHANG]),
        ("Position and parts", &[PART_GRADIENT, WORLD_HEIGHT, PART_RANDOM, ID]),
        ("Patterns", &[STREAKS, NOISE, GRAIN, CELLS, SCRATCHES]),
    ]
};

/// A new smart mask of a preset; a Custom Bake Map starts on the project's
/// first custom map.
fn new_generator(kind: GenKind, custom: &[bake::CustomMap]) -> crate::engine::Generator {
    let mut g = kind.default_generator();
    if let (GenKind::CustomMap, Some(c)) = (kind, custom.first()) {
        g.custom[0] = crate::engine::CustomInput { id: c.id, weight: 1.0 };
    }
    g
}

/// A custom bake map's name as masks list it ("missing" once removed).
fn custom_input_name(id: u32, custom: &[bake::CustomMap]) -> String {
    match custom.iter().find(|c| c.id == id) {
        Some(c) => format!("{}  {}", bake_live::kind_icon(c.kind), c.name),
        None => "(removed map)".to_string(),
    }
}

/// Which custom bake map a Custom Bake Map mask reads: the project's maps,
/// by name.
fn custom_map_picker(ui: &mut egui::Ui, g: &mut crate::engine::Generator, salt: usize, custom: &[bake::CustomMap]) -> bool {
    let mut dirty = false;
    // The input the mask reads most.
    let current = g.custom.iter().enumerate().filter(|(_, c)| c.id != 0).max_by(|a, b| a.1.weight.abs().total_cmp(&b.1.weight.abs())).map(|(k, c)| (k, c.id));
    if custom.is_empty() {
        labeled(ui, "Map", |ui| {
            ui.label(egui::RichText::new("No custom maps yet: add one in the bake window (Bake Mesh Maps → Custom maps).").small().color(theme::TEXT_DIM))
        });
        return false;
    }
    labeled(ui, "Map", |ui| {
        let text = current.map_or("Choose…".to_string(), |(_, id)| custom_input_name(id, custom));
        egui::ComboBox::from_id_salt(("gen custom map", salt)).selected_text(text).width(220.0).height(400.0).show_ui(ui, |ui| {
            for c in custom {
                let on = current.is_some_and(|(_, id)| id == c.id);
                let r = ui.add(egui::Button::selectable(on, custom_input_name(c.id, custom))).on_hover_text(c.kind.name());
                if r.clicked() && !on {
                    // Replace the map the mask reads most, keeping its weight.
                    let (k, weight) = current.map_or((g.custom.iter().position(|c| c.id == 0).unwrap_or(0), 1.0), |(k, _)| (k, g.custom[k].weight));
                    g.custom[k] = crate::engine::CustomInput { id: c.id, weight };
                    dirty = true;
                }
            }
        });
    });
    if current.is_some_and(|(_, id)| custom.iter().all(|c| c.id != id)) {
        ui.label(egui::RichText::new("This map was removed in the bake window; pick another.").small().color(Color32::from_rgb(240, 180, 80)));
    }
    dirty
}

/// One row of a mask's Mixes: remove button, name, weight slider.
fn input_row(ui: &mut egui::Ui, name: String, tip: &str, weight: &mut f32) -> (bool, bool) {
    let mut remove = false;
    let mut changed = false;
    ui.horizontal(|ui| {
        // Remove button first, so it never ends up past the panel's edge.
        if ui.add(egui::Button::new(icons::X).frame(false)).on_hover_text("Remove this input").clicked() {
            remove = true;
        }
        let label = egui::Label::new(egui::RichText::new(name).color(theme::TEXT)).truncate();
        ui.allocate_ui_with_layout(vec2(140.0, 18.0), egui::Layout::left_to_right(egui::Align::Center), |ui| {
            ui.set_width(140.0);
            ui.add(label).on_hover_text(tip);
        });
        // The slider takes what's left, next to its value box.
        ui.spacing_mut().slider_width = (ui.available_width() - 60.0).max(40.0);
        changed = ui.add(egui::Slider::new(weight, -2.0..=2.0).max_decimals(2)).changed();
    });
    (remove, changed)
}

/// The inputs a smart mask mixes: only the ones it uses, each with its
/// weight and a remove button, and a menu to add more by kind.
fn generator_inputs_ui(ui: &mut egui::Ui, g: &mut crate::engine::Generator, custom: &[bake::CustomMap]) -> bool {
    let mut dirty = false;
    ui.add_space(4.0);
    ui.label(egui::RichText::new("Mixes").small().color(theme::TEXT_DIM))
        .on_hover_text("What this mask is made of: each input adds (or with a negative weight, takes away) its share.");
    for i in 0..GEN_INPUTS.len() {
        if g.weights[i] == 0.0 {
            continue;
        }
        let (remove, changed) = input_row(ui, GEN_INPUTS[i].0.to_string(), GEN_INPUTS[i].1, &mut g.weights[i]);
        dirty |= changed;
        if remove {
            g.weights[i] = 0.0;
            dirty = true;
        }
    }
    for k in 0..g.custom.len() {
        if g.custom[k].id == 0 {
            continue;
        }
        let name = custom_input_name(g.custom[k].id, custom);
        let (remove, changed) = input_row(ui, name, "A custom bake map (bake window → Custom maps)", &mut g.custom[k].weight);
        dirty |= changed;
        if remove {
            g.custom[k] = crate::engine::CustomInput::default();
            dirty = true;
        }
    }
    let free = g.custom.iter().position(|c| c.id == 0);
    ui.menu_button(theme::with_icon(icons::PLUS, "Add input"), |ui| {
        ui.set_min_width(240.0);
        egui::ScrollArea::vertical().max_height(420.0).show(ui, |ui| {
            for (group, inputs) in INPUT_GROUPS {
                ui.label(egui::RichText::new(group).small().strong().color(theme::TEXT_DIM));
                for &i in inputs {
                    let used = g.weights[i] != 0.0;
                    let r = ui.add_enabled(!used, egui::Button::new(GEN_INPUTS[i].0).frame(false)).on_hover_text(GEN_INPUTS[i].1);
                    if r.clicked() {
                        g.weights[i] = 1.0;
                        dirty = true;
                        ui.close();
                    }
                }
                ui.add_space(4.0);
            }
            ui.label(egui::RichText::new("Custom maps").small().strong().color(theme::TEXT_DIM));
            if custom.is_empty() {
                ui.label(egui::RichText::new("None yet: add them in the bake window.").small().color(theme::TEXT_DIM));
            }
            for c in custom {
                let used = g.custom.iter().any(|x| x.id == c.id);
                let tip = if free.is_none() { "A mask mixes up to four custom maps" } else { c.kind.name() };
                let r = ui.add_enabled(!used && free.is_some(), egui::Button::new(custom_input_name(c.id, custom)).frame(false)).on_hover_text(tip);
                if let (true, Some(k)) = (r.clicked(), free) {
                    g.custom[k] = crate::engine::CustomInput { id: c.id, weight: 1.0 };
                    dirty = true;
                    ui.close();
                }
            }
        });
    })
    .response
    .on_hover_text("Mix in another baked map or pattern");
    dirty
}

fn slider(ui: &mut egui::Ui, label: &str, v: &mut f32, range: std::ops::RangeInclusive<f32>) -> bool {
    labeled(ui, label, |ui| ui.add(egui::Slider::new(v, range).max_decimals(2)).changed())
}

fn color_edit(ui: &mut egui::Ui, c: &mut [f32; 3]) -> bool { color_button(ui, c, None) }
fn color_button(ui: &mut egui::Ui, c: &mut [f32; 3], tip: Option<&str>) -> bool {
    let rgb = c.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8);
    let response = ui.add(egui::Button::new(egui::RichText::new("   ").background_color(Color32::from_rgb(rgb[0], rgb[1], rgb[2])))
        .fill(Color32::from_rgb(rgb[0], rgb[1], rgb[2])).min_size(vec2(28.0, 18.0)));
    let response = if let Some(tip) = tip { response.on_hover_text(tip) } else { response };
    let mut changed = false;
    egui::Popup::from_toggle_button_response(&response)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(|ui| {
            ui.set_width(color_picker::WIDTH + 28.0);
            egui::ScrollArea::vertical().max_height(600.0).show(ui, |ui| {
                // The hue is remembered between frames so it survives passing through gray or black.
                let memory = response.id.with("picker_hsv");
                let mut hsv = match ui.data(|d| d.get_temp::<[f32; 3]>(memory)) {
                    Some(previous) => hsv_keeping(previous, *c),
                    None => rgb_to_hsv(*c),
                };
                let picked = color_picker::picker(ui, &mut hsv);
                if picked { *c = hsv_to_rgb(hsv); changed = true; }
                let mut hsv_values = [hsv[0] * 360.0, hsv[1] * 100.0, hsv[2] * 100.0];
                for (label, index, range) in [("H", 0usize, 0.0..=360.0), ("S", 1usize, 0.0..=100.0), ("V", 2usize, 0.0..=100.0)] {
                    changed |= labeled(ui, label, |ui| ui.add(egui::Slider::new(&mut hsv_values[index], range).max_decimals(2)).changed());
                }
                if changed && !picked {
                    hsv = [hsv_values[0] / 360.0, hsv_values[1] / 100.0, hsv_values[2] / 100.0];
                    *c = hsv_to_rgb(hsv);
                }
                ui.separator();
                let mut channels = c.map(|v| v * 255.0);
                for (label, index) in [("R", 0usize), ("G", 1usize), ("B", 2usize)] {
                    changed |= labeled(ui, label, |ui| ui.add(egui::Slider::new(&mut channels[index], 0.0..=255.0).max_decimals(0)).changed());
                }
                if changed { *c = channels.map(|v| (v / 255.0).clamp(0.0, 1.0)); }
                let mut hex = color_picker::hex_string(*c);
                ui.horizontal(|ui| {
                    ui.label("HEX");
                    if ui.add(egui::TextEdit::singleline(&mut hex).desired_width(90.0)).changed() {
                        if let Some(parsed) = parse_color_hex(&hex) { *c = parsed; changed = true; }
                    }
                });
                ui.data_mut(|d| d.insert_temp(memory, if changed { hsv_keeping(hsv, *c) } else { hsv }));
            });
        });
    changed
}
fn swap_brushes(active: &mut Brush, other: &mut Brush) {
    std::mem::swap(active, other);
    let left = &*other;
    active.mat = MatValues { enabled: active.mat.enabled, ..left.mat };
    active.mask_value = left.mask_value;
    active.fix_value = left.fix_value;
    active.symmetry = left.symmetry;
    active.projection = left.projection;
    active.paint_through = left.paint_through;
    active.angle_limit = left.angle_limit;
    active.stabilize = left.stabilize;
}
fn swap_button(ui: &mut egui::Ui) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(vec2(20.0, 20.0), egui::Sense::click());
    let stroke = egui::Stroke::new(1.5, if response.hovered() { Color32::WHITE } else { theme::TEXT });
    let center = rect.center();
    ui.painter().arrow(center + vec2(5.0, -3.5), vec2(-10.0, 0.0), stroke);
    ui.painter().arrow(center + vec2(-5.0, 3.5), vec2(10.0, 0.0), stroke);
    response
}
#[derive(Default)]
struct ColorSliders {
    hsv: [f32; 3],
    out: [f32; 3],
    temp: f32,
    base: [f32; 3],
    shifting: bool,
    hsv_mode: bool,
    hex_edit: String,
    hex_dirty: bool,
    /// Color when the panel was opened, for the old | new swatch.
    opened: [f32; 3],
}
fn parse_color_hex(value: &str) -> Option<[f32; 3]> {
    let raw = value.trim().strip_prefix('#').unwrap_or(value.trim());
    if raw.len() != 6 { return None; }
    let packed = u32::from_str_radix(raw, 16).ok()?;
    Some([
        ((packed >> 16) & 0xff) as f32 / 255.0,
        ((packed >> 8) & 0xff) as f32 / 255.0,
        (packed & 0xff) as f32 / 255.0,
    ])
}
fn rgb_to_hsv(c: [f32; 3]) -> [f32; 3] {
    let max = c[0].max(c[1]).max(c[2]);
    let delta = max - c[0].min(c[1]).min(c[2]);
    let h = if delta <= 0.0 { 0.0 }
    else if max == c[0] { ((c[1] - c[2]) / delta).rem_euclid(6.0) }
    else if max == c[1] { (c[2] - c[0]) / delta + 2.0 }
    else { (c[0] - c[1]) / delta + 4.0 };
    [h / 6.0, if max > 0.0 { delta / max } else { 0.0 }, max]
}
fn hsv_to_rgb([h, s, v]: [f32; 3]) -> [f32; 3] {
    let channel = |n: f32| { let k = (n + h * 6.0) % 6.0; v - v * s * k.min(4.0 - k).clamp(0.0, 1.0) };
    [channel(5.0), channel(3.0), channel(1.0)]
}
fn hsv_keeping(prev: [f32; 3], color: [f32; 3]) -> [f32; 3] {
    let next = rgb_to_hsv(color);
    [if next[1] > 0.0 && next[2] > 0.0 { next[0] } else { prev[0] },
     if next[2] > 0.0 { next[1] } else { prev[1] }, next[2]]
}
fn shift_temperature(color: [f32; 3], amount: f32) -> [f32; 3] {
    let gain = [1.0 + 0.30 * amount, 1.0 + 0.05 * amount, 1.0 - 0.30 * amount];
    std::array::from_fn(|i| (color[i] * gain[i]).clamp(0.0, 1.0))
}

/// Channel toggles in Substance order: color, height, rough, metal, opacity.
fn channel_toggles(ui: &mut egui::Ui, enabled: &mut [bool; crate::engine::CHANNELS]) -> bool {
    let mut changed = false;
    ui.horizontal_wrapped(|ui| {
        use crate::engine::{NORMAL, SPECULAR};
        for (c, name) in [(0, "color"), (3, "height"), (1, "rough"), (2, "metal"), (SPECULAR, "specular"), (NORMAL, "normal"), (4, "opacity"), (5, "emission")] {
            let on = enabled[c];
            let text = egui::RichText::new(name).color(if on { Color32::WHITE } else { theme::TEXT_DIM });
            let b = egui::Button::new(text).fill(if on { theme::ACCENT_DIM } else { theme::WIDGET }).min_size(vec2(56.0, 20.0));
            let b = ui.add(b);
            let b = match c {
                NORMAL => b.on_hover_text("Normal detail from a normal-map image (fill layers). A brush paints it flat, smoothing the layer's normal detail away."),
                SPECULAR => b.on_hover_text("How much non-metals reflect: 0.5 is the usual 4%"),
                _ => b,
            };
            if b.clicked() {
                enabled[c] = !on;
                changed = true;
            }
        }
    });
    changed
}

fn material_controls(ui: &mut egui::Ui, m: &mut MatValues, show_values: bool) -> bool {
    let mut changed = channel_toggles(ui, &mut m.enabled);
    if !show_values {
        return changed;
    }
    ui.add_space(4.0);
    if m.enabled[0] {
        changed |= labeled(ui, "Base color", |ui| color_edit(ui, &mut m.color));
    }
    if m.enabled[3] {
        changed |= slider(ui, "Height", &mut m.height, -1.0..=1.0);
    }
    if m.enabled[1] {
        changed |= slider(ui, "Roughness", &mut m.rough, 0.0..=1.0);
    }
    if m.enabled[2] {
        changed |= slider(ui, "Metallic", &mut m.metal, 0.0..=1.0);
    }
    if m.enabled[crate::engine::SPECULAR] {
        // 0.5 is the usual 4% reflectance of non-metals; metals reflect their color.
        changed |= slider(ui, "Specular", &mut m.specular, 0.0..=1.0);
    }
    if m.enabled[4] {
        // 0 is fully see-through; exports as the base color's alpha or an opacity map.
        changed |= slider(ui, "Opacity", &mut m.opacity, 0.0..=1.0);
    }
    if m.enabled[5] {
        // Black emits nothing; brightness in the view is the view's emission strength.
        changed |= labeled(ui, "Emission", |ui| color_edit(ui, &mut m.emission));
    }
    changed
}

/// Adds a smart material as a folder holding its layers, at `at` inside
/// `parent`, as one undo step; returns the folder's index.
pub(crate) fn add_smart_material(
    engine: &mut Engine,
    undo: &mut UndoStack,
    smart: &presets::SmartMaterial,
    at: usize,
    parent: Option<u64>,
) -> usize {
    undo.begin_group(format!("Add {}", smart.name));
    let mut folder = engine.add_folder(at, parent, smart.name.to_string(), undo);
    let folder_id = engine.layers[folder].id;
    for l in smart.layers.iter().chain(&presets::MICRO_DETAIL) {
        // Each layer goes on top of the folder's contents, right below the folder.
        let idx = engine.add_fill_layer(folder, Some(folder_id), l.name.to_string(), l.fill(), undo);
        folder = idx + 1;
        if let Some(scan) = l.scan {
            // Maps the set lacks (or a missing assets folder) leave the
            // material's own values in place.
            if let Some(dir) = crate::environment::asset_dir("pbr").map(|d| d.join(scan.id)) {
                for (ch, file) in presets::Scan::FILES {
                    let path = dir.join(file);
                    if path.is_file()
                        && let Ok(img) = engine.shared_fill_image(&path)
                    {
                        engine.set_fill_image(idx, ch, Some(img), undo);
                    }
                }
            }
        }
        let layer = &mut engine.layers[idx];
        layer.opacity = l.opacity;
        layer.blend = l.blend;
        if let Some(g) = l.generator() {
            engine.add_mask(idx, true, Some(g), undo);
        }
    }
    engine.commit_props(undo);
    undo.end_group();
    folder
}

/// What was dropped on `resp` this frame, if it's a `T`. egui's
/// `dnd_release_payload` takes the payload whatever its type, so asking a
/// widget for one kind (a layer being moved) would throw away another (a
/// shelf item) that a later check on the same widget wants.
fn dnd_release<T: std::any::Any + Send + Sync>(resp: &egui::Response) -> Option<Arc<T>> {
    egui::DragAndDrop::has_payload_of_type::<T>(&resp.ctx).then(|| resp.dnd_release_payload::<T>()).flatten()
}

/// Something dragged off the shelf.
#[derive(Clone, Copy)]
pub(crate) enum ShelfItem {
    /// A smart material, as a folder of fill layers with smart masks (its
    /// live graph opens in the Material Designer).
    Smart(usize),
    /// A smart material as one graph fill layer.
    SmartGraph(usize),
    Material(usize),
    Generator(GenKind),
    /// A brush tip (`brushes::BRUSH_ALPHAS`).
    Brush(usize),
    /// A tileable texture (`textures::TEXTURES`), added as a fill layer's height.
    Texture(usize),
    /// A saved material graph (`GraphStore::shelf`).
    Graph(usize),
    /// A downloaded material (`LibraryUi::local`).
    Library(usize),
    /// A smart material the user saved or downloaded (`user_smarts`).
    UserSmart(usize),
}

/// A layer being dragged in the layer stack.
#[derive(Clone, Copy)]
struct LayerDrag(usize);

impl PainterApp {
    /// Switches workspace (offscreen UI tests).
    #[cfg(test)]
    pub(crate) fn set_workspace(&mut self, ws: Workspace) {
        self.workspace = ws;
    }

    /// Waits for every built-in HDRI picture, then shows the Preview tab
    /// (offscreen UI tests).
    #[cfg(test)]
    pub(crate) fn test_show_hdris(&mut self, ctx: &egui::Context) {
        if let Some(rx) = self.hdri_thumb_rx.take() {
            for (name, px) in rx.iter() {
                let tex = ctx.load_texture(format!("hdri {name}"), egui::ColorImage::from_rgba_unmultiplied(HDRI_THUMB, &px), Default::default());
                self.hdri_thumbs.insert(name, tex);
            }
        }
        self.docks.show_pane(Workspace::Paint, dock::Pane::Preview);
        self.export.open = false;
    }

    /// Turns on the UV seams overlay in 3D (offscreen UI tests).
    #[cfg(test)]
    pub(crate) fn test_show_seams(&mut self) {
        self.view.show_seams_3d = true;
    }

    /// Shows the plain materials on the shelf, every type (offscreen UI tests).
    #[cfg(test)]
    pub(crate) fn test_shelf_materials(&mut self) {
        self.shelf.cat = ShelfCat::Materials;
        self.shelf.kind = None;
    }

    /// Closes the welcome window (offscreen UI tests).
    #[cfg(test)]
    pub(crate) fn test_close_welcome(&mut self) {
        self.show_welcome = false;
    }

    /// Opens the asset library and shows the Downloaded shelf (offscreen UI
    /// tests); true once the open tab's catalog has arrived.
    #[cfg(test)]
    pub(crate) fn test_open_library(&mut self) -> bool {
        self.export.open = false;
        self.library.open = true;
        self.shelf.cat = ShelfCat::Downloaded;
        self.library.test_ready()
    }

    /// Saves a built-in scanned smart material as a user smart material and
    /// applies it again (offscreen UI tests). Returns a summary of the
    /// original and the applied block: (layers, masks, generators, filters,
    /// images, folders).
    #[cfg(test)]
    pub(crate) fn test_smart_roundtrip(&mut self, ctx: &egui::Context) -> [[usize; 6]; 2] {
        let k = presets::SMART_MATERIALS.iter().position(|s| s.is_scanned()).expect("a scanned smart material");
        let at = self.engine.layers.len();
        self.selected = add_smart_material(&mut self.engine, &mut self.undo, &presets::SMART_MATERIALS[k], at, None);
        // A mask filter, which smart materials must keep.
        let first = self.engine.block(self.selected).start;
        if let Some(m) = self.engine.layers[first..self.selected].iter_mut().find_map(|l| l.mask.as_mut()) {
            m.filters.push(crate::engine::MaskFilter::new(crate::engine::FilterKind::ALL[0]));
        }
        let summary = |e: &Engine, top: usize| {
            let b = e.block(top);
            let ls = &e.layers[b];
            [
                ls.len(),
                ls.iter().filter(|l| l.mask.is_some()).count(),
                ls.iter().filter(|l| l.mask.as_ref().is_some_and(|m| m.generator.is_some())).count(),
                ls.iter().map(|l| l.mask.as_ref().map_or(0, |m| m.filters.len())).sum(),
                ls.iter().map(|l| l.images.iter().flatten().count()).sum(),
                ls.iter().filter(|l| l.is_folder()).count(),
            ]
        };
        let before = summary(&self.engine, self.selected);
        self.save_smart_material(ctx);
        let path = self.user_smarts.iter().find(|s| s.name.starts_with(presets::SMART_MATERIALS[k].name)).expect("saved").path.clone();
        let at = self.engine.layers.len();
        self.apply_smart_material(&path, at, None);
        [before, summary(&self.engine, self.selected)]
    }

    /// Bakes again without ambient occlusion and edge distance (offscreen UI
    /// tests); true once that bake is in.
    #[cfg(test)]
    pub(crate) fn test_bake_without(&mut self, ctx: &egui::Context, maps: u32) -> bool {
        if self.bake_settings.maps != crate::bake::ALL_MAPS & !maps {
            self.bake_settings.maps = crate::bake::ALL_MAPS & !maps;
            self.start_bake(ctx);
        }
        self.bake_job.is_none() && self.engine.missing_maps() == maps
    }

    /// Sets up two custom bake maps, renames the first and shows the bake
    /// window with it open (offscreen UI tests).
    #[cfg(test)]
    pub(crate) fn test_show_custom(&mut self) {
        self.show_bake = true;
        self.add_custom_map(bake::CustomKind::Flow);
        self.add_custom_map(bake::CustomKind::Deposit);
        self.bake_settings.custom[0].name = "Rain from above".into();
        self.select_custom(self.bake_settings.custom[0].id);
    }

    /// A fill layer masked by the Custom Bake Map preset reading the second
    /// custom map, mixed with occlusion, selected with its mask (offscreen UI tests).
    #[cfg(test)]
    pub(crate) fn test_custom_mask(&mut self) {
        use crate::engine::gin;
        self.show_bake = false;
        let fill = FillParams { mat: MatValues { color: [0.8, 0.6, 0.3], ..Default::default() }, ..Default::default() };
        let idx = self.engine.add_fill_layer(self.engine.layers.len(), None, "Sand".into(), fill, &mut self.undo);
        let mut g = GenKind::CustomMap.default_generator();
        g.custom[0] = crate::engine::CustomInput { id: self.bake_settings.custom[1].id, weight: 1.0 };
        g.weights[gin::OCCLUSION] = 0.4;
        self.engine.add_mask(idx, true, Some(g), &mut self.undo);
        self.selected = idx;
        self.target = StrokeTarget::Mask;
        self.docks.show_pane(Workspace::Paint, dock::Pane::Layers);
    }

    /// Opens or closes the bake window (offscreen UI tests).
    #[cfg(test)]
    pub(crate) fn test_set_bake_window(&mut self, open: bool) {
        self.show_bake = open;
    }

    /// Shows smart materials on the shelf, the bake window, and selects the
    /// top layer of a smart material that uses the edge distance (offscreen
    /// UI tests).
    #[cfg(test)]
    pub(crate) fn test_show_missing(&mut self) {
        self.shelf.cat = ShelfCat::SmartMaterials;
        self.show_bake = true;
        let k = presets::SMART_MATERIALS.iter().position(|s| s.needs_maps() & (1 << 5) != 0).expect("a smart material with edge wear");
        let at = self.engine.layers.len();
        let folder = add_smart_material(&mut self.engine, &mut self.undo, &presets::SMART_MATERIALS[k], at, None);
        // The layer in it whose mask reads the edges.
        let block = self.engine.block(folder);
        self.selected = block
            .clone()
            .find(|&i| self.engine.layers[i].mask.as_ref().and_then(|m| m.generator).is_some_and(|g| g.needs_maps() & (1 << 5) != 0))
            .unwrap_or(folder);
        self.target = StrokeTarget::Mask;
    }

    /// Opens a graph in the Material Designer (offscreen UI tests).
    #[cfg(test)]
    pub(crate) fn test_open_graph(&mut self, id: u64) {
        self.open_graph(id);
    }

    /// Path tool on a new paint layer with the Stitch tip: a path across the
    /// ball in the 3D view or a curve on the UV map (offscreen UI tests).
    #[cfg(test)]
    pub(crate) fn test_path_setup(&mut self, kind_3d: bool) {
        let at = self.engine.layers.len();
        self.selected = self.engine.add_paint_layer(at, None, Some("Stitches".into()), &mut self.undo);
        self.target = StrokeTarget::Content;
        self.set_tool(Tool::Path);
        self.brush.tip = crate::brushes::BRUSH_ALPHAS.iter().position(|b| b.name == "Stitch");
        self.brush.size = 34.0;
        self.brush.spacing = 1.25;
        self.brush.angle = 0.0;
        self.brush.mat.enabled = [true, true, false, true, false, false, false, false];
        self.brush.mat.color = [0.95, 0.85, 0.3];
        self.brush.mat.height = 1.0;
        self.path.clear();
        let (kind, size) = if kind_3d { (ViewKind::V3d, self.views.view3d.size) } else { (ViewKind::V2d, self.views.view2d.size) };
        let (w, h) = (size[0] as f32, size[1] as f32);
        let marks: &[(f32, f32)] = if kind_3d { &[(0.25, 0.42), (0.4, 0.34), (0.6, 0.34), (0.75, 0.42)] } else { &[(0.2, 0.7), (0.35, 0.55), (0.55, 0.62), (0.75, 0.5)] };
        self.path.view = Some(kind);
        for &(x, y) in marks {
            if let Some(p) = self.path_from_screen(kind, Vec2::new(x * w, y * h)) {
                self.path.points.push(p);
            }
        }
    }

    /// Applies the path (offscreen UI tests); returns the status line.
    #[cfg(test)]
    pub(crate) fn test_path_apply(&mut self) -> String {
        self.apply_path();
        self.status.clone()
    }

    /// Closes the library and adds the first downloaded material as a layer
    /// (offscreen UI tests).
    #[cfg(test)]
    pub(crate) fn test_add_downloaded(&mut self) {
        self.library.open = false;
        self.pending.push(Action::AddOnTop(ShelfItem::Library(0)));
    }

    /// Opens the export window (offscreen UI tests).
    #[cfg(test)]
    pub(crate) fn test_open_export(&mut self) {
        self.export.open = true;
    }

    /// Queues an action by name (offscreen UI tests).
    #[cfg(test)]
    pub(crate) fn test_add_preset(&mut self, i: usize) {
        self.pending.push(Action::AddGraphPreset(i));
    }

    pub fn new(cc: &eframe::CreationContext<'_>) -> Result<Self, String> {
        let rs = cc.wgpu_render_state.clone().ok_or("Material Painter requires the wgpu renderer")?;
        let gpu = rs.adapter.get_info();
        crate::crash::set_context(format!("{} ({:?}, driver {} {})", gpu.name, gpu.backend, gpu.driver, gpu.driver_info));
        theme::apply(&cc.egui_ctx);

        let prefs = Preferences::load();
        let mesh = Primitive::MaterialBall.build();
        let gpu_mesh = GpuMesh::new(&rs.device, &mesh);
        let bvh = Bvh::from_mesh(&mesh);
        let mut engine = Engine::new(&rs, &gpu_mesh, (mesh.bounds_min, mesh.bounds_max), prefs.default_resolution);
        engine.add_paint_layer(0, None, None, &mut UndoStack::default());
        let views = ViewRenderer::new(&rs);
        let env_baker = EnvBaker::new(&rs.device);
        // A real studio HDRI (plus the key light) when installed, else the
        // procedural studio.
        let default_env = environment::BUILTIN
            .iter()
            .position(|b| b.0 == prefs.default_hdri)
            .or_else(|| environment::BUILTIN.iter().position(|b| b.0 == "Small Studio"))
            .filter(|i| environment::available_builtins().contains(i))
            .unwrap_or(0);
        let env = env_baker.upload(&rs, &environment::decode(&EnvSource::Builtin(default_env))?);
        let mut raytracer = RayTracer::new(&rs);
        raytracer.set_mesh(&rs.device, &mesh, &bvh);
        let baker = GpuBaker::new(&rs.device);

        let ctx = &cc.egui_ctx;
        let load = |name: &str, img: egui::ColorImage| ctx.load_texture(name, img, egui::TextureOptions::LINEAR);
        let materials = presets::MATERIALS
            .iter()
            .map(|m| load(m.name, presets::render_preview(&[PreviewLayer { fill: m.fill(), opacity: 1.0, generator: None }])))
            .collect();
        let smart = presets::SMART_MATERIALS
            .iter()
            .map(|s| {
                let layers: Vec<PreviewLayer> = s
                    .layers
                    .iter()
                    .map(|l| PreviewLayer { fill: l.fill(), opacity: l.opacity, generator: l.generator() })
                    .collect();
                load(s.name, presets::render_preview(&layers))
            })
            .collect();
        let generators = GenKind::ALL.iter().map(|g| load(g.name(), presets::render_generator_preview(*g))).collect();
        // Brush tips show as white shapes; textures as their grayscale tile.
        const THUMB: u32 = 96;
        let brushes = crate::brushes::BRUSH_ALPHAS
            .iter()
            .map(|b| {
                let px: Vec<Color32> = b.render(THUMB).into_iter().map(|a| Color32::from_white_alpha(a)).collect();
                load(b.name, egui::ColorImage::new([THUMB as usize; 2], px))
            })
            .collect();
        let textures = crate::textures::TEXTURES
            .iter()
            .map(|t| {
                let px: Vec<Color32> = t.render(THUMB).into_iter().map(Color32::from_gray).collect();
                load(t.name, egui::ColorImage::new([THUMB as usize; 2], px))
            })
            .collect();

        let shelf_previews = shelf_previews::ShelfPreviews::start(&rs, ctx, &env_baker);

        let mut app = PainterApp {
            rs,
            mesh: Arc::new(mesh),
            bvh: Arc::new(bvh),
            gpu_mesh,
            engine,
            views,
            raytracer,
            rt: RtSettings::default(),
            undo: UndoStack::default(),
            selected: 0,
            target: StrokeTarget::Content,
            tool: Tool::Paint,
            brush: Brush::default(),
            other_brush: Brush::default(),
            color2: [0.95, 0.95, 0.95],
            color_sliders: ColorSliders::default(),
            id_color_pick_target: None,
            vertex_color_preview_active: false,
            preview_update_at: None,
            cam: OrbitCam::default(),
            cam2d: Cam2d::default(),
            layout: Layout::Split,
            view: ViewSettings::default(),
            env,
            env_baker,
            env_load: None,
            env_builtins: environment::available_builtins(),
            hdri_thumbs: Default::default(),
            hdri_thumb_rx: Some(start_hdri_thumbs(&cc.egui_ctx, environment::available_builtins())),
            hdri_new_thumb: None,
            stroke: None,
            bake_settings: BakeSettings::default(),
            bake_job: None,
            baker,
            show_bake: false,
            shelf: Shelf { cat: ShelfCat::SmartMaterials, search: String::new(), kind: None, icon_size: 72.0, smart, materials, generators, brushes, textures },
            shelf_previews: Some(shelf_previews),
            export: ExportDialog::new("MaterialBall".into()),
            prefs_applied: prefs.clone(),
            prefs,
            prefs_ui: PrefsUi { open: false, page: prefs::Page::General },
            graphs: GraphStore::new(&cc.egui_ctx),
            workspace: Workspace::Paint,
            docks: dock::Docks::load(),
            drew_3d: false,
            pending: Vec::new(),
            layer_clipboard: None,
            status: "Ready. Paint with LMB, orbit with Alt+LMB or RMB, pan with MMB.".into(),
            show_help: false,
            show_about: false,
            show_welcome: false,
            welcome_tex: None,
            library: Default::default(),
            user_smarts: smart_io::list(&cc.egui_ctx),
            path: Default::default(),
            bake_preview: Default::default(),
            custom_ui: Default::default(),
            spp: Default::default(),
            preview: Default::default(),
            crash_report: crate::crash::take_report(),
            frame_ms: 0.0,
            ppp: cc.egui_ctx.pixels_per_point(),
            mcp: None,
            mcp_bake_waiters: Vec::new(),
            mcp_env_waiters: Vec::new(),
            project_path: None,
            saved_serial: 0,
            confirm: None,
            allow_close: false,
            recent: project_io::load_recent(),
            autosaved_serial: 0,
            recovery: project_io::pending_recovery(),
            window_title: String::new(),
            mcp_views: None,
            overlays: Overlays::default(),
            cage_preview: CagePreview { show: true, probe: None, job: None, failed: None, drawn: None, stats: None },
            cage_brush: Default::default(),
            poly: PolyFillState {
                mode: FillMode::Polygon,
                box_drag: None,
                erase: false,
                topo: None,
                frame_hover: None,
                shown: None,
                filled: Default::default(),
                stroke_erase: false,
                angle: 30.0,
                lasso: false,
                through: false,
            },
            uploaded_tip: None,
            uploaded_grain: None,
            sets: Default::default(),
            pressure: 1.0,
            line_anchor: None,
        };
        app.apply_startup_preferences(ctx);
        match crate::mcp::McpServer::start(&cc.egui_ctx, app.prefs.mcp_port, app.prefs.mcp_on_start) {
            Ok(server) => app.mcp = Some(server),
            Err(e) => app.status = e,
        }
        if app.prefs.bake_on_load {
            app.start_bake(ctx);
        }
        Ok(app)
    }

    // ------------------------------------------------------------------ jobs

    /// Bakes the texture set being edited.
    fn start_bake(&mut self, ctx: &egui::Context) {
        self.sets.bake_queue.clear();
        self.start_bake_set(self.sets.active, ctx);
    }

    /// Bakes every texture set, the edited one first.
    fn start_bake_all(&mut self, ctx: &egui::Context) {
        let active = self.sets.active;
        self.sets.bake_queue = (0..self.mesh.sets.len()).filter(|&k| k != active).collect();
        self.start_bake_set(active, ctx);
    }

    fn start_bake_set(&mut self, set: usize, ctx: &egui::Context) {
        if set == self.sets.active {
            self.preview_full_bake_started();
        }
        let Some(res) = self.set_engine_mut(set).map(|e| e.res.min(bake::MAX_BAKE_RES)) else { return };
        self.sets.baking = set;
        let mesh = self.mesh.clone();
        let settings = self.bake_settings.clone();
        let (tx, rx) = mpsc::channel();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let key = bake_cache::key(&mesh, &settings, res, set);
            let prepared = match bake_cache::load(&key) {
                Some(entry) => Ok(Prepared::Cached(entry)),
                None => bake::prepare(&mesh, &settings, set).map(|scene| Prepared::Scene(scene, key)),
            };
            let _ = tx.send(prepared);
            ctx.request_repaint();
        });
        // A newer job replaces any running one; its results are dropped.
        self.bake_job = Some(BakeJob::Preparing(rx));
        self.status = if self.mesh.sets.len() > 1 {
            format!("Preparing bake of {}…", self.mesh.sets[set].name)
        } else {
            "Preparing bake…".into()
        };
    }

    /// A bake finished (or failed): the next queued set starts.
    fn bake_next(&mut self, ctx: &egui::Context) {
        self.bake_job = None;
        if let Some(set) = self.sets.bake_queue.pop_front() {
            self.start_bake_set(set, ctx);
        }
    }

    fn poll_bake(&mut self, ctx: &egui::Context) {
        match &mut self.bake_job {
            None => {}
            Some(BakeJob::Preparing(rx)) => match rx.try_recv() {
                Ok(Ok(Prepared::Cached(entry))) => {
                    let rs = &self.rs;
                    match out_of_memory_guard(&rs.device, || entry.upload(&rs.device, &rs.queue)) {
                        Ok((maps, report)) => {
                            self.status = format!("Loaded {} cached mesh maps ({}², from {}).", report.baked.count_ones(), report.resolution, report.source);
                            if let Some(e) = self.set_engine_mut(self.sets.baking) {
                                e.set_maps(maps, Arc::new(report));
                            }
                        }
                        Err(e) => {
                            self.fail_mcp_bake(&e);
                            self.status = e;
                        }
                    }
                    self.bake_next(ctx);
                }
                Ok(Ok(Prepared::Scene(scene, key))) => {
                    let res = self.set_engines().get(self.sets.baking).map_or(self.engine.res, |e| e.res);
                    let (baker, rs, mesh, settings) = (&self.baker, &self.rs, &self.gpu_mesh, &self.bake_settings);
                    match out_of_memory_guard(&rs.device, || baker.start(&rs.device, &rs.queue, mesh, &scene, settings, res)) {
                        Ok(run) => {
                            self.bake_job = Some(BakeJob::Running(run, key));
                            ctx.request_repaint();
                        }
                        Err(e) => {
                            self.fail_mcp_bake(&e);
                            self.status = e;
                            self.bake_job = None;
                        }
                    }
                }
                Ok(Err(e)) => {
                    self.fail_mcp_bake(&e);
                    self.status = e;
                    self.bake_job = None;
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.fail_mcp_bake("The bake preparation stopped unexpectedly.");
                    self.bake_job = None;
                }
            },
            Some(BakeJob::Running(run, key)) => {
                let (baker, rs) = (&self.baker, &self.rs);
                let pumped = match out_of_memory_guard(&rs.device, || baker.pump(run, &rs.device, &rs.queue, ctx)) {
                    Ok(maps) => maps,
                    Err(e) => {
                        self.fail_mcp_bake(&e);
                        self.status = e;
                        self.bake_job = None;
                        return;
                    }
                };
                if let Some(maps) = pumped {
                    let report = run.report.clone();
                    let matched = match report.matched_names {
                        0 => String::new(),
                        n => format!(", {n} parts matched by name"),
                    };
                    let set_res = texture_sets::engines(&self.engine, &self.sets, self.mesh.sets.len()).get(self.sets.baking).map_or(self.engine.res, |e| e.res);
                    let upscaled = if report.resolution < set_res { format!(", upscaled to {set_res}²") } else { String::new() };
                    let set_name = match self.mesh.sets.get(self.sets.baking) {
                        Some(s) if self.mesh.sets.len() > 1 => format!(" for {}", s.name),
                        _ => String::new(),
                    };
                    self.status = format!(
                        "Baked {} mesh maps{set_name} from {} ({} tris, {} parts{matched}{}) at {}²{upscaled}.",
                        report.baked.count_ones(),
                        report.source,
                        report.triangles,
                        report.parts,
                        if report.used_cage { ", cage" } else { "" },
                        report.resolution
                    );
                    bake_cache::store(key.clone(), &maps, report.clone(), &self.rs.device, &self.rs.queue);
                    if let Some(e) = self.set_engine_mut(self.sets.baking) {
                        e.set_maps(maps, report);
                    }
                    self.bake_next(ctx);
                }
            }
        }
    }

    fn bake_progress(&self) -> Option<(f32, &'static str)> {
        match &self.bake_job {
            None => None,
            Some(BakeJob::Preparing(_)) => Some((0.0, "Preparing")),
            Some(BakeJob::Running(run, _)) => Some((run.progress(), "Baking")),
        }
    }

    fn load_mesh(&mut self, mesh: Mesh, ctx: &egui::Context) {
        self.load_mesh_sized(mesh, ctx, &[]);
    }

    /// Loads `mesh`, making its new texture sets at `sizes` (per set, in
    /// order) instead of the default size.
    fn load_mesh_sized(&mut self, mesh: Mesh, ctx: &egui::Context, sizes: &[u32]) {
        if self.stroke.is_some() {
            return;
        }
        // A bake of the old mesh can't land on the new one.
        self.bake_job = None;
        self.end_preview(false);
        self.gpu_mesh = GpuMesh::new(&self.rs.device, &mesh);
        self.bvh = Arc::new(Bvh::from_mesh(&mesh));
        self.raytracer.set_mesh(&self.rs.device, &mesh, &self.bvh);
        self.export.mesh_name = mesh.name.replace(' ', "");
        self.status = match mesh.sets.len() {
            1 => format!("Loaded '{}' ({} triangles).", mesh.name, mesh.triangle_count()),
            n => format!("Loaded '{}' ({} triangles, {n} texture sets).", mesh.name, mesh.triangle_count()),
        };
        let old_names: Vec<String> = self.mesh.sets.iter().map(|s| s.name.clone()).collect();
        self.mesh = Arc::new(mesh);
        self.rebuild_sets(&old_names, sizes);
        // Mesh-dependent helpers rebuild for the new mesh on demand.
        self.poly.topo = None;
        self.poly.shown = None;
        self.overlays.clear(OverlaySlot::Selection);
        self.overlays.clear(OverlaySlot::VertexColors);
        self.vertex_color_preview_active = false;
        // A painted cage belongs to the mesh it was painted on.
        if self.bake_settings.cage_paint.as_deref().is_some_and(|p| !p.fits(self.mesh.vertices.len())) {
            self.bake_settings.cage_paint = None;
        }
        self.cam = OrbitCam { fov_y: self.cam.fov_y, ..OrbitCam::default() };
        if self.prefs.bake_on_load {
            self.start_bake_all(ctx);
        }
    }

    // ------------------------------------------------------------------ actions

    fn process_actions(&mut self, ctx: &egui::Context) {
        for action in std::mem::take(&mut self.pending) {
            if self.stroke.is_some() {
                continue;
            }
            let (at, parent) = self.insert_place();
            match action {
                a @ (Action::NewProject(_) | Action::OpenProject(_) | Action::OpenSpp(_) | Action::Exit) if self.is_modified() => {
                    self.confirm = Some(a);
                }
                Action::Confirmed(a) => self.run_unchecked(*a, ctx),
                a @ (Action::NewProject(_) | Action::OpenProject(_) | Action::OpenSpp(_) | Action::Exit) => self.run_unchecked(a, ctx),
                Action::SaveProject => {
                    self.save(false);
                }
                Action::SaveProjectAs => {
                    self.save(true);
                }
                Action::SaveThen(a) => {
                    if self.save(false) {
                        self.pending.push(Action::Confirmed(a));
                    }
                }
                Action::ChangeMesh(p) => self.load_mesh(p.build(), ctx),
                Action::ImportMesh => {
                    if let Some(path) = rfd::FileDialog::new().add_filter("Mesh (FBX, OBJ)", &crate::import::EXTENSIONS).pick_file() {
                        match Mesh::load_file(&path) {
                            Ok(mesh) => self.load_mesh(mesh, ctx),
                            Err(e) => self.status = e,
                        }
                    }
                }
                Action::Undo => {
                    self.switch_for_history(false);
                    self.engine.undo(&mut self.undo);
                    self.after_history_step();
                }
                Action::Redo => {
                    self.switch_for_history(true);
                    self.engine.redo(&mut self.undo);
                    self.after_history_step();
                }
                Action::AddPaint => {
                    self.selected = self.engine.add_paint_layer(at, parent, None, &mut self.undo);
                    self.target = StrokeTarget::Content;
                }
                Action::AddFill(fill, name) => {
                    self.selected = self.engine.add_fill_layer(at, parent, name, fill, &mut self.undo);
                    self.target = StrokeTarget::Content;
                }
                Action::AddFolder => {
                    let n = self.engine.layers.iter().filter(|l| l.is_folder()).count() + 1;
                    self.selected = self.engine.add_folder(at, parent, format!("Folder {n}"), &mut self.undo);
                    self.target = StrokeTarget::Content;
                }
                Action::GroupIntoFolder => {
                    if let Some(layer) = self.engine.layers.get(self.selected) {
                        let (parent, label) = (layer.parent, format!("Group {}", layer.name));
                        self.undo.begin_group(label);
                        let n = self.engine.layers.iter().filter(|l| l.is_folder()).count() + 1;
                        let folder = self.engine.add_folder(self.selected + 1, parent, format!("Folder {n}"), &mut self.undo);
                        let folder_id = self.engine.layers[folder].id;
                        self.engine.move_to(self.selected, Drop::Into(folder_id), &mut self.undo);
                        self.undo.end_group();
                        self.selected = self.engine.index_of(folder_id).unwrap_or(folder);
                        self.target = StrokeTarget::Content;
                    }
                }
                Action::AddMask { white } => {
                    if self.selected < self.engine.layers.len() {
                        self.engine.add_mask(self.selected, white, None, &mut self.undo);
                        self.target = StrokeTarget::Mask;
                    }
                }
                Action::AddGenerator(kind) => {
                    let g = new_generator(kind, &self.bake_settings.custom);
                    if let Some(layer) = self.engine.layers.get_mut(self.selected) {
                        match &mut layer.mask {
                            Some(m) => {
                                m.generator = Some(g);
                                self.engine.mark_dirty();
                            }
                            None => self.engine.add_mask(self.selected, true, Some(g), &mut self.undo),
                        }
                        self.target = StrokeTarget::Mask;
                    } else {
                        self.status = "Select a layer to add a generator mask to.".into();
                    }
                }
                Action::RemoveMask => {
                    self.engine.remove_mask(self.selected, &mut self.undo);
                    self.target = StrokeTarget::Content;
                }
                Action::SaveSmart => self.save_smart_material(ctx),
                Action::Duplicate => {
                    if let Some(i) = self.engine.duplicate_layer(self.selected, &mut self.undo) {
                        self.selected = i;
                    }
                }
                Action::CopyLayer => {
                    if self.selected < self.engine.layers.len() {
                        let name = self.engine.layers[self.selected].name.clone();
                        self.layer_clipboard = self.engine.copy_layer_block(self.selected);
                        self.status = format!("Copied '{name}'. Switch texture sets and press Ctrl+V to paste.");
                    } else {
                        self.layer_clipboard = None;
                        self.status = "Select a layer to copy.".into();
                    }
                }
                Action::PasteLayer => {
                    if let Some(layers) = self.layer_clipboard.take() {
                        if layers.is_empty() {
                            self.status = "The layer clipboard is empty.".into();
                        } else {
                            let ids = self.engine.insert_layers_resampled(
                                at, parent, layers, "Paste layers".into(), &mut self.undo,
                            );
                            if let Some(id) = ids.last().copied() {
                                self.selected = self.engine.index_of(id).unwrap_or(self.selected);
                                // Keep the clipboard reusable after a paste, at the destination resolution.
                                self.layer_clipboard = self.engine.copy_layer_block(self.selected);
                                self.target = StrokeTarget::Content;
                                self.status = "Pasted layer block.".into();
                            }
                        }
                    } else {
                        self.status = "Nothing to paste. Select a layer and press Ctrl+C first.".into();
                    }
                }
                Action::Delete => {
                    self.engine.delete_layer(self.selected, &mut self.undo);
                    self.selected = self.selected.min(self.engine.layers.len().saturating_sub(1));
                    self.target = StrokeTarget::Content;
                }
                Action::Clear => self.engine.clear_layer(self.selected, &mut self.undo),
                // Shelf content added by clicking or dropping on a view goes
                // on top of the stack, outside any folder.
                Action::AddOnTop(item) => {
                    self.pending.push(Action::ShelfDrop { item, at: self.engine.layers.len(), parent: None });
                }
                Action::ShelfDrop { item, at, parent } => match item {
                    ShelfItem::Smart(i) => {
                        let smart = &presets::SMART_MATERIALS[i];
                        self.selected = add_smart_material(&mut self.engine, &mut self.undo, smart, at, parent);
                        self.target = StrokeTarget::Content;
                        self.status = format!("Added smart material '{}'. Edit its layers, or the folder's graph (Ctrl+G).", smart.name);
                    }
                    ShelfItem::SmartGraph(i) => {
                        let smart = &presets::SMART_MATERIALS[i];
                        self.add_smart_graph(smart, at, parent);
                        self.status = format!("Added smart material '{}' as one graph layer. Tune it in the layer properties.", smart.name);
                    }
                    ShelfItem::Material(i) => {
                        let m = &presets::MATERIALS[i];
                        self.selected = self.engine.add_fill_layer(at, parent, m.name.to_string(), m.fill(), &mut self.undo);
                        self.target = StrokeTarget::Content;
                    }
                    ShelfItem::Generator(g) => self.pending.push(Action::AddGenerator(g)),
                    ShelfItem::Brush(i) => self.pending.push(Action::SetBrushTip(Some(i))),
                    ShelfItem::Texture(i) => self.add_texture_layer(i, at, parent),
                    ShelfItem::Graph(i) => {
                        if let Some(path) = self.graphs.shelf.get(i).map(|g| g.path.clone()) {
                            self.add_shelf_graph(&path, at, parent);
                        }
                    }
                    ShelfItem::Library(i) => self.add_library_material(i, at, parent),
                    ShelfItem::UserSmart(i) => {
                        if let Some(path) = self.user_smarts.get(i).map(|s| s.path.clone()) {
                            self.apply_smart_material(&path, at, parent);
                        }
                    }
                },
                Action::AddGraphLayer => self.add_graph_layer(None, at, parent),
                Action::AddGraphPreset(i) => {
                    if let Some(p) = crate::graph::presets::PRESETS.get(i) {
                        self.add_graph_layer(Some((p.build)()), at, parent);
                    }
                }
                Action::AddGraphMask => self.add_graph_mask(),
                Action::EditGraph => self.edit_selected_graph(),
                Action::GraphUndo => self.graph_undo(false),
                Action::GraphRedo => self.graph_undo(true),
                Action::GraphEditCode(id) => self.open_code(&id),
                Action::GraphCopyNode(id) => self.new_custom_node(Some(&id), true),
                Action::GraphRefresh(id) => self.refresh_graph(id),
                Action::GraphAddNode(def) => self.add_node_to_open_graph(&def),
                Action::SetBrushTip(tip) => {
                    self.brush.tip = tip;
                    if !matches!(self.tool, Tool::Paint | Tool::Eraser) {
                        self.set_tool(Tool::Paint);
                    }
                    let name = tip.map_or("Round", |i| crate::brushes::BRUSH_ALPHAS[i].name);
                    self.status = format!("Brush tip: {name}.");
                }
                Action::TextureToChannel { texture, ch } => {
                    let is_fill = self.engine.layers.get(self.selected).is_some_and(|l| l.is_fill());
                    if !is_fill {
                        self.status = "Select a fill layer to put the texture on.".into();
                    } else {
                        match self.texture_image(texture) {
                            Ok(img) => {
                                let idx = self.selected;
                                if let Content::Fill(f) = &mut self.engine.layers[idx].content {
                                    f.mat.enabled[ch] = true;
                                }
                                self.engine.set_fill_image(idx, ch, Some(img), &mut self.undo);
                            }
                            Err(e) => self.status = e,
                        }
                    }
                }
                Action::SetFillImage { idx, ch, path } => match path {
                    None => self.engine.set_fill_image(idx, ch, None, &mut self.undo),
                    Some(path) => {
                        let name = path.file_name().map_or("image".into(), |n| n.to_string_lossy().into_owned());
                        let loaded = std::fs::read(&path)
                            .map_err(|e| format!("can't read {}: {e}", path.display()))
                            .and_then(|bytes| self.engine.decode_fill_image(name, Arc::new(bytes)));
                        match loaded {
                            Ok(img) => self.engine.set_fill_image(idx, ch, Some(img), &mut self.undo),
                            Err(e) => self.status = e,
                        }
                    }
                },
                Action::MoveTo { from, drop } => {
                    if let Some(i) = self.engine.move_to(from, drop, &mut self.undo) {
                        self.selected = i;
                    }
                }
                Action::SetResolution(r) => {
                    self.engine.set_resolution(r, &self.gpu_mesh, &mut self.undo);
                    self.repush_graphs();
                    if self.engine.has_bake {
                        // Mesh maps are baked at the texture set size; refresh them.
                        self.start_bake(ctx);
                    }
                    self.status = format!("Texture set resolution: {r}x{r}.");
                }
                Action::Bake => self.start_bake(ctx),
                Action::BakeAll => self.start_bake_all(ctx),
            }
        }
    }

    /// Saves to the project file, asking for one if there is none (or `as_new`).
    /// Returns whether it saved.
    fn save(&mut self, as_new: bool) -> bool {
        let path = match (&self.project_path, as_new) {
            (Some(p), false) => Some(p.clone()),
            _ => self.save_dialog(),
        };
        let Some(path) = path else { return false };
        match self.save_project(&path) {
            Ok(()) => true,
            Err(e) => {
                self.status = e;
                false
            }
        }
    }

    /// New, open and exit, once unsaved changes were saved or discarded.
    fn run_unchecked(&mut self, action: Action, ctx: &egui::Context) {
        match action {
            Action::NewProject(p) => {
                // The first set is the one cleared below and kept for the new mesh.
                self.switch_set(0);
                while !self.engine.layers.is_empty() {
                    self.engine.delete_layer(0, &mut self.undo);
                }
                self.graphs.clear(&self.rs);
                if self.engine.res != self.prefs.default_resolution {
                    self.engine.set_resolution(self.prefs.default_resolution, &self.gpu_mesh, &mut self.undo);
                }
                self.selected = self.engine.add_paint_layer(0, None, None, &mut self.undo);
                self.undo.clear();
                self.saved_serial = self.undo.state_serial();
                self.project_path = None;
                self.target = StrokeTarget::Content;
                self.apply_project_defaults();
                self.load_mesh(p.build(), ctx);
            }
            Action::OpenSpp(path) => self.start_spp_open(path, ctx),
            Action::OpenProject(path) => {
                if let Some(path) = path.or_else(Self::open_dialog) {
                    if let Err(e) = self.open_project(&path, ctx) {
                        self.status = e;
                    }
                    self.target = StrokeTarget::Content;
                }
            }
            Action::Exit => {
                self.allow_close = true;
                project_io::remove_autosave();
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
            other => self.pending.push(other),
        }
    }

    /// Offers to restore work left in the autosave by a crash.
    fn recovery_dialog(&mut self, ctx: &egui::Context) {
        let Some(path) = &self.recovery else { return };
        let when = std::fs::metadata(path).and_then(|m| m.modified()).ok().and_then(|t| t.elapsed().ok());
        let ago = when.map_or(String::new(), |d| match d.as_secs() {
            s if s < 120 => " a moment ago".into(),
            s if s < 7200 => format!(" {} minutes ago", s / 60),
            s => format!(" {} hours ago", s / 3600),
        });
        let mut choice = None;
        egui::Modal::new(egui::Id::unique("recover autosave")).show(ctx, |ui| {
            ui.set_width(360.0);
            ui.heading("Recover unsaved work?");
            ui.label(format!("Material Painter didn't close normally. An autosave was made{ago}."));
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("Recover").clicked() {
                    choice = Some(true);
                }
                if ui.button("Discard").clicked() {
                    choice = Some(false);
                }
            });
        });
        if let Some(recover) = choice {
            let path = self.recovery.take().expect("checked above");
            if recover {
                if let Err(e) = self.open_project(&path, ctx) {
                    self.status = e;
                }
            } else {
                project_io::remove_recovery(&path);
            }
        }
    }

    /// "Save changes?" before an action that would drop them.
    fn confirm_dialog(&mut self, ctx: &egui::Context) {
        let Some(action) = &self.confirm else { return };
        let what = match action {
            Action::Exit => "closing",
            Action::OpenProject(_) => "opening another project",
            Action::OpenSpp(_) => "opening a Substance 3D Painter project",
            _ => "starting a new project",
        };
        let name = self.project_path.as_ref().and_then(|p| p.file_name()).map_or("this project".to_string(), |n| n.to_string_lossy().into_owned());
        let mut choice = None;
        egui::Modal::new(egui::Id::unique("unsaved changes")).show(ctx, |ui| {
            ui.set_width(340.0);
            ui.heading("Save changes?");
            ui.label(format!("{name} has unsaved changes. Save them before {what}?"));
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("Save").clicked() {
                    choice = Some(0);
                }
                if ui.button("Don't save").clicked() {
                    choice = Some(1);
                }
                if ui.button("Cancel").clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    choice = Some(2);
                }
            });
        });
        if let Some(c) = choice {
            let action = self.confirm.take().expect("checked above");
            match c {
                0 => self.pending.push(Action::SaveThen(Box::new(action))),
                1 => self.pending.push(Action::Confirmed(Box::new(action))),
                _ => {}
            }
        }
    }

    /// Where a new layer goes: above the selected layer, next to it, or at
    /// the top of the selected folder.
    /// Where a new layer goes: into a folder only when that folder is
    /// selected; otherwise at the top level, above the selected layer (or
    /// above the folder holding it), or on top when nothing is selected.
    fn insert_place(&self) -> (usize, Option<u64>) {
        match self.engine.layers.get(self.selected) {
            None => (self.engine.layers.len(), None),
            Some(l) if l.is_folder() => (self.selected, Some(l.id)),
            Some(_) => {
                let top = self.engine.ancestors(self.selected).max().unwrap_or(self.selected);
                (top + 1, None)
            }
        }
    }

    /// Keeps the selection valid after undo/redo changed the layer stack.
    fn after_history_step(&mut self) {
        if self.selected != NO_LAYER {
            self.selected = self.selected.min(self.engine.layers.len().saturating_sub(1));
        }
        let has_mask = self.engine.layers.get(self.selected).is_some_and(|l| l.mask.is_some());
        if self.target == StrokeTarget::Mask && !has_mask {
            self.target = StrokeTarget::Content;
        }
    }

    fn shortcuts(&mut self, ctx: &egui::Context) {
        use egui::{Key, KeyboardShortcut, Modifiers};
        ctx.input_mut(|i| {
            if i.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND | Modifiers::SHIFT, Key::E)) {
                self.export.open = true;
            }
            if i.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::Comma)) {
                self.prefs_ui.open = !self.prefs_ui.open;
            }
            if i.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND | Modifiers::SHIFT, Key::S)) {
                self.pending.push(Action::SaveProjectAs);
            }
            if i.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::S)) {
                self.pending.push(Action::SaveProject);
            }
            if i.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::O)) {
                self.pending.push(Action::OpenProject(None));
            }
            // These shortcuts belong to the layer stack, not text fields; the
            // keyboard-focus guard below keeps normal text copy/paste intact.
            // Over the graph editor, undo and redo act on the graph.
            let graph = self.graphs.editor.hovered;
            if i.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND | Modifiers::SHIFT, Key::Z))
                || i.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::Y))
            {
                self.pending.push(if graph { Action::GraphRedo } else { Action::Redo });
            }
            if i.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::Z)) {
                self.pending.push(if graph { Action::GraphUndo } else { Action::Undo });
            }
            if i.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::G)) {
                if self.workspace == Workspace::Designer {
                    self.workspace = Workspace::Paint;
                } else {
                    self.pending.push(Action::EditGraph);
                }
            }
            // Ctrl+PageUp / PageDown step through the workspaces.
            let step = if i.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::PageDown)) {
                1
            } else if i.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::PageUp)) {
                Workspace::ALL.len() - 1
            } else {
                0
            };
            if step > 0 {
                let at = Workspace::ALL.iter().position(|w| *w == self.workspace).unwrap_or(0);
                self.workspace = Workspace::ALL[(at + step) % Workspace::ALL.len()];
            }
        });
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        // Layer shortcuts only in the paint workspace, and not over the graph
        // editor: there Delete removes nodes and Ctrl+C / Ctrl+V copy them, and
        // consuming the key here would hide it from the editor.
        let layer_keys = self.workspace == Workspace::Paint && !self.graphs.editor.hovered;
        ctx.input_mut(|i| {
            if layer_keys && i.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::C)) {
                self.pending.push(Action::CopyLayer);
            }
            if layer_keys && i.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::V)) {
                self.pending.push(Action::PasteLayer);
            }
            if layer_keys && i.consume_shortcut(&KeyboardShortcut::new(Modifiers::default(), Key::Delete)) {
                if self.selected < self.engine.layers.len() {
                    self.pending.push(Action::Delete);
                }
            }
            if i.modifiers.any() {
                return;
            }
            if i.key_pressed(Key::Num1) {
                self.set_tool(Tool::Paint);
            }
            if i.key_pressed(Key::Num2) {
                self.set_tool(Tool::Eraser);
            }
            if i.key_pressed(Key::OpenBracket) {
                self.brush.size = (self.brush.size / 1.15).max(2.0);
            }
            if i.key_pressed(Key::CloseBracket) {
                self.brush.size = (self.brush.size * 1.15).min(800.0);
            }
            if i.key_pressed(Key::I) {
                self.set_tool(Tool::Picker);
            }
            if i.key_pressed(Key::Num3) || i.key_pressed(Key::P) {
                self.set_tool(Tool::PolyFill);
            }
            if i.key_pressed(Key::Num4) || i.key_pressed(Key::L) {
                self.set_tool(Tool::Path);
            }
            if i.key_pressed(Key::S) {
                self.swap_colors();
            }
            if i.key_pressed(Key::W) {
                self.view.show_wire_3d = !self.view.show_wire_3d;
            }
            if i.key_pressed(Key::F) {
                self.cam = OrbitCam { yaw: self.cam.yaw, pitch: self.cam.pitch, fov_y: self.cam.fov_y, ..OrbitCam::default() };
                self.cam2d.zoom = 0.0;
            }
            if i.key_pressed(Key::M) {
                self.view.mode = ViewMode::Material;
            }
            if i.key_pressed(Key::C) {
                let all = ViewMode::all();
                let pos = all.iter().position(|m| *m == self.view.mode).unwrap_or(0);
                self.view.mode = all[(pos + 1) % all.len()];
            }
            if i.key_pressed(Key::F1) {
                self.layout = Layout::Split;
            }
            if i.key_pressed(Key::F2) {
                self.layout = Layout::View3d;
            }
            if i.key_pressed(Key::F10) {
                self.rt.enabled = !self.rt.enabled;
            }
            if i.key_pressed(Key::F3) {
                self.layout = Layout::View2d;
            }
            if i.key_pressed(Key::F5) {
                self.preview_update_at = None;
                self.engine.mark_dirty();
                self.status = "Preview recomposition forced (F5).".into();
            }
        });
        if ctx.input(|i| i.key_pressed(Key::F5) && !i.modifiers.any()) {
            ctx.request_repaint();
        }
    }

    // ------------------------------------------------------------------ panels

    fn menu_bar(&mut self, ui: &mut egui::Ui) {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.add(egui::Button::new(theme::with_icon(icons::FOLDER_OPEN, "Open project…")).shortcut_text("Ctrl+O")).clicked() {
                    self.pending.push(Action::OpenProject(None));
                    ui.close();
                }
                let recent = self.recent.clone();
                ui.add_enabled_ui(!recent.is_empty(), |ui| {
                    ui.menu_button("Open recent", |ui| {
                        for p in recent {
                            let name = p.file_name().map_or_else(|| p.display().to_string(), |n| n.to_string_lossy().into_owned());
                            if ui.button(name).on_hover_text(p.display().to_string()).clicked() {
                                self.pending.push(Action::OpenProject(Some(p)));
                                ui.close();
                            }
                        }
                    });
                });
                if ui.add(egui::Button::new(theme::with_icon(icons::FLOPPY_DISK, "Save project")).shortcut_text("Ctrl+S")).clicked() {
                    self.pending.push(Action::SaveProject);
                    ui.close();
                }
                if ui.add(egui::Button::new("Save project as…").shortcut_text("Ctrl+Shift+S")).clicked() {
                    self.pending.push(Action::SaveProjectAs);
                    ui.close();
                }
                ui.separator();
                ui.menu_button("New project", |ui| {
                    for p in Primitive::ALL {
                        if ui.button(p.name()).clicked() {
                            self.pending.push(Action::NewProject(p));
                            ui.close();
                        }
                    }
                });
                ui.menu_button("Swap preview mesh (keep layers)", |ui| {
                    for p in Primitive::ALL {
                        if ui.button(p.name()).clicked() {
                            self.pending.push(Action::ChangeMesh(p));
                            ui.close();
                        }
                    }
                });
                if ui.button(theme::with_icon(icons::CUBE, "Import mesh (FBX, OBJ)…")).clicked() {
                    self.pending.push(Action::ImportMesh);
                    ui.close();
                }
                let spp = ui
                    .add_enabled(!self.spp_busy(), egui::Button::new(theme::with_icon(icons::FOLDER_OPEN, "Open Substance 3D Painter project (.spp)…")))
                    .on_hover_text("Brings in the model, the final look of its color, roughness, metallic and height, and as much of the layer stack as maps onto Material Painter's. A report lists what came across and what didn't.");
                if spp.clicked() {
                    self.pending.push(Action::OpenSpp(None));
                    ui.close();
                }
                ui.separator();
                if ui.add(egui::Button::new(theme::with_icon(icons::EXPORT, "Export textures…")).shortcut_text("Ctrl+Shift+E")).clicked() {
                    self.export.open = true;
                    ui.close();
                }
                ui.separator();
                if let Some(server) = &self.mcp {
                    let mut on = server.enabled.load(std::sync::atomic::Ordering::Relaxed);
                    if ui.checkbox(&mut on, "MCP server (AI agents)").on_hover_text(server.url()).changed() {
                        server.enabled.store(on, std::sync::atomic::Ordering::Relaxed);
                    }
                    ui.separator();
                }
                if ui.button("Exit").clicked() {
                    self.pending.push(Action::Exit);
                    ui.close();
                }
            });
            ui.menu_button("Edit", |ui| {
                let undo_label = self.undo.undo.last().map_or("Undo".to_string(), |e| format!("Undo {}", e.label));
                let redo_label = self.undo.redo.last().map_or("Redo".to_string(), |e| format!("Redo {}", e.label));
                let can_undo = !self.undo.undo.is_empty();
                let can_redo = !self.undo.redo.is_empty();
                if ui.add_enabled(can_undo, egui::Button::new(theme::with_icon(icons::UNDO, &undo_label)).shortcut_text("Ctrl+Z")).clicked() {
                    self.pending.push(Action::Undo);
                    ui.close();
                }
                if ui.add_enabled(can_redo, egui::Button::new(theme::with_icon(icons::REDO, &redo_label)).shortcut_text("Ctrl+Y")).clicked() {
                    self.pending.push(Action::Redo);
                    ui.close();
                }
                ui.separator();
                if ui.add(egui::Button::new(theme::with_icon(icons::GEAR, "Preferences…")).shortcut_text("Ctrl+,")).clicked() {
                    self.prefs_ui.open = true;
                    ui.close();
                }
            });
            ui.menu_button("Layer", |ui| {
                self.layer_menu_items(ui);
            });
            ui.menu_button("View", |ui| {
                if ui.button("Reset panel layout").on_hover_text("Puts this workspace's panels back where they started").clicked() {
                    self.docks.reset(self.workspace);
                    ui.close();
                }
                ui.separator();
                for (l, name, key) in [(Layout::Split, "3D / 2D", "F1"), (Layout::View3d, "3D only", "F2"), (Layout::View2d, "2D only", "F3")] {
                    if ui.add(theme::selectable(self.layout == l, name).shortcut_text(key)).clicked() {
                        self.layout = l;
                        ui.close();
                    }
                }
                ui.separator();
                for m in ViewMode::all() {
                    if ui.add(theme::selectable(self.view.mode == m, m.name())).clicked() {
                        self.view.mode = m;
                        ui.close();
                    }
                }
                ui.separator();
                ui.checkbox(&mut self.view.show_wire_3d, "Wireframe (3D)  W");
                ui.checkbox(&mut self.view.show_wire, "UV wireframe (2D)");
                ui.checkbox(&mut self.view.show_seams_2d, "UV seams (2D)");
                ui.checkbox(&mut self.view.show_seams_3d, "UV seams (3D)");
                ui.horizontal(|ui| {
                    ui.label("UV wire opacity");
                    ui.add(egui::Slider::new(&mut self.view.uv_wire_opacity, 0.05..=1.0).max_decimals(2));
                });
                ui.add(egui::Checkbox::new(&mut self.rt.enabled, "Ray traced preview (F10)"));
                if ui.add(egui::Button::new("Frame mesh").shortcut_text("F")).clicked() {
                    self.cam = OrbitCam { fov_y: self.cam.fov_y, ..OrbitCam::default() };
                    self.cam2d.zoom = 0.0;
                    ui.close();
                }
            });
            ui.menu_button("Help", |ui| {
                if ui.button(theme::with_icon(icons::INFO, "Controls")).clicked() {
                    self.show_help = true;
                    ui.close();
                }
                if ui.button("Source code and releases").clicked() {
                    ui.ctx().open_url(egui::OpenUrl::new_tab(REPO_URL));
                    ui.close();
                }
                if ui.button(egui::RichText::new(theme::with_icon(icons::HEART, "Support development on Ko-fi")).color(theme::DONATE)).clicked() {
                    ui.ctx().open_url(egui::OpenUrl::new_tab(welcome::DONATE_URL));
                    ui.close();
                }
                if ui.button("Open crash reports folder").clicked() {
                    if let Some(dir) = project_io::data_dir() {
                        open_in_file_browser(&dir);
                    }
                    ui.close();
                }
                ui.separator();
                if ui.button("Welcome screen").clicked() {
                    self.show_welcome = true;
                    ui.close();
                }
                if ui.button("About Material Painter").clicked() {
                    self.show_about = true;
                    ui.close();
                }
            });
            // Workspace tabs.
            ui.add_space(24.0);
            ui.separator();
            for ws in Workspace::ALL {
                let resp = theme::pill(ui, &theme::with_icon(ws.icon(), ws.name()), self.workspace == ws);
                let keys = match ws {
                    Workspace::Designer => "Ctrl+G opens the selected layer's graph; Ctrl+PageUp / PageDown switch workspaces",
                    _ => "Ctrl+PageUp / PageDown switch workspaces",
                };
                if resp.on_hover_text(keys).clicked() {
                    self.workspace = ws;
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let resp = theme::donate_button(ui, "Support", vec2(0.0, 22.0));
                if resp.on_hover_text("Material Painter is free; donations on Ko-fi keep it going").clicked() {
                    ui.ctx().open_url(egui::OpenUrl::new_tab(welcome::DONATE_URL));
                }
            });
        });
    }

    fn layer_menu_items(&mut self, ui: &mut egui::Ui) {
        let has = self.selected < self.engine.layers.len();
        let is_paint = has && self.engine.layers[self.selected].is_paint();
        let has_mask = has && self.engine.layers[self.selected].mask.is_some();
        let mut item = |ui: &mut egui::Ui, enabled: bool, text: &str, action: fn() -> Action| {
            if ui.add_enabled(enabled, egui::Button::new(text)).clicked() {
                self.pending.push(action());
                ui.close();
            }
        };
        item(ui, true, "Add paint layer", || Action::AddPaint);
        item(ui, true, "Add fill layer", || Action::AddFill(FillParams::default(), "Fill layer".into()));
        item(ui, true, "Add graph fill layer", || Action::AddGraphLayer);
        item(ui, true, "Add folder", || Action::AddFolder);
        item(ui, has, "Group into folder", || Action::GroupIntoFolder);
        ui.separator();
        item(ui, has && !has_mask, "Add black mask", || Action::AddMask { white: false });
        item(ui, has && !has_mask, "Add white mask", || Action::AddMask { white: true });
        item(ui, has && !has_mask, "Add graph mask", || Action::AddGraphMask);
        item(ui, true, "Open graph in Material Designer (Ctrl+G)", || Action::EditGraph);
        item(ui, has_mask, "Remove mask", || Action::RemoveMask);
        ui.separator();
        item(ui, has, "Save as smart material", || Action::SaveSmart);
        item(ui, has, "Duplicate layer", || Action::Duplicate);
        item(ui, has, "Copy layer (Ctrl+C)", || Action::CopyLayer);
        item(ui, self.layer_clipboard.as_ref().is_some_and(|layers| !layers.is_empty()), "Paste layer (Ctrl+V)", || Action::PasteLayer);
        item(ui, is_paint, "Clear layer", || Action::Clear);
        item(ui, has, "Delete layer", || Action::Delete);
    }

    fn set_tool(&mut self, tool: Tool) {
        if (self.tool == Tool::Eraser) != (tool == Tool::Eraser) { swap_brushes(&mut self.brush,&mut self.other_brush); }
        self.tool=tool;
    }
    /// Debounce expensive material preview recomposites without blocking UI controls.
    fn defer_preview_update(&mut self) {
        self.preview_update_at = Some(Instant::now() + MATERIAL_PREVIEW_DELAY);
    }
    fn schedule_preview_update(&mut self, ctx: &egui::Context) {
        self.defer_preview_update();
        ctx.request_repaint_after(MATERIAL_PREVIEW_DELAY);
    }
    /// Service the material preview timer once per app frame, even when the
    /// paint viewport is hidden. A held pointer postpones the rebuild until release.
    fn service_preview_debounce(&mut self, ctx: &egui::Context) {
        let Some(deadline) = self.preview_update_at else { return; };
        // Keep extending the debounce while a pointer interaction is active.
        // This guarantees the full delay starts after the slider is released,
        // even when someone holds the knob without moving it.
        if ctx.input(|i| i.pointer.any_down()) {
            self.defer_preview_update();
            ctx.request_repaint_after(MATERIAL_PREVIEW_DELAY);
            return;
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            self.preview_update_at = None;
            self.engine.mark_dirty();
            ctx.request_repaint();
        } else {
            ctx.request_repaint_after(remaining);
        }
    }
    fn active_color(&self) -> [f32; 3] {
        if self.target == StrokeTarget::Content {
            if let Some(layer) = self.engine.layers.get(self.selected) {
                if let Content::Fill(fill) = &layer.content { return fill.mat.color; }
            }
        }
        self.brush.mat.color
    }
    fn set_active_color(&mut self, color: [f32; 3]) {
        if self.target == StrokeTarget::Content {
            if let Some(layer) = self.engine.layers.get_mut(self.selected) {
                if let Content::Fill(fill) = &mut layer.content {
                    fill.mat.color = color;
                    self.defer_preview_update();
                    return;
                }
            }
        }
        self.brush.mat.color = color;
    }
    fn swap_colors(&mut self) { let current=self.active_color(); self.set_active_color(self.color2); self.color2=current; }
    fn color_slots(&mut self,ui:&mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x=3.0;
            let mut active=self.active_color();
            if color_button(ui,&mut active,Some("Active layer / brush color")){self.set_active_color(active);}
            if theme::icon_button(ui,icons::EYEDROPPER,false,20.0).on_hover_text("Pick material color from the mesh (I)").clicked(){self.set_tool(Tool::Picker);}
            if swap_button(ui).on_hover_text("Swap the active color and the second color (S)").clicked(){self.swap_colors();}
            color_button(ui,&mut self.color2,Some("Second color"));
        });
    }
    fn color_hud(&mut self,ui:&mut egui::Ui) {
        let fill_target=self.target==StrokeTarget::Content && self.engine.layers.get(self.selected).is_some_and(|layer|matches!(&layer.content,Content::Fill(_)));
        let initial_active_color=self.active_color();
        ui.set_width(420.0);
        egui::ScrollArea::vertical().max_height(620.0).show(ui, |ui| {
        self.color_slots(ui); ui.add_space(4.0);
        let mut state=std::mem::take(&mut self.color_sliders); let mut color=self.active_color();
        if state.out!=color { state.hsv=hsv_keeping(state.hsv,color); state.temp=0.0; state.shifting=false; }
        ui.horizontal(|ui| {
            ui.selectable_value(&mut state.hsv_mode,false,"RGB");
            ui.selectable_value(&mut state.hsv_mode,true,"HSV");
            ui.separator();
        });
        let picked=color_picker::picker(ui,&mut state.hsv);
        let mut numeric_changed=false;
        if state.hsv_mode {
            let (mut h,mut s,mut v)=(state.hsv[0]*360.0,state.hsv[1]*100.0,state.hsv[2]*100.0);
            numeric_changed |= labeled(ui,"H",|ui|ui.add(egui::Slider::new(&mut h,0.0..=360.0).max_decimals(2)).changed());
            numeric_changed |= labeled(ui,"S",|ui|ui.add(egui::Slider::new(&mut s,0.0..=100.0).max_decimals(2)).changed());
            numeric_changed |= labeled(ui,"V",|ui|ui.add(egui::Slider::new(&mut v,0.0..=100.0).max_decimals(2)).changed());
            if numeric_changed { state.hsv=[h/360.0,s/100.0,v/100.0]; }
        } else {
            let mut rgb=color.map(|v|v*255.0);
            numeric_changed |= labeled(ui,"R",|ui|ui.add(egui::Slider::new(&mut rgb[0],0.0..=255.0).max_decimals(0)).changed());
            numeric_changed |= labeled(ui,"G",|ui|ui.add(egui::Slider::new(&mut rgb[1],0.0..=255.0).max_decimals(0)).changed());
            numeric_changed |= labeled(ui,"B",|ui|ui.add(egui::Slider::new(&mut rgb[2],0.0..=255.0).max_decimals(0)).changed());
            if numeric_changed { color=rgb.map(|v|v/255.0); state.hsv=hsv_keeping(state.hsv,color); }
        }
        if picked||numeric_changed { color=hsv_to_rgb(state.hsv); }
        if !state.hex_dirty {
            state.hex_edit = color_picker::hex_string(color);
        }
        ui.horizontal(|ui| {
            if color_picker::compare_swatch(ui,state.opened,color) { color=state.opened; state.hsv=hsv_keeping(state.hsv,color); }
            ui.label(egui::RichText::new("HEX").small().color(theme::TEXT_DIM));
            let response = ui.add(egui::TextEdit::singleline(&mut state.hex_edit).desired_width(94.0).font(egui::TextStyle::Monospace));
            if response.changed() {
                state.hex_dirty = true;
                if let Some(parsed) = parse_color_hex(&state.hex_edit) {
                    color = parsed;
                    state.hsv = hsv_keeping(state.hsv, color);
                }
            }
            if response.lost_focus() {
                state.hex_dirty = false;
                state.hex_edit = color_picker::hex_string(color);
            }
            if ui.button("Copy").on_hover_text("Copy the hex code").clicked() { ui.ctx().copy_text(color_picker::hex_string(color)); }
        });
        let temp=labeled(ui,"Temperature",|ui|ui.add(egui::Slider::new(&mut state.temp,-100.0..=100.0).max_decimals(0)));
        if temp.changed() {
            if !state.shifting { state.base=color; state.shifting=true; }
            color=shift_temperature(state.base,state.temp/100.0);
            state.hsv=hsv_keeping(state.hsv,color);
        }
        if temp.drag_stopped()||temp.lost_focus() { state.temp=0.0; state.shifting=false; }
        ui.add_space(2.0);
        let second=self.color2;
        let from_mix=color_picker::mix_strip(ui,color,second);
        let from_harmony=color_picker::harmony_row(ui,color);
        if let Some((picked,as_second))=from_mix.or(from_harmony) {
            if as_second { self.color2=picked; } else { color=picked; state.hsv=hsv_keeping(state.hsv,color); }
        }
        if color != self.active_color() { self.set_active_color(color); }
        state.out=color; self.color_sliders=state;
        ui.separator();
        let mut material_changed=false;
        if fill_target {
            if let Some(layer)=self.engine.layers.get_mut(self.selected) {
                if let Content::Fill(fill)=&mut layer.content {
                    let material=&mut fill.mat;
                    material_changed |= slider(ui,"Opacity",&mut material.opacity,0.0..=1.0);
                    material_changed |= channel_toggles(ui,&mut material.enabled);
                    if material.enabled[3] { material_changed |= slider(ui,"Height",&mut material.height,-1.0..=1.0); }
                    if material.enabled[1] { material_changed |= slider(ui,"Roughness",&mut material.rough,0.0..=1.0); }
                    if material.enabled[2] { material_changed |= slider(ui,"Metallic",&mut material.metal,0.0..=1.0); }
                }
            }
        } else {
            let material=&mut self.brush.mat;
            slider(ui,"Opacity",&mut material.opacity,0.0..=1.0);
            channel_toggles(ui,&mut material.enabled);
            if material.enabled[3] { slider(ui,"Height",&mut material.height,-1.0..=1.0); }
            if material.enabled[1] { slider(ui,"Roughness",&mut material.rough,0.0..=1.0); }
            if material.enabled[2] { slider(ui,"Metallic",&mut material.metal,0.0..=1.0); }
        }
        ui.separator(); self.swatch_tray(ui);
        if fill_target && (material_changed || self.active_color()!=initial_active_color) {
            self.schedule_preview_update(ui.ctx());
        }
        });
    }
    fn swatch_tray(&mut self,ui:&mut egui::Ui) {
        let current=self.active_color().map(|v|(v.clamp(0.0,1.0)*255.0).round() as u8);
        let shift=ui.input(|i|i.modifiers.shift); let (mut pick,mut remove,mut add)=(None,None,false);
        ui.horizontal_wrapped(|ui|{
            ui.spacing_mut().item_spacing=vec2(3.0,3.0);
            for (index,color) in self.prefs.swatches.iter().enumerate() {
                let (rect,response)=ui.allocate_exact_size(vec2(18.0,18.0),egui::Sense::click());
                let selected=*color==current;
                ui.painter().rect_filled(rect,2.0,Color32::from_rgb(color[0],color[1],color[2]));
                ui.painter().rect_stroke(rect,2.0,egui::Stroke::new(if selected{2.0}else{1.0},if selected{Color32::WHITE}else{Color32::from_gray(80)}),egui::StrokeKind::Inside);
                let hint=format!("#{:02x}{:02x}{:02x}\nClick: brush color\nShift+click: second color\nRight-click: remove",color[0],color[1],color[2]);
                let response=response.on_hover_text(hint);
                if response.clicked(){pick=Some((index,shift));}
                if response.secondary_clicked(){remove=Some(index);}
            }
            let can_add=self.prefs.swatches.len()<prefs::MAX_SWATCHES&&!self.prefs.swatches.contains(&current);
            let plus=egui::Button::new(icons::PLUS).min_size(vec2(18.0,18.0));
            if ui.add_enabled(can_add,plus).on_hover_text("Save the brush color in the tray").clicked(){add=true;}
        });
        if self.prefs.swatches.is_empty(){ui.label(egui::RichText::new("Press + to keep the brush color here.").small().color(theme::TEXT_DIM));}
        if let Some((index,second))=pick {
            let c=self.prefs.swatches[index].map(|v|v as f32/255.0);
            if second{self.color2=c;}else{self.set_active_color(c);}
        }
        if let Some(index)=remove{self.prefs.swatches.remove(index);}
        if add{self.prefs.swatches.push(current);}
    }

    fn context_toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_centered(|ui| {
            // The right-hand controls go first; the tool controls get what's
            // left and clip instead of running under them on narrow windows.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                // Ray trace remains in View and on F10; this toolbar slot opens the Color HUD.
                if matches!(self.tool, Tool::Paint | Tool::Picker) && self.target == StrokeTarget::Content {
                    let color_btn=ui.add(theme::selectable(false,theme::with_icon(icons::DROP,"Color")))
                        .on_hover_text("Two colors, HSV, temperature, channel values and saved swatches");
                    if color_btn.clicked() { self.color_sliders.opened=self.active_color(); }
                    egui::Popup::from_toggle_button_response(&color_btn)
                        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
                        .show(|ui|self.color_hud(ui));
                }
                if ui
                    .add(theme::selectable(self.view.show_wire, theme::with_icon(icons::GRID_FOUR, "UV wire")))
                    .on_hover_text("Show the UV wireframe in the 2D view")
                    .clicked()
                {
                    self.view.show_wire = !self.view.show_wire;
                }
                if ui
                    .add(theme::selectable(self.view.show_wire_3d, theme::with_icon(icons::CUBE_TRANSPARENT, "Wireframe")))
                    .on_hover_text("Show the mesh's edges in the 3D view (W)")
                    .clicked()
                {
                    self.view.show_wire_3d = !self.view.show_wire_3d;
                }
                if ui
                    .add(theme::selectable(self.view.show_seams_3d, theme::with_icon(icons::POLYGON, "Seams")))
                    .on_hover_text("UV seams over the 3D view")
                    .clicked()
                {
                    self.view.show_seams_3d = !self.view.show_seams_3d;
                }
                ui.separator();
                for (l, icon, tip) in [
                    (Layout::View2d, icons::SQUARE, "2D view only (F3)"),
                    (Layout::View3d, icons::CUBE, "3D view only (F2)"),
                    (Layout::Split, icons::COLUMNS, "3D and 2D side by side (F1)"),
                ] {
                    if theme::icon_button(ui, icon, self.layout == l, 24.0).on_hover_text(tip).clicked() {
                        self.layout = l;
                    }
                }
                ui.separator();
                egui::ComboBox::from_id_salt("viewmode")
                    .selected_text(self.mode_name(self.view.mode))
                    .width(130.0)
                    .height(600.0)
                    .show_ui(ui, |ui| {
                        ui.label(egui::RichText::new("MATERIAL").small().color(theme::TEXT_DIM));
                        for m in ViewMode::CHANNELS {
                            ui.selectable_value(&mut self.view.mode, m, m.name());
                        }
                        ui.separator();
                        ui.label(egui::RichText::new("MESH MAPS").small().color(theme::TEXT_DIM));
                        for m in ViewMode::all().into_iter().skip(ViewMode::CHANNELS.len()) {
                            ui.selectable_value(&mut self.view.mode, m, m.name());
                        }
                        if !self.bake_settings.custom.is_empty() {
                            ui.separator();
                            ui.label(egui::RichText::new("CUSTOM MAPS").small().color(theme::TEXT_DIM));
                        }
                        for c in &self.bake_settings.custom {
                            ui.selectable_value(&mut self.view.mode, ViewMode::Custom(c.id), &c.name);
                        }
                    })
                    .response
                    .on_hover_text("What the views show (M: material, C: next channel)");
                // Color selection is consolidated in the Color popup at the right end
                // of this toolbar; do not show a second, competing color chip here.
                ui.label(egui::RichText::new("Display").color(theme::TEXT_DIM));
                ui.separator();
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    ui.set_clip_rect(ui.max_rect().intersect(ui.clip_rect()));
                    ui.spacing_mut().slider_width = 90.0;
                    let fix_name;
                    let name = match (self.tool, self.target) {
                        (Tool::Paint, StrokeTarget::BakeFix(m)) => {
                            fix_name = format!("Fix bake: {}", MAPS[m].name);
                            fix_name.as_str()
                        }
                        (Tool::Eraser, StrokeTarget::BakeFix(_)) => "Restore bake",
                        (Tool::Paint, StrokeTarget::Mask) => "Paint (mask)",
                        (Tool::Paint, _) => "Paint",
                        (Tool::Eraser, _) => "Eraser",
                        (Tool::Picker, _) => "Pick material (click the surface)",
                        (Tool::PolyFill, StrokeTarget::Mask) => "Polygon fill (mask)",
                        (Tool::PolyFill, _) => "Polygon fill",
                        (Tool::Path, StrokeTarget::Mask) => "Path (mask)",
                        (Tool::Path, _) => "Path",
                    };
                    let icon = match self.tool {
                        Tool::Paint => icons::PAINT_BRUSH,
                        Tool::Eraser => icons::ERASER,
                        Tool::Picker => icons::EYEDROPPER,
                        Tool::PolyFill => icons::POLYGON,
                        Tool::Path => icons::PATH,
                    };
                    ui.label(egui::RichText::new(theme::with_icon(icon, name)).strong().color(Color32::WHITE));
                    ui.separator();
                    if self.tool == Tool::PolyFill {
                        ui.spacing_mut().item_spacing.x = 2.0;
                        for (mode, icon, tip) in [
                            (FillMode::Triangle, icons::TRIANGLE, "Triangle: fill one triangle"),
                            (FillMode::Polygon, icons::SQUARE, "Polygon: fill a flat quad (two coplanar triangles)"),
                            (FillMode::Loop, icons::COLUMNS, "Face loop: a strip of quads through the edge nearest the cursor, as far as the quads go (a ring around a cylinder, a row of panels). Point near another edge of the quad to turn it."),
                            (FillMode::Flat, icons::MAGIC_WAND, "Flat area: spreads from the face until an edge sharper than the angle: a panel, a bevel, a side of a box"),
                            (FillMode::Part, icons::CUBE, "Loose part: a connected piece of one object"),
                            (FillMode::Object, icons::CUBE_TRANSPARENT, "Object: a whole imported object"),
                            (FillMode::UvIsland, icons::GRID_FOUR, "UV island: a whole island of the UV layout"),
                            (FillMode::Set, icons::SPHERE, "Texture set: every face of the material"),
                        ] {
                            let on = self.poly.mode == mode;
                            if theme::icon_button(ui, icon, on, 24.0).on_hover_text(tip).clicked() {
                                self.poly.mode = mode;
                            }
                        }
                        ui.add_space(4.0);
                        ui.label(egui::RichText::new(self.poly.mode.name()).color(Color32::WHITE));
                        if self.poly.mode == FillMode::Flat {
                            ui.add(egui::Slider::new(&mut self.poly.angle, 1.0..=90.0).max_decimals(0).suffix("°"))
                                .on_hover_text("The sharpest edge a flat area spreads across");
                        }
                        ui.separator();
                        ui.spacing_mut().item_spacing.x = 6.0;
                        ui.label(egui::RichText::new("Shift+drag").color(theme::TEXT_DIM));
                        if theme::icon_button(ui, icons::SELECTION, !self.poly.lasso, 24.0).on_hover_text("Shift+drag a box").clicked() {
                            self.poly.lasso = false;
                        }
                        if theme::icon_button(ui, icons::LASSO, self.poly.lasso, 24.0).on_hover_text("Shift+drag a lasso: a free shape").clicked() {
                            self.poly.lasso = true;
                        }
                        if theme::icon_button(ui, icons::SELECTION_ALL, self.poly.through, 24.0)
                            .on_hover_text("Select through: the box or lasso also takes faces hidden behind others (in 3D)")
                            .clicked()
                        {
                            self.poly.through = !self.poly.through;
                        }
                        ui.separator();
                        let text = theme::with_icon(icons::ERASER, "Erase");
                        if ui.add(theme::selectable(self.poly.erase, text)).on_hover_text("Remove paint instead (or hold Ctrl while clicking)").clicked() {
                            self.poly.erase = !self.poly.erase;
                        }
                        ui.add_space(6.0);
                        ui.label(egui::RichText::new("Opacity").color(theme::TEXT_DIM));
                        let mut p = self.brush.opacity * 100.0;
                        if ui.add(egui::Slider::new(&mut p, 1.0..=100.0).max_decimals(0).suffix("%")).changed() {
                            self.brush.opacity = p / 100.0;
                        }
                        ui.label(egui::RichText::new("Click or drag across faces, Shift+drag a box or lasso. Uses the brush material.").small().color(theme::TEXT_DIM));
                    } else if self.tool == Tool::Path {
                        self.path_toolbar(ui);
                    } else {
                    let s = |ui: &mut egui::Ui, label: &str, v: &mut f32, r: std::ops::RangeInclusive<f32>, pct: bool| {
                        ui.label(egui::RichText::new(label).color(theme::TEXT_DIM));
                        let resp = if pct {
                            let mut p = *v * 100.0;
                            let lo = *r.start() * 100.0;
                            let hi = *r.end() * 100.0;
                            let resp = ui.add(egui::Slider::new(&mut p, lo..=hi).max_decimals(0).suffix("%"));
                            if resp.changed() {
                                *v = p / 100.0;
                            }
                            resp
                        } else {
                            ui.add(egui::Slider::new(v, r).max_decimals(0))
                        };
                        ui.add_space(6.0);
                        resp
                    };
                    // Keep the two brush colors and their swap control beside Paint.
                    if matches!(self.tool, Tool::Paint | Tool::Picker) && self.target == StrokeTarget::Content {
                        self.color_slots(ui);
                        ui.separator();
                    }
                    s(ui, "Size", &mut self.brush.size, 2.0..=800.0, false)
                        .on_hover_text("Brush size ([ and ] to shrink / grow, X + right-drag in the view)");
                    s(ui, "Flow", &mut self.brush.flow, 0.01..=1.0, true);
                    s(ui, "Stroke opacity", &mut self.brush.opacity, 0.01..=1.0, true)
                        .on_hover_text("Hold Z and right-drag horizontally in a view to adjust stroke opacity");
                    s(ui, "Hardness", &mut self.brush.hardness, 0.0..=0.99, true);
                    s(ui, "Spacing", &mut self.brush.spacing, 0.02..=3.0, true);
                    ui.label(egui::RichText::new("Symmetry").color(theme::TEXT_DIM));
                    egui::ComboBox::from_id_salt("symmetry")
                        .width(52.0)
                        .selected_text(["Off", "X", "Y", "Z"][self.brush.symmetry as usize])
                        .show_ui(ui, |ui| {
                            for (i, name) in ["Off", "X", "Y", "Z"].iter().enumerate() {
                                ui.selectable_value(&mut self.brush.symmetry, i as u8, *name);
                            }
                        })
                        .response
                        .on_hover_text("Mirror strokes across the mesh's center");
                    ui.add_space(6.0);
                    ui.label(egui::RichText::new("Projection").color(theme::TEXT_DIM));
                    projection_combo(ui, "projection_bar", &mut self.brush.projection);
                    }
                });
            });
        });
    }

    fn tool_strip(&mut self, ui: &mut egui::Ui) {
        ui.add_space(6.0);
        ui.spacing_mut().item_spacing.y = 4.0;
        for (tool, icon, tip) in [
            (Tool::Paint, icons::PAINT_BRUSH, "Paint (1)"),
            (Tool::Eraser, icons::ERASER, "Eraser (2)"),
            (Tool::PolyFill, icons::POLYGON, "Polygon fill: triangles, quads, mesh parts or UV islands (3 / P)"),
            (Tool::Path, icons::PATH, "Path: place points, then paint along them; stitches, seams, panel lines (4 / L)"),
            (Tool::Picker, icons::EYEDROPPER, "Pick material from the surface (I)"),
        ] {
            if theme::icon_button(ui, icon, self.tool == tool, 34.0).on_hover_text(tip).clicked() {
                self.set_tool(tool);
            }
        }
        ui.add_space(4.0);
        ui.separator();
        ui.add_space(4.0);
        if theme::icon_button(ui, icons::CUBE_TRANSPARENT, self.view.show_wire_3d, 34.0).on_hover_text("Wireframe in the 3D view (W)").clicked() {
            self.view.show_wire_3d = !self.view.show_wire_3d;
        }
        if theme::icon_button(ui, icons::FIRE, self.show_bake, 34.0).on_hover_text("Bake mesh maps…").clicked() {
            self.show_bake = !self.show_bake;
        }
        if theme::icon_button(ui, icons::EXPORT, self.export.open, 34.0).on_hover_text("Export textures… (Ctrl+Shift+E)").clicked() {
            self.export.open = !self.export.open;
        }
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_centered(|ui| {
            ui.label(egui::RichText::new(&self.status).color(theme::TEXT_DIM));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let mb = |b: u64| b as f64 / (1024.0 * 1024.0);
                ui.label(
                    egui::RichText::new(format!(
                        "{:.1} ms  │  VRAM {:.0} MB  │  Undo {:.1} MB RAM ({} steps)  │  {}²  │  {} tris",
                        self.frame_ms,
                        mb(self.engine.vram_bytes()),
                        mb(self.undo.ram_bytes() as u64),
                        self.undo.undo.len(),
                        self.engine.res,
                        self.mesh.triangle_count(),
                    ))
                    .color(theme::TEXT_DIM)
                    .small(),
                );
                if let Some(server) = &self.mcp {
                    if server.enabled.load(std::sync::atomic::Ordering::Relaxed) {
                        ui.label(egui::RichText::new(format!("MCP :{}", server.port)).color(theme::ACCENT).small())
                            .on_hover_text(format!("AI agents can connect at {}", server.url()));
                    }
                }
                if let Some((p, what)) = self.bake_progress() {
                    ui.add(egui::ProgressBar::new(p).desired_width(140.0).text(format!("{what} {:.0}%", p * 100.0)));
                }
            });
        });
    }

    // ------------------------------------------------------------------ layers

    fn layers_panel(&mut self, ui: &mut egui::Ui) {
        // Which texture set the stack below belongs to.
        if self.mesh.sets.len() > 1 {
            let mut pick = None;
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Texture set").color(theme::TEXT_DIM));
                let current = self.mesh.sets[self.sets.active].name.clone();
                egui::ComboBox::from_id_salt("texture set")
                    .selected_text(current)
                    .width(ui.available_width())
                    .show_ui(ui, |ui| {
                        for (k, set) in self.mesh.sets.iter().enumerate() {
                            let resp = ui.selectable_label(k == self.sets.active, &set.name);
                            // Hovering an entry shows its part of the mesh.
                            if resp.hovered() {
                                self.sets.hover = Some(k);
                            }
                            if resp.clicked() {
                                pick = Some(k);
                            }
                        }
                    })
                    .response
                    .on_hover_text("Each material of the mesh has its own layers, size and mesh maps");
            });
            if let Some(k) = pick {
                self.pick_set(k, ui.ctx());
            }
            ui.add_space(2.0);
        }
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            let mut btn = |ui: &mut egui::Ui, icon: &str, tip: &str, action: Option<Action>| {
                let enabled = action.is_some();
                let resp = ui.add_enabled_ui(enabled, |ui| theme::icon_button(ui, icon, false, 28.0)).inner;
                if resp.on_hover_text(tip).clicked() {
                    if let Some(a) = action {
                        self.pending.push(a);
                    }
                }
            };
            let has = self.selected < self.engine.layers.len();
            let no_mask = has && self.engine.layers[self.selected].mask.is_none();
            btn(ui, icons::PAINT_BRUSH, "Add paint layer", Some(Action::AddPaint));
            btn(ui, icons::PAINT_BUCKET, "Add fill layer", Some(Action::AddFill(FillParams::default(), "Fill layer".into())));
            btn(ui, icons::SPARKLE, "Add graph fill layer (a material graph)", Some(Action::AddGraphLayer));
            btn(ui, icons::FOLDER_PLUS, "Add folder (inside the selected folder)", Some(Action::AddFolder));
            btn(ui, icons::MASK, "Add black mask", no_mask.then_some(Action::AddMask { white: false }));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                btn(ui, icons::TRASH, "Delete layer (Delete)", has.then_some(Action::Delete));
                btn(ui, icons::FILE_PLUS, "Paste copied layer (Ctrl+V)", self.layer_clipboard.as_ref().is_some_and(|layers| !layers.is_empty()).then_some(Action::PasteLayer));
                btn(ui, icons::COPY, "Copy layer (Ctrl+C)", has.then_some(Action::CopyLayer));
                btn(ui, icons::COPY, "Duplicate layer", has.then_some(Action::Duplicate));
            });
        });
        ui.add_space(4.0);
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 1.0;
            for idx in (0..self.engine.layers.len()).rev() {
                let collapsed = self
                    .engine
                    .ancestors(idx)
                    .any(|a| matches!(self.engine.layers[a].content, Content::Folder(Folder { open: false, .. })));
                if !collapsed {
                    self.layer_row(ui, idx);
                }
            }
            // Clicking the empty space below the layers selects none.
            let rest = ui.available_rect_before_wrap();
            if rest.height() > 2.0 && ui.allocate_rect(rest, Sense::click()).clicked() {
                self.selected = NO_LAYER;
                self.target = StrokeTarget::Content;
            }
            if self.engine.layers.is_empty() {
                ui.label(egui::RichText::new("No layers. Add a paint or fill layer, or drop a smart material from the shelf.").color(theme::TEXT_DIM));
            }
        });
    }

    fn layer_row(&mut self, ui: &mut egui::Ui, idx: usize) {
        let h = 46.0;
        let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), h), Sense::click_and_drag());
        let indent = self.engine.depth(idx) as f32 * 16.0;
        let layer = &self.engine.layers[idx];
        let (id, visible, name) = (layer.id, layer.visible, layer.name.clone());
        resp.widget_info(|| egui::WidgetInfo::selected(egui::Role::ListItem, true, idx == self.selected, &name));
        let folder_open = match layer.content {
            Content::Folder(f) => Some(f.open),
            _ => None,
        };
        let selected = idx == self.selected;
        let base_id = ui.scope_id().with(("layer", id));

        let painter = ui.painter_at(rect);
        let bg = if selected {
            theme::SELECT_BG
        } else if resp.hovered() {
            theme::WIDGET
        } else {
            theme::PANEL_DARK
        };
        painter.rect_filled(rect.shrink2(vec2(0.0, 1.0)), theme::RADIUS, bg);
        if selected {
            painter.rect_stroke(rect.shrink2(vec2(0.0, 1.0)), theme::RADIUS, Stroke::new(1.0, theme::ACCENT), egui::StrokeKind::Inside);
        }
        if selected {
            painter.rect_filled(Rect::from_min_size(rect.min + vec2(0.0, 6.0), vec2(3.0, h - 12.0)), 1.5, theme::ACCENT);
        }

        // Visibility toggle.
        let eye_rect = Rect::from_center_size(pos2(rect.min.x + 16.0, rect.center().y), vec2(22.0, 22.0));
        let eye = ui.interact(eye_rect, base_id.with("eye"), Sense::click());
        let (eye_icon, eye_col) = if visible { (icons::EYE, theme::TEXT) } else { (icons::EYE_SLASH, theme::TEXT_DIM.gamma_multiply(0.6)) };
        let eye_col = if eye.hovered() { Color32::WHITE } else { eye_col };
        painter.text(eye_rect.center(), egui::Align2::CENTER_CENTER, eye_icon, egui::FontId::proportional(15.0), eye_col);

        // Content thumbnail, indented by folder depth.
        let thumb = Rect::from_min_size(pos2(eye_rect.max.x + 6.0 + indent, rect.min.y + 5.0), vec2(36.0, 36.0));
        let thumb_resp = ui.interact(thumb, base_id.with("thumb"), Sense::click());
        if indent > 0.0 {
            let x = eye_rect.max.x + indent - 4.0;
            painter.line_segment([pos2(x, rect.top()), pos2(x, rect.bottom())], Stroke::new(1.0, theme::BORDER));
        }
        match &layer.content {
            Content::Paint(_) => {
                paint_checker(&painter, thumb);
                if let Some(tid) = layer.thumb {
                    painter.image(tid, thumb, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
                }
            }
            Content::Fill(f) => match self.engine.content_thumb(id) {
                // The fill as it comes out on the texture set (images, graph, variation).
                Some(tid) => {
                    paint_checker(&painter, thumb);
                    painter.image(tid, thumb, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
                }
                None => {
                    painter.rect_filled(thumb, 0.0, srgb_u8(f.mat.color));
                }
            },
            Content::Folder(f) => {
                // Folder icon, open or closed; clicking it toggles.
                let col = if visible { Color32::from_rgb(0xe8, 0xb9, 0x4a) } else { theme::TEXT_DIM };
                let icon = if f.open { icons::FOLDER_OPEN } else { icons::FOLDER };
                painter.text(thumb.center(), egui::Align2::CENTER_CENTER, icon, egui::FontId::proportional(24.0), col);
            }
        }
        if selected && self.target == StrokeTarget::Content {
            painter.rect_stroke(thumb.expand(1.0), 0.0, Stroke::new(2.0, theme::ACCENT), egui::StrokeKind::Outside);
        }

        // Mask thumbnail.
        let mut x = thumb.max.x + 6.0;
        let mut mask_resp = None;
        if let Some(mask) = &layer.mask {
            let mrect = Rect::from_min_size(pos2(x, rect.min.y + 5.0), vec2(36.0, 36.0));
            mask_resp = Some(ui.interact(mrect, base_id.with("mask"), Sense::click()));
            painter.rect_filled(mrect, 0.0, Color32::BLACK);
            match self.engine.mask_thumb(id) {
                // The mask as it comes out (painting, smart mask, layers, filters).
                Some(tid) => {
                    painter.image(tid, mrect, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
                }
                None if mask.paint.is_some() => {
                    painter.circle_filled(mrect.center(), 10.0, Color32::WHITE);
                }
                None => {}
            }
            // A smart mask says so in the corner.
            if mask.generator.is_some() || !mask.layers.is_empty() {
                let at = mrect.right_bottom() - vec2(7.0, 7.0);
                painter.circle_filled(at, 6.5, Color32::from_black_alpha(190));
                painter.text(at, egui::Align2::CENTER_CENTER, icons::SPARKLE, egui::FontId::proportional(9.5), theme::ACCENT);
            }
            if selected && self.target == StrokeTarget::Mask {
                painter.rect_stroke(mrect.expand(1.0), 0.0, Stroke::new(2.0, theme::ACCENT), egui::StrokeKind::Outside);
            }
            x = mrect.max.x + 6.0;
        }

        // Name and blend info.
        let text_col = if visible { Color32::WHITE } else { theme::TEXT_DIM };
        let title = theme::elided(&painter, &name, egui::FontId::proportional(12.5), text_col, rect.max.x - x - 8.0);
        let name_cut = title.elided;
        painter.galley(pos2(x, rect.min.y + 13.0 - title.size().y * 0.5), title, text_col);
        let mb = self.engine.layer_vram_bytes(idx) as f64 / (1024.0 * 1024.0);
        let info = match &layer.content {
            Content::Folder(_) => {
                let inside = self.engine.block(idx).len() - 1;
                format!("Folder  ·  {inside} layers  ·  {:.0}%", layer.opacity * 100.0)
            }
            Content::Fill(_) => format!("Fill  {} {:.0}%  ·  {:.0} MB", layer.blend.short(), layer.opacity * 100.0, mb),
            Content::Paint(_) => format!("Paint  {} {:.0}%  ·  {:.0} MB", layer.blend.short(), layer.opacity * 100.0, mb),
        };
        painter.text(pos2(x, rect.min.y + 31.0), egui::Align2::LEFT_CENTER, info, egui::FontId::proportional(10.5), theme::TEXT_DIM);
        painter.line_segment([rect.left_bottom(), rect.right_bottom()], Stroke::new(1.0, theme::BORDER));
        let resp = if name_cut { resp.on_hover_text(&name) } else { resp };

        // Interaction.
        if eye.clicked() {
            self.engine.layers[idx].visible = !visible;
            self.engine.mark_dirty();
        } else if mask_resp.as_ref().is_some_and(|r| r.clicked()) {
            self.selected = idx;
            self.target = StrokeTarget::Mask;
        } else if thumb_resp.clicked() {
            if let Content::Folder(f) = &mut self.engine.layers[idx].content {
                f.open = !f.open;
            }
            self.selected = idx;
            self.target = StrokeTarget::Content;
        } else if resp.double_clicked() && folder_open.is_some() {
            if let Content::Folder(f) = &mut self.engine.layers[idx].content {
                f.open = !f.open;
            }
        } else if resp.clicked() {
            if self.selected != idx || matches!(self.target, StrokeTarget::BakeFix(_)) {
                self.target = StrokeTarget::Content;
            }
            self.selected = idx;
        }

        // Drag to reorder: the top or bottom part of a row drops above or below
        // it; the middle of a folder (or anywhere below an open one) drops inside.
        resp.dnd_set_drag_payload(LayerDrag(idx));
        let drop_at = |y: f32| {
            let t = (y - rect.top()) / rect.height();
            match folder_open {
                Some(open) if t >= 0.25 && (open || t <= 0.75) => Drop::Into(id),
                _ if t < 0.5 => Drop::Above(id),
                _ => Drop::Below(id),
            }
        };
        if let (Some(from), Some(p)) = (resp.dnd_hover_payload::<LayerDrag>(), ui.ctx().pointer_interact_pos()) {
            if from.0 != idx {
                let accent = Stroke::new(2.0, theme::ACCENT);
                match drop_at(p.y) {
                    Drop::Above(_) => {
                        ui.painter().line_segment([rect.left_top(), rect.right_top()], accent);
                    }
                    Drop::Below(_) => {
                        ui.painter().line_segment([rect.left_bottom(), rect.right_bottom()], accent);
                    }
                    Drop::Into(_) => {
                        ui.painter().rect_stroke(rect.shrink(1.0), 0.0, accent, egui::StrokeKind::Inside);
                    }
                }
            }
        }
        if let (Some(from), Some(p)) = (dnd_release::<LayerDrag>(&resp), ui.ctx().pointer_interact_pos()) {
            if from.0 != idx {
                self.pending.push(Action::MoveTo { from: from.0, drop: drop_at(p.y) });
            }
        }
        // Shelf items land where they're dropped; a generator masks the row's layer.
        if let (Some(item), Some(p)) = (resp.dnd_hover_payload::<ShelfItem>(), ui.ctx().pointer_interact_pos()) {
            let accent = Stroke::new(2.0, theme::ACCENT);
            match (*item, drop_at(p.y)) {
                (ShelfItem::Generator(_), _) | (_, Drop::Into(_)) => {
                    ui.painter().rect_stroke(rect.shrink(1.0), 0.0, accent, egui::StrokeKind::Inside);
                }
                (_, Drop::Above(_)) => {
                    ui.painter().line_segment([rect.left_top(), rect.right_top()], accent);
                }
                (_, Drop::Below(_)) => {
                    ui.painter().line_segment([rect.left_bottom(), rect.right_bottom()], accent);
                }
            }
        }
        if let (Some(item), Some(p)) = (dnd_release::<ShelfItem>(&resp), ui.ctx().pointer_interact_pos()) {
            if let ShelfItem::Generator(g) = *item {
                self.selected = idx;
                self.pending.push(Action::AddGenerator(g));
            } else {
                let row_parent = self.engine.layers[idx].parent;
                let (at, parent) = match drop_at(p.y) {
                    Drop::Above(_) => (idx + 1, row_parent),
                    Drop::Below(_) => (self.engine.block(idx).start, row_parent),
                    Drop::Into(_) => (idx, Some(id)),
                };
                self.pending.push(Action::ShelfDrop { item: *item, at, parent });
            }
        }
        resp.context_menu(|ui| {
            self.selected = idx;
            self.layer_menu_items(ui);
        });
    }

    fn texture_set_panel(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            if self.mesh.sets.len() > 1 {
                section(ui, "Texture sets");
                self.texture_set_list(ui);
            }
            section(ui, "Texture set");
            labeled(ui, "Mesh", |ui| ui.label(&self.mesh.name));
            let set = &self.mesh.sets[self.sets.active.min(self.mesh.sets.len() - 1)];
            if self.mesh.sets.len() > 1 {
                labeled(ui, "Material", |ui| ui.label(&set.name));
            }
            labeled(ui, "Triangles", |ui| ui.label((set.tris.end - set.tris.start).to_string()));
            let res = self.engine.res;
            labeled(ui, "Size", |ui| {
                egui::ComboBox::from_id_salt("res").selected_text(format!("{res} x {res}")).show_ui(ui, |ui| {
                    for r in RESOLUTIONS {
                        if ui.selectable_label(r == res, format!("{r} x {r}")).clicked() && r != res {
                            self.pending.push(Action::SetResolution(r));
                        }
                    }
                });
            });
            ui.label(egui::RichText::new("Changing size resamples every painted layer; stroke history is cleared.").small().color(theme::TEXT_DIM));

            section(ui, "Normal");
            slider(ui, "Height scale", &mut self.view.height_strength, 0.0..=8.0);

            section(ui, "Mesh maps");
            if ui.button("Open baker…").clicked() {
                self.show_bake = true;
            }
        });
    }

    // ------------------------------------------------------------------ preview

    /// Starts decoding an HDRI; it replaces the current one when ready.
    pub(crate) fn load_env(&mut self, source: EnvSource, ctx: &egui::Context) {
        let (tx, rx) = mpsc::channel();
        let ctx = ctx.clone();
        self.status = format!("Loading environment '{}'…", source.name());
        std::thread::spawn(move || {
            let _ = tx.send(environment::decode(&source));
            ctx.request_repaint();
        });
        // A newer load replaces any running one; its result is dropped.
        self.env_load = Some(rx);
    }

    fn poll_hdri_thumbs(&mut self, ctx: &egui::Context) {
        if let Some((name, px)) = self.hdri_new_thumb.take() {
            let tex = ctx.load_texture(format!("hdri {name}"), egui::ColorImage::from_rgba_unmultiplied(HDRI_THUMB, &px), Default::default());
            self.hdri_thumbs.insert(name, tex);
        }
        let Some(rx) = &self.hdri_thumb_rx else { return };
        let mut done = false;
        loop {
            match rx.try_recv() {
                Ok((name, px)) => {
                    let tex = ctx.load_texture(format!("hdri {name}"), egui::ColorImage::from_rgba_unmultiplied(HDRI_THUMB, &px), Default::default());
                    self.hdri_thumbs.entry(name).or_insert(tex);
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    done = true;
                    break;
                }
            }
        }
        if done {
            self.hdri_thumb_rx = None;
        }
    }

    fn poll_env(&mut self) {
        let Some(rx) = &self.env_load else { return };
        let result = match rx.try_recv() {
            Ok(r) => r,
            Err(mpsc::TryRecvError::Empty) => return,
            Err(mpsc::TryRecvError::Disconnected) => Err("The environment loader stopped unexpectedly.".into()),
        };
        self.env_load = None;
        match result {
            Ok(img) => {
                // Its picture (a loaded file has none yet); made into a texture by poll_hdri_thumbs.
                self.hdri_new_thumb = Some((img.source.name(), img.thumbnail(HDRI_THUMB[0], HDRI_THUMB[1])));
                self.env = self.env_baker.upload(&self.rs, &img);
                self.status = format!("Environment: {}.", self.env.source.name());
                for w in self.mcp_env_waiters.drain(..) {
                    let _ = w.send(Ok(vec![crate::mcp::Content::Text(format!("Environment '{}' in use.", self.env.source.name()))]));
                }
            }
            Err(e) => {
                for w in self.mcp_env_waiters.drain(..) {
                    let _ = w.send(Err(e.clone()));
                }
                self.status = e;
            }
        }
    }

    fn preview_panel(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            section(ui, "Camera");
            let mut fov = self.cam.fov_y.to_degrees();
            if slider(ui, "Field of view", &mut fov, 10.0..=90.0) {
                self.cam.fov_y = fov.to_radians();
            }

            section(ui, "Shader");
            self.preview_shader_section(ui);
            let v = &mut self.view;

            section(ui, "Environment");
            let current = self.env_load.as_ref().map_or_else(|| self.env.source.name(), |_| "Loading…".to_string());
            let mut pick = None;
            // The HDRI in use; clicking it opens a scrolling grid to choose from.
            let full = ui.available_width();
            let (r, resp) = ui.allocate_exact_size(vec2(full, full * 0.5), Sense::click());
            match self.hdri_thumbs.get(&self.env.source.name()) {
                Some(t) => {
                    ui.painter().image(t.id(), r, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
                }
                None => {
                    ui.painter().rect_filled(r, 0.0, theme::PANEL_DARK);
                }
            }
            let edge = if resp.hovered() { Stroke::new(1.5, theme::ACCENT) } else { Stroke::new(1.0, theme::OUTLINE) };
            ui.painter().rect_stroke(r, 0.0, edge, egui::StrokeKind::Inside);
            ui.painter().text(
                r.right_bottom() - vec2(8.0, 8.0),
                egui::Align2::RIGHT_BOTTOM,
                "Change ▾",
                egui::FontId::proportional(12.0),
                Color32::WHITE,
            );
            let resp = resp.on_hover_text("Choose an HDRI");
            ui.label(egui::RichText::new(current).strong().color(Color32::WHITE));
            egui::Popup::menu(&resp).close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside).show(|ui| {
                ui.set_width(360.0);
                ui.label(egui::RichText::new("Environment").strong().color(Color32::WHITE));
                egui::ScrollArea::vertical().max_height(340.0).show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing = vec2(6.0, 6.0);
                        let w = 170.0;
                        for &i in &self.env_builtins {
                            let src = EnvSource::Builtin(i);
                            let name = environment::BUILTIN[i].0;
                            let (rect, resp) = ui.allocate_exact_size(vec2(w, w * 0.5 + 18.0), Sense::click());
                            let img = Rect::from_min_size(rect.min, vec2(w, w * 0.5));
                            let p = ui.painter();
                            match self.hdri_thumbs.get(name) {
                                Some(t) => {
                                    p.image(t.id(), img, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
                                }
                                None => {
                                    p.rect_filled(img, 0.0, theme::PANEL_DARK);
                                    p.text(img.center(), egui::Align2::CENTER_CENTER, "…", egui::FontId::proportional(14.0), theme::TEXT_DIM);
                                }
                            }
                            let selected = self.env.source == src;
                            let stroke = if selected {
                                Stroke::new(2.0, theme::ACCENT)
                            } else if resp.hovered() {
                                Stroke::new(1.0, Color32::from_gray(120))
                            } else {
                                Stroke::new(1.0, theme::OUTLINE)
                            };
                            p.rect_stroke(img, 0.0, stroke, egui::StrokeKind::Inside);
                            p.text(
                                pos2(rect.min.x + 2.0, img.max.y + 2.0),
                                egui::Align2::LEFT_TOP,
                                name,
                                egui::FontId::proportional(11.5),
                                if selected { Color32::WHITE } else { theme::TEXT_DIM },
                            );
                            if resp.clicked() {
                                pick = Some(src);
                                ui.close();
                            }
                        }
                    });
                });
                ui.separator();
                if ui.button("Load HDRI file…").on_hover_text("An equirectangular .hdr or .exr").clicked() {
                    if let Some(path) = rfd::FileDialog::new().add_filter("HDRI", &["hdr", "exr"]).pick_file() {
                        pick = Some(EnvSource::File(path));
                    }
                    ui.close();
                }
            });
            let mut deg = v.env_rotation.to_degrees();
            if slider(ui, "Rotation", &mut deg, -180.0..=180.0) {
                v.env_rotation = deg.to_radians();
            }
            slider(ui, "Intensity", &mut v.env_intensity, 0.0..=4.0);
            labeled(ui, "Ambient occlusion", |ui| {
                ui.add(egui::Slider::new(&mut v.ao_strength, 0.0..=1.0).custom_formatter(|x, _| format!("{:.0}%", x * 100.0)))
            })
            .on_hover_text("How much the baked ambient occlusion darkens crevices in the 3D view. Only the view: bakes and exports keep the full map.");
            labeled(ui, "Emission", |ui| ui.add(egui::Slider::new(&mut v.emission, 0.0..=20.0).logarithmic(true).max_decimals(2)))
                .on_hover_text("How bright emissive parts glow in the 3D view and the ray tracer. Only the view: exports keep the emission color as painted.");

            section(ui, "Key light");
            ui.checkbox(&mut v.key.on, theme::with_icon(icons::SUN, "Directional light with shadows"))
                .on_hover_text("A sun-like light on top of the environment: crisp highlights and contact shadows. It turns with the environment rotation (Shift+right-drag).");
            if v.key.on {
                slider(ui, "Intensity", &mut v.key.intensity, 0.0..=8.0);
                slider(ui, "Direction", &mut v.key.azimuth, -180.0..=180.0);
                slider(ui, "Height", &mut v.key.elevation, 5.0..=90.0);
                labeled(ui, "Color", |ui| color_edit(ui, &mut v.key.color));
            }

            section(ui, "Background");
            labeled(ui, "Show", |ui| {
                egui::ComboBox::from_id_salt("background").selected_text(v.background.name()).show_ui(ui, |ui| {
                    for b in Background::ALL {
                        ui.selectable_value(&mut v.background, b, b.name());
                    }
                });
            });
            match v.background {
                Background::Environment => {
                    slider(ui, "Blur", &mut v.bg_blur, 0.0..=1.0);
                    slider(ui, "Brightness", &mut v.bg_intensity, 0.0..=2.0);
                }
                Background::Color => {
                    labeled(ui, "Color", |ui| color_edit(ui, &mut v.bg_color));
                }
                Background::Transparent => {}
            }

            section(ui, "Post processing");
            labeled(ui, "Exposure", |ui| {
                ui.add(egui::Slider::new(&mut v.exposure, 0.05..=8.0).logarithmic(true).max_decimals(2));
            });
            let p = &mut v.post;
            labeled(ui, "Tone mapping", |ui| {
                egui::ComboBox::from_id_salt("tonemap").selected_text(p.tonemap.name()).show_ui(ui, |ui| {
                    for t in Tonemap::ALL {
                        ui.selectable_value(&mut p.tonemap, t, t.name()).on_hover_text(t.hint());
                    }
                });
            });
            slider(ui, "Contrast", &mut p.contrast, 0.5..=1.5);
            slider(ui, "Saturation", &mut p.saturation, 0.0..=2.0);
            slider(ui, "Temperature", &mut p.temperature, -1.0..=1.0);
            slider(ui, "Tint", &mut p.tint, -1.0..=1.0);
            slider(ui, "Bloom", &mut p.bloom, 0.0..=1.0);
            if p.bloom > 0.0 {
                slider(ui, "Bloom threshold", &mut p.bloom_threshold, 0.0..=4.0);
            }
            slider(ui, "Vignette", &mut p.vignette, 0.0..=1.0);
            slider(ui, "Chromatic ab.", &mut p.chromatic, 0.0..=1.0);
            slider(ui, "Sharpen", &mut p.sharpen, 0.0..=1.0);
            slider(ui, "Film grain", &mut p.grain, 0.0..=1.0);
            ui.add_space(6.0);
            if ui.button("Reset preview settings").clicked() {
                let keep = (v.mode, v.height_strength, v.show_wire, v.use_baked_normal);
                *v = ViewSettings { mode: keep.0, height_strength: keep.1, show_wire: keep.2, use_baked_normal: keep.3, ..ViewSettings::default() };
                self.cam.fov_y = self.prefs.fov_deg.to_radians();
            }
            if let Some(src) = pick {
                self.load_env(src, ui.ctx());
            }
        });
    }

    // ------------------------------------------------------------------ mesh maps

    fn mesh_maps_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.add(egui::Button::new(egui::RichText::new("Bake mesh maps…").strong()).min_size(vec2(130.0, 24.0))).clicked() {
                self.show_bake = true;
            }
            if let Some((p, what)) = self.bake_progress() {
                ui.add(egui::ProgressBar::new(p).text(format!("{what} {:.0}%", p * 100.0)));
            }
        });
        if let Some(r) = &self.engine.bake_report {
            let time = if r.from_cache {
                "loaded from the bake cache".to_string()
            } else {
                r.seconds().map_or("…".to_string(), |s| format!("{:.2}s GPU + {:.2}s prep", s, r.prep_seconds))
            };
            ui.label(
                egui::RichText::new(format!(
                    "{} · {} tris · {} parts · {} feature edges · {}² · {} rays/texel · {}",
                    r.source, r.triangles, r.parts, r.hard_edges, r.resolution, r.rays_per_texel, time
                ))
                .small()
                .color(theme::TEXT_DIM),
            );
            if let Some(filled) = r.filled_percent().filter(|f| *f > 0.0) {
                ui.label(
                    egui::RichText::new(format!("{filled:.2}% of texels missed by every ray took the nearest high-poly surface"))
                        .small()
                        .color(theme::TEXT_DIM),
                )
                .on_hover_text("Raise the cage there (Paint cage or Auto-fix) for an exact projection.");
            }
            if let Some(missed) = r.missed_percent() {
                let (col, text) = if missed > 0.05 {
                    (Color32::from_rgb(235, 120, 90), format!("⚠ {missed:.2}% of texels missed the bake source"))
                } else {
                    (Color32::from_rgb(120, 200, 120), "✔ No missed rays".to_string())
                };
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(text).color(col));
                    if missed > 0.05 && ui.small_button("Show").clicked() {
                        self.view.mode = ViewMode::MissedRays;
                    }
                });
            }
        } else if self.bake_job.is_none() {
            ui.label(egui::RichText::new("No maps baked yet.").color(theme::TEXT_DIM));
        }
        ui.checkbox(&mut self.view.use_baked_normal, "Use baked normal in material");
        ui.separator();

        let fixing = match self.target {
            StrokeTarget::BakeFix(m) => Some(m),
            _ => None,
        };
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            for (i, info) in MAPS.iter().enumerate() {
                let shown = self.view.mode == ViewMode::Map(i);
                let has_fix = self.engine.map_fix[i].is_some();
                let frame = egui::Frame::new()
                    .fill(if fixing == Some(i) { theme::ACCENT_DIM } else { theme::PANEL_DARK })
                    .inner_margin(egui::Margin::symmetric(6, 4));
                frame.show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.horizontal(|ui| {
                        let eye = egui::RichText::new(if shown { "◉" } else { "○" }).color(if shown { theme::ACCENT } else { theme::TEXT_DIM });
                        if ui.add(egui::Button::new(eye).frame(false)).on_hover_text("Display this map").clicked() {
                            self.view.mode = if shown { ViewMode::Material } else { ViewMode::Map(i) };
                        }
                        ui.label(egui::RichText::new(info.name).strong()).on_hover_text(info.desc);
                        if has_fix {
                            ui.label(egui::RichText::new("fixed").small().color(theme::ACCENT));
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if info.kind != MapKind::Info {
                                let label = if fixing == Some(i) { "Done" } else { "Fix" };
                                if ui
                                    .add_enabled(self.engine.has_bake, egui::Button::new(label))
                                    .on_hover_text("Paint corrections onto this map. The eraser restores the original bake.")
                                    .clicked()
                                {
                                    if fixing == Some(i) {
                                        self.target = StrokeTarget::Content;
                                    } else {
                                        self.target = StrokeTarget::BakeFix(i);
                                        self.view.mode = ViewMode::Map(i);
                                        self.set_tool(Tool::Paint);
                                    }
                                }
                                if has_fix && ui.small_button("✖").on_hover_text("Clear painted fixes").clicked() {
                                    self.engine.clear_map_fix(i, &mut self.undo);
                                }
                            }
                        });
                    });
                    if info.kind == MapKind::Scalar {
                        let lv = &mut self.engine.map_levels[i];
                        let mut changed = false;
                        egui::CollapsingHeader::new(egui::RichText::new("Levels").small())
                            .id_salt(("levels", i))
                            .default_open(*lv != Levels::default())
                            .show(ui, |ui| {
                                changed |= slider(ui, "Black", &mut lv.black, 0.0..=1.0);
                                changed |= slider(ui, "White", &mut lv.white, 0.0..=1.0);
                                changed |= slider(ui, "Gamma", &mut lv.gamma, 0.1..=4.0);
                                if ui.small_button("Reset").clicked() {
                                    *lv = Levels::default();
                                    changed = true;
                                }
                            });
                        if changed {
                            self.engine.map_levels_changed();
                        }
                    }
                });
                ui.add_space(2.0);
            }
            // The custom maps the bake made, by name.
            let baked: Vec<&bake::CustomMap> =
                self.bake_settings.custom.iter().filter(|c| self.engine.custom_index(c.id).is_some()).collect();
            if !baked.is_empty() {
                ui.label(egui::RichText::new("CUSTOM MAPS").small().color(theme::TEXT_DIM));
            }
            let mut show = None;
            for c in baked {
                let shown = self.view.mode == ViewMode::Custom(c.id);
                egui::Frame::new().fill(theme::PANEL_DARK).inner_margin(egui::Margin::symmetric(6, 4)).show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.horizontal(|ui| {
                        let eye = egui::RichText::new(if shown { "◉" } else { "○" }).color(if shown { theme::ACCENT } else { theme::TEXT_DIM });
                        if ui.add(egui::Button::new(eye).frame(false)).on_hover_text("Display this map").clicked() {
                            show = Some(if shown { ViewMode::Material } else { ViewMode::Custom(c.id) });
                        }
                        ui.label(egui::RichText::new(format!("{}  {}", bake_live::kind_icon(c.kind), c.name)).strong()).on_hover_text(c.kind.help());
                    });
                });
                ui.add_space(2.0);
            }
            if let Some(m) = show {
                self.view.mode = m;
            }
        });
    }

    fn bake_window(&mut self, ctx: &egui::Context) {
        const BAKE_WIDTH: f32 = 520.0;
        let mut open = self.show_bake;
        // Against the right edge, so the 3D view (and the gizmos in it) stays visible.
        let screen = ctx.content_rect();
        egui::Window::new("Bake Mesh Maps")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            // A fixed width (text wraps to it), so nothing inside can widen the window or
            // leave the bars above narrower than what's below them.
            .min_width(BAKE_WIDTH)
            .max_width(BAKE_WIDTH)
            .default_pos(egui::pos2((screen.max.x - BAKE_WIDTH - 30.0).max(screen.min.x), screen.min.y + 60.0))
            .show(ctx, |ui| {
            ui.set_width(BAKE_WIDTH);
            self.preview_bar(ui);
            // The settings scroll within the screen; the Bake button stays below them.
            let settings_height = (screen.height() - 260.0).max(200.0);
            egui::ScrollArea::vertical().max_height(settings_height).auto_shrink([false, true]).show(ui, |ui| {
            // Inside the scroll area too: text wraps here instead of running past the window.
            ui.set_max_width(BAKE_WIDTH - 14.0);
            let s = &mut self.bake_settings;
            section(ui, "Source");
            ui.horizontal(|ui| {
                if ui.radio(s.high_poly.is_none(), "Low poly only (geometry info)").clicked() {
                    s.high_poly = None;
                }
                if ui.radio(s.high_poly.is_some(), "High poly → low poly").clicked() && s.high_poly.is_none() {
                    s.high_poly = rfd::FileDialog::new().add_filter("Mesh (FBX, OBJ)", &crate::import::EXTENSIONS).pick_file();
                }
            });
            if let Some(path) = s.high_poly.clone() {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(path.display().to_string()).small().color(theme::TEXT_DIM));
                    if ui.small_button("Change…").clicked() {
                        if let Some(p) = rfd::FileDialog::new().add_filter("Mesh (FBX, OBJ)", &crate::import::EXTENSIONS).pick_file() {
                            s.high_poly = Some(p);
                        }
                    }
                });
                ui.label(
                    egui::RichText::new("The high poly must be in the same space as the low poly (no UVs needed).")
                        .small()
                        .color(theme::TEXT_DIM),
                );
                section(ui, "Projection");
                ui.checkbox(&mut s.match_by_name, "Match by mesh name")
                    .on_hover_text("Each low-poly part only receives detail from the high-poly part with the same name (_low / _high suffixes are ignored), so parts close together can't bleed into each other.");
                labeled(ui, "Cage mesh", |ui| {
                    match s.cage.clone() {
                        Some(p) => {
                            let name = p.file_name().map_or_else(|| p.display().to_string(), |n| n.to_string_lossy().into_owned());
                            ui.add(egui::Label::new(name).truncate()).on_hover_text(p.display().to_string());
                            if ui.small_button("✕").on_hover_text("Use the front/rear distances instead").clicked() {
                                s.cage = None;
                            }
                        }
                        None => {
                            if ui.button("Load cage…").on_hover_text("The low poly with its vertices pushed out to enclose the high poly (same topology)").clicked() {
                                s.cage = rfd::FileDialog::new().add_filter("Mesh (FBX, OBJ)", &crate::import::EXTENSIONS).pick_file();
                            }
                        }
                    }
                });
                if s.cage.is_none() {
                    slider(ui, "Front", &mut s.front, 0.001..=0.5);
                } else {
                    ui.label(egui::RichText::new("Rays start on the cage and travel to the low poly.").small().color(theme::TEXT_DIM));
                }
                slider(ui, "Rear", &mut s.rear, 0.001..=0.5);
                if s.cage.is_none() {
                    ui.checkbox(&mut s.average_normals, "Average normals").on_hover_text(
                        "Rays search along normals averaged over every face at a vertex, so the cage stays closed over hard edges: no gaps or doubled detail along them. Off: each vertex's own normal.",
                    );
                }
                ui.checkbox(&mut s.ignore_backfaces, "Ignore backfaces").on_hover_text(
                    "Rays skip faces they reach from behind (the inside of a neighboring part, the far side of a thin one). Where that leaves nothing, they count them after all, so flipped faces still bake.",
                );
                ui.checkbox(&mut s.fill_misses, "Fill misses from the nearest surface").on_hover_text(
                    "A texel whose rays find nothing takes the high poly's nearest point instead of staying flat, so the maps have no holes. The report counts them.",
                );
                if s.builds_cage() || (s.cage.is_none() && !s.average_normals) {
                    let painted = s.cage_paint.is_some();
                    let probed = self.cage_preview.probe.as_ref().is_some_and(|p| p.has_high);
                    cage_paint::cage_brush_ui(ui, &mut self.cage_brush, painted, probed);
                }
                let c = &mut self.cage_preview;
                ui.checkbox(&mut c.show, theme::with_icon(icons::CUBE_TRANSPARENT, "Show cage in the 3D view"))
                    .on_hover_text("The shell the bake rays start from, colored by what they find from each low-poly vertex.");
                if c.show {
                    if c.job.is_some() {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label(egui::RichText::new("Probing the high poly…").small().color(theme::TEXT_DIM));
                        });
                    } else if let Some((_, e)) = &c.failed {
                        ui.label(egui::RichText::new(e).small().color(Color32::from_rgb(255, 110, 90)));
                    } else if let Some(st) = c.stats {
                        ui.horizontal_wrapped(|ui| {
                            ui.spacing_mut().item_spacing.x = 10.0;
                            for (status, label, share) in [
                                (crate::cage::Status::Ok, "covered", st.ok),
                                (crate::cage::Status::Tight, "tight", st.tight),
                                (crate::cage::Status::Missed, "missed", st.missed),
                                (crate::cage::Status::Overlap, "overlapping", st.overlap),
                            ] {
                                let [r, g, b] = status.color();
                                ui.label(egui::RichText::new(format!("■ {label} {:.1}%", share * 100.0)).small().color(Color32::from_rgb(r, g, b)));
                            }
                        });
                    }
                }
                ui.label(egui::RichText::new("Red: the cage misses the high poly, raise Front/Rear. Amber: close to the limit. Magenta: another part is reached first and bleeds in; lower Front, load a cage or match by name.").small().color(theme::TEXT_DIM));
            }
            labeled(ui, "ID from", |ui| {
                ui.radio_value(&mut s.id_source, IdSource::MeshParts, "Mesh parts");
                ui.radio_value(&mut s.id_source, IdSource::Materials, "Materials");
                ui.radio_value(&mut s.id_source, IdSource::VertexColors, "Vertex colors");
            });
            if s.id_source == IdSource::VertexColors {
                if self.mesh.tri_vertex_color.is_some() {
                    ui.label(egui::RichText::new("Vertex colors detected. Bake the ID map to use the imported colors in ID masks.")
                        .small().color(theme::ACCENT));
                    ui.label(egui::RichText::new("The 3D viewport displays the imported vertex colors while this source is selected.")
                        .small().color(theme::TEXT_DIM));
                } else {
                    ui.label(egui::RichText::new("No vertex colors were found on the imported mesh. Assign a vertex color attribute in Blender, export FBX/OBJ with colors, then reimport.")
                        .small().color(Color32::from_rgb(255, 125, 90)));
                }
            }

            section(ui, "Maps");
            ui.horizontal_wrapped(|ui| {
                for &m in crate::bake::OPTIONAL_MAPS.iter() {
                    let bit = 1u32 << m;
                    let mut on = s.maps & bit != 0;
                    if ui.checkbox(&mut on, MAPS[m].name).on_hover_text(MAPS[m].desc).changed() {
                        s.maps ^= bit;
                    }
                }
            });
            ui.horizontal_wrapped(|ui| {
                if ui.small_button("All").clicked() {
                    s.maps = crate::bake::ALL_MAPS;
                }
                if ui.small_button("None").clicked() {
                    s.maps = crate::bake::ALL_MAPS & !crate::bake::OPTIONAL_MAPS.iter().fold(0, |a, m| a | (1 << m));
                }
                ui.label(
                    egui::RichText::new("Maps left out bake faster and stay neutral; masks that read them show a warning. Normal, height, ID and part maps are always baked.")
                        .small()
                        .color(theme::TEXT_DIM),
                );
            });

            self.custom_maps_section(ui);
            let s = &mut self.bake_settings;

            section(ui, "Quality");
            labeled(ui, "Preset", |ui| {
                egui::ComboBox::from_id_salt("bake quality").selected_text(s.quality.name()).show_ui(ui, |ui| {
                    for q in Quality::ALL {
                        if ui.selectable_label(s.quality == q, q.name()).clicked() {
                            s.apply_quality(q);
                        }
                    }
                });
            });
            labeled(ui, "Anti-aliasing", |ui| {
                ui.add(egui::Slider::new(&mut s.aa, 1..=8).custom_formatter(|v, _| format!("{0}x{0}", v as u32)))
            });
            labeled(ui, "AO rays", |ui| ui.add(egui::Slider::new(&mut s.ao_rays, 4..=512)));
            labeled(ui, "Thickness rays", |ui| ui.add(egui::Slider::new(&mut s.thickness_rays, 0..=256)));
            labeled(ui, "Sky rays", |ui| ui.add(egui::Slider::new(&mut s.sky_rays, 0..=256)));
            ui.label(
                egui::RichText::new(format!(
                    "{} rays per texel · {:.0}M rays at {}²",
                    s.total_rays_per_texel(),
                    s.total_rays_per_texel() as f64 * (self.engine.res as f64).powi(2) / 1e6,
                    self.engine.res
                ))
                .small()
                .color(theme::TEXT_DIM),
            );

            egui::CollapsingHeader::new("Map settings").default_open(false).show(ui, |ui| {
                slider(ui, "AO distance", &mut s.ao_distance, 0.02..=2.0);
                slider(ui, "Cavity distance", &mut s.cavity_distance, 0.005..=0.3);
                slider(ui, "Thickness dist.", &mut s.thickness_distance, 0.02..=2.0);
                slider(ui, "Sky cone (°)", &mut s.sky_cone_deg, 1.0..=89.0);
                slider(ui, "Curv. radius (px)", &mut s.curvature_radius, 1.0..=16.0);
                slider(ui, "Curv. contrast", &mut s.curvature_scale, 0.02..=2.0);
                slider(ui, "Edge sharpness", &mut s.edge_threshold, 0.05..=0.95);
                labeled(ui, "Edge search dist.", |ui| {
                    ui.checkbox(&mut s.edge_auto, "Auto")
                        .on_hover_text("Scale with the mesh: a tenth of its thinnest dimension.");
                    if s.edge_auto {
                        let auto = s.edge_distance_for(&self.mesh);
                        ui.label(egui::RichText::new(format!("{auto:.3}")).color(theme::TEXT_DIM));
                    } else {
                        ui.add(egui::Slider::new(&mut s.edge_max_distance, 0.005..=1.0).max_decimals(3));
                    }
                });
                slider(ui, "Hard edge angle (°)", &mut s.hard_edge_angle, 5.0..=120.0);
                ui.label(egui::RichText::new("Distances are in mesh units; the mesh is normalized to a 2-unit box.").small().color(theme::TEXT_DIM));
            });

            });

            ui.separator();
            ui.horizontal_wrapped(|ui| {
                let busy = self.bake_job.is_some();
                let sets = self.mesh.sets.len();
                let label = if sets > 1 { format!("Bake {}", self.mesh.sets[self.sets.active].name) } else { "Bake".into() };
                let bake = egui::Button::new(egui::RichText::new(theme::with_icon(icons::FIRE, &label)).strong()).fill(theme::ACCENT_DIM).min_size(vec2(120.0, 30.0));
                if ui.add_enabled(!busy, bake).on_hover_text("Bake the texture set being painted").clicked() {
                    self.pending.push(Action::Bake);
                }
                if sets > 1 {
                    let all = egui::Button::new(format!("Bake all {sets} sets")).min_size(vec2(0.0, 30.0));
                    if ui.add_enabled(!busy, all).on_hover_text("Bake every texture set, one after another").clicked() {
                        self.pending.push(Action::BakeAll);
                    }
                }
                if let Some((p, what)) = self.bake_progress() {
                    let set = match self.mesh.sets.get(self.sets.baking) {
                        Some(s) if sets > 1 => format!(" {}", s.name),
                        _ => String::new(),
                    };
                    ui.add(egui::ProgressBar::new(p).text(format!("{what}{set} {:.0}%", p * 100.0)));
                } else if self.engine.has_bake {
                    ui.label(egui::RichText::new("Painted fixes and levels are kept when you re-bake.").small().color(theme::TEXT_DIM));
                }
            });
        });
        self.show_bake = open;
    }

    fn fix_properties(&mut self, ui: &mut egui::Ui, map: usize) {
        let info = &MAPS[map];
        section(ui, "Bake fix");
        ui.label(egui::RichText::new(format!("Painting corrections on {}.", info.name)).color(Color32::WHITE));
        ui.label(
            egui::RichText::new("Paint (1) sets the value below, Eraser (2) restores the original bake. Fixes survive re-baking.")
                .small()
                .color(theme::TEXT_DIM),
        );
        if info.kind == MapKind::Normal {
            ui.label(egui::RichText::new("Painting flattens the normal: use it to remove projection artifacts.").color(theme::TEXT));
        } else {
            ui.horizontal(|ui| {
                ui.label("Value");
                ui.add(egui::Slider::new(&mut self.brush.fix_value, 0.0..=1.0));
            });
            ui.horizontal(|ui| {
                for (label, v) in [("Black", 0.0), ("50%", 0.5), ("White", 1.0)] {
                    if ui.button(label).clicked() {
                        self.brush.fix_value = v;
                    }
                }
            });
        }
        if ui.button("Done fixing").clicked() {
            self.target = StrokeTarget::Content;
        }
        self.brush_properties(ui, false);
    }

    /// A smart mask's settings; `salt` keeps several apart on one panel.
    /// `custom` names the custom bake maps the mask can read.
    fn generator_ui(
        ui: &mut egui::Ui,
        g: &mut crate::engine::Generator,
        salt: usize,
        custom: &[bake::CustomMap],
        target: (usize, Option<usize>),
        pick_target: &mut Option<(usize, Option<usize>)>,
        bake_settings: &mut BakeSettings,
        show_bake: &mut bool,
    ) -> bool {
        use crate::engine::gin;
        let mut dirty = false;
        labeled(ui, "Preset", |ui| {
            egui::ComboBox::from_id_salt(("gen", salt)).selected_text(g.kind.name()).show_ui(ui, |ui| {
                for k in GenKind::ALL {
                    if ui.selectable_label(g.kind == k, k.name()).clicked() && g.kind != k {
                        *g = new_generator(k, custom);
                        if k == GenKind::IdMask {
                            bake_settings.id_source = bake::IdSource::VertexColors;
                        }
                        dirty = true;
                    }
                }
            });
        });
        if g.kind == GenKind::CustomMap {
            dirty |= custom_map_picker(ui, g, salt, custom);
        }
        dirty |= slider(ui, "Level", &mut g.level, -1.0..=2.0);
        dirty |= slider(ui, "Contrast", &mut g.contrast, 0.001..=0.6);
        dirty |= slider(ui, "Noise breakup", &mut g.noise, 0.0..=2.0);
        dirty |= slider(ui, "Noise scale", &mut g.scale, 0.2..=20.0);
        dirty |= slider(ui, "Seed", &mut g.seed, 0.0..=100.0);
        dirty |= labeled(ui, "Invert", |ui| ui.checkbox(&mut g.invert, "").changed());
        let uses_width = g.weights[gin::EDGE] != 0.0 || g.weights[gin::CREASE] != 0.0;
        if uses_width {
            dirty |= slider(ui, "Edge width", &mut g.edge_width, 0.002..=0.5);
        }
        if g.weights[gin::ID] != 0.0 {
            dirty |= labeled(ui, "ID color", |ui| color_edit(ui, &mut g.id_color));
            dirty |= slider(ui, "ID tolerance", &mut g.id_tolerance, 0.01..=1.0);
            if ui.button(theme::with_icon(icons::EYEDROPPER, "Sample from mesh")).on_hover_text("Click a colored point on the 3D model. Its imported vertex color becomes this ID mask color.").clicked() {
                *pick_target = Some(target);
                bake_settings.id_source = bake::IdSource::VertexColors;
                *show_bake = false;
            }
            if *pick_target == Some(target) {
                ui.label(egui::RichText::new("Click a colored area in the 3D viewport. Then bake the ID map to refresh the mask preview.").small().color(theme::ACCENT));
            }
        }
        dirty | generator_inputs_ui(ui, g, custom)
    }

    // ------------------------------------------------------------------ properties

    fn properties(&mut self, ui: &mut egui::Ui) {
        if let StrokeTarget::BakeFix(m) = self.target {
            ui.label(egui::RichText::new(format!("Properties – Fix · {}", MAPS[m].name)).strong().color(Color32::WHITE));
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| self.fix_properties(ui, m));
            return;
        }
        let title = match self.engine.layers.get(self.selected) {
            None => "Properties".to_string(),
            Some(l) if self.target == StrokeTarget::Mask => format!("Properties – Mask of {}", l.name),
            Some(l) if l.is_fill() => format!("Properties – Fill · {}", l.name),
            Some(l) => format!("Properties – Paint · {}", l.name),
        };
        ui.label(egui::RichText::new(title).strong().color(Color32::WHITE));
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            let idx = self.selected;
            if idx >= self.engine.layers.len() {
                ui.label(egui::RichText::new("Select a layer.").color(theme::TEXT_DIM));
                return;
            }
            let mut dirty = false;

            section(ui, "Layer");
            {
                let layer = &mut self.engine.layers[idx];
                labeled(ui, "Name", |ui| ui.text_edit_singleline(&mut layer.name));
                labeled(ui, "Blend mode", |ui| {
                    egui::ComboBox::from_id_salt("blend").selected_text(layer.blend.name()).show_ui(ui, |ui| {
                        for b in Blend::ALL {
                            dirty |= ui.selectable_value(&mut layer.blend, b, b.name()).changed();
                        }
                    });
                });
                let mut op = layer.opacity * 100.0;
                if labeled(ui, "Opacity", |ui| ui.add(egui::Slider::new(&mut op, 0.0..=100.0).max_decimals(0).suffix("%")).changed()) {
                    layer.opacity = op / 100.0;
                    dirty = true;
                }
            }

            if self.target == StrokeTarget::Mask {
                dirty |= self.mask_properties(ui, idx);
                self.brush_properties(ui, false);
            } else {
                if self.engine.layers[idx].is_folder() {
                    section(ui, "Material graph");
                    labeled(ui, "Graph", |ui| {
                        if ui.button("Edit as graph").on_hover_text("The fill layers inside as nodes in the Material Designer (Ctrl+G)").clicked() {
                            self.pending.push(Action::EditGraph);
                        }
                    });
                    ui.label(
                        egui::RichText::new("Its fill layers show as Layer and Smart Mask nodes. Edits in the graph change the layers, and the other way round; wire your own nodes into a Layer node's mask.")
                            .small()
                            .color(theme::TEXT_DIM),
                    );
                }
                let is_fill = self.engine.layers[idx].is_fill();
                if is_fill {
                    let names: [Option<String>; crate::engine::CHANNELS] =
                        std::array::from_fn(|i| self.engine.layers[idx].images[i].as_ref().map(|im| im.name.clone()));
                    let graph_name = match &self.engine.layers[idx].content {
                        Content::Fill(f) => f.graph.map(|g| self.graphs.doc(g).map_or("(missing)".to_string(), |d| d.graph.name.clone())),
                        _ => None,
                    };
                    if let Some(name) = &graph_name {
                        section(ui, "Material graph");
                        labeled(ui, "Graph", |ui| {
                            ui.label(egui::RichText::new(name).color(Color32::WHITE));
                            if ui.button("Edit graph").on_hover_text("Ctrl+G").clicked() {
                                self.pending.push(Action::EditGraph);
                            }
                        });
                        ui.label(egui::RichText::new("The graph's Output nodes fill the channels below; uncheck a channel to leave it out.").small().color(theme::TEXT_DIM));
                        if let Some(gid) = match &self.engine.layers[idx].content {
                            Content::Fill(f) => f.graph,
                            _ => None,
                        } {
                            self.exposed_controls(ui, gid);
                        }
                    }
                    if let Content::Fill(f) = &mut self.engine.layers[idx].content {
                        section(ui, "Fill material");
                        dirty |= material_controls(ui, &mut f.mat, true);
                        if graph_name.is_none() {
                        section(ui, "Images");
                        let shelf_textures = &self.shelf.textures;
                        for (ch, label) in ["Color", "Roughness", "Metallic", "Height", "Opacity", "Emission", "Normal", "Specular"].into_iter().enumerate() {
                            // A shelf texture dropped on the row feeds the channel.
                            let row = ui.scope(|ui| labeled(ui, label, |ui| {
                                match &names[ch] {
                                    Some(name) => {
                                        ui.add(egui::Label::new(name.as_str()).truncate()).on_hover_text(name.as_str());
                                        if ui.small_button("✕").on_hover_text("Remove image").clicked() {
                                            self.pending.push(Action::SetFillImage { idx, ch, path: None });
                                        }
                                    }
                                    None => {
                                        if ui.button("Load image…").clicked()
                                            && let Some(path) = rfd::FileDialog::new().add_filter("Image", &IMAGE_EXT).pick_file()
                                        {
                                            self.pending.push(Action::SetFillImage { idx, ch, path: Some(path) });
                                        }
                                    }
                                }
                                // Any shelf texture can feed the channel.
                                ui.menu_button("Texture ▾", |ui| {
                                    ui.label(egui::RichText::new(format!("Shelf texture as {}", label.to_lowercase())).small().color(theme::TEXT_DIM));
                                    egui::ScrollArea::vertical().max_height(380.0).show(ui, |ui| {
                                        ui.set_width(4.0 * 70.0);
                                        ui.horizontal_wrapped(|ui| {
                                            ui.spacing_mut().item_spacing = vec2(6.0, 6.0);
                                            for (i, t) in crate::textures::TEXTURES.iter().enumerate() {
                                                let (r, resp) = ui.allocate_exact_size(vec2(64.0, 64.0), Sense::click());
                                                if let Some(h) = shelf_textures.get(i) {
                                                    ui.painter().image(h.id(), r, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
                                                }
                                                let stroke = if resp.hovered() { Stroke::new(1.5, theme::ACCENT) } else { Stroke::new(1.0, theme::OUTLINE) };
                                                ui.painter().rect_stroke(r, 0.0, stroke, egui::StrokeKind::Inside);
                                                if resp.on_hover_text(format!("{} · {}", t.name, t.category)).clicked() {
                                                    self.pending.push(Action::TextureToChannel { texture: i, ch });
                                                    ui.close();
                                                }
                                            }
                                        });
                                    });
                                });
                            }))
                            .response
                            .rect;
                            let drop = ui.interact(row, ui.scope_id().with(("image_drop", ch)), Sense::hover());
                            if let Some(ShelfItem::Texture(_)) = drop.dnd_hover_payload::<ShelfItem>().as_deref() {
                                ui.painter().rect_stroke(row.expand(1.0), 0.0, Stroke::new(2.0, theme::ACCENT), egui::StrokeKind::Inside);
                            }
                            if let Some(ShelfItem::Texture(texture)) = dnd_release::<ShelfItem>(&drop).as_deref() {
                                self.pending.push(Action::TextureToChannel { texture: *texture, ch });
                            }
                        }
                        if names[crate::engine::NORMAL].is_some() {
                            dirty |= slider(ui, "Normal strength", &mut f.mat.normal, 0.0..=2.0);
                        }
                        if names.iter().any(Option::is_some) {
                            labeled(ui, "Projection", |ui| {
                                egui::ComboBox::from_id_salt("projection").selected_text(f.projection.name()).show_ui(ui, |ui| {
                                    for p in Projection::ALL {
                                        dirty |= ui.selectable_value(&mut f.projection, p, p.name()).on_hover_text(p.tip()).changed();
                                    }
                                });
                            });
                            dirty |= labeled(ui, "Tiling", |ui| {
                                ui.add(egui::Slider::new(&mut f.tiling, 0.05..=64.0).logarithmic(true).max_decimals(2)).changed()
                            });
                            dirty |= slider(ui, "Rotation", &mut f.rotation, -180.0..=180.0);
                            dirty |= slider(ui, "Offset U", &mut f.offset[0], -1.0..=1.0);
                            dirty |= slider(ui, "Offset V", &mut f.offset[1], -1.0..=1.0);
                            if f.projection == Projection::Triplanar {
                                dirty |= slider(ui, "Blend sharpness", &mut f.tri_sharpness, 1.0..=16.0);
                            }
                        }
                        }
                        section(ui, "Procedural variation");
                        dirty |= slider(ui, "Color var.", &mut f.color_var, 0.0..=1.0);
                        dirty |= slider(ui, "Rough var.", &mut f.rough_var, 0.0..=1.0);
                        dirty |= slider(ui, "Height var.", &mut f.height_var, 0.0..=1.0);
                        dirty |= slider(ui, "Scale", &mut f.var_scale, 0.5..=60.0);
                        dirty |= slider(ui, "Seed", &mut f.seed, 0.0..=100.0);
                        dirty |= slider(ui, "Per-part var.", &mut f.part_var, 0.0..=1.0);
                    }
                    ui.add_space(6.0);
                    ui.label(egui::RichText::new("Fill layers cost no texture memory. Add a mask to paint where they show.").small().color(theme::TEXT_DIM));
                } else if !self.engine.layers[idx].is_folder() {
                    self.brush_properties(ui, true);
                }
            }
            if dirty {
                self.schedule_preview_update(ui.ctx());
            }
        });
    }

    /// A warning badge on a shelf swatch that reads maps the bake left out.
    fn missing_badge(ui: &egui::Ui, resp: &egui::Response, missing: u32) {
        if missing == 0 {
            return;
        }
        let at = resp.rect.right_top() + vec2(-8.0, 6.0);
        let r = Rect::from_center_size(at + vec2(-5.0, 6.0), vec2(16.0, 16.0));
        ui.painter().rect_filled(r, 0.0, Color32::from_rgb(230, 160, 40));
        ui.painter().text(r.center(), egui::Align2::CENTER_CENTER, "!", egui::FontId::proportional(12.0), Color32::BLACK);
        resp.clone().on_hover_text(format!("Needs maps that weren't baked: {}. Bake them for this to look right.", crate::engine::map_names(missing)));
    }

    /// A note under a mask that reads maps the bake left out.
    /// What a view mode is called; custom maps by their names.
    fn mode_name(&self, mode: ViewMode) -> String {
        match mode {
            ViewMode::Custom(id) => self.bake_settings.custom.iter().find(|c| c.id == id).map_or("Custom map".to_string(), |c| c.name.clone()),
            m => m.name().to_string(),
        }
    }

    fn missing_maps_note(&mut self, ui: &mut egui::Ui, idx: usize) {
        let Some(mask) = self.engine.layers.get(idx).and_then(|l| l.mask.as_ref()) else { return };
        let mut needs = mask.generator.map_or(0, |g| g.needs_maps());
        for ml in &mask.layers {
            if let crate::engine::MaskSource::Generator(g) = &ml.source {
                needs |= g.needs_maps();
            }
        }
        let missing = needs & self.engine.missing_maps();
        // Custom maps it reads that the bake didn't make (or that were removed).
        let mut gens: Vec<&crate::engine::Generator> = mask.generator.iter().collect();
        gens.extend(mask.layers.iter().filter_map(|ml| match &ml.source {
            crate::engine::MaskSource::Generator(g) => Some(g),
            _ => None,
        }));
        let mut custom: Vec<String> = gens
            .iter()
            .flat_map(|g| self.engine.missing_custom(g))
            .map(|id| self.bake_settings.custom.iter().find(|c| c.id == id).map_or("a removed custom map".to_string(), |c| c.name.clone()))
            .collect();
        custom.dedup();
        if missing == 0 && custom.is_empty() {
            return;
        }
        let mut names: Vec<String> = (missing != 0).then(|| crate::engine::map_names(missing)).into_iter().collect();
        names.extend(custom);
        egui::Frame::new().fill(Color32::from_rgb(60, 45, 20)).inner_margin(egui::Margin::same(6)).show(ui, |ui| {
            ui.label(
                egui::RichText::new(format!("This mask reads maps that weren't baked: {}.", names.join(", ")))
                    .color(Color32::from_rgb(240, 190, 90)),
            );
            if ui.button("Open the baker").clicked() {
                self.show_bake = true;
            }
        });
    }

    fn mask_properties(&mut self, ui: &mut egui::Ui, idx: usize) -> bool {
        let mut dirty = false;
        let custom = self.bake_settings.custom.clone();
        section(ui, "Mask");
        self.missing_maps_note(ui, idx);
        let Some(mask) = self.engine.layers[idx].mask.as_mut() else {
            ui.label("This layer has no mask.");
            return false;
        };
        if let Some(gid) = mask.graph {
            let name = self.graphs.doc(gid).map_or("(missing)".to_string(), |d| d.graph.name.clone());
            labeled(ui, "Graph", |ui| {
                ui.label(egui::RichText::new(name).color(Color32::WHITE));
                if ui.button("Edit graph").on_hover_text("Ctrl+G").clicked() {
                    self.pending.push(Action::EditGraph);
                }
            });
            ui.label(egui::RichText::new("A material graph renders this mask (its Output set to Mask). Painting on it is replaced when the graph changes.").small().color(theme::TEXT_DIM));
            self.exposed_controls(ui, gid);
            let layers = self.mask_layers_ui(ui, idx);
            return self.mask_filters_ui(ui, idx) | layers;
        }
        let has_paint = mask.paint.is_some();
        if let Some(g) = &mut mask.generator {
            dirty |= Self::generator_ui(ui, g, 0, &custom, (idx, None), &mut self.id_color_pick_target, &mut self.bake_settings, &mut self.show_bake);
        }
        ui.horizontal(|ui| {
            if !has_paint && ui.button("Add paint to mask").clicked() {
                self.engine.ensure_mask_paint(idx);
            }
            if ui.button("Remove mask").clicked() {
                self.pending.push(Action::RemoveMask);
            }
        });
        if self.engine.layers[idx].mask.as_ref().is_some_and(|m| m.generator.is_some()) && !self.engine.has_bake {
            ui.label(egui::RichText::new("Mesh maps are still baking; edge and dirt masks refine when done.").small().color(theme::TEXT_DIM));
        }
        dirty |= self.mask_layers_ui(ui, idx);
        dirty |= self.mask_filters_ui(ui, idx);
        dirty
    }

    /// The mask's stack: generators or flat values blended over the base
    /// mask (painting × smart mask), bottom first.
    fn mask_layers_ui(&mut self, ui: &mut egui::Ui, idx: usize) -> bool {
        use crate::engine::{MaskBlend, MaskLayer, MaskSource};
        if self.engine.layers[idx].is_folder() {
            return false;
        }
        let Some(mask) = self.engine.layers[idx].mask.as_mut() else { return false };
        section(ui, "Mask layers");
        let mut dirty = false;
        let custom = self.bake_settings.custom.clone();
        let (mut remove, mut swap) = (None, None);
        let count = mask.layers.len();
        // Top of the stack first, like the layer list.
        for i in (0..count).rev() {
            let ml = &mut mask.layers[i];
            egui::Frame::new().fill(theme::PANEL_DARK).stroke(Stroke::new(1.0, theme::OUTLINE)).inner_margin(egui::Margin::same(6)).show(ui, |ui| {
                ui.horizontal(|ui| {
                    dirty |= ui.checkbox(&mut ml.enabled, "").on_hover_text("Switch the mask layer on or off").changed();
                    let name = match &ml.source {
                        MaskSource::Generator(g) => g.kind.name(),
                        MaskSource::Fill(_) => "Fill",
                    };
                    ui.label(egui::RichText::new(name).strong().color(if ml.enabled { Color32::WHITE } else { theme::TEXT_DIM }));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button(icons::TRASH).on_hover_text("Remove").clicked() {
                            remove = Some(i);
                        }
                        if ui.add_enabled(i + 1 < count, egui::Button::new("▲").small()).on_hover_text("Move up").clicked() {
                            swap = Some(i);
                        }
                        if ui.add_enabled(i > 0, egui::Button::new("▼").small()).on_hover_text("Move down").clicked() {
                            swap = Some(i - 1);
                        }
                    });
                });
                ui.add_enabled_ui(ml.enabled, |ui| {
                    labeled(ui, "Blend", |ui| {
                        egui::ComboBox::from_id_salt(("mask blend", i)).selected_text(ml.blend.name()).show_ui(ui, |ui| {
                            for b in MaskBlend::ALL {
                                dirty |= ui.selectable_value(&mut ml.blend, b, b.name()).changed();
                            }
                        });
                    });
                    dirty |= slider(ui, "Opacity", &mut ml.opacity, 0.0..=1.0);
                    match &mut ml.source {
                        MaskSource::Generator(g) => dirty |= Self::generator_ui(ui, g, 100 + i, &custom, (idx, Some(i)), &mut self.id_color_pick_target, &mut self.bake_settings, &mut self.show_bake),
                        MaskSource::Fill(v) => dirty |= slider(ui, "Value", v, 0.0..=1.0),
                    }
                });
            });
            ui.add_space(3.0);
        }
        if let Some(i) = remove {
            mask.layers.remove(i);
            dirty = true;
        }
        if let Some(i) = swap {
            mask.layers.swap(i, i + 1);
            dirty = true;
        }
        ui.menu_button(theme::with_icon(icons::STACK, "Add mask layer ▾"), |ui| {
            for k in GenKind::ALL {
                if ui.button(k.name()).clicked() {
                    mask.layers.push(MaskLayer { source: MaskSource::Generator(new_generator(k, &custom)), blend: MaskBlend::Multiply, opacity: 1.0, enabled: true });
                    dirty = true;
                    ui.close();
                }
            }
            ui.separator();
            if ui.button("Fill (one value)").clicked() {
                mask.layers.push(MaskLayer { source: MaskSource::Fill(1.0), blend: MaskBlend::Normal, opacity: 1.0, enabled: true });
                dirty = true;
                ui.close();
            }
        });
        if count == 0 {
            ui.label(egui::RichText::new("Stack more smart masks on this one: each blends over the mask below it (Multiply keeps only where both are, Add joins them, Subtract cuts one out).").small().color(theme::TEXT_DIM));
        }
        dirty
    }

    /// The mask's filter stack: add, reorder, switch off, remove and tune
    /// filters that apply after painting, smart mask and graph.
    fn mask_filters_ui(&mut self, ui: &mut egui::Ui, idx: usize) -> bool {
        use crate::engine::{FilterKind, MaskFilter};
        if self.engine.layers[idx].is_folder() {
            return false;
        }
        let Some(mask) = self.engine.layers[idx].mask.as_mut() else { return false };
        section(ui, "Mask filters");
        let mut dirty = false;
        let (mut remove, mut up) = (None, None);
        let count = mask.filters.len();
        for (i, f) in mask.filters.iter_mut().enumerate() {
            egui::Frame::new().fill(theme::PANEL_DARK).stroke(Stroke::new(1.0, theme::OUTLINE)).inner_margin(egui::Margin::same(6)).show(ui, |ui| {
                ui.horizontal(|ui| {
                    dirty |= ui.checkbox(&mut f.enabled, "").on_hover_text("Switch the filter on or off").changed();
                    ui.label(egui::RichText::new(f.kind.name()).strong().color(if f.enabled { Color32::WHITE } else { theme::TEXT_DIM }))
                        .on_hover_text(f.kind.tip());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button(icons::TRASH).on_hover_text("Remove").clicked() {
                            remove = Some(i);
                        }
                        if ui.add_enabled(i > 0, egui::Button::new("▲").small()).on_hover_text("Apply earlier").clicked() {
                            up = Some(i);
                        }
                        if ui.add_enabled(i + 1 < count, egui::Button::new("▼").small()).on_hover_text("Apply later").clicked() {
                            up = Some(i + 1);
                        }
                    });
                });
                ui.add_enabled_ui(f.enabled, |ui| {
                    for (k, (name, lo, hi)) in f.kind.params().iter().enumerate() {
                        let v = match k {
                            0 => &mut f.a,
                            1 => &mut f.b,
                            _ => &mut f.c,
                        };
                        dirty |= slider(ui, name, v, *lo..=*hi);
                    }
                });
            });
            ui.add_space(3.0);
        }
        if let Some(i) = remove {
            mask.filters.remove(i);
            dirty = true;
        }
        if let Some(i) = up {
            mask.filters.swap(i - 1, i);
            dirty = true;
        }
        ui.menu_button(theme::with_icon(icons::MAGIC_WAND, "Add filter ▾"), |ui| {
            for k in FilterKind::ALL {
                if ui.button(k.name()).on_hover_text(k.tip()).clicked() {
                    mask.filters.push(MaskFilter::new(k));
                    dirty = true;
                    ui.close();
                }
            }
        });
        if mask.filters.is_empty() {
            ui.label(egui::RichText::new("Filters run in order on the finished mask: levels, blur, warp, noise breakup and more.").small().color(theme::TEXT_DIM));
        }
        dirty
    }

    /// The current tip with a grid of all tips to choose from.
    fn tip_picker(&mut self, ui: &mut egui::Ui) {
        let name = self.brush.tip.map_or("Round", |i| crate::brushes::BRUSH_ALPHAS[i].name);
        labeled(ui, "Tip", |ui| {
            let tex = self.brush.tip.map(|i| self.shelf.brushes[i].id());
            let (rect, resp) = ui.allocate_exact_size(vec2(40.0, 40.0), Sense::click());
            ui.painter().rect_filled(rect, theme::RADIUS, theme::PANEL_DARK);
            match tex {
                Some(t) => {
                    ui.painter().image(t, rect.shrink(3.0), Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
                }
                None => {
                    ui.painter().circle_filled(rect.center(), 13.0, Color32::WHITE);
                }
            }
            let popup = egui::Popup::menu(&resp).close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside);
            ui.label(name);
            popup.show(|ui| {
                ui.set_max_width(300.0);
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = vec2(3.0, 3.0);
                    let cell = |ui: &mut egui::Ui, tex: Option<egui::TextureId>, selected: bool, tip: &str| {
                        let (rect, resp) = ui.allocate_exact_size(vec2(44.0, 44.0), Sense::click());
                        let bg = if selected { theme::ACCENT_DIM } else if resp.hovered() { theme::WIDGET_HOVER } else { theme::PANEL_DARK };
                        ui.painter().rect_filled(rect, theme::RADIUS, bg);
                        match tex {
                            Some(t) => {
                                ui.painter().image(t, rect.shrink(4.0), Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
                            }
                            None => {
                                ui.painter().circle_filled(rect.center(), 15.0, Color32::WHITE);
                            }
                        }
                        resp.on_hover_text(tip).clicked()
                    };
                    if cell(ui, None, self.brush.tip.is_none(), "Round") {
                        self.pending.push(Action::SetBrushTip(None));
                    }
                    for (i, b) in crate::brushes::BRUSH_ALPHAS.iter().enumerate() {
                        if cell(ui, Some(self.shelf.brushes[i].id()), self.brush.tip == Some(i), b.name) {
                            self.pending.push(Action::SetBrushTip(Some(i)));
                        }
                    }
                });
            });
        });
    }

    fn brush_properties(&mut self, ui: &mut egui::Ui, content: bool) {
        let _ = content;
        section(ui, if self.tool == Tool::Eraser { "Eraser" } else { "Brush" });
        slider(ui, "Size", &mut self.brush.size, 2.0..=800.0);
        slider(ui, "Hardness", &mut self.brush.hardness, 0.0..=0.99);
        slider(ui, "Flow", &mut self.brush.flow, 0.01..=1.0);
        slider(ui, "Stroke opacity", &mut self.brush.opacity, 0.01..=1.0);
        slider(ui, "Spacing", &mut self.brush.spacing, 0.02..=1.0);
        section(ui, "Brush tip");
        self.tip_picker(ui);
        if self.brush.tip.is_some() {
            slider(ui, "Angle", &mut self.brush.angle, -180.0..=180.0);
            labeled(ui, "Follow stroke", |ui| ui.checkbox(&mut self.brush.follow, "").on_hover_text("Turn the tip along the stroke direction"));
            slider(ui, "Angle jitter", &mut self.brush.angle_jitter, 0.0..=1.0);
        }
        slider(ui, "Size jitter", &mut self.brush.size_jitter, 0.0..=1.0);
        slider(ui, "Scatter", &mut self.brush.scatter, 0.0..=2.0);

        section(ui, "Projection");
        labeled(ui, "Mode", |ui| projection_combo(ui, "projection_props", &mut self.brush.projection));
        if self.brush.projection == BrushProjection::Screen {
            labeled(ui, "Paint through", |ui| {
                ui.checkbox(&mut self.brush.paint_through, "").on_hover_text("Also paint surfaces hidden behind others")
            });
        }
        labeled(ui, "Angle limit", |ui| {
            ui.add(egui::Slider::new(&mut self.brush.angle_limit, 10.0..=90.0).max_decimals(0).suffix("°"))
                .on_hover_text("Paint fades out on surfaces turned further than this from the brush, so it doesn't smear on slopes")
        });

        section(ui, "Stroke");
        labeled(ui, "Stabilizer", |ui| {
            ui.add(egui::Slider::new(&mut self.brush.stabilize, 0.0..=200.0).max_decimals(0))
                .on_hover_text("Lazy mouse: the brush trails the cursor on a string for smooth lines")
        });
        labeled(ui, "Pen pressure", |ui| {
            ui.checkbox(&mut self.brush.pressure_size, "Size");
            ui.checkbox(&mut self.brush.pressure_flow, "Flow");
        });
        ui.label(egui::RichText::new("Shift+click draws a straight line from the last stroke.").small().color(theme::TEXT_DIM));

        section(ui, "Grain");
        labeled(ui, "Texture", |ui| {
            let name = self.brush.grain.map_or("None", |i| crate::textures::TEXTURES[i].name);
            egui::ComboBox::from_id_salt("brush_grain").selected_text(name).height(400.0).show_ui(ui, |ui| {
                ui.selectable_value(&mut self.brush.grain, None, "None");
                for (i, t) in crate::textures::TEXTURES.iter().enumerate() {
                    ui.selectable_value(&mut self.brush.grain, Some(i), t.name);
                }
            })
        });
        if self.brush.grain.is_some() {
            slider(ui, "Strength", &mut self.brush.grain_strength, 0.0..=1.0);
            labeled(ui, "Scale", |ui| {
                ui.add(egui::Slider::new(&mut self.brush.grain_scale, 0.25..=64.0).logarithmic(true).max_decimals(2))
                    .on_hover_text("Repeats across the mesh (or the UV square with the UV projection)")
            });
            labeled(ui, "Invert", |ui| ui.checkbox(&mut self.brush.grain_invert, ""));
        } else {
            ui.label(egui::RichText::new("A texture fixed to the surface that breaks up the paint: scratches, dust, chipping.").small().color(theme::TEXT_DIM));
        }

        if content {
            section(ui, if self.tool == Tool::Eraser { "Erase channels" } else { "Material" });
            material_controls(ui, &mut self.brush.mat, self.tool != Tool::Eraser);
            if self.tool != Tool::Eraser {
                ui.label(egui::RichText::new("Tip: right-click a material in the shelf → Use as brush material.").small().color(theme::TEXT_DIM));
            }
        } else if self.tool != Tool::Eraser && self.target == StrokeTarget::Mask {
            section(ui, "Mask value");
            ui.horizontal(|ui| {
                if ui.button("Black").clicked() {
                    self.brush.mask_value = 0.0;
                }
                if ui.button("White").clicked() {
                    self.brush.mask_value = 1.0;
                }
                ui.add(egui::Slider::new(&mut self.brush.mask_value, 0.0..=1.0));
            });
        }
    }

    // ------------------------------------------------------------------ shelf

    fn shelf_panel(&mut self, ui: &mut egui::Ui) {
        // A narrow shelf keeps its room for thumbnails: categories as icons
        // (names on hover), a header without the hint, - / + for the size.
        let compact = ui.available_width() < SHELF_COMPACT;
        egui::Panel::left("shelf_cats")
            .exact_size(if compact { 40.0 } else { 176.0 })
            .resizable(false)
            .frame(egui::Frame::new().fill(theme::PANEL_DARK).inner_margin(egui::Margin::symmetric(if compact { 4 } else { 8 }, 8)))
            .show(ui, |ui| {
                if !compact {
                    ui.label(egui::RichText::new("SHELF").small().color(theme::TEXT_DIM));
                }
                ui.spacing_mut().item_spacing.y = 2.0;
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                for (cat, icon, name, count) in [
                    (ShelfCat::SmartMaterials, icons::SPARKLE, "Smart materials", presets::SMART_MATERIALS.len() + self.user_smarts.len()),
                    (ShelfCat::Materials, icons::SPHERE, "Materials", presets::MATERIALS.len()),
                    (ShelfCat::Generators, icons::MAGIC_WAND, "Generators", GenKind::ALL.len()),
                    (ShelfCat::Brushes, icons::PAINT_BRUSH, "Brushes", crate::brushes::BRUSH_ALPHAS.len()),
                    (ShelfCat::Textures, icons::IMAGES, "Textures", crate::textures::TEXTURES.len()),
                    (ShelfCat::Graphs, icons::SPARKLE, "Graphs", self.graphs.shelf.len()),
                    (ShelfCat::Downloaded, icons::DOWNLOAD_SIMPLE, "Downloaded", self.library.local.len()),
                ] {
                    let on = self.shelf.cat == cat;
                    let color = if on { Color32::WHITE } else { theme::TEXT };
                    let resp = if compact {
                        ui.add_sized([32.0, 26.0], theme::selectable(on, egui::RichText::new(icon).size(15.0).color(color)))
                            .on_hover_text(format!("{name} ({count})"))
                    } else {
                        let resp = ui.add_sized([158.0, 21.0], theme::selectable(on, egui::RichText::new(format!("{}  {name}", icon)).color(color)));
                        ui.painter().text(
                            resp.rect.right_center() - vec2(8.0, 0.0),
                            egui::Align2::RIGHT_CENTER,
                            count.to_string(),
                            egui::FontId::proportional(10.0),
                            theme::TEXT_DIM,
                        );
                        resp
                    };
                    if resp.clicked() {
                        self.shelf.cat = cat;
                    }
                }
                });
            });
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(theme::PANEL).inner_margin(egui::Margin::symmetric(6, 4)))
            .show(ui, |ui| {
                // Wraps rather than overlapping when even the compact header doesn't fit.
                ui.horizontal_wrapped(|ui| {
                    let label = if compact { icons::GLOBE.to_string() } else { theme::with_icon(icons::GLOBE, "Get more assets") };
                    let browse = egui::Button::new(egui::RichText::new(label).color(Color32::WHITE)).fill(theme::ACCENT.gamma_multiply(0.55));
                    if ui.add(browse).on_hover_text("Get more assets: thousands of free materials from Poly Haven and ambientCG, and graphs shared by the community").clicked() {
                        self.library.open = true;
                    }
                    ui.add(egui::TextEdit::singleline(&mut self.shelf.search).hint_text("Search…").desired_width(if compact { 110.0 } else { 200.0 }));
                    let hint = match self.shelf.cat {
                        ShelfCat::SmartMaterials => "Double-click or drag onto a view to add a smart material (a folder of layers). Save your own: right-click a folder in Layers → Save as smart material.",
                        ShelfCat::Materials => "Double-click or drag onto a view to add as a fill layer. Right-click for more.",
                        ShelfCat::Generators => "Double-click or drag onto a view to mask the selected layer.",
                        ShelfCat::Brushes => "Double-click or drag onto a view to paint with this brush tip.",
                        ShelfCat::Textures => "Double-click or drag to add as a height fill layer. Right-click to put it on the selected fill layer.",
                        ShelfCat::Graphs => "Material graphs you saved or downloaded. Double-click or drag onto a view to add as a graph fill layer. Right-click to share.",
                        ShelfCat::Downloaded => "Free materials from the asset library. Double-click or drag onto a view to add as a fill layer.",
                    };
                    if matches!(self.shelf.cat, ShelfCat::SmartMaterials | ShelfCat::Materials | ShelfCat::Downloaded) {
                        let text = self.shelf.kind.map_or("All types", |k| k.name());
                        egui::ComboBox::from_id_salt("shelf type").selected_text(text).width(if compact { 96.0 } else { 130.0 }).show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.shelf.kind, None, "All types").on_hover_text("Everything, grouped by type");
                            for k in presets::MaterialType::ALL {
                                ui.selectable_value(&mut self.shelf.kind, Some(k), k.name());
                            }
                        });
                    }
                    if compact {
                        // Thumbnail size in steps; the hint stays on the hover.
                        let step = |size: f32, by: f32| (size + by).clamp(ICON_MIN, ICON_MAX);
                        if ui.small_button("-").on_hover_text("Smaller thumbnails (Ctrl+scroll over the shelf)").clicked() {
                            self.shelf.icon_size = step(self.shelf.icon_size, -16.0);
                        }
                        if ui.small_button("+").on_hover_text("Bigger thumbnails (Ctrl+scroll over the shelf)").clicked() {
                            self.shelf.icon_size = step(self.shelf.icon_size, 16.0);
                        }
                        ui.label(egui::RichText::new(icons::INFO).color(theme::TEXT_DIM)).on_hover_text(hint);
                    } else {
                    // The size slider at the right edge first; the hint fits in what's left,
                    // shortened with an ellipsis (whole on hover) instead of running under it.
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add(egui::Slider::new(&mut self.shelf.icon_size, ICON_MIN..=ICON_MAX).show_value(false))
                            .on_hover_text("Thumbnail size (Ctrl+scroll over the shelf)");
                        ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                            ui.add(egui::Label::new(egui::RichText::new(hint).small().color(theme::TEXT_DIM)).truncate()).on_hover_text(hint);
                        });
                    });
                    }
                });
                ui.add_space(2.0);
                // Ctrl+scroll arrives as zoom; it resizes the thumbnails.
                if ui.rect_contains_pointer(ui.max_rect()) {
                    let zoom = ui.input(|i| i.zoom_delta());
                    if zoom != 1.0 {
                        self.shelf.icon_size = (self.shelf.icon_size * zoom).clamp(ICON_MIN, ICON_MAX);
                    }
                }
                let icon = self.shelf.icon_size;
                let search = self.shelf.search.to_lowercase();
                let kind = self.shelf.kind;
                let wanted = |k: presets::MaterialType| kind.is_none_or(|f| f == k);
                // The type's name as a heading of its own when the type changes
                // (all types shown). It takes the full width, so the wrapped
                // layout puts it on a fresh row and starts the next items below it.
                let group = |ui: &mut egui::Ui, last: &mut Option<presets::MaterialType>, k: presets::MaterialType| {
                    if kind.is_none() && *last != Some(k) {
                        let width = ui.max_rect().width();
                        let h = if last.is_some() { 30.0 } else { 20.0 };
                        let (r, _) = ui.allocate_exact_size(vec2(width, h), Sense::hover());
                        let base = r.left_bottom() + vec2(2.0, -5.0);
                        let g = ui.painter().layout_no_wrap(k.name().to_uppercase(), egui::FontId::new(11.0, theme::heading_family()), theme::TEXT_DIM);
                        let line_y = base.y - g.size().y * 0.5;
                        let text_w = g.size().x;
                        ui.painter().galley(pos2(base.x, base.y - g.size().y), g, theme::TEXT_DIM);
                        ui.painter().line_segment(
                            [pos2(base.x + text_w + 8.0, line_y), pos2(r.right() - 4.0, line_y)],
                            Stroke::new(1.0, theme::OUTLINE),
                        );
                        *last = Some(k);
                    }
                };
                let mut share = None;
                let mut share_smart = None;
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing = vec2(4.0, 4.0);
                        match self.shelf.cat {
                            ShelfCat::Graphs => {
                                if self.graphs.shelf.is_empty() {
                                    ui.label(
                                        egui::RichText::new("No saved graphs yet. In the Material Graph window (Ctrl+G), use Save to shelf.")
                                            .color(theme::TEXT_DIM),
                                    );
                                }
                                let fallback = self.shelf.materials.first().map(|t| t.id());
                                for (i, g) in self.graphs.shelf.iter().enumerate() {
                                    if !g.name.to_lowercase().contains(&search) {
                                        continue;
                                    }
                                    let Some(tex) = g.thumb.as_ref().map(|t| t.id()).or(fallback) else { continue };
                                    let resp = Self::swatch(ui, tex, &g.name, icon);
                                    resp.dnd_set_drag_payload(ShelfItem::Graph(i));
                                    if resp.double_clicked() {
                                        self.pending.push(Action::AddOnTop(ShelfItem::Graph(i)));
                                    }
                                    resp.context_menu(|ui| {
                                        if ui.button(theme::with_icon(icons::SHARE_NETWORK, "Share to community…")).clicked() {
                                            share = Some(i);
                                            ui.close();
                                        }
                                    });
                                }
                            }
                            ShelfCat::Downloaded => {
                                self.library.ensure_local(ui.ctx());
                                if self.library.local.is_empty() {
                                    ui.label(
                                        egui::RichText::new("Nothing downloaded yet. Click Get more assets above for thousands of free materials.")
                                            .color(theme::TEXT_DIM),
                                    );
                                }
                                let fallback = self.shelf.materials.first().map(|t| t.id());
                                let kinds: Vec<presets::MaterialType> = self.library.local.iter().map(|m| presets::MaterialType::of(&m.name, None)).collect();
                                let mut order: Vec<usize> = (0..self.library.local.len())
                                    .filter(|&i| self.library.local[i].name.to_lowercase().contains(&search) && wanted(kinds[i]))
                                    .collect();
                                order.sort_by_key(|&i| kinds[i]);
                                let mut last = None;
                                for i in order {
                                    group(ui, &mut last, kinds[i]);
                                    let m = &self.library.local[i];
                                    let thumb = self.library.local_thumbs.get(i).and_then(|t| t.as_ref()).map(|t| t.id());
                                    let Some(tex) = thumb.or(fallback) else { continue };
                                    let resp = Self::swatch(ui, tex, &m.name.clone(), icon).on_hover_text(format!("{}, from {}", m.name, m.source));
                                    resp.dnd_set_drag_payload(ShelfItem::Library(i));
                                    if resp.double_clicked() {
                                        self.pending.push(Action::AddOnTop(ShelfItem::Library(i)));
                                    }
                                    resp.context_menu(|ui| self.library_material_menu(ui, i));
                                }
                            }
                            ShelfCat::SmartMaterials => {
                                // Your own first, marked with a badge.
                                let fallback = self.shelf.smart.first().map(|t| t.id());
                                for i in 0..self.user_smarts.len() {
                                    let s = &self.user_smarts[i];
                                    if !s.name.to_lowercase().contains(&search) || !wanted(presets::MaterialType::of(&s.name, None)) {
                                        continue;
                                    }
                                    let Some(tex) = s.thumb.as_ref().map(|t| t.id()).or(fallback) else { continue };
                                    let resp = Self::swatch(ui, tex, &s.name.clone(), icon);
                                    let at = resp.rect.left_top() + vec2(8.0, 6.0);
                                    let g = ui.painter().layout_no_wrap("MINE".into(), egui::FontId::proportional(8.5), Color32::WHITE);
                                    let r = Rect::from_min_size(at, g.size() + vec2(6.0, 2.0));
                                    ui.painter().rect_filled(r, 0.0, theme::DONATE.gamma_multiply(0.9));
                                    ui.painter().galley(r.min + vec2(3.0, 1.0), g, Color32::WHITE);
                                    resp.dnd_set_drag_payload(ShelfItem::UserSmart(i));
                                    if resp.double_clicked() {
                                        self.pending.push(Action::AddOnTop(ShelfItem::UserSmart(i)));
                                    }
                                    resp.context_menu(|ui| self.user_smart_menu(ui, i, &mut share_smart));
                                }
                                let mut order: Vec<usize> = (0..presets::SMART_MATERIALS.len())
                                    .filter(|&i| presets::SMART_MATERIALS[i].name.to_lowercase().contains(&search) && wanted(presets::SMART_MATERIALS[i].kind()))
                                    .collect();
                                order.sort_by_key(|&i| presets::SMART_MATERIALS[i].kind());
                                // Your own stay first, apart from the groups.
                                let mut last = None;
                                for (i, s) in order.into_iter().map(|i| (i, &presets::SMART_MATERIALS[i])) {
                                    group(ui, &mut last, s.kind());
                                    let tex = self.shelf.smart[i].id();
                                    let resp = Self::swatch(ui, tex, s.name, icon);
                                    if s.is_scanned() {
                                        // Badge: built on a photo-scanned texture set.
                                        let at = resp.rect.left_top() + vec2(8.0, 6.0);
                                        let font = egui::FontId::proportional(8.5);
                                        let g = ui.painter().layout_no_wrap("SCAN".into(), font, Color32::WHITE);
                                        let r = Rect::from_min_size(at, g.size() + vec2(6.0, 2.0));
                                        ui.painter().rect_filled(r, 0.0, theme::ACCENT.gamma_multiply(0.85));
                                        ui.painter().galley(r.min + vec2(3.0, 1.0), g, Color32::WHITE);
                                    }
                                    Self::missing_badge(ui, &resp, s.needs_maps() & self.engine.missing_maps());
                                    resp.dnd_set_drag_payload(ShelfItem::Smart(i));
                                    if resp.double_clicked() {
                                        self.pending.push(Action::AddOnTop(ShelfItem::Smart(i)));
                                    }
                                    resp.context_menu(|ui| {
                                        if ui.button("Add as layer folder").on_hover_text("A folder of fill layers with smart masks; Ctrl+G shows them as a graph you can edit too").clicked() {
                                            self.pending.push(Action::AddOnTop(ShelfItem::Smart(i)));
                                            ui.close();
                                        }
                                        if ui.button("Add as one graph layer").on_hover_text("One layer driven by a graph; its settings as material parameters in the layer properties").clicked() {
                                            self.pending.push(Action::AddOnTop(ShelfItem::SmartGraph(i)));
                                            ui.close();
                                        }
                                    });
                                }
                            }
                            ShelfCat::Materials => {
                                let mut order: Vec<usize> =
                                    (0..presets::MATERIALS.len()).filter(|&i| presets::MATERIALS[i].name.to_lowercase().contains(&search) && wanted(presets::MATERIALS[i].kind())).collect();
                                order.sort_by_key(|&i| presets::MATERIALS[i].kind());
                                let mut last = None;
                                for (i, m) in order.into_iter().map(|i| (i, &presets::MATERIALS[i])) {
                                    group(ui, &mut last, m.kind());
                                    let tex = self.shelf.materials[i].id();
                                    let resp = Self::swatch(ui, tex, m.name, icon);
                                    resp.dnd_set_drag_payload(ShelfItem::Material(i));
                                    if resp.double_clicked() {
                                        self.pending.push(Action::AddOnTop(ShelfItem::Material(i)));
                                    }
                                    resp.context_menu(|ui| {
                                        if ui.button("Add as fill layer").clicked() {
                                            self.pending.push(Action::AddOnTop(ShelfItem::Material(i)));
                                            ui.close();
                                        }
                                        if ui.button("Use as brush material").clicked() {
                                            let f = m.fill();
                                            let mut enabled = self.brush.mat.enabled;
                                            enabled[..3].fill(true);
                                            self.brush.mat = MatValues {
                                                enabled,
                                                height: self.brush.mat.height,
                                                opacity: self.brush.mat.opacity,
                                                emission: self.brush.mat.emission,
                                                ..f.mat
                                            };
                                            self.set_tool(Tool::Paint);
                                            ui.close();
                                        }
                                    });
                                }
                            }
                            ShelfCat::Brushes => {
                                for (i, b) in crate::brushes::BRUSH_ALPHAS.iter().enumerate() {
                                    if !b.name.to_lowercase().contains(&search) {
                                        continue;
                                    }
                                    let resp = Self::swatch(ui, self.shelf.brushes[i].id(), b.name, icon);
                                    if self.brush.tip == Some(i) {
                                        ui.painter().rect_stroke(resp.rect.shrink(2.0), theme::RADIUS, Stroke::new(1.5, theme::ACCENT), egui::StrokeKind::Inside);
                                    }
                                    resp.dnd_set_drag_payload(ShelfItem::Brush(i));
                                    if resp.double_clicked() {
                                        self.pending.push(Action::SetBrushTip(Some(i)));
                                    }
                                }
                            }
                            ShelfCat::Textures => {
                                for (i, t) in crate::textures::TEXTURES.iter().enumerate() {
                                    if !t.name.to_lowercase().contains(&search) && !t.category.to_lowercase().contains(&search) {
                                        continue;
                                    }
                                    let resp = Self::swatch(ui, self.shelf.textures[i].id(), t.name, icon);
                                    resp.dnd_set_drag_payload(ShelfItem::Texture(i));
                                    if resp.double_clicked() {
                                        self.pending.push(Action::AddOnTop(ShelfItem::Texture(i)));
                                    }
                                    resp.context_menu(|ui| {
                                        if ui.button("Add as height fill layer").clicked() {
                                            self.pending.push(Action::AddOnTop(ShelfItem::Texture(i)));
                                            ui.close();
                                        }
                                        ui.separator();
                                        for (ch, label) in [(3, "height"), (0, "color"), (1, "roughness"), (2, "metallic")] {
                                            if ui.button(format!("Use as {label} on the selected fill layer")).clicked() {
                                                self.pending.push(Action::TextureToChannel { texture: i, ch });
                                                ui.close();
                                            }
                                        }
                                    });
                                }
                            }
                            ShelfCat::Generators => {
                                for (i, g) in GenKind::ALL.iter().enumerate() {
                                    if !g.name().to_lowercase().contains(&search) {
                                        continue;
                                    }
                                    let tex = self.shelf.generators[i].id();
                                    let resp = Self::swatch(ui, tex, g.name(), icon);
                                    Self::missing_badge(ui, &resp, g.default_generator().needs_maps() & self.engine.missing_maps());
                                    resp.dnd_set_drag_payload(ShelfItem::Generator(*g));
                                    if resp.double_clicked() {
                                        self.pending.push(Action::AddGenerator(*g));
                                    }
                                }
                            }
                        }
                    });
                });
                if let Some(i) = share {
                    self.share_graph(ui.ctx(), i);
                }
                if let Some(path) = share_smart {
                    self.share_smart(ui.ctx(), &path);
                }
            });
    }

    /// A shelf thumbnail `size` points wide with its name below; hovering
    /// shows it at twice the size.
    fn swatch(ui: &mut egui::Ui, tex: egui::TextureId, name: &str, size: f32) -> egui::Response {
        let named = size >= NAMED_MIN;
        let (rect, resp) = ui.allocate_exact_size(vec2(size + 12.0, size + if named { 22.0 } else { 8.0 }), Sense::click_and_drag());
        let p = ui.painter();
        if resp.hovered() {
            p.rect_filled(rect, 0.0, theme::WIDGET_HOVER);
        }
        let img = Rect::from_min_size(rect.min + vec2(6.0, 4.0), vec2(size, size));
        p.image(tex, img, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
        // One line, cut with "…"; the hover shows the full name.
        if named {
            let galley = theme::elided(p, name, egui::FontId::proportional(10.5), theme::TEXT, size + 8.0);
            let pos = pos2(rect.center().x - galley.size().x * 0.5, img.max.y + 3.0);
            p.galley(pos, galley, theme::TEXT);
        }
        if resp.dragged() {
            // The thumbnail follows the pointer while it's dragged to a view.
            if let Some(p) = resp.ctx.pointer_interact_pos() {
                let ghost = Rect::from_center_size(p, vec2(size, size));
                let painter = resp.ctx.layer_painter(egui::LayerId::new(egui::Order::Tooltip, resp.id.with("drag ghost")));
                painter.image(tex, ghost, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::from_white_alpha(220));
            }
            return resp;
        }
        resp.on_hover_ui(|ui| {
            ui.add(egui::Image::new((tex, vec2(size * 2.0, size * 2.0))));
            ui.label(name);
        })
    }

    // ------------------------------------------------------------------ viewports

    fn stamp_uniform(&self, kind: ViewKind) -> StampUniform {
        let ppp_radius = self.brush.size * 0.5 * self.ppp;
        let brush = [ppp_radius.max(0.5), self.brush.hardness.min(0.99), self.brush.flow, self.brush.symmetry as f32];
        let clip = [NEAR, FAR, if self.brush.tip.is_some() { 1.0 } else { 0.0 }, 0.0];
        let proj = [
            match self.brush.projection {
                BrushProjection::Screen => 0.0,
                BrushProjection::Surface => 1.0,
                BrushProjection::Uv => 2.0,
            },
            if self.brush.paint_through { 1.0 } else { 0.0 },
            self.brush.angle_limit.to_radians().cos(),
            0.0,
        ];
        match kind {
            ViewKind::V3d => {
                let [w, h] = self.views.view3d.size;
                let vp = self.cam.view_proj(w as f32 / h as f32);
                StampUniform {
                    view_proj: vp.to_cols_array_2d(),
                    cam_pos: self.cam.eye().extend(1.0).to_array(),
                    view: [w as f32, h as f32, 0.0, 0.0],
                    view2d: [0.0; 4],
                    brush,
                    clip,
                    proj,
                }
            }
            ViewKind::V2d => {
                let [w, h] = self.views.view2d.size;
                StampUniform {
                    view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
                    cam_pos: [0.0; 4],
                    view: [w as f32, h as f32, 1.0, 0.0],
                    view2d: [self.cam2d.zoom, self.cam2d.pan.x, self.cam2d.pan.y, 0.0],
                    brush,
                    clip,
                    proj,
                }
            }
        }
    }

    // ------------------------------------------------------------------ shelf textures

    /// A shelf texture rendered at working resolution as a fill image.
    fn texture_image(&self, i: usize) -> Result<crate::engine::FillImage, String> {
        use image::ImageEncoder;
        const SIZE: u32 = 1024;
        let t = &crate::textures::TEXTURES[i];
        let mut png = Vec::new();
        image::codecs::png::PngEncoder::new(&mut png)
            .write_image(&t.render(SIZE), SIZE, SIZE, image::ExtendedColorType::L8)
            .map_err(|e| format!("can't encode {}: {e}", t.name))?;
        self.engine.decode_fill_image(format!("{}.png", t.name), Arc::new(png))
    }

    /// A fill layer whose height comes from a shelf texture, projected
    /// tri-planar so it tiles seamlessly on any mesh.
    fn add_texture_layer(&mut self, i: usize, at: usize, parent: Option<u64>) {
        let t = &crate::textures::TEXTURES[i];
        let img = match self.texture_image(i) {
            Ok(img) => img,
            Err(e) => {
                self.status = e;
                return;
            }
        };
        let fill = FillParams {
            mat: MatValues { enabled: [false, false, false, true, false, false, false, false], height: 0.5, ..FillParams::default().mat },
            projection: Projection::Triplanar,
            tiling: 2.0,
            ..FillParams::default()
        };
        self.undo.begin_group(format!("Add {}", t.name));
        let idx = self.engine.add_fill_layer(at, parent, t.name.to_string(), fill, &mut self.undo);
        self.engine.set_fill_image(idx, 3, Some(img), &mut self.undo);
        self.undo.end_group();
        self.selected = idx;
        self.target = StrokeTarget::Content;
        self.status = format!("Added '{}' as height. Change the channel, projection and tiling in the layer properties.", t.name);
    }

    /// Keeps the engine's brush tip texture in step with the brush.
    fn sync_brush_tip(&mut self) {
        if self.uploaded_tip == Some(self.brush.tip) {
            return;
        }
        match self.brush.tip {
            Some(i) => {
                const SIZE: u32 = 256;
                let px = crate::brushes::BRUSH_ALPHAS[i].render(SIZE);
                self.engine.set_brush_alpha(Some((&px, SIZE)));
            }
            None => self.engine.set_brush_alpha(None),
        }
        self.uploaded_tip = Some(self.brush.tip);
    }

    /// Keeps the engine's brush grain texture in step with the brush.
    fn sync_brush_grain(&mut self) {
        if self.uploaded_grain == Some(self.brush.grain) {
            return;
        }
        match self.brush.grain {
            Some(i) => {
                const SIZE: u32 = 512;
                let px = crate::textures::TEXTURES[i].render(SIZE);
                self.engine.set_brush_grain(Some((&px, SIZE)));
            }
            None => self.engine.set_brush_grain(None),
        }
        self.uploaded_grain = Some(self.brush.grain);
    }

    /// The brush grain as `BrushPaint::grain`.
    fn grain_params(&self) -> [f32; 4] {
        let b = &self.brush;
        if b.grain.is_none() {
            return [0.0; 4];
        }
        let uv = b.projection == BrushProjection::Uv;
        let (lo, hi) = self.engine.bounds();
        let tiling = if uv { b.grain_scale } else { b.grain_scale / (hi - lo).max_element().max(1e-6) };
        [b.grain_strength, tiling, if uv { 1.0 } else { 0.0 }, if b.grain_invert { 1.0 } else { 0.0 }]
    }

    /// Identifies a view's camera, so a line anchor goes stale when it moves.
    fn view_key(&self, kind: ViewKind) -> [f32; 8] {
        match kind {
            ViewKind::V3d => {
                let c = &self.cam;
                let [w, h] = self.views.view3d.size;
                [c.yaw, c.pitch, c.dist, c.target.x, c.target.y, c.target.z, c.fov_y, w as f32 / h.max(1) as f32]
            }
            ViewKind::V2d => {
                let c = &self.cam2d;
                let [w, h] = self.views.view2d.size;
                [c.zoom, c.pan.x, c.pan.y, w as f32, h as f32, 0.0, 0.0, 0.0]
            }
        }
    }

    /// One stamp at screen position `p`: tip angle (optionally along the
    /// stroke direction `dir`) plus the brush's size, angle and position
    /// jitter, and pen pressure. Returns the stamp and its flow scale.
    fn make_stamp(&mut self, p: Vec2, dir: Option<Vec2>, pressure: f32) -> ([f32; 4], f32) {
        let radius = self.brush.size * 0.5 * self.ppp;
        let b = &mut self.brush;
        let mut rnd = || {
            b.rng ^= b.rng << 13;
            b.rng ^= b.rng >> 17;
            b.rng ^= b.rng << 5;
            b.rng as f32 / u32::MAX as f32
        };
        let (r1, r2, r3, r4) = (rnd(), rnd(), rnd(), rnd());
        let scale = (1.0 - b.size_jitter * r1) * if b.pressure_size { pressure } else { 1.0 };
        let offset = Vec2::new(r2 - 0.5, r3 - 0.5) * 2.0 * b.scatter * radius;
        let mut angle = b.angle.to_radians() + (r4 - 0.5) * std::f32::consts::TAU * b.angle_jitter;
        if b.follow {
            if let Some(d) = dir {
                angle += d.y.atan2(d.x);
            }
        }
        ([p.x + offset.x, p.y + offset.y, scale, angle], if b.pressure_flow { pressure } else { 1.0 })
    }

    // ------------------------------------------------------------------ cage preview

    /// Keeps the cage overlay in step with the bake settings: probes the high
    /// poly on a worker thread when the inputs change, recolors when the
    /// distances change, and hides it with the bake window.
    fn update_cage_preview(&mut self, ctx: &egui::Context) {
        self.apply_cage_action();
        let c = &mut self.cage_preview;
        if let Some((key, rx)) = &c.job {
            match rx.try_recv() {
                Ok(Ok(probe)) => {
                    c.probe = Some(Arc::new(probe));
                    c.job = None;
                    c.drawn = None;
                }
                Ok(Err(e)) => {
                    c.failed = Some((key.clone(), e));
                    c.job = None;
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => c.job = None,
            }
        }
        // A cage only exists for high-poly bakes.
        if !(self.show_bake && c.show && self.bake_settings.high_poly.is_some()) {
            if self.overlays.is_set(OverlaySlot::Cage) {
                self.overlays.clear(OverlaySlot::Cage);
                ctx.request_repaint();
            }
            c.drawn = None;
            return;
        }
        let key = ProbeKey::new(&self.mesh, &self.bake_settings);
        let current = c.probe.as_ref().is_some_and(|p| p.key == key);
        if !current {
            if self.overlays.is_set(OverlaySlot::Cage) {
                self.overlays.clear(OverlaySlot::Cage);
                ctx.request_repaint();
            }
            c.drawn = None;
            c.stats = None;
            let running = c.job.as_ref().is_some_and(|(k, _)| *k == key);
            let failed = c.failed.as_ref().is_some_and(|(k, _)| *k == key);
            if !running && !failed {
                c.failed = None;
                let (tx, rx) = mpsc::channel();
                let (mesh, k, ctx) = (self.mesh.clone(), key.clone(), ctx.clone());
                std::thread::spawn(move || {
                    let _ = tx.send(CageProbe::build(&mesh, k));
                    ctx.request_repaint();
                });
                c.job = Some((key, rx));
            }
            return;
        }
        let reach = crate::cage::Reach::new(&self.bake_settings);
        if c.drawn.as_ref() != Some(&reach) {
            let probe = c.probe.as_ref().expect("current");
            let (tris, lines, stats) = probe.overlay(&reach);
            self.overlays.set(&self.rs.device, OverlaySlot::Cage, &tris, &lines);
            c.drawn = Some(reach);
            c.stats = probe.has_high.then_some(stats);
            ctx.request_repaint();
        }
    }

    /// Shows the imported vertex colors directly on the 3D mesh when they are the active ID source.
    /// This is a viewport-only preview; the baked ID texture still uses the same per-corner colors.
    fn sync_vertex_color_preview(&mut self, ctx: &egui::Context) {
        let has_colors = self.mesh.tri_vertex_color.as_ref().is_some_and(|colors| {
            colors.len() == self.mesh.indices.len() / 3
        });
        let show = self.bake_settings.id_source == bake::IdSource::VertexColors && has_colors;
        if show == self.vertex_color_preview_active {
            return;
        }
        if show {
            let mut triangles = Vec::with_capacity(self.mesh.indices.len());
            if let Some(colors) = &self.mesh.tri_vertex_color {
                for (tri_index, tri) in self.mesh.indices.chunks_exact(3).enumerate() {
                    let tri_colors = colors[tri_index];
                    for corner in 0..3 {
                        let Some(vertex) = self.mesh.vertices.get(tri[corner] as usize) else { continue; };
                        let rgb = tri_colors[corner];
                        triangles.push(OverlayVertex {
                            pos: vertex.pos,
                            color: [rgb[0], rgb[1], rgb[2], 255],
                        });
                    }
                }
            }
            self.overlays.set(&self.rs.device, OverlaySlot::VertexColors, &triangles, &[]);
        } else {
            self.overlays.clear(OverlaySlot::VertexColors);
        }
        self.vertex_color_preview_active = show;
        ctx.request_repaint();
    }

    // ------------------------------------------------------------------ polygon fill

    fn poly_topology(&mut self) -> Arc<Topology> {
        self.poly.topo.get_or_insert_with(|| Arc::new(Topology::build(&self.mesh))).clone()
    }

    /// The triangle under a view pixel, and the 3D point there.
    fn pick_triangle(&mut self, kind: ViewKind, at: Vec2) -> Option<(u32, Vec3)> {
        match kind {
            ViewKind::V3d => {
                let size = Vec2::new(self.views.view3d.size[0] as f32, self.views.view3d.size[1] as f32);
                let (o, d) = self.cam.ray(at, size);
                self.bvh.raycast(o, d).map(|h| (h.tri, h.pos))
            }
            ViewKind::V2d => {
                let size = self.views.view2d.size;
                let center = Vec2::new(size[0] as f32, size[1] as f32) * 0.5;
                let uv = (at - center - self.cam2d.pan) / self.cam2d.zoom.max(1.0) + Vec2::splat(0.5);
                let topo = self.poly_topology();
                // The 2D view shows the edited texture set: pick among its faces.
                let t = topo.pick_uv(uv)?;
                Some((t, topo.point_at_uv(t, uv)))
            }
        }
    }

    /// The texture set whose part of the mesh is under a 3D view pixel. The
    /// 2D view shows only the edited set, so it has none to pick.
    fn set_under(&mut self, kind: ViewKind, at: Vec2) -> Option<usize> {
        if kind != ViewKind::V3d || self.mesh.sets.len() < 2 {
            return None;
        }
        self.pick_triangle(kind, at).map(|(t, _)| self.mesh.set_of(t))
    }

    fn poly_opts(&self, at: Option<Vec3>) -> SelectOpts {
        SelectOpts { at, angle: self.poly.angle }
    }

    /// Fills the faces the fill mode selects around each seed triangle,
    /// skipping any already filled during this drag. `at`: where a single
    /// pick landed (face loops run through the edge nearest it).
    fn poly_fill(&mut self, seeds: &[u32], at: Option<Vec3>) {
        let topo = self.poly_topology();
        let opts = self.poly_opts(at);
        let mut tris = Vec::new();
        for &s in seeds {
            // A filled seed's whole selection is already filled.
            if !self.poly.filled.contains(&s) {
                tris.extend(topo.select_with(self.poly.mode, s, &opts).into_iter().filter(|t| self.poly.filled.insert(*t)));
            }
        }
        // Only the edited texture set takes paint, and only where it's shown.
        if let Some(set) = self.mesh.sets.get(self.sets.active) {
            tris.retain(|t| set.tris.contains(t));
        }
        tris.retain(|&t| !self.tri_hidden(t));
        if tris.is_empty() {
            return;
        }
        // Padding past UV island borders hides seams.
        let geometry = topo.uv_geometry(&tris, self.prefs.poly_padding / self.engine.res as f32);
        let mut mat = self.brush.mat;
        if let StrokeTarget::BakeFix(_) = self.target {
            mat.rough = self.brush.fix_value;
        }
        let paint = BrushPaint {
            mat,
            mask_value: self.brush.mask_value,
            opacity: self.brush.opacity,
            erase: self.poly.stroke_erase,
            grain: self.grain_params(),
        };
        self.engine.stroke_polys(&geometry, &paint);
    }

    /// Triangles whose centers fall inside a box or lasso of view pixels
    /// (`outline`: the lasso, else the box `a`..`b`). In 3D only the ones the
    /// camera sees, as with a click, unless selecting through.
    fn region_triangles(&mut self, kind: ViewKind, a: Vec2, b: Vec2, outline: &[Vec2]) -> Vec<u32> {
        use rayon::prelude::*;
        let topo = self.poly_topology();
        let (lo, hi) = if outline.len() >= 3 {
            (outline.iter().fold(Vec2::splat(f32::MAX), |m, p| m.min(*p)), outline.iter().fold(Vec2::splat(f32::MIN), |m, p| m.max(*p)))
        } else {
            (a.min(b), a.max(b))
        };
        let inside = |p: Vec2| !(p.cmplt(lo).any() || p.cmpgt(hi).any()) && (outline.len() < 3 || crate::polyfill::in_outline(p, outline));
        let through = self.poly.through;
        let n = topo.triangle_count() as u32;
        match kind {
            ViewKind::V3d => {
                let size = Vec2::new(self.views.view3d.size[0] as f32, self.views.view3d.size[1] as f32);
                let vp = self.cam.view_proj(size.x / size.y);
                let eye = self.cam.eye();
                let bvh = &self.bvh;
                (0..n)
                    .into_par_iter()
                    .filter(|&t| {
                        let c = topo.centroid(t);
                        let q = vp * c.extend(1.0);
                        if q.w <= 0.0 {
                            return false;
                        }
                        let px = Vec2::new((q.x / q.w * 0.5 + 0.5) * size.x, (0.5 - q.y / q.w * 0.5) * size.y);
                        if !inside(px) {
                            return false;
                        }
                        if through {
                            return true;
                        }
                        let to = c - eye;
                        let dist = to.length();
                        bvh.raycast(eye, to / dist).is_none_or(|h| h.tri == t || h.pos.distance(eye) >= dist * 0.999)
                    })
                    .collect()
            }
            ViewKind::V2d => {
                let size = self.views.view2d.size;
                let center = Vec2::new(size[0] as f32, size[1] as f32) * 0.5;
                let (pan, zoom) = (self.cam2d.pan, self.cam2d.zoom.max(1.0));
                // Back from UV to view pixels, where the box or lasso is.
                let to_px = |uv: Vec2| (uv - Vec2::splat(0.5)) * zoom + center + pan;
                (0..n).into_par_iter().filter(|&t| inside(to_px(topo.uv_centroid(t)))).collect()
            }
        }
    }

    /// Shows the faces the fill tool would fill under the cursor.
    fn update_poly_highlight(&mut self) {
        let hover = self.poly.frame_hover.take().filter(|_| self.tool == Tool::PolyFill);
        let erase = if self.stroke.is_some() { self.poly.stroke_erase } else { self.poly.erase };
        let mode = self.poly.mode;
        let want = hover.map(|(tri, at)| PolyShown {
            tri,
            mode,
            erase,
            edge: (mode == FillMode::Loop).then(|| self.poly_topology().loop_edge(tri, at)),
            angle: if mode == FillMode::Flat { self.poly.angle.round() as u32 } else { 0 },
        });
        if want == self.poly.shown && hover.is_some() == self.poly.shown.is_some() {
            return;
        }
        self.poly.shown = want;
        match (want, hover) {
            (Some(w), Some((_, at))) => {
                let topo = self.poly_topology();
                let color = if erase { [255, 90, 80] } else { [80, 160, 255] };
                let mut tris = topo.select_with(w.mode, w.tri, &self.poly_opts(Some(at)));
                if let Some(set) = self.mesh.sets.get(self.sets.active) {
                    tris.retain(|t| set.tris.contains(t));
                }
                let (fill, outline) = topo.overlay(&tris, color);
                self.overlays.set(&self.rs.device, OverlaySlot::Selection, &fill, &outline);
            }
            _ => self.overlays.clear(OverlaySlot::Selection),
        }
    }

    fn try_begin_stroke(&mut self, kind: ViewKind, at: Vec2) {
        let erase = self.tool == Tool::Eraser;
        if self.begin_stroke_on_target(kind, at, erase) {
            let stamp = self.make_stamp(at, None, self.pressure);
            self.paint_stamps(kind, &[stamp]);
        }
    }

    /// Starts a stroke on the current target (layer content, mask or bake
    /// fix), or says in the status bar why it can't.
    fn begin_stroke_on_target(&mut self, kind: ViewKind, at: Vec2, erase: bool) -> bool {
        let idx = self.selected;
        if let StrokeTarget::BakeFix(_) = self.target {
            if self.engine.begin_stroke(0, self.target, [false; crate::engine::CHANNELS], erase) {
                self.stroke = Some(StrokeInput { view: kind, last: at, lazy: at, pressure: self.pressure });
                return true;
            } else if !self.engine.has_bake {
                self.status = "Bake mesh maps before fixing them.".into();
            }
            return false;
        }
        let Some(layer) = self.engine.layers.get(idx) else {
            self.status = if self.engine.layers.is_empty() { "Add a layer to paint on." } else { "Select a layer to paint on." }.into();
            return false;
        };
        if self.target == StrokeTarget::Content && !layer.is_paint() {
            self.status = "Fill layers are procedural. Add a mask to paint on it, or select a paint layer.".into();
            return false;
        }
        if self.engine.begin_stroke(idx, self.target, self.brush.mat.enabled, erase) {
            self.stroke = Some(StrokeInput { view: kind, last: at, lazy: at, pressure: self.pressure });
            true
        } else {
            self.status = if erase {
                "Nothing to erase on the enabled channels.".into()
            } else {
                "Enable at least one channel in the brush material.".into()
            };
            false
        }
    }

    /// Samples the imported per-corner vertex color and assigns it to the active ID mask.
    fn pick_vertex_id_color(&mut self, kind: ViewKind, at: Vec2) {
        if kind != ViewKind::V3d {
            self.status = "Vertex ID sampling is available in the 3D view.".into();
            return;
        }
        let size = Vec2::new(self.views.view3d.size[0] as f32, self.views.view3d.size[1] as f32);
        let (origin, direction) = self.cam.ray(at, size);
        let Some(hit) = self.bvh.raycast(origin, direction) else {
            self.status = "Click a colored area on the mesh.".into();
            return;
        };
        let Some(colors) = self.mesh.tri_vertex_color.as_ref().and_then(|all| all.get(hit.tri as usize)) else {
            self.status = "This mesh has no imported vertex colors. Assign colors in Blender and reimport the FBX/OBJ.".into();
            self.id_color_pick_target = None;
            return;
        };
        let weights = [hit.bary.x, hit.bary.y, hit.bary.z];
        let mut color = [0.0f32; 3];
        for k in 0..3 {
            for ch in 0..3 { color[ch] += colors[k][ch] as f32 / 255.0 * weights[k]; }
        }
        let Some((layer_idx, mask_layer)) = self.id_color_pick_target.take() else { return; };
        let Some(mask) = self.engine.layers.get_mut(layer_idx).and_then(|layer| layer.mask.as_mut()) else {
            self.status = "The target ID mask no longer exists.".into();
            return;
        };
        let generator = match mask_layer {
            None => mask.generator.as_mut(),
            Some(i) => mask.layers.get_mut(i).and_then(|layer| match &mut layer.source {
                crate::engine::MaskSource::Generator(g) => Some(g),
                crate::engine::MaskSource::Fill(_) => None,
            }),
        };
        let Some(generator) = generator else {
            self.status = "The selected mask generator no longer exists.".into();
            return;
        };
        generator.id_color = color;
        generator.id_tolerance = generator.id_tolerance.max(0.015);
        self.bake_settings.id_source = bake::IdSource::VertexColors;
        self.view.mode = ViewMode::Map(4);
        self.engine.mark_dirty();
        self.show_bake = true;
        let rgb = color.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8);
        self.status = format!("Vertex ID #{:02X}{:02X}{:02X} sampled. Bake mesh maps to update the ID preview.", rgb[0], rgb[1], rgb[2]);
    }

    /// Eyedropper: copies the final color, roughness and metallic under the
    /// cursor into the brush, then goes back to painting.
    fn pick_material(&mut self, kind: ViewKind, at: Vec2) {
        let mut set = self.sets.active;
        let uv = match kind {
            ViewKind::V3d => {
                let size = Vec2::new(self.views.view3d.size[0] as f32, self.views.view3d.size[1] as f32);
                let (o, d) = self.cam.ray(at, size);
                let Some(hit) = self.bvh.raycast(o, d) else {
                    self.status = "Pick on the mesh.".into();
                    return;
                };
                let corner = |k: usize| Vec2::from_array(self.mesh.vertices[self.mesh.indices[hit.tri as usize * 3 + k] as usize].uv);
                set = self.mesh.set_of(hit.tri);
                corner(0) * hit.bary.x + corner(1) * hit.bary.y + corner(2) * hit.bary.z
            }
            ViewKind::V2d => {
                let size = self.views.view2d.size;
                let center = Vec2::new(size[0] as f32, size[1] as f32) * 0.5;
                (at - center - self.cam2d.pan) / self.cam2d.zoom.max(1.0) + Vec2::splat(0.5)
            }
        };
        if !(0.0..=1.0).contains(&uv.x) || !(0.0..=1.0).contains(&uv.y) {
            self.status = "Pick inside the texture.".into();
            return;
        }
        let engines = self.set_engines();
        let (color, rough, metal) = engines.get(set).unwrap_or(&&self.engine).sample_final(uv);
        self.brush.mat.color = color;
        self.brush.mat.rough = rough;
        self.brush.mat.metal = metal;
        self.set_tool(Tool::Paint);
        let hex = color.map(|c| (c * 255.0).round() as u8);
        self.status = format!(
            "Picked #{:02X}{:02X}{:02X}, roughness {:.2}, metallic {:.2}.",
            hex[0], hex[1], hex[2], rough, metal
        );
    }

    fn paint_stamps(&mut self, kind: ViewKind, raw: &[([f32; 4], f32)]) {
        let su = self.stamp_uniform(kind);
        let stamps = match kind {
            ViewKind::V3d => {
                let [w, h] = self.views.view3d.size;
                let radius = self.brush.size * 0.5 * self.ppp;
                resolve_stamps(&self.mesh, &self.bvh, &self.cam, Vec2::new(w as f32, h as f32), self.brush.projection, self.brush.symmetry, radius, raw)
            }
            // The 2D view is the texture: stamps land where they are drawn.
            ViewKind::V2d => resolve_stamps(&self.mesh, &self.bvh, &self.cam, Vec2::ONE, BrushProjection::Screen, 0, 0.0, raw),
        };
        let mut mat = self.brush.mat;
        if let StrokeTarget::BakeFix(_) = self.target {
            mat.rough = self.brush.fix_value;
        }
        let paint = BrushPaint {
            mat,
            mask_value: self.brush.mask_value,
            opacity: self.brush.opacity,
            erase: self.tool == Tool::Eraser,
            grain: self.grain_params(),
        };
        let region = match kind {
            ViewKind::V3d => {
                let [w, h] = self.views.view3d.size;
                let radius = self.brush.size * 0.5 * self.ppp;
                let size = Vec2::new(w as f32, h as f32);
                stroke_region(&self.mesh, &self.bvh, &self.cam, size, self.brush.projection, self.brush.symmetry, radius, &stamps)
            }
            ViewKind::V2d => {
                // The 2D view shows the texture: the stamps' own UV squares.
                let [w, h] = self.views.view2d.size;
                let center = Vec2::new(w as f32, h as f32) * 0.5;
                let zoom = self.cam2d.zoom.max(1.0);
                let radius = self.brush.size * 0.5 * self.ppp;
                stamps.iter().fold(None, |acc, s| {
                    let uv = (Vec2::new(s[0][0], s[0][1]) - center - self.cam2d.pan) / zoom + Vec2::splat(0.5);
                    let r = radius * s[0][2] / zoom;
                    Some(grow(acc, uv - Vec2::splat(r), uv + Vec2::splat(r)))
                })
            }
        };
        self.engine.stroke_step(&stamps, su, self.views.view3d.depth_view(), &paint, region);
    }

    fn continue_stroke(&mut self, at: Vec2) {
        let Some(s) = &mut self.stroke else { return };
        // Stabilizer: the brush follows the cursor on a string, only moving
        // once the cursor pulls the string tight.
        let string = self.brush.stabilize * self.ppp;
        let pull = at - s.lazy;
        if pull.length() > string {
            s.lazy += pull * (1.0 - string / pull.length());
        }
        let at = s.lazy;
        let pressure = self.pressure;
        let radius = self.brush.size * 0.5 * self.ppp * if self.brush.pressure_size { pressure.max(0.1) } else { 1.0 };
        let mut step = (self.brush.spacing * radius * 2.0).max(1.0);
        let delta = at - s.last;
        let dist = delta.length();
        if dist < step {
            return;
        }
        // Keep each frame's stamp list bounded on very fast strokes.
        let max = crate::engine::MAX_STAMPS as f32;
        if dist / step > max {
            step = dist / max;
        }
        let dir = delta / dist;
        let mut points = Vec::new();
        let mut travelled = step;
        let p0 = s.pressure;
        while travelled <= dist {
            // Pressure eases from the last stamp's to the current one.
            points.push((s.last + dir * travelled, p0 + (pressure - p0) * travelled / dist));
            travelled += step;
        }
        s.last += dir * (travelled - step);
        s.pressure = points.last().map_or(p0, |p| p.1);
        let kind = s.view;
        let stamps: Vec<([f32; 4], f32)> = points.into_iter().map(|(p, pr)| self.make_stamp(p, Some(dir), pr)).collect();
        self.paint_stamps(kind, &stamps);
    }

    fn end_stroke(&mut self) {
        if let Some(s) = self.stroke.take() {
            if matches!(self.tool, Tool::Paint | Tool::Eraser) {
                self.line_anchor = Some(LineAnchor { view: s.view, at: s.last, key: self.view_key(s.view) });
            }
            if let Some(entry) = self.engine.end_stroke() {
                self.undo.push("Paint stroke", crate::undo::Edit::Stroke(entry));
            }
        }
    }

    /// Scroll zoom per point of scrolling, with the user's speed and direction.
    fn zoom_rate(&self) -> f32 {
        0.0015 * self.prefs.zoom_speed * if self.prefs.invert_zoom { -1.0 } else { 1.0 }
    }

    fn view_input(&mut self, ui: &egui::Ui, kind: ViewKind, rect: Rect, resp: &egui::Response) {
        let (hover, pressed, down, mods, delta, scroll, x_down, z_down) = ui.input(|i| {
            (
                i.pointer.hover_pos(),
                i.pointer.primary_pressed(),
                i.pointer.primary_down(),
                i.modifiers,
                i.pointer.delta(),
                i.smooth_scroll_delta.y,
                i.key_down(egui::Key::X),
                i.key_down(egui::Key::Z),
            )
        });
        let ppp = self.ppp;
        let local = |p: egui::Pos2| Vec2::new((p.x - rect.min.x) * ppp, (p.y - rect.min.y) * ppp);
        use egui::PointerButton as B;

        // One brush value per gesture, all with the right mouse button and a
        // horizontal drag (right: more): X changes only the size, Z only the
        // stroke opacity, Ctrl only the hardness.
        let rmb = resp.dragged_by(B::Secondary);
        let adjust_xz = rmb && !mods.any() && (x_down || z_down);
        let adjust_hardness = rmb && mods.ctrl && !mods.alt && !mods.shift && !x_down && !z_down;
        let adjust_brush = adjust_xz || adjust_hardness;
        if adjust_xz && x_down {
            self.brush.size = (self.brush.size + delta.x).clamp(2.0, 800.0);
            self.status = format!("Brush size {:.0}", self.brush.size);
        } else if adjust_xz {
            self.brush.opacity = (self.brush.opacity + delta.x * 0.0025).clamp(0.01, 1.0);
            self.status = format!("Stroke opacity {:.0}%", self.brush.opacity * 100.0);
        } else if adjust_hardness {
            self.brush.hardness = (self.brush.hardness + delta.x * 0.004).clamp(0.0, 1.0);
            self.status = format!("Brush hardness {:.0}%", self.brush.hardness * 100.0);
        }

        // Pen pressure arrives with touch events; mice never send them.
        let touch = ui.input(|i| {
            i.events.iter().rev().find_map(|e| match e {
                egui::Event::Touch { phase, force, .. } => Some((*phase, *force)),
                _ => None,
            })
        });
        match touch {
            Some((egui::TouchPhase::End | egui::TouchPhase::Cancel, _)) => self.pressure = 1.0,
            Some((_, Some(force))) => self.pressure = force.clamp(0.02, 1.0),
            _ => {}
        }

        // Navigation.
        match kind {
            ViewKind::V3d => {
                let orbit = !adjust_brush
                    && ((resp.dragged_by(B::Primary) && mods.alt)
                        || (resp.dragged_by(B::Secondary) && !mods.alt && !mods.shift && !mods.ctrl));
                if orbit {
                    let k = 0.008 * self.prefs.orbit_speed;
                    self.cam.yaw -= delta.x * k;
                    self.cam.pitch = (self.cam.pitch + delta.y * k).clamp(-1.55, 1.55);
                }
                if resp.dragged_by(B::Middle) {
                    let (right, up) = self.cam.basis();
                    let k = self.cam.dist * 0.0016;
                    self.cam.target += (-right * delta.x + up * delta.y) * k;
                }
                if resp.dragged_by(B::Secondary) && mods.alt && !adjust_brush {
                    self.cam.dist = (self.cam.dist * (delta.y * 0.01).exp()).clamp(0.3, 40.0);
                }
                if resp.dragged_by(B::Secondary) && mods.shift && !adjust_brush {
                    self.view.env_rotation += delta.x * 0.01;
                }
                if resp.hovered() && scroll != 0.0 {
                    self.cam.dist = (self.cam.dist * (-scroll * self.zoom_rate()).exp()).clamp(0.3, 40.0);
                }
                if resp.double_clicked() && mods.alt {
                    if let Some(p) = hover {
                        let size = Vec2::new(self.views.view3d.size[0] as f32, self.views.view3d.size[1] as f32);
                        let (o, d) = self.cam.ray(local(p), size);
                        if let Some(hit) = self.bvh.raycast(o, d) {
                            self.cam.target = hit.pos;
                        }
                    }
                }
            }
            ViewKind::V2d => {
                if resp.dragged_by(B::Middle) || (resp.dragged_by(B::Secondary) && !adjust_brush) || (resp.dragged_by(B::Primary) && mods.alt) {
                    self.cam2d.pan += Vec2::new(delta.x, delta.y) * ppp;
                }
                if resp.hovered() && scroll != 0.0 {
                    if let Some(p) = hover {
                        // Zoom about the cursor.
                        let size = self.views.view2d.size;
                        let center = Vec2::new(size[0] as f32, size[1] as f32) * 0.5;
                        let cursor = local(p);
                        let f = (scroll * self.zoom_rate()).exp();
                        let anchor = cursor - center - self.cam2d.pan;
                        self.cam2d.zoom *= f;
                        self.cam2d.pan -= anchor * (f - 1.0);
                    }
                }
            }
        }

        // The bake window's gizmo (direction arrow, point marker) comes first.
        if kind == ViewKind::V3d {
            self.custom_ui.view_rect = Some(rect);
        }
        if kind == ViewKind::V3d && !mods.alt {
            let cursor = hover.filter(|p| rect.contains(*p)).map(local);
            if self.gizmo_input(ui, cursor, pressed, down) {
                return;
            }
        }
        // Painting the bake cage takes drags over the model while it's on.
        if kind == ViewKind::V3d && !mods.alt && self.cage_brush_active() {
            let at = hover.filter(|p| rect.contains(*p) && resp.hovered());
            if self.cage_brush_input(ui, at.map(local), at, down) {
                return;
            }
        }

        // The path tool places and drags points instead of painting.
        if self.tool == Tool::Path {
            let cursor = hover.filter(|p| rect.contains(*p) && !mods.alt).map(local);
            let secondary = ui.input(|i| i.pointer.secondary_clicked()) && !mods.any();
            self.path_input(kind, cursor, pressed && !mods.alt, down && !mods.alt, secondary, resp.hovered());
            return;
        }

        // Polygon fill picks faces instead of laying down stamps.
        if self.tool == Tool::PolyFill {
            // Only the view the box started in handles it (both views run
            // their input every frame).
            if self.poly.box_drag.as_ref().is_some_and(|b| b.view != kind) {
                return;
            }
            if let Some(mut b) = self.poly.box_drag.take() {
                if down {
                    if let Some(p) = hover {
                        let p = rect.clamp(p);
                        b.end = p;
                        // A lasso keeps points a few pixels apart.
                        if self.poly.lasso && b.lasso.last().is_none_or(|q| q.distance(p) >= 3.0) {
                            b.lasso.push(p);
                        }
                    }
                    self.poly.box_drag = Some(b);
                    return;
                }
                // Released: fill what the box or lasso covers. A Shift+click
                // without dragging fills the face under it, like a plain click.
                let (a, z) = (local(b.start), local(b.end));
                let outline: Vec<Vec2> = b.lasso.iter().map(|p| local(*p)).collect();
                let span = outline.iter().fold(a.distance(z), |m, p| m.max(a.distance(*p)));
                let (seeds, at) = if span < 3.0 * ppp {
                    match self.pick_triangle(kind, a) {
                        Some((t, at)) => (vec![t], Some(at)),
                        None => (Vec::new(), None),
                    }
                } else {
                    (self.region_triangles(kind, a, z, if self.poly.lasso { &outline } else { &[] }), None)
                };
                if !seeds.is_empty() && self.begin_stroke_on_target(kind, a, b.erase) {
                    self.poly.filled.clear();
                    self.poly.stroke_erase = b.erase;
                    self.poly_fill(&seeds, at);
                    self.end_stroke();
                }
                return;
            }
            let at = hover.filter(|p| rect.contains(*p) && !mods.alt).map(local);
            let tri = at.and_then(|p| self.pick_triangle(kind, p));
            if at.is_some() {
                self.poly.frame_hover = tri;
            }
            if pressed && mods.shift && at.is_some() && resp.hovered() && self.stroke.is_none() {
                let p = hover.unwrap_or_default();
                let lasso = if self.poly.lasso { vec![p] } else { Vec::new() };
                self.poly.box_drag = Some(BoxDrag { view: kind, start: p, end: p, lasso, erase: self.poly.erase != mods.ctrl });
            } else if pressed && at.is_some() && resp.hovered() && self.stroke.is_none() {
                if let Some((t, pos)) = tri {
                    let erase = self.poly.erase != mods.ctrl;
                    if self.begin_stroke_on_target(kind, at.unwrap_or_default(), erase) {
                        self.poly.filled.clear();
                        self.poly.stroke_erase = erase;
                        self.poly_fill(&[t], Some(pos));
                    }
                }
            } else if down && self.stroke.as_ref().is_some_and(|s| s.view == kind) {
                if let Some((t, pos)) = tri {
                    self.poly_fill(&[t], Some(pos));
                }
            }
            return;
        }

        // Painting.
        if let Some(p) = hover {
            let in_view = rect.contains(p);
            if pressed && in_view && !mods.alt && self.stroke.is_none() && resp.hovered() && self.id_color_pick_target.is_some() {
                self.pick_vertex_id_color(kind, local(p));
            } else if pressed && in_view && !mods.alt && self.stroke.is_none() && resp.hovered() && self.tool == Tool::Picker {
                self.pick_material(kind, local(p));
            } else if pressed && in_view && !mods.alt && self.stroke.is_none() && resp.hovered() {
                // Shift+click: a straight line from where the last stroke ended.
                let anchor = self.line_anchor.filter(|a| mods.shift && a.view == kind && a.key == self.view_key(kind));
                match anchor {
                    Some(a) => {
                        self.try_begin_stroke(kind, a.at);
                        if let Some(s) = &mut self.stroke {
                            s.lazy = local(p);
                        }
                        self.continue_stroke(local(p));
                    }
                    None => self.try_begin_stroke(kind, local(p)),
                }
            } else if down && self.stroke.as_ref().is_some_and(|s| s.view == kind) {
                self.continue_stroke(local(p));
            }
        }
    }

    /// One viewport (a dock tab): input, the GPU render at the tab's size,
    /// overlays and the brush cursor.
    fn view_pane(&mut self, ui: &mut egui::Ui, kind: ViewKind) {
        let rect = ui.available_rect_before_wrap();
        if rect.width() < 4.0 || rect.height() < 4.0 {
            return;
        }
        let resp = ui.allocate_rect(rect, Sense::click_and_drag());
        self.view_input(ui, kind, rect, &resp);
        // A shelf item goes to the texture set under the pointer, which lights up while it's dragged over.
        let drop_set = resp
            .dnd_hover_payload::<ShelfItem>()
            // Masks and brush tips go to the selected layer and the brush instead.
            .filter(|item| !matches!(**item, ShelfItem::Generator(_) | ShelfItem::Brush(_)))
            .and_then(|_| ui.ctx().pointer_interact_pos())
            .and_then(|p| self.set_under(kind, Vec2::new((p.x - rect.min.x) * self.ppp, (p.y - rect.min.y) * self.ppp)));
        if let Some(k) = drop_set {
            self.sets.hover = Some(k);
        }
        if let Some(item) = dnd_release::<ShelfItem>(&resp) {
            if let Some(k) = drop_set {
                self.switch_set(k);
            }
            self.pending.push(Action::AddOnTop(*item));
        }
        self.update_poly_highlight();

        // GPU: apply a settled material edit once, then render this view.
        if self.engine.dirty {
            self.engine.composite();
        }
        self.composite_parked();
        let size = [(rect.width() * self.ppp).round() as u32, (rect.height() * self.ppp).round() as u32];
        let mut rt_view = None;
        match kind {
            ViewKind::V3d => {
                self.views.set_looks = self.set_looks(ui.ctx());
                let engines = texture_sets::engines(&self.engine, &self.sets, self.mesh.sets.len());
                self.views.render_3d(&self.rs, size, &engines, &self.env, &self.gpu_mesh, &self.cam, &self.view, Some(&self.overlays));
                self.drew_3d = true;
                // Path tracing replaces the 3D view while it's idle. Painting and
                // the map display modes use the raster view, which reacts instantly.
                if self.rt.enabled && self.stroke.is_none() && self.view.mode == ViewMode::Material && self.preview_update_at.is_none() {
                    let engines = texture_sets::engines(&self.engine, &self.sets, self.mesh.sets.len());
                    if self.raytracer.render(&self.rs, size, &engines, &self.env, &self.cam, &self.view, &self.rt) {
                        ui.ctx().request_repaint();
                    }
                    rt_view = self.raytracer.egui_id;
                }
            }
            ViewKind::V2d => {
                if self.cam2d.zoom <= 0.0 {
                    self.cam2d.zoom = size[0].min(size[1]) as f32 * 0.9;
                    self.cam2d.pan = Vec2::ZERO;
                }
                self.views.render_2d(&self.rs, size, &self.engine, &self.env, &self.gpu_mesh, self.sets.active, &self.cam2d, &self.view);
            }
        }

        let painter = ui.painter_at(rect);
        let uv = Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));
        let id = match kind {
            ViewKind::V3d => rt_view.unwrap_or(self.views.view3d.egui_id),
            ViewKind::V2d => self.views.view2d.egui_id,
        };
        painter.image(id, rect, uv, Color32::WHITE);
        self.draw_path(&painter, kind, rect);
        self.draw_gizmo(&painter, kind, rect);
        self.draw_cage_brush(&painter, kind);
        if let Some(b) = self.poly.box_drag.as_ref().filter(|b| b.view == kind) {
            let color = if b.erase { Color32::from_rgb(255, 90, 80) } else { Color32::from_rgb(80, 160, 255) };
            if b.lasso.len() >= 2 {
                let mut pts = b.lasso.clone();
                pts.push(b.lasso[0]);
                painter.add(egui::Shape::line(pts, Stroke::new(1.5, color)));
            } else if b.lasso.is_empty() {
                let r = Rect::from_two_pos(b.start, b.end);
                painter.rect_filled(r, 0.0, color.gamma_multiply(0.15));
                painter.rect_stroke(r, 0.0, Stroke::new(1.0, color), egui::StrokeKind::Inside);
            }
        }
        let label = match kind {
            ViewKind::V3d if rt_view.is_some() => {
                let state = if self.raytracer.converged(&self.rt) { "done" } else { "refining" };
                format!("Ray traced · {}/{} spp ({state})", self.raytracer.samples, self.rt.max_samples)
            }
            _ => self.mode_name(self.view.mode),
        };
        painter.text(rect.min + vec2(10.0, 8.0), egui::Align2::LEFT_TOP, label, egui::FontId::proportional(11.5), Color32::from_white_alpha(150));
        if let Some(item) = resp.dnd_hover_payload::<ShelfItem>() {
            let set = drop_set.filter(|_| self.mesh.sets.len() > 1).map(|k| self.mesh.sets[k].name.clone());
            let what = match (*item, set) {
                (ShelfItem::Generator(_), _) => "Drop to mask the selected layer".to_owned(),
                (ShelfItem::Brush(_), _) => "Drop to paint with this brush tip".to_owned(),
                (_, Some(name)) => format!("Drop to add to {name}"),
                _ => "Drop to add".to_owned(),
            };
            painter.rect_stroke(rect.shrink(2.0), 0.0, Stroke::new(2.0, theme::ACCENT), egui::StrokeKind::Inside);
            painter.text(rect.center_bottom() - vec2(0.0, 24.0), egui::Align2::CENTER_CENTER, what, egui::FontId::proportional(13.0), Color32::WHITE);
        }

        // Brush cursor.
        let alt = ui.input(|i| i.modifiers.alt);
        if let Some(p) = resp.hover_pos() {
            if self.tool == Tool::PolyFill && !alt {
                ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
            } else if !alt {
                let r = self.brush.size * 0.5;
                painter.circle_stroke(p, r + 1.0, Stroke::new(1.0, Color32::from_black_alpha(140)));
                painter.circle_stroke(p, r, Stroke::new(1.0, Color32::from_white_alpha(220)));
                let inner = r * self.brush.hardness;
                if inner > 2.0 {
                    painter.circle_stroke(p, inner, Stroke::new(1.0, Color32::from_white_alpha(70)));
                }
                ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
                if let Some(s) = self.stroke.as_ref().filter(|s| s.view == kind && self.brush.stabilize > 0.0) {
                    let lazy = rect.min + vec2(s.lazy.x, s.lazy.y) / self.ppp;
                    painter.line_segment([lazy, p], Stroke::new(1.0, Color32::from_white_alpha(120)));
                    painter.circle_filled(lazy, 2.5, Color32::WHITE);
                }
            }
        }
    }

    /// After every view drew: strokes end when the button is released, and
    /// the 3D depth the brush reads stays valid when no 3D view is showing.
    fn views_end_frame(&mut self, ctx: &egui::Context) {
        if self.stroke.is_some() && !ctx.input(|i| i.pointer.primary_down()) {
            self.end_stroke();
        }
        self.update_poly_highlight();
        if !self.drew_3d {
            let size = self.views.view3d.size;
            let engines = texture_sets::engines(&self.engine, &self.sets, self.mesh.sets.len());
            self.views.render_3d(&self.rs, size, &engines, &self.env, &self.gpu_mesh, &self.cam, &self.view, Some(&self.overlays));
        }
        self.drew_3d = false;
    }

    // ------------------------------------------------------------------ windows

    fn about_window(&mut self, ctx: &egui::Context) {
        let mut open = self.show_about;
        egui::Window::new("About").open(&mut open).collapsible(false).resizable(false).show(ctx, |ui| {
            ui.set_width(360.0);
            ui.label(egui::RichText::new("Material Painter").heading().color(Color32::WHITE));
            ui.label(egui::RichText::new(format!("Version {}", env!("CARGO_PKG_VERSION"))).color(theme::TEXT_DIM));
            if let Some(gpu) = crate::crash::context() {
                // Selectable, so it can be copied into bug reports.
                ui.add(egui::Label::new(egui::RichText::new(format!("GPU: {gpu}")).color(theme::TEXT_DIM)).selectable(true));
            }
            ui.add_space(8.0);
            ui.label("A PBR texture painter: layers, smart materials and masks, mesh map baking, material graphs and custom node shaders.");
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new("Free software under the GNU General Public License, version 3 or later. It comes with no warranty.")
                    .small()
                    .color(theme::TEXT_DIM),
            );
            ui.label(
                egui::RichText::new("Bundled HDRIs and textures: Poly Haven (CC0). Icons: Phosphor (MIT).").small().color(theme::TEXT_DIM),
            );
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("Source code").clicked() {
                    ui.ctx().open_url(egui::OpenUrl::new_tab(REPO_URL));
                }
                if ui.button("Third-party licenses").clicked() {
                    match notices_path() {
                        Some(p) => ui.ctx().open_url(egui::OpenUrl::new_tab(format!("file:///{}", p.display().to_string().replace('\\', "/")))),
                        None => ui.ctx().open_url(egui::OpenUrl::new_tab(format!("{REPO_URL}/blob/master/THIRD_PARTY_NOTICES.html"))),
                    }
                }
            });
        });
        self.show_about = open;
    }

    /// Tells about a crash in the last run, once, with its report.
    fn crash_window(&mut self, ctx: &egui::Context) {
        let Some(path) = self.crash_report.clone() else { return };
        let mut close = false;
        egui::Window::new("Material Painter closed unexpectedly").collapsible(false).resizable(false).show(ctx, |ui| {
            ui.set_width(420.0);
            ui.label("Last time, the app stopped because of an error. A report was saved; sending it helps fix the problem.");
            ui.add_space(4.0);
            ui.label(egui::RichText::new(path.display().to_string()).small().monospace().color(theme::TEXT_DIM));
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui.button("Open the report").clicked() {
                    ui.ctx().open_url(egui::OpenUrl::new_tab(format!("file:///{}", path.display().to_string().replace('\\', "/"))));
                }
                if ui.button("Report the problem").on_hover_text("Opens a new issue on the project page; attach the report").clicked() {
                    ui.ctx().open_url(egui::OpenUrl::new_tab(format!("{REPO_URL}/issues/new")));
                }
                if ui.button("Close").clicked() {
                    close = true;
                }
            });
        });
        if close {
            self.crash_report = None;
        }
    }

    fn help_window(&mut self, ctx: &egui::Context) {
        let mut open = self.show_help;
        egui::Window::new("Controls").open(&mut open).collapsible(false).resizable(false).show(ctx, |ui| {
            egui::Grid::new("help").striped(true).show(ui, |ui| {
                for (k, v) in [
                    ("LMB", "Paint / erase"),
                    ("Alt + LMB  or  RMB", "Orbit (3D) · pan (2D)"),
                    ("MMB", "Pan"),
                    ("Alt + RMB / Wheel", "Zoom"),
                    ("Shift + RMB", "Rotate environment"),
                    ("Ctrl + RMB", "Brush size (→) / hardness (↑)"),
                    ("Alt + double-click", "Set orbit center on the mesh"),
                    ("[  ]", "Brush size"),
                    ("Shift + click", "Straight line from the last stroke"),
                    ("Ctrl+G", "Material Designer (the selected layer's graph)"),
                    ("Ctrl+PageUp / PageDown", "Switch workspace"),
                    ("1 / 2 / I", "Paint / Eraser / Pick material"),
                    ("X", "Toggle X symmetry"),
                    ("F", "Frame mesh"),
                    ("M / C", "Material view / cycle channels"),
                    ("F1 / F2 / F3", "3D+2D / 3D / 2D"),
                    ("Ctrl+Z / Ctrl+Y", "Undo / redo"),
                    ("Ctrl+S / Ctrl+Shift+S", "Save / save as"),
                    ("Ctrl+O", "Open project"),
                    ("Ctrl+Shift+E", "Export textures"),
                    ("Double-click (shelf)", "Add to the layer stack"),
                    ("Drag (shelf)", "Drop on a view or in the layer stack"),
                    ("Ctrl + wheel (shelf)", "Thumbnail size"),
                    ("Drag (layers)", "Reorder; drop on a folder's middle to put it inside"),
                ] {
                    ui.label(egui::RichText::new(k).strong());
                    ui.label(v);
                    ui.end_row();
                }
            });
        });
        self.show_help = open;
    }
}

impl eframe::App for PainterApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let start = Instant::now();
        let ctx = ui.ctx().clone();
        self.ppp = ctx.pixels_per_point();
        let _ = self.rs.device.poll(wgpu::PollType::Poll);
        self.undo.update();
        self.poll_bake(&ctx);
        self.poll_spp_open(&ctx);
        self.update_preview(&ctx);
        self.poll_env();
        self.poll_hdri_thumbs(&ctx);
        self.poll_library(&ctx);
        // Closing the window with unsaved changes asks first.
        if ctx.input(|i| i.viewport().close_requested()) && !self.allow_close {
            if self.is_modified() {
                ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
                self.confirm = Some(Action::Exit);
            } else {
                project_io::remove_autosave();
            }
        }
        self.recovery_dialog(&ctx);
        self.confirm_dialog(&ctx);
        self.autosave_if_due();
        let title = format!(
            "{}{} — Material Painter",
            self.project_path.as_ref().and_then(|p| p.file_stem()).map_or("Untitled".to_string(), |n| n.to_string_lossy().into_owned()),
            if self.is_modified() { " *" } else { "" }
        );
        if title != self.window_title {
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
            self.window_title = title;
        }
        if let Some(p) = &mut self.shelf_previews
            && !p.pump(&self.rs, &ctx, &mut self.shelf)
        {
            self.shelf_previews = None;
        }
        self.path_keys(&ctx);
        self.shortcuts(&ctx);
        self.process_actions(&ctx);
        self.process_mcp(&ctx);

        egui::Panel::top("menu").frame(theme::bar().inner_margin(egui::Margin::symmetric(10, 5))).show(ui, |ui| self.menu_bar(ui));
        let paint = self.workspace == Workspace::Paint;
        if paint {
            egui::Panel::top("toolbar")
                .exact_size(46.0)
                .frame(theme::card(theme::PANEL).inner_margin(egui::Margin::symmetric(12, 4)))
                .show(ui, |ui| self.context_toolbar(ui));
        }
        egui::Panel::bottom("status").exact_size(24.0).frame(theme::bar()).show(ui, |ui| self.status_bar(ui));
        if paint {
        egui::Panel::left("tools")
            .exact_size(56.0)
            .frame(theme::card(theme::PANEL).inner_margin(egui::Margin::symmetric(7, 10)))
            .show(ui, |ui| self.tool_strip(ui));
        }

        // Panel actions (add layer, etc.) apply before this frame's GPU work.
        self.process_actions(&ctx);
        self.sync_brush_tip();
        self.sync_brush_grain();
        self.update_graphs();
        self.update_preview_shader();
        self.update_cage_preview(&ctx);
        self.sync_vertex_color_preview(&ctx);
        match self.workspace {
            Workspace::Paint => {
                egui::CentralPanel::default().frame(egui::Frame::new().fill(theme::BG).inner_margin(egui::Margin::same(4))).show(ui, |ui| {
                    self.dock_ui(ui, Workspace::Paint);
                });
            }
            Workspace::Designer => {
                egui::CentralPanel::default().frame(egui::Frame::new().fill(theme::BG)).show(ui, |ui| self.designer(ui));
            }
            Workspace::ShaderMaker => {
                egui::CentralPanel::default().frame(egui::Frame::new().fill(theme::BG)).show(ui, |ui| self.shader_maker(ui));
            }
        }
        if self.workspace != Workspace::Designer {
            self.graphs.editor.hovered = false;
        }
        self.views_end_frame(&ctx);
        self.docks.save_if_changed(&ctx);

        self.export_window(&ctx);
        self.preferences_window(&ctx);
        self.apply_preferences(&ctx);
        self.bake_window(&ctx);
        self.spp_report_window(&ctx);
        self.help_window(&ctx);
        self.about_window(&ctx);
        self.crash_window(&ctx);
        self.welcome_window(&ctx);
        self.library_window(&ctx);
        // Settings changed in the bake window show on the next frame.
        self.update_cage_preview(&ctx);
        self.sync_vertex_color_preview(&ctx);
        self.service_preview_debounce(&ctx);

        // Property edits become one undo step once the interaction ends: no
        // button held (slider drags) and no text field focused (typing).
        let idle = !ctx.input(|i| i.pointer.any_down()) && ctx.memory(|m| m.focused().is_none());
        if idle && self.stroke.is_none() {
            self.engine.commit_props(&mut self.undo);
        }
        let ms = start.elapsed().as_secs_f32() * 1000.0;
        self.frame_ms = self.frame_ms * 0.9 + ms * 0.1;
    }
}

#[cfg(test)]
mod color_hud_tests{
use super::*;
fn close(a:[f32;3],b:[f32;3])->bool{a.iter().zip(b).all(|(x,y)|(x-y).abs()<1e-4)}
#[test]fn hsv_round_trips(){for c in [[0.8,0.2,0.1],[0.0,0.0,0.0],[1.0,1.0,1.0],[0.2,0.9,0.4],[0.1,0.3,0.95],[0.5,0.5,0.5],[1.0,0.0,0.0]]{assert!(close(hsv_to_rgb(rgb_to_hsv(c)),c),"{c:?}");}}
#[test]fn gray_keeps_previous_hue(){let p=[0.6,0.7,0.8];assert!(close(hsv_keeping(p,[0.4,0.4,0.4]),[0.6,0.0,0.4]));assert!(close(hsv_keeping(p,[0.0,0.0,0.0]),[0.6,0.7,0.0]));}
#[test]fn temperature_warms_and_cools(){let c=[0.5,0.5,0.5];assert_eq!(shift_temperature(c,0.0),c);let w=shift_temperature(c,1.0);let k=shift_temperature(c,-1.0);assert!(w[0]>c[0]&&w[2]<c[2]);assert!(k[0]<c[0]&&k[2]>c[2]);}
}
