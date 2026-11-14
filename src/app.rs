//! Application shell: Substance-Painter-style layout, input handling and
//! the per-frame GPU flow (stamps -> composite -> viewport render).

use crate::bake::{self, BakeRun, BakeSettings, GpuBaker, IdSource, PreparedScene, Quality};
use crate::bake_cache;
use crate::bvh::Bvh;
use crate::cage::{CageProbe, CageStats, ProbeKey};
use crate::environment::{self, EnvBaker, EnvImage, EnvSource, Environment};
use crate::engine::{Drop, FillImage, Folder, Layer, Projection};
use crate::engine::{
    Blend, BrushPaint, Content, Engine, FillParams, GEN_INPUTS, GenKind, Levels, MAPS, MapKind, MatValues, Stamp,
    StampUniform, MAX_STAMPS,
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
use crate::viewport::{Background, Cam2d, FAR, NEAR, OrbitCam, OverlaySlot, Overlays, ViewMode, ViewRenderer, ViewSettings};
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
mod color_ui;
mod bake_jobs;
mod mask_ui;
mod properties_ui;
mod layers_ui;
mod shelf_ui;
mod view_ui;
mod windows_ui;
use color_ui::{ColorSliders, color_edit, hsv_to_rgb, rgb_to_hsv};
mod vertex_id;

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
    /// A mask layer of this kind; makes the (black) mask first if the layer has none.
    AddMaskLayer(GenKind),
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

/// Running average and peak of one measured time, in milliseconds.
#[derive(Default, Clone, Copy)]
struct Stat {
    sum: f64,
    peak: f64,
    n: u32,
}

impl Stat {
    fn add(&mut self, v: f64) {
        self.sum += v;
        self.peak = self.peak.max(v);
        self.n += 1;
    }

    fn text(&self) -> String {
        if self.n == 0 { "-".into() } else { format!("{:.1}/{:.0}", self.sum / f64::from(self.n), self.peak) }
    }
}

/// F6: where the main thread's time goes while painting. Everything here is
/// CPU time (recording and submitting work); it never waits for the GPU, so
/// it does not disturb what it measures. A frame that is slow while these
/// are small is waiting on the GPU.
#[derive(Default)]
struct StrokeMeter {
    on: bool,
    /// `stroke_step` calls since the last recomposition.
    steps_now: u32,
    steps_per_frame: Stat,
    /// CPU time spent mapping cursor samples to surface stamps.
    resolve_cpu: Stat,
    /// CPU time spent calculating the changed UV region before GPU submission.
    region_cpu: Stat,
    step_cpu: Stat,
    begin_cpu: Stat,
    composite_cpu: Stat,
    end_cpu: Stat,
    /// The recomposition right after letting go is still due.
    after_pending: bool,
    after_release_cpu: Stat,
    gpu_stamp: Stat,
    gpu_apply: Stat,
    gpu_composite: Stat,
    gpu_composite_total: Stat,
    /// Engine counters (full recompositions, thumbnails rendered) when F6
    /// went on, and the latest values.
    counters_base: Option<(u32, u32)>,
    counters: (u32, u32),
    engine_context: Option<(u64, usize)>,
}

impl StrokeMeter {
    /// The recomposition ran: counts the frame's steps while painting, or
    /// times the first one after releasing. `counters` are the engine's
    /// full-recomposition and thumbnail-render totals.
    fn composited(&mut self, painting: bool, cpu_ms: f64, counters: (u32, u32), engine_epoch: u64, texture_set: usize) {
        // The active Engine can be swapped for another one at the same address
        // and texture-set index. Track explicit instance changes instead of
        // relying only on counters moving backwards.
        let engine_context = (engine_epoch, texture_set);
        let counters_reset = self.counters_base.is_some()
            && (self.engine_context != Some(engine_context) || counters.0 < self.counters.0 || counters.1 < self.counters.1);
        if counters_reset {
            self.counters_base = Some(counters);
        } else {
            self.counters_base.get_or_insert(counters);
        }
        self.counters = counters;
        self.engine_context = Some(engine_context);
        if painting {
            self.composite_cpu.add(cpu_ms);
            self.steps_per_frame.add(f64::from(std::mem::take(&mut self.steps_now)));
        } else if std::mem::take(&mut self.after_pending) {
            self.after_release_cpu.add(cpu_ms);
        }
    }

    /// Status-bar text: average/peak in ms (steps: calls per frame).
    fn text(
        &self,
        res: u32,
        paint_scissor_pixels: u64,
        paint_scissor_rects: u32,
        scissor_pixels: u64,
        bbox_pixels: u64,
        scissor_rects: u32,
        layer_passes: u32,
        folder_passes: u32,
        mask_cache_hits: u32,
        mask_cache_misses: u32,
        mask_gpu_passes: u32,
    ) -> String {
        let (full, thumbs) = self.counters_base.map_or((0, 0), |b| (self.counters.0.saturating_sub(b.0), self.counters.1.saturating_sub(b.1)));
        let texture_pixels = u64::from(res) * u64::from(res);
        let paint_percent = if texture_pixels == 0 { 0.0 } else { paint_scissor_pixels as f64 * 100.0 / texture_pixels as f64 };
        let patch_percent = if texture_pixels == 0 { 0.0 } else { scissor_pixels as f64 * 100.0 / texture_pixels as f64 };
        let bbox_percent = if texture_pixels == 0 { 0.0 } else { bbox_pixels as f64 * 100.0 / texture_pixels as f64 };
        format!(
            "F6 CPU avg/peak ms: steps/frame {}, resolve {}, region {}, stamp {}, begin {}, recomp {}, end {}, recomp after release {} | GPU avg/peak ms: stamp {}, apply {}, composite layer {}, composite total {} | paint {:.1}%/{} scissors | composite {:.1}% (bbox {:.1}%, {} scissors) | stack {}L/{}F | masks {}h/{}m/{}p | since F6: full recomps {full}, thumbnails {thumbs}",
            self.steps_per_frame.text(),
            self.resolve_cpu.text(),
            self.region_cpu.text(),
            self.step_cpu.text(),
            self.begin_cpu.text(),
            self.composite_cpu.text(),
            self.end_cpu.text(),
            self.after_release_cpu.text(),
            self.gpu_stamp.text(),
            self.gpu_apply.text(),
            self.gpu_composite.text(),
            self.gpu_composite_total.text(),
            paint_percent,
            paint_scissor_rects,
            patch_percent,
            bbox_percent,
            scissor_rects,
            layer_passes,
            folder_passes,
            mask_cache_hits,
            mask_cache_misses,
            mask_gpu_passes
        )
    }
}

#[cfg(test)]
mod stroke_meter_tests {
    use super::StrokeMeter;

    #[test]
    fn counter_baseline_resets_when_engine_epoch_changes_even_if_counters_increase() {
        let mut meter = StrokeMeter::default();
        meter.composited(false, 0.0, (3, 5), 0, 0);
        meter.composited(false, 0.0, (6, 8), 0, 0);
        assert_eq!(meter.counters_base, Some((3, 5)));

        // A replacement engine on the same set can start with larger counters.
        meter.composited(false, 0.0, (10, 10), 1, 0);
        assert_eq!(meter.counters_base, Some((10, 10)));
        meter.composited(false, 0.0, (12, 11), 1, 0);

        let text = meter.text(2048, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert!(text.contains("since F6: full recomps 2, thumbnails 1"), "{text}");
    }

    #[test]
    fn counter_baseline_resets_when_switching_to_a_busier_texture_set() {
        let mut meter = StrokeMeter::default();
        meter.composited(false, 0.0, (3, 2), 0, 0);
        meter.composited(false, 0.0, (5, 4), 0, 0);
        meter.composited(false, 0.0, (17, 11), 1, 1);
        assert_eq!(meter.counters_base, Some((17, 11)));
        meter.composited(false, 0.0, (19, 12), 1, 1);

        let text = meter.text(2048, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert!(text.contains("since F6: full recomps 2, thumbnails 1"), "{text}");
    }
}


#[derive(Default)]
struct DiagnosticCapture {
    active: bool,
    started_context: String,
    samples: Vec<String>,
    dropped_samples: u32,
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
    /// Created only when the user first enables ray-traced preview.
    raytracer: Option<RayTracer>,
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
    vertex_id: vertex_id::VertexIdState,
    /// ID source of the maps that were baked last.
    baked_id_source: IdSource,
    /// Latest material edit; the expensive preview rebuild waits until this settles.
    preview_update_at: Option<Instant>,
    /// The next rebuild reports how long it took (F5).
    time_composite: bool,
    /// F6: CPU time of the paint path.
    meter: StrokeMeter,
    /// Temporary ray-hit trace used only by the diagnostic hotkey.
    diagnostics: DiagnosticCapture,
    /// Changes whenever the active Engine instance is replaced or swapped.
    engine_epoch: u64,
    /// The running bake only makes the ID map (from vertex colors).
    bake_id_only: bool,
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
    /// Compiled on demand when a mesh bake is actually requested.
    baker: Option<GpuBaker>,
    show_bake: bool,
    shelf: Shelf,
    /// CPU-only material and generator previews arrive from a worker.
    cpu_preview_rx: Option<mpsc::Receiver<CpuShelfPreview>>,
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
    /// The slowest recent frame (decays), to spot stutter that the average hides.
    frame_peak_ms: f32,
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
/// Which CPU-generated shelf preview a worker result updates.
enum CpuShelfPreviewTarget {
    Smart(usize),
    Material(usize),
    Generator(usize),
}

struct CpuShelfPreview {
    target: CpuShelfPreviewTarget,
    image: egui::ColorImage,
}

/// CPU-only previews are generated away from PainterApp::new. The egui
/// texture handles are created and updated only on the UI thread.
fn start_cpu_shelf_previews(ctx: &egui::Context) -> mpsc::Receiver<CpuShelfPreview> {
    let (tx, rx) = mpsc::channel();
    let ctx = ctx.clone();
    std::thread::Builder::new()
        .name("shelf-cpu-previews".into())
        .spawn(move || {
            let _worker = crate::startup_trace::span("cpu worker: total");
            for (i, material) in presets::MATERIALS.iter().enumerate() {
                let image = {
                    let _s = crate::startup_trace::span_d("cpu preview: material", material.name.to_string());
                    presets::render_preview(&[PreviewLayer {
                        fill: material.fill(),
                        opacity: 1.0,
                        generator: None,
                    }])
                };
                if tx.send(CpuShelfPreview { target: CpuShelfPreviewTarget::Material(i), image }).is_err() {
                    return;
                }
                ctx.request_repaint();
            }
            for (i, smart) in presets::SMART_MATERIALS.iter().enumerate() {
                let image = {
                    let _s = crate::startup_trace::span_d("cpu preview: smart", smart.name.to_string());
                    let layers: Vec<PreviewLayer> = smart
                        .layers
                        .iter()
                        .map(|layer| PreviewLayer { fill: layer.fill(), opacity: layer.opacity, generator: layer.generator() })
                        .collect();
                    presets::render_preview(&layers)
                };
                if tx.send(CpuShelfPreview { target: CpuShelfPreviewTarget::Smart(i), image }).is_err() {
                    return;
                }
                ctx.request_repaint();
            }
            for (i, generator) in GenKind::ALL.iter().enumerate() {
                let image = {
                    let _s = crate::startup_trace::span_d("cpu preview: generator", generator.name().to_string());
                    presets::render_generator_preview(*generator)
                };
                if tx.send(CpuShelfPreview { target: CpuShelfPreviewTarget::Generator(i), image }).is_err() {
                    return;
                }
                ctx.request_repaint();
            }
        })
        .expect("could not start CPU shelf preview worker");
    rx
}

#[cfg(test)]
mod cpu_shelf_preview_tests {
    use super::*;

    fn assert_same_pixels(actual: &egui::ColorImage, expected: &egui::ColorImage, label: &str) {
        assert_eq!(actual.size, expected.size, "{label}: dimensions");
        assert_eq!(actual.pixels, expected.pixels, "{label}: pixel bytes");
    }

    #[test]
    fn background_cpu_previews_match_the_synchronous_reference() {
        let ctx = egui::Context::default();
        let updates: Vec<CpuShelfPreview> = start_cpu_shelf_previews(&ctx).into_iter().collect();
        assert_eq!(
            updates.len(),
            presets::MATERIALS.len() + presets::SMART_MATERIALS.len() + GenKind::ALL.len(),
            "worker must return every shelf preview"
        );

        let mut materials = std::collections::HashSet::new();
        let mut smart_materials = std::collections::HashSet::new();
        let mut generators = std::collections::HashSet::new();

        for update in updates {
            match update.target {
                CpuShelfPreviewTarget::Material(i) => {
                    assert!(materials.insert(i), "duplicate material preview {i}");
                    let material = &presets::MATERIALS[i];
                    let expected = presets::render_preview(&[PreviewLayer {
                        fill: material.fill(),
                        opacity: 1.0,
                        generator: None,
                    }]);
                    assert_same_pixels(&update.image, &expected, material.name);
                }
                CpuShelfPreviewTarget::Smart(i) => {
                    assert!(smart_materials.insert(i), "duplicate smart material preview {i}");
                    let smart = &presets::SMART_MATERIALS[i];
                    let layers: Vec<PreviewLayer> = smart
                        .layers
                        .iter()
                        .map(|layer| PreviewLayer { fill: layer.fill(), opacity: layer.opacity, generator: layer.generator() })
                        .collect();
                    let expected = presets::render_preview(&layers);
                    assert_same_pixels(&update.image, &expected, smart.name);
                }
                CpuShelfPreviewTarget::Generator(i) => {
                    assert!(generators.insert(i), "duplicate generator preview {i}");
                    let expected = presets::render_generator_preview(GenKind::ALL[i]);
                    assert_same_pixels(&update.image, &expected, GenKind::ALL[i].name());
                }
            }
        }

        assert_eq!(materials.len(), presets::MATERIALS.len());
        assert_eq!(smart_materials.len(), presets::SMART_MATERIALS.len());
        assert_eq!(generators.len(), GenKind::ALL.len());
    }
}

/// Wait until material controls have been idle before rebuilding the preview.
const MATERIAL_PREVIEW_DELAY: Duration = Duration::from_millis(650);

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

/// UV tile resolution for conservative multi-rectangle stroke bounds.
const STROKE_REGION_GRID: usize = 64;
/// Avoid turning a complex UV layout into hundreds of tiny GPU draws.
const MAX_STROKE_REGION_RECTS: usize = 64;

fn mark_uv_tiles(tiles: &mut [bool], lo: Vec2, hi: Vec2) {
    const N: usize = STROKE_REGION_GRID;
    if hi.x < 0.0 || hi.y < 0.0 || lo.x > 1.0 || lo.y > 1.0 {
        return;
    }
    let cell = |v: f32| ((v.clamp(0.0, 1.0) * N as f32).floor() as usize).min(N - 1);
    let (x0, x1) = (cell(lo.x), cell(hi.x));
    let (y0, y1) = (cell(lo.y), cell(hi.y));
    for y in y0.min(y1)..=y0.max(y1) {
        for x in x0.min(x1)..=x0.max(x1) {
            tiles[y * N + x] = true;
        }
    }
}

/// Compresses occupied UV tiles into disjoint rectangles. If a pathological
/// layout creates too many runs, fall back to one conservative bounding box.
fn uv_tile_rectangles(tiles: &[bool]) -> Option<Vec<[f32; 4]>> {
    const N: usize = STROKE_REGION_GRID;
    let mut rects: Vec<[usize; 4]> = Vec::new();
    for y in 0..N {
        let mut x = 0;
        while x < N {
            if !tiles[y * N + x] {
                x += 1;
                continue;
            }
            let x0 = x;
            while x < N && tiles[y * N + x] {
                x += 1;
            }
            if let Some(r) = rects.iter_mut().rev().find(|r| r[0] == x0 && r[2] == x && r[3] == y) {
                r[3] = y + 1;
            } else {
                rects.push([x0, y, x, y + 1]);
                if rects.len() > MAX_STROKE_REGION_RECTS {
                    // The run count cannot shrink later; bound pathological
                    // checkerboard layouts to one fixed 64x64 scan.
                    let mut bounds = [N, N, 0, 0];
                    for (i, &occupied) in tiles.iter().enumerate() {
                        if occupied {
                            let (tx, ty) = (i % N, i / N);
                            bounds[0] = bounds[0].min(tx);
                            bounds[1] = bounds[1].min(ty);
                            bounds[2] = bounds[2].max(tx + 1);
                            bounds[3] = bounds[3].max(ty + 1);
                        }
                    }
                    return Some(vec![
                        [
                            bounds[0] as f32 / N as f32,
                            bounds[1] as f32 / N as f32,
                            bounds[2] as f32 / N as f32,
                            bounds[3] as f32 / N as f32,
                        ],
                    ]);
                }
            }
        }
    }
    if rects.is_empty() {
        return None;
    }
    Some(
        rects.into_iter()
            .map(|r| [r[0] as f32 / N as f32, r[1] as f32 / N as f32, r[2] as f32 / N as f32, r[3] as f32 / N as f32])
            .collect(),
    )
}

/// An empty tile set is a known empty paint region, not an unknown one.
fn uv_tile_rectangles_or_empty(tiles: &[bool]) -> Vec<[f32; 4]> {
    uv_tile_rectangles(tiles).unwrap_or_default()
}

/// Conservative UV rectangles for a frame's 3D stamps. Candidate triangle
/// UV bounds are rasterized into a coarse grid and compressed into disjoint
/// rectangles, preserving coverage while avoiding one large bbox across islands.
/// Screen stamps include all depths; the shader's depth test remains authoritative.
fn uv_bounds_inside_tile(lo: Vec2, hi: Vec2) -> bool {
    lo.x >= 0.0 && lo.y >= 0.0 && hi.x <= 1.0 && hi.y <= 1.0
}

fn mark_candidate_triangle_uv(
    tiles: &mut [bool],
    mesh: &Mesh,
    triangle_indices: &std::ops::Range<u32>,
    triangle: u32,
) -> bool {
    let first_index = triangle * 3;
    if first_index < triangle_indices.start || first_index + 3 > triangle_indices.end {
        return true;
    }
    let mut lo = Vec2::splat(f32::INFINITY);
    let mut hi = Vec2::splat(f32::NEG_INFINITY);
    for k in 0..3 {
        let uv = Vec2::from(mesh.vertices[mesh.indices[triangle as usize * 3 + k] as usize].uv);
        lo = lo.min(uv);
        hi = hi.max(uv);
    }
    if !uv_bounds_inside_tile(lo, hi) {
        return false;
    }
    mark_uv_tiles(tiles, lo, hi);
    true
}

/// Conservative UV rectangles for a frame's 3D stamps. Candidate triangle
/// UV bounds are rasterized into a coarse grid and compressed into disjoint
/// rectangles, preserving coverage while avoiding one large bbox across islands.
/// Screen stamps include all depths; the shader's depth test remains authoritative.
pub(crate) fn stroke_region(
    mesh: &Mesh,
    bvh: &Bvh,
    triangle_indices: std::ops::Range<u32>,
    cam: &OrbitCam,
    size: Vec2,
    projection: BrushProjection,
    symmetry: u8,
    radius: f32,
    stamps: &[Stamp],
) -> Option<Vec<[f32; 4]>> {
    // Keep region calculation aligned with the GPU's actual stamp limit.
    let stamps = &stamps[..stamps.len().min(MAX_STAMPS)];
    if projection == BrushProjection::Uv {
        let mut tiles = [false; STROKE_REGION_GRID * STROKE_REGION_GRID];
        for s in stamps.iter().filter(|s| s[1][3] > 0.0) {
            let (uv, r) = (Vec2::new(s[4][0], s[4][1]), s[2][3]);
            let (lo, hi) = (uv - Vec2::splat(r), uv + Vec2::splat(r));
            if !uv_bounds_inside_tile(lo, hi) {
                return None;
            }
            mark_uv_tiles(&mut tiles, lo, hi);
        }
        return Some(uv_tile_rectangles_or_empty(&tiles));
    }

    let mut tiles = [false; STROKE_REGION_GRID * STROKE_REGION_GRID];
    let mut needs_full_region = false;
    match projection {
        BrushProjection::Surface => {
            // Resolve mirrored stamps before querying; no symmetry transform is
            // needed here, and the sphere test stays conservative for each disc.
            let mut spheres = [(Vec3::ZERO, 0.0); MAX_STAMPS];
            let mut sphere_count = 0;
            for s in stamps.iter().filter(|s| s[1][3] > 0.0) {
                spheres[sphere_count] = (Vec3::new(s[1][0], s[1][1], s[1][2]), s[1][3] * 1.5);
                sphere_count += 1;
            }
            let spheres = &spheres[..sphere_count];
            bvh.query(
                |lo, hi| spheres.iter().any(|&(c, r)| c.clamp(lo, hi).distance_squared(c) <= r * r),
                |t| needs_full_region |= !mark_candidate_triangle_uv(&mut tiles, mesh, &triangle_indices, t),
            );
        }
        _ => {
            // Screen box of all circles; boxes crossing the eye plane are
            // kept conservatively, while boxes wholly behind it are rejected.
            let (mut a, mut b) = (Vec2::MAX, Vec2::MIN);
            for s in stamps {
                let r = radius * s[0][2];
                a = a.min(Vec2::new(s[0][0] - r, s[0][1] - r));
                b = b.max(Vec2::new(s[0][0] + r, s[0][1] + r));
            }
            let vp = cam.view_proj(size.x / size.y.max(1.0));
            let on_screen = |lo: Vec3, hi: Vec3| {
                // Clip-space w is linear over the AABB, so its extrema occur
                // at corners. Keep boxes crossing the eye plane conservatively.
                let (mut w_min, mut w_max) = (f32::INFINITY, f32::NEG_INFINITY);
                for k in 0..8 {
                    let p = Vec3::new(
                        if k & 1 == 0 { lo.x } else { hi.x },
                        if k & 2 == 0 { lo.y } else { hi.y },
                        if k & 4 == 0 { lo.z } else { hi.z },
                    );
                    let w = (vp * p.extend(1.0)).w;
                    w_min = w_min.min(w);
                    w_max = w_max.max(w);
                }
                if w_max <= 0.0 {
                    return false;
                }
                if w_min <= 1e-4 {
                    return true;
                }
                let (mut smin, mut smax) = (Vec2::MAX, Vec2::MIN);
                for k in 0..8 {
                    let p = Vec3::new(
                        if k & 1 == 0 { lo.x } else { hi.x },
                        if k & 2 == 0 { lo.y } else { hi.y },
                        if k & 4 == 0 { lo.z } else { hi.z },
                    );
                    let c = vp * p.extend(1.0);
                    let q = (Vec2::new(c.x, -c.y) / c.w * 0.5 + 0.5) * size;
                    smin = smin.min(q);
                    smax = smax.max(q);
                }
                smin.x <= b.x && smax.x >= a.x && smin.y <= b.y && smax.y >= a.y
            };
            let mut flip = Vec3::ONE;
            if symmetry > 0 {
                flip[symmetry as usize - 1] = -1.0;
            }
            bvh.query(
                |lo, hi| {
                    on_screen(lo, hi)
                        || (symmetry > 0 && {
                            let (m, n) = (lo * flip, hi * flip);
                            on_screen(m.min(n), m.max(n))
                        })
                },
                |t| needs_full_region |= !mark_candidate_triangle_uv(&mut tiles, mesh, &triangle_indices, t),
            );
        }
    }
    if needs_full_region {
        None
    } else {
        Some(uv_tile_rectangles_or_empty(&tiles))
    }
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
    add_smart_material_inner(engine, undo, smart, at, parent, None)
}

/// Adds a smart material using a caller-owned scan image cache. This is used
/// by shelf previews so deleting one thumbnail's layers does not expire the
/// GPU textures needed by the next thumbnail.
pub(crate) fn add_smart_material_with_images(
    engine: &mut Engine,
    undo: &mut UndoStack,
    smart: &presets::SmartMaterial,
    at: usize,
    parent: Option<u64>,
    images: &std::collections::HashMap<std::path::PathBuf, FillImage>,
) -> usize {
    add_smart_material_inner(engine, undo, smart, at, parent, Some(images))
}

fn add_smart_material_inner(
    engine: &mut Engine,
    undo: &mut UndoStack,
    smart: &presets::SmartMaterial,
    at: usize,
    parent: Option<u64>,
    cached_images: Option<&std::collections::HashMap<std::path::PathBuf, FillImage>>,
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
                    if let Some(images) = cached_images {
                        if let Some(img) = images.get(&path) {
                            engine.set_fill_image(idx, ch, Some(img.clone()), undo);
                        }
                    } else if path.is_file()
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
        let _new_total = crate::startup_trace::span("PainterApp::new total (after device)");
        let rs = cc.wgpu_render_state.clone().ok_or("Material Painter requires the wgpu renderer")?;
        let gpu = rs.adapter.get_info();
        crate::crash::set_context(format!("{} ({:?}, driver {} {})", gpu.name, gpu.backend, gpu.driver, gpu.driver_info));
        crate::startup_trace::event("PainterApp::new begin (window+device already created)");
        { let _s = crate::startup_trace::span("new: theme::apply"); theme::apply(&cc.egui_ctx); }

        let prefs = { let _s = crate::startup_trace::span("new: Preferences::load"); Preferences::load() };
        let mesh = Arc::new({
            let _s = crate::startup_trace::span("new: MaterialBall.build");
            Primitive::MaterialBall.build()
        });
        let gpu_mesh = { let _s = crate::startup_trace::span("new: GpuMesh::new"); GpuMesh::new(&rs.device, &mesh) };
        let bvh = { let _s = crate::startup_trace::span("new: Bvh::from_mesh"); Bvh::from_mesh(&mesh) };
        let mut engine = {
            let _s = crate::startup_trace::span("new: Engine::new (main editor)");
            Engine::new(&rs, &gpu_mesh, (mesh.bounds_min, mesh.bounds_max), prefs.default_resolution)
        };
        {
            let _s = crate::startup_trace::span("new: Engine::add_paint_layer (initial layer)");
            engine.add_paint_layer(0, None, None, &mut UndoStack::default());
        }
        let views = {
            let _s = crate::startup_trace::span("new: ViewRenderer::new (main editor)");
            ViewRenderer::new(&rs)
        };
        let env_baker = { let _s = crate::startup_trace::span("new: EnvBaker::new"); EnvBaker::new(&rs.device) };
        // A real studio HDRI (plus the key light) when installed, else the
        // procedural studio.
        let default_env = environment::BUILTIN
            .iter()
            .position(|b| b.0 == prefs.default_hdri)
            .or_else(|| environment::BUILTIN.iter().position(|b| b.0 == "Small Studio"))
            .filter(|i| environment::available_builtins().contains(i))
            .unwrap_or(0);
        let _s_env = crate::startup_trace::span("new: env decode+upload");
        let env = env_baker.upload(&rs, &environment::decode(&EnvSource::Builtin(default_env))?);
        drop(_s_env);


        let ctx = &cc.egui_ctx;
        let load = |name: &str, img: egui::ColorImage| ctx.load_texture(name, img, egui::TextureOptions::LINEAR);
        let _s_cpu = crate::startup_trace::span("new: start CPU preview worker + placeholder textures");
        let cpu_preview_rx = start_cpu_shelf_previews(ctx);
        let placeholder = || egui::ColorImage::new([72, 72], vec![Color32::from_gray(28); 72 * 72]);
        let materials = presets::MATERIALS.iter().map(|m| load(m.name, placeholder())).collect();
        let smart = presets::SMART_MATERIALS.iter().map(|s| load(s.name, placeholder())).collect();
        let generators = GenKind::ALL.iter().map(|g| load(g.name(), placeholder())).collect();
        drop(_s_cpu);
        // Brush tips show as white shapes; textures as their grayscale tile.
        const THUMB: u32 = 96;
        let brushes = {
            let _s = crate::startup_trace::span("new: brush thumbnail render + texture handles");
            crate::brushes::BRUSH_ALPHAS
            .iter()
            .map(|b| {
                let px: Vec<Color32> = b.render(THUMB).into_iter().map(Color32::from_white_alpha).collect();
                load(b.name, egui::ColorImage::new([THUMB as usize; 2], px))
            })
            .collect()
        };
        let textures = {
            let _s = crate::startup_trace::span("new: procedural texture thumbnail render + texture handles");
            crate::textures::TEXTURES
            .iter()
            .map(|t| {
                let px: Vec<Color32> = t.render(THUMB).into_iter().map(Color32::from_gray).collect();
                load(t.name, egui::ColorImage::new([THUMB as usize; 2], px))
            })
            .collect()
        };

        let _s_sp = crate::startup_trace::span("new: ShelfPreviews::start");
        let shelf_previews = shelf_previews::ShelfPreviews::start(ctx, mesh.clone());
        drop(_s_sp);

        let mut app = PainterApp {
            rs,
            mesh,
            bvh: Arc::new(bvh),
            gpu_mesh,
            engine,
            views,
            raytracer: None,
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
            vertex_id: Default::default(),
            baked_id_source: IdSource::MeshParts,
            preview_update_at: None,
            time_composite: false,
            meter: StrokeMeter::default(),
            diagnostics: DiagnosticCapture::default(),
            engine_epoch: 0,
            bake_id_only: false,
            cam: OrbitCam::default(),
            cam2d: Cam2d::default(),
            layout: Layout::Split,
            view: ViewSettings::default(),
            env,
            env_baker,
            env_load: None,
            env_builtins: environment::available_builtins(),
            hdri_thumbs: Default::default(),
            hdri_thumb_rx: Some({
                let _s = crate::startup_trace::span("new: start HDRI thumbnail worker");
                start_hdri_thumbs(&cc.egui_ctx, environment::available_builtins())
            }),
            hdri_new_thumb: None,
            stroke: None,
            bake_settings: BakeSettings::default(),
            bake_job: None,
            baker: None,
            show_bake: false,
            shelf: Shelf { cat: ShelfCat::SmartMaterials, search: String::new(), kind: None, icon_size: 72.0, smart, materials, generators, brushes, textures },
            cpu_preview_rx: Some(cpu_preview_rx),
            shelf_previews: Some(shelf_previews),
            export: ExportDialog::new("MaterialBall".into()),
            prefs_applied: prefs.clone(),
            prefs,
            prefs_ui: PrefsUi { open: false, page: prefs::Page::General },
            graphs: { let _s = crate::startup_trace::span("new: GraphStore::new"); GraphStore::new(&cc.egui_ctx) },
            workspace: Workspace::Paint,
            docks: { let _s = crate::startup_trace::span("new: Docks::load"); dock::Docks::load() },
            drew_3d: false,
            pending: Vec::new(),
            layer_clipboard: None,
            status: "Ready. Paint with LMB, orbit with Alt+LMB or RMB, pan with MMB.".into(),
            show_help: false,
            show_about: false,
            show_welcome: false,
            welcome_tex: None,
            library: Default::default(),
            user_smarts: { let _s = crate::startup_trace::span("new: smart_io::list"); smart_io::list(&cc.egui_ctx) },
            path: Default::default(),
            bake_preview: Default::default(),
            custom_ui: Default::default(),
            spp: Default::default(),
            preview: Default::default(),
            crash_report: crate::crash::take_report(),
            frame_ms: 0.0,
            frame_peak_ms: 0.0,
            ppp: cc.egui_ctx.pixels_per_point(),
            mcp: None,
            mcp_bake_waiters: Vec::new(),
            mcp_env_waiters: Vec::new(),
            project_path: None,
            saved_serial: 0,
            confirm: None,
            allow_close: false,
            recent: { let _s = crate::startup_trace::span("new: project_io::load_recent"); project_io::load_recent() },
            autosaved_serial: 0,
            recovery: { let _s = crate::startup_trace::span("new: project_io::pending_recovery"); project_io::pending_recovery() },
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
        { let _s = crate::startup_trace::span("new: apply_startup_preferences"); app.apply_startup_preferences(ctx); }
        match {
            let _s = crate::startup_trace::span("new: McpServer::start");
            crate::mcp::McpServer::start(&cc.egui_ctx, app.prefs.mcp_port, app.prefs.mcp_on_start)
        } {
            Ok(server) => app.mcp = Some(server),
            Err(e) => app.status = e,
        }
        if app.prefs.bake_on_load {
            let _s = crate::startup_trace::span("new: start_bake (bake_on_load)");
            app.start_bake(ctx);
        }
        crate::startup_trace::event("PainterApp::new end");
        Ok(app)
    }

    // ------------------------------------------------------------------ jobs

    fn load_mesh(&mut self, mesh: Mesh, ctx: &egui::Context) {
        self.load_mesh_sized(mesh, ctx, &[]);
    }

    /// Loads `mesh`, making its new texture sets at `sizes` (per set, in
    /// order) instead of the default size.
    fn load_mesh_sized(&mut self, mesh: Mesh, ctx: &egui::Context, sizes: &[u32]) {
        if self.stroke.is_some() {
            return;
        }
        self.engine_epoch = self.engine_epoch.wrapping_add(1);
        // A bake of the old mesh can't land on the new one.
        self.bake_job = None;
        self.end_preview(false);
        self.gpu_mesh = GpuMesh::new(&self.rs.device, &mesh);
        self.bvh = Arc::new(Bvh::from_mesh(&mesh));
        if let Some(raytracer) = self.raytracer.as_mut() {
            raytracer.set_mesh(&self.rs.device, &mesh, &self.bvh);
        }
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
        self.vertex_id.shown = None;
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
                Action::AddMaskLayer(kind) => {
                    if self.selected < self.engine.layers.len() {
                        if self.engine.layers[self.selected].mask.is_none() {
                            self.engine.add_mask(self.selected, false, None, &mut self.undo);
                        }
                        use crate::engine::{MaskBlend, MaskLayer, MaskSource};
                        let g = new_generator(kind, &self.bake_settings.custom);
                        if let Some(mask) = self.engine.layers[self.selected].mask.as_mut() {
                            let blend = if mask.black_base && mask.layers.is_empty() { MaskBlend::Add } else { MaskBlend::Multiply };
                            mask.layers.push(MaskLayer { source: MaskSource::Generator(g), blend, opacity: 1.0, enabled: true });
                            self.engine.mark_dirty();
                        }
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


    fn diagnostic_context_report(&self, phase: &str) -> String {
        let layout = match self.layout {
            Layout::Split => "Split (3D + UV)",
            Layout::View3d => "3D only",
            Layout::View2d => "UV only",
        };
        let target = match self.target {
            StrokeTarget::Content => "Content",
            StrokeTarget::Mask => "Mask",
            StrokeTarget::BakeFix(_) => "Bake Fix",
        };
        let tool = match self.tool {
            Tool::Paint => "Paint",
            Tool::Eraser => "Eraser",
            Tool::Picker => "Picker",
            Tool::PolyFill => "Polygon Fill",
            Tool::Path => "Path",
        };
        let project = self.project_path.as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "(not saved)".into());
        let active_set = self.mesh.sets.get(self.sets.active)
            .map(|s| s.name.as_str())
            .unwrap_or("(missing)");
        let selected_layer = self.engine.layers.get(self.selected)
            .map(|l| l.name.as_str())
            .unwrap_or("(none)");
        let eye = self.cam.eye();

        format!(
            "MATERIAL PAINTER — DIAGNOSTIC SNAPSHOT\n\
             phase: {phase}\n\
             unix_time: {}\n\
             app_version: {}\n\
             os: {} / {}\n\
             project: {project}\n\
             mesh: {}\n\
             workspace: {}\n\
             layout: {layout}\n\
             viewport_mode: {}\n\
             ray_traced_preview: {}\n\
             viewport_3d_pixels: {:?}\n\
             viewport_uv_pixels: {:?}\n\
             pixels_per_point: {:.3}\n\
             texture_sets: {}\n\
             active_texture_set_index: {}\n\
             active_texture_set_name: {active_set}\n\
             active_texture_resolution: {}\n\
             active_layer_index: {}\n\
             active_layer_name: {selected_layer}\n\
             stroke_target: {target}\n\
             tool: {tool}\n\
             brush_projection: {}\n\
             brush_paint_through: {}\n\
             brush_angle_limit_degrees: {:.2}\n\
             brush_size_points: {:.2}\n\
             camera_eye: {:?}\n\
             camera_target: {:?}\n\
             camera_yaw_pitch_fov_radians: {:.6}, {:.6}, {:.6}\n\
             camera_distance: {:.6}\n\
             uv_camera_zoom: {:.6}\n\
             uv_camera_pan: {:?}\n\
             environment: {}\n\
             environment_intensity: {:.4}\n\
             environment_rotation_radians: {:.4}\n\
             exposure: {:.4}\n\
             key_light_enabled: {}\n\
             key_light_intensity: {:.4}\n\
             key_light_azimuth_degrees: {:.3}\n\
             key_light_elevation_degrees: {:.3}\n\
             ao_strength: {:.4}\n\
             emission_display_strength: {:.4}\n\
             background: {}\n\
             background_color_linear_rgb: {:?}\n\
             baked_normal_display_enabled: {}\n\
             uv_seams_enabled: {}\n\
             uv_wire_opacity: {:.4}\n\
             visible_layer_count: {}\n\
             total_layer_count: {}\n\
             engine_dirty: {}\n\
             F6_meter_enabled: {}\n\
             frame_time_ema_ms: {:.3}\n\
             recent_frame_peak_ms: {:.3}\n",
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs()).unwrap_or(0),
            env!("CARGO_PKG_VERSION"),
            std::env::consts::OS,
            std::env::consts::ARCH,
            self.mesh.name,
            self.workspace.name(),
            self.view.mode.name(),
            self.rt.enabled,
            self.views.view3d.size,
            self.views.view2d.size,
            self.ppp,
            self.mesh.sets.len(),
            self.sets.active,
            self.engine.res,
            self.selected,
            self.brush.projection.name(),
            self.brush.paint_through,
            self.brush.angle_limit,
            self.brush.size,
            eye,
            self.cam.target,
            self.cam.yaw,
            self.cam.pitch,
            self.cam.fov_y,
            self.cam.dist,
            self.cam2d.zoom,
            self.cam2d.pan,
            self.env.source.name(),
            self.view.env_intensity,
            self.view.env_rotation,
            self.view.exposure,
            self.view.key.on,
            self.view.key.intensity,
            self.view.key.azimuth,
            self.view.key.elevation,
            self.view.ao_strength,
            self.view.emission,
            self.view.background.name(),
            self.view.bg_color,
            self.view.use_baked_normal,
            self.view.show_seams_2d,
            self.view.uv_wire_opacity,
            self.engine.layers.iter().filter(|l| l.visible).count(),
            self.engine.layers.len(),
            self.engine.dirty,
            self.meter.on,
            self.frame_ms,
            self.frame_peak_ms,
        )
    }

    fn diagnostic_mesh_report(&self) -> String {
        use std::collections::HashMap;

        let mut uv_min = Vec2::splat(f32::INFINITY);
        let mut uv_max = Vec2::splat(f32::NEG_INFINITY);
        let mut uv_outside_triangles = 0u32;
        let mut uv_degenerate_triangles = 0u32;
        let mut uv_positive_triangles = 0u32;
        let mut uv_negative_triangles = 0u32;
        let mut geometry_degenerate_triangles = 0u32;
        let mut zero_vertex_normal_corners = 0u32;
        let mut normals_aligned = 0u32;
        let mut normals_opposed = 0u32;
        let mut normals_mixed = 0u32;
        let mut exact_uv_keys: HashMap<[i64; 6], u32> = HashMap::new();
        let mut exact_uv_keys_by_set: HashMap<(usize, [i64; 6]), u32> = HashMap::new();
        // [triangles, zero normal corners, aligned, opposed, mixed]
        let mut object_normal_stats = vec![[0u32; 5]; self.mesh.objects.len()];

        for tri in 0..self.mesh.triangle_count() {
            let base = tri * 3;
            let v = [
                self.mesh.vertices[self.mesh.indices[base] as usize],
                self.mesh.vertices[self.mesh.indices[base + 1] as usize],
                self.mesh.vertices[self.mesh.indices[base + 2] as usize],
            ];
            let p = v.map(|v| Vec3::from_array(v.pos));
            let uv = v.map(|v| Vec2::from_array(v.uv));
            for t in uv {
                uv_min = uv_min.min(t);
                uv_max = uv_max.max(t);
            }

            let signed_uv_area = (uv[1].x - uv[0].x) * (uv[2].y - uv[0].y)
                - (uv[1].y - uv[0].y) * (uv[2].x - uv[0].x);
            if signed_uv_area.abs() < 1.0e-10 {
                uv_degenerate_triangles += 1;
            } else if signed_uv_area > 0.0 {
                uv_positive_triangles += 1;
            } else {
                uv_negative_triangles += 1;
            }
            if uv.iter().any(|t| t.x < 0.0 || t.x > 1.0 || t.y < 0.0 || t.y > 1.0) {
                uv_outside_triangles += 1;
            }

            let face = (p[1] - p[0]).cross(p[2] - p[0]);
            if face.length() <= 1.0e-10 {
                geometry_degenerate_triangles += 1;
            }
            let face_n = face.normalize_or_zero();
            let mut triangle_zero_normals = 0u32;
            let alignments = v.map(|vertex| {
                let normal = Vec3::from_array(vertex.normal);
                if normal.length_squared() <= 1.0e-12 {
                    zero_vertex_normal_corners += 1;
                    triangle_zero_normals += 1;
                }
                normal.normalize_or_zero().dot(face_n)
            });
            let normal_alignment = alignments.iter().sum::<f32>() / 3.0;
            let mixed_or_opposed = alignments.iter().any(|&n| n < -0.25)
                && alignments.iter().any(|&n| n > 0.25);
            let normal_class = if mixed_or_opposed || normal_alignment.abs() <= 0.25 {
                normals_mixed += 1;
                4
            } else if normal_alignment < -0.25 {
                normals_opposed += 1;
                3
            } else {
                normals_aligned += 1;
                2
            };
            if let Some(&object) = self.mesh.tri_object.get(tri)
                && let Some(stats) = object_normal_stats.get_mut(object as usize)
            {
                stats[0] += 1;
                stats[1] += triangle_zero_normals;
                stats[normal_class] += 1;
            }

            // Exact UV triangle identity (quantized to 1e-6), independent of vertex order.
            // This detects exact stacks, not partial triangle intersections.
            let mut q = uv.map(|t| [(t.x * 1_000_000.0).round() as i64, (t.y * 1_000_000.0).round() as i64]);
            q.sort_unstable();
            let key = [q[0][0], q[0][1], q[1][0], q[1][1], q[2][0], q[2][1]];
            *exact_uv_keys.entry(key).or_default() += 1;
            *exact_uv_keys_by_set.entry((self.mesh.set_of(tri as u32), key)).or_default() += 1;
        }

        let stacked_groups = exact_uv_keys.values().filter(|&&n| n > 1).count();
        let stacked_faces = exact_uv_keys.values().filter(|&&n| n > 1).map(|&n| n as usize).sum::<usize>();
        let same_set_stacked_groups = exact_uv_keys_by_set.values().filter(|&&n| n > 1).count();
        let same_set_stacked_faces = exact_uv_keys_by_set.values().filter(|&&n| n > 1).map(|&n| n as usize).sum::<usize>();
        let max_stack_depth = exact_uv_keys.values().copied().max().unwrap_or(0);

        let mut out = format!(
            "\nMESH / UV / NORMAL SUMMARY\n\
             vertices_after_tangent_split: {}\n\
             triangles: {}\n\
             indices: {}\n\
             mesh_texture_sets: {}\n\
             uv_min: {:?}\n\
             uv_max: {:?}\n\
             triangles_with_any_uv_outside_0_1: {}\n\
             uv_degenerate_triangles: {}\n\
             uv_positive_winding_triangles: {}\n\
             uv_negative_winding_triangles: {}\n\
             geometric_degenerate_triangles: {}\n\
             zero_vertex_normal_corners: {}\n\
             triangle_normals_aligned_with_vertex_normals: {}\n\
             triangle_normals_opposed_to_vertex_normals: {}\n\
             triangle_normals_mixed_or_uncertain: {}\n\
             exact_duplicate_uv_triangle_groups_global: {}\n\
             triangles_in_exact_uv_groups_global: {}\n\
             exact_duplicate_uv_triangle_groups_same_texture_set: {}\n\
             triangles_in_exact_uv_groups_same_texture_set: {}\n\
             maximum_exact_uv_stack_depth: {}\n\
             exact_stack_detection_tolerance: 1e-6 UV units\n\
             note: exact duplicate UV triangle detection does not detect partial overlaps\n",
            self.mesh.vertices.len(),
            self.mesh.triangle_count(),
            self.mesh.indices.len(),
            self.mesh.sets.len(),
            uv_min,
            uv_max,
            uv_outside_triangles,
            uv_degenerate_triangles,
            uv_positive_triangles,
            uv_negative_triangles,
            geometry_degenerate_triangles,
            zero_vertex_normal_corners,
            normals_aligned,
            normals_opposed,
            normals_mixed,
            stacked_groups,
            stacked_faces,
            same_set_stacked_groups,
            same_set_stacked_faces,
            max_stack_depth,
        );

        out.push_str("texture_set_ranges:\n");
        for (index, set) in self.mesh.sets.iter().enumerate() {
            let state = if index == self.sets.active {
                "ACTIVE"
            } else if self.sets.parked.get(index).and_then(Option::as_ref).is_some() {
                "parked engine loaded"
            } else {
                "engine not currently parked"
            };
            out.push_str(&format!(
                "  [{}] {} | triangles {}..{} (count {}) | {}\n",
                index, set.name, set.tris.start, set.tris.end, set.tris.end - set.tris.start, state
            ));
        }

        out.push_str("normal_provenance_by_object:\n");
        for (index, name) in self.mesh.objects.iter().enumerate() {
            let ufbx_generated = self.mesh.object_ufbx_generated_normals.get(index).copied();
            let reversed = self.mesh.object_reversed_winding.get(index).copied();
            let negative_transform = self.mesh.object_negative_transform_determinant.get(index).copied();
            let smooth_faces = self.mesh.object_smooth_faces.get(index).copied();
            let face_smoothing_entries = self.mesh.object_face_smoothing_entries.get(index).copied();
            let smooth_edges = self.mesh.object_smooth_edges.get(index).copied();
            let edge_smoothing_entries = self.mesh.object_edge_smoothing_entries.get(index).copied();
            let fallback = self.mesh.object_fallback_normals_generated.get(index).copied();
            let fmt = |value: Option<bool>| match value {
                Some(true) => "yes",
                Some(false) => "no",
                None => "unknown",
            };
            let stats = object_normal_stats.get(index).copied().unwrap_or([0; 5]);
            out.push_str(&format!(
                "  [{}] {} | triangles={} aligned={} opposed={} mixed={} zero_normal_corners={} | soft_faces={}/{} soft_edges={}/{} | ufbx_generated_normals={} | ufbx_reversed_winding={} | negative_transform_determinant={} | importer_fallback_normals={}\n",
                index, name, stats[0], stats[2], stats[3], stats[4], stats[1],
                smooth_faces.map_or_else(|| "unknown".to_string(), |v| v.to_string()),
                face_smoothing_entries.map_or_else(|| "unknown".to_string(), |v| v.to_string()),
                smooth_edges.map_or_else(|| "unknown".to_string(), |v| v.to_string()),
                edge_smoothing_entries.map_or_else(|| "unknown".to_string(), |v| v.to_string()),
                fmt(ufbx_generated), fmt(reversed), fmt(negative_transform), fmt(fallback)
            ));
        }
        if self.mesh.objects.is_empty() {
            out.push_str("  (no imported object metadata; primitive or saved mesh)\n");
        }
        out
    }

    fn record_diagnostic_ray_samples(&mut self, raw: &[([f32; 4], f32)]) {
        if !self.diagnostics.active {
            return;
        }
        let [width, height] = self.views.view3d.size;
        let size = Vec2::new(width as f32, height as f32);

        for (stamp, _) in raw {
            if self.diagnostics.samples.len() >= 256 {
                self.diagnostics.dropped_samples = self.diagnostics.dropped_samples.saturating_add(1);
                continue;
            }
            let (origin, direction) = self.cam.ray(Vec2::new(stamp[0], stamp[1]), size);
            let nearest = self.bvh.raycast(origin, direction);
            let mut hits = Vec::new();
            self.bvh.all_hits(origin, direction, FAR, &mut hits);
            hits.sort_by(|a, b| a.0.total_cmp(&b.0));

            let nearest_text = match nearest {
                Some(hit) => {
                    let tri = hit.tri as usize;
                    let base = tri * 3;
                    let v = [
                        self.mesh.vertices[self.mesh.indices[base] as usize],
                        self.mesh.vertices[self.mesh.indices[base + 1] as usize],
                        self.mesh.vertices[self.mesh.indices[base + 2] as usize],
                    ];
                    let p = v.map(|v| Vec3::from_array(v.pos));
                    let face_n = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
                    let (shading_n, uv, uv_scale) = surface_at(&self.mesh, hit.tri, hit.bary);
                    let alignment = v.iter()
                        .map(|vertex| Vec3::from_array(vertex.normal).normalize_or_zero().dot(face_n))
                        .sum::<f32>() / 3.0;
                    let facing = face_n.dot(-direction);
                    let object = self.mesh.tri_object.get(tri).copied().unwrap_or(u32::MAX);
                    let object_name = self.mesh.objects.get(object as usize).map(String::as_str).unwrap_or("(unknown)");
                    let set = self.mesh.set_of(hit.tri);
                    let set_name = self.mesh.sets.get(set).map(|s| s.name.as_str()).unwrap_or("(missing)");
                    format!(
                        "tri={} object={}({}) set={}({}) t={:.6} bary={:?} uv={:?} geomN={:?} shadeN={:?} NdotGeom={:.4} geomNdotToCamera={:.4} uvScale={:.6}",
                        hit.tri, object, object_name, set, set_name, (hit.pos - origin).length(), hit.bary, uv, face_n, shading_n, alignment, facing, uv_scale
                    )
                }
                None => "no hit".into(),
            };

            let candidates = hits.iter().take(8).map(|(distance, tri)| {
                let base = *tri as usize * 3;
                let v = [
                    self.mesh.vertices[self.mesh.indices[base] as usize],
                    self.mesh.vertices[self.mesh.indices[base + 1] as usize],
                    self.mesh.vertices[self.mesh.indices[base + 2] as usize],
                ];
                let p = v.map(|v| Vec3::from_array(v.pos));
                let uv = v.map(|v| Vec2::from_array(v.uv));
                let hit_pos = origin + direction * *distance;
                let e0 = p[1] - p[0];
                let e1 = p[2] - p[0];
                let vp = hit_pos - p[0];
                let d00 = e0.dot(e0);
                let d01 = e0.dot(e1);
                let d11 = e1.dot(e1);
                let d20 = vp.dot(e0);
                let d21 = vp.dot(e1);
                let denominator = d00 * d11 - d01 * d01;
                let (bary, exact_uv) = if denominator.abs() > 1.0e-12 {
                    let b = (d11 * d20 - d01 * d21) / denominator;
                    let c = (d00 * d21 - d01 * d20) / denominator;
                    let bary = Vec3::new(1.0 - b - c, b, c);
                    let exact_uv = uv[0] * bary.x + uv[1] * bary.y + uv[2] * bary.z;
                    (bary, exact_uv)
                } else {
                    (Vec3::ZERO, (uv[0] + uv[1] + uv[2]) / 3.0)
                };
                let face_n = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
                let shading_n = (Vec3::from_array(v[0].normal) * bary.x
                    + Vec3::from_array(v[1].normal) * bary.y
                    + Vec3::from_array(v[2].normal) * bary.z).normalize_or_zero();
                let n_dot_face = shading_n.dot(face_n);
                let facing = face_n.dot(-direction);
                let object = self.mesh.tri_object.get(*tri as usize).copied().unwrap_or(u32::MAX);
                let object_name = self.mesh.objects.get(object as usize).map(String::as_str).unwrap_or("(unknown)");
                let set = self.mesh.set_of(*tri);
                let set_name = self.mesh.sets.get(set).map(|s| s.name.as_str()).unwrap_or("(missing)");
                format!(
                    "tri={} object={}({}) set={}({}) t={:.6} bary={:?} uv={:?} geomNdotToCamera={:.4} shadeNdotGeomN={:.4}",
                    tri, object, object_name, set, set_name, distance, bary, exact_uv, facing, n_dot_face
                )
            }).collect::<Vec<_>>().join(" ; ");

            self.diagnostics.samples.push(format!(
                "sample {} cursor_px=({:.2},{:.2}) projection={} paint_through={} angle_limit_deg={:.2} active_set={} ray_origin={:?} ray_dir={:?}\n  nearest: {}\n  all_intersections={} first_up_to_8=[{}]",
                self.diagnostics.samples.len() + 1, stamp[0], stamp[1],
                self.brush.projection.name(), self.brush.paint_through, self.brush.angle_limit,
                self.sets.active, origin, direction,
                nearest_text, hits.len(), candidates
            ));
        }
    }

    fn save_diagnostic_report(&mut self, ctx: &egui::Context, stem: &str, report: String) {
        let timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs()).unwrap_or(0);
        let file_name = format!("{stem}_{timestamp}.txt");
        let path = rfd::FileDialog::new()
            .add_filter("Text report", &["txt"])
            .set_file_name(&file_name)
            .save_file();
        match path {
            Some(path) => match std::fs::write(&path, report.as_bytes()) {
                Ok(()) => self.status = format!("Diagnostic report saved to {}.", path.display()),
                Err(error) => self.status = format!("Could not save diagnostic report: {error}"),
            },
            None => self.status = "Diagnostic report not saved; report copied to clipboard.".into(),
        }
        // Queue the clipboard update only after the native save dialog has closed.
        ctx.copy_text(report);
    }

    fn start_diagnostic_capture(&mut self) {
        let mut context = self.diagnostic_context_report("ray capture started");
        context.push_str(&self.diagnostic_mesh_report());
        context.push_str("\nRAY-HIT CAPTURE\nRecord only 3D brush stamp positions. Maximum retained samples: 256.\n");
        context.push_str(&format!(
            "brush_projection_at_start: {}\npaint_through_at_start: {}\nangle_limit_at_start: {:.3}\n",
            self.brush.projection.name(), self.brush.paint_through, self.brush.angle_limit
        ));
        self.diagnostics = DiagnosticCapture {
            active: true,
            started_context: context,
            samples: Vec::new(),
            dropped_samples: 0,
        };
        self.status = "Ctrl+F2 capture ON. Paint both sides in 3D, then press Ctrl+F2 again to save the trace.".into();
    }

    fn stop_diagnostic_capture(&mut self, ctx: &egui::Context) {
        let mut capture = std::mem::take(&mut self.diagnostics);
        capture.active = false;
        let mut report = capture.started_context;
        report.push_str(&format!(
            "\nRAY CAPTURE SUMMARY\nretained_samples: {}\ndropped_samples_after_limit: {}\n",
            capture.samples.len(), capture.dropped_samples
        ));
        for sample in capture.samples {
            report.push_str(&sample);
            report.push('\n');
        }
        report.push_str("\nCAPTURE END STATE\n");
        report.push_str(&self.diagnostic_context_report("ray capture stopped"));
        self.save_diagnostic_report(ctx, "material_painter_ray_trace", report);
    }

    fn shortcuts(&mut self, ctx: &egui::Context) {
        use egui::{Key, KeyboardShortcut, Modifiers};
        let mut capture_view_snapshot = false;
        let mut toggle_ray_capture = false;
        ctx.input_mut(|i| {
            if i.consume_shortcut(&KeyboardShortcut::new(Modifiers::CTRL, Key::F1)) {
                capture_view_snapshot = true;
            }
            if i.consume_shortcut(&KeyboardShortcut::new(Modifiers::CTRL, Key::F2)) {
                toggle_ray_capture = true;
            }
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
        if capture_view_snapshot {
            let mut report = self.diagnostic_context_report("viewport snapshot");
            report.push_str("\nNOTE: capture once in Material mode and again in Base Color mode when diagnosing a black UV viewport.\n");
            self.save_diagnostic_report(ctx, "material_painter_view_snapshot", report);
        }
        if toggle_ray_capture {
            if self.diagnostics.active {
                self.stop_diagnostic_capture(ctx);
            } else {
                self.start_diagnostic_capture();
            }
        }
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
            if layer_keys && i.consume_shortcut(&KeyboardShortcut::new(Modifiers::default(), Key::Delete))
                && self.selected < self.engine.layers.len() {
                    self.pending.push(Action::Delete);
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
            if i.key_pressed(Key::F6) {
                self.meter = StrokeMeter { on: !self.meter.on, ..Default::default() };
                self.status = if self.meter.on {
                    "F6 meter on (CPU times, bottom right). Paint, then press F6 again to stop.".into()
                } else {
                    "F6 meter off.".into()
                };
            }
            if i.key_pressed(Key::F5) {
                self.preview_update_at = None;
                self.engine.mark_dirty();
                self.time_composite = true;
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
    /// paint viewport is hidden. While a control is held the rebuild waits.
    fn service_preview_debounce(&mut self, ctx: &egui::Context) {
        let Some(deadline) = self.preview_update_at else { return; };
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            if ctx.input(|i| i.pointer.any_down()) {
                self.preview_update_at = Some(Instant::now() + MATERIAL_PREVIEW_DELAY);
                ctx.request_repaint_after(MATERIAL_PREVIEW_DELAY);
            } else {
                self.preview_update_at = None;
                self.engine.mark_dirty();
                ctx.request_repaint();
            }
        } else {
            ctx.request_repaint_after(remaining);
        }
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
                    .on_hover_text("What the 3D viewport shows (M: material, C: next channel)");
                if matches!(self.layout, Layout::Split | Layout::View2d) {
                    ui.separator();
                    egui::ComboBox::from_id_salt("uvviewmode")
                        .selected_text(self.mode_name(self.view.uv_mode))
                        .width(130.0)
                        .height(600.0)
                        .show_ui(ui, |ui| {
                            ui.label(egui::RichText::new("MATERIAL").small().color(theme::TEXT_DIM));
                            for m in ViewMode::CHANNELS {
                                ui.selectable_value(&mut self.view.uv_mode, m, m.name());
                            }
                            ui.separator();
                            ui.label(egui::RichText::new("MESH MAPS").small().color(theme::TEXT_DIM));
                            for m in ViewMode::all().into_iter().skip(ViewMode::CHANNELS.len()) {
                                ui.selectable_value(&mut self.view.uv_mode, m, m.name());
                            }
                            if !self.bake_settings.custom.is_empty() {
                                ui.separator();
                                ui.label(egui::RichText::new("CUSTOM MAPS").small().color(theme::TEXT_DIM));
                            }
                            for map in &self.bake_settings.custom {
                                ui.selectable_value(&mut self.view.uv_mode, ViewMode::Custom(map.id), &map.name);
                            }
                        })
                        .response
                        .on_hover_text("What the UV viewport shows");
                    ui.label(egui::RichText::new("UV display").color(theme::TEXT_DIM));
                }
                // Color selection is consolidated in the Color popup at the right end
                // of this toolbar; do not show a second, competing color chip here.
                ui.label(egui::RichText::new("3D display").color(theme::TEXT_DIM));
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
                if self.diagnostics.active {
                    ui.label(egui::RichText::new("RAY CAPTURE ON  ·  Ctrl+F2 TO STOP").strong().color(theme::ACCENT));
                }
                let mb = |b: u64| b as f64 / (1024.0 * 1024.0);
                ui.label(
                    egui::RichText::new(format!(
                        "{}{:.1} ms (peak {:.0})  │  GPU est. {:.0} MB  │  Undo {:.1} MB RAM ({} steps)  │  {}²  │  {} tris",
                        if self.meter.on { format!("{}  │  ", self.meter.text(
                            self.engine.res,
                            self.engine.last_paint_scissor_pixels,
                            self.engine.last_paint_scissor_rects,
                            self.engine.last_scissor_pixels,
                            self.engine.last_scissor_bbox_pixels,
                            self.engine.last_scissor_rects,
                            self.engine.last_composite_layer_passes,
                            self.engine.last_composite_folder_passes,
                            self.engine.last_mask_cache_hits,
                            self.engine.last_mask_cache_misses,
                            self.engine.last_mask_gpu_passes
                        )) } else { String::new() },
                        self.frame_ms,
                        self.frame_peak_ms,
                        mb(self.engine.vram_bytes() + self.sets.parked.iter().flatten().map(|p| p.engine.vram_bytes()).sum::<u64>()),
                        mb(self.undo.ram_bytes() as u64),
                        self.undo.undo.len(),
                        self.engine.res,
                        self.mesh.triangle_count(),
                    ))
                    .color(theme::TEXT_DIM)
                    .small(),
                );
                if let Some(server) = &self.mcp
                    && server.enabled.load(std::sync::atomic::Ordering::Relaxed) {
                        ui.label(egui::RichText::new(format!("MCP :{}", server.port)).color(theme::ACCENT).small())
                            .on_hover_text(format!("AI agents can connect at {}", server.url()));
                    }
                if let Some((p, what)) = self.bake_progress() {
                    ui.add(egui::ProgressBar::new(p).desired_width(140.0).text(format!("{what} {:.0}%", p * 100.0)));
                }
            });
        });
    }

    // ------------------------------------------------------------------ layers

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
                    if ui.small_button("Change…").clicked()
                        && let Some(p) = rfd::FileDialog::new().add_filter("Mesh (FBX, OBJ)", &crate::import::EXTENSIONS).pick_file() {
                            s.high_poly = Some(p);
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

    // ------------------------------------------------------------------ properties

    // ------------------------------------------------------------------ shelf

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
        if b.follow
            && let Some(d) = dir {
                angle += d.y.atan2(d.x);
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
        let began = Instant::now();
        let begun = self.begin_stroke_on_target(kind, at, erase);
        if self.meter.on && begun {
            self.meter.begin_cpu.add(began.elapsed().as_secs_f64() * 1000.0);
            self.meter.after_pending = false;
        }
        if begun {
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
        if kind == ViewKind::V3d {
            self.record_diagnostic_ray_samples(raw);
        }
        let su = self.stamp_uniform(kind);
        let resolve_started = self.meter.on.then(Instant::now);
        let stamps = match kind {
            ViewKind::V3d => {
                let [w, h] = self.views.view3d.size;
                let radius = self.brush.size * 0.5 * self.ppp;
                resolve_stamps(&self.mesh, &self.bvh, &self.cam, Vec2::new(w as f32, h as f32), self.brush.projection, self.brush.symmetry, radius, raw)
            }
            // The 2D view is the texture: stamps land where they are drawn.
            ViewKind::V2d => resolve_stamps(&self.mesh, &self.bvh, &self.cam, Vec2::ONE, BrushProjection::Screen, 0, 0.0, raw),
        };
        if let Some(started) = resolve_started {
            self.meter.resolve_cpu.add(started.elapsed().as_secs_f64() * 1000.0);
        }
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
        let region_started = self.meter.on.then(Instant::now);
        let region = match kind {
            ViewKind::V3d => {
                let [w, h] = self.views.view3d.size;
                let radius = self.brush.size * 0.5 * self.ppp;
                let size = Vec2::new(w as f32, h as f32);
                stroke_region(&self.mesh, &self.bvh, self.engine.triangles(), &self.cam, size, self.brush.projection, self.brush.symmetry, radius, &stamps)
            }
            ViewKind::V2d => {
                // The 2D view shows the texture: the stamps' own UV squares.
                let [w, h] = self.views.view2d.size;
                let center = Vec2::new(w as f32, h as f32) * 0.5;
                let zoom = self.cam2d.zoom.max(1.0);
                let radius = self.brush.size * 0.5 * self.ppp;
                stamps
                    .iter()
                    .fold(None, |acc, s| {
                        let uv = (Vec2::new(s[0][0], s[0][1]) - center - self.cam2d.pan) / zoom + Vec2::splat(0.5);
                        let r = radius * s[0][2] / zoom;
                        Some(grow(acc, uv - Vec2::splat(r), uv + Vec2::splat(r)))
                    })
                    .map(|region| vec![region])
            }
        };
        if let Some(started) = region_started {
            self.meter.region_cpu.add(started.elapsed().as_secs_f64() * 1000.0);
        }
        let stamp_started = Instant::now();
        if kind == ViewKind::V3d && self.brush.projection == BrushProjection::Screen {
            if self.meter.on {
                self.engine.stroke_step_screen_profiled(
                    &self.gpu_mesh,
                    &stamps,
                    su,
                    self.views.view3d.depth_view(),
                    &paint,
                    region,
                );
            } else {
                self.engine.stroke_step_screen(
                    &self.gpu_mesh,
                    &stamps,
                    su,
                    self.views.view3d.depth_view(),
                    &paint,
                    region,
                );
            }
        } else if self.meter.on {
            self.engine.stroke_step_profiled(&stamps, su, self.views.view3d.depth_view(), &paint, region);
        } else {
            self.engine.stroke_step(&stamps, su, self.views.view3d.depth_view(), &paint, region);
        }
        if self.meter.on {
            self.meter.steps_now += 1;
            self.meter.step_cpu.add(stamp_started.elapsed().as_secs_f64() * 1000.0);
        }
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
            let ending = Instant::now();
            if let Some(entry) = self.engine.end_stroke() {
                self.undo.push("Paint stroke", crate::undo::Edit::Stroke(entry));
            }
            if self.meter.on {
                self.meter.end_cpu.add(ending.elapsed().as_secs_f64() * 1000.0);
                self.meter.after_pending = true;
            }
        }
    }

    // ------------------------------------------------------------------ windows

}

impl PainterApp {
    fn poll_cpu_shelf_previews(&mut self, ctx: &egui::Context) {
        let Some(rx) = self.cpu_preview_rx.as_ref() else { return };
        let mut finished = false;
        loop {
            match rx.try_recv() {
                Ok(update) => {
                    let handle = match update.target {
                        CpuShelfPreviewTarget::Smart(i) => &mut self.shelf.smart[i],
                        CpuShelfPreviewTarget::Material(i) => &mut self.shelf.materials[i],
                        CpuShelfPreviewTarget::Generator(i) => &mut self.shelf.generators[i],
                    };
                    {
                        let _s = crate::startup_trace::span("cpu preview: TextureHandle::set (UI thread)");
                        handle.set(update.image, egui::TextureOptions::LINEAR);
                    }
                    crate::startup_trace::event("cpu preview applied");
                    ctx.request_repaint();
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    crate::startup_trace::event("all cpu previews applied (worker finished)");
                    finished = true;
                    break;
                }
            }
        }
        if finished {
            self.cpu_preview_rx = None;
        }
    }
}

impl eframe::App for PainterApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let _trace_frame = crate::startup_trace::frame();
        let start = Instant::now();
        let ctx = ui.ctx().clone();
        self.ppp = ctx.pixels_per_point();
        let _ = self.rs.device.poll(wgpu::PollType::Poll);
        {
            let _s = crate::startup_trace::span("frame: async job polling");
            self.undo.update();
            self.poll_bake(&ctx);
            self.poll_spp_open(&ctx);
            self.update_preview(&ctx);
            self.poll_env();
            self.poll_hdri_thumbs(&ctx);
            self.poll_library(&ctx);
            self.poll_cpu_shelf_previews(&ctx);
        }
        // Closing the window with unsaved changes asks first.
        if ctx.input(|i| i.viewport().close_requested()) && !self.allow_close {
            if self.is_modified() {
                ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
                self.confirm = Some(Action::Exit);
            } else {
                project_io::remove_autosave();
            }
        }
        {
            let _s = crate::startup_trace::span("frame: recovery/confirm/autosave UI");
            self.recovery_dialog(&ctx);
            self.confirm_dialog(&ctx);
            self.autosave_if_due();
        }
        let title = format!(
            "{}{} — Material Painter",
            self.project_path.as_ref().and_then(|p| p.file_stem()).map_or("Untitled".to_string(), |n| n.to_string_lossy().into_owned()),
            if self.is_modified() { " *" } else { "" }
        );
        if title != self.window_title {
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
            self.window_title = title;
        }
        let pumped = self.shelf_previews.is_some();
        let keep = {
            let _s = crate::startup_trace::span("frame: shelf pump (UI thread)");
            self.shelf_previews.as_mut().is_none_or(|p| p.pump(&self.rs, &ctx, &mut self.shelf, &self.env_baker, &self.gpu_mesh, &self.engine))
        };
        if pumped && !keep {
            self.shelf_previews = None;
            crate::startup_trace::shelf_done();
        }
        {
            let _s = crate::startup_trace::span("frame: shortcuts/actions/MCP");
            self.path_keys(&ctx);
            self.shortcuts(&ctx);
            self.process_actions(&ctx);
            self.process_mcp(&ctx);
        }

        {
            let _s = crate::startup_trace::span("frame: menu and top/bottom toolbars");
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
        }

        // Panel actions (add layer, etc.) apply before this frame's GPU work.
        {
            let _s = crate::startup_trace::span("frame: process panels and update previews");
            self.process_actions(&ctx);
            self.sync_brush_tip();
            self.sync_brush_grain();
            self.update_graphs();
            self.update_preview_shader();
            self.update_cage_preview(&ctx);
            self.sync_vertex_color_preview(&ctx);
        }
        // The ID mask panel says again, below, whether it is open.
        self.vertex_id.panel = None;
        {
            let _s = crate::startup_trace::span("frame: workspace/dock UI");
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
        }
        if self.workspace != Workspace::Designer {
            self.graphs.editor.hovered = false;
        }
        self.views_end_frame(&ctx);
        self.docks.save_if_changed(&ctx);

        {
            let _s = crate::startup_trace::span("frame: secondary windows and preview services");
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
        }

        // Property edits become one undo step once the interaction ends: no
        // button held (slider drags) and no text field focused (typing).
        let idle = !ctx.input(|i| i.pointer.any_down()) && ctx.memory(|m| m.focused().is_none());
        if idle && self.stroke.is_none() {
            self.engine.commit_props(&mut self.undo);
        }
        let ms = start.elapsed().as_secs_f32() * 1000.0;
        self.frame_ms = self.frame_ms * 0.9 + ms * 0.1;
        self.frame_peak_ms = ms.max(self.frame_peak_ms * 0.97);
    }
}



#[cfg(test)]
mod stroke_region_tile_tests {
    use super::*;

    const CELLS: usize = STROKE_REGION_GRID * STROKE_REGION_GRID;

    #[test]
    fn tile_rectangles_cover_each_marked_cell_exactly_once() {
        let mut tiles = [false; CELLS];
        mark_uv_tiles(&mut tiles, Vec2::new(0.02, 0.03), Vec2::new(0.16, 0.12));
        mark_uv_tiles(&mut tiles, Vec2::new(0.72, 0.68), Vec2::new(0.91, 0.87));
        let rects = uv_tile_rectangles(&tiles).expect("occupied tiles produce rectangles");
        let mut covered = [false; CELLS];

        for [u0, v0, u1, v1] in rects {
            let x0 = (u0 * STROKE_REGION_GRID as f32).round() as usize;
            let y0 = (v0 * STROKE_REGION_GRID as f32).round() as usize;
            let x1 = (u1 * STROKE_REGION_GRID as f32).round() as usize;
            let y1 = (v1 * STROKE_REGION_GRID as f32).round() as usize;
            for y in y0..y1 {
                for x in x0..x1 {
                    let i = y * STROKE_REGION_GRID + x;
                    assert!(!covered[i], "tile rectangles overlap at ({x}, {y})");
                    covered[i] = true;
                }
            }
        }
        assert_eq!(covered, tiles);
    }

    #[test]
    fn empty_tile_set_is_a_known_empty_region() {
        assert!(uv_tile_rectangles_or_empty(&[false; CELLS]).is_empty());
    }

    #[test]
    fn uv_bounds_outside_tile_require_full_region_fallback() {
        assert!(uv_bounds_inside_tile(Vec2::ZERO, Vec2::ONE));
        assert!(!uv_bounds_inside_tile(Vec2::new(-0.01, 0.2), Vec2::new(0.1, 0.3)));
        assert!(!uv_bounds_inside_tile(Vec2::new(0.8, 0.2), Vec2::new(1.01, 0.3)));
    }

    #[test]
    fn uv_bounds_at_zero_and_one_stay_inside_the_grid() {
        let mut tiles = [false; CELLS];
        mark_uv_tiles(&mut tiles, Vec2::ZERO, Vec2::ZERO);
        mark_uv_tiles(&mut tiles, Vec2::ONE, Vec2::ONE);
        assert!(tiles[0]);
        assert!(tiles[CELLS - 1]);
        let rects = uv_tile_rectangles(&tiles).expect("edge tiles are occupied");
        assert!(rects.iter().all(|r| r.iter().all(|v| (0.0..=1.0).contains(v))));
    }

    #[test]
    fn fragmented_grid_falls_back_to_one_conservative_rectangle() {
        let mut tiles = [false; CELLS];
        for y in (0..STROKE_REGION_GRID).step_by(2) {
            for x in (0..STROKE_REGION_GRID).step_by(2) {
                tiles[y * STROKE_REGION_GRID + x] = true;
            }
        }
        let rects = uv_tile_rectangles(&tiles).expect("checkerboard has occupied tiles");
        assert_eq!(rects.len(), 1);
        let [x0, y0, x1, y1] = rects[0];
        assert_eq!([x0, y0], [0.0, 0.0]);
        assert!((x1 - 63.0 / STROKE_REGION_GRID as f32).abs() < f32::EPSILON);
        assert!((y1 - 63.0 / STROKE_REGION_GRID as f32).abs() < f32::EPSILON);
    }
}
