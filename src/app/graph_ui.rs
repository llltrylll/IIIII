//! Material graphs in the app: the graphs behind graph fill layers and graph
//! masks (evaluated and pushed into their layers), the node editor window,
//! the custom node code editor, hot reload of user nodes and the shelf's
//! saved graphs.

use super::{Action, PainterApp, labeled, section, theme};
use crate::engine::{Content, FillParams, MAPS};
use crate::gpu::Tex;
use crate::graph::{self, Graph, Library, ParamKind, Runtime, Value};
use crate::undo::StrokeTarget;
use eframe::egui;
use egui::{Color32, Pos2, Rect, Sense, Stroke, pos2, vec2};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, mpsc};

pub(super) struct GraphDoc {
    pub id: u64,
    pub graph: Graph,
    rt: Runtime,
    /// Edited since the last evaluation.
    pub(super) dirty: bool,
    /// Its layers need the outputs again (new layer, new texture set).
    push: bool,
    maps_seen: u64,
    /// A preview shader's graph (shading nodes), not a texture graph.
    pub shading: bool,
    /// Last evaluated without the editor showing it: only the outputs kept
    /// their textures (the editor's thumbnails need a new evaluation).
    lean: bool,
    /// A folder's live graph as last brought in line with its layers: what
    /// differs since are the artist's edits.
    pub(super) seen: Option<Graph>,
}

/// The floating add-node menu: where it opened (canvas units and screen)
/// and the output a dropped link came from, to wire the new node to.
struct AddMenu {
    at: Pos2,
    screen: Pos2,
    link_from: Option<u64>,
    /// Focus the search field on the first frame.
    focus: bool,
}

/// A saved graph on the shelf.
pub(super) struct ShelfGraph {
    pub path: PathBuf,
    pub name: String,
    pub thumb: Option<egui::TextureHandle>,
}

pub(super) struct GraphEditor {
    pub doc: Option<u64>,
    /// Canvas offset (points) and scale.
    pan: egui::Vec2,
    zoom: f32,
    /// The node whose settings show (the last one picked).
    selected: Option<u64>,
    /// Every selected node, `selected` included.
    multi: Vec<u64>,
    /// Box selection in progress, from this screen point.
    box_from: Option<Pos2>,
    /// Copied nodes (Ctrl+C), pasted with Ctrl+V.
    clipboard: Vec<graph::Node>,
    /// The add-node menu, when open.
    menu: Option<AddMenu>,
    /// A link being dragged out of this node's output.
    link_from: Option<u64>,
    /// Where the add menu opened, in canvas units.
    add_at: Pos2,
    search: String,
    /// Graph snapshots for undo inside the editor (of `history_doc`).
    history: Vec<Graph>,
    future: Vec<Graph>,
    last: Option<Graph>,
    history_doc: Option<u64>,
    /// The pointer is over the editor: Ctrl+Z / Ctrl+Y act on the graph.
    pub hovered: bool,
    /// Show the start page instead of the open graph.
    start_page: bool,
    /// Fit the graph in view on the next canvas frame (a graph just opened).
    frame_next: bool,
    thumbs: HashMap<(u64, u64), (egui::TextureId, usize)>,
}

pub(super) struct GraphStore {
    pub lib: Library,
    pub docs: Vec<GraphDoc>,
    next_id: u64,
    pub editor: GraphEditor,
    pub shelf: Vec<ShelfGraph>,
    /// The Shader Maker workspace's state.
    pub shader: super::shader_ui::ShaderMaker,
    /// Thumbnails for the start pages.
    pub gallery: super::gallery::Gallery,
    /// Keeps the user node folder watched; changes arrive on `changes`.
    _watcher: Option<notify::RecommendedWatcher>,
    changes: Option<mpsc::Receiver<()>>,
    /// Repaints when a graph's image has been decoded.
    repaint: egui::Context,
}

fn graphs_dir() -> Option<PathBuf> {
    Some(super::project_io::data_dir()?.join("graphs"))
}

impl GraphStore {
    pub fn new(ctx: &egui::Context) -> GraphStore {
        let user_dir = super::project_io::data_dir().map(|d| d.join("nodes"));
        let (mut watcher, mut changes) = (None, None);
        if let Some(dir) = &user_dir {
            let _ = std::fs::create_dir_all(dir);
            use notify::Watcher;
            let (tx, rx) = mpsc::channel();
            let ctx = ctx.clone();
            let w = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
                if res.is_ok_and(|e| e.paths.iter().any(|p| p.extension().is_some_and(|x| x == "wgsl"))) {
                    let _ = tx.send(());
                    ctx.request_repaint();
                }
            });
            if let Ok(mut w) = w
                && w.watch(dir, notify::RecursiveMode::NonRecursive).is_ok() {
                    watcher = Some(w);
                    changes = Some(rx);
                }
        }
        let mut store = GraphStore {
            lib: Library::load(user_dir),
            docs: Vec::new(),
            next_id: 1,
            editor: GraphEditor {
                doc: None,
                pan: vec2(60.0, 200.0),
                zoom: 1.0,
                selected: None,
                multi: Vec::new(),
                box_from: None,
                clipboard: Vec::new(),
                menu: None,
                link_from: None,
                add_at: Pos2::ZERO,
                search: String::new(),
                history: Vec::new(),
                future: Vec::new(),
                last: None,
                history_doc: None,
                hovered: false,
                start_page: false,
                frame_next: true,
                thumbs: HashMap::new(),
            },
            shelf: Vec::new(),
            shader: super::shader_ui::ShaderMaker::new(),
            gallery: Default::default(),
            _watcher: watcher,
            changes,
            repaint: ctx.clone(),
        };
        store.reload_shelf(ctx);
        store
    }

    /// Rereads the saved graphs and their thumbnails.
    pub fn reload_shelf(&mut self, ctx: &egui::Context) {
        self.shelf.clear();
        let Some(dir) = graphs_dir() else { return };
        let Ok(entries) = std::fs::read_dir(&dir) else { return };
        let mut paths: Vec<PathBuf> =
            entries.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "mpgraph")).collect();
        paths.sort();
        for path in paths {
            let name = path.file_stem().unwrap_or_default().to_string_lossy().into_owned();
            let thumb = image::open(path.with_extension("png")).ok().map(|img| {
                let img = img.into_rgba8();
                let size = [img.width() as usize, img.height() as usize];
                ctx.load_texture(format!("graph {name}"), egui::ColorImage::from_rgba_unmultiplied(size, &img), Default::default())
            });
            self.shelf.push(ShelfGraph { path, name, thumb });
        }
    }

    pub fn doc(&self, id: u64) -> Option<&GraphDoc> {
        self.docs.iter().find(|d| d.id == id)
    }

    fn doc_mut(&mut self, id: u64) -> Option<&mut GraphDoc> {
        self.docs.iter_mut().find(|d| d.id == id)
    }

    /// Adds a graph under a new id (or `id`, when loading a project).
    pub fn insert(&mut self, rs: &eframe::egui_wgpu::RenderState, graph: Graph, id: Option<u64>) -> u64 {
        let id = id.unwrap_or(self.next_id);
        self.next_id = self.next_id.max(id + 1);
        let mut rt = Runtime::new(&rs.device, &rs.queue);
        let ctx = self.repaint.clone();
        rt.set_waker(Arc::new(move || ctx.request_repaint()));
        let shading = graph::shading::is_shading(&graph, &self.lib);
        self.docs.push(GraphDoc { id, graph, rt, dirty: true, push: true, maps_seen: u64::MAX, shading, lean: false, seen: None });
        id
    }

    /// Forgets every graph (new or loaded project).
    pub fn clear(&mut self, rs: &eframe::egui_wgpu::RenderState) {
        let mut renderer = rs.renderer.write();
        for (_, (tex, _)) in self.editor.thumbs.drain() {
            renderer.free_texture(&tex);
        }
        self.docs.clear();
        self.editor.doc = None;
        self.editor.selected = None;
    }

    /// Projects store their graphs with the layers that use them.
    pub fn to_json(&self, used: &[u64]) -> serde_json::Value {
        let list: Vec<serde_json::Value> = self
            .docs
            .iter()
            .filter(|d| used.contains(&d.id))
            .map(|d| serde_json::json!({"id": d.id, "graph": d.graph.to_json(&self.lib)}))
            .collect();
        serde_json::Value::Array(list)
    }
}

/// A mask graph to start from: the curvature map scanned into a mask of the
/// convex edges.
fn mask_starter(res: u32) -> Graph {
    let mut g = Graph::new("Mask graph");
    g.res = res;
    g.match_set = true;
    let map = g.add("mesh_map", [0.0, 0.0]);
    let scan = g.add("histogram_scan", [220.0, 0.0]);
    let out = g.add("output", [440.0, 0.0]);
    let curvature = MAPS.iter().position(|m| m.name == "Curvature").unwrap_or(0);
    g.node_mut(map).unwrap().params.insert("map".into(), Value::MeshMap(curvature));
    g.node_mut(scan).unwrap().params.insert("position".into(), Value::Float(0.55));
    g.node_mut(out).unwrap().params.insert("usage".into(), Value::Enum(4));
    g.connect(map, scan, "input");
    g.connect(scan, out, "input");
    g
}

fn half_to_f32(b: [u8; 2]) -> f32 {
    let h = u16::from_le_bytes(b);
    let (s, e, m) = (h >> 15, ((h >> 10) & 31) as i32, (h & 1023) as f32);
    let v = match e {
        0 => m / 1024.0 * 2f32.powi(-14),
        31 => f32::INFINITY,
        _ => (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    };
    if s == 1 { -v } else { v }
}

/// The editor for one node parameter; true when it changed. `salt` keeps
/// widget ids apart between nodes.
/// A node parameter's editor; `custom` lists the custom bake maps a Mesh
/// Map parameter can pick.
pub(super) fn param_widget(ui: &mut egui::Ui, p: &graph::def::Param, v: &mut Value, salt: u64, custom: &[crate::bake::CustomMap]) -> bool {
    // Seeds get a button that rolls a new one.
    if let (ParamKind::Int { min, max }, Value::Int(i)) = (&p.kind, &mut *v)
        && p.name.contains("seed") {
            let mut ch = ui.add(egui::Slider::new(i, *min..=*max)).changed();
            if ui.small_button("🎲").on_hover_text("New random seed").clicked() {
                let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.subsec_nanos());
                *i = *min + (t % ((*max - *min).max(1) as u32 + 1)) as i32;
                ch = true;
            }
            return ch;
        }
    match (&p.kind, v) {
                    (ParamKind::Float { min, max }, Value::Float(f)) => ui.add(egui::Slider::new(f, *min..=*max).max_decimals(3)).changed(),
                    (ParamKind::Int { min, max }, Value::Int(i)) => ui.add(egui::Slider::new(i, *min..=*max)).changed(),
                    (ParamKind::Bool, Value::Bool(b)) => ui.checkbox(b, "").changed(),
                    (ParamKind::Color, Value::Color(c)) => super::color_edit(ui, c),
                    (ParamKind::Enum(opts), Value::Enum(i)) => {
                        let mut ch = false;
                        egui::ComboBox::from_id_salt(("param enum", salt, p.name.as_str())).selected_text(opts.get(*i).cloned().unwrap_or_default()).show_ui(ui, |ui| {
                            for (k, o) in opts.iter().enumerate() {
                                ch |= ui.selectable_value(i, k, o).changed();
                            }
                        });
                        ch
                    }
                    (ParamKind::MeshMap, Value::MeshMap(i)) => {
                        let mut ch = false;
                        let base = crate::engine::CUSTOM_MAP_VALUE;
                        let current = match MAPS.get(*i) {
                            Some(m) => m.name.to_string(),
                            None => custom.iter().find(|c| *i == base + c.id as usize).map_or("(removed map)".to_string(), |c| c.name.clone()),
                        };
                        egui::ComboBox::from_id_salt(("param map", salt, p.name.as_str())).selected_text(current).height(400.0).show_ui(ui, |ui| {
                            for (k, m) in MAPS.iter().enumerate() {
                                ch |= ui.selectable_value(i, k, m.name).on_hover_text(m.desc).changed();
                            }
                            if !custom.is_empty() {
                                ui.separator();
                                ui.label(egui::RichText::new("CUSTOM MAPS").small().color(theme::TEXT_DIM));
                            }
                            for c in custom {
                                ch |= ui.selectable_value(i, base + c.id as usize, &c.name).on_hover_text(c.kind.name()).changed();
                            }
                        });
                        ch
                    }
                    (ParamKind::Image, Value::Image(path)) => {
                        let name = path.as_ref().and_then(|p| p.file_name()).map_or("None".into(), |n| n.to_string_lossy().into_owned());
                        ui.add(egui::Label::new(name).truncate());
                        if ui.small_button("Choose…").clicked()
                            && let Some(p) = rfd::FileDialog::new().add_filter("Images", &["png", "jpg", "jpeg", "tga", "bmp", "tif", "tiff", "exr", "hdr"]).pick_file() {
                                *path = Some(p);
                                return true;
                            }
                        false
                    }
                    _ => false,
    }
}

/// A user node file name not taken yet: `base`, else `base_2`, `base_3`...
pub(super) fn unique_stem(lib: &Library, base: &str) -> String {
    let taken = |s: &str| lib.find(&format!("user/{s}")).is_some() || lib.user_path(&format!("user/{s}")).is_some_and(|p| p.exists());
    let mut stem = base.to_string();
    let mut k = 2;
    while taken(&stem) {
        stem = format!("{base}_{k}");
        k += 1;
    }
    stem
}

/// Draws the add-node menu if open; returns the node definition picked.
fn add_menu(ui: &egui::Ui, lib: &Library, e: &mut GraphEditor, shading: bool) -> Option<String> {
    let menu = e.menu.as_mut()?;
    let mut picked = None;
    let mut close = false;
    let area = egui::Area::new(egui::Id::unique("graph add menu")).order(egui::Order::Foreground).fixed_pos(menu.screen).show(ui.ctx(), |ui| {
        egui::Frame::popup(ui.style()).show(ui, |ui| {
            ui.set_width(230.0);
            let r = ui.add(egui::TextEdit::singleline(&mut e.search).hint_text("Search nodes…"));
            if menu.focus {
                r.request_focus();
                menu.focus = false;
            }
            let q = e.search.to_lowercase();
            egui::ScrollArea::vertical().max_height(380.0).show(ui, |ui| {
                let mut first = None;
                for (cat, ids) in lib.categories() {
                    let defs: Vec<_> = ids
                        .iter()
                        .filter_map(|id| lib.find(id))
                        .filter(|d| d.shading == shading)
                        .filter(|d| q.is_empty() || d.name.to_lowercase().contains(&q) || d.category.to_lowercase().contains(&q))
                        .collect();
                    if defs.is_empty() {
                        continue;
                    }
                    ui.label(egui::RichText::new(cat.to_uppercase()).small().color(category_color(&cat).gamma_multiply(1.6)));
                    for d in defs {
                        first.get_or_insert(d.id.clone());
                        if ui.add(egui::Button::new(&d.name).frame(false)).on_hover_text(&d.description).clicked() {
                            picked = Some(d.id.clone());
                        }
                    }
                }
                if ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    picked = picked.clone().or(first);
                }
            });
        });
    });
    let outside = ui.input(|i| i.pointer.any_pressed())
        && ui.input(|i| i.pointer.interact_pos()).is_some_and(|p| !area.response.rect.contains(p));
    if ui.input(|i| i.key_pressed(egui::Key::Escape)) || (outside && picked.is_none()) {
        close = true;
    }
    if close {
        e.menu = None;
    }
    picked
}

/// Copies of `nodes` moved by `offset`, under new ids; links among them
/// follow the copies, links from other nodes still in the graph stay.
fn paste_nodes(graph: &mut Graph, nodes: &[graph::Node], offset: egui::Vec2) -> Vec<u64> {
    let mut map = HashMap::new();
    for n in nodes {
        let id = graph.add(&n.def, [n.pos[0] + offset.x, n.pos[1] + offset.y]);
        map.insert(n.id, id);
    }
    for n in nodes {
        let new = map[&n.id];
        let mut links = HashMap::new();
        for (input, from) in &n.links {
            match map.get(from) {
                Some(f) => {
                    links.insert(input.clone(), *f);
                }
                None if graph.node(*from).is_some() => {
                    links.insert(input.clone(), *from);
                }
                None => {}
            }
        }
        let node = graph.node_mut(new).expect("just added");
        node.params = n.params.clone();
        node.links = links;
    }
    nodes.iter().map(|n| map[&n.id]).collect()
}

const NODE_W: f32 = 150.0;
const HEADER: f32 = 22.0;
const ROW: f32 = 18.0;
const PREVIEW: f32 = 126.0;

/// A node's box: header, an input row each, and a preview picture (shading
/// nodes make no picture, so they're compact).
fn node_size(inputs: usize, picture: bool) -> egui::Vec2 {
    let preview = if picture { PREVIEW + 12.0 } else { 8.0 };
    vec2(NODE_W, HEADER + inputs as f32 * ROW + preview)
}

fn category_color(cat: &str) -> Color32 {
    match cat {
        "Generators" => Color32::from_rgb(70, 110, 160),
        "Adjust" => Color32::from_rgb(110, 90, 150),
        "Blend" => Color32::from_rgb(150, 100, 60),
        "Filters" => Color32::from_rgb(60, 130, 110),
        "Inputs" => Color32::from_rgb(120, 120, 60),
        "Outputs" => Color32::from_rgb(160, 60, 70),
        _ => Color32::from_rgb(90, 90, 100),
    }
}

impl PainterApp {
    // ------------------------------------------------------------------ store

    /// Evaluates the graphs that are shown or used and pushes changed
    /// outputs into their layers. Runs once a frame, before compositing.
    pub(super) fn update_graphs(&mut self) {
        self.sync_folder_graphs();
        if let Some(rx) = &self.graphs.changes
            && rx.try_iter().count() > 0 {
                self.graphs.lib.reload_user();
                for d in &mut self.graphs.docs {
                    d.dirty = true;
                }
                self.shader_files_changed();
            }
        self.fill_parked_graphs();
        let maps_version = self.engine.maps_version;
        for i in 0..self.graphs.docs.len() {
            let id = self.graphs.docs[i].id;
            let in_use = self.engine.layers.iter().any(|l| {
                matches!(&l.content, Content::Fill(f) if f.graph == Some(id)) || l.mask.as_ref().is_some_and(|m| m.graph == Some(id))
            });
            let shown = self.workspace == super::Workspace::Designer && self.graphs.editor.doc == Some(id);
            if !in_use && !shown {
                continue;
            }
            let d = &mut self.graphs.docs[i];
            if d.shading {
                continue;
            }
            // Images decoded since: the nodes waiting for them render now.
            if d.rt.take_images(&self.rs.device, &self.rs.queue) {
                d.dirty = true;
            }
            if d.graph.match_set && d.graph.res != self.engine.res {
                d.graph.res = self.engine.res;
                d.dirty = true;
            }
            if d.dirty || d.maps_seen != maps_version || (d.lean && shown) {
                let engine = &self.engine;
                let maps = |m: usize| engine.mesh_map(m);
                let surface = engine.graph_surface();
                let lean = !shown;
                if d.rt.evaluate(&self.rs.device, &self.rs.queue, &d.graph, &self.graphs.lib, &maps, Some(&surface), maps_version, lean) {
                    d.push = true;
                }
                d.dirty = false;
                d.lean = lean;
                d.maps_seen = maps_version;
            }
            if d.push && in_use {
                self.push_graph(i);
            }
        }
        if self.workspace == super::Workspace::ShaderMaker {
            self.update_shader_preview();
        }
    }

    /// Texture sets not being edited keep the graph outputs they had; one
    /// that has none yet (an opened project) gets them rendered on its own
    /// mesh maps.
    fn fill_parked_graphs(&mut self) {
        let (lib, docs) = (&self.graphs.lib, &mut self.graphs.docs);
        for p in self.sets.parked.iter_mut().flatten() {
            let engine = &mut p.engine;
            for li in 0..engine.layers.len() {
                let l = &engine.layers[li];
                let Content::Fill(f) = &l.content else { continue };
                let Some(id) = f.graph.filter(|_| l.images.iter().all(Option::is_none)) else { continue };
                let Some(d) = docs.iter_mut().find(|d| d.id == id && !d.shading) else { continue };
                let version = engine.maps_version;
                if d.graph.match_set {
                    d.graph.res = engine.res;
                }
                {
                    let maps = |m: usize| engine.mesh_map(m);
                    let surface = engine.graph_surface();
                    d.rt.evaluate(&self.rs.device, &self.rs.queue, &d.graph, lib, &maps, Some(&surface), version, true);
                }
                // The edited set's outputs are made again on its own maps.
                d.maps_seen = u64::MAX;
                let outs: [Option<Arc<Tex>>; graph::USAGES.len()] = d.graph.outputs(lib).map(|o| o.and_then(|n| d.rt.output(n).cloned()));
                let name = d.graph.name.clone();
                engine.set_graph_images(li, graph::channel_outputs(&outs), &name);
            }
        }
    }

    /// Sends a graph's outputs to the fill layers and masks that use it.
    fn push_graph(&mut self, i: usize) {
        let d = &self.graphs.docs[i];
        let outs: [Option<Arc<Tex>>; graph::USAGES.len()] = d.graph.outputs(&self.graphs.lib).map(|o| o.and_then(|n| d.rt.output(n).cloned()));
        let (id, name) = (d.id, d.graph.name.clone());
        for li in 0..self.engine.layers.len() {
            let l = &self.engine.layers[li];
            let (fill, mask) = (matches!(&l.content, Content::Fill(f) if f.graph == Some(id)), l.mask.as_ref().is_some_and(|m| m.graph == Some(id)));
            if fill {
                self.engine.set_graph_images(li, graph::channel_outputs(&outs), &name);
            }
            if mask
                && let Some(t) = outs[graph::MASK_USAGE].as_ref().or(outs[0].as_ref()) {
                    self.engine.set_graph_mask(li, t);
                }
        }
        self.graphs.docs[i].push = false;
    }

    /// The graph's parameters as controls, under their group headings
    /// (layer properties).
    pub(super) fn exposed_controls(&mut self, ui: &mut egui::Ui, id: u64) {
        let custom = self.bake_settings.custom.clone();
        let lib = &self.graphs.lib;
        let Some(d) = self.graphs.docs.iter_mut().find(|d| d.id == id) else { return };
        if d.graph.params.is_empty() {
            ui.label(egui::RichText::new("Tip: mark node settings with ◇ in the Material Designer to tweak them here.").small().color(theme::TEXT_DIM));
            return;
        }
        // Groups in the order their first parameter comes.
        let mut groups: Vec<&str> = Vec::new();
        for p in &d.graph.params {
            if !groups.contains(&p.group.as_str()) {
                groups.push(&p.group);
            }
        }
        let groups: Vec<String> = groups.into_iter().map(str::to_string).collect();
        let mut set: Option<(usize, Value)> = None;
        for group in &groups {
            let mut rows = |ui: &mut egui::Ui| {
                for i in (0..d.graph.params.len()).filter(|&i| &d.graph.params[i].group == group) {
                    let Some((p, mut v)) = d.graph.param_value(i, lib) else { continue };
                    let gp = &d.graph.params[i];
                    let changed = labeled(ui, &gp.name, |ui| {
                        let mut ch = param_widget(ui, &p, &mut v, id ^ (i as u64) << 32, &custom);
                        if let Some(def) = gp.default.as_ref().filter(|def| **def != v)
                            && ui.small_button("⟲").on_hover_text("Back to the material's value").clicked() {
                                v = def.clone();
                                ch = true;
                            }
                        ch
                    });
                    if changed {
                        set = Some((i, v));
                    }
                }
            };
            if group.is_empty() {
                rows(ui);
            } else {
                egui::CollapsingHeader::new(egui::RichText::new(group).strong()).id_salt(("graph group", id, group.as_str())).default_open(true).show(ui, rows);
            }
        }
        if let Some((i, v)) = set {
            d.graph.set_param(i, v, lib);
            d.dirty = true;
        }
    }

    /// The open graph's parameters: rename, group, see what they drive,
    /// remove (Node pane).
    fn graph_params_panel(&mut self, ui: &mut egui::Ui, doc: u64) {
        let lib = &self.graphs.lib;
        let Some(d) = self.graphs.docs.iter_mut().find(|d| d.id == doc) else { return };
        section(ui, "Material parameters");
        if d.graph.params.is_empty() {
            ui.label(
                egui::RichText::new("None yet. Click ◇ next to a node setting to show it in the layer properties under a name of your choice; several settings can share one parameter.")
                    .small()
                    .color(theme::TEXT_DIM),
            );
            return;
        }
        let mut remove = None;
        let mut select = None;
        let mut move_up = None;
        for i in 0..d.graph.params.len() {
            let drives: Vec<(u64, String)> = d.graph.params[i]
                .binds
                .iter()
                .map(|(n, k)| {
                    let node = d.graph.node(*n).and_then(|n| lib.find(&n.def)).map_or("?".to_string(), |def| def.name.clone());
                    (*n, format!("{node} · {}", k.replace('_', " ")))
                })
                .collect();
            let p = &mut d.graph.params[i];
            ui.push_id(("graph param", i), |ui| {
                theme::card(theme::WIDGET).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.add(egui::TextEdit::singleline(&mut p.name).desired_width(ui.available_width() - 64.0)).on_hover_text("Its name in the layer properties");
                        if ui.add_enabled(i > 0, egui::Button::new("▲").small()).on_hover_text("Show it earlier").clicked() {
                            move_up = Some(i);
                        }
                        if ui.small_button("✕").on_hover_text("Remove the parameter (the nodes keep their values)").clicked() {
                            remove = Some(i);
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Group").small().color(theme::TEXT_DIM));
                        ui.add(egui::TextEdit::singleline(&mut p.group).hint_text("none").desired_width(ui.available_width()))
                            .on_hover_text("Parameters with the same group show together under that heading");
                    });
                    for (n, what) in drives {
                        if ui.add(egui::Label::new(egui::RichText::new(format!("→ {what}")).small().color(theme::TEXT_DIM)).sense(Sense::click())).on_hover_text("Select the node").clicked() {
                            select = Some(n);
                        }
                    }
                });
            });
        }
        if let Some(i) = move_up {
            d.graph.params.swap(i - 1, i);
        }
        if let Some(i) = remove {
            d.graph.params.remove(i);
        }
        if let Some(n) = select {
            self.graphs.editor.selected = Some(n);
        }
    }

    pub(super) fn refresh_graph(&mut self, id: u64) {
        if let Some(d) = self.graphs.doc_mut(id) {
            d.dirty = true;
        }
    }

    /// Drops a node into the open graph, next to the others, and shows it.
    pub(super) fn add_node_to_open_graph(&mut self, def: &str) {
        let Some(doc) = self.graphs.editor.doc else { return };
        let Some(d) = self.graphs.doc_mut(doc) else { return };
        let x = d.graph.nodes.iter().map(|n| n.pos[0]).fold(0.0f32, f32::max) + 220.0;
        let id = d.graph.add(def, [x, 0.0]);
        d.dirty = true;
        self.graphs.editor.selected = Some(id);
        self.workspace = super::Workspace::Designer;
    }

    /// Graphs that layers or masks use (what projects store).
    pub(super) fn used_graphs(&self) -> Vec<u64> {
        let mut ids = Vec::new();
        for l in &self.engine.layers {
            if let Content::Fill(f) = &l.content {
                ids.extend(f.graph);
            }
            if let Content::Folder(f) = &l.content {
                ids.extend(f.graph);
            }
            ids.extend(l.mask.as_ref().and_then(|m| m.graph));
        }
        ids.extend(self.preview.graph);
        ids
    }

    /// Every graph sends its outputs again (texture set resized, project loaded).
    pub(super) fn repush_graphs(&mut self) {
        for d in &mut self.graphs.docs {
            d.push = true;
        }
    }

    pub(super) fn open_graph(&mut self, id: u64) {
        self.workspace = super::Workspace::Designer;
        let e = &mut self.graphs.editor;
        e.start_page = false;
        if e.doc != Some(id) {
            e.frame_next = true;
            e.doc = Some(id);
            e.selected = None;
        }
    }

    /// A fill layer driven by `graph` (the starter graph when `None`).
    pub(super) fn add_graph_layer(&mut self, graph: Option<Graph>, at: usize, parent: Option<u64>) {
        let mut graph = graph.unwrap_or_else(|| Graph::starter("Graph material"));
        graph.match_set = true;
        graph.res = self.engine.res;
        let name = graph.name.clone();
        let outs = graph.outputs(&self.graphs.lib);
        let id = self.graphs.insert(&self.rs, graph, None);
        let mut fill = FillParams { graph: Some(id), ..FillParams::default() };
        fill.mat.enabled = graph::channel_outputs(&outs).map(|o| o.is_some());
        // An emission output is the emitted color itself: no tint.
        fill.mat.emission = [1.0; 3];
        self.selected = self.engine.add_fill_layer(at, parent, name, fill, &mut self.undo);
        self.target = StrokeTarget::Content;
        self.open_graph(id);
        self.status = "Added a graph fill layer. Edit its graph in the Material Graph window.".into();
    }

    /// A smart material as one graph fill layer: its settings are the
    /// layer's material parameters; its graph opens in the Material Designer
    /// on request.
    pub(super) fn add_smart_graph(&mut self, smart: &crate::presets::SmartMaterial, at: usize, parent: Option<u64>) {
        let mut graph = graph::smart::graph(smart, &self.graphs.lib);
        // As sharp as the texture set, following it when it's resized.
        graph.match_set = true;
        graph.res = self.engine.res;
        let name = graph.name.clone();
        let outs = graph.outputs(&self.graphs.lib);
        let id = self.graphs.insert(&self.rs, graph, None);
        let mut fill = FillParams { graph: Some(id), ..FillParams::default() };
        fill.mat.enabled = graph::channel_outputs(&outs).map(|o| o.is_some());
        self.selected = self.engine.add_fill_layer(at, parent, name, fill, &mut self.undo);
        self.target = StrokeTarget::Content;
    }

    /// A mask on the selected layer, rendered by a new mask graph.
    pub(super) fn add_graph_mask(&mut self) {
        let idx = self.selected;
        if idx >= self.engine.layers.len() || self.engine.layers[idx].mask.is_some() {
            return;
        }
        let id = self.graphs.insert(&self.rs, mask_starter(self.engine.res), None);
        self.engine.add_mask(idx, true, None, &mut self.undo);
        if let Some(m) = self.engine.layers[idx].mask.as_mut() {
            m.graph = Some(id);
        }
        self.target = StrokeTarget::Mask;
        self.open_graph(id);
        self.status = "Added a graph mask: its Output (Mask) drives where the layer shows.".into();
    }

    /// Opens the graph of the selected layer (its fill, or its mask when
    /// the mask is targeted).
    pub(super) fn edit_selected_graph(&mut self) {
        let Some(l) = self.engine.layers.get(self.selected) else { return };
        if l.is_folder() {
            self.open_folder_graph(self.selected);
            return;
        }
        let id = if self.target == StrokeTarget::Mask {
            l.mask.as_ref().and_then(|m| m.graph)
        } else {
            match &l.content {
                Content::Fill(f) => f.graph,
                _ => None,
            }
        };
        match id {
            Some(id) => self.open_graph(id),
            None => self.workspace = super::Workspace::Designer,
        }
    }

    pub(super) fn add_shelf_graph(&mut self, path: &std::path::Path, at: usize, parent: Option<u64>) {
        match graph::load_file(path, &self.graphs.lib) {
            Ok(g) => self.add_graph_layer(Some(g), at, parent),
            Err(e) => self.status = e,
        }
    }

    // ------------------------------------------------------------------ window

    /// The Material Designer workspace: the material on the mesh, the node
    /// canvas and the selected node's settings.
    pub(super) fn designer(&mut self, ui: &mut egui::Ui) {
        // Show a graph of the project when there is one and none is picked.
        if self.graphs.editor.doc.filter(|d| self.graphs.doc(*d).is_some()).is_none() && !self.graphs.editor.start_page {
            self.graphs.editor.doc = self.graphs.docs.last().map(|d| d.id);
        }
        egui::Panel::top("designer bar")
            .frame(theme::card(theme::PANEL).inner_margin(egui::Margin::symmetric(12, 6)))
            .show(ui, |ui| self.graph_toolbar(ui));
        self.graphs.editor.hovered = false;
        egui::CentralPanel::default().frame(egui::Frame::new().fill(theme::BG).inner_margin(egui::Margin::same(4))).show(ui, |ui| {
            self.dock_ui(ui, super::Workspace::Designer);
        });
        self.snapshot_graph(ui.ctx());
    }

    /// The open graph (or none: the start page).
    fn designer_doc(&self) -> Option<u64> {
        self.graphs.editor.doc.filter(|d| self.graphs.doc(*d).is_some() && !self.graphs.editor.start_page)
    }

    /// The Graph tab: the node canvas, or the start page.
    pub(super) fn designer_graph_pane(&mut self, ui: &mut egui::Ui) {
        match self.designer_doc() {
            Some(doc) => {
                self.canvas(ui, doc);
                self.graphs.editor.hovered = ui.rect_contains_pointer(ui.max_rect());
            }
            None => {
                egui::Frame::new().inner_margin(egui::Margin::symmetric(24, 8)).show(ui, |ui| {
                    ui.set_min_size(ui.available_size());
                    self.designer_start(ui);
                });
            }
        }
    }

    /// The Node tab: the selected node's settings.
    pub(super) fn designer_settings_pane(&mut self, ui: &mut egui::Ui) {
        match self.designer_doc() {
            Some(doc) => {
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    self.node_panel(ui, doc);
                    // A folder's graph is tuned through its layers.
                    let folder = self.engine.layers.iter().any(|l| matches!(&l.content, Content::Folder(f) if f.graph == Some(doc)));
                    if !folder {
                        ui.add_space(10.0);
                        self.graph_params_panel(ui, doc);
                    }
                });
            }
            None => {
                ui.label(egui::RichText::new("Pick a material on the start page, or open a graph from the toolbar.").color(theme::TEXT_DIM));
            }
        }
    }

    /// The Designer's start page: materials to start from, the project's
    /// graphs and the saved ones.
    fn designer_start(&mut self, ui: &mut egui::Ui) {
        self.ensure_gallery();
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            use super::gallery::{card, heading, preset_key};
            heading(ui, "Start a material", "Each one becomes a graph fill layer on the mesh. Change any node afterwards.");
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = vec2(12.0, 12.0);
                for (i, p) in graph::presets::PRESETS.iter().enumerate() {
                    if card(ui, self.gallery_thumb(&preset_key(i)), p.name, p.description, 168.0).clicked() {
                        self.pending.push(Action::AddGraphPreset(i));
                        self.graphs.editor.start_page = false;
                    }
                }
            });
            if !self.graphs.docs.is_empty() {
                heading(ui, "In this project", "Graphs your layers and masks use.");
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = vec2(12.0, 12.0);
                    for d in &self.graphs.docs {
                        let outs = d.graph.outputs(&self.graphs.lib);
                        let thumb = outs.iter().flatten().next().and_then(|n| self.graphs.editor.thumbs.get(&(d.id, *n))).map(|t| t.0);
                        let what = if outs[4].is_some() { "Mask graph" } else { "Material graph" };
                        if card(ui, thumb, &d.graph.name, &format!("{what} · {} nodes", d.graph.nodes.len()), 140.0).clicked() {
                            self.graphs.editor.doc = Some(d.id);
                            self.graphs.editor.start_page = false;
                            self.graphs.editor.selected = None;
                            self.graphs.editor.frame_next = true;
                        }
                    }
                });
            }
            if !self.graphs.shelf.is_empty() {
                heading(ui, "Saved graphs", "From your shelf.");
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = vec2(12.0, 12.0);
                    for (i, g) in self.graphs.shelf.iter().enumerate() {
                        if card(ui, g.thumb.as_ref().map(|t| t.id()), &g.name, "Adds it as a graph fill layer.", 140.0).clicked() {
                            self.pending.push(Action::AddOnTop(super::ShelfItem::Graph(i)));
                            self.graphs.editor.start_page = false;
                        }
                    }
                });
            }
            heading(ui, "Masks", "A graph can also decide where a layer shows, from the baked mesh maps.");
            if ui.add(egui::Button::new(egui::RichText::new("  Add a graph mask to the selected layer  ").size(14.0)).min_size(vec2(0.0, 30.0))).clicked() {
                self.pending.push(Action::AddGraphMask);
                self.graphs.editor.start_page = false;
            }
            ui.add_space(20.0);
        });
    }

    fn graph_toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.menu_button(egui::RichText::new("+ New").strong(), |ui| {
                for (i, p) in graph::presets::PRESETS.iter().enumerate() {
                    if ui.button(p.name).on_hover_text(p.description).clicked() {
                        self.pending.push(Action::AddGraphPreset(i));
                        self.graphs.editor.start_page = false;
                        ui.close();
                    }
                }
                ui.separator();
                if ui.button("Graph mask on the selected layer").clicked() {
                    self.pending.push(Action::AddGraphMask);
                    self.graphs.editor.start_page = false;
                    ui.close();
                }
            });
            let home = theme::selectable(self.graphs.editor.start_page || self.graphs.editor.doc.is_none(), "Start page");
            if ui.add(home).clicked() {
                self.graphs.editor.start_page = true;
            }
            ui.separator();
            let current = self.graphs.editor.doc.and_then(|d| self.graphs.doc(d)).map_or("None".to_string(), |d| d.graph.name.clone());
            ui.label(egui::RichText::new("Graph").color(theme::TEXT_DIM));
            egui::ComboBox::from_id_salt("graph doc").selected_text(current).width(170.0).show_ui(ui, |ui| {
                for d in &self.graphs.docs {
                    if ui.selectable_label(self.graphs.editor.doc == Some(d.id), &d.graph.name).clicked() {
                        self.graphs.editor.doc = Some(d.id);
                        self.graphs.editor.selected = None;
                        self.graphs.editor.start_page = false;
                        self.graphs.editor.frame_next = true;
                    }
                }
            });
            if let Some(id) = self.graphs.editor.doc
                && let Some(d) = self.graphs.doc_mut(id) {
                    ui.add(egui::TextEdit::singleline(&mut d.graph.name).desired_width(140.0));
                }
            let shading = self.graphs.editor.doc.and_then(|id| self.graphs.doc(id)).is_some_and(|d| d.shading);
            if shading {
                ui.separator();
                ui.label(egui::RichText::new("Preview shader: the 3D view uses it while you texture").color(theme::TEXT_DIM));
            }
            if let Some(id) = self.graphs.editor.doc.filter(|_| !shading) {
                let res_now = self.engine.res;
                if let Some(d) = self.graphs.doc_mut(id) {
                    ui.label(egui::RichText::new("Size").color(theme::TEXT_DIM));
                    let before = (d.graph.res, d.graph.match_set);
                    let text = if d.graph.match_set { format!("Texture set ({})", d.graph.res) } else { d.graph.res.to_string() };
                    egui::ComboBox::from_id_salt("graph res").selected_text(text).width(120.0).show_ui(ui, |ui| {
                        if ui
                            .selectable_label(d.graph.match_set, "Texture set")
                            .on_hover_text("Render at the texture set's resolution, following it when it changes")
                            .clicked()
                        {
                            d.graph.match_set = true;
                            d.graph.res = res_now;
                        }
                        for r in [256, 512, 1024, 2048, 4096] {
                            if ui.selectable_label(!d.graph.match_set && d.graph.res == r, r.to_string()).clicked() {
                                d.graph.match_set = false;
                                d.graph.res = r;
                            }
                        }
                    });
                    if (d.graph.res, d.graph.match_set) != before {
                        d.dirty = true;
                    }
                }
                ui.separator();
                if ui.button("Save to shelf").on_hover_text("Save this graph as a shelf material").clicked() {
                    self.save_graph_to_shelf(ui.ctx(), id);
                }
                if ui.button("Export outputs…").on_hover_text("Write the graph's outputs as image files").clicked() {
                    self.export_graph_outputs(id);
                }
            }
            ui.separator();
            if ui.button("New node…").on_hover_text("Write a custom node in WGSL (opens the Shader Maker)").clicked() {
                self.new_custom_node(None, true);
            }
            if ui.button("Nodes folder").on_hover_text("Custom nodes live here; edits on disk reload automatically").clicked()
                && let Some(dir) = &self.graphs.lib.user_dir {
                    super::open_in_file_browser(dir);
                }
            if !self.graphs.lib.errors.is_empty() {
                let text: Vec<String> = self.graphs.lib.errors.iter().map(|(p, e)| format!("{}: {e}", p.display())).collect();
                ui.label(egui::RichText::new(format!("⚠ {} node file(s) with errors", text.len())).color(Color32::from_rgb(240, 120, 90)))
                    .on_hover_text(text.join("\n"));
            }
        });
    }

    /// Records undo snapshots of the open graph once an interaction ends.
    fn snapshot_graph(&mut self, ctx: &egui::Context) {
        let e = &mut self.graphs.editor;
        let Some(id) = e.doc else { return };
        if e.history_doc != Some(id) {
            e.history.clear();
            e.future.clear();
            e.last = None;
            e.history_doc = Some(id);
        }
        if ctx.input(|i| i.pointer.any_down()) || ctx.memory(|m| m.focused().is_some()) {
            return;
        }
        let Some(d) = self.graphs.docs.iter().find(|d| d.id == id) else { return };
        match &e.last {
            Some(last) if *last == d.graph => {}
            Some(last) => {
                e.history.push(last.clone());
                if e.history.len() > 100 {
                    e.history.remove(0);
                }
                e.future.clear();
                e.last = Some(d.graph.clone());
            }
            None => e.last = Some(d.graph.clone()),
        }
    }

    pub(super) fn graph_undo(&mut self, redo: bool) {
        let e = &mut self.graphs.editor;
        let Some(id) = e.doc else { return };
        let (from, to) = if redo { (&mut e.future, &mut e.history) } else { (&mut e.history, &mut e.future) };
        let Some(g) = from.pop() else { return };
        if let Some(d) = self.graphs.docs.iter_mut().find(|d| d.id == id) {
            to.push(std::mem::replace(&mut d.graph, g.clone()));
            d.dirty = true;
            e.last = Some(g);
        }
    }

    // ------------------------------------------------------------------ canvas

    fn canvas(&mut self, ui: &mut egui::Ui, doc: u64) {
        let rect = ui.available_rect_before_wrap();
        let bg = ui.interact(rect, ui.scope_id().with("graph canvas"), Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, theme::CARD_RADIUS, theme::PANEL_DARK);
        let typing = ui.ctx().egui_wants_keyboard_input();
        let over = ui.rect_contains_pointer(rect);

        // Pan with the middle or right button, zoom with the wheel about the cursor.
        let e = &mut self.graphs.editor;
        if bg.dragged_by(egui::PointerButton::Middle) || bg.dragged_by(egui::PointerButton::Secondary) {
            e.pan += bg.drag_delta();
        }
        if let Some(p) = bg.hover_pos() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                let f = (scroll * 0.0015).exp();
                let new = (e.zoom * f).clamp(0.2, 2.5);
                let world = (p - rect.min - e.pan) / e.zoom;
                e.zoom = new;
                e.pan = p - rect.min - world * new;
            }
        }

        // Thumbnails for this frame.
        self.sync_thumbs(doc);

        let lib = &self.graphs.lib;
        let d = self.graphs.docs.iter_mut().find(|d| d.id == doc).expect("doc exists");
        let e = &mut self.graphs.editor;
        e.multi.retain(|id| d.graph.node(*id).is_some());
        let size_of = |n: &graph::Node| lib.find(&n.def).map_or(node_size(0, true), |def| node_size(def.inputs.len(), !def.shading));

        // F frames the selection, or everything; a graph just opened is framed.
        let frame_key = over && !typing && ui.input(|i| i.key_pressed(egui::Key::F));
        if (frame_key || e.frame_next) && !d.graph.nodes.is_empty() && rect.width() > 100.0 {
            e.frame_next = false;
            let pick: Vec<&graph::Node> = if e.multi.is_empty() {
                d.graph.nodes.iter().collect()
            } else {
                d.graph.nodes.iter().filter(|n| e.multi.contains(&n.id)).collect()
            };
            let mut bb = Rect::NOTHING;
            for n in pick {
                bb = bb.union(Rect::from_min_size(pos2(n.pos[0], n.pos[1]), size_of(n)));
            }
            let zoom = ((rect.width() - 80.0) / bb.width()).min((rect.height() - 80.0) / bb.height()).clamp(0.2, 1.5);
            e.zoom = zoom;
            e.pan = rect.size() * 0.5 - bb.center().to_vec2() * zoom;
        }
        let (pan, zoom) = (e.pan, e.zoom);
        let to_screen = move |w: Pos2| rect.min + pan + w.to_vec2() * zoom;
        let to_world = move |s: Pos2| ((s - rect.min - pan) / zoom).to_pos2();

        // Grid.
        let step = 32.0 * zoom;
        if step > 6.0 {
            let mut x = rect.min.x + pan.x.rem_euclid(step);
            while x < rect.max.x {
                painter.line_segment([pos2(x, rect.min.y + 8.0), pos2(x, rect.max.y - 8.0)], Stroke::new(1.0, Color32::from_gray(24)));
                x += step;
            }
            let mut y = rect.min.y + pan.y.rem_euclid(step);
            while y < rect.max.y {
                painter.line_segment([pos2(rect.min.x + 8.0, y), pos2(rect.max.x - 8.0, y)], Stroke::new(1.0, Color32::from_gray(24)));
                y += step;
            }
        }

        let mut changed = false;
        let mut edit_code = None;
        let mods = ui.input(|i| i.modifiers);

        // Pin positions, for links and drop targets.
        let mut out_pins: HashMap<u64, Pos2> = HashMap::new();
        let mut in_pins: Vec<(u64, String, Pos2)> = Vec::new();
        for n in &d.graph.nodes {
            let p = pos2(n.pos[0], n.pos[1]);
            out_pins.insert(n.id, to_screen(p + vec2(NODE_W, HEADER * 0.5)));
            if let Some(def) = lib.find(&n.def) {
                for (i, inp) in def.inputs.iter().enumerate() {
                    in_pins.push((n.id, inp.name.clone(), to_screen(p + vec2(0.0, HEADER + ROW * (i as f32 + 0.5)))));
                }
            }
        }
        let wire = |painter: &egui::Painter, a: Pos2, b: Pos2, color: Color32| {
            let dx = ((b.x - a.x).abs() * 0.5).max(40.0 * zoom);
            let pts = [a, a + vec2(dx, 0.0), b - vec2(dx, 0.0), b];
            painter.add(egui::epaint::CubicBezierShape::from_points_stroke(pts, false, Color32::TRANSPARENT, Stroke::new(2.0 * zoom.max(0.6), color)));
        };
        for n in &d.graph.nodes {
            for (input, from) in &n.links {
                if let (Some(a), Some((_, _, b))) = (out_pins.get(from), in_pins.iter().find(|(id, name, _)| *id == n.id && name == input)) {
                    let lit = e.multi.contains(from) || e.multi.contains(&n.id);
                    wire(&painter, *a, *b, if lit { Color32::from_gray(225) } else { Color32::from_gray(150) });
                }
            }
        }

        // Nodes, in order (later ones on top).
        let font = |s: f32| egui::FontId::proportional(s * zoom);
        let mut drag_by = egui::Vec2::ZERO;
        for ni in 0..d.graph.nodes.len() {
            let n = &d.graph.nodes[ni];
            let def = lib.find(&n.def);
            let ins = def.map_or(0, |def| def.inputs.len());
            let picture = !def.is_some_and(|def| def.shading);
            let r = Rect::from_min_size(to_screen(pos2(n.pos[0], n.pos[1])), node_size(ins, picture) * zoom);
            let id = n.id;
            let resp = ui.interact(r, ui.scope_id().with(("graph node", doc, id)), Sense::click_and_drag());
            if resp.drag_started_by(egui::PointerButton::Primary) && !e.multi.contains(&id) {
                if !(mods.shift || mods.command) {
                    e.multi.clear();
                }
                e.multi.push(id);
                e.selected = Some(id);
            }
            if resp.dragged_by(egui::PointerButton::Primary) && e.link_from.is_none() {
                drag_by += resp.drag_delta() / zoom;
            }
            if resp.clicked() {
                if mods.shift || mods.command {
                    // Shift / Ctrl+click adds to or removes from the selection.
                    if let Some(k) = e.multi.iter().position(|m| *m == id) {
                        e.multi.remove(k);
                        if e.selected == Some(id) {
                            e.selected = e.multi.last().copied();
                        }
                    } else {
                        e.multi.push(id);
                        e.selected = Some(id);
                    }
                } else {
                    e.multi = vec![id];
                    e.selected = Some(id);
                }
            }
            if resp.double_clicked() && d.graph.nodes[ni].def.starts_with("user/") {
                edit_code = Some(d.graph.nodes[ni].def.clone());
            }
            let n = &d.graph.nodes[ni];
            let primary = e.selected == Some(id);
            let selected = e.multi.contains(&id);
            let error = d.rt.errors.get(&id);
            let color = def.map_or(Color32::from_rgb(120, 50, 50), |def| category_color(&def.category));
            painter.rect_filled(r, 0.0, theme::HEADER);
            let head = Rect::from_min_size(r.min, vec2(r.width(), HEADER * zoom));
            // Gray headers; the category shows as a thin stripe.
            painter.rect_filled(head, 0.0, theme::WIDGET);
            painter.rect_filled(Rect::from_min_size(head.min, vec2(3.0 * zoom, head.height())), 0.0, color);
            let title = def.map_or_else(|| format!("? {}", n.def), |d| d.name.clone());
            let t = theme::elided(&painter, &title, font(12.0), Color32::WHITE, head.width() - 26.0 * zoom);
            let title_cut = t.elided.then_some(title);
            painter.galley(head.left_center() + vec2(7.0 * zoom, -t.size().y * 0.5), t, Color32::WHITE);
            if d.graph.is_bound(id) {
                painter.text(head.right_center() - vec2(7.0 * zoom, 0.0), egui::Align2::RIGHT_CENTER, "◆", font(10.0), Color32::from_rgb(255, 220, 120));
            }
            let border = if error.is_some() {
                Stroke::new(2.0, Color32::from_rgb(240, 90, 70))
            } else if primary {
                Stroke::new(2.5, theme::ACCENT)
            } else if selected {
                Stroke::new(1.5, theme::ACCENT.gamma_multiply(0.7))
            } else {
                Stroke::new(1.0, theme::OUTLINE)
            };
            painter.rect_stroke(r, 0.0, border, egui::StrokeKind::Outside);
            if let Some(def) = def {
                for (i, inp) in def.inputs.iter().enumerate() {
                    let y = r.min.y + (HEADER + ROW * (i as f32 + 0.5)) * zoom;
                    let pin = pos2(r.min.x, y);
                    let c = if inp.kind == graph::Kind::Gray { Color32::from_gray(200) } else { Color32::from_rgb(230, 190, 90) };
                    painter.circle_filled(pin, 5.0 * zoom, if n.links.contains_key(&inp.name) { c } else { Color32::from_gray(70) });
                    painter.circle_stroke(pin, 5.0 * zoom, Stroke::new(1.0, c));
                    painter.text(pin + vec2(10.0 * zoom, 0.0), egui::Align2::LEFT_CENTER, inp.name.replace('_', " "), font(11.0), theme::TEXT);
                }
                let out_c = if def.output == graph::Kind::Gray { Color32::from_gray(200) } else { Color32::from_rgb(230, 190, 90) };
                painter.circle_filled(out_pins[&id], 5.5 * zoom, out_c);
            }
            let img = Rect::from_min_size(
                pos2(r.min.x + (NODE_W - PREVIEW) * 0.5 * zoom, r.min.y + (HEADER + ins as f32 * ROW + 6.0) * zoom),
                vec2(PREVIEW, PREVIEW) * zoom,
            );
            match e.thumbs.get(&(doc, id)) {
                _ if !picture => {}
                Some((tex, _)) if error.is_none() => {
                    painter.image(*tex, img, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
                }
                _ => {
                    painter.rect_filled(img, 0.0, Color32::from_gray(28));
                    if error.is_some() {
                        painter.text(img.center(), egui::Align2::CENTER_CENTER, "error", font(12.0), Color32::from_rgb(240, 110, 90));
                    }
                }
            }
            match (error, title_cut) {
                (Some(err), _) => {
                    resp.on_hover_text(err.as_str());
                }
                (None, Some(full)) => {
                    resp.on_hover_text(full);
                }
                _ => {}
            }

            // Output pin: drag a link out of it.
            let out_r = Rect::from_center_size(out_pins[&id], vec2(16.0, 16.0) * zoom.max(0.7));
            let out_resp = ui.interact(out_r, ui.scope_id().with(("graph out", doc, id)), Sense::drag());
            if out_resp.drag_started() {
                e.link_from = Some(id);
            }
            // Input pins: dragging a linked one picks the link back up.
            if let Some(def) = def {
                for inp in &def.inputs {
                    let Some((_, _, pin)) = in_pins.iter().find(|(nid, name, _)| *nid == id && *name == inp.name) else { continue };
                    let ir = Rect::from_center_size(*pin, vec2(16.0, 16.0) * zoom.max(0.7));
                    let ir_resp = ui.interact(ir, ui.scope_id().with(("graph in", doc, id, inp.name.as_str())), Sense::drag());
                    if ir_resp.drag_started()
                        && let Some(from) = d.graph.nodes[ni].links.remove(&inp.name) {
                            e.link_from = Some(from);
                            changed = true;
                        }
                }
            }
        }
        // Dragging a selected node moves the whole selection.
        if drag_by != egui::Vec2::ZERO {
            for n in d.graph.nodes.iter_mut().filter(|n| e.multi.contains(&n.id)) {
                n.pos[0] += drag_by.x;
                n.pos[1] += drag_by.y;
            }
            changed = true;
        }

        // A link in flight: follow the pointer; on release connect to the
        // input under it, or offer to add a node there, already linked.
        if let Some(from) = e.link_from {
            if let (Some(a), Some(p)) = (out_pins.get(&from), ui.input(|i| i.pointer.latest_pos())) {
                wire(&painter, *a, p, theme::ACCENT);
            }
            if ui.input(|i| i.pointer.any_released()) {
                if let Some(p) = ui.input(|i| i.pointer.latest_pos()) {
                    let hit = in_pins.iter().filter(|(_, _, pin)| pin.distance(p) < 12.0 * zoom.max(0.7)).min_by(|a, b| {
                        a.2.distance(p).total_cmp(&b.2.distance(p))
                    });
                    match hit {
                        Some((to, input, _)) => changed |= d.graph.connect(from, *to, input),
                        None if rect.contains(p) => {
                            e.menu = Some(AddMenu { at: to_world(p), screen: p, link_from: Some(from), focus: true });
                            e.search.clear();
                        }
                        None => {}
                    }
                }
                e.link_from = None;
            }
        }

        // Box selection: drag on empty canvas.
        if bg.drag_started_by(egui::PointerButton::Primary) {
            e.box_from = bg.interact_pointer_pos();
        }
        if let Some(a) = e.box_from {
            let b = ui.input(|i| i.pointer.latest_pos()).unwrap_or(a);
            let sel = Rect::from_two_pos(a, b);
            painter.rect_filled(sel, 0.0, theme::ACCENT.gamma_multiply(0.12));
            painter.rect_stroke(sel, 0.0, Stroke::new(1.0, theme::ACCENT), egui::StrokeKind::Inside);
            if !ui.input(|i| i.pointer.primary_down()) {
                if !(mods.shift || mods.command) {
                    e.multi.clear();
                }
                for n in &d.graph.nodes {
                    let r = Rect::from_min_size(to_screen(pos2(n.pos[0], n.pos[1])), size_of(n) * zoom);
                    if r.intersects(sel) && !e.multi.contains(&n.id) {
                        e.multi.push(n.id);
                    }
                }
                e.selected = e.multi.last().copied();
                e.box_from = None;
            }
        }

        // Empty canvas: click deselects; right-click, Space or Tab adds a node.
        if bg.clicked() {
            e.selected = None;
            e.multi.clear();
            e.menu = None;
        }
        let key_menu = over && !typing && e.menu.is_none() && ui.input(|i| i.key_pressed(egui::Key::Space) || i.key_pressed(egui::Key::Tab));
        if (bg.secondary_clicked() || key_menu)
            && let Some(p) = ui.input(|i| i.pointer.latest_pos()).filter(|p| rect.contains(*p)) {
                e.menu = Some(AddMenu { at: to_world(p), screen: p, link_from: None, focus: true });
                e.search.clear();
            }
        if let Some(def) = add_menu(ui, lib, e, d.shading) {
            let menu = e.menu.take().expect("menu was open");
            let id = d.graph.add(&def, [menu.at.x, menu.at.y]);
            // Dropped from a link: wire it to the new node's first input.
            if let (Some(from), Some(first)) = (menu.link_from, lib.find(&def).and_then(|d| d.inputs.first())) {
                d.graph.connect(from, id, &first.name);
            }
            e.selected = Some(id);
            e.multi = vec![id];
            changed = true;
        }

        // Keys over the canvas: delete, duplicate, copy and paste.
        if over && !typing {
            let (del, dup, copy, paste) = ui.input(|i| {
                let cmd = |k| i.modifiers.command && i.key_pressed(k);
                (
                    i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace),
                    cmd(egui::Key::D),
                    cmd(egui::Key::C) || i.events.iter().any(|e| matches!(e, egui::Event::Copy)),
                    cmd(egui::Key::V) || i.events.iter().any(|e| matches!(e, egui::Event::Paste(_))),
                )
            });
            if del && !e.multi.is_empty() {
                for id in std::mem::take(&mut e.multi) {
                    // A folder's layers are removed in the layer stack.
                    if d.graph.tag_of(id).is_none() {
                        d.graph.remove(id);
                    }
                }
                e.selected = None;
                changed = true;
            }
            if copy && !e.multi.is_empty() {
                e.clipboard = d.graph.nodes.iter().filter(|n| e.multi.contains(&n.id)).cloned().collect();
            }
            let source = if dup {
                Some((d.graph.nodes.iter().filter(|n| e.multi.contains(&n.id)).cloned().collect::<Vec<_>>(), vec2(40.0, 40.0)))
            } else if paste && !e.clipboard.is_empty() {
                // Pasted with its top-left corner at the pointer.
                let min = e.clipboard.iter().fold(pos2(f32::MAX, f32::MAX), |m, n| pos2(m.x.min(n.pos[0]), m.y.min(n.pos[1])));
                let at = ui.input(|i| i.pointer.latest_pos()).map_or(min + vec2(40.0, 40.0), to_world);
                Some((e.clipboard.clone(), at - min))
            } else {
                None
            };
            if let Some((nodes, offset)) = source.filter(|(n, _)| !n.is_empty()) {
                let ids = paste_nodes(&mut d.graph, &nodes, offset);
                e.selected = ids.last().copied();
                e.multi = ids;
                changed = true;
            }
        }

        if d.graph.nodes.is_empty() {
            painter.text(rect.center(), egui::Align2::CENTER_CENTER, "Right-click (or Space) to add nodes", egui::FontId::proportional(14.0), theme::TEXT_DIM);
        }
        painter.text(
            rect.left_bottom() + vec2(8.0, -6.0),
            egui::Align2::LEFT_BOTTOM,
            "Right-click / Space: add node  ·  Drag empty: select  ·  F: frame  ·  Ctrl+D / C / V  ·  Del  ·  Middle drag: pan",
            egui::FontId::proportional(11.0),
            theme::TEXT_DIM,
        );
        if changed {
            d.dirty = true;
        }
        if let Some(id) = edit_code {
            self.pending.push(Action::GraphEditCode(id));
        }
    }

    /// Keeps egui textures of the open graph's node thumbnails current.
    fn sync_thumbs(&mut self, doc: u64) {
        let Some(d) = self.graphs.docs.iter().find(|d| d.id == doc) else { return };
        let mut renderer = self.rs.renderer.write();
        let thumbs = &mut self.graphs.editor.thumbs;
        thumbs.retain(|(dd, n), (tex, _)| {
            let keep = *dd != doc || d.graph.node(*n).is_some();
            if !keep {
                renderer.free_texture(tex);
            }
            keep
        });
        for n in &d.graph.nodes {
            let Some(t) = d.rt.thumb(n.id) else { continue };
            let ptr = Arc::as_ptr(t) as usize;
            match thumbs.get_mut(&(doc, n.id)) {
                Some((_, p)) if *p == ptr => {}
                Some((tex, p)) => {
                    renderer.update_egui_texture_from_wgpu_texture(&self.rs.device, &t.view, eframe::wgpu::FilterMode::Linear, *tex);
                    *p = ptr;
                }
                None => {
                    let tex = renderer.register_native_texture(&self.rs.device, &t.view, eframe::wgpu::FilterMode::Linear);
                    thumbs.insert((doc, n.id), (tex, ptr));
                }
            }
        }
    }

    // ------------------------------------------------------------------ side panel

    fn node_panel(&mut self, ui: &mut egui::Ui, doc: u64) {
        let custom = self.bake_settings.custom.clone();
        let Some(sel) = self.graphs.editor.selected else {
            ui.label(egui::RichText::new("Select a node to edit it.").color(theme::TEXT_DIM));
            ui.add_space(8.0);
            ui.label(egui::RichText::new("Output nodes send their input to the layer's base color, roughness, metallic, height, or (for graph masks) the mask.").small().color(theme::TEXT_DIM));
            return;
        };
        let preview = self.graphs.editor.thumbs.get(&(doc, sel)).map(|t| t.0);
        let lib = &self.graphs.lib;
        let Some(d) = self.graphs.docs.iter_mut().find(|d| d.id == doc) else { return };
        let error = d.rt.errors.get(&sel).cloned();
        let Some(ni) = d.graph.nodes.iter().position(|n| n.id == sel) else { return };
        let node = &d.graph.nodes[ni];
        let Some(def) = lib.find(&node.def) else {
            ui.label(egui::RichText::new(format!("Unknown node '{}'. Its file is missing or has errors.", node.def)).color(Color32::from_rgb(240, 120, 90)));
            return;
        };
        ui.label(egui::RichText::new(&def.name).heading().color(Color32::WHITE));
        ui.label(egui::RichText::new(format!("{} · {}", def.category, if def.id.starts_with("user/") { "custom node" } else { "built-in" })).small().color(theme::TEXT_DIM));
        if let Some(t) = d.graph.tag_of(sel) {
            let what = match t.role {
                graph::Role::Mask => "its smart mask".to_string(),
                graph::Role::Channel(k) => format!("its {}", ["base color", "roughness", "metallic", "height", "emission"].get(k).unwrap_or(&"channel")),
                _ => "the folder's output".to_string(),
            };
            let layer = self.engine.layers.iter().find(|l| l.id == t.layer).map_or("the folder".to_string(), |l| format!("layer '{}'", l.name));
            ui.label(egui::RichText::new(format!("Shows {what} of {layer}: changes here change the layer, and the other way round.")).small().color(theme::ACCENT));
        }
        if !def.description.is_empty() {
            ui.label(egui::RichText::new(&def.description).color(theme::TEXT));
        }
        if let Some(e) = &error {
            ui.label(egui::RichText::new(e).color(Color32::from_rgb(240, 120, 90)).monospace().small());
        }
        let mut changed = false;
        // The pane scrolls as a whole (node settings, preview, material parameters).
        ui.scope(|ui| {
            if !def.params.is_empty() {
                section(ui, "Parameters");
            }
            // What the ◇ menus asked for, done once the node isn't borrowed.
            enum Bind {
                New,
                To(usize),
                Off,
            }
            let mut bind: Option<(String, Bind)> = None;
            for p in &def.params {
                let mut v = lib.value(&d.graph.nodes[ni], &p.name).unwrap_or_else(|| p.default.clone());
                let label = p.name.replace('_', " ");
                let bound = d.graph.param_of(sel, &p.name);
                // Graph parameters this one could join: the same kind of value.
                let joinable: Vec<(usize, String)> = d
                    .graph
                    .params
                    .iter()
                    .enumerate()
                    .filter(|(i, gp)| Some(*i) != bound && gp.binds.iter().any(|(n, k)| {
                        d.graph.nodes.iter().find(|x| x.id == *n).and_then(|x| lib.find(&x.def)).and_then(|def| def.params.iter().find(|q| &q.name == k))
                            .is_some_and(|q| std::mem::discriminant(&q.default) == std::mem::discriminant(&p.default))
                    }))
                    .map(|(i, gp)| (i, gp.name.clone()))
                    .collect();
                let c = labeled(ui, &label, |ui| {
                    let star = egui::RichText::new(if bound.is_some() { "◆" } else { "◇" })
                        .color(if bound.is_some() { Color32::from_rgb(255, 220, 120) } else { theme::TEXT_DIM });
                    let tip = match bound {
                        Some(i) => format!("Set by the material parameter \"{}\"", d.graph.params[i].name),
                        None => "Show this setting in the layer properties (Material Paint)".to_string(),
                    };
                    ui.menu_button(star, |ui| {
                        if let Some(i) = bound {
                            ui.label(egui::RichText::new(format!("Set by \"{}\"", d.graph.params[i].name)).small().color(theme::TEXT_DIM));
                            if ui.button("Stop showing it").clicked() {
                                bind = Some((p.name.clone(), Bind::Off));
                            }
                        } else if ui.button("Show as a new parameter").clicked() {
                            bind = Some((p.name.clone(), Bind::New));
                        }
                        if !joinable.is_empty() {
                            ui.separator();
                            ui.label(egui::RichText::new("Let a parameter set it too").small().color(theme::TEXT_DIM));
                            for (i, name) in &joinable {
                                if ui.button(name).clicked() {
                                    bind = Some((p.name.clone(), Bind::To(*i)));
                                }
                            }
                        }
                    })
                    .response
                    .on_hover_text(tip);
                    param_widget(ui, p, &mut v, sel, &custom)
                });
                if c {
                    d.graph.nodes[ni].params.insert(p.name.clone(), v.clone());
                    changed = true;
                    // The parameter's other nodes follow.
                    if let Some(i) = bound {
                        d.graph.set_param(i, v, lib);
                    }
                }
            }
            if let Some((name, b)) = bind {
                match b {
                    Bind::New => {
                        d.graph.expose(sel, &name, lib);
                    }
                    Bind::To(i) => d.graph.bind(i, sel, &name, lib),
                    Bind::Off => d.graph.unbind(sel, &name),
                }
                changed = true;
            }
            let node = &mut d.graph.nodes[ni];
            if !def.inputs.is_empty() {
                section(ui, "Inputs");
                for inp in &def.inputs {
                    let linked = node.links.contains_key(&inp.name);
                    labeled(ui, &inp.name.replace('_', " "), |ui| {
                        ui.label(egui::RichText::new(if linked { "connected" } else { "default value" }).color(theme::TEXT_DIM));
                        if linked && ui.small_button("✕").on_hover_text("Disconnect").clicked() {
                            node.links.remove(&inp.name);
                            changed = true;
                        }
                    });
                }
            }
        });
        let user = def.id.starts_with("user/");
        let def_id = def.id.clone();
        ui.horizontal(|ui| {
            if user {
                if ui.button("Edit code").clicked() {
                    self.pending.push(Action::GraphEditCode(def_id.clone()));
                }
            } else if ui.button("Edit a copy as custom node").on_hover_text("Copies this node's WGSL into your nodes folder and opens it").clicked() {
                self.pending.push(Action::GraphCopyNode(def_id.clone()));
            }
        });
        if changed {
            d.dirty = true;
        }
        if let Some(tex) = preview {
            let size = ui.available_width().min(270.0);
            ui.add_space(4.0);
            ui.add(egui::Image::new((tex, vec2(size, size))));
        }
    }

    // ------------------------------------------------------------------ custom nodes

    /// A new user node (a copy of `from`'s source, or the template), opened
    /// in the Shader Maker; with `to_graph`, also put into the open graph
    /// (replacing the selected node when that is the copied one).
    pub(super) fn new_custom_node(&mut self, from: Option<&str>, to_graph: bool) {
        let lib = &mut self.graphs.lib;
        let base = from.map_or("my_node".to_string(), |id| id.rsplit('/').next().unwrap_or(id).to_string());
        let stem = unique_stem(lib, &base);
        let source = match from.and_then(|id| lib.find(id)) {
            Some(d) => d.source.replacen(&format!("// @name {}", d.name), &format!("// @name {} (custom)", d.name), 1).replacen(
                &format!("// @category {}", d.category),
                "// @category Custom",
                1,
            ),
            None => graph::template("My Node"),
        };
        match lib.save_user(&stem, &source) {
            Ok(id) => {
                if let Some(doc) = self.graphs.editor.doc.filter(|_| to_graph)
                    && let Some(d) = self.graphs.docs.iter_mut().find(|d| d.id == doc) {
                        // Copying a selected node swaps it for the copy; else add one.
                        let sel = self.graphs.editor.selected.filter(|s| from.is_some() && d.graph.node(*s).is_some_and(|n| Some(n.def.as_str()) == from));
                        match sel {
                            Some(s) => d.graph.node_mut(s).unwrap().def = id.clone(),
                            None => {
                                let n = d.graph.add(&id, [self.graphs.editor.add_at.x, self.graphs.editor.add_at.y]);
                                self.graphs.editor.selected = Some(n);
                            }
                        }
                        d.dirty = true;
                    }
                self.open_code(&id);
            }
            Err(e) => self.status = format!("Couldn't create the node: {e}"),
        }
    }

    // ------------------------------------------------------------------ shelf and export

    fn save_graph_to_shelf(&mut self, ctx: &egui::Context, id: u64) {
        let Some(dir) = graphs_dir() else { return };
        let Some(d) = self.graphs.doc(id) else { return };
        let _ = std::fs::create_dir_all(&dir);
        let stem: String = d.graph.name.chars().map(|c| if c.is_alphanumeric() || c == ' ' || c == '-' || c == '_' { c } else { '_' }).collect();
        let path = dir.join(format!("{}.mpgraph", stem.trim()));
        if let Err(e) = graph::save_file(&path, &d.graph, &self.graphs.lib) {
            self.status = e;
            return;
        }
        // Thumbnail: the base color output, else the first output.
        let outs = d.graph.outputs(&self.graphs.lib);
        if let Some(t) = outs.iter().flatten().next().and_then(|n| d.rt.thumb(*n)) {
            let bytes = crate::gpu::read_texture(&self.rs.device, &self.rs.queue, t);
            let _ = image::save_buffer(path.with_extension("png"), &bytes, t.width, t.height, image::ExtendedColorType::Rgba8);
        }
        let name = d.graph.name.clone();
        self.graphs.reload_shelf(ctx);
        self.status = format!("Saved '{name}' to the shelf (Graphs).");
    }

    fn export_graph_outputs(&mut self, id: u64) {
        let Some(d) = self.graphs.doc(id) else { return };
        let Some(folder) = rfd::FileDialog::new().set_title("Export graph outputs").pick_folder() else { return };
        let outs = d.graph.outputs(&self.graphs.lib);
        let mut written = 0;
        for (usage, node) in outs.iter().enumerate() {
            let Some(tex) = node.and_then(|n| d.rt.output(n)) else { continue };
            let bytes = crate::gpu::read_texture(&self.rs.device, &self.rs.queue, tex);
            let px: Vec<[f32; 4]> = bytes.chunks_exact(8).map(|c| [0, 1, 2, 3].map(|k| half_to_f32([c[k * 2], c[k * 2 + 1]]).clamp(0.0, 1.0))).collect();
            let name = format!("{}_{}.png", d.graph.name, graph::USAGES[usage].replace(' ', ""));
            let path = folder.join(name);
            let (w, h) = (tex.width, tex.height);
            let result = match usage {
                0 => image::save_buffer(&path, &px.iter().flat_map(|p| [0, 1, 2].map(|k| (p[k] * 255.0).round() as u8)).collect::<Vec<u8>>(), w, h, image::ExtendedColorType::Rgb8),
                3 => {
                    let v: Vec<u16> = px.iter().map(|p| (p[0] * 65535.0).round() as u16).collect();
                    image::ImageBuffer::<image::Luma<u16>, _>::from_raw(w, h, v).map_or(Ok(()), |img| img.save(&path))
                }
                _ => image::save_buffer(&path, &px.iter().map(|p| (p[0] * 255.0).round() as u8).collect::<Vec<u8>>(), w, h, image::ExtendedColorType::L8),
            };
            match result {
                Ok(()) => written += 1,
                Err(e) => {
                    self.status = format!("{}: {e}", path.display());
                    return;
                }
            }
        }
        self.status = format!("Exported {written} graph output(s) to {}.", folder.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pasting_remaps_links_among_the_copies() {
        let mut g = Graph::new("T");
        let a = g.add("noise", [0.0, 0.0]);
        let b = g.add("levels", [100.0, 0.0]);
        let c = g.add("invert", [200.0, 0.0]);
        g.connect(a, b, "input");
        g.connect(b, c, "input");
        g.node_mut(b).unwrap().params.insert("gamma".into(), Value::Float(2.0));
        // Copy b and c: c's link follows the copy of b, b's link to a stays.
        let copies: Vec<graph::Node> = g.nodes.iter().filter(|n| n.id != a).cloned().collect();
        let ids = paste_nodes(&mut g, &copies, vec2(10.0, 20.0));
        assert_eq!(ids.len(), 2);
        let (b2, c2) = (g.node(ids[0]).unwrap(), g.node(ids[1]).unwrap());
        assert_eq!(b2.links["input"], a);
        assert_eq!(c2.links["input"], ids[0]);
        assert_eq!(b2.params["gamma"], Value::Float(2.0));
        assert_eq!(b2.pos, [110.0, 20.0]);
        assert_eq!(unique_stem(&Library::load(None), "noise"), "noise");
    }
}
