//! Opening a Substance 3D Painter project (`.spp`) as a new project: reading it on a worker
//! thread (`crate::spp_import`), then the model, the flattened result as a paint layer, the
//! layer stack rebuilt as far as it maps onto Material Painter's layers, a bake, and a report
//! that says plainly what came across and what didn't.
//!
//! The rebuilt stack sits in a hidden folder above the flattened layer: the flattened layer is
//! the project exactly as it looked in Painter; the folder is its structure to edit (uniform
//! fills exactly, smart masks approximately, everything else as named, hidden placeholders).

use super::{PainterApp, theme};
use crate::engine::{Blend, FillParams, FilterKind, GenKind, MaskFilter, MatValues};
use crate::spp::document::ActionKind;
use crate::spp::{LayerInfo, LayerKind, SourceInfo, channel_bits};
use crate::spp_import::Imported;
use crate::undo::StrokeTarget;
use eframe::egui;
use egui::Color32;
use std::path::PathBuf;
use std::sync::mpsc;

/// Name of the folder holding the rebuilt Painter stack.
pub(super) const REBUILT: &str = "Painter layers (rebuilt)";
/// Name of the layer holding Painter's flattened result.
pub(super) const FLATTENED: &str = "Painter result (flattened)";

#[derive(Default)]
pub(super) struct SppOpen {
    job: Option<(PathBuf, mpsc::Receiver<Result<Imported, String>>)>,
    pub report: Option<Report>,
}

pub(super) struct Report {
    name: String,
    came: Vec<String>,
    left: Vec<String>,
    stack: Rebuilt,
}

/// What rebuilding the stack made.
#[derive(Default)]
pub(super) struct Rebuilt {
    folders: usize,
    fills: usize,
    /// Fills and masks made from something only close to Painter's (smart masks, partial fills).
    approximate: usize,
    masks: usize,
    /// Layers kept only as hidden, empty placeholders, by name and why.
    placeholders: Vec<(String, &'static str)>,
    /// Mask steps that couldn't be rebuilt.
    mask_steps_left: usize,
    other_blends: usize,
}

impl Rebuilt {
    /// Adds another texture set's counts.
    fn add(&mut self, o: Rebuilt) {
        self.folders += o.folders;
        self.fills += o.fills;
        self.approximate += o.approximate;
        self.masks += o.masks;
        self.placeholders.extend(o.placeholders);
        self.mask_steps_left += o.mask_steps_left;
        self.other_blends += o.other_blends;
    }
}

fn linear_to_srgb(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}

/// Painter's blending mode id as Material Painter's (ids as found by SubCraft); `None` when
/// there is no close one.
fn blend_of(mode: i32) -> Option<Blend> {
    Some(match mode {
        25 => Blend::Normal,
        4 => Blend::Multiply,
        29 => Blend::Subtract,
        2 | 11 | 15 => Blend::Overlay,
        _ => return None,
    })
}

/// A fill from a layer's uniform sources (`None` when it has none), and whether other sources
/// (images, procedural materials) were left out.
fn fill_of(l: &LayerInfo) -> Option<(FillParams, bool)> {
    let mut mat = MatValues { enabled: [false; crate::engine::CHANNELS], ..MatValues::default() };
    let mut partial = false;
    for s in &l.sources {
        match s {
            SourceInfo::Uniform { channels, color } => {
                if channels & channel_bits::BASE_COLOR != 0 {
                    mat.enabled[0] = true;
                    // Painter keeps uniform colors linear; fills here are sRGB.
                    mat.color = color.map(linear_to_srgb);
                }
                if channels & channel_bits::ROUGHNESS != 0 {
                    mat.enabled[1] = true;
                    mat.rough = color[0];
                }
                if channels & channel_bits::METALLIC != 0 {
                    mat.enabled[2] = true;
                    mat.metal = color[0];
                }
                if channels & channel_bits::HEIGHT != 0 {
                    mat.enabled[3] = true;
                    mat.height = color[0];
                }
            }
            _ => partial = true,
        }
    }
    mat.enabled.iter().any(|&e| e).then(|| (FillParams { mat, ..FillParams::default() }, partial))
}

/// A Painter smart-mask graph as a generator preset (approximately), by its name.
fn generator_of(name: &str) -> Option<GenKind> {
    Some(match name {
        "mg_dirt" | "mg_ambient_occlusion" => GenKind::Dirt,
        "mg_curvature" => GenKind::EdgeWear,
        "mg_position" => GenKind::PositionGradient,
        n if n.contains("edge_wear") => GenKind::EdgeWear,
        n if n.contains("dust") => GenKind::Dust,
        n if n.contains("leak") || n.contains("drip") => GenKind::Drips,
        n if n.contains("noise") || n.contains("grunge") || n.contains("fractal") => GenKind::Clouds,
        _ => return None,
    })
}

/// A layer's mask as Material Painter's: starting value, smart mask, filters. `None`: no mask.
fn mask_of(l: &LayerInfo, st: &mut Rebuilt) -> Option<(bool, Option<crate::engine::Generator>, Vec<MaskFilter>)> {
    if !l.has_mask || !l.mask_enabled {
        return None;
    }
    let mut white = l.mask_initial_white;
    let mut generator = None;
    let mut filters = Vec::new();
    for a in l.mask.iter().filter(|a| a.enabled) {
        match &a.kind {
            ActionKind::Fill { sources, .. } => {
                for s in sources {
                    match s {
                        SourceInfo::Uniform { color, .. } => white = color[0] > 0.5,
                        SourceInfo::Procedural { graph, .. } if generator.is_none() => match generator_of(graph.name()) {
                            Some(kind) => {
                                let mut g = kind.default_generator();
                                g.invert = ["invert", "global_invert"].iter().any(|k| graph.value(k).is_some_and(|v| v > 0.5));
                                generator = Some(g);
                                st.approximate += 1;
                            }
                            None => st.mask_steps_left += 1,
                        },
                        _ => st.mask_steps_left += 1,
                    }
                }
            }
            ActionKind::Levels(lv) => {
                let mut f = MaskFilter::new(FilterKind::Levels);
                (f.a, f.b, f.c) = (lv.in_min[0], lv.in_max[0], lv.gamma[0]);
                filters.push(f);
            }
            ActionKind::Filter { graph } if graph.name().contains("blur") => {
                filters.push(MaskFilter::new(FilterKind::Blur));
                st.approximate += 1;
            }
            _ => st.mask_steps_left += 1,
        }
    }
    st.masks += 1;
    Some((white, generator, filters))
}

impl PainterApp {
    /// Asks for a `.spp` (unless given) and reads it on a worker thread.
    pub(super) fn start_spp_open(&mut self, path: Option<PathBuf>, ctx: &egui::Context) {
        let Some(path) = path.or_else(|| rfd::FileDialog::new().add_filter("Substance 3D Painter project", &["spp"]).pick_file()) else {
            return;
        };
        let (tx, rx) = mpsc::channel();
        let (p, ctx) = (path.clone(), ctx.clone());
        std::thread::spawn(move || {
            let _ = tx.send(crate::spp_import::import(&p));
            ctx.request_repaint();
        });
        self.status = format!("Opening {}…", path.display());
        self.spp.job = Some((path, rx));
    }

    /// Picks up a finished read; call every frame.
    pub(super) fn poll_spp_open(&mut self, ctx: &egui::Context) {
        let Some((path, rx)) = &self.spp.job else { return };
        match rx.try_recv() {
            Ok(Ok(imported)) => {
                let path = path.clone();
                self.spp.job = None;
                self.apply_spp(&path, imported, ctx);
            }
            Ok(Err(e)) => {
                self.status = e;
                self.spp.job = None;
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                self.status = "Reading the Painter project stopped unexpectedly.".into();
                self.spp.job = None;
            }
        }
    }

    pub(super) fn spp_busy(&self) -> bool {
        self.spp.job.is_some()
    }

    /// A new project from an imported Painter project: the model, then each texture set's
    /// size, flattened result and rebuilt stack.
    fn apply_spp(&mut self, path: &std::path::Path, imported: Imported, ctx: &egui::Context) {
        let Imported { mesh, sets, came, left } = imported;
        self.graphs.clear(&self.rs);
        self.apply_project_defaults();
        // An imported Painter document is a new project: don't inherit channel
        // removals from whichever project happened to be open before it.
        self.active_channels = [true; crate::engine::CHANNELS];
        let sizes: Vec<u32> = sets.iter().map(|s| s.resolution).collect();
        self.load_mesh_sized(mesh, ctx, &sizes);
        let mut stack = Rebuilt::default();
        for (k, set) in sets.into_iter().enumerate() {
            self.switch_set(k);
            while !self.engine.layers.is_empty() {
                self.engine.delete_layer(0, &mut self.undo);
            }
            if self.engine.res != set.resolution {
                self.engine.set_resolution(set.resolution, &self.gpu_mesh, &mut self.undo);
            }
            let flat = self.engine.add_paint_layer_from(0, None, FLATTENED.into(), set.channels, &mut self.undo);
            stack.add(self.rebuild_spp_stack(&set.layers));
            self.selected = self.engine.layers.iter().position(|l| l.name == FLATTENED).unwrap_or(flat);
            self.target = StrokeTarget::Content;
        }
        self.switch_set(0);
        self.undo.clear();
        self.saved_serial = self.undo.state_serial();
        self.project_path = None;
        // Every set at its own size (set above), so after the sizes.
        self.start_bake_all(ctx);
        let name = path.file_name().map_or_else(|| self.mesh.name.clone(), |n| n.to_string_lossy().into_owned());
        self.status = format!("Opened {name} from Substance 3D Painter: see the report for what came across.");
        self.spp.report = Some(Report { name, came, left, stack });
    }

    /// Rebuilds Painter's layers (bottom to top) in a hidden folder on top of the stack.
    fn rebuild_spp_stack(&mut self, layers: &[LayerInfo]) -> Rebuilt {
        let mut st = Rebuilt::default();
        if layers.is_empty() {
            return st;
        }
        let at = self.engine.layers.len();
        let folder = self.engine.add_folder(at, None, REBUILT.into(), &mut self.undo);
        let folder = self.add_spp_layers(folder, layers, &mut st);
        self.engine.layers[folder].visible = false;
        self.engine.commit_props(&mut self.undo);
        st
    }

    /// Adds Painter layers into the folder at `folder`; returns the folder's index after.
    fn add_spp_layers(&mut self, mut folder: usize, layers: &[LayerInfo], st: &mut Rebuilt) -> usize {
        let fid = self.engine.layers[folder].id;
        for l in layers {
            // Each layer goes on top of the folder's contents, right below the folder.
            let idx = match l.kind {
                LayerKind::Group => {
                    let child = self.engine.add_folder(folder, Some(fid), l.name.clone(), &mut self.undo);
                    st.folders += 1;
                    self.add_spp_layers(child, &l.children, st)
                }
                LayerKind::Fill => match fill_of(l) {
                    Some((fill, partial)) => {
                        st.fills += 1;
                        if partial {
                            st.approximate += 1;
                        }
                        self.engine.add_fill_layer(folder, Some(fid), l.name.clone(), fill, &mut self.undo)
                    }
                    None => self.spp_placeholder(folder, fid, l, "procedural or image fill", st),
                },
                LayerKind::Paint => self.spp_placeholder(folder, fid, l, "brush strokes", st),
                LayerKind::Filter => self.spp_placeholder(folder, fid, l, "filter", st),
                LayerKind::Empty => self.spp_placeholder(folder, fid, l, "empty in Painter", st),
            };
            folder = idx + 1;
            let layer = &mut self.engine.layers[idx];
            if layer.visible {
                layer.visible = l.visible;
            }
            if let Some(&(_, mode, opacity)) = l.blends.first() {
                layer.opacity = opacity.clamp(0.0, 1.0);
                match blend_of(mode) {
                    Some(b) => layer.blend = b,
                    None => st.other_blends += 1,
                }
            }
            if let Some((white, generator, filters)) = mask_of(l, st) {
                self.engine.add_mask(idx, white, generator, &mut self.undo);
                if let Some(m) = self.engine.layers[idx].mask.as_mut() {
                    m.filters = filters;
                }
            }
        }
        folder
    }

    /// A hidden, empty layer standing for one that can't be rebuilt, named after it.
    fn spp_placeholder(&mut self, folder: usize, fid: u64, l: &LayerInfo, why: &'static str, st: &mut Rebuilt) -> usize {
        let idx = self.engine.add_paint_layer(folder, Some(fid), Some(format!("{} (not rebuilt: {why})", l.name)), &mut self.undo);
        self.engine.layers[idx].visible = false;
        st.placeholders.push((l.name.clone(), why));
        idx
    }

    /// The report of an import: what came across, what was rebuilt, what wasn't.
    pub(super) fn spp_report_window(&mut self, ctx: &egui::Context) {
        let Some(r) = &self.spp.report else { return };
        let mut open = true;
        let mut close = false;
        egui::Window::new("Opened from Substance 3D Painter")
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_width(520.0)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                ui.label(egui::RichText::new(&r.name).strong().color(Color32::WHITE));
                ui.label(
                    egui::RichText::new(format!(
                        "The layer \"{FLATTENED}\" shows the project as it looked in Painter. To edit its structure instead, show the folder \"{REBUILT}\" and hide the flattened layer."
                    ))
                    .color(theme::TEXT),
                );
                ui.add_space(6.0);
                egui::ScrollArea::vertical().max_height(380.0).show(ui, |ui| {
                    let green = Color32::from_rgb(120, 210, 140);
                    let amber = Color32::from_rgb(240, 190, 90);
                    ui.label(egui::RichText::new("Came across").strong().color(green));
                    for line in &r.came {
                        ui.label(format!("✓ {line}"));
                    }
                    let s = &r.stack;
                    if s.folders + s.fills + s.placeholders.len() > 0 {
                        ui.label(format!(
                            "✓ The layer stack, rebuilt: {} folder{}, {} fill{}{}, {} mask{}.",
                            s.folders,
                            if s.folders == 1 { "" } else { "s" },
                            s.fills,
                            if s.fills == 1 { "" } else { "s" },
                            if s.approximate > 0 { format!(" ({} part{} only approximately)", s.approximate, if s.approximate == 1 { "" } else { "s" }) } else { String::new() },
                            s.masks,
                            if s.masks == 1 { "" } else { "s" },
                        ));
                    }
                    ui.add_space(6.0);
                    ui.label(egui::RichText::new("Not imported").strong().color(amber));
                    for line in &r.left {
                        ui.label(format!("• {line}"));
                    }
                    if !s.placeholders.is_empty() {
                        ui.label(format!("• {} layer{} kept only as hidden, empty placeholders with their names:", s.placeholders.len(), if s.placeholders.len() == 1 { "" } else { "s" }));
                        for (name, why) in s.placeholders.iter().take(20) {
                            ui.label(egui::RichText::new(format!("    {name} ({why})")).small().color(theme::TEXT_DIM));
                        }
                        if s.placeholders.len() > 20 {
                            ui.label(egui::RichText::new(format!("    and {} more", s.placeholders.len() - 20)).small().color(theme::TEXT_DIM));
                        }
                    }
                    if s.mask_steps_left > 0 {
                        ui.label(format!("• {} mask step{} (painted mask strokes, unknown generators) not rebuilt.", s.mask_steps_left, if s.mask_steps_left == 1 { "" } else { "s" }));
                    }
                    if s.other_blends > 0 {
                        ui.label(format!("• {} blend mode{} with no equivalent here, set to Normal.", s.other_blends, if s.other_blends == 1 { "" } else { "s" }));
                    }
                });
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new(
                        "Material Painter is not affiliated with or endorsed by Adobe; Substance 3D Painter is Adobe's trademark. \
                         Only your own project's data is read: the contents of Substance (.sbsar) materials are never opened. \
                         Reading .spp files is adapted from SubCraft (github.com/Aalzard/subcraft, MIT).",
                    )
                    .small()
                    .color(theme::TEXT_DIM),
                );
                if ui.button("Close").clicked() {
                    close = true;
                }
            });
        if !open || close {
            self.spp.report = None;
        }
    }
}

#[cfg(test)]
impl PainterApp {
    /// Opens a Painter project (offscreen tests).
    pub(crate) fn test_open_spp(&mut self, path: &std::path::Path, ctx: &egui::Context) {
        self.start_spp_open(Some(path.to_path_buf()), ctx);
    }

    /// The import finished: its report is up (offscreen tests).
    pub(crate) fn test_spp_done(&self) -> bool {
        self.spp.job.is_none() && self.spp.report.is_some()
    }

    /// Names, kinds and visibility of the layers, bottom to top (offscreen tests).
    pub(crate) fn test_layers(&self) -> Vec<(String, &'static str, bool)> {
        use crate::engine::Content;
        self.engine
            .layers
            .iter()
            .map(|l| {
                let kind = match &l.content {
                    Content::Paint(ch) if ch.iter().any(Option::is_some) => "paint",
                    Content::Paint(_) => "empty paint",
                    Content::Fill(_) => "fill",
                    Content::Folder(_) => "folder",
                };
                (l.name.clone(), kind, l.visible)
            })
            .collect()
    }
}
