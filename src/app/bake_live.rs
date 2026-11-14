//! Bake window helpers that react while you tune a bake:
//!
//! - Live preview: every settings change bakes the document again at preview
//!   quality (512² at most, a few rays a texel) and shows it in place of the
//!   full bake, so masks and the views respond while you drag a slider. The
//!   CPU scene preparation (BVH, edges) is kept between previews and only
//!   redone when a setting it reads changes. A change during a preview bake
//!   starts the next one when it finishes, so dragging never queues work.
//!   Once the settings stop changing, the preview is baked again at higher
//!   resolutions (1024², then 2048²): at 512² thin UV islands (bevels) are a
//!   texel or two wide and show as hard bands when stretched over the texture
//!   set. A change drops any refinement in progress.
//!   Closing the window (or switching the preview off) brings the full bake
//!   back; Bake replaces both.
//! - The custom map slots: cards you add, pick and remove, and the settings
//!   of the picked one.
//! - 3D gizmos for the picked slot: an arrow to drag for where the fluid or
//!   wind comes from, a marker to drag over the mesh for a point.

use super::{PainterApp, ViewKind, labeled, section, slider, theme};
use crate::bake::{self, BakeReport, BakeRun, BakedMaps, CustomKind, CustomMap, PreparedScene};
use crate::icons;
use crate::viewport::ViewMode;
use eframe::egui;
use egui::{Color32, Rect, Stroke, vec2};
use glam::{Vec2, Vec3};
use std::sync::{Arc, mpsc};

/// Preview resolutions: the quick one baked on every change, then the ones
/// it's refined to while the settings stay put (each capped at the texture
/// set's).
const PREVIEW_STEPS: [u32; 3] = [512, 1024, 2048];
const PREVIEW_RES: u32 = PREVIEW_STEPS[0];
/// Grab distance for gizmo handles, in screen points.
const GRAB: f32 = 11.0;

/// The live bake preview (see the module docs).
pub(super) struct LivePreview {
    pub on: bool,
    /// The prepared scene previews reuse, under its [`prepare_key`](bake::BakeSettings::prepare_key).
    scene: Option<(String, Arc<PreparedScene>)>,
    preparing: Option<(String, mpsc::Receiver<Result<PreparedScene, String>>)>,
    /// The preview bake in progress: the settings it bakes and its resolution.
    run: Option<(String, u32, BakeRun)>,
    /// The settings and resolution of the preview on screen (`None`: no preview shown).
    shown: Option<(String, u32)>,
    /// Settings whose preview failed, and why (not retried until they change).
    failed: Option<(String, String)>,
    /// The full bake the preview covers, put back when it ends (`Some(None)`:
    /// there was none).
    saved: Option<Option<(BakedMaps, Arc<BakeReport>)>>,
    /// The settings of the last full bake: while they match, the full bake is
    /// what's shown and there is nothing to preview.
    pub baked: Option<String>,
    /// Preview bakes and scene preparations started (for tests: one per change).
    starts: u32,
}

impl Default for LivePreview {
    fn default() -> Self {
        Self { on: true, scene: None, preparing: None, run: None, shown: None, failed: None, saved: None, baked: None, starts: 0 }
    }
}

/// The bake window's custom-map editing state.
#[derive(Default)]
pub(super) struct CustomUi {
    /// The custom map (by id) whose settings are open and whose gizmo the 3D view shows.
    pub open: Option<u32>,
    /// A gizmo handle being dragged.
    drag: Option<Handle>,
    /// The next click on the mesh places the point.
    picking: bool,
    /// Where the 3D view was last laid out (for tests that drag a handle).
    pub view_rect: Option<Rect>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Handle {
    Direction,
    Point,
}

pub(super) fn kind_icon(kind: CustomKind) -> &'static str {
    match kind {
        CustomKind::Flow => icons::DROP,
        CustomKind::Exposure => icons::WIND,
        CustomKind::Deposit => icons::SNOWFLAKE,
        CustomKind::Contact => icons::MAGNET,
        CustomKind::Point => icons::CROSSHAIR,
    }
}

fn kind_color(kind: CustomKind) -> Color32 {
    match kind {
        CustomKind::Flow => Color32::from_rgb(90, 170, 255),
        CustomKind::Exposure => Color32::from_rgb(255, 176, 70),
        CustomKind::Deposit => Color32::from_rgb(235, 215, 165),
        CustomKind::Contact => Color32::from_rgb(200, 130, 255),
        CustomKind::Point => Color32::from_rgb(255, 95, 85),
    }
}

/// One line on what a kind is for, for the add menu.
fn kind_blurb(kind: CustomKind) -> &'static str {
    match kind {
        CustomKind::Flow => "Rain streaks, rust runs, leaks",
        CustomKind::Exposure => "Wind and sand erosion, sun bleaching",
        CustomKind::Deposit => "Sand, snow and dust drifts",
        CustomKind::Contact => "Grime where parts meet",
        CustomKind::Point => "Impact damage, scorching",
    }
}

/// A gizmo hint next to its handle, readable over any surface.
fn caption(painter: &egui::Painter, at: egui::Pos2, text: &str) {
    let pos = at + vec2(11.0, -9.0);
    let font = egui::FontId::proportional(12.0);
    painter.text(pos + vec2(1.0, 1.0), egui::Align2::LEFT_BOTTOM, text, font.clone(), Color32::from_black_alpha(220));
    painter.text(pos, egui::Align2::LEFT_BOTTOM, text, font, Color32::from_gray(235));
}

impl PainterApp {
    /// The settings of the bake on screen or about to be: the bake settings
    /// and the mesh they apply to.
    fn bake_key(&self, prepare: bool) -> String {
        let mesh = Arc::as_ptr(&self.mesh) as usize;
        let set = self.sets.active;
        if prepare { format!("{mesh}|{set}|{}", self.bake_settings.prepare_key()) } else { format!("{mesh}|{set}|{:?}", self.bake_settings) }
    }

    /// A full bake is starting: it replaces the preview (which stays on
    /// screen until the bake is done).
    pub(super) fn preview_full_bake_started(&mut self) {
        self.bake_preview.baked = Some(self.bake_key(false));
        self.bake_preview.run = None;
        self.bake_preview.preparing = None;
        self.bake_preview.saved = None;
        self.bake_preview.shown = None;
    }

    /// The refined preview of the current settings is on screen, and nothing
    /// more is baking (offscreen tests).
    #[cfg(test)]
    pub(crate) fn test_preview_shown(&self) -> bool {
        let sharpest = self.engine.res.min(PREVIEW_STEPS[PREVIEW_STEPS.len() - 1]);
        self.bake_preview.run.is_none() && self.bake_preview.shown == Some((self.bake_key(false), sharpest))
    }

    /// Where the gizmo's handle is on screen (offscreen tests drag it).
    #[cfg(test)]
    pub(crate) fn test_gizmo_handle(&self) -> Option<egui::Pos2> {
        let rect = self.custom_ui.view_rect?;
        let (_, c) = self.gizmo_slot()?;
        let (_, at) = self.gizmo_handle(&c)?;
        let p = self.view3d_px(at)?;
        Some(rect.min + vec2(p.x / self.ppp, p.y / self.ppp))
    }

    /// A custom map's settings, by position (offscreen tests).
    #[cfg(test)]
    pub(crate) fn test_custom(&self, k: usize) -> Option<CustomMap> {
        self.bake_settings.custom.get(k).cloned()
    }

    /// Preview bakes and preparations started so far (offscreen tests).
    #[cfg(test)]
    pub(crate) fn test_preview_starts(&self) -> u32 {
        self.bake_preview.starts
    }

    /// The full bake is on screen again, with no preview left over (offscreen tests).
    #[cfg(test)]
    pub(crate) fn test_preview_ended(&self) -> bool {
        self.bake_preview.shown.is_none() && self.bake_preview.saved.is_none()
    }

    /// Ends the preview; with `restore`, the full bake it covered comes back.
    pub(super) fn end_preview(&mut self, restore: bool) {
        self.bake_preview.run = None;
        self.bake_preview.preparing = None;
        self.bake_preview.shown = None;
        if let Some(saved) = self.bake_preview.saved.take()
            && restore {
                match saved {
                    Some((maps, report)) => self.engine.set_maps(maps, report),
                    None => self.engine.clear_maps(),
                }
            }
    }

    /// Advances the live preview; call every frame.
    pub(super) fn update_preview(&mut self, ctx: &egui::Context) {
        if !(self.bake_preview.on && self.show_bake && self.bake_job.is_none()) {
            if self.bake_job.is_none() {
                self.end_preview(true);
            }
            return;
        }
        let key = self.bake_key(false);
        if self.bake_preview.baked.as_ref() == Some(&key) {
            // Back to the settings of the full bake: show it.
            self.end_preview(true);
            return;
        }

        if let Some((pkey, rx)) = &self.bake_preview.preparing {
            match rx.try_recv() {
                Ok(Ok(scene)) => {
                    self.bake_preview.scene = Some((pkey.clone(), Arc::new(scene)));
                    self.bake_preview.preparing = None;
                }
                Ok(Err(e)) => {
                    self.bake_preview.failed = Some((key.clone(), e));
                    self.bake_preview.preparing = None;
                }
                Err(mpsc::TryRecvError::Empty) => return,
                Err(mpsc::TryRecvError::Disconnected) => self.bake_preview.preparing = None,
            }
        }
        let quick = self.engine.res.min(PREVIEW_RES);
        // A refinement of settings that have changed since is dropped: the
        // quick preview of the new ones comes first.
        if self.bake_preview.run.as_ref().is_some_and(|(k, res, _)| *k != key && *res > quick) {
            self.bake_preview.run = None;
        }
        if let Some((rkey, rres, run)) = &mut self.bake_preview.run {
            let Some(maps) = self.baker.as_ref().expect("a running live preview owns an initialized GPU baker")
                .pump(run, &self.rs.device, &self.rs.queue, ctx) else { return };
            let (shown, report) = ((rkey.clone(), *rres), run.report.clone());
            if self.bake_preview.saved.is_none() {
                self.bake_preview.saved = Some(self.engine.current_maps());
            }
            self.engine.set_maps(maps, report);
            self.bake_preview.shown = Some(shown);
            self.bake_preview.run = None;
        }

        // A quick preview of new settings, then sharper ones of the same.
        let res = match &self.bake_preview.shown {
            Some((k, r)) if *k == key => match PREVIEW_STEPS.iter().map(|s| (*s).min(self.engine.res)).find(|s| s > r) {
                Some(next) => next,
                None => return,
            },
            _ => quick,
        };
        if self.bake_preview.failed.as_ref().is_some_and(|(k, _)| *k == key) {
            return;
        }
        self.bake_preview.failed = None;
        let pkey = self.bake_key(true);
        if self.bake_preview.scene.as_ref().is_some_and(|(k, _)| *k == pkey) {
            self.ensure_baker();
        }
        match &self.bake_preview.scene {
            Some((k, scene)) if *k == pkey => {
                let run = self.baker.as_ref().expect("live preview initializes the GPU baker above")
                    .start(&self.rs.device, &self.rs.queue, &self.gpu_mesh, scene, &self.bake_settings.preview(), res);
                self.bake_preview.run = Some((key, res, run));
                self.bake_preview.starts += 1;
                ctx.request_repaint();
            }
            _ => {
                let (mesh, settings, set) = (self.mesh.clone(), self.bake_settings.clone(), self.sets.active);
                let (tx, rx) = mpsc::channel();
                let ctx = ctx.clone();
                std::thread::spawn(move || {
                    let _ = tx.send(bake::prepare(&mesh, &settings, set));
                    ctx.request_repaint();
                });
                self.bake_preview.preparing = Some((pkey, rx));
                self.bake_preview.starts += 1;
            }
        }
    }

    /// The live preview switch and what it's doing, for the top of the bake window.
    pub(super) fn preview_bar(&mut self, ui: &mut egui::Ui) {
        let frame = egui::Frame::new().fill(theme::PANEL_DARK).corner_radius(4.0).inner_margin(egui::Margin::symmetric(8, 6));
        frame.show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.bake_preview.on, egui::RichText::new(theme::with_icon(icons::LIGHTNING, "Live preview")).strong())
                    .on_hover_text("Bake a fast, low-resolution preview whenever a setting changes, and show it in the views and masks.");
                let (text, color) = if !self.bake_preview.on {
                    ("Off: changes show after you bake.".to_string(), theme::TEXT_DIM)
                } else if self.bake_job.is_some() {
                    ("Paused while baking.".to_string(), theme::TEXT_DIM)
                } else if let Some((_, e)) = &self.bake_preview.failed {
                    (format!("Can't preview: {e}"), Color32::from_rgb(255, 110, 90))
                } else if self.bake_preview.preparing.is_some() {
                    ("Preparing the mesh…".to_string(), theme::TEXT)
                } else if self.bake_preview.run.as_ref().is_some_and(|(_, res, _)| *res > PREVIEW_RES) {
                    ("Sharpening…".to_string(), theme::TEXT)
                } else if self.bake_preview.run.is_some() {
                    ("Updating…".to_string(), theme::TEXT)
                } else if self.bake_preview.shown.is_some() {
                    ("Showing a preview. Bake for full quality.".to_string(), theme::ACCENT)
                } else {
                    ("Showing the full bake.".to_string(), theme::TEXT_DIM)
                };
                if self.bake_preview.on && (self.bake_preview.preparing.is_some() || self.bake_preview.run.is_some()) {
                    ui.spinner();
                }
                ui.label(egui::RichText::new(text).small().color(color));
            });
        });
    }

    // ------------------------------------------------------------------ custom maps

    /// The custom maps section of the bake window.
    pub(super) fn custom_maps_section(&mut self, ui: &mut egui::Ui) {
        section(ui, "Custom maps");
        let count = self.bake_settings.custom.len();
        ui.horizontal(|ui| {
            ui.add_enabled_ui(count < bake::MAX_CUSTOM_MAPS, |ui| {
                ui.menu_button(theme::with_icon(icons::PLUS, "Add map"), |ui| {
                    ui.set_min_width(260.0);
                    for kind in CustomKind::ALL {
                        let text = egui::RichText::new(format!("{}  {}", kind_icon(kind), kind.name())).color(kind_color(kind));
                        let r = ui.add(egui::Button::new(text).right_text(egui::RichText::new(kind_blurb(kind)).small().color(theme::TEXT_DIM)));
                        if r.on_hover_text(kind.help()).clicked() {
                            self.add_custom_map(kind);
                            ui.close();
                        }
                    }
                })
                .response
                .on_hover_text("Bake a map of what happens around the model: fluid flow, wind, drifts, parts touching, a point. Add as many as you need.");
            });
            let text = match count {
                0 => String::new(),
                1 => "1 map".to_string(),
                n => format!("{n} maps"),
            };
            ui.label(egui::RichText::new(text).small().color(theme::TEXT_DIM));
        });
        if count == 0 {
            ui.label(
                egui::RichText::new("None yet. Add one to bake rain streaks, wind erosion, sand and snow drifts, or grime where parts meet; masks then read it with the Custom Bake Map preset.")
                    .small()
                    .color(theme::TEXT_DIM),
            );
            return;
        }
        let ids: Vec<u32> = self.bake_settings.custom.iter().map(|c| c.id).collect();
        for id in ids {
            self.custom_card(ui, id);
        }
    }

    /// Adds a custom map of a kind and opens it.
    pub(super) fn add_custom_map(&mut self, kind: CustomKind) {
        let i = self.bake_settings.add_custom(kind);
        if kind == CustomKind::Point {
            let (lo, hi) = (self.mesh.bounds_min, self.mesh.bounds_max);
            let c = &mut self.bake_settings.custom[i];
            c.point = ((lo + hi) * 0.5).to_array();
            c.distance = (hi - lo).length() * 0.35;
            self.custom_ui.picking = true;
        }
        self.select_custom(self.bake_settings.custom[i].id);
    }

    /// Opens a custom map's settings and shows it.
    pub(super) fn select_custom(&mut self, id: u32) {
        self.custom_ui.open = Some(id);
        self.view.mode = ViewMode::Custom(id);
    }

    fn custom_pos(&self, id: u32) -> Option<usize> {
        self.bake_settings.custom.iter().position(|c| c.id == id)
    }

    /// One custom map: a header to open, show, copy or remove it, and the
    /// open map's name and settings.
    fn custom_card(&mut self, ui: &mut egui::Ui, id: u32) {
        let Some(i) = self.custom_pos(id) else { return };
        let (kind, name) = (self.bake_settings.custom[i].kind, self.bake_settings.custom[i].name.clone());
        let open = self.custom_ui.open == Some(id);
        let shown = self.view.mode == ViewMode::Custom(id);
        let color = kind_color(kind);
        let frame = egui::Frame::new()
            .fill(if open { theme::PANEL_DARK } else { Color32::TRANSPARENT })
            .stroke(Stroke::new(1.0, if open { color.gamma_multiply(0.6) } else { theme::PANEL_DARK }))
            .corner_radius(4.0)
            .inner_margin(egui::Margin::symmetric(8, 6));
        let (mut remove, mut copy) = (false, false);
        frame.show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(kind_icon(kind)).color(color));
                if open {
                    // The open map's name, to type over.
                    let edit = egui::TextEdit::singleline(&mut self.bake_settings.custom[i].name).desired_width(200.0).hint_text(kind.name());
                    ui.add(edit).on_hover_text("This map's name, as masks and graphs list it");
                } else {
                    let title = egui::RichText::new(&name).color(color).strong();
                    if ui.add(egui::Button::new(title).frame(false)).on_hover_text("Edit this map (and show its gizmo in the 3D view)").clicked() {
                        self.select_custom(id);
                    }
                    ui.label(egui::RichText::new(kind.name()).small().color(theme::TEXT_DIM));
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.add(egui::Button::new(icons::TRASH).frame(false)).on_hover_text("Remove this map").clicked() {
                        remove = true;
                    }
                    let can_copy = self.bake_settings.custom.len() < bake::MAX_CUSTOM_MAPS;
                    if ui.add_enabled(can_copy, egui::Button::new(icons::COPY).frame(false)).on_hover_text("Duplicate this map (same settings, a new name)").clicked() {
                        copy = true;
                    }
                    let eye = if shown { icons::EYE } else { icons::EYE_SLASH };
                    if ui.add(egui::Button::new(eye).frame(false)).on_hover_text("Show this map in the views").clicked() {
                        self.view.mode = if shown { ViewMode::Material } else { ViewMode::Custom(id) };
                    }
                    if open && ui.add(egui::Button::new(icons::CHECK).frame(false)).on_hover_text("Done: close this map's settings").clicked() {
                        self.custom_ui.open = None;
                    }
                });
            });
            if open {
                self.custom_settings(ui, id);
            }
        });
        if remove {
            self.bake_settings.custom.remove(i);
            if open {
                self.custom_ui.open = None;
            }
            if shown {
                self.view.mode = ViewMode::Material;
            }
        } else if copy {
            let mut c = self.bake_settings.custom[i].clone();
            c.id = self.bake_settings.custom.iter().map(|c| c.id).max().unwrap_or(0) + 1;
            let taken = |n: &str| self.bake_settings.custom.iter().any(|c| c.name == n);
            c.name = (2..).map(|k| format!("{} {k}", name.trim_end_matches(|ch: char| ch.is_ascii_digit()).trim_end())).find(|n| !taken(n)).expect("a free name");
            let new_id = c.id;
            self.bake_settings.custom.insert(i + 1, c);
            self.select_custom(new_id);
        }
        ui.add_space(2.0);
    }

    /// The open map's settings, in words that say what they do.
    fn custom_settings(&mut self, ui: &mut egui::Ui, id: u32) {
        let Some(i) = self.custom_pos(id) else { return };
        let picking = &mut self.custom_ui.picking;
        let c = &mut self.bake_settings.custom[i];
        ui.label(egui::RichText::new(c.kind.help()).small().color(theme::TEXT_DIM));
        ui.add_space(2.0);
        if c.kind.has_direction() {
            let what = if c.kind == CustomKind::Flow { "Fluid comes from" } else { "Wind comes from" };
            labeled(ui, what, |ui| {
                for (name, az, el) in [
                    ("Above", None, 90.0),
                    ("Front", Some(0.0), 0.0),
                    ("Back", Some(180.0), 0.0),
                    ("Left", Some(270.0), 0.0),
                    ("Right", Some(90.0), 0.0),
                ] {
                    let on = (c.elevation - el).abs() < 0.5 && az.is_none_or(|a: f32| (c.azimuth - a).abs() < 0.5);
                    if ui.selectable_label(on, name).clicked() {
                        if let Some(az) = az {
                            c.azimuth = az;
                        }
                        c.elevation = el;
                    }
                }
            });
            ui.label(egui::RichText::new("Or drag the arrow's handle in the 3D view.").small().color(theme::TEXT_DIM));
            egui::CollapsingHeader::new(egui::RichText::new("Exact angle").small()).id_salt(("custom angle", id)).show(ui, |ui| {
                slider(ui, "Compass (°)", &mut c.azimuth, 0.0..=360.0);
                slider(ui, "Height (°)", &mut c.elevation, -90.0..=90.0);
            });
        }
        match c.kind {
            CustomKind::Flow => {
                slider(ui, "Streak length", &mut c.distance, 0.05..=2.0);
                slider(ui, "Wetness", &mut c.strength, 0.5..=12.0);
                slider(ui, "Meander (°)", &mut c.wander, 0.0..=60.0);
                slider(ui, "Rain angle spread (°)", &mut c.spread, 0.0..=60.0);
            }
            CustomKind::Exposure => {
                slider(ui, "Gustiness (°)", &mut c.spread, 0.0..=85.0);
                slider(ui, "Shelter distance", &mut c.distance, 0.05..=4.0);
            }
            CustomKind::Deposit => {
                slider(ui, "Gustiness (°)", &mut c.spread, 0.0..=85.0);
                slider(ui, "Shelter distance", &mut c.distance, 0.02..=2.0);
            }
            CustomKind::Contact => {
                slider(ui, "Reach", &mut c.distance, 0.005..=0.5);
            }
            CustomKind::Point => {
                labeled(ui, "Point", |ui| {
                    let text = if *picking { "Click the model…" } else { "Pick on model" };
                    if ui.selectable_label(*picking, egui::RichText::new(theme::with_icon(icons::CROSSHAIR, text))).clicked() {
                        *picking = !*picking;
                    }
                });
                ui.label(egui::RichText::new("Or drag the marker over the model in the 3D view.").small().color(theme::TEXT_DIM));
                slider(ui, "Radius", &mut c.distance, 0.05..=4.0);
            }
        }
    }

    // ------------------------------------------------------------------ 3D gizmos

    /// The custom map whose gizmo the 3D view shows: its index and settings.
    fn gizmo_slot(&self) -> Option<(usize, CustomMap)> {
        if !self.show_bake {
            return None;
        }
        let i = self.custom_pos(self.custom_ui.open?)?;
        Some((i, self.bake_settings.custom[i].clone()))
    }

    /// The mesh's center and how far from it the direction handle sits:
    /// just outside the mesh, but pulled in when the camera is close so the
    /// handle stays on screen.
    fn gizmo_frame(&self) -> (Vec3, f32) {
        let (lo, hi) = (self.mesh.bounds_min, self.mesh.bounds_max);
        let center = (lo + hi) * 0.5;
        let r = ((hi - lo).length() * 0.5).max(0.1);
        let visible = self.cam.eye().distance(center) * (self.cam.fov_y * 0.5).tan();
        (center, (visible * 0.75).clamp(r * 0.5, r * 1.35))
    }

    /// A world point in 3D view pixels.
    fn view3d_px(&self, w: Vec3) -> Option<Vec2> {
        let [sw, sh] = self.views.view3d.size;
        let size = Vec2::new(sw as f32, sh as f32);
        let clip = self.cam.view_proj(size.x / size.y) * w.extend(1.0);
        if clip.w <= 1e-5 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        Some(Vec2::new((ndc.x + 1.0) * 0.5 * size.x, (1.0 - ndc.y) * 0.5 * size.y))
    }

    fn gizmo_handle(&self, c: &CustomMap) -> Option<(Handle, Vec3)> {
        let (center, reach) = self.gizmo_frame();
        if c.kind.has_direction() {
            Some((Handle::Direction, center + c.direction() * reach))
        } else if c.kind == CustomKind::Point {
            Some((Handle::Point, Vec3::from_array(c.point)))
        } else {
            None
        }
    }

    /// Gizmo input in the 3D view (`cursor` in view pixels). True when the
    /// gizmo takes the input, so it doesn't paint or orbit.
    pub(super) fn gizmo_input(&mut self, ui: &egui::Ui, cursor: Option<Vec2>, pressed: bool, down: bool) -> bool {
        let Some((k, c)) = self.gizmo_slot() else {
            self.custom_ui.drag = None;
            return false;
        };
        let [w, h] = self.views.view3d.size;
        let size = Vec2::new(w as f32, h as f32);
        let mesh_hit = |app: &PainterApp, px: Vec2| {
            let (o, d) = app.cam.ray(px, size);
            app.bvh.raycast(o, d).map(|hit| hit.pos)
        };

        if let Some(handle) = self.custom_ui.drag {
            if !down {
                self.custom_ui.drag = None;
                return true;
            }
            ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
            let Some(px) = cursor else { return true };
            match handle {
                Handle::Direction => {
                    let (center, rad) = self.gizmo_frame();
                    let (o, d) = self.cam.ray(px, size);
                    // The handle rides on a sphere around the mesh: where the
                    // cursor's ray meets it, or the nearest point beside it.
                    let oc = o - center;
                    let b = oc.dot(d);
                    let disc = b * b - (oc.length_squared() - rad * rad);
                    let on = if disc >= 0.0 { o + d * (-b - disc.sqrt()) } else { o + d * (-b) };
                    let dir = (on - center).normalize_or_zero();
                    if dir != Vec3::ZERO {
                        let slot = &mut self.bake_settings.custom[k];
                        slot.elevation = dir.y.clamp(-1.0, 1.0).asin().to_degrees();
                        slot.azimuth = dir.x.atan2(dir.z).to_degrees().rem_euclid(360.0);
                    }
                }
                Handle::Point => {
                    if let Some(p) = mesh_hit(self, px) {
                        self.bake_settings.custom[k].point = p.to_array();
                    }
                }
            }
            return true;
        }

        let Some(px) = cursor else { return false };
        if self.custom_ui.picking && c.kind == CustomKind::Point {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
            if pressed {
                if let Some(p) = mesh_hit(self, px) {
                    self.bake_settings.custom[k].point = p.to_array();
                    self.custom_ui.picking = false;
                }
                return true;
            }
            return false;
        }
        let Some((handle, at)) = self.gizmo_handle(&c) else { return false };
        let near = self.view3d_px(at).is_some_and(|s| s.distance(px) <= GRAB * self.ppp);
        if near {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
            if pressed {
                self.custom_ui.drag = Some(handle);
                return true;
            }
        }
        false
    }

    /// Draws the picked slot's gizmo over the 3D view.
    pub(super) fn draw_gizmo(&self, painter: &egui::Painter, kind: ViewKind, rect: Rect) {
        if kind != ViewKind::V3d {
            return;
        }
        let Some((_, c)) = self.gizmo_slot() else { return };
        let ppp = self.ppp;
        let ui_at = |w: Vec3| self.view3d_px(w).map(|p| rect.min + vec2(p.x / ppp, p.y / ppp));
        let color = kind_color(c.kind);
        let shadow = Color32::from_black_alpha(150);
        let hot = self.custom_ui.drag.is_some();
        let arrow = |from: egui::Pos2, to: egui::Pos2, width: f32, color: Color32| {
            let d = to - from;
            let len = d.length();
            if len < 2.0 {
                return;
            }
            let u = d / len;
            let n = vec2(-u.y, u.x);
            let head = (len * 0.3).clamp(5.0, 14.0);
            let base = to - u * head;
            for (w, col) in [(width + 2.0, shadow), (width, color)] {
                painter.line_segment([from, base], Stroke::new(w, col));
            }
            let tri = vec![to, base + n * head * 0.5, base - n * head * 0.5];
            painter.add(egui::Shape::convex_polygon(tri, color, Stroke::new(1.0, shadow)));
        };
        if c.kind.has_direction() {
            let (center, reach) = self.gizmo_frame();
            let dir = c.direction();
            // The main arrow, pointing the way the fluid or wind travels,
            // and a few parallel ones around it suggesting the whole flow.
            let (a, b) = dir.any_orthonormal_pair();
            for (off, width, alpha) in [
                (Vec3::ZERO, 3.0, 1.0),
                (a * reach * 0.45, 1.5, 0.55),
                (-a * reach * 0.45, 1.5, 0.55),
                (b * reach * 0.45, 1.5, 0.55),
                (-b * reach * 0.45, 1.5, 0.55),
            ] {
                let start = if off == Vec3::ZERO { 1.0 } else { 0.9 };
                if let (Some(p0), Some(p1)) = (ui_at(center + off + dir * reach * start), ui_at(center + off + dir * reach * 0.55)) {
                    arrow(p0, p1, width, color.gamma_multiply(alpha));
                }
            }
            if let Some(h) = ui_at(center + dir * reach) {
                painter.circle_filled(h, if hot { 8.0 } else { 7.0 }, shadow);
                painter.circle_filled(h, if hot { 6.5 } else { 5.5 }, color);
                let label = if c.kind == CustomKind::Flow { "drag: where fluid comes from" } else { "drag: where wind comes from" };
                caption(painter, h, label);
            }
        } else if c.kind == CustomKind::Point {
            let p = Vec3::from_array(c.point);
            let Some(at) = ui_at(p) else { return };
            // The radius, as a circle facing the camera.
            let (right, _) = self.cam.basis();
            if let Some(edge) = ui_at(p + right * c.distance) {
                let rad = at.distance(edge);
                painter.circle_stroke(at, rad, Stroke::new(3.0, shadow));
                painter.circle_stroke(at, rad, Stroke::new(1.5, color.gamma_multiply(0.8)));
            }
            painter.circle_filled(at, if hot { 8.0 } else { 7.0 }, shadow);
            painter.circle_filled(at, if hot { 6.5 } else { 5.5 }, color);
            let label = if self.custom_ui.picking { "click the model to place" } else { "drag over the model" };
            caption(painter, at, label);
        }
    }
}
