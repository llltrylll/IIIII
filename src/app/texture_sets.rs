//! Texture sets: one document (layer stack, resolution, mesh maps) per
//! material of the mesh. The set being edited lives in `PainterApp::engine`
//! (with `selected` and `target`); the others wait here until switched to.
//! History is shared: undoing a step of another set switches to it first.

use super::{PainterApp, icons, theme};
use crate::bvh::Bvh;
use crate::engine::Engine;
use crate::undo::{StrokeTarget, UndoStack};
use crate::viewport::{OverlaySlot, SetLook};
use eframe::egui;
use std::collections::VecDeque;
use std::sync::Arc;

/// How long a texture set flashes after it's picked, in seconds.
const FLASH_SECONDS: f64 = 1.6;
/// Highlight of a pinned texture set.
const PINNED: f32 = 0.7;

/// A texture set that isn't being edited, with the panel state it had.
pub(super) struct Parked {
    pub engine: Engine,
    pub selected: usize,
    pub target: StrokeTarget,
}

#[derive(Default)]
pub(super) struct TextureSets {
    /// The set in `PainterApp::engine`, by position in `Mesh::sets`.
    pub active: usize,
    /// One slot per mesh set; `None` for the active one.
    pub parked: Vec<Option<Parked>>,
    /// The set the running bake is for.
    pub baking: usize,
    /// Sets to bake after it ("Bake all").
    pub bake_queue: VecDeque<usize>,
    /// Per set: not drawn, not pickable (display only, not saved).
    pub hidden: Vec<bool>,
    /// Per set: always highlighted.
    pub pinned: Vec<bool>,
    /// The set flashing since a time (`egui` input time).
    pub flash: Option<(usize, f64)>,
    /// The set under the pointer in a set list this frame.
    pub hover: Option<usize>,
}

/// The engines of `count` sets: `active` (the edited one) and the parked
/// ones. Borrows fields only, so a view can render while they're borrowed.
pub(super) fn engines<'a>(active: &'a Engine, sets: &'a TextureSets, count: usize) -> Vec<&'a Engine> {
    (0..count)
        .map(|k| match sets.parked.get(k).and_then(Option::as_ref) {
            Some(p) => &p.engine,
            None => active,
        })
        .collect()
}

impl PainterApp {
    /// Every set's engine, in mesh order (what the 3D view draws).
    pub(super) fn set_engines(&self) -> Vec<&Engine> {
        engines(&self.engine, &self.sets, self.mesh.sets.len())
    }

    /// Set `k`'s engine, active or parked.
    pub(super) fn set_engine_mut(&mut self, k: usize) -> Option<&mut Engine> {
        if k == self.sets.active {
            return Some(&mut self.engine);
        }
        self.sets.parked.get_mut(k)?.as_mut().map(|p| &mut p.engine)
    }

    /// Makes set `k` the one being edited.
    pub(super) fn switch_set(&mut self, k: usize) {
        if k == self.sets.active || self.stroke.is_some() {
            return;
        }
        let Some(mut p) = self.sets.parked.get_mut(k).and_then(Option::take) else { return };
        // A bake preview shows on the set being edited; the full bake comes back first.
        self.end_preview(true);
        self.engine.commit_props(&mut self.undo);
        std::mem::swap(&mut self.engine, &mut p.engine);
        std::mem::swap(&mut self.selected, &mut p.selected);
        std::mem::swap(&mut self.target, &mut p.target);
        self.sets.parked[self.sets.active] = Some(p);
        self.sets.active = k;
        self.undo.set = k;
        // Polygon fill selections are triangles of the set that was edited.
        self.poly.shown = None;
        self.poly.filled.clear();
        self.overlays.clear(OverlaySlot::Selection);
        // Graphs evaluate on the edited set's mesh maps.
        for d in &mut self.graphs.docs {
            d.dirty = true;
        }
        self.status = format!("Texture set: {}.", self.mesh.sets[k].name);
    }

    /// Before an undo or redo: the step belongs to `set`, which becomes the
    /// edited one so the step applies to its layers.
    pub(super) fn switch_for_history(&mut self, redo: bool) {
        let stack = if redo { &self.undo.redo } else { &self.undo.undo };
        if let Some(set) = stack.last().map(|e| e.set) {
            self.switch_set(set);
        }
    }

    /// Composites the parked sets that changed (new mesh, bake, load).
    pub(super) fn composite_parked(&mut self) {
        for p in self.sets.parked.iter_mut().flatten() {
            if p.engine.dirty {
                p.engine.composite();
            }
        }
    }

    /// After the mesh changed (`old_names`: the previous sets): every set
    /// keeps its layers when a new set has its name, else the next unused
    /// one in order; new sets start with one paint layer, at `sizes[k]` when
    /// given (a project or import about to fill them), else the default
    /// size. The edited set stays edited when it survives.
    pub(super) fn rebuild_sets(&mut self, old_names: &[String], sizes: &[u32]) {
        let new: Vec<(String, std::ops::Range<u32>)> =
            self.mesh.sets.iter().map(|s| (s.name.clone(), s.index_range())).collect();
        let mut used = vec![false; old_names.len()];
        let mut assign: Vec<Option<usize>> = new
            .iter()
            .map(|(name, _)| {
                let j = old_names.iter().enumerate().position(|(j, n)| !used[j] && n == name)?;
                used[j] = true;
                Some(j)
            })
            .collect();
        for a in assign.iter_mut().filter(|a| a.is_none()) {
            if let Some(j) = used.iter().position(|u| !u) {
                used[j] = true;
                *a = Some(j);
            }
        }
        let old_active = self.sets.active;
        let mut old = std::mem::take(&mut self.sets.parked);
        old.resize_with(old_names.len().max(old_active + 1), || None);
        let active = assign.iter().position(|a| *a == Some(old_active)).unwrap_or(0);
        // Made at their final size: making every set at the default size and
        // resizing it after costs several sets' worth of GPU memory at once.
        let size = |k: usize| sizes.get(k).copied().unwrap_or(self.prefs.default_resolution);
        let fresh = |engine: &Engine, range: std::ops::Range<u32>, mesh: &crate::gpu::GpuMesh, res: u32| {
            let mut e = engine.sibling(mesh, range, res);
            e.add_paint_layer(0, None, None, &mut UndoStack::default());
            Parked { engine: e, selected: 0, target: StrokeTarget::Content }
        };
        let mut parked = Vec::with_capacity(new.len());
        for (i, (_, range)) in new.iter().enumerate() {
            if i == active {
                parked.push(None);
                continue;
            }
            let kept = assign[i].filter(|&j| j != old_active).and_then(|j| old[j].take());
            let p = match kept {
                Some(p) => p,
                None => fresh(&self.engine, range.clone(), &self.gpu_mesh, size(i)),
            };
            parked.push(Some(p));
        }
        if assign[active] != Some(old_active) {
            let p = match assign[active].and_then(|j| old[j].take()) {
                Some(p) => p,
                None => fresh(&self.engine, new[active].1.clone(), &self.gpu_mesh, size(active)),
            };
            self.engine = p.engine;
            self.selected = p.selected;
            self.target = p.target;
        }
        // History steps name sets by position; they only stay valid when
        // every set kept its place.
        let identity = assign.len() == old_names.len() && assign.iter().enumerate().all(|(i, a)| *a == Some(i));
        if !identity {
            self.undo.clear();
        }
        self.sets.parked = parked;
        self.sets.active = active;
        self.sets.bake_queue.clear();
        self.sets.hidden = vec![false; new.len()];
        self.sets.pinned = vec![false; new.len()];
        self.sets.flash = None;
        self.undo.set = active;
        let bounds = (self.mesh.bounds_min, self.mesh.bounds_max);
        for (k, (_, range)) in new.into_iter().enumerate() {
            let gpu_mesh = &self.gpu_mesh;
            let e = match self.sets.parked[k].as_mut() {
                Some(p) => &mut p.engine,
                None => &mut self.engine,
            };
            e.set_bounds(bounds);
            e.set_triangles(range, gpu_mesh);
        }
        for d in &mut self.graphs.docs {
            d.dirty = true;
        }
    }

    /// Flashes texture set `k` in the 3D view, to show which part of the mesh it is.
    pub(super) fn flash_set(&mut self, k: usize, ctx: &egui::Context) {
        self.sets.flash = Some((k, ctx.input(|i| i.time)));
        ctx.request_repaint();
    }

    /// Edits texture set `k` (from a set list) and flashes it.
    pub(super) fn pick_set(&mut self, k: usize, ctx: &egui::Context) {
        self.switch_set(k);
        self.flash_set(k, ctx);
    }

    /// Shows or hides texture set `k`'s part of the mesh. Hidden parts can't
    /// be painted or picked through: the picking BVH and the path tracer
    /// leave them out.
    pub(super) fn set_hidden(&mut self, k: usize, hidden: bool) {
        let n = self.mesh.sets.len();
        self.sets.hidden.resize(n, false);
        if k >= n || self.sets.hidden[k] == hidden {
            return;
        }
        self.sets.hidden[k] = hidden;
        let mesh = self.mesh.clone();
        let shown: Vec<std::ops::Range<u32>> =
            mesh.sets.iter().zip(&self.sets.hidden).filter(|(_, h)| !**h).map(|(s, _)| s.tris.clone()).collect();
        self.bvh = Arc::new(if shown.len() == n {
            Bvh::from_mesh(&mesh)
        } else {
            Bvh::from_mesh_where(&mesh, |t| shown.iter().any(|r| r.contains(&(t as u32))))
        });
        self.raytracer.set_mesh(&self.rs.device, &mesh, &self.bvh);
        self.poly.shown = None;
        self.status = format!("{} {}.", if hidden { "Hid" } else { "Showed" }, mesh.sets[k].name);
    }

    /// Whether triangle `t` belongs to a hidden texture set.
    pub(super) fn tri_hidden(&self, t: u32) -> bool {
        self.sets.hidden.iter().any(|h| *h) && self.sets.hidden.get(self.mesh.set_of(t)).copied().unwrap_or(false)
    }

    /// How the 3D view draws each set this frame: hidden ones left out,
    /// pinned ones tinted, a just-picked one pulsing for a moment.
    pub(super) fn set_looks(&mut self, ctx: &egui::Context) -> Vec<SetLook> {
        let n = self.mesh.sets.len();
        let mut looks: Vec<SetLook> = (0..n)
            .map(|k| SetLook {
                hidden: self.sets.hidden.get(k).copied().unwrap_or(false),
                highlight: if self.sets.pinned.get(k).copied().unwrap_or(false) { PINNED } else { 0.0 },
            })
            .collect();
        if let Some(k) = self.sets.hover.take().filter(|&k| k < n) {
            looks[k].highlight = looks[k].highlight.max(0.45);
        }
        if let Some((k, start)) = self.sets.flash {
            let t = ctx.input(|i| i.time) - start;
            if t < FLASH_SECONDS && k < n {
                // Pulses (on at once), fading out.
                let pulse = 0.5 + 0.5 * (t * std::f64::consts::TAU * 1.5).cos();
                let fade = 1.0 - t / FLASH_SECONDS;
                looks[k].highlight = looks[k].highlight.max((pulse * fade) as f32);
                ctx.request_repaint();
            } else {
                self.sets.flash = None;
            }
        }
        looks
    }

    /// The texture set list: click a set to edit it (it flashes in the 3D
    /// view), hide its part of the mesh, or pin a highlight on it.
    pub(super) fn texture_set_list(&mut self, ui: &mut egui::Ui) {
        let n = self.mesh.sets.len();
        self.sets.hidden.resize(n, false);
        self.sets.pinned.resize(n, false);
        let shown = self.sets.hidden.iter().filter(|h| !**h).count();
        let (mut pick, mut hide, mut pin) = (None, None, None);
        for (k, set) in self.mesh.sets.iter().enumerate() {
            let tris = set.tris.end - set.tris.start;
            let hidden = self.sets.hidden[k];
            let pinned = self.sets.pinned[k];
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 2.0;
                let eye = if hidden { icons::EYE_SLASH } else { icons::EYE };
                let tip = if hidden { "Show this part of the mesh" } else { "Hide this part of the mesh (it can't be painted while hidden)" };
                // The last part shown stays: hiding everything leaves nothing to paint.
                let can = hidden || shown > 1;
                let eye_resp = ui.add_enabled_ui(can, |ui| theme::icon_button(ui, eye, false, 22.0)).inner;
                if eye_resp.on_hover_text(tip).clicked() {
                    hide = Some((k, !hidden));
                }
                if theme::icon_button(ui, icons::LIGHTBULB, pinned, 22.0).on_hover_text("Keep this texture set highlighted in the 3D view").clicked() {
                    pin = Some(k);
                }
                let text = if hidden { egui::RichText::new(&set.name).color(theme::TEXT_DIM) } else { egui::RichText::new(&set.name) };
                let row = theme::selectable(k == self.sets.active, text).min_size(egui::vec2(ui.available_width(), 22.0));
                let resp = ui
                    .add(row)
                    .on_hover_text(format!("{} ({tris} triangles). Click to paint this texture set; it flashes in the 3D view.", set.name));
                if resp.hovered() {
                    self.sets.hover = Some(k);
                }
                if resp.clicked() {
                    pick = Some(k);
                }
            });
        }
        if let Some((k, h)) = hide {
            self.set_hidden(k, h);
        }
        if let Some(k) = pin {
            self.sets.pinned[k] = !self.sets.pinned[k];
            ui.ctx().request_repaint();
        }
        if let Some(k) = pick {
            self.pick_set(k, ui.ctx());
        }
    }
}

#[cfg(test)]
impl PainterApp {
    /// Each texture set's name, size and layer names, bottom to top (offscreen tests).
    pub(crate) fn test_sets(&self) -> Vec<(String, u32, Vec<String>)> {
        self.mesh
            .sets
            .iter()
            .zip(self.set_engines())
            .map(|(s, e)| (s.name.clone(), e.res, e.layers.iter().map(|l| l.name.clone()).collect()))
            .collect()
    }

    /// Uses the user's preferences file instead of the test defaults (offscreen tests).
    pub(crate) fn test_user_prefs(&mut self, json: &str) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(json) {
            self.prefs = super::prefs::Preferences::from_json(&v);
            self.apply_project_defaults();
        }
    }

    /// Each texture set's name and texture memory breakdown (offscreen tests).
    pub(crate) fn test_vram(&self) -> Vec<(String, Vec<(&'static str, u64)>)> {
        self.mesh.sets.iter().zip(self.set_engines()).map(|(s, e)| (s.name.clone(), e.vram_breakdown())).collect()
    }

    /// Saves the project to `path` and opens it again (offscreen tests).
    pub(crate) fn test_save_and_reopen(&mut self, path: &std::path::Path, ctx: &egui::Context) -> Result<(), String> {
        self.save_project(path)?;
        self.open_project(path, ctx)
    }

    /// A bake is running or queued (offscreen tests).
    pub(crate) fn test_baking(&self) -> bool {
        self.bake_job.is_some() || !self.sets.bake_queue.is_empty()
    }

    /// Turns the path-traced preview on or off (offscreen tests).
    pub(crate) fn test_ray_trace(&mut self, on: bool) {
        self.rt.enabled = on;
    }

    /// The path-traced preview has all its samples (offscreen tests).
    pub(crate) fn test_ray_trace_done(&self) -> bool {
        self.raytracer.converged(&self.rt)
    }

    /// Loads `mesh` as the paintable mesh (offscreen tests).
    pub(crate) fn test_load_mesh(&mut self, mesh: crate::mesh::Mesh, ctx: &egui::Context) {
        self.load_mesh(mesh, ctx);
    }

    /// Changes with every recorded edit (offscreen tests).
    pub(crate) fn test_undo_serial(&self) -> u64 {
        self.undo.state_serial()
    }

    /// Picks the polygon fill tool in `mode` (offscreen tests).
    pub(crate) fn test_poly_tool(&mut self, mode: crate::polyfill::FillMode, lasso: bool) {
        self.set_tool(super::Tool::PolyFill);
        self.poly.mode = mode;
        self.poly.lasso = lasso;
    }

    /// Per set: hidden, flashing or highlighted this frame (offscreen tests).
    pub(crate) fn test_set_state(&self) -> Vec<(bool, bool)> {
        (0..self.mesh.sets.len())
            .map(|k| (self.sets.hidden.get(k).copied().unwrap_or(false), self.sets.flash.is_some_and(|(f, _)| f == k)))
            .collect()
    }

    /// Hides or shows set `k` (offscreen tests).
    pub(crate) fn set_hidden_for_test(&mut self, k: usize, hidden: bool) {
        self.set_hidden(k, hidden);
    }

    /// Starts dragging the first shelf material, as a press on its tile does (offscreen tests).
    pub(crate) fn test_drag_material(&self, ctx: &egui::Context) {
        self.test_drag(ctx, super::ShelfItem::Material(0));
    }

    /// Starts dragging shelf item `item`, as a press on its tile does (offscreen tests).
    pub(crate) fn test_drag(&self, ctx: &egui::Context, item: super::ShelfItem) {
        egui::DragAndDrop::set_payload(ctx, item);
    }

    /// Per layer, bottom to top: whether it has a generator mask, and which
    /// fill channels have an image (offscreen tests).
    pub(crate) fn test_layer_sources(&self) -> Vec<(bool, Vec<usize>)> {
        self.engine
            .layers
            .iter()
            .map(|l| {
                let generator = l.mask.as_ref().is_some_and(|m| m.generator.is_some());
                (generator, (0..crate::engine::CHANNELS).filter(|&ch| l.images[ch].is_some()).collect())
            })
            .collect()
    }

    /// The texture set under a point of the 3D view (offscreen tests).
    pub(crate) fn test_set_at(&mut self, at: egui::Pos2) -> Option<usize> {
        let rect = self.test_view_rect()?;
        self.set_under(super::ViewKind::V3d, glam::Vec2::new(at.x - rect.min.x, at.y - rect.min.y) * self.ppp)
    }

    /// The texture set being edited (offscreen tests).
    pub(crate) fn test_active_set(&self) -> usize {
        self.sets.active
    }

    /// The picking BVH's triangle count (offscreen tests).
    pub(crate) fn test_pickable(&self) -> usize {
        self.bvh.order.len()
    }

    /// Closes the import report and the autosave prompt (offscreen screenshots).
    pub(crate) fn test_close_dialogs(&mut self) {
        self.show_welcome = false;
        self.recovery = None;
        self.spp.report = None;
    }
}
