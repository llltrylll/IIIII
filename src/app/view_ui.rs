//! The 3D and 2D viewports: input, drawing of the panes and end-of-frame bookkeeping.
use super::*;

impl PainterApp {
    pub(super) fn view_input(&mut self, ui: &egui::Ui, kind: ViewKind, rect: Rect, resp: &egui::Response) {
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
                if resp.double_clicked() && mods.alt
                    && let Some(p) = hover {
                        let size = Vec2::new(self.views.view3d.size[0] as f32, self.views.view3d.size[1] as f32);
                        let (o, d) = self.cam.ray(local(p), size);
                        if let Some(hit) = self.bvh.raycast(o, d) {
                            self.cam.target = hit.pos;
                        }
                    }
            }
            ViewKind::V2d => {
                if resp.dragged_by(B::Middle) || (resp.dragged_by(B::Secondary) && !adjust_brush) || (resp.dragged_by(B::Primary) && mods.alt) {
                    self.cam2d.pan += Vec2::new(delta.x, delta.y) * ppp;
                }
                if resp.hovered() && scroll != 0.0
                    && let Some(p) = hover {
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
            } else if down && self.stroke.as_ref().is_some_and(|s| s.view == kind)
                && let Some((t, pos)) = tri {
                    self.poly_fill(&[t], Some(pos));
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
    pub(super) fn view_pane(&mut self, ui: &mut egui::Ui, kind: ViewKind) {
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
            let started = Instant::now();
            self.engine.composite();
            if std::mem::take(&mut self.time_composite) {
                // F5 measures the whole rebuild, GPU included.
                let cpu = started.elapsed();
                self.engine.wait_idle();
                let total = started.elapsed();
                self.status = format!(
                    "Recomposition (F5): {:.0} ms total ({:.0} ms preparing, {:.0} ms on the GPU), {} layers, {}².",
                    total.as_secs_f64() * 1000.0,
                    cpu.as_secs_f64() * 1000.0,
                    (total - cpu).as_secs_f64() * 1000.0,
                    self.engine.layers.len(),
                    self.engine.res
                );
            }
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
    pub(super) fn views_end_frame(&mut self, ctx: &egui::Context) {
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

    /// Scroll zoom per point of scrolling, with the user's speed and direction.
    pub(super) fn zoom_rate(&self) -> f32 {
        0.0015 * self.prefs.zoom_speed * if self.prefs.invert_zoom { -1.0 } else { 1.0 }
    }
}
