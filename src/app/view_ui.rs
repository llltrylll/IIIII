//! The 3D and 2D viewports: input, drawing of the panes and end-of-frame bookkeeping.
use super::*;

/// Project a world-space point-light position into the 3D viewport's UI coordinates.
fn point_light_screen_position(cam: &OrbitCam, position: Vec3, rect: Rect) -> Option<egui::Pos2> {
    if rect.width() <= 1.0 || rect.height() <= 1.0 || !position.is_finite() {
        return None;
    }
    let clip = cam.view_proj(rect.width() / rect.height()) * position.extend(1.0);
    if !clip.w.is_finite() || clip.w <= 1e-5 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    if !ndc.is_finite()
        || ndc.z < 0.0
        || ndc.z > 1.0
        || ndc.x < -1.0
        || ndc.x > 1.0
        || ndc.y < -1.0
        || ndc.y > 1.0
    {
        return None;
    }
    Some(pos2(
        rect.left() + (ndc.x * 0.5 + 0.5) * rect.width(),
        rect.top() + (0.5 - ndc.y * 0.5) * rect.height(),
    ))
}

/// Points of the opacity arc, starting at the top and going clockwise.
/// `turn_segments` is how many straight pieces a full turn would use.
fn stroke_opacity_arc_points_dense(center: egui::Pos2, radius: f32, opacity: f32, turn_segments: f32) -> Vec<egui::Pos2> {
    let opacity = opacity.clamp(0.0, 1.0);
    if opacity <= 0.0 || !radius.is_finite() || radius <= 0.0 {
        return Vec::new();
    }
    let segments = ((turn_segments * opacity).ceil() as usize).max(1);
    let sweep = std::f32::consts::TAU * opacity;
    (0..=segments)
        .map(|step| {
            let angle = -std::f32::consts::FRAC_PI_2 + sweep * step as f32 / segments as f32;
            pos2(center.x + radius * angle.cos(), center.y + radius * angle.sin())
        })
        .collect()
}

/// The coarse arc (48 pieces per turn) the original tests describe.
#[cfg(test)]
fn stroke_opacity_arc_points(center: egui::Pos2, radius: f32, opacity: f32) -> Vec<egui::Pos2> {
    stroke_opacity_arc_points_dense(center, radius, opacity, 48.0)
}

/// Pieces per turn for the drawn arc: about one piece per 1.5 px of length so
/// large cursors stay round, never fewer than the original 48.
fn stroke_opacity_turn_segments(radius: f32) -> f32 {
    (std::f32::consts::TAU * radius / 1.5).max(48.0)
}

/// The white endpoint shows the exact end of the blue opacity sweep.
fn stroke_opacity_marker_position(center: egui::Pos2, radius: f32, opacity: f32) -> Option<egui::Pos2> {
    if !center.is_finite() || !radius.is_finite() || radius <= 0.0 || !opacity.is_finite() {
        return None;
    }
    let opacity = opacity.clamp(0.0, 1.0);
    if opacity <= 0.0 {
        return None;
    }
    let angle = -std::f32::consts::FRAC_PI_2 + std::f32::consts::TAU * opacity;
    Some(pos2(center.x + radius * angle.cos(), center.y + radius * angle.sin()))
}

/// Short tick marks on the opacity track at 0, 25, 50 and 75 %, as
/// (inner, outer) point pairs, so the arc can be read against a scale.
fn stroke_opacity_tick_segments(center: egui::Pos2, radius: f32) -> Vec<(egui::Pos2, egui::Pos2)> {
    if !radius.is_finite() || radius <= 0.0 {
        return Vec::new();
    }
    (0..4)
        .map(|quarter| {
            let angle = -std::f32::consts::FRAC_PI_2 + std::f32::consts::FRAC_PI_2 * quarter as f32;
            let (cos, sin) = (angle.cos(), angle.sin());
            (
                pos2(center.x + (radius - 1.5) * cos, center.y + (radius - 1.5) * sin),
                pos2(center.x + (radius + 2.5) * cos, center.y + (radius + 2.5) * sin),
            )
        })
        .collect()
}

/// Full opacity is self-evident from the closed ring, so only partial values
/// need a nearby percentage label.
fn stroke_opacity_label(opacity: f32) -> Option<String> {
    if !opacity.is_finite() {
        return None;
    }
    let percent = (opacity.clamp(0.0, 1.0) * 100.0).round() as u32;
    (percent < 100).then(|| format!("{percent}%"))
}

/// Keep the value just outside the lower-right quadrant, like the brush
/// preview cards, rather than attaching it to the moving end of the arc.
fn stroke_opacity_label_position(center: egui::Pos2, radius: f32) -> egui::Pos2 {
    let offset = radius * 0.72 + 4.0;
    pos2(center.x + offset, center.y + offset)
}

impl PainterApp {
    fn finish_walk_navigation(&mut self) {
        // The first-person camera already maintains a virtual focus point on
        // its look ray. Re-anchor to a real surface only when the hit is
        // plausible; never fall back to the mesh bounds center/origin.
        let eye = self.cam.eye();
        let forward = (self.cam.target - eye).normalize_or_zero();
        if forward.length_squared() < 0.5 {
            return;
        }
        if let Some(hit) = self.bvh.raycast(eye, forward) {
            let distance = eye.distance(hit.pos);
            if distance.is_finite() && distance >= crate::viewport::CAMERA_MIN_DISTANCE {
                self.cam.focus_on(hit.pos);
            }
        }
    }

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

        // Moving a light is a one-click placement mode, not a paint gesture.
        // Use the mesh hit when possible; otherwise place it on the camera's
        // focal plane so lights can also be moved into empty space.
        if self.move_point_light && kind == ViewKind::V3d {
            if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                self.move_point_light = false;
                self.status = "Point-light move mode cancelled.".into();
                ui.ctx().request_repaint();
                return;
            }
            if pressed && resp.hovered() && let Some(pointer) = hover {
                let size = Vec2::new(rect.width() * ppp, rect.height() * ppp);
                let (origin, direction) = self.cam.ray(local(pointer), size);
                let position = if let Some(hit) = self.bvh.raycast(origin, direction) {
                    Some(hit.pos - direction * 0.25)
                } else {
                    let forward = (self.cam.target - origin).normalize_or_zero();
                    let denominator = direction.dot(forward);
                    if denominator.abs() > 1e-5 {
                        let distance = (self.cam.target - origin).dot(forward) / denominator;
                        (distance > 0.0).then_some(origin + direction * distance)
                    } else {
                        None
                    }
                };
                let count = self.view.point_light_count.min(crate::viewport::MAX_POINT_LIGHTS);
                if count > 0 && self.selected_point_light < count && let Some(position) = position.filter(|p| p.is_finite()) {
                    self.view.point_lights[self.selected_point_light].position = position;
                    self.move_point_light = false;
                    self.status = format!(
                        "Point {} moved to X {:.2}, Y {:.2}, Z {:.2}.",
                        self.selected_point_light + 1, position.x, position.y, position.z
                    );
                }
                ui.ctx().request_repaint();
            }
            ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
            return;
        }

        // Clicking a marker selects its light and is deliberately intercepted
        // before the regular paint input path.
        if kind == ViewKind::V3d && self.show_point_light_markers && pressed && resp.hovered()
            && let Some(pointer) = hover
        {
            let count = self.view.point_light_count.min(crate::viewport::MAX_POINT_LIGHTS);
            for index in 0..count {
                if point_light_screen_position(&self.cam, self.view.point_lights[index].position, rect)
                    .is_some_and(|marker| marker.distance(pointer) <= 12.0)
                {
                    self.selected_point_light = index;
                    self.status = format!("Point light {} selected.", index + 1);
                    ui.ctx().request_repaint();
                    return;
                }
            }
        }

        // Ctrl+' toggles first-person navigation while the pointer is over
        // the 3D viewport. While active, the pointer is locked/hidden and
        // keyboard motion is routed exclusively to the camera.
        let toggle_walk = kind == ViewKind::V3d
            && (self.walk_navigation || resp.hovered())
            && !ui.ctx().egui_wants_keyboard_input()
            && ui.input_mut(|i| {
                i.consume_shortcut(&egui::KeyboardShortcut::new(
                    egui::Modifiers::CTRL,
                    egui::Key::Quote,
                ))
            });
        if toggle_walk {
            self.walk_navigation = !self.walk_navigation;
            if self.walk_navigation {
                self.walk_skip_mouse_delta = true;
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::CursorGrab(
                    egui::viewport::CursorGrab::Locked,
                ));
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::CursorVisible(false));
                self.status = "First-person navigation: WASD move, mouse look, E/Q vertical, Shift faster, Esc exit.".into();
            } else {
                self.walk_skip_mouse_delta = false;
                self.finish_walk_navigation();
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::CursorGrab(
                    egui::viewport::CursorGrab::None,
                ));
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::CursorVisible(true));
                self.status = "First-person navigation ended.".into();
            }
            ui.ctx().request_repaint();
            return;
        }

        if self.walk_navigation && kind == ViewKind::V3d {
            let (look_delta, dt, forward, backward, left, right, up, down, fast, slow, exit, scroll) =
                ui.input(|i| (
                    // Locked cursors do not change their screen position. Use
                    // egui's raw mouse motion for first-person look, falling
                    // back to pointer delta on integrations without raw motion.
                    i.pointer.motion().unwrap_or_else(|| i.pointer.delta()),
                    i.stable_dt.clamp(0.0, 0.05),
                    i.key_down(egui::Key::W),
                    i.key_down(egui::Key::S),
                    i.key_down(egui::Key::A),
                    i.key_down(egui::Key::D),
                    i.key_down(egui::Key::E),
                    i.key_down(egui::Key::Q),
                    i.modifiers.shift,
                    i.modifiers.ctrl,
                    i.key_pressed(egui::Key::Escape)
                        || i.events.iter().any(|event| matches!(
                            event,
                            egui::Event::PointerButton { pressed: true, .. }
                        )),
                    i.smooth_scroll_delta.y,
                ));
            if exit {
                self.walk_navigation = false;
                self.walk_skip_mouse_delta = false;
                self.finish_walk_navigation();
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::CursorGrab(
                    egui::viewport::CursorGrab::None,
                ));
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::CursorVisible(true));
                self.status = "First-person navigation ended.".into();
            } else {
                let look_delta = if std::mem::take(&mut self.walk_skip_mouse_delta) {
                    Vec2::ZERO
                } else {
                    Vec2::new(look_delta.x, look_delta.y)
                };
                self.cam.walk_look(look_delta, 0.0025);
                if scroll != 0.0 {
                    self.walk_speed = (self.walk_speed * (scroll * 0.004).exp()).clamp(0.25, 24.0);
                    self.status = format!("First-person speed: {:.1}", self.walk_speed);
                }
                let speed = self.walk_speed * if fast { 4.0 } else if slow { 0.25 } else { 1.0 };
                self.cam.walk_move(
                    forward as i32 as f32 - backward as i32 as f32,
                    right as i32 as f32 - left as i32 as f32,
                    up as i32 as f32 - down as i32 as f32,
                    dt,
                    speed,
                );
            }
            ui.ctx().request_repaint();
            return;
        }

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
                    self.cam.set_distance(self.cam.dist * (delta.y * 0.01).exp());
                }
                if resp.dragged_by(B::Secondary) && mods.shift && !adjust_brush {
                    self.view.env_rotation += delta.x * 0.01;
                }
                if resp.hovered() && scroll != 0.0 {
                    self.cam.set_distance(self.cam.dist * (-scroll * self.zoom_rate()).exp());
                }
                if resp.double_clicked() && mods.alt
                    && let Some(p) = hover {
                        let size = Vec2::new(self.views.view3d.size[0] as f32, self.views.view3d.size[1] as f32);
                        let (o, d) = self.cam.ray(local(p), size);
                        if let Some(hit) = self.bvh.raycast(o, d) {
                            self.cam.focus_on(hit.pos);
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
                    self.poly_fill(&seeds, at, kind);
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
                        self.poly_fill(&[t], Some(pos), kind);
                    }
                }
            } else if down && self.stroke.as_ref().is_some_and(|s| s.view == kind)
                && let Some((t, pos)) = tri {
                    self.poly_fill(&[t], Some(pos), kind);
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
            } else if self.stroke.as_ref().is_some_and(|s| s.view == kind) {
                // Process the release-frame cursor position too. Otherwise,
                // the last movement delivered with mouse-up is discarded
                // immediately before end_stroke() captures the final state.
                self.continue_stroke(local(p));
            }
        }
    }

    /// Whether the current 3D preview samples world-space position/normal maps.
    fn parked_surface_maps_needed(&self) -> bool {
        matches!(self.view.mode, ViewMode::WorldNormal | ViewMode::Position)
            || (self.view.mode == ViewMode::Material && self.views.has_custom_shader())
    }

    /// The auxiliary depth pass exists for screen-projected 3D brush strokes.
    /// Rebuilding it during ordinary camera orbit was rendering the entire mesh
    /// a second time on every navigation frame, despite no brush using the result.
    fn screen_paint_depth_needed(&self) -> bool {
        self.stroke.as_ref().is_some_and(|stroke| stroke.view == ViewKind::V3d)
            && self.brush.projection == BrushProjection::Screen
            && !self.walk_navigation
    }

    /// Reframes both viewports on the normalized mesh bounds. The camera is
    /// fitted to the narrower dock axis so a portrait 3D pane does not crop it.
    pub(super) fn frame_mesh_views(&mut self) {
        let size = self.views.view3d.size;
        let aspect = if size[1] > 0 { size[0] as f32 / size[1] as f32 } else { 1.0 };
        self.frame_mesh_views_with_aspect(aspect);
    }

    fn frame_mesh_views_with_aspect(&mut self, aspect: f32) {
        if self.mesh.vertices.is_empty() {
            return;
        }
        self.cam.frame_bounds(self.mesh.bounds_min, self.mesh.bounds_max, aspect);
        self.cam2d.pan = Vec2::ZERO;
        self.cam2d.zoom = 0.0;
        self.view_framed_mesh_id = Some(self.gpu_mesh.mesh_id);
        self.views.invalidate_view_cache();
        self.status = "3D and UV views framed to the mesh.".into();
    }

    /// One viewport (a dock tab): input, the GPU render at the tab's size,
    /// overlays and the brush cursor.
    pub(super) fn view_pane(&mut self, ui: &mut egui::Ui, kind: ViewKind) {
        let rect = ui.available_rect_before_wrap();
        if rect.width() < 4.0 || rect.height() < 4.0 {
            return;
        }
        // Camera and UV pan can outlive a project reload. Fit the first frame
        // of each new GPU mesh to its actual bounds rather than showing empty space.
        if kind == ViewKind::V3d
            && !self.mesh.vertices.is_empty()
            && self.view_framed_mesh_id != Some(self.gpu_mesh.mesh_id)
        {
            self.frame_mesh_views_with_aspect(rect.width() / rect.height().max(1.0));
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
            if self.meter.on {
                let counters = (self.engine.full_recomps, self.engine.thumb_renders());
                self.meter.composited(self.stroke.is_some(), started.elapsed().as_secs_f64() * 1000.0, counters, self.engine_epoch, self.sets.active);
            }
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
        let size = [(rect.width() * self.ppp).round() as u32, (rect.height() * self.ppp).round() as u32];
        let mut rt_view = None;
        match kind {
            ViewKind::V3d => {
                self.views.set_looks = self.set_looks(ui.ctx());
                let needs_surface_maps = self.parked_surface_maps_needed();
                self.set_parked_surface_maps_resident(needs_surface_maps);
                let engines = texture_sets::engines(&self.engine, &self.sets, self.mesh.sets.len());
                {
                    let _s = crate::startup_trace::span("app viewport: 3D render call");
                    self.views.render_3d_with_object_visibility_and_paint_depth(
                        &self.rs, size, &engines, &self.env, &self.gpu_mesh, self.mesh.as_ref(),
                        &self.cam, &self.view, Some(&self.overlays), &self.hidden_objects,
                        self.screen_paint_depth_needed(),
                    );
                }
                self.drew_3d = true;
                // Path tracing replaces the 3D view while it's idle. Painting and
                // the map display modes use the raster view, which reacts instantly.
                if self.rt.enabled && self.stroke.is_none() && self.view.mode == ViewMode::Material && self.preview_update_at.is_none() {
                    if self.raytracer.is_none() {
                        let _s = crate::startup_trace::span("raytraced preview: lazy RayTracer::new + set_mesh");
                        let mut raytracer = RayTracer::new(&self.rs);
                        raytracer.set_mesh(&self.rs.device, &self.mesh, &self.bvh);
                        self.raytracer = Some(raytracer);
                    }
                    let engines = texture_sets::engines(&self.engine, &self.sets, self.mesh.sets.len());
                    let raytracer = self.raytracer.as_mut().expect("ray tracer was initialized above");
                    if raytracer.render(&self.rs, size, &engines, &self.env, &self.cam, &self.view, &self.rt) {
                        ui.ctx().request_repaint();
                    }
                    rt_view = raytracer.egui_id;
                }
            }
            ViewKind::V2d => {
                if self.cam2d.zoom <= 0.0 {
                    self.cam2d.zoom = size[0].min(size[1]) as f32 * 0.9;
                    self.cam2d.pan = Vec2::ZERO;
                }
                {
                    let _s = crate::startup_trace::span("app viewport: 2D render call");
                    self.views.render_2d(&self.rs, size, &self.engine, &self.env, &self.gpu_mesh, self.sets.active, &self.cam2d, &self.view);
                }
            }
        }

        let painter = ui.painter_at(rect);
        let uv = Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));
        let id = match kind {
            ViewKind::V3d => rt_view.unwrap_or(self.views.view3d.egui_id),
            ViewKind::V2d => self.views.view2d.egui_id,
        };
        painter.image(id, rect, uv, Color32::WHITE);
        if kind == ViewKind::V3d && self.show_point_light_markers {
            let count = self.view.point_light_count.min(crate::viewport::MAX_POINT_LIGHTS);
            for index in 0..count {
                let light = self.view.point_lights[index];
                let Some(marker) = point_light_screen_position(&self.cam, light.position, rect) else { continue };
                let tint = Color32::from_rgba_unmultiplied(
                    (light.color[0].clamp(0.0, 1.0) * 255.0) as u8,
                    (light.color[1].clamp(0.0, 1.0) * 255.0) as u8,
                    (light.color[2].clamp(0.0, 1.0) * 255.0) as u8,
                    if light.on { 255 } else { 105 },
                );
                let selected = index == self.selected_point_light;
                let outline = if selected { theme::ACCENT } else { tint };
                painter.circle_filled(marker, 8.5, Color32::from_black_alpha(220));
                painter.circle_stroke(marker, 8.0, Stroke::new(if selected { 2.0 } else { 1.5 }, outline));
                painter.circle_filled(marker, 2.8, tint);
                for (dx, dy) in [(0.0, -12.0), (12.0, 0.0), (0.0, 12.0), (-12.0, 0.0)] {
                    let start = marker + vec2(dx * 0.62, dy * 0.62);
                    let end = marker + vec2(dx, dy);
                    painter.line_segment([start, end], Stroke::new(1.4, outline));
                }
                let label_rect = Rect::from_min_size(marker + vec2(11.0, -9.0), vec2(26.0, 17.0));
                painter.rect_filled(label_rect, 3.0, Color32::from_black_alpha(205));
                painter.text(
                    label_rect.center(),
                    egui::Align2::CENTER_CENTER,
                    format!("L{}", index + 1),
                    egui::FontId::proportional(10.0),
                    if light.on { Color32::WHITE } else { Color32::from_gray(165) },
                );
            }
        }
        if kind == ViewKind::V3d && self.move_point_light {
            let prompt = format!("Move Point {} · click the model to place · Esc cancels", self.selected_point_light + 1);
            let prompt_rect = Rect::from_min_size(rect.left_bottom() + vec2(8.0, -28.0), vec2(290.0, 20.0));
            painter.rect_filled(prompt_rect, 3.0, Color32::from_black_alpha(205));
            painter.text(prompt_rect.center(), egui::Align2::CENTER_CENTER, prompt, egui::FontId::proportional(11.0), Color32::WHITE);
        }
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
                let raytracer = self.raytracer.as_ref().expect("ray-traced view requires an initialized tracer");
                let state = if raytracer.converged(&self.rt) { "done" } else { "refining" };
                format!("Ray traced · {}/{} spp ({state})", raytracer.samples, self.rt.max_samples)
            }
            ViewKind::V3d => self.mode_name(self.view.mode),
            ViewKind::V2d => self.mode_name(self.view.uv_mode),
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
                    // Two faint outlines keep the hardness ring legible on
                    // both dark and light materials without filling the brush.
                    painter.circle_stroke(p, inner + 0.5, Stroke::new(1.0, Color32::from_black_alpha(85)));
                    painter.circle_stroke(p, inner, Stroke::new(1.0, Color32::from_white_alpha(125)));
                }

                // Opacity indicator: a quiet full track, blue progress arc,
                // white endpoint and a small percentage on partial opacity.
                let opacity_radius = r + 5.0;
                painter.circle_stroke(
                    p,
                    opacity_radius + 0.5,
                    Stroke::new(1.0, Color32::from_black_alpha(95)),
                );
                painter.circle_stroke(
                    p,
                    opacity_radius,
                    Stroke::new(1.0, Color32::from_white_alpha(85)),
                );
                for (inner, outer) in stroke_opacity_tick_segments(p, opacity_radius) {
                    painter.line_segment([inner, outer], Stroke::new(1.0, Color32::from_white_alpha(120)));
                }
                let opacity_arc = stroke_opacity_arc_points_dense(
                    p,
                    opacity_radius,
                    self.brush.opacity,
                    stroke_opacity_turn_segments(opacity_radius),
                );
                if opacity_arc.len() >= 2 {
                    painter.add(egui::Shape::line(opacity_arc, Stroke::new(2.2, theme::ACCENT)));
                }
                if let Some(marker) = stroke_opacity_marker_position(p, opacity_radius, self.brush.opacity) {
                    painter.circle_filled(marker, 3.0, Color32::from_black_alpha(220));
                    painter.circle_filled(marker, 1.8, Color32::WHITE);
                }
                if let Some(label) = stroke_opacity_label(self.brush.opacity) {
                    let label_pos = stroke_opacity_label_position(p, opacity_radius);
                    let font = egui::FontId::proportional(9.5);
                    // A dark outline gives the light label contrast on both
                    // dark and bright viewport surfaces.
                    for offset in [vec2(-0.7, 0.0), vec2(0.7, 0.0), vec2(0.0, -0.7), vec2(0.0, 0.7)] {
                        painter.text(
                            label_pos + offset,
                            egui::Align2::LEFT_CENTER,
                            &label,
                            font.clone(),
                            Color32::from_black_alpha(220),
                        );
                    }
                    painter.text(
                        label_pos,
                        egui::Align2::LEFT_CENTER,
                        label,
                        font,
                        Color32::from_white_alpha(235),
                    );
                }

                ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
                if let Some(s) = self.stroke.as_ref().filter(|s| s.view == kind && self.brush.stabilize > 0.0) {
                    let lazy = rect.min + vec2(s.lazy.x, s.lazy.y) / self.ppp;
                    painter.line_segment([lazy, p], Stroke::new(1.0, Color32::from_white_alpha(120)));
                    painter.circle_filled(lazy, 2.5, Color32::WHITE);
                }
            }
        }

        if kind == ViewKind::V3d && self.color_hud_open {
            let width = 436.0;
            let left = (rect.right() - width - 8.0).max(rect.left() + 8.0);
            egui::Area::new(egui::Id::unique("color_hud_overlay"))
                .order(egui::Order::Foreground)
                .fixed_pos(pos2(left, rect.top() + 8.0))
                .show(ui.ctx(), |ui| {
                    egui::Frame::popup(ui.style()).show(ui, |ui| {
                        ui.set_width(420.0);
                        self.color_hud(ui);
                    });
                });
        }
    }

    /// After every view drew: strokes end when the button is released, and
    /// the 3D depth the brush reads stays valid when no 3D view is showing.
    pub(super) fn views_end_frame(&mut self, ctx: &egui::Context) {
        if let Some((forced_composite, stamp_ms, apply_ms, composite_ms, composite_total_ms, composite_clear_ms, composite_body_ms, pass_times)) =
            self.engine.poll_gpu_stroke_times()
        {
            if forced_composite {
                let total_ms = composite_total_ms.unwrap_or(0.0);
                let clear_ms = composite_clear_ms.unwrap_or(0.0);
                let body_ms = composite_body_ms.unwrap_or(0.0);
                self.last_composite_gpu_profile = Some(CompositeGpuProfile {
                    wall_status: self.status.clone(),
                    total_ms,
                    clear_ms,
                    body_ms,
                    layer_passes: self.engine.last_composite_layer_passes,
                    folder_passes: self.engine.last_composite_folder_passes,
                    mask_cache_hits: self.engine.last_mask_cache_hits,
                    mask_cache_misses: self.engine.last_mask_cache_misses,
                    mask_gpu_passes: self.engine.last_mask_gpu_passes,
                    stages: pass_times,
                });
                if let Some(profile) = self.last_composite_gpu_profile.as_ref() {
                    let slowest = profile.stages.iter().max_by(|a, b| a.1.total_cmp(&b.1));
                    self.status = if let Some((label, ms)) = slowest {
                        format!("{} | Composite GPU {:.2} ms · clear {:.2} ms · body {:.2} ms · {} stages · slowest {} {:.2} ms", profile.wall_status, profile.total_ms, profile.clear_ms, profile.body_ms, profile.stages.len(), label, ms)
                    } else {
                        format!("{} | Composite GPU timestamps collected · {:.2} ms · no per-pass spans", profile.wall_status, profile.total_ms)
                    };
                }
            } else if self.meter.on {
                self.meter.gpu_stamp.add(stamp_ms);
                self.meter.gpu_apply.add(apply_ms);
                self.meter.gpu_composite.add(composite_ms);
                if let Some(ms) = composite_total_ms {
                    self.meter.gpu_composite_total.add(ms);
                }
                if let Some(ms) = composite_clear_ms {
                    self.meter.gpu_composite_clear.add(ms);
                }
                if let Some(ms) = composite_body_ms {
                    self.meter.gpu_composite_body.add(ms);
                }
                if let Some((label, ms)) = pass_times.into_iter().max_by(|a, b| a.1.total_cmp(&b.1)) {
                    if self.meter.gpu_composite_slowest_label != label {
                        self.meter.gpu_composite_slowest = Default::default();
                        self.meter.gpu_composite_slowest_label = label;
                    }
                    self.meter.gpu_composite_slowest.add(ms);
                }
            }
        }
        if self.engine.gpu_stroke_profile_pending() {
            ctx.request_repaint();
        }
        if self.stroke.is_some() && !ctx.input(|i| i.pointer.primary_down()) {
            self.end_stroke();
        }
        self.update_poly_highlight();
        if !self.drew_3d {
            let size = self.views.view3d.size;
            self.views.set_looks = self.set_looks(ctx);
            let needs_surface_maps = self.parked_surface_maps_needed();
            self.set_parked_surface_maps_resident(needs_surface_maps);
            let engines = texture_sets::engines(&self.engine, &self.sets, self.mesh.sets.len());
            {
                let _s = crate::startup_trace::span("app viewport: fallback 3D render call");
                self.views.render_3d_with_object_visibility_and_paint_depth(
                    &self.rs, size, &engines, &self.env, &self.gpu_mesh, self.mesh.as_ref(),
                    &self.cam, &self.view, Some(&self.overlays), &self.hidden_objects,
                    self.screen_paint_depth_needed(),
                );
            }
        }
        // Parked Texture Sets can need a full-resolution composite after a
        // switch or property edit. Do not let that background work compete
        // with a live brush stroke, camera navigation, or an active pointer
        // drag; the visible Engine is composited synchronously when selected.
        let pointer_busy = ctx.input(|i| i.pointer.any_down());
        if self.stroke.is_none() && !self.walk_navigation && !pointer_busy {
            self.composite_parked_one(ctx);
        }
        self.drew_3d = false;
    }

    /// Scroll zoom per point of scrolling, with the user's speed and direction.
    pub(super) fn zoom_rate(&self) -> f32 {
        0.0015 * self.prefs.zoom_speed * if self.prefs.invert_zoom { -1.0 } else { 1.0 }
    }
}

#[cfg(test)]
mod cursor_opacity_tests {
    use super::{
        stroke_opacity_arc_points, stroke_opacity_arc_points_dense, stroke_opacity_label,
        stroke_opacity_marker_position, stroke_opacity_turn_segments,
    };
    use eframe::egui::pos2;

    #[test]
    fn zero_opacity_has_no_progress_arc() {
        assert!(stroke_opacity_arc_points(pos2(10.0, 20.0), 12.0, 0.0).is_empty());
    }

    #[test]
    fn half_opacity_draws_half_a_ring() {
        let points = stroke_opacity_arc_points(pos2(0.0, 0.0), 10.0, 0.5);
        assert_eq!(points.len(), 25);
        assert!(points.first().unwrap().y < 0.0);
        assert!(points.last().unwrap().y > 0.0);
    }

    #[test]
    fn full_opacity_draws_a_complete_ring() {
        let points = stroke_opacity_arc_points(pos2(0.0, 0.0), 10.0, 1.0);
        assert_eq!(points.len(), 49);
        assert!(points.first().unwrap().distance(*points.last().unwrap()) < 1e-4);
    }

    #[test]
    fn opacity_is_clamped_to_supported_range() {
        let over = stroke_opacity_arc_points(pos2(0.0, 0.0), 10.0, 2.0);
        let full = stroke_opacity_arc_points(pos2(0.0, 0.0), 10.0, 1.0);
        assert_eq!(over, full);
        assert!(stroke_opacity_arc_points(pos2(0.0, 0.0), 10.0, -1.0).is_empty());
    }

    #[test]
    fn large_cursor_arc_has_short_pieces() {
        let radius = 80.0;
        let points = stroke_opacity_arc_points_dense(pos2(0.0, 0.0), radius, 1.0, stroke_opacity_turn_segments(radius));
        let longest = points.windows(2).map(|pair| pair[0].distance(pair[1])).fold(0.0_f32, f32::max);
        assert!(longest <= 1.6, "pieces of {longest} px would look polygonal");
        let coarse = stroke_opacity_arc_points(pos2(0.0, 0.0), radius, 1.0);
        assert!(points.len() > coarse.len());
    }

    #[test]
    fn small_cursor_keeps_the_original_piece_count() {
        assert_eq!(stroke_opacity_turn_segments(5.0), 48.0);
        let points = stroke_opacity_arc_points_dense(pos2(0.0, 0.0), 5.0, 0.5, stroke_opacity_turn_segments(5.0));
        assert_eq!(points.len(), stroke_opacity_arc_points(pos2(0.0, 0.0), 5.0, 0.5).len());
    }

    #[test]
    fn opacity_marker_tracks_the_arc_endpoint() {
        let center = pos2(50.0, 60.0);
        let quarter = stroke_opacity_marker_position(center, 20.0, 0.25).unwrap();
        assert!((quarter.x - 70.0).abs() < 1e-4);
        assert!((quarter.y - 60.0).abs() < 1e-4);
        let full = stroke_opacity_marker_position(center, 20.0, 1.0).unwrap();
        assert!((full.x - center.x).abs() < 1e-4);
        assert!((full.y - 40.0).abs() < 1e-4);
        assert!(stroke_opacity_marker_position(center, 20.0, 0.0).is_none());
    }

    #[test]
    fn percentage_label_is_hidden_only_for_a_full_ring() {
        assert_eq!(stroke_opacity_label(0.25).as_deref(), Some("25%"));
        assert_eq!(stroke_opacity_label(0.65).as_deref(), Some("65%"));
        assert_eq!(stroke_opacity_label(1.0), None);
        assert_eq!(stroke_opacity_label(1.5), None);
    }

    #[test]
    fn ticks_mark_the_four_quarters_outside_the_inner_edge() {
        let ticks = super::stroke_opacity_tick_segments(pos2(0.0, 0.0), 10.0);
        assert_eq!(ticks.len(), 4);
        assert!(ticks[0].1.y < -10.0 && ticks[0].1.x.abs() < 1e-4);
        assert!(ticks[1].1.x > 10.0);
        assert!(ticks[2].1.y > 10.0);
        assert!(ticks[3].1.x < -10.0);
        assert!(super::stroke_opacity_tick_segments(pos2(0.0, 0.0), 0.0).is_empty());
    }
}
