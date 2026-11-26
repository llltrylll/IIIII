//! Camera lens and editable viewport lighting controls.
use super::*;
use crate::viewport::{OrbitCam, PointLight, MAX_POINT_LIGHTS};
use glam::Vec3;

/// Keep auto-created lights near the model focus even when the camera is zoomed far out.
fn camera_light_position(cam: &OrbitCam, horizontal_offset: f32, vertical_offset: f32) -> Vec3 {
    let (right, up) = cam.basis();
    let forward = (cam.target - cam.eye()).normalize_or_zero();
    let in_front_distance = (cam.dist * 0.4).clamp(0.25, 3.0);
    cam.target - forward * in_front_distance + right * horizontal_offset + up * vertical_offset
}

impl PainterApp {
    pub(super) fn toggle_point_light_move(&mut self) {
        self.drag_point_light = None;
        self.point_light_pointer_capture = false;
        let count = self.view.point_light_count.min(MAX_POINT_LIGHTS);
        if count == 0 {
            self.move_point_light = false;
            self.status = "Add a point light before moving it.".into();
            return;
        }
        self.selected_point_light = self.selected_point_light.min(count - 1);
        self.move_point_light = !self.move_point_light;
        self.status = if self.move_point_light {
            format!("Move Point {}: click a surface in the 3D view. Press Esc to cancel.", self.selected_point_light + 1)
        } else {
            "Point-light move mode cancelled.".into()
        };
    }

    pub(super) fn lighting_panel(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Scene lighting controls").strong().color(Color32::WHITE));
        ui.label(
            egui::RichText::new("Position and tune viewport light sources. Camera FOV is on the top bar.")
                .small()
                .color(theme::TEXT_DIM),
        );

        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            section(ui, "Environment");
            slider(ui, "Intensity", &mut self.view.env_intensity, 0.0..=5.0);
            let mut rotation = self.view.env_rotation.to_degrees();
            if slider(ui, "Rotation", &mut rotation, -180.0..=180.0) {
                self.view.env_rotation = rotation.to_radians();
            }

            section(ui, "Directional key");
            ui.checkbox(
                &mut self.view.key.on,
                theme::with_icon(icons::SUN, "Directional light with shadows"),
            )
            .on_hover_text("A sun-like light with shadows. Its direction turns with the environment rotation.");
            slider(ui, "Azimuth", &mut self.view.key.azimuth, -180.0..=180.0);
            slider(ui, "Elevation", &mut self.view.key.elevation, -89.0..=89.0);
            slider(ui, "Intensity", &mut self.view.key.intensity, 0.0..=8.0);
            labeled(ui, "Color", |ui| color_edit(ui, &mut self.view.key.color));

            section(ui, "Point lights");
            ui.checkbox(&mut self.show_point_light_markers, "Show light markers in 3D view");
            let existing = self.view.point_light_count.min(MAX_POINT_LIGHTS);
            ui.horizontal(|ui| {
                if ui.button("＋ Add point light").clicked() && existing < MAX_POINT_LIGHTS {
                    let offset = existing as f32 * 0.65;
                    let position = camera_light_position(&self.cam, offset, 1.5);
                    self.view.point_lights[existing] = PointLight::new(position);
                    self.view.point_light_count = existing + 1;
                    self.selected_point_light = existing;
                }
                let move_label = if self.move_point_light { "Cancel placement" } else { "Place with click" };
                if ui.add_enabled(existing > 0, egui::Button::new(move_label)).clicked() {
                    self.toggle_point_light_move();
                }
                if ui.add_enabled(existing > 0, egui::Button::new("Remove")).clicked() {
                    let selected = self.selected_point_light.min(existing - 1);
                    self.drag_point_light = None;
                    self.point_light_pointer_capture = false;
                    for index in selected..(existing - 1) {
                        self.view.point_lights[index] = self.view.point_lights[index + 1];
                    }
                    self.view.point_lights[existing - 1] = PointLight::default();
                    self.view.point_light_count = existing - 1;
                    self.selected_point_light = selected.min(existing.saturating_sub(2));
                    self.move_point_light = false;
                }
            });
            let count = self.view.point_light_count.min(MAX_POINT_LIGHTS);
            if count == 0 {
                ui.label(egui::RichText::new("No point lights. Add one to create a local light source.").small().color(theme::TEXT_DIM));
            } else {
                self.selected_point_light = self.selected_point_light.min(count - 1);
                ui.horizontal_wrapped(|ui| {
                    for index in 0..count {
                        let label = format!("Point {}", index + 1);
                        if ui.selectable_label(self.selected_point_light == index, label).clicked() {
                            self.selected_point_light = index;
                        }
                    }
                });

                let selected = self.selected_point_light;
                section(ui, &format!("Point light {}", selected + 1));
                ui.horizontal(|ui| {
                    let move_button = egui::Button::new(if self.move_point_light { "Cancel placement" } else { "Place with click" })
                        .min_size(vec2(150.0, 26.0));
                    if ui.add(move_button).clicked() {
                        self.toggle_point_light_move();
                    }
                    if ui.button(if self.view.point_lights[selected].on { "Turn off" } else { "Turn on" }).clicked() {
                        self.view.point_lights[selected].on = !self.view.point_lights[selected].on;
                    }
                });
                ui.label(
                    egui::RichText::new("Drag the marker in the 3D view to move it. Right-click the marker to toggle it; Place with click snaps it to a surface.")
                        .small()
                        .color(theme::TEXT_DIM),
                );
                let light = &mut self.view.point_lights[selected];
                labeled(ui, "Enabled", |ui| ui.checkbox(&mut light.on, ""));
                labeled(ui, "Color", |ui| color_edit(ui, &mut light.color));
                slider(ui, "Intensity", &mut light.intensity, 0.0..=200.0);
                slider(ui, "Range", &mut light.range, 0.1..=1_000.0);

                section(ui, "Position");
                ui.add(egui::DragValue::new(&mut light.position.x).speed(0.1).range(-1_000_000.0..=1_000_000.0).prefix("X "));
                ui.add(egui::DragValue::new(&mut light.position.y).speed(0.1).range(-1_000_000.0..=1_000_000.0).prefix("Y "));
                ui.add(egui::DragValue::new(&mut light.position.z).speed(0.1).range(-1_000_000.0..=1_000_000.0).prefix("Z "));
            }

            ui.add_space(8.0);
            if self.move_point_light {
                ui.label(
                    egui::RichText::new(format!("Placement mode: click a surface in the 3D view to place Point {}. Esc cancels.", self.selected_point_light + 1))
                        .small()
                        .color(theme::ACCENT),
                );
            }
            ui.label(
                egui::RichText::new(format!("{} / {} point lights. Local point lights do not cast shadows.", count, MAX_POINT_LIGHTS))
                    .small()
                    .color(theme::TEXT_DIM),
            );
        });
    }
}


#[cfg(test)]
mod tests {
    use super::camera_light_position;
    use crate::viewport::OrbitCam;

    #[test]
    fn auto_placed_light_stays_near_focus_when_camera_is_far_away() {
        let mut cam = OrbitCam::default();
        cam.set_distance(500.0);
        let position = camera_light_position(&cam, 0.0, 0.0);
        assert!((position.distance(cam.target) - 3.0).abs() < 1e-4);
    }

    #[test]
    fn auto_placed_light_stays_far_enough_from_focus_for_close_camera() {
        let mut cam = OrbitCam::default();
        cam.set_distance(0.01);
        let position = camera_light_position(&cam, 0.0, 0.0);
        assert!((position.distance(cam.target) - 0.25).abs() < 1e-4);
    }

    #[test]
    fn auto_placed_light_tracks_normal_camera_distance() {
        let cam = OrbitCam::default();
        let position = camera_light_position(&cam, 0.0, 0.0);
        assert!((position.distance(cam.target) - cam.dist * 0.4).abs() < 1e-4);
    }
}
