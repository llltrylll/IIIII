//! Camera lens and editable viewport lighting controls.
use super::*;

impl PainterApp {
    pub(super) fn lighting_panel(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Camera and scene lighting").strong().color(Color32::WHITE));
        ui.label(
            egui::RichText::new("Adjust the viewport camera and the lights used by Material preview.")
                .small()
                .color(theme::TEXT_DIM),
        );

        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            section(ui, "Camera");
            let mut fov_degrees = self.cam.fov_y.to_degrees();
            if slider(ui, "Field of view", &mut fov_degrees, 10.0..=120.0) {
                self.cam.fov_y = fov_degrees.to_radians();
            }
            ui.horizontal(|ui| {
                if ui.small_button("Reset FOV").clicked() {
                    self.cam.fov_y = self.prefs.fov_deg.clamp(10.0, 120.0).to_radians();
                }
                if ui.small_button("Frame model").clicked() {
                    self.frame_mesh_views();
                }
            });
            ui.label(
                egui::RichText::new("Higher values show more of the scene; lower values give a tighter, flatter view.")
                    .small()
                    .color(theme::TEXT_DIM),
            );

            section(ui, "Environment");
            slider(ui, "Intensity", &mut self.view.env_intensity, 0.0..=5.0);
            let mut rotation = self.view.env_rotation.to_degrees();
            if slider(ui, "Rotation", &mut rotation, -180.0..=180.0) {
                self.view.env_rotation = rotation.to_radians();
            }

            section(ui, "Directional key");
            labeled(ui, "Enabled", |ui| ui.checkbox(&mut self.view.key.on, ""));
            slider(ui, "Azimuth", &mut self.view.key.azimuth, -180.0..=180.0);
            slider(ui, "Elevation", &mut self.view.key.elevation, -89.0..=89.0);
            slider(ui, "Intensity", &mut self.view.key.intensity, 0.0..=8.0);
            labeled(ui, "Color", |ui| color_edit(ui, &mut self.view.key.color));

            section(ui, "Point lights");
            let existing = self.view.point_light_count.min(MAX_POINT_LIGHTS);
            ui.horizontal(|ui| {
                if ui.button("＋ Add point light").clicked() && existing < MAX_POINT_LIGHTS {
                    let (right, up) = self.cam.basis();
                    let offset = existing as f32 * 0.65;
                    let position = self.cam.target + right * (1.5 + offset) + up * 1.5;
                    self.view.point_lights[existing] = PointLight::new(position);
                    self.view.point_light_count = existing + 1;
                    self.selected_point_light = existing;
                }
                if ui.button("Remove").clicked() && existing > 0 {
                    let selected = self.selected_point_light.min(existing - 1);
                    for index in selected..(existing - 1) {
                        self.view.point_lights[index] = self.view.point_lights[index + 1];
                    }
                    self.view.point_lights[existing - 1] = PointLight::default();
                    self.view.point_light_count = existing - 1;
                    self.selected_point_light = selected.min(existing.saturating_sub(2));
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
                let light = &mut self.view.point_lights[selected];
                section(ui, &format!("Point light {}", selected + 1));
                labeled(ui, "Enabled", |ui| ui.checkbox(&mut light.on, ""));
                labeled(ui, "Color", |ui| color_edit(ui, &mut light.color));
                slider(ui, "Intensity", &mut light.intensity, 0.0..=200.0);
                slider(ui, "Range", &mut light.range, 0.1..=20.0);

                section(ui, "Position");
                slider(ui, "X", &mut light.position.x, -20.0..=20.0);
                slider(ui, "Y", &mut light.position.y, -20.0..=20.0);
                slider(ui, "Z", &mut light.position.z, -20.0..=20.0);
                if ui.small_button("Move in front of camera").clicked() {
                    let (right, up) = self.cam.basis();
                    light.position = self.cam.target + right * 1.5 + up * 1.5;
                }
            }

            ui.add_space(8.0);
            ui.label(
                egui::RichText::new(format!("{} / {} point lights. Local point lights currently do not cast shadows.", count, MAX_POINT_LIGHTS))
                    .small()
                    .color(theme::TEXT_DIM),
            );
        });
    }
}
