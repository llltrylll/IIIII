//! Project-wide channel and object visibility controls.
use super::*;

const CHANNEL_LABELS: [&str; crate::engine::CHANNELS] =
    ["Base Color", "Roughness", "Metallic", "Height", "Opacity", "Emission", "Normal", "Specular"];
const CHANNEL_ORDER: [usize; crate::engine::CHANNELS] = [0, 6, 1, 2, HEIGHT_CHANNEL, 4, 5, 7];
const HEIGHT_CHANNEL: usize = crate::engine::HEIGHT;

fn apply_project_channel_mask(fill: &mut FillParams, active: &[bool; crate::engine::CHANNELS]) {
    for ch in 0..crate::engine::CHANNELS {
        if !active[ch] {
            fill.mat.enabled[ch] = false;
        }
    }
}

/// Intersects a brush's own channel choices with the channels this project permits.
pub(super) fn effective_channel_mask(
    brush_enabled: [bool; crate::engine::CHANNELS],
    active: &[bool; crate::engine::CHANNELS],
) -> [bool; crate::engine::CHANNELS] {
    std::array::from_fn(|ch| brush_enabled[ch] && active[ch])
}

/// Removes disabled-channel data from layer blocks before they enter the project.
pub(super) fn filter_layer_channels(
    layers: &mut [Layer],
    active: &[bool; crate::engine::CHANNELS],
) {
    for layer in layers {
        for ch in 0..crate::engine::CHANNELS {
            if active[ch] {
                continue;
            }
            match &mut layer.content {
                Content::Paint(channels) => channels[ch] = None,
                Content::Fill(fill) => fill.mat.enabled[ch] = false,
                Content::Folder(_) => {}
            }
            layer.images[ch] = None;
        }
    }
}

#[cfg(test)]
mod channel_mask_tests {
    use super::apply_project_channel_mask;
    use crate::engine::{FillParams, CHANNELS};
    use super::HEIGHT_CHANNEL;

    #[test]
    fn disabled_height_is_removed_from_a_shelf_fill_material() {
        let mut fill = FillParams::default();
        fill.mat.enabled = [true; CHANNELS];
        let mut active = [true; CHANNELS];
        active[HEIGHT_CHANNEL] = false;

        apply_project_channel_mask(&mut fill, &active);

        assert!(!fill.mat.enabled[HEIGHT_CHANNEL]);
        for ch in 0..CHANNELS {
            if ch != HEIGHT_CHANNEL {
                assert!(fill.mat.enabled[ch], "active channel {ch} was unexpectedly removed");
            }
        }
    }

    #[test]
    fn project_mask_blocks_disabled_channels_without_changing_brush_preferences() {
        use super::effective_channel_mask;

        let brush = [true; CHANNELS];
        let active = [true, true, true, false, true, true, true, true];
        let filtered = effective_channel_mask(brush, &active);

        assert!(!filtered[crate::engine::HEIGHT]);
        assert!(filtered[0]);
        assert!(brush[crate::engine::HEIGHT], "filtering must not mutate brush preferences");
    }

    #[test]
    fn enabled_channels_keep_their_material_flags() {
        let mut fill = FillParams::default();
        fill.mat.enabled = [false; CHANNELS];
        fill.mat.enabled[HEIGHT_CHANNEL] = true;
        apply_project_channel_mask(&mut fill, &[true; CHANNELS]);
        assert!(fill.mat.enabled[HEIGHT_CHANNEL]);
    }
}

impl PainterApp {
    /// Keeps a material preset from introducing channels excluded by the project.
    pub(super) fn filter_fill_channels(&self, fill: &mut FillParams) {
        apply_project_channel_mask(fill, &self.active_channels);
    }

    pub(super) fn project_settings_panel(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            ui.heading("Project Settings");
            ui.add_space(8.0);

            section(ui, "Channels");
            ui.label(
                egui::RichText::new("Only active channels are accepted from shelf materials and stored in the project file.")
                    .small()
                    .color(theme::TEXT_DIM),
            );
            let mut remove_channel = None;
            ui.menu_button(theme::with_icon(icons::PLUS, "Add Channel"), |ui| {
                let mut added = None;
                for ch in CHANNEL_ORDER {
                    if !self.active_channels[ch] && ui.button(CHANNEL_LABELS[ch]).clicked() {
                        added = Some(ch);
                        ui.close();
                    }
                }
                if let Some(ch) = added {
                    self.active_channels[ch] = true;
                    self.project_settings_serial = self.project_settings_serial.wrapping_add(1);
                    self.status = format!("Added {} channel. It starts empty; shelf materials can now provide it.", CHANNEL_LABELS[ch]);
                }
            });

            let active_count = self.active_channels.iter().filter(|active| **active).count();
            if active_count == 0 {
                ui.colored_label(Color32::from_rgb(240, 170, 80), "No channels are active. Add at least one before painting.");
            }
            for ch in CHANNEL_ORDER {
                if !self.active_channels[ch] {
                    continue;
                }
                ui.horizontal(|ui| {
                    ui.label(CHANNEL_LABELS[ch]);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("×").on_hover_text(format!("Remove {} from this project", CHANNEL_LABELS[ch])).clicked() {
                            remove_channel = Some(ch);
                        }
                    });
                });
            }
            ui.add_space(3.0);
            ui.label(
                egui::RichText::new("Removing a channel permanently discards its paint and material maps from every texture set. Adding it again starts empty.")
                    .small()
                    .color(theme::TEXT_DIM),
            );

            ui.add_space(14.0);
            section(ui, "Mesh Parts / Objects");
            let object_count = self.mesh.objects.len();
            if object_count == 0 || self.mesh.tri_object.len() != self.mesh.triangle_count() {
                ui.label(
                    egui::RichText::new("This mesh has no separately imported objects. Import an FBX or OBJ containing multiple objects to manage them here.")
                        .small()
                        .color(theme::TEXT_DIM),
                );
            } else {
                self.hidden_objects.resize(object_count, false);
                let visible_count = self.hidden_objects.iter().filter(|hidden| !**hidden).count();
                ui.horizontal(|ui| {
                    ui.label(format!("{visible_count} of {object_count} visible"));
                    if ui.small_button("Show all").on_hover_text("Restore every object").clicked() {
                        let changed = self.hidden_objects.iter().any(|hidden| *hidden);
                        if changed {
                            self.hidden_objects.fill(false);
                            self.rebuild_visible_bvh();
                            self.poly.shown = None;
                            self.overlays.clear(OverlaySlot::Selection);
                            self.vertex_id.shown = None;
                            self.project_settings_serial = self.project_settings_serial.wrapping_add(1);
                            self.status = "Showed all mesh objects.".into();
                        }
                    }
                });

                let mut toggle_object = None;
                for (index, name) in self.mesh.objects.iter().enumerate() {
                    let hidden = self.hidden_objects[index];
                    let (rect, response) =
                        ui.allocate_exact_size(vec2(ui.available_width(), 25.0), Sense::hover());
                    if response.hovered() {
                        ui.painter().rect_filled(rect, 3.0, Color32::from_gray(42));
                    }
                    let color = if hidden { theme::TEXT_DIM } else { theme::TEXT };
                    ui.painter().text(
                        rect.left_center() + vec2(7.0, 0.0),
                        egui::Align2::LEFT_CENTER,
                        name,
                        egui::FontId::proportional(13.0),
                        color,
                    );
                    // The eye appears on row hover; it remains visible for hidden
                    // rows so they can always be restored without hunting.
                    if response.hovered() || hidden {
                        let eye_rect = Rect::from_min_size(
                            pos2(rect.right() - 27.0, rect.top() + 1.0),
                            vec2(25.0, 23.0),
                        );
                        let icon = if hidden { icons::EYE_SLASH } else { icons::EYE };
                        let button = ui.put(eye_rect, egui::Button::new(icon).frame(false));
                        if button.on_hover_text(if hidden { "Show this object" } else { "Hide this object" }).clicked() {
                            toggle_object = Some((index, !hidden));
                        }
                    }
                }
                if let Some((index, hidden)) = toggle_object {
                    self.set_object_hidden(index, hidden);
                }
            }

            if let Some(ch) = remove_channel {
                self.pending_channel_remove = Some(ch);
            }
        });

        let mut decision = None;
        if let Some(ch) = self.pending_channel_remove {
            egui::Window::new(format!("Remove {}?", CHANNEL_LABELS[ch]))
                .id(egui::Id::unique("remove_project_channel"))
                .collapsible(false)
                .resizable(false)
                .show(ui.ctx(), |ui| {
                    ui.label(format!(
                        "This permanently discards {} data from every layer and texture set. This action cannot be undone. Adding the channel again starts empty.",
                        CHANNEL_LABELS[ch]
                    ));
                    ui.horizontal(|ui| {
                        if ui.button("Remove Channel").clicked() {
                            decision = Some(true);
                        }
                        if ui.button("Cancel").clicked() {
                            decision = Some(false);
                        }
                    });
                });
        }
        if let (Some(ch), Some(remove)) = (self.pending_channel_remove, decision) {
            self.pending_channel_remove = None;
            if remove {
                self.remove_project_channel(ch);
            }
        }
    }

    fn remove_project_channel(&mut self, ch: usize) {
        if ch >= crate::engine::CHANNELS || !self.active_channels[ch] {
            return;
        }
        let active_set = self.sets.active;
        let set_count = self.mesh.sets.len().max(1);
        for set in 0..set_count {
            self.undo.set = set;
            if set == active_set {
                Self::discard_channel_from_engine(&mut self.engine, &mut self.undo, ch);
            } else if let Some(parked) = self.sets.parked.get_mut(set).and_then(Option::as_mut) {
                Self::discard_channel_from_engine(&mut parked.engine, &mut self.undo, ch);
            }
        }
        self.undo.set = active_set;
        // Remove this channel from every historical snapshot/edit before it
        // can be restored by undo or redo; unrelated edits remain available.
        self.undo.discard_channel(ch);
        self.active_channels[ch] = false;
        if let Some(layers) = self.layer_clipboard.as_mut() {
            filter_layer_channels(layers, &self.active_channels);
        }
        self.project_settings_serial = self.project_settings_serial.wrapping_add(1);
        self.status = format!("Removed {} channel and discarded its data from all texture sets.", CHANNEL_LABELS[ch]);
    }

    fn discard_channel_from_engine(engine: &mut Engine, undo: &mut UndoStack, ch: usize) {
        // Preserve any property edits that were already pending before removal.
        engine.commit_props(undo);
        let mut changed = false;
        for layer in &mut engine.layers {
            match &mut layer.content {
                Content::Paint(channels) => {
                    changed |= channels[ch].take().is_some();
                }
                Content::Fill(fill) => {
                    if fill.mat.enabled[ch] {
                        fill.mat.enabled[ch] = false;
                        changed = true;
                    }
                }
                Content::Folder(_) => {}
            }
            changed |= layer.images[ch].take().is_some();
        }
        if changed {
            engine.commit_props_without_undo();
        }
    }
}
