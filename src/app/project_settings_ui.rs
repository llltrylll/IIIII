//! Project-wide channel and object visibility controls.
use super::*;

const CHANNEL_LABELS: [&str; crate::engine::CHANNELS] =
    ["Base Color", "Roughness", "Metallic", "Height", "Opacity", "Emission", "Normal", "Specular"];
const CHANNEL_ORDER: [usize; crate::engine::CHANNELS] = [0, 6, 1, 2, HEIGHT_CHANNEL, 4, 5, 7];
const HEIGHT_CHANNEL: usize = crate::engine::HEIGHT;

/// Per-layer channel outputs that stay stored but remain inactive in this project.
pub(super) fn disabled_fill_channels(
    fill: &FillParams,
    active: &[bool; crate::engine::CHANNELS],
) -> [bool; crate::engine::CHANNELS] {
    std::array::from_fn(|ch| !active[ch] && fill.mat.enabled[ch])
}

/// Intersects a brush's own channel choices with the channels this project permits.
pub(super) fn effective_channel_mask(
    brush_enabled: [bool; crate::engine::CHANNELS],
    active: &[bool; crate::engine::CHANNELS],
) -> [bool; crate::engine::CHANNELS] {
    std::array::from_fn(|ch| brush_enabled[ch] && active[ch])
}

/// Fixed-size mask of channels that contain data while disabled in Project.
/// This is display-only and does not strip or alter imported material data.
pub(super) fn disabled_map_flags(
    layer: &Layer,
    active: &[bool; crate::engine::CHANNELS],
) -> [bool; crate::engine::CHANNELS] {
    std::array::from_fn(|ch| {
        if active[ch] {
            return false;
        }
        layer.disabled_channels[ch] || match &layer.content {
            Content::Paint(channels) => channels[ch].is_some(),
            Content::Fill(fill) => fill.mat.enabled[ch] || layer.images[ch].is_some(),
            Content::Folder(_) => false,
        }
    })
}

/// Converts the fixed channel mask to labels only when a tooltip is requested.
pub(super) fn disabled_map_names(
    flags: &[bool; crate::engine::CHANNELS],
) -> Vec<&'static str> {
    CHANNEL_ORDER.iter().copied().filter_map(|ch| flags[ch].then_some(CHANNEL_LABELS[ch])).collect()
}

/// Marks inactive channels on pasted layers without deleting their maps or paint.
pub(super) fn mark_inactive_layer_channels(
    layers: &mut [Layer],
    active: &[bool; crate::engine::CHANNELS],
) {
    for layer in layers {
        for ch in 0..crate::engine::CHANNELS {
            if active[ch] {
                layer.disabled_channels[ch] = false;
                continue;
            }
            let has_data = match &layer.content {
                Content::Paint(channels) => channels[ch].is_some(),
                Content::Fill(fill) => fill.mat.enabled[ch] || layer.images[ch].is_some(),
                Content::Folder(_) => false,
            };
            layer.disabled_channels[ch] = has_data;
        }
    }
}

#[cfg(test)]
mod channel_mask_tests {
    use super::disabled_fill_channels;
    use crate::engine::{Blend, Content, FillParams, Layer, CHANNELS};
    use super::HEIGHT_CHANNEL;

    #[test]
    fn disabled_height_is_recorded_without_mutating_the_source_material() {
        let mut fill = FillParams::default();
        fill.mat.enabled = [true; CHANNELS];
        let mut active = [true; CHANNELS];
        active[HEIGHT_CHANNEL] = false;

        let disabled = disabled_fill_channels(&fill, &active);

        assert!(disabled[HEIGHT_CHANNEL]);
        assert!(!disabled[0]);
        assert!(fill.mat.enabled[HEIGHT_CHANNEL], "the source map must remain stored and enabled");
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
    fn clipboard_marks_disabled_outputs_without_stripping_their_maps() {
        use super::mark_inactive_layer_channels;

        let mut fill = FillParams::default();
        fill.mat.enabled = [true; CHANNELS];
        let mut layer = Layer {
            id: 1,
            name: "Copied fill".into(),
            visible: true,
            opacity: 1.0,
            blend: Blend::Normal,
            content: Content::Fill(fill),
            mask: None,
            disabled_channels: [false; CHANNELS],
            parent: None,
            images: Default::default(),
            thumb: None,
        };
        let mut active = [true; CHANNELS];
        active[HEIGHT_CHANNEL] = false;

        mark_inactive_layer_channels(std::slice::from_mut(&mut layer), &active);

        {
            let Content::Fill(fill) = &layer.content else { panic!("expected fill layer") };
            assert!(fill.mat.enabled[HEIGHT_CHANNEL], "clipboard must retain the source map flag");
            assert!(fill.mat.enabled[0]);
        }
        assert!(layer.disabled_channels[HEIGHT_CHANNEL]);
        assert!(!layer.disabled_channels[0]);

        active[HEIGHT_CHANNEL] = true;
        mark_inactive_layer_channels(std::slice::from_mut(&mut layer), &active);
        assert!(!layer.disabled_channels[HEIGHT_CHANNEL], "re-enabled channels must be restored in clipboard layers");
    }

    #[test]
    fn inactive_map_indicator_only_marks_disabled_channels_with_data() {
        use super::{disabled_map_flags, disabled_map_names};

        let mut fill = FillParams::default();
        fill.mat.enabled[HEIGHT_CHANNEL] = true;
        let mut disabled_channels = [false; CHANNELS];
        disabled_channels[HEIGHT_CHANNEL] = true;
        let layer = Layer {
            id: 1,
            name: "Preset height".into(),
            visible: true,
            opacity: 1.0,
            blend: Blend::Normal,
            content: Content::Fill(fill),
            mask: None,
            disabled_channels,
            parent: None,
            images: Default::default(),
            thumb: None,
        };
        let mut active = [true; CHANNELS];
        active[HEIGHT_CHANNEL] = false;

        let flags = disabled_map_flags(&layer, &active);
        assert!(flags[HEIGHT_CHANNEL]);
        assert_eq!(disabled_map_names(&flags), vec!["Height"]);
        active[HEIGHT_CHANNEL] = true;
        let flags = disabled_map_flags(&layer, &active);
        assert!(!flags[HEIGHT_CHANNEL]);
        assert!(disabled_map_names(&flags).is_empty());
    }

    #[test]
    fn active_project_channels_do_not_mark_source_maps_inactive() {
        let mut fill = FillParams::default();
        fill.mat.enabled = [false; CHANNELS];
        fill.mat.enabled[HEIGHT_CHANNEL] = true;
        let disabled = disabled_fill_channels(&fill, &[true; CHANNELS]);
        assert!(!disabled[HEIGHT_CHANNEL]);
        assert!(fill.mat.enabled[HEIGHT_CHANNEL]);
    }
}

impl PainterApp {
    pub(super) fn project_settings_panel(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Project");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if theme::icon_button(ui, icons::X, false, 22.0)
                        .on_hover_text("Close Project panel")
                        .clicked()
                    {
                        self.project_panel_open = false;
                    }
                });
            });
            ui.add_space(8.0);

            section(ui, "Channels");
            ui.label(
                egui::RichText::new("Inactive channels stay stored in their layers, but are excluded from the material output.")
                    .small()
                    .color(theme::TEXT_DIM),
            );
            let mut disable_channel = None;
            ui.menu_button(theme::with_icon(icons::PLUS, "Enable Channel"), |ui| {
                let mut added = None;
                for ch in CHANNEL_ORDER {
                    if !self.active_channels[ch] && ui.button(CHANNEL_LABELS[ch]).clicked() {
                        added = Some(ch);
                        ui.close();
                    }
                }
                if let Some(ch) = added {
                    self.enable_project_channel(ch);
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
                        if ui.small_button("×").on_hover_text(format!("Disable {} output but preserve its maps", CHANNEL_LABELS[ch])).clicked() {
                            disable_channel = Some(ch);
                        }
                    });
                });
            }
            ui.add_space(3.0);
            ui.label(
                egui::RichText::new("Disabling a channel only hides its output. Its maps and paint remain stored and are restored when it is enabled again.")
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
                            self.views.invalidate_shadow();
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
                    // Keep the hit target alive while the cursor enters it. If the
                    // button is created only while the row is hovered, its own hover
                    // state can make the row response false and remove the button.
                    let eye_rect = Rect::from_min_size(
                        pos2(rect.right() - 27.0, rect.top() + 1.0),
                        vec2(25.0, 23.0),
                    );
                    let eye_id = ui.scope_id().with(("project_mesh_object_eye", index));
                    let eye = ui.interact(eye_rect, eye_id, Sense::click());
                    let icon = if hidden { icons::EYE_SLASH } else { icons::EYE };
                    if eye.hovered() {
                        ui.painter().rect_filled(eye_rect, 3.0, Color32::from_gray(58));
                    }
                    ui.painter().text(
                        eye_rect.center(),
                        egui::Align2::CENTER_CENTER,
                        icon,
                        egui::FontId::proportional(15.0),
                        if hidden { theme::TEXT_DIM } else { theme::TEXT },
                    );
                    if eye.on_hover_text(if hidden { "Show this object" } else { "Hide this object" }).clicked() {
                        toggle_object = Some((index, !hidden));
                    }
                }
                if let Some((index, hidden)) = toggle_object {
                    self.set_object_hidden(index, hidden);
                }
            }

            if let Some(ch) = disable_channel {
                self.pending_channel_disable = Some(ch);
            }
        });

        let mut decision = None;
        if let Some(ch) = self.pending_channel_disable {
            egui::Window::new(format!("Disable {}?", CHANNEL_LABELS[ch]))
                .id(egui::Id::unique("disable_project_channel"))
                .collapsible(false)
                .resizable(false)
                .show(ui.ctx(), |ui| {
                    ui.label(format!(
                        "This disables {} in the material output. All existing maps and paint remain stored and return when the channel is enabled again.",
                        CHANNEL_LABELS[ch]
                    ));
                    ui.horizontal(|ui| {
                        if ui.button("Disable Channel").clicked() {
                            decision = Some(true);
                        }
                        if ui.button("Cancel").clicked() {
                            decision = Some(false);
                        }
                    });
                });
        }
        if let (Some(ch), Some(disable)) = (self.pending_channel_disable, decision) {
            self.pending_channel_disable = None;
            if disable {
                self.disable_project_channel(ch);
            }
        }
    }

    fn disable_project_channel(&mut self, ch: usize) {
        if ch >= crate::engine::CHANNELS || !self.active_channels[ch] {
            return;
        }
        let active_set = self.sets.active;
        let set_count = self.mesh.sets.len().max(1);
        for set in 0..set_count {
            self.undo.set = set;
            if set == active_set {
                Self::mark_channel_disabled(&mut self.engine, &mut self.undo, ch);
            } else if let Some(parked) = self.sets.parked.get_mut(set).and_then(Option::as_mut) {
                Self::mark_channel_disabled(&mut parked.engine, &mut self.undo, ch);
            }
        }
        self.undo.set = active_set;
        self.active_channels[ch] = false;
        if let Some(layers) = self.layer_clipboard.as_mut() {
            mark_inactive_layer_channels(layers, &self.active_channels);
        }
        self.project_settings_serial = self.project_settings_serial.wrapping_add(1);
        self.status = format!("Disabled {} output. Existing maps and paint were preserved.", CHANNEL_LABELS[ch]);
    }

    fn enable_project_channel(&mut self, ch: usize) {
        if ch >= crate::engine::CHANNELS || self.active_channels[ch] {
            return;
        }
        let active_set = self.sets.active;
        let set_count = self.mesh.sets.len().max(1);
        for set in 0..set_count {
            self.undo.set = set;
            if set == active_set {
                Self::mark_channel_enabled(&mut self.engine, &mut self.undo, ch);
            } else if let Some(parked) = self.sets.parked.get_mut(set).and_then(Option::as_mut) {
                Self::mark_channel_enabled(&mut parked.engine, &mut self.undo, ch);
            }
        }
        self.undo.set = active_set;
        self.active_channels[ch] = true;
        if let Some(layers) = self.layer_clipboard.as_mut() {
            for layer in layers {
                layer.disabled_channels[ch] = false;
            }
        }
        self.project_settings_serial = self.project_settings_serial.wrapping_add(1);
        self.status = format!("Enabled {} output. Preserved maps and paint are available again.", CHANNEL_LABELS[ch]);
    }

    fn mark_channel_disabled(engine: &mut Engine, undo: &mut UndoStack, ch: usize) {
        engine.commit_props(undo);
        for layer in &mut engine.layers {
            let has_data = match &layer.content {
                Content::Paint(channels) => channels[ch].is_some(),
                Content::Fill(fill) => fill.mat.enabled[ch] || layer.images[ch].is_some(),
                Content::Folder(_) => false,
            };
            layer.disabled_channels[ch] = has_data;
        }
        engine.mark_dirty();
    }

    fn mark_channel_enabled(engine: &mut Engine, undo: &mut UndoStack, ch: usize) {
        engine.commit_props(undo);
        for layer in &mut engine.layers {
            layer.disabled_channels[ch] = false;
        }
        engine.mark_dirty();
    }
}
