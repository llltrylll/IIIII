//! The Layers tab: the layer list and rows, and the texture-set picker.
use super::*;

impl PainterApp {
    pub(super) fn layers_panel(&mut self, ui: &mut egui::Ui) {
        // Which texture set the stack below belongs to.
        if self.mesh.sets.len() > 1 {
            let mut pick = None;
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Texture set").color(theme::TEXT_DIM));
                let current = self.mesh.sets[self.sets.active].name.clone();
                egui::ComboBox::from_id_salt("texture set")
                    .selected_text(current)
                    .width(ui.available_width())
                    .show_ui(ui, |ui| {
                        for (k, set) in self.mesh.sets.iter().enumerate() {
                            let resp = ui.selectable_label(k == self.sets.active, &set.name);
                            // Hovering an entry shows its part of the mesh.
                            if resp.hovered() {
                                self.sets.hover = Some(k);
                            }
                            if resp.clicked() {
                                pick = Some(k);
                            }
                        }
                    })
                    .response
                    .on_hover_text("Each material of the mesh has its own layers, size and mesh maps");
            });
            if let Some(k) = pick {
                self.pick_set(k, ui.ctx());
            }
            ui.add_space(2.0);
        }
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            let mut btn = |ui: &mut egui::Ui, icon: &str, tip: &str, action: Option<Action>| {
                let enabled = action.is_some();
                let resp = ui.add_enabled_ui(enabled, |ui| theme::icon_button(ui, icon, false, 28.0)).inner;
                if resp.on_hover_text(tip).clicked()
                    && let Some(a) = action {
                        self.pending.push(a);
                    }
            };
            let has = self.selected < self.engine.layers.len();
            let no_mask = has && self.engine.layers[self.selected].mask.is_none();
            btn(ui, icons::PAINT_BRUSH, "Add paint layer", Some(Action::AddPaint));
            btn(ui, icons::PAINT_BUCKET, "Add fill layer", Some(Action::AddFill(FillParams::default(), "Fill layer".into())));
            btn(ui, icons::SPARKLE, "Add graph fill layer (a material graph)", Some(Action::AddGraphLayer));
            btn(ui, icons::FOLDER_PLUS, "Add folder (inside the selected folder)", Some(Action::AddFolder));
            btn(ui, icons::MASK, "Add black mask", no_mask.then_some(Action::AddMask { white: false }));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                btn(ui, icons::TRASH, "Delete layer (Delete)", has.then_some(Action::Delete));
                btn(ui, icons::FILE_PLUS, "Paste copied layer (Ctrl+V)", self.layer_clipboard.as_ref().is_some_and(|layers| !layers.is_empty()).then_some(Action::PasteLayer));
                btn(ui, icons::COPY, "Copy layer (Ctrl+C)", has.then_some(Action::CopyLayer));
                btn(ui, icons::COPY, "Duplicate layer", has.then_some(Action::Duplicate));
            });
        });
        ui.add_space(4.0);
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 1.0;
            for idx in (0..self.engine.layers.len()).rev() {
                let collapsed = self
                    .engine
                    .ancestors(idx)
                    .any(|a| matches!(self.engine.layers[a].content, Content::Folder(Folder { open: false, .. })));
                if !collapsed {
                    self.layer_row(ui, idx);
                }
            }
            // Clicking the empty space below the layers selects none.
            let rest = ui.available_rect_before_wrap();
            if rest.height() > 2.0 && ui.allocate_rect(rest, Sense::click()).clicked() {
                self.selected = NO_LAYER;
                self.target = StrokeTarget::Content;
            }
            if self.engine.layers.is_empty() {
                ui.label(egui::RichText::new("No layers. Add a paint or fill layer, or drop a smart material from the shelf.").color(theme::TEXT_DIM));
            }
        });
    }

    pub(super) fn layer_row(&mut self, ui: &mut egui::Ui, idx: usize) {
        let h = 46.0;
        let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), h), Sense::click_and_drag());
        let indent = self.engine.depth(idx) as f32 * 16.0;
        let layer = &self.engine.layers[idx];
        let (id, visible, name) = (layer.id, layer.visible, layer.name.clone());
        resp.widget_info(|| egui::WidgetInfo::selected(egui::Role::ListItem, true, idx == self.selected, &name));
        let folder_open = match layer.content {
            Content::Folder(f) => Some(f.open),
            _ => None,
        };
        let selected = idx == self.selected;
        let base_id = ui.scope_id().with(("layer", id));

        let painter = ui.painter_at(rect);
        let bg = if selected {
            theme::SELECT_BG
        } else if resp.hovered() {
            theme::WIDGET
        } else {
            theme::PANEL_DARK
        };
        painter.rect_filled(rect.shrink2(vec2(0.0, 1.0)), theme::RADIUS, bg);
        if selected {
            painter.rect_stroke(rect.shrink2(vec2(0.0, 1.0)), theme::RADIUS, Stroke::new(1.0, theme::ACCENT), egui::StrokeKind::Inside);
        }
        if selected {
            painter.rect_filled(Rect::from_min_size(rect.min + vec2(0.0, 6.0), vec2(3.0, h - 12.0)), 1.5, theme::ACCENT);
        }

        // Visibility toggle.
        let eye_rect = Rect::from_center_size(pos2(rect.min.x + 16.0, rect.center().y), vec2(22.0, 22.0));
        let eye = ui.interact(eye_rect, base_id.with("eye"), Sense::click());
        let (eye_icon, eye_col) = if visible { (icons::EYE, theme::TEXT) } else { (icons::EYE_SLASH, theme::TEXT_DIM.gamma_multiply(0.6)) };
        let eye_col = if eye.hovered() { Color32::WHITE } else { eye_col };
        painter.text(eye_rect.center(), egui::Align2::CENTER_CENTER, eye_icon, egui::FontId::proportional(15.0), eye_col);

        // Content thumbnail, indented by folder depth.
        let thumb = Rect::from_min_size(pos2(eye_rect.max.x + 6.0 + indent, rect.min.y + 5.0), vec2(36.0, 36.0));
        let thumb_resp = ui.interact(thumb, base_id.with("thumb"), Sense::click());
        if indent > 0.0 {
            let x = eye_rect.max.x + indent - 4.0;
            painter.line_segment([pos2(x, rect.top()), pos2(x, rect.bottom())], Stroke::new(1.0, theme::BORDER));
        }
        match &layer.content {
            Content::Paint(_) => {
                paint_checker(&painter, thumb);
                if let Some(tid) = layer.thumb {
                    painter.image(tid, thumb, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
                }
            }
            Content::Fill(f) => match self.engine.content_thumb(id) {
                // The fill as it comes out on the texture set (images, graph, variation).
                Some(tid) => {
                    paint_checker(&painter, thumb);
                    painter.image(tid, thumb, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
                }
                None => {
                    painter.rect_filled(thumb, 0.0, srgb_u8(f.mat.color));
                }
            },
            Content::Folder(f) => {
                // Folder icon, open or closed; clicking it toggles.
                let col = if visible { Color32::from_rgb(0xe8, 0xb9, 0x4a) } else { theme::TEXT_DIM };
                let icon = if f.open { icons::FOLDER_OPEN } else { icons::FOLDER };
                painter.text(thumb.center(), egui::Align2::CENTER_CENTER, icon, egui::FontId::proportional(24.0), col);
            }
        }
        if selected && self.target == StrokeTarget::Content {
            painter.rect_stroke(thumb.expand(1.0), 0.0, Stroke::new(2.0, theme::ACCENT), egui::StrokeKind::Outside);
        }

        // Mask thumbnail.
        let mut x = thumb.max.x + 6.0;
        let mut mask_resp = None;
        if let Some(mask) = &layer.mask {
            let mrect = Rect::from_min_size(pos2(x, rect.min.y + 5.0), vec2(36.0, 36.0));
            mask_resp = Some(ui.interact(mrect, base_id.with("mask"), Sense::click()));
            painter.rect_filled(mrect, 0.0, Color32::BLACK);
            match self.engine.mask_thumb(id) {
                // The mask as it comes out (painting, smart mask, layers, filters).
                Some(tid) => {
                    painter.image(tid, mrect, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
                }
                None if mask.paint.is_some() => {
                    painter.circle_filled(mrect.center(), 10.0, Color32::WHITE);
                }
                None => {}
            }
            // A smart mask says so in the corner.
            if mask.generator.is_some() || !mask.layers.is_empty() {
                let at = mrect.right_bottom() - vec2(7.0, 7.0);
                painter.circle_filled(at, 6.5, Color32::from_black_alpha(190));
                painter.text(at, egui::Align2::CENTER_CENTER, icons::SPARKLE, egui::FontId::proportional(9.5), theme::ACCENT);
            }
            if selected && self.target == StrokeTarget::Mask {
                painter.rect_stroke(mrect.expand(1.0), 0.0, Stroke::new(2.0, theme::ACCENT), egui::StrokeKind::Outside);
            }
            x = mrect.max.x + 6.0;
        }

        // Name and blend info.
        let text_col = if visible { Color32::WHITE } else { theme::TEXT_DIM };
        let title = theme::elided(&painter, &name, egui::FontId::proportional(12.5), text_col, rect.max.x - x - 8.0);
        let name_cut = title.elided;
        painter.galley(pos2(x, rect.min.y + 13.0 - title.size().y * 0.5), title, text_col);
        let mb = self.engine.layer_vram_bytes(idx) as f64 / (1024.0 * 1024.0);
        let info = match &layer.content {
            Content::Folder(_) => {
                let inside = self.engine.block(idx).len() - 1;
                format!("Folder  ·  {inside} layers  ·  {:.0}%", layer.opacity * 100.0)
            }
            Content::Fill(_) => format!("Fill  {} {:.0}%  ·  {:.0} MB", layer.blend.short(), layer.opacity * 100.0, mb),
            Content::Paint(_) => format!("Paint  {} {:.0}%  ·  {:.0} MB", layer.blend.short(), layer.opacity * 100.0, mb),
        };
        painter.text(pos2(x, rect.min.y + 31.0), egui::Align2::LEFT_CENTER, info, egui::FontId::proportional(10.5), theme::TEXT_DIM);
        painter.line_segment([rect.left_bottom(), rect.right_bottom()], Stroke::new(1.0, theme::BORDER));
        let resp = if name_cut { resp.on_hover_text(&name) } else { resp };

        // Interaction.
        if eye.clicked() {
            self.engine.layers[idx].visible = !visible;
            self.engine.mark_dirty();
        } else if mask_resp.as_ref().is_some_and(|r| r.clicked()) {
            self.selected = idx;
            self.target = StrokeTarget::Mask;
        } else if thumb_resp.clicked() {
            if let Content::Folder(f) = &mut self.engine.layers[idx].content {
                f.open = !f.open;
            }
            self.selected = idx;
            self.target = StrokeTarget::Content;
        } else if resp.double_clicked() && folder_open.is_some() {
            if let Content::Folder(f) = &mut self.engine.layers[idx].content {
                f.open = !f.open;
            }
        } else if resp.clicked() {
            if self.selected != idx || matches!(self.target, StrokeTarget::BakeFix(_)) {
                self.target = StrokeTarget::Content;
            }
            self.selected = idx;
        }

        // Drag to reorder: the top or bottom part of a row drops above or below
        // it; the middle of a folder (or anywhere below an open one) drops inside.
        resp.dnd_set_drag_payload(LayerDrag(idx));
        let drop_at = |y: f32| {
            let t = (y - rect.top()) / rect.height();
            match folder_open {
                Some(open) if t >= 0.25 && (open || t <= 0.75) => Drop::Into(id),
                _ if t < 0.5 => Drop::Above(id),
                _ => Drop::Below(id),
            }
        };
        if let (Some(from), Some(p)) = (resp.dnd_hover_payload::<LayerDrag>(), ui.ctx().pointer_interact_pos())
            && from.0 != idx {
                let accent = Stroke::new(2.0, theme::ACCENT);
                match drop_at(p.y) {
                    Drop::Above(_) => {
                        ui.painter().line_segment([rect.left_top(), rect.right_top()], accent);
                    }
                    Drop::Below(_) => {
                        ui.painter().line_segment([rect.left_bottom(), rect.right_bottom()], accent);
                    }
                    Drop::Into(_) => {
                        ui.painter().rect_stroke(rect.shrink(1.0), 0.0, accent, egui::StrokeKind::Inside);
                    }
                }
            }
        if let (Some(from), Some(p)) = (dnd_release::<LayerDrag>(&resp), ui.ctx().pointer_interact_pos())
            && from.0 != idx {
                self.pending.push(Action::MoveTo { from: from.0, drop: drop_at(p.y) });
            }
        // Shelf items land where they're dropped; a generator masks the row's layer.
        if let (Some(item), Some(p)) = (resp.dnd_hover_payload::<ShelfItem>(), ui.ctx().pointer_interact_pos()) {
            let accent = Stroke::new(2.0, theme::ACCENT);
            match (*item, drop_at(p.y)) {
                (ShelfItem::Generator(_), _) | (_, Drop::Into(_)) => {
                    ui.painter().rect_stroke(rect.shrink(1.0), 0.0, accent, egui::StrokeKind::Inside);
                }
                (_, Drop::Above(_)) => {
                    ui.painter().line_segment([rect.left_top(), rect.right_top()], accent);
                }
                (_, Drop::Below(_)) => {
                    ui.painter().line_segment([rect.left_bottom(), rect.right_bottom()], accent);
                }
            }
        }
        if let (Some(item), Some(p)) = (dnd_release::<ShelfItem>(&resp), ui.ctx().pointer_interact_pos()) {
            if let ShelfItem::Generator(g) = *item {
                self.selected = idx;
                self.pending.push(Action::AddGenerator(g));
            } else {
                let row_parent = self.engine.layers[idx].parent;
                let (at, parent) = match drop_at(p.y) {
                    Drop::Above(_) => (idx + 1, row_parent),
                    Drop::Below(_) => (self.engine.block(idx).start, row_parent),
                    Drop::Into(_) => (idx, Some(id)),
                };
                self.pending.push(Action::ShelfDrop { item: *item, at, parent });
            }
        }
        resp.context_menu(|ui| {
            self.selected = idx;
            self.layer_menu_items(ui);
        });
    }

    pub(super) fn texture_set_panel(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            if self.mesh.sets.len() > 1 {
                section(ui, "Texture sets");
                self.texture_set_list(ui);
            }
            section(ui, "Texture set");
            labeled(ui, "Mesh", |ui| ui.label(&self.mesh.name));
            let set = &self.mesh.sets[self.sets.active.min(self.mesh.sets.len() - 1)];
            if self.mesh.sets.len() > 1 {
                labeled(ui, "Material", |ui| ui.label(&set.name));
            }
            labeled(ui, "Triangles", |ui| ui.label((set.tris.end - set.tris.start).to_string()));
            let res = self.engine.res;
            labeled(ui, "Size", |ui| {
                egui::ComboBox::from_id_salt("res").selected_text(format!("{res} x {res}")).show_ui(ui, |ui| {
                    for r in RESOLUTIONS {
                        if ui.selectable_label(r == res, format!("{r} x {r}")).clicked() && r != res {
                            self.pending.push(Action::SetResolution(r));
                        }
                    }
                });
            });
            ui.label(egui::RichText::new("Changing size resamples every painted layer; stroke history is cleared.").small().color(theme::TEXT_DIM));

            section(ui, "Normal");
            slider(ui, "Height scale", &mut self.view.height_strength, 0.0..=8.0);

            section(ui, "Mesh maps");
            if ui.button("Open baker…").clicked() {
                self.show_bake = true;
            }
        });
    }
}
