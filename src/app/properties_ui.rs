//! The Properties panel: layer, fill material, brush and map-level controls.
use super::*;

impl PainterApp {
    pub(super) fn fix_properties(&mut self, ui: &mut egui::Ui, map: usize) {
        let info = &MAPS[map];
        section(ui, "Bake fix");
        ui.label(egui::RichText::new(format!("Painting corrections on {}.", info.name)).color(Color32::WHITE));
        ui.label(
            egui::RichText::new("Paint (1) sets the value below, Eraser (2) restores the original bake. Fixes survive re-baking.")
                .small()
                .color(theme::TEXT_DIM),
        );
        if info.kind == MapKind::Normal {
            ui.label(egui::RichText::new("Painting flattens the normal: use it to remove projection artifacts.").color(theme::TEXT));
        } else {
            ui.horizontal(|ui| {
                ui.label("Value");
                ui.add(egui::Slider::new(&mut self.brush.fix_value, 0.0..=1.0));
            });
            ui.horizontal(|ui| {
                for (label, v) in [("Black", 0.0), ("50%", 0.5), ("White", 1.0)] {
                    if ui.button(label).clicked() {
                        self.brush.fix_value = v;
                    }
                }
            });
        }
        if ui.button("Done fixing").clicked() {
            self.target = StrokeTarget::Content;
        }
        self.brush_properties(ui, false);
    }

    pub(super) fn properties(&mut self, ui: &mut egui::Ui) {
        if let StrokeTarget::BakeFix(m) = self.target {
            ui.label(egui::RichText::new(format!("Properties – Fix · {}", MAPS[m].name)).strong().color(Color32::WHITE));
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| self.fix_properties(ui, m));
            return;
        }
        let title = match self.engine.layers.get(self.selected) {
            None => "Properties".to_string(),
            Some(l) if self.target == StrokeTarget::Mask => format!("Properties – Mask of {}", l.name),
            Some(l) if l.is_fill() => format!("Properties – Fill · {}", l.name),
            Some(l) => format!("Properties – Paint · {}", l.name),
        };
        ui.label(egui::RichText::new(title).strong().color(Color32::WHITE));
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            let idx = self.selected;
            if idx >= self.engine.layers.len() {
                ui.label(egui::RichText::new("Select a layer.").color(theme::TEXT_DIM));
                return;
            }
            let mut dirty = false;

            section(ui, "Layer");
            {
                let layer = &mut self.engine.layers[idx];
                labeled(ui, "Name", |ui| ui.text_edit_singleline(&mut layer.name));
                labeled(ui, "Blend mode", |ui| {
                    egui::ComboBox::from_id_salt("blend").selected_text(layer.blend.name()).show_ui(ui, |ui| {
                        for b in Blend::ALL {
                            dirty |= ui.selectable_value(&mut layer.blend, b, b.name()).changed();
                        }
                    });
                });
                let mut op = layer.opacity * 100.0;
                if labeled(ui, "Opacity", |ui| ui.add(egui::Slider::new(&mut op, 0.0..=100.0).max_decimals(0).suffix("%")).changed()) {
                    layer.opacity = op / 100.0;
                    dirty = true;
                }
            }

            if self.target == StrokeTarget::Content && self.engine.layers[idx].mask.is_none() {
                // The mask is made on its own: no separate "Add mask" step to remember.
                section(ui, "Mask");
                ui.menu_button(theme::with_icon(icons::STACK, "Add mask layer ▾"), |ui| {
                    for k in GenKind::ALL {
                        if ui.button(k.name()).clicked() {
                            self.pending.push(Action::AddMaskLayer(k));
                            ui.close();
                        }
                    }
                })
                .response
                .on_hover_text("Adds a mask to this layer if it has none, with this mask layer on it (e.g. Vertex Color ID Mask)");
            }
            if self.target == StrokeTarget::Mask {
                dirty |= self.mask_properties(ui, idx);
                self.brush_properties(ui, false);
            } else {
                if self.engine.layers[idx].is_folder() {
                    section(ui, "Material graph");
                    labeled(ui, "Graph", |ui| {
                        if ui.button("Edit as graph").on_hover_text("The fill layers inside as nodes in the Material Designer (Ctrl+G)").clicked() {
                            self.pending.push(Action::EditGraph);
                        }
                    });
                    ui.label(
                        egui::RichText::new("Its fill layers show as Layer and Smart Mask nodes. Edits in the graph change the layers, and the other way round; wire your own nodes into a Layer node's mask.")
                            .small()
                            .color(theme::TEXT_DIM),
                    );
                }
                let is_fill = self.engine.layers[idx].is_fill();
                if is_fill {
                    let names: [Option<String>; crate::engine::CHANNELS] =
                        std::array::from_fn(|i| self.engine.layers[idx].images[i].as_ref().map(|im| im.name.clone()));
                    let graph_name = match &self.engine.layers[idx].content {
                        Content::Fill(f) => f.graph.map(|g| self.graphs.doc(g).map_or("(missing)".to_string(), |d| d.graph.name.clone())),
                        _ => None,
                    };
                    if let Some(name) = &graph_name {
                        section(ui, "Material graph");
                        labeled(ui, "Graph", |ui| {
                            ui.label(egui::RichText::new(name).color(Color32::WHITE));
                            if ui.button("Edit graph").on_hover_text("Ctrl+G").clicked() {
                                self.pending.push(Action::EditGraph);
                            }
                        });
                        ui.label(egui::RichText::new("The graph's Output nodes fill the channels below; uncheck a channel to leave it out.").small().color(theme::TEXT_DIM));
                        if let Some(gid) = match &self.engine.layers[idx].content {
                            Content::Fill(f) => f.graph,
                            _ => None,
                        } {
                            self.exposed_controls(ui, gid);
                        }
                    }
                    if let Content::Fill(f) = &mut self.engine.layers[idx].content {
                        section(ui, "Fill material");
                        dirty |= material_controls(ui, &mut f.mat, true);
                        if graph_name.is_none() {
                        section(ui, "Images");
                        let shelf_textures = &self.shelf.textures;
                        for (ch, label) in ["Color", "Roughness", "Metallic", "Height", "Opacity", "Emission", "Normal", "Specular"].into_iter().enumerate() {
                            // A shelf texture dropped on the row feeds the channel.
                            let row = ui.scope(|ui| labeled(ui, label, |ui| {
                                match &names[ch] {
                                    Some(name) => {
                                        ui.add(egui::Label::new(name.as_str()).truncate()).on_hover_text(name.as_str());
                                        if ui.small_button("✕").on_hover_text("Remove image").clicked() {
                                            self.pending.push(Action::SetFillImage { idx, ch, path: None });
                                        }
                                    }
                                    None => {
                                        if ui.button("Load image…").clicked()
                                            && let Some(path) = rfd::FileDialog::new().add_filter("Image", &IMAGE_EXT).pick_file()
                                        {
                                            self.pending.push(Action::SetFillImage { idx, ch, path: Some(path) });
                                        }
                                    }
                                }
                                // Any shelf texture can feed the channel.
                                ui.menu_button("Texture ▾", |ui| {
                                    ui.label(egui::RichText::new(format!("Shelf texture as {}", label.to_lowercase())).small().color(theme::TEXT_DIM));
                                    egui::ScrollArea::vertical().max_height(380.0).show(ui, |ui| {
                                        ui.set_width(4.0 * 70.0);
                                        ui.horizontal_wrapped(|ui| {
                                            ui.spacing_mut().item_spacing = vec2(6.0, 6.0);
                                            for (i, t) in crate::textures::TEXTURES.iter().enumerate() {
                                                let (r, resp) = ui.allocate_exact_size(vec2(64.0, 64.0), Sense::click());
                                                if let Some(h) = shelf_textures.get(i) {
                                                    ui.painter().image(h.id(), r, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
                                                }
                                                let stroke = if resp.hovered() { Stroke::new(1.5, theme::ACCENT) } else { Stroke::new(1.0, theme::OUTLINE) };
                                                ui.painter().rect_stroke(r, 0.0, stroke, egui::StrokeKind::Inside);
                                                if resp.on_hover_text(format!("{} · {}", t.name, t.category)).clicked() {
                                                    self.pending.push(Action::TextureToChannel { texture: i, ch });
                                                    ui.close();
                                                }
                                            }
                                        });
                                    });
                                });
                            }))
                            .response
                            .rect;
                            let drop = ui.interact(row, ui.scope_id().with(("image_drop", ch)), Sense::hover());
                            if let Some(ShelfItem::Texture(_)) = drop.dnd_hover_payload::<ShelfItem>().as_deref() {
                                ui.painter().rect_stroke(row.expand(1.0), 0.0, Stroke::new(2.0, theme::ACCENT), egui::StrokeKind::Inside);
                            }
                            if let Some(ShelfItem::Texture(texture)) = dnd_release::<ShelfItem>(&drop).as_deref() {
                                self.pending.push(Action::TextureToChannel { texture: *texture, ch });
                            }
                        }
                        if names[crate::engine::NORMAL].is_some() {
                            dirty |= slider(ui, "Normal strength", &mut f.mat.normal, 0.0..=2.0);
                        }
                        if names.iter().any(Option::is_some) {
                            labeled(ui, "Projection", |ui| {
                                egui::ComboBox::from_id_salt("projection").selected_text(f.projection.name()).show_ui(ui, |ui| {
                                    for p in Projection::ALL {
                                        dirty |= ui.selectable_value(&mut f.projection, p, p.name()).on_hover_text(p.tip()).changed();
                                    }
                                });
                            });
                            dirty |= labeled(ui, "Tiling", |ui| {
                                ui.add(egui::Slider::new(&mut f.tiling, 0.05..=64.0).logarithmic(true).max_decimals(2)).changed()
                            });
                            dirty |= slider(ui, "Rotation", &mut f.rotation, -180.0..=180.0);
                            dirty |= slider(ui, "Offset U", &mut f.offset[0], -1.0..=1.0);
                            dirty |= slider(ui, "Offset V", &mut f.offset[1], -1.0..=1.0);
                            if f.projection == Projection::Triplanar {
                                dirty |= slider(ui, "Blend sharpness", &mut f.tri_sharpness, 1.0..=16.0);
                            }
                        }
                        }
                        section(ui, "Procedural variation");
                        dirty |= slider(ui, "Color var.", &mut f.color_var, 0.0..=1.0);
                        dirty |= slider(ui, "Rough var.", &mut f.rough_var, 0.0..=1.0);
                        dirty |= slider(ui, "Height var.", &mut f.height_var, 0.0..=1.0);
                        dirty |= slider(ui, "Scale", &mut f.var_scale, 0.5..=60.0);
                        dirty |= slider(ui, "Seed", &mut f.seed, 0.0..=100.0);
                        dirty |= slider(ui, "Per-part var.", &mut f.part_var, 0.0..=1.0);
                    }
                    ui.add_space(6.0);
                    ui.label(egui::RichText::new("Fill layers cost no texture memory. Add a mask to paint where they show.").small().color(theme::TEXT_DIM));
                } else if !self.engine.layers[idx].is_folder() {
                    self.brush_properties(ui, true);
                }
            }
            if dirty {
                self.schedule_preview_update(ui.ctx());
            }
        });
    }

    /// A warning badge on a shelf swatch that reads maps the bake left out.
    pub(super) fn missing_badge(ui: &egui::Ui, resp: &egui::Response, missing: u32) {
        if missing == 0 {
            return;
        }
        let at = resp.rect.right_top() + vec2(-8.0, 6.0);
        let r = Rect::from_center_size(at + vec2(-5.0, 6.0), vec2(16.0, 16.0));
        ui.painter().rect_filled(r, 0.0, Color32::from_rgb(230, 160, 40));
        ui.painter().text(r.center(), egui::Align2::CENTER_CENTER, "!", egui::FontId::proportional(12.0), Color32::BLACK);
        resp.clone().on_hover_text(format!("Needs maps that weren't baked: {}. Bake them for this to look right.", crate::engine::map_names(missing)));
    }

    /// A note under a mask that reads maps the bake left out.
    /// What a view mode is called; custom maps by their names.
    pub(super) fn mode_name(&self, mode: ViewMode) -> String {
        match mode {
            ViewMode::Custom(id) => self.bake_settings.custom.iter().find(|c| c.id == id).map_or("Custom map".to_string(), |c| c.name.clone()),
            m => m.name().to_string(),
        }
    }

    pub(super) fn missing_maps_note(&mut self, ui: &mut egui::Ui, idx: usize) {
        let Some(mask) = self.engine.layers.get(idx).and_then(|l| l.mask.as_ref()) else { return };
        let mut needs = mask.generator.map_or(0, |g| g.needs_maps());
        for ml in &mask.layers {
            if let crate::engine::MaskSource::Generator(g) = &ml.source {
                needs |= g.needs_maps();
            }
        }
        let missing = needs & self.engine.missing_maps();
        // Custom maps it reads that the bake didn't make (or that were removed).
        let mut gens: Vec<&crate::engine::Generator> = mask.generator.iter().collect();
        gens.extend(mask.layers.iter().filter_map(|ml| match &ml.source {
            crate::engine::MaskSource::Generator(g) => Some(g),
            _ => None,
        }));
        let mut custom: Vec<String> = gens
            .iter()
            .flat_map(|g| self.engine.missing_custom(g))
            .map(|id| self.bake_settings.custom.iter().find(|c| c.id == id).map_or("a removed custom map".to_string(), |c| c.name.clone()))
            .collect();
        custom.dedup();
        if missing == 0 && custom.is_empty() {
            return;
        }
        let mut names: Vec<String> = (missing != 0).then(|| crate::engine::map_names(missing)).into_iter().collect();
        names.extend(custom);
        egui::Frame::new().fill(Color32::from_rgb(60, 45, 20)).inner_margin(egui::Margin::same(6)).show(ui, |ui| {
            ui.label(
                egui::RichText::new(format!("This mask reads maps that weren't baked: {}.", names.join(", ")))
                    .color(Color32::from_rgb(240, 190, 90)),
            );
            if ui.button("Open the baker").clicked() {
                self.show_bake = true;
            }
        });
    }

    /// The current tip with a grid of all tips to choose from.
    pub(super) fn tip_picker(&mut self, ui: &mut egui::Ui) {
        let name = self.brush.tip.map_or("Round", |i| crate::brushes::BRUSH_ALPHAS[i].name);
        labeled(ui, "Tip", |ui| {
            let tex = self.brush.tip.map(|i| self.shelf.brushes[i].id());
            let (rect, resp) = ui.allocate_exact_size(vec2(40.0, 40.0), Sense::click());
            ui.painter().rect_filled(rect, theme::RADIUS, theme::PANEL_DARK);
            match tex {
                Some(t) => {
                    ui.painter().image(t, rect.shrink(3.0), Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
                }
                None => {
                    ui.painter().circle_filled(rect.center(), 13.0, Color32::WHITE);
                }
            }
            let popup = egui::Popup::menu(&resp).close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside);
            ui.label(name);
            popup.show(|ui| {
                ui.set_max_width(300.0);
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = vec2(3.0, 3.0);
                    let cell = |ui: &mut egui::Ui, tex: Option<egui::TextureId>, selected: bool, tip: &str| {
                        let (rect, resp) = ui.allocate_exact_size(vec2(44.0, 44.0), Sense::click());
                        let bg = if selected { theme::ACCENT_DIM } else if resp.hovered() { theme::WIDGET_HOVER } else { theme::PANEL_DARK };
                        ui.painter().rect_filled(rect, theme::RADIUS, bg);
                        match tex {
                            Some(t) => {
                                ui.painter().image(t, rect.shrink(4.0), Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
                            }
                            None => {
                                ui.painter().circle_filled(rect.center(), 15.0, Color32::WHITE);
                            }
                        }
                        resp.on_hover_text(tip).clicked()
                    };
                    if cell(ui, None, self.brush.tip.is_none(), "Round") {
                        self.pending.push(Action::SetBrushTip(None));
                    }
                    for (i, b) in crate::brushes::BRUSH_ALPHAS.iter().enumerate() {
                        if cell(ui, Some(self.shelf.brushes[i].id()), self.brush.tip == Some(i), b.name) {
                            self.pending.push(Action::SetBrushTip(Some(i)));
                        }
                    }
                });
            });
        });
    }

    pub(super) fn brush_properties(&mut self, ui: &mut egui::Ui, content: bool) {
        let _ = content;
        section(ui, if self.tool == Tool::Eraser { "Eraser" } else { "Brush" });
        slider(ui, "Size", &mut self.brush.size, 2.0..=800.0);
        slider(ui, "Hardness", &mut self.brush.hardness, 0.0..=0.99);
        slider(ui, "Flow", &mut self.brush.flow, 0.01..=1.0);
        slider(ui, "Stroke opacity", &mut self.brush.opacity, 0.01..=1.0);
        slider(ui, "Spacing", &mut self.brush.spacing, 0.02..=1.0);
        section(ui, "Brush tip");
        self.tip_picker(ui);
        if self.brush.tip.is_some() {
            slider(ui, "Angle", &mut self.brush.angle, -180.0..=180.0);
            labeled(ui, "Follow stroke", |ui| ui.checkbox(&mut self.brush.follow, "").on_hover_text("Turn the tip along the stroke direction"));
            slider(ui, "Angle jitter", &mut self.brush.angle_jitter, 0.0..=1.0);
        }
        slider(ui, "Size jitter", &mut self.brush.size_jitter, 0.0..=1.0);
        slider(ui, "Scatter", &mut self.brush.scatter, 0.0..=2.0);

        section(ui, "Projection");
        labeled(ui, "Mode", |ui| projection_combo(ui, "projection_props", &mut self.brush.projection));
        if self.brush.projection == BrushProjection::Screen {
            labeled(ui, "Paint through", |ui| {
                ui.checkbox(&mut self.brush.paint_through, "").on_hover_text("Also paint surfaces hidden behind others")
            });
        }
        labeled(ui, "Angle limit", |ui| {
            ui.add(egui::Slider::new(&mut self.brush.angle_limit, 10.0..=90.0).max_decimals(0).suffix("°"))
                .on_hover_text("Paint fades out on surfaces turned further than this from the brush, so it doesn't smear on slopes")
        });

        section(ui, "Stroke");
        labeled(ui, "Stabilizer", |ui| {
            ui.add(egui::Slider::new(&mut self.brush.stabilize, 0.0..=200.0).max_decimals(0))
                .on_hover_text("Lazy mouse: the brush trails the cursor on a string for smooth lines")
        });
        labeled(ui, "Pen pressure", |ui| {
            ui.checkbox(&mut self.brush.pressure_size, "Size");
            ui.checkbox(&mut self.brush.pressure_flow, "Flow");
        });
        ui.label(egui::RichText::new("Shift+click draws a straight line from the last stroke.").small().color(theme::TEXT_DIM));

        section(ui, "Grain");
        labeled(ui, "Texture", |ui| {
            let name = self.brush.grain.map_or("None", |i| crate::textures::TEXTURES[i].name);
            egui::ComboBox::from_id_salt("brush_grain").selected_text(name).height(400.0).show_ui(ui, |ui| {
                ui.selectable_value(&mut self.brush.grain, None, "None");
                for (i, t) in crate::textures::TEXTURES.iter().enumerate() {
                    ui.selectable_value(&mut self.brush.grain, Some(i), t.name);
                }
            })
        });
        if self.brush.grain.is_some() {
            slider(ui, "Strength", &mut self.brush.grain_strength, 0.0..=1.0);
            labeled(ui, "Scale", |ui| {
                ui.add(egui::Slider::new(&mut self.brush.grain_scale, 0.25..=64.0).logarithmic(true).max_decimals(2))
                    .on_hover_text("Repeats across the mesh (or the UV square with the UV projection)")
            });
            labeled(ui, "Invert", |ui| ui.checkbox(&mut self.brush.grain_invert, ""));
        } else {
            ui.label(egui::RichText::new("A texture fixed to the surface that breaks up the paint: scratches, dust, chipping.").small().color(theme::TEXT_DIM));
        }

        if content {
            section(ui, if self.tool == Tool::Eraser { "Erase channels" } else { "Material" });
            material_controls(ui, &mut self.brush.mat, self.tool != Tool::Eraser);
            if self.tool != Tool::Eraser {
                ui.label(egui::RichText::new("Tip: right-click a material in the shelf → Use as brush material.").small().color(theme::TEXT_DIM));
            }
        } else if self.tool != Tool::Eraser && self.target == StrokeTarget::Mask {
            section(ui, "Mask value");
            ui.horizontal(|ui| {
                if ui.button("Black").clicked() {
                    self.brush.mask_value = 0.0;
                }
                if ui.button("White").clicked() {
                    self.brush.mask_value = 1.0;
                }
                ui.add(egui::Slider::new(&mut self.brush.mask_value, 0.0..=1.0));
            });
        }
    }
}
