//! The Shelf: materials, brushes, smart masks and textures to browse and drop in.
use super::*;

impl PainterApp {
    pub(super) fn shelf_panel(&mut self, ui: &mut egui::Ui) {
        // A narrow shelf keeps its room for thumbnails: categories as icons
        // (names on hover), a header without the hint, - / + for the size.
        let compact = ui.available_width() < SHELF_COMPACT;
        egui::Panel::left("shelf_cats")
            .exact_size(if compact { 40.0 } else { 176.0 })
            .resizable(false)
            .frame(egui::Frame::new().fill(theme::PANEL_DARK).inner_margin(egui::Margin::symmetric(if compact { 4 } else { 8 }, 8)))
            .show(ui, |ui| {
                if !compact {
                    ui.label(egui::RichText::new("SHELF").small().color(theme::TEXT_DIM));
                }
                ui.spacing_mut().item_spacing.y = 2.0;
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                for (cat, icon, name, count) in [
                    (ShelfCat::SmartMaterials, icons::SPARKLE, "Smart materials", presets::SMART_MATERIALS.len() + self.user_smarts.len()),
                    (ShelfCat::Materials, icons::SPHERE, "Materials", presets::MATERIALS.len()),
                    (ShelfCat::Generators, icons::MAGIC_WAND, "Generators", GenKind::ALL.len()),
                    (ShelfCat::Brushes, icons::PAINT_BRUSH, "Brushes", crate::brushes::BRUSH_ALPHAS.len()),
                    (ShelfCat::Textures, icons::IMAGES, "Textures", crate::textures::TEXTURES.len()),
                    (ShelfCat::Graphs, icons::SPARKLE, "Graphs", self.graphs.shelf.len()),
                    (ShelfCat::Downloaded, icons::DOWNLOAD_SIMPLE, "Downloaded", self.library.local.len()),
                ] {
                    let on = self.shelf.cat == cat;
                    let color = if on { Color32::WHITE } else { theme::TEXT };
                    let resp = if compact {
                        ui.add_sized([32.0, 26.0], theme::selectable(on, egui::RichText::new(icon).size(15.0).color(color)))
                            .on_hover_text(format!("{name} ({count})"))
                    } else {
                        let resp = ui.add_sized([158.0, 21.0], theme::selectable(on, egui::RichText::new(format!("{}  {name}", icon)).color(color)));
                        ui.painter().text(
                            resp.rect.right_center() - vec2(8.0, 0.0),
                            egui::Align2::RIGHT_CENTER,
                            count.to_string(),
                            egui::FontId::proportional(10.0),
                            theme::TEXT_DIM,
                        );
                        resp
                    };
                    if resp.clicked() {
                        self.shelf.cat = cat;
                    }
                }
                });
            });
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(theme::PANEL).inner_margin(egui::Margin::symmetric(6, 4)))
            .show(ui, |ui| {
                // Wraps rather than overlapping when even the compact header doesn't fit.
                ui.horizontal_wrapped(|ui| {
                    let label = if compact { icons::GLOBE.to_string() } else { theme::with_icon(icons::GLOBE, "Get more assets") };
                    let browse = egui::Button::new(egui::RichText::new(label).color(Color32::WHITE)).fill(theme::ACCENT.gamma_multiply(0.55));
                    if ui.add(browse).on_hover_text("Get more assets: thousands of free materials from Poly Haven and ambientCG, and graphs shared by the community").clicked() {
                        self.library.open = true;
                    }
                    ui.add(egui::TextEdit::singleline(&mut self.shelf.search).hint_text("Search…").desired_width(if compact { 110.0 } else { 200.0 }));
                    let hint = match self.shelf.cat {
                        ShelfCat::SmartMaterials => "Double-click or drag onto a view to add a smart material (a folder of layers). Save your own: right-click a folder in Layers → Save as smart material.",
                        ShelfCat::Materials => "Double-click or drag onto a view to add as a fill layer. Right-click for more.",
                        ShelfCat::Generators => "Double-click or drag onto a view to mask the selected layer.",
                        ShelfCat::Brushes => "Double-click or drag onto a view to paint with this brush tip.",
                        ShelfCat::Textures => "Double-click or drag to add as a height fill layer. Right-click to put it on the selected fill layer.",
                        ShelfCat::Graphs => "Material graphs you saved or downloaded. Double-click or drag onto a view to add as a graph fill layer. Right-click to share.",
                        ShelfCat::Downloaded => "Free materials from the asset library. Double-click or drag onto a view to add as a fill layer.",
                    };
                    if matches!(self.shelf.cat, ShelfCat::SmartMaterials | ShelfCat::Materials | ShelfCat::Downloaded) {
                        let text = self.shelf.kind.map_or("All types", |k| k.name());
                        egui::ComboBox::from_id_salt("shelf type").selected_text(text).width(if compact { 96.0 } else { 130.0 }).show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.shelf.kind, None, "All types").on_hover_text("Everything, grouped by type");
                            for k in presets::MaterialType::ALL {
                                ui.selectable_value(&mut self.shelf.kind, Some(k), k.name());
                            }
                        });
                    }
                    if compact {
                        // Thumbnail size in steps; the hint stays on the hover.
                        let step = |size: f32, by: f32| (size + by).clamp(ICON_MIN, ICON_MAX);
                        if ui.small_button("-").on_hover_text("Smaller thumbnails (Ctrl+scroll over the shelf)").clicked() {
                            self.shelf.icon_size = step(self.shelf.icon_size, -16.0);
                        }
                        if ui.small_button("+").on_hover_text("Bigger thumbnails (Ctrl+scroll over the shelf)").clicked() {
                            self.shelf.icon_size = step(self.shelf.icon_size, 16.0);
                        }
                        ui.label(egui::RichText::new(icons::INFO).color(theme::TEXT_DIM)).on_hover_text(hint);
                    } else {
                    // The size slider at the right edge first; the hint fits in what's left,
                    // shortened with an ellipsis (whole on hover) instead of running under it.
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add(egui::Slider::new(&mut self.shelf.icon_size, ICON_MIN..=ICON_MAX).show_value(false))
                            .on_hover_text("Thumbnail size (Ctrl+scroll over the shelf)");
                        ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                            ui.add(egui::Label::new(egui::RichText::new(hint).small().color(theme::TEXT_DIM)).truncate()).on_hover_text(hint);
                        });
                    });
                    }
                });
                ui.add_space(2.0);
                // Ctrl+scroll arrives as zoom; it resizes the thumbnails.
                if ui.rect_contains_pointer(ui.max_rect()) {
                    let zoom = ui.input(|i| i.zoom_delta());
                    if zoom != 1.0 {
                        self.shelf.icon_size = (self.shelf.icon_size * zoom).clamp(ICON_MIN, ICON_MAX);
                    }
                }
                let icon = self.shelf.icon_size;
                let search = self.shelf.search.to_lowercase();
                let kind = self.shelf.kind;
                let wanted = |k: presets::MaterialType| kind.is_none_or(|f| f == k);
                // The type's name as a heading of its own when the type changes
                // (all types shown). It takes the full width, so the wrapped
                // layout puts it on a fresh row and starts the next items below it.
                let group = |ui: &mut egui::Ui, last: &mut Option<presets::MaterialType>, k: presets::MaterialType| {
                    if kind.is_none() && *last != Some(k) {
                        let width = ui.max_rect().width();
                        let h = if last.is_some() { 30.0 } else { 20.0 };
                        let (r, _) = ui.allocate_exact_size(vec2(width, h), Sense::hover());
                        let base = r.left_bottom() + vec2(2.0, -5.0);
                        let g = ui.painter().layout_no_wrap(k.name().to_uppercase(), egui::FontId::new(11.0, theme::heading_family()), theme::TEXT_DIM);
                        let line_y = base.y - g.size().y * 0.5;
                        let text_w = g.size().x;
                        ui.painter().galley(pos2(base.x, base.y - g.size().y), g, theme::TEXT_DIM);
                        ui.painter().line_segment(
                            [pos2(base.x + text_w + 8.0, line_y), pos2(r.right() - 4.0, line_y)],
                            Stroke::new(1.0, theme::OUTLINE),
                        );
                        *last = Some(k);
                    }
                };
                let mut share = None;
                let mut share_smart = None;
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing = vec2(4.0, 4.0);
                        match self.shelf.cat {
                            ShelfCat::Graphs => {
                                if self.graphs.shelf.is_empty() {
                                    ui.label(
                                        egui::RichText::new("No saved graphs yet. In the Material Graph window (Ctrl+G), use Save to shelf.")
                                            .color(theme::TEXT_DIM),
                                    );
                                }
                                let fallback = self.shelf.materials.first().map(|t| t.id());
                                for (i, g) in self.graphs.shelf.iter().enumerate() {
                                    if !g.name.to_lowercase().contains(&search) {
                                        continue;
                                    }
                                    let Some(tex) = g.thumb.as_ref().map(|t| t.id()).or(fallback) else { continue };
                                    let resp = Self::swatch(ui, tex, &g.name, icon);
                                    resp.dnd_set_drag_payload(ShelfItem::Graph(i));
                                    if resp.double_clicked() {
                                        self.pending.push(Action::AddOnTop(ShelfItem::Graph(i)));
                                    }
                                    resp.context_menu(|ui| {
                                        if ui.button(theme::with_icon(icons::SHARE_NETWORK, "Share to community…")).clicked() {
                                            share = Some(i);
                                            ui.close();
                                        }
                                    });
                                }
                            }
                            ShelfCat::Downloaded => {
                                self.library.ensure_local(ui.ctx());
                                if self.library.local.is_empty() {
                                    ui.label(
                                        egui::RichText::new("Nothing downloaded yet. Click Get more assets above for thousands of free materials.")
                                            .color(theme::TEXT_DIM),
                                    );
                                }
                                let fallback = self.shelf.materials.first().map(|t| t.id());
                                let kinds: Vec<presets::MaterialType> = self.library.local.iter().map(|m| presets::MaterialType::of(&m.name, None)).collect();
                                let mut order: Vec<usize> = (0..self.library.local.len())
                                    .filter(|&i| self.library.local[i].name.to_lowercase().contains(&search) && wanted(kinds[i]))
                                    .collect();
                                order.sort_by_key(|&i| kinds[i]);
                                let mut last = None;
                                for i in order {
                                    group(ui, &mut last, kinds[i]);
                                    let m = &self.library.local[i];
                                    let thumb = self.library.local_thumbs.get(i).and_then(|t| t.as_ref()).map(|t| t.id());
                                    let Some(tex) = thumb.or(fallback) else { continue };
                                    let resp = Self::swatch(ui, tex, &m.name.clone(), icon).on_hover_text(format!("{}, from {}", m.name, m.source));
                                    resp.dnd_set_drag_payload(ShelfItem::Library(i));
                                    if resp.double_clicked() {
                                        self.pending.push(Action::AddOnTop(ShelfItem::Library(i)));
                                    }
                                    resp.context_menu(|ui| self.library_material_menu(ui, i));
                                }
                            }
                            ShelfCat::SmartMaterials => {
                                // Your own first, marked with a badge.
                                let fallback = self.shelf.smart.first().map(|t| t.id());
                                for i in 0..self.user_smarts.len() {
                                    let s = &self.user_smarts[i];
                                    if !s.name.to_lowercase().contains(&search) || !wanted(presets::MaterialType::of(&s.name, None)) {
                                        continue;
                                    }
                                    let Some(tex) = s.thumb.as_ref().map(|t| t.id()).or(fallback) else { continue };
                                    let resp = Self::swatch(ui, tex, &s.name.clone(), icon);
                                    let at = resp.rect.left_top() + vec2(8.0, 6.0);
                                    let g = ui.painter().layout_no_wrap("MINE".into(), egui::FontId::proportional(8.5), Color32::WHITE);
                                    let r = Rect::from_min_size(at, g.size() + vec2(6.0, 2.0));
                                    ui.painter().rect_filled(r, 0.0, theme::DONATE.gamma_multiply(0.9));
                                    ui.painter().galley(r.min + vec2(3.0, 1.0), g, Color32::WHITE);
                                    resp.dnd_set_drag_payload(ShelfItem::UserSmart(i));
                                    if resp.double_clicked() {
                                        self.pending.push(Action::AddOnTop(ShelfItem::UserSmart(i)));
                                    }
                                    resp.context_menu(|ui| self.user_smart_menu(ui, i, &mut share_smart));
                                }
                                let mut order: Vec<usize> = (0..presets::SMART_MATERIALS.len())
                                    .filter(|&i| presets::SMART_MATERIALS[i].name.to_lowercase().contains(&search) && wanted(presets::SMART_MATERIALS[i].kind()))
                                    .collect();
                                order.sort_by_key(|&i| presets::SMART_MATERIALS[i].kind());
                                // Your own stay first, apart from the groups.
                                let mut last = None;
                                for (i, s) in order.into_iter().map(|i| (i, &presets::SMART_MATERIALS[i])) {
                                    group(ui, &mut last, s.kind());
                                    let tex = self.shelf.smart[i].id();
                                    let resp = Self::swatch(ui, tex, s.name, icon);
                                    if s.is_scanned() {
                                        // Badge: built on a photo-scanned texture set.
                                        let at = resp.rect.left_top() + vec2(8.0, 6.0);
                                        let font = egui::FontId::proportional(8.5);
                                        let g = ui.painter().layout_no_wrap("SCAN".into(), font, Color32::WHITE);
                                        let r = Rect::from_min_size(at, g.size() + vec2(6.0, 2.0));
                                        ui.painter().rect_filled(r, 0.0, theme::ACCENT.gamma_multiply(0.85));
                                        ui.painter().galley(r.min + vec2(3.0, 1.0), g, Color32::WHITE);
                                    }
                                    Self::missing_badge(ui, &resp, s.needs_maps() & self.engine.missing_maps());
                                    resp.dnd_set_drag_payload(ShelfItem::Smart(i));
                                    if resp.double_clicked() {
                                        self.pending.push(Action::AddOnTop(ShelfItem::Smart(i)));
                                    }
                                    resp.context_menu(|ui| {
                                        if ui.button("Add as layer folder").on_hover_text("A folder of fill layers with smart masks; Ctrl+G shows them as a graph you can edit too").clicked() {
                                            self.pending.push(Action::AddOnTop(ShelfItem::Smart(i)));
                                            ui.close();
                                        }
                                        if ui.button("Add as one graph layer").on_hover_text("One layer driven by a graph; its settings as material parameters in the layer properties").clicked() {
                                            self.pending.push(Action::AddOnTop(ShelfItem::SmartGraph(i)));
                                            ui.close();
                                        }
                                    });
                                }
                            }
                            ShelfCat::Materials => {
                                let mut order: Vec<usize> =
                                    (0..presets::MATERIALS.len()).filter(|&i| presets::MATERIALS[i].name.to_lowercase().contains(&search) && wanted(presets::MATERIALS[i].kind())).collect();
                                order.sort_by_key(|&i| presets::MATERIALS[i].kind());
                                let mut last = None;
                                for (i, m) in order.into_iter().map(|i| (i, &presets::MATERIALS[i])) {
                                    group(ui, &mut last, m.kind());
                                    let tex = self.shelf.materials[i].id();
                                    let resp = Self::swatch(ui, tex, m.name, icon);
                                    resp.dnd_set_drag_payload(ShelfItem::Material(i));
                                    if resp.double_clicked() {
                                        self.pending.push(Action::AddOnTop(ShelfItem::Material(i)));
                                    }
                                    resp.context_menu(|ui| {
                                        if ui.button("Add as fill layer").clicked() {
                                            self.pending.push(Action::AddOnTop(ShelfItem::Material(i)));
                                            ui.close();
                                        }
                                        if ui.button("Use as brush material").clicked() {
                                            let f = m.fill();
                                            let mut enabled = self.brush.mat.enabled;
                                            enabled[..3].fill(true);
                                            self.brush.mat = MatValues {
                                                enabled,
                                                height: self.brush.mat.height,
                                                opacity: self.brush.mat.opacity,
                                                emission: self.brush.mat.emission,
                                                ..f.mat
                                            };
                                            self.set_tool(Tool::Paint);
                                            ui.close();
                                        }
                                    });
                                }
                            }
                            ShelfCat::Brushes => {
                                for (i, b) in crate::brushes::BRUSH_ALPHAS.iter().enumerate() {
                                    if !b.name.to_lowercase().contains(&search) {
                                        continue;
                                    }
                                    let resp = Self::swatch(ui, self.shelf.brushes[i].id(), b.name, icon);
                                    if self.brush.tip == Some(i) {
                                        ui.painter().rect_stroke(resp.rect.shrink(2.0), theme::RADIUS, Stroke::new(1.5, theme::ACCENT), egui::StrokeKind::Inside);
                                    }
                                    resp.dnd_set_drag_payload(ShelfItem::Brush(i));
                                    if resp.double_clicked() {
                                        self.pending.push(Action::SetBrushTip(Some(i)));
                                    }
                                }
                            }
                            ShelfCat::Textures => {
                                for (i, t) in crate::textures::TEXTURES.iter().enumerate() {
                                    if !t.name.to_lowercase().contains(&search) && !t.category.to_lowercase().contains(&search) {
                                        continue;
                                    }
                                    let resp = Self::swatch(ui, self.shelf.textures[i].id(), t.name, icon);
                                    resp.dnd_set_drag_payload(ShelfItem::Texture(i));
                                    if resp.double_clicked() {
                                        self.pending.push(Action::AddOnTop(ShelfItem::Texture(i)));
                                    }
                                    resp.context_menu(|ui| {
                                        if ui.button("Add as height fill layer").clicked() {
                                            self.pending.push(Action::AddOnTop(ShelfItem::Texture(i)));
                                            ui.close();
                                        }
                                        ui.separator();
                                        for (ch, label) in [(3, "height"), (0, "color"), (1, "roughness"), (2, "metallic")] {
                                            if ui.button(format!("Use as {label} on the selected fill layer")).clicked() {
                                                self.pending.push(Action::TextureToChannel { texture: i, ch });
                                                ui.close();
                                            }
                                        }
                                    });
                                }
                            }
                            ShelfCat::Generators => {
                                for (i, g) in GenKind::ALL.iter().enumerate() {
                                    if !g.name().to_lowercase().contains(&search) {
                                        continue;
                                    }
                                    let tex = self.shelf.generators[i].id();
                                    let resp = Self::swatch(ui, tex, g.name(), icon);
                                    Self::missing_badge(ui, &resp, g.default_generator().needs_maps() & self.engine.missing_maps());
                                    resp.dnd_set_drag_payload(ShelfItem::Generator(*g));
                                    if resp.double_clicked() {
                                        self.pending.push(Action::AddGenerator(*g));
                                    }
                                }
                            }
                        }
                    });
                });
                if let Some(i) = share {
                    self.share_graph(ui.ctx(), i);
                }
                if let Some(path) = share_smart {
                    self.share_smart(ui.ctx(), &path);
                }
            });
    }

    /// A shelf thumbnail `size` points wide with its name below; hovering
    /// shows it at twice the size.
    pub(super) fn swatch(ui: &mut egui::Ui, tex: egui::TextureId, name: &str, size: f32) -> egui::Response {
        let named = size >= NAMED_MIN;
        let (rect, resp) = ui.allocate_exact_size(vec2(size + 12.0, size + if named { 22.0 } else { 8.0 }), Sense::click_and_drag());
        let p = ui.painter();
        if resp.hovered() {
            p.rect_filled(rect, 0.0, theme::WIDGET_HOVER);
        }
        let img = Rect::from_min_size(rect.min + vec2(6.0, 4.0), vec2(size, size));
        p.image(tex, img, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
        // One line, cut with "…"; the hover shows the full name.
        if named {
            let galley = theme::elided(p, name, egui::FontId::proportional(10.5), theme::TEXT, size + 8.0);
            let pos = pos2(rect.center().x - galley.size().x * 0.5, img.max.y + 3.0);
            p.galley(pos, galley, theme::TEXT);
        }
        if resp.dragged() {
            // The thumbnail follows the pointer while it's dragged to a view.
            if let Some(p) = resp.ctx.pointer_interact_pos() {
                let ghost = Rect::from_center_size(p, vec2(size, size));
                let painter = resp.ctx.layer_painter(egui::LayerId::new(egui::Order::Tooltip, resp.id.with("drag ghost")));
                painter.image(tex, ghost, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::from_white_alpha(220));
            }
            return resp;
        }
        resp.on_hover_ui(|ui| {
            ui.add(egui::Image::new((tex, vec2(size * 2.0, size * 2.0))));
            ui.label(name);
        })
    }
}
