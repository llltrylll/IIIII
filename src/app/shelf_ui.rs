//! The Shelf: materials, brushes, smart masks and textures to browse and drop in.
use super::*;

#[derive(Clone, Copy)]
enum ShelfTile {
    Graph(usize),
    Library(usize),
    UserSmart(usize),
    Smart(usize),
    Material(usize),
    Brush(usize),
    Texture(usize),
    Generator(usize),
}

impl ShelfTile {
    fn id_salt(self) -> (&'static str, usize) {
        match self {
            Self::Graph(i) => ("graph", i),
            Self::Library(i) => ("library", i),
            Self::UserSmart(i) => ("user-smart", i),
            Self::Smart(i) => ("smart", i),
            Self::Material(i) => ("material", i),
            Self::Brush(i) => ("brush", i),
            Self::Texture(i) => ("texture", i),
            Self::Generator(i) => ("generator", i),
        }
    }
}

enum ShelfRow {
    Heading { kind: presets::MaterialType, height: f32 },
    Items(Vec<ShelfTile>),
}

impl ShelfRow {
    fn height(&self, tile_height: f32) -> f32 {
        match self {
            Self::Heading { height, .. } => *height,
            Self::Items(_) => tile_height,
        }
    }
}

fn flush_shelf_row(rows: &mut Vec<ShelfRow>, current: &mut Vec<ShelfTile>) {
    if !current.is_empty() {
        rows.push(ShelfRow::Items(std::mem::take(current)));
    }
}

fn push_shelf_tile(
    rows: &mut Vec<ShelfRow>,
    current: &mut Vec<ShelfTile>,
    tile: ShelfTile,
    columns: usize,
) {
    current.push(tile);
    if current.len() >= columns {
        flush_shelf_row(rows, current);
    }
}

fn push_shelf_group(
    rows: &mut Vec<ShelfRow>,
    current: &mut Vec<ShelfTile>,
    last: &mut Option<presets::MaterialType>,
    kind: presets::MaterialType,
    enabled: bool,
) {
    if enabled && *last != Some(kind) {
        flush_shelf_row(rows, current);
        let height = if last.is_some() { 30.0 } else { 20.0 };
        rows.push(ShelfRow::Heading { kind, height });
        *last = Some(kind);
    }
}

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
                let matches_search = |name: &str| search.is_empty() || name.to_lowercase().contains(&search);
                let kind = self.shelf.kind;
                let cat = self.shelf.cat;
                let wanted = |k: presets::MaterialType| kind.is_none_or(|filter| filter == k);

                // Build a compact row index. The ScrollArea below virtualizes
                // the rows, so off-screen entries never create widgets or IDs.
                let spacing = 4.0;
                let tile_width = icon + 12.0;
                let named = icon >= NAMED_MIN;
                let tile_height = icon + if named { 22.0 } else { 8.0 };
                let columns = (((ui.available_width() + spacing) / (tile_width + spacing)).floor() as usize).max(1);
                let mut rows = Vec::new();
                let mut current = Vec::with_capacity(columns);
                let mut last_group = None;

                match cat {
                    ShelfCat::Graphs => {
                        for (i, graph) in self.graphs.shelf.iter().enumerate() {
                            if matches_search(&graph.name) {
                                push_shelf_tile(&mut rows, &mut current, ShelfTile::Graph(i), columns);
                            }
                        }
                    }
                    ShelfCat::Downloaded => {
                        self.library.ensure_local(ui.ctx());
                        let kinds: Vec<presets::MaterialType> = self.library.local.iter()
                            .map(|material| presets::MaterialType::of(&material.name, None))
                            .collect();
                        let mut order: Vec<usize> = (0..self.library.local.len())
                            .filter(|&i| matches_search(&self.library.local[i].name) && wanted(kinds[i]))
                            .collect();
                        order.sort_by_key(|&i| kinds[i]);
                        for i in order {
                            push_shelf_group(&mut rows, &mut current, &mut last_group, kinds[i], kind.is_none());
                            push_shelf_tile(&mut rows, &mut current, ShelfTile::Library(i), columns);
                        }
                    }
                    ShelfCat::SmartMaterials => {
                        // User materials stay before the built-in groups.
                        for (i, smart) in self.user_smarts.iter().enumerate() {
                            if matches_search(&smart.name) && wanted(presets::MaterialType::of(&smart.name, None)) {
                                push_shelf_tile(&mut rows, &mut current, ShelfTile::UserSmart(i), columns);
                            }
                        }
                        let mut order: Vec<usize> = (0..presets::SMART_MATERIALS.len())
                            .filter(|&i| matches_search(presets::SMART_MATERIALS[i].name)
                                && wanted(presets::SMART_MATERIALS[i].kind()))
                            .collect();
                        order.sort_by_key(|&i| presets::SMART_MATERIALS[i].kind());
                        for i in order {
                            let smart = &presets::SMART_MATERIALS[i];
                            push_shelf_group(&mut rows, &mut current, &mut last_group, smart.kind(), kind.is_none());
                            push_shelf_tile(&mut rows, &mut current, ShelfTile::Smart(i), columns);
                        }
                    }
                    ShelfCat::Materials => {
                        let mut order: Vec<usize> = (0..presets::MATERIALS.len())
                            .filter(|&i| matches_search(presets::MATERIALS[i].name)
                                && wanted(presets::MATERIALS[i].kind()))
                            .collect();
                        order.sort_by_key(|&i| presets::MATERIALS[i].kind());
                        for i in order {
                            let material = &presets::MATERIALS[i];
                            push_shelf_group(&mut rows, &mut current, &mut last_group, material.kind(), kind.is_none());
                            push_shelf_tile(&mut rows, &mut current, ShelfTile::Material(i), columns);
                        }
                    }
                    ShelfCat::Brushes => {
                        for (i, brush) in crate::brushes::BRUSH_ALPHAS.iter().enumerate() {
                            if matches_search(brush.name) {
                                push_shelf_tile(&mut rows, &mut current, ShelfTile::Brush(i), columns);
                            }
                        }
                    }
                    ShelfCat::Textures => {
                        for (i, texture) in crate::textures::TEXTURES.iter().enumerate() {
                            if matches_search(texture.name) || matches_search(texture.category) {
                                push_shelf_tile(&mut rows, &mut current, ShelfTile::Texture(i), columns);
                            }
                        }
                    }
                    ShelfCat::Generators => {
                        for (i, generator) in GenKind::ALL.iter().enumerate() {
                            if matches_search(generator.name()) {
                                push_shelf_tile(&mut rows, &mut current, ShelfTile::Generator(i), columns);
                            }
                        }
                    }
                }
                flush_shelf_row(&mut rows, &mut current);

                if cat == ShelfCat::Graphs && self.graphs.shelf.is_empty() {
                    ui.label(egui::RichText::new("No saved graphs yet. In the Material Graph window (Ctrl+G), use Save to shelf.")
                        .color(theme::TEXT_DIM));
                } else if cat == ShelfCat::Downloaded && self.library.local.is_empty() {
                    ui.label(egui::RichText::new("Nothing downloaded yet. Click Get more assets above for thousands of free materials.")
                        .color(theme::TEXT_DIM));
                }

                let total_height = rows.iter().map(|row| row.height(tile_height) + spacing).sum::<f32>()
                    - if rows.is_empty() { 0.0 } else { spacing };
                let fallback_material = self.shelf.materials.first().map(|texture| texture.id());
                let fallback_smart = self.shelf.smart.first().map(|texture| texture.id());
                let mut share = None;
                let mut share_smart = None;

                egui::ScrollArea::vertical().auto_shrink([false, false]).show_viewport(ui, |content_ui, viewport| {
                    content_ui.set_height(total_height.max(0.0));
                    let max_rect = content_ui.max_rect();
                    let clip_rect = content_ui.clip_rect();
                    let left = max_rect.left();
                    let top = max_rect.top();
                    let width = max_rect.width();
                    let mut y = 0.0;

                    for row in &rows {
                        let row_height = row.height(tile_height);
                        let relative = Rect::from_min_size(pos2(0.0, y), vec2(width, row_height));
                        if viewport.intersects(relative) {
                            match row {
                                ShelfRow::Heading { kind, .. } => {
                                    let rect = Rect::from_min_size(pos2(left, top + y), vec2(width, row_height));
                                    if clip_rect.intersects(rect) {
                                        let base = rect.left_bottom() + vec2(2.0, -5.0);
                                        let galley = content_ui.painter().layout_no_wrap(
                                            kind.name().to_uppercase(),
                                            egui::FontId::new(11.0, theme::heading_family()),
                                            theme::TEXT_DIM,
                                        );
                                        let line_y = base.y - galley.size().y * 0.5;
                                        let text_width = galley.size().x;
                                        content_ui.painter().galley(
                                            pos2(base.x, base.y - galley.size().y),
                                            galley,
                                            theme::TEXT_DIM,
                                        );
                                        content_ui.painter().line_segment(
                                            [pos2(base.x + text_width + 8.0, line_y), pos2(rect.right() - 4.0, line_y)],
                                            Stroke::new(1.0, theme::OUTLINE),
                                        );
                                    }
                                }
                                ShelfRow::Items(items) => {
                                    for (column, tile) in items.iter().copied().enumerate() {
                                        let rect = Rect::from_min_size(
                                            pos2(left + column as f32 * (tile_width + spacing), top + y),
                                            vec2(tile_width, tile_height),
                                        );
                                        if !clip_rect.intersects(rect) {
                                            continue;
                                        }
                                        let id = content_ui.scope_id().with(("shelf-swatch", tile.id_salt()));
                                        match tile {
                                            ShelfTile::Graph(i) => {
                                                let graph = &self.graphs.shelf[i];
                                                let Some(texture) = graph.thumb.as_ref().map(|thumb| thumb.id()).or(fallback_material) else { continue };
                                                let response = Self::swatch_at(content_ui, id, rect, texture, &graph.name, icon);
                                                if response.double_clicked() {
                                                    self.pending.push(Action::AddOnTop(ShelfItem::Graph(i)));
                                                }
                                                response.context_menu(|ui| {
                                                    if ui.button(theme::with_icon(icons::SHARE_NETWORK, "Share to community…")).clicked() {
                                                        share = Some(i);
                                                        ui.close();
                                                    }
                                                });
                                                if clip_rect.intersects(response.rect) {
                                                    response.dnd_set_drag_payload(ShelfItem::Graph(i));
                                                }
                                            }
                                            ShelfTile::Library(i) => {
                                                let material = &self.library.local[i];
                                                let texture = self.library.local_thumbs.get(i).and_then(|thumb| thumb.as_ref())
                                                    .map(|thumb| thumb.id()).or(fallback_material);
                                                let Some(texture) = texture else { continue };
                                                let response = Self::swatch_at(content_ui, id, rect, texture, &material.name, icon);
                                                if clip_rect.intersects(response.rect) {
                                                    let response = response.on_hover_text(format!("{}, from {}", material.name, material.source));
                                                    response.dnd_set_drag_payload(ShelfItem::Library(i));
                                                    if response.double_clicked() {
                                                        self.pending.push(Action::AddOnTop(ShelfItem::Library(i)));
                                                    }
                                                    response.context_menu(|ui| self.library_material_menu(ui, i));
                                                }
                                            }
                                            ShelfTile::UserSmart(i) => {
                                                let smart = &self.user_smarts[i];
                                                let texture = smart.thumb.as_ref().map(|thumb| thumb.id()).or(fallback_smart);
                                                let Some(texture) = texture else { continue };
                                                let response = Self::swatch_at(content_ui, id, rect, texture, &smart.name, icon);
                                                if clip_rect.intersects(response.rect) {
                                                    let badge = content_ui.painter().layout_no_wrap(
                                                        "MINE".into(),
                                                        egui::FontId::proportional(8.5),
                                                        Color32::WHITE,
                                                    );
                                                    let at = response.rect.left_top() + vec2(8.0, 6.0);
                                                    let badge_rect = Rect::from_min_size(at, badge.size() + vec2(6.0, 2.0));
                                                    content_ui.painter().rect_filled(badge_rect, 0.0, theme::DONATE.gamma_multiply(0.9));
                                                    content_ui.painter().galley(badge_rect.min + vec2(3.0, 1.0), badge, Color32::WHITE);
                                                    response.dnd_set_drag_payload(ShelfItem::UserSmart(i));
                                                    if response.double_clicked() {
                                                        self.pending.push(Action::AddOnTop(ShelfItem::UserSmart(i)));
                                                    }
                                                    response.context_menu(|ui| self.user_smart_menu(ui, i, &mut share_smart));
                                                }
                                            }
                                            ShelfTile::Smart(i) => {
                                                let smart = &presets::SMART_MATERIALS[i];
                                                let texture = self.shelf.smart[i].id();
                                                let response = Self::swatch_at(content_ui, id, rect, texture, smart.name, icon);
                                                if smart.is_scanned() {
                                                    let badge = content_ui.painter().layout_no_wrap(
                                                        "SCAN".into(),
                                                        egui::FontId::proportional(8.5),
                                                        Color32::WHITE,
                                                    );
                                                    let at = response.rect.left_top() + vec2(8.0, 6.0);
                                                    let badge_rect = Rect::from_min_size(at, badge.size() + vec2(6.0, 2.0));
                                                    content_ui.painter().rect_filled(badge_rect, 0.0, theme::ACCENT.gamma_multiply(0.85));
                                                    content_ui.painter().galley(badge_rect.min + vec2(3.0, 1.0), badge, Color32::WHITE);
                                                }
                                                Self::missing_badge(content_ui, &response, smart.needs_maps() & self.engine.missing_maps());
                                                if clip_rect.intersects(response.rect) {
                                                    response.dnd_set_drag_payload(ShelfItem::Smart(i));
                                                    if response.double_clicked() {
                                                        self.pending.push(Action::AddOnTop(ShelfItem::Smart(i)));
                                                    }
                                                    response.context_menu(|ui| {
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
                                            ShelfTile::Material(i) => {
                                                let material = &presets::MATERIALS[i];
                                                let texture = self.shelf.materials[i].id();
                                                let response = Self::swatch_at(content_ui, id, rect, texture, material.name, icon);
                                                if clip_rect.intersects(response.rect) {
                                                    response.dnd_set_drag_payload(ShelfItem::Material(i));
                                                    if response.double_clicked() {
                                                        self.pending.push(Action::AddOnTop(ShelfItem::Material(i)));
                                                    }
                                                    response.context_menu(|ui| {
                                                        if ui.button("Add as fill layer").clicked() {
                                                            self.pending.push(Action::AddOnTop(ShelfItem::Material(i)));
                                                            ui.close();
                                                        }
                                                        if ui.button("Use as brush material").clicked() {
                                                            let fill = material.fill();
                                                            let mut enabled = self.brush.mat.enabled;
                                                            enabled[..3].fill(true);
                                                            self.brush.mat = MatValues {
                                                                enabled,
                                                                height: self.brush.mat.height,
                                                                opacity: self.brush.mat.opacity,
                                                                emission: self.brush.mat.emission,
                                                                ..fill.mat
                                                            };
                                                            self.set_tool(Tool::Paint);
                                                            ui.close();
                                                        }
                                                    });
                                                }
                                            }
                                            ShelfTile::Brush(i) => {
                                                let brush = &crate::brushes::BRUSH_ALPHAS[i];
                                                let response = Self::swatch_at(content_ui, id, rect, self.shelf.brushes[i].id(), brush.name, icon);
                                                if self.brush.tip == Some(i) {
                                                    content_ui.painter().rect_stroke(
                                                        response.rect.shrink(2.0),
                                                        theme::RADIUS,
                                                        Stroke::new(1.5, theme::ACCENT),
                                                        egui::StrokeKind::Inside,
                                                    );
                                                }
                                                if clip_rect.intersects(response.rect) {
                                                    response.dnd_set_drag_payload(ShelfItem::Brush(i));
                                                    if response.double_clicked() {
                                                        self.pending.push(Action::SetBrushTip(Some(i)));
                                                    }
                                                }
                                            }
                                            ShelfTile::Texture(i) => {
                                                let texture_def = &crate::textures::TEXTURES[i];
                                                let response = Self::swatch_at(content_ui, id, rect, self.shelf.textures[i].id(), texture_def.name, icon);
                                                if clip_rect.intersects(response.rect) {
                                                    response.dnd_set_drag_payload(ShelfItem::Texture(i));
                                                    if response.double_clicked() {
                                                        self.pending.push(Action::AddOnTop(ShelfItem::Texture(i)));
                                                    }
                                                    response.context_menu(|ui| {
                                                        if ui.button("Add as height fill layer").clicked() {
                                                            self.pending.push(Action::AddOnTop(ShelfItem::Texture(i)));
                                                            ui.close();
                                                        }
                                                        ui.separator();
                                                        for (channel, label) in [(3, "height"), (0, "color"), (1, "roughness"), (2, "metallic")] {
                                                            if ui.button(format!("Use as {label} on the selected fill layer")).clicked() {
                                                                self.pending.push(Action::TextureToChannel { texture: i, ch: channel });
                                                                ui.close();
                                                            }
                                                        }
                                                    });
                                                }
                                            }
                                            ShelfTile::Generator(i) => {
                                                let generator = GenKind::ALL[i];
                                                let response = Self::swatch_at(content_ui, id, rect, self.shelf.generators[i].id(), generator.name(), icon);
                                                Self::missing_badge(content_ui, &response, generator.default_generator().needs_maps() & self.engine.missing_maps());
                                                if clip_rect.intersects(response.rect) {
                                                    response.dnd_set_drag_payload(ShelfItem::Generator(generator));
                                                    if response.double_clicked() {
                                                        self.pending.push(Action::AddGenerator(generator));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        y += row_height + spacing;
                    }
                });

                if let Some(i) = share {
                    self.share_graph(ui.ctx(), i);
                }
                if let Some(path) = share_smart {
                    self.share_smart(ui.ctx(), &path);
                }
            });
    }

    fn swatch_at(
        ui: &mut egui::Ui,
        id: egui::Id,
        rect: Rect,
        tex: egui::TextureId,
        name: &str,
        size: f32,
    ) -> egui::Response {
        let named = size >= NAMED_MIN;
        let visible = ui.is_rect_visible(rect);
        let response = ui.interact(rect, id, if visible { Sense::click_and_drag() } else { Sense::hover() });
        if !visible {
            return response;
        }
        let painter = ui.painter();
        if response.hovered() {
            painter.rect_filled(rect, 0.0, theme::WIDGET_HOVER);
        }
        let image_rect = Rect::from_min_size(rect.min + vec2(6.0, 4.0), vec2(size, size));
        painter.image(tex, image_rect, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
        if named {
            let galley = theme::elided(painter, name, egui::FontId::proportional(10.5), theme::TEXT, size + 8.0);
            let pos = pos2(rect.center().x - galley.size().x * 0.5, image_rect.max.y + 3.0);
            painter.galley(pos, galley, theme::TEXT);
        }
        if response.dragged() {
            if let Some(pointer) = response.ctx.pointer_interact_pos() {
                let ghost = Rect::from_center_size(pointer, vec2(size, size));
                let drag_painter = response.ctx.layer_painter(egui::LayerId::new(egui::Order::Tooltip, response.id.with("drag ghost")));
                drag_painter.image(tex, ghost, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::from_white_alpha(220));
            }
            return response;
        }
        response.on_hover_ui(|ui| {
            ui.add(egui::Image::new((tex, vec2(size * 2.0, size * 2.0))));
            ui.label(name);
        })
    }
}
