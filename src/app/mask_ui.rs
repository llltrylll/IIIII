//! Mask Properties: smart-mask generators, mask layers, filters and the vertex-color ID mask plumbing.
use super::*;

impl PainterApp {
    /// A smart mask's settings; `salt` keeps several apart on one panel.
    /// `custom` names the custom bake maps the mask can read.
    pub(super) fn generator_ui(
        ui: &mut egui::Ui,
        g: &mut crate::engine::Generator,
        salt: usize,
        custom: &[bake::CustomMap],
        target: (usize, Option<usize>),
        pick_target: &mut Option<(usize, Option<usize>)>,
        vertex_id: &mut vertex_id::VertexIdState,
        info: vertex_id::Info,
    ) -> bool {
        use crate::engine::gin;
        let mut dirty = false;
        labeled(ui, "Preset", |ui| {
            egui::ComboBox::from_id_salt(("gen", salt)).selected_text(g.kind.name()).show_ui(ui, |ui| {
                for k in GenKind::ALL {
                    if ui.selectable_label(g.kind == k, k.name()).clicked() && g.kind != k {
                        *g = new_generator(k, custom);
                        dirty = true;
                    }
                }
            });
        });
        if g.kind == GenKind::CustomMap {
            dirty |= custom_map_picker(ui, g, salt, custom);
        }
        dirty |= slider(ui, "Level", &mut g.level, -1.0..=2.0);
        dirty |= slider(ui, "Contrast", &mut g.contrast, 0.001..=0.6);
        dirty |= slider(ui, "Noise breakup", &mut g.noise, 0.0..=2.0);
        dirty |= slider(ui, "Noise scale", &mut g.scale, 0.2..=20.0);
        dirty |= slider(ui, "Seed", &mut g.seed, 0.0..=100.0);
        dirty |= labeled(ui, "Invert", |ui| ui.checkbox(&mut g.invert, "").changed());
        let uses_width = g.weights[gin::EDGE] != 0.0 || g.weights[gin::CREASE] != 0.0;
        if uses_width {
            dirty |= slider(ui, "Edge width", &mut g.edge_width, 0.002..=0.5);
        }
        if g.weights[gin::ID] != 0.0 {
            dirty |= vertex_id::panel(ui, g, vertex_id, info, pick_target, target);
        }
        dirty | generator_inputs_ui(ui, g, custom)
    }

    pub(super) fn mask_properties(&mut self, ui: &mut egui::Ui, idx: usize) -> bool {
        let mut dirty = false;
        let custom = self.bake_settings.custom.clone();
        let palette = self.vertex_id.palette_for(&self.mesh).clone();
        let id_info = vertex_id::Info { palette: &palette, has_colors: vertex_id::has_colors(&self.mesh), baked_source: self.baked_id_source };
        section(ui, "Mask");
        self.missing_maps_note(ui, idx);
        let Some(mask) = self.engine.layers[idx].mask.as_mut() else {
            ui.label("This layer has no mask.");
            return false;
        };
        if let Some(gid) = mask.graph {
            let name = self.graphs.doc(gid).map_or("(missing)".to_string(), |d| d.graph.name.clone());
            labeled(ui, "Graph", |ui| {
                ui.label(egui::RichText::new(name).color(Color32::WHITE));
                if ui.button("Edit graph").on_hover_text("Ctrl+G").clicked() {
                    self.pending.push(Action::EditGraph);
                }
            });
            ui.label(egui::RichText::new("A material graph renders this mask (its Output set to Mask). Painting on it is replaced when the graph changes.").small().color(theme::TEXT_DIM));
            self.exposed_controls(ui, gid);
            let layers = self.mask_layers_ui(ui, idx);
            return self.mask_filters_ui(ui, idx) | layers;
        }
        let has_paint = mask.paint.is_some();
        if let Some(g) = &mut mask.generator {
            dirty |= Self::generator_ui(ui, g, 0, &custom, (idx, None), &mut self.id_color_pick_target, &mut self.vertex_id, id_info);
        }
        ui.horizontal(|ui| {
            if !has_paint && ui.button("Add paint to mask").clicked() {
                self.engine.ensure_mask_paint(idx);
            }
            if ui.button("Remove mask").clicked() {
                self.pending.push(Action::RemoveMask);
            }
        });
        if self.engine.layers[idx].mask.as_ref().is_some_and(|m| m.generator.is_some()) && !self.engine.has_bake {
            ui.label(egui::RichText::new("Mesh maps are still baking; edge and dirt masks refine when done.").small().color(theme::TEXT_DIM));
        }
        dirty |= self.mask_layers_ui(ui, idx);
        dirty |= self.mask_filters_ui(ui, idx);
        dirty
    }

    /// The mask's stack: generators or flat values blended over the base
    /// mask (painting × smart mask), bottom first.
    pub(super) fn mask_layers_ui(&mut self, ui: &mut egui::Ui, idx: usize) -> bool {
        use crate::engine::{MaskBlend, MaskLayer, MaskSource};
        let Some(mask) = self.engine.layers[idx].mask.as_mut() else { return false };
        section(ui, "Mask layers");
        let mut dirty = false;
        let custom = self.bake_settings.custom.clone();
        let palette = self.vertex_id.palette_for(&self.mesh).clone();
        let id_info = vertex_id::Info { palette: &palette, has_colors: vertex_id::has_colors(&self.mesh), baked_source: self.baked_id_source };
        let (mut remove, mut swap) = (None, None);
        let count = mask.layers.len();
        // Top of the stack first, like the layer list.
        for i in (0..count).rev() {
            let ml = &mut mask.layers[i];
            egui::Frame::new().fill(theme::PANEL_DARK).stroke(Stroke::new(1.0, theme::OUTLINE)).inner_margin(egui::Margin::same(6)).show(ui, |ui| {
                ui.horizontal(|ui| {
                    dirty |= ui.checkbox(&mut ml.enabled, "").on_hover_text("Switch the mask layer on or off").changed();
                    let name = match &ml.source {
                        MaskSource::Generator(g) => g.kind.name(),
                        MaskSource::Fill(_) => "Fill",
                    };
                    ui.label(egui::RichText::new(name).strong().color(if ml.enabled { Color32::WHITE } else { theme::TEXT_DIM }));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button(icons::TRASH).on_hover_text("Remove").clicked() {
                            remove = Some(i);
                        }
                        if ui.add_enabled(i + 1 < count, egui::Button::new("▲").small()).on_hover_text("Move up").clicked() {
                            swap = Some(i);
                        }
                        if ui.add_enabled(i > 0, egui::Button::new("▼").small()).on_hover_text("Move down").clicked() {
                            swap = Some(i - 1);
                        }
                    });
                });
                ui.add_enabled_ui(ml.enabled, |ui| {
                    labeled(ui, "Blend", |ui| {
                        egui::ComboBox::from_id_salt(("mask blend", i)).selected_text(ml.blend.name()).show_ui(ui, |ui| {
                            for b in MaskBlend::ALL {
                                dirty |= ui.selectable_value(&mut ml.blend, b, b.name()).changed();
                            }
                        });
                    });
                    dirty |= slider(ui, "Opacity", &mut ml.opacity, 0.0..=1.0);
                    match &mut ml.source {
                        MaskSource::Generator(g) => dirty |= Self::generator_ui(ui, g, 100 + i, &custom, (idx, Some(i)), &mut self.id_color_pick_target, &mut self.vertex_id, id_info),
                        MaskSource::Fill(v) => dirty |= slider(ui, "Value", v, 0.0..=1.0),
                    }
                });
            });
            ui.add_space(3.0);
        }
        if let Some(i) = remove {
            mask.layers.remove(i);
            dirty = true;
        }
        if let Some(i) = swap {
            mask.layers.swap(i, i + 1);
            dirty = true;
        }
        ui.menu_button(theme::with_icon(icons::STACK, "Add mask layer ▾"), |ui| {
            for k in GenKind::ALL {
                if ui.button(k.name()).clicked() {
                    // On a black mask nothing shows through Multiply: reveal with Add instead.
                    let blend = if mask.black_base && mask.layers.is_empty() { MaskBlend::Add } else { MaskBlend::Multiply };
                    mask.layers.push(MaskLayer { source: MaskSource::Generator(new_generator(k, &custom)), blend, opacity: 1.0, enabled: true });
                    dirty = true;
                    ui.close();
                }
            }
            ui.separator();
            if ui.button("Fill (one value)").clicked() {
                mask.layers.push(MaskLayer { source: MaskSource::Fill(1.0), blend: MaskBlend::Normal, opacity: 1.0, enabled: true });
                dirty = true;
                ui.close();
            }
        });
        if count == 0 {
            ui.label(egui::RichText::new("Stack more smart masks on this one: each blends over the mask below it (Multiply keeps only where both are, Add joins them, Subtract cuts one out).").small().color(theme::TEXT_DIM));
        }
        dirty
    }

    /// The mask's filter stack: add, reorder, switch off, remove and tune
    /// filters that apply after painting, smart mask and graph.
    pub(super) fn mask_filters_ui(&mut self, ui: &mut egui::Ui, idx: usize) -> bool {
        use crate::engine::{FilterKind, MaskFilter};
        let Some(mask) = self.engine.layers[idx].mask.as_mut() else { return false };
        section(ui, "Mask filters");
        let mut dirty = false;
        let (mut remove, mut up) = (None, None);
        let count = mask.filters.len();
        for (i, f) in mask.filters.iter_mut().enumerate() {
            egui::Frame::new().fill(theme::PANEL_DARK).stroke(Stroke::new(1.0, theme::OUTLINE)).inner_margin(egui::Margin::same(6)).show(ui, |ui| {
                ui.horizontal(|ui| {
                    dirty |= ui.checkbox(&mut f.enabled, "").on_hover_text("Switch the filter on or off").changed();
                    ui.label(egui::RichText::new(f.kind.name()).strong().color(if f.enabled { Color32::WHITE } else { theme::TEXT_DIM }))
                        .on_hover_text(f.kind.tip());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button(icons::TRASH).on_hover_text("Remove").clicked() {
                            remove = Some(i);
                        }
                        if ui.add_enabled(i > 0, egui::Button::new("▲").small()).on_hover_text("Apply earlier").clicked() {
                            up = Some(i);
                        }
                        if ui.add_enabled(i + 1 < count, egui::Button::new("▼").small()).on_hover_text("Apply later").clicked() {
                            up = Some(i + 1);
                        }
                    });
                });
                ui.add_enabled_ui(f.enabled, |ui| {
                    for (k, (name, lo, hi)) in f.kind.params().iter().enumerate() {
                        let v = match k {
                            0 => &mut f.a,
                            1 => &mut f.b,
                            _ => &mut f.c,
                        };
                        dirty |= slider(ui, name, v, *lo..=*hi);
                    }
                });
            });
            ui.add_space(3.0);
        }
        if let Some(i) = remove {
            mask.filters.remove(i);
            dirty = true;
        }
        if let Some(i) = up {
            mask.filters.swap(i - 1, i);
            dirty = true;
        }
        ui.menu_button(theme::with_icon(icons::MAGIC_WAND, "Add filter ▾"), |ui| {
            for k in FilterKind::ALL {
                if ui.button(k.name()).on_hover_text(k.tip()).clicked() {
                    mask.filters.push(MaskFilter::new(k));
                    dirty = true;
                    ui.close();
                }
            }
        });
        if mask.filters.is_empty() {
            ui.label(egui::RichText::new("Filters run in order on the finished mask: levels, blur, warp, noise breakup and more.").small().color(theme::TEXT_DIM));
        }
        dirty
    }

    /// What the 3D model shows of the vertex colors: the colors themselves while
    /// sampling (or in the baker with the Vertex colors source), the ID mask's
    /// selection while its panel is open. A viewport-only preview.
    pub(super) fn sync_vertex_color_preview(&mut self, ctx: &egui::Context) {
        self.service_vertex_id_requests(ctx);
        let want = if !vertex_id::has_colors(&self.mesh) {
            None
        } else if self.id_color_pick_target.is_some() || (self.show_bake && self.bake_settings.id_source == IdSource::VertexColors) {
            Some(vertex_id::Shown::Colors)
        } else if self.vertex_id.preview_off {
            None
        } else {
            self.vertex_id.panel.as_ref().map(|(ids, tol)| vertex_id::Shown::Mask(ids.iter().map(|c| c.map(f32::to_bits)).collect(), tol.to_bits()))
        };
        if want == self.vertex_id.shown {
            return;
        }
        match &want {
            Some(shown) => {
                let triangles = vertex_id::overlay(&self.mesh, shown);
                self.overlays.set(&self.rs.device, OverlaySlot::VertexColors, &triangles, &[]);
            }
            None => self.overlays.clear(OverlaySlot::VertexColors),
        }
        self.vertex_id.shown = want;
        ctx.request_repaint();
    }

    /// Things the ID mask panel asked for: baking from vertex colors, and
    /// reading the colors from the mesh file for projects saved without them.
    pub(super) fn service_vertex_id_requests(&mut self, ctx: &egui::Context) {
        if std::mem::take(&mut self.vertex_id.want_rebake) {
            self.bake_settings.id_source = IdSource::VertexColors;
            self.start_bake_id_only(ctx);
            self.status = "Baking the ID map from the vertex colors (the other maps stay as they are).".into();
        }
        if std::mem::take(&mut self.vertex_id.want_import) {
            let Some(path) = rfd::FileDialog::new().add_filter("Mesh (FBX, OBJ)", &crate::import::EXTENSIONS).set_title("The mesh file this project was made from").pick_file() else {
                return;
            };
            match vertex_id::colors_from_file(&self.mesh, &path) {
                Ok(colors) => {
                    let found = vertex_id::palette(&colors).distinct;
                    let mut mesh = (*self.mesh).clone();
                    mesh.tri_vertex_color = Some(colors);
                    self.mesh = Arc::new(mesh);
                    self.vertex_id.shown = None;
                    // The ID map has to be baked again from the new colors.
                    self.vertex_id.want_rebake = true;
                    self.status = format!("Read {found} vertex colors from the file. Baking the maps from them.");
                }
                Err(e) => self.status = e,
            }
            ctx.request_repaint();
        }
    }

    /// Samples the imported per-corner vertex color and assigns it to the active ID mask.
    pub(super) fn pick_vertex_id_color(&mut self, kind: ViewKind, at: Vec2) {
        if kind != ViewKind::V3d {
            self.status = "Vertex ID sampling is available in the 3D view.".into();
            return;
        }
        let size = Vec2::new(self.views.view3d.size[0] as f32, self.views.view3d.size[1] as f32);
        let (origin, direction) = self.cam.ray(at, size);
        let Some(hit) = self.bvh.raycast(origin, direction) else {
            self.status = "Click a colored area on the mesh.".into();
            return;
        };
        let Some(colors) = self.mesh.tri_vertex_color.as_ref().and_then(|all| all.get(hit.tri as usize)) else {
            self.status = "This mesh has no vertex colors. Open the ID mask in Properties and use \"Import vertex colors from file\".".into();
            self.id_color_pick_target = None;
            return;
        };
        let weights = [hit.bary.x, hit.bary.y, hit.bary.z];
        let mut color = [0.0f32; 3];
        for k in 0..3 {
            for ch in 0..3 { color[ch] += colors[k][ch] as f32 / 255.0 * weights[k]; }
        }
        let Some((layer_idx, mask_layer)) = self.id_color_pick_target.take() else { return; };
        let Some(mask) = self.engine.layers.get_mut(layer_idx).and_then(|layer| layer.mask.as_mut()) else {
            self.status = "The target ID mask no longer exists.".into();
            return;
        };
        let generator = match mask_layer {
            None => mask.generator.as_mut(),
            Some(i) => mask.layers.get_mut(i).and_then(|layer| match &mut layer.source {
                crate::engine::MaskSource::Generator(g) => Some(g),
                crate::engine::MaskSource::Fill(_) => None,
            }),
        };
        let Some(generator) = generator else {
            self.status = "The selected mask generator no longer exists.".into();
            return;
        };
        let mut ids = generator.id_colors();
        let rgb = color.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8);
        if vertex_id::selected_index(&ids, rgb).is_none() {
            if ids.len() < crate::engine::MAX_ID_COLORS {
                ids.push(color);
            } else {
                *ids.last_mut().expect("at least one") = color;
            }
            generator.set_id_colors(&ids);
        }
        generator.id_tolerance = generator.id_tolerance.max(0.015);
        self.engine.mark_dirty();
        self.status = format!("Vertex ID #{:02X}{:02X}{:02X} is selected.", rgb[0], rgb[1], rgb[2]);
    }
}
