//! The asset library window: free CC0 materials from Poly Haven and
//! ambientCG, and the community library of shared graphs and nodes. Plus the
//! shelf's Downloaded category and sharing a graph with the community.

use super::export_ui::open_in_file_browser;
use super::{PainterApp, theme};
use crate::engine::{FillParams, Projection};
use crate::undo::StrokeTarget;
use crate::icons;
use crate::library::{self, Entry, Event, Job, Kind, LocalMaterial, Net, Res, Source};
use eframe::egui;
use egui::{Color32, Rect, Sense, vec2};
use std::collections::{HashMap, HashSet};

enum Catalog {
    Loading,
    Ready(Vec<Entry>),
    Failed(String),
}

enum Thumb {
    Pending,
    Ready(egui::TextureHandle),
    Missing,
}

/// Thumbnails requested at once; more wait until these arrive.
const MAX_THUMBS_IN_FLIGHT: usize = 24;
/// Smallest card width; cards stretch to fill each row.
const CARD: f32 = 150.0;

pub(super) struct LibraryUi {
    pub open: bool,
    net: Option<Net>,
    tab: Source,
    search: String,
    res: Res,
    catalogs: HashMap<Source, Catalog>,
    thumbs: HashMap<String, Thumb>,
    in_flight: usize,
    /// Downloads in progress: key -> (done, total).
    downloads: HashMap<String, (u64, u64)>,
    /// Community entries installed this session.
    installed: HashSet<String>,
    /// Downloaded materials, for the shelf.
    pub local: Vec<LocalMaterial>,
    pub local_thumbs: Vec<Option<egui::TextureHandle>>,
    local_loaded: bool,
}

impl Default for LibraryUi {
    fn default() -> Self {
        LibraryUi {
            open: false,
            net: None,
            tab: Source::PolyHaven,
            search: String::new(),
            res: Res::K2,
            catalogs: HashMap::new(),
            thumbs: HashMap::new(),
            in_flight: 0,
            downloads: HashMap::new(),
            installed: HashSet::new(),
            local: Vec::new(),
            local_thumbs: Vec::new(),
            local_loaded: false,
        }
    }
}

impl LibraryUi {
    /// Rereads the downloaded materials (and their shelf thumbnails).
    pub fn reload_local(&mut self, ctx: &egui::Context) {
        self.local = library::local_materials();
        self.local_thumbs = self
            .local
            .iter()
            .map(|m| {
                let img = image::open(m.dir.join("thumb.png")).ok()?.into_rgba8();
                let size = [img.width() as usize, img.height() as usize];
                Some(ctx.load_texture(format!("library {}", m.key), egui::ColorImage::from_rgba_unmultiplied(size, &img), Default::default()))
            })
            .collect();
        self.local_loaded = true;
    }

    pub fn ensure_local(&mut self, ctx: &egui::Context) {
        if !self.local_loaded {
            self.reload_local(ctx);
        }
    }

    /// The open tab's catalog has arrived (offscreen UI tests).
    #[cfg(test)]
    pub fn test_ready(&self) -> bool {
        matches!(self.catalogs.get(&self.tab), Some(Catalog::Ready(_)))
    }

    fn net(&mut self, ctx: &egui::Context) -> &Net {
        self.net.get_or_insert_with(|| Net::new(ctx))
    }

    fn request_catalog(&mut self, ctx: &egui::Context, source: Source) {
        self.catalogs.insert(source, Catalog::Loading);
        self.net(ctx).send(Job::Catalog(source));
    }

    fn is_installed(&self, e: &Entry) -> bool {
        match e.kind {
            Kind::Material => self.local.iter().any(|m| m.key == e.key()),
            _ => self.installed.contains(&e.key()) || library::installed_path(e).is_some_and(|p| p.is_file()),
        }
    }
}

impl PainterApp {
    /// Applies finished catalog, thumbnail and download jobs.
    pub(super) fn poll_library(&mut self, ctx: &egui::Context) {
        let Some(net) = &self.library.net else { return };
        let events: Vec<Event> = net.events.try_iter().collect();
        for event in events {
            match event {
                Event::Catalog(source, result) => {
                    let state = match result {
                        Ok(entries) => Catalog::Ready(entries),
                        Err(e) => Catalog::Failed(e),
                    };
                    self.library.catalogs.insert(source, state);
                }
                Event::Thumb { key, image } => {
                    self.library.in_flight = self.library.in_flight.saturating_sub(1);
                    let thumb = match image {
                        Some(img) => Thumb::Ready(ctx.load_texture(format!("thumb {key}"), img, Default::default())),
                        None => Thumb::Missing,
                    };
                    self.library.thumbs.insert(key, thumb);
                }
                Event::Progress { key, done, total } => {
                    self.library.downloads.insert(key, (done, total));
                }
                Event::Downloaded { key, kind, result } => {
                    self.library.downloads.remove(&key);
                    match result {
                        Ok(_) => {
                            self.library.installed.insert(key);
                            match kind {
                                Kind::Material => {
                                    self.library.reload_local(ctx);
                                    self.status = "Downloaded. It's on the shelf under Downloaded: drag it onto the mesh.".into();
                                }
                                Kind::Graph => {
                                    self.graphs.reload_shelf(ctx);
                                    self.status = "Downloaded. It's on the shelf under Graphs.".into();
                                }
                                Kind::Node => {
                                    self.status = "Downloaded. The node is in the Material Designer's add-node menu.".into();
                                }
                                Kind::Smart => {
                                    self.reload_user_smarts(ctx);
                                    self.status = "Downloaded. It's first in the shelf's Smart materials.".into();
                                }
                            }
                        }
                        Err(e) => self.status = format!("Download failed: {e}"),
                    }
                }
            }
        }
    }

    pub(super) fn library_window(&mut self, ctx: &egui::Context) {
        if !self.library.open {
            return;
        }
        self.library.ensure_local(ctx);
        let tab = self.library.tab;
        if !self.library.catalogs.contains_key(&tab) {
            self.library.request_catalog(ctx, tab);
        }
        let mut open = true;
        egui::Window::new(theme::with_icon(icons::GLOBE, "Asset library"))
            .open(&mut open)
            .default_size([980.0, 660.0])
            .min_size([520.0, 360.0])
            .resizable(true)
            .collapsible(false)
            .show(ctx, |ui| self.library_contents(ui, ctx));
        self.library.open &= open;
    }

    fn library_contents(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.horizontal(|ui| {
            for s in Source::ALL {
                let label = match s {
                    Source::Community => theme::with_icon(icons::USERS_THREE, s.name()),
                    _ => s.name().to_string(),
                };
                if theme::pill(ui, &label, self.library.tab == s).clicked() && self.library.tab != s {
                    self.library.tab = s;
                    if !self.library.catalogs.contains_key(&s) {
                        self.library.request_catalog(ctx, s);
                    }
                }
            }
            ui.add_space(12.0);
            ui.add(egui::TextEdit::singleline(&mut self.library.search).hint_text("Search…").desired_width(220.0));
            if self.library.tab != Source::Community {
                ui.add_space(8.0);
                ui.label(egui::RichText::new("Resolution").color(theme::TEXT_DIM));
                egui::ComboBox::from_id_salt("library res").selected_text(self.library.res.label()).width(60.0).show_ui(ui, |ui| {
                    for r in Res::ALL {
                        ui.selectable_value(&mut self.library.res, r, r.label());
                    }
                });
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.hyperlink_to(theme::with_icon(icons::GLOBE, "Website"), self.library.tab.site());
            });
        });
        let note = match self.library.tab {
            Source::Community => "Smart materials, graphs and nodes made by Material Painter users. Share yours: right-click it on the shelf → Share to community.",
            _ => "Free CC0 (public domain) materials: use them for anything, no credit needed. Click to download; they go on the shelf under Downloaded.",
        };
        ui.label(egui::RichText::new(note).small().color(theme::TEXT_DIM));
        ui.separator();

        let tab = self.library.tab;
        let search = self.library.search.to_lowercase();
        let entries: Vec<Entry> = match self.library.catalogs.get(&tab) {
            Some(Catalog::Ready(all)) => {
                all.iter().filter(|e| search.split_whitespace().all(|w| e.search.contains(w))).cloned().collect()
            }
            Some(Catalog::Failed(e)) => {
                let e = e.clone();
                ui.vertical_centered(|ui| {
                    ui.add_space(40.0);
                    let msg = if tab == Source::Community && e.contains("404") {
                        "The community library is empty for now. Be the first to share a graph!".to_string()
                    } else {
                        format!("Couldn't load the catalog. Check your internet connection.\n{e}")
                    };
                    ui.label(egui::RichText::new(msg).color(theme::TEXT_DIM));
                    if ui.button("Try again").clicked() {
                        self.library.request_catalog(ctx, tab);
                    }
                });
                return;
            }
            _ => {
                ui.vertical_centered(|ui| {
                    ui.add_space(40.0);
                    ui.spinner();
                    ui.label(egui::RichText::new(format!("Loading {}…", tab.name())).color(theme::TEXT_DIM));
                });
                return;
            }
        };
        if entries.is_empty() {
            let msg = if tab == Source::Community && search.is_empty() {
                "The community library is empty for now. Be the first: right-click a smart material or graph on the shelf → Share to community."
            } else {
                "Nothing matches your search."
            };
            ui.label(egui::RichText::new(msg).color(theme::TEXT_DIM));
            return;
        }
        let spacing = 8.0;
        let avail = ui.available_width() - 14.0; // room for the scroll bar
        let cols = (((avail + spacing) / (CARD + spacing)).floor() as usize).max(1);
        let card = ((avail - spacing * (cols - 1) as f32) / cols as f32).floor();
        let rows = entries.len().div_ceil(cols);
        let row_h = card + 46.0 + spacing;
        egui::ScrollArea::vertical().auto_shrink([false, false]).show_rows(ui, row_h, rows, |ui, range| {
            for row in range {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = spacing;
                    for e in entries.iter().skip(row * cols).take(cols) {
                        self.library_card(ui, ctx, e, card);
                    }
                });
                ui.add_space(spacing);
            }
        });
    }

    fn library_card(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, e: &Entry, card: f32) {
        let (rect, resp) = ui.allocate_exact_size(vec2(card, card + 46.0), Sense::click());
        let key = e.key();
        if !self.library.thumbs.contains_key(&key) && self.library.in_flight < MAX_THUMBS_IN_FLIGHT {
            self.library.thumbs.insert(key.clone(), Thumb::Pending);
            self.library.in_flight += 1;
            let job = Job::Thumb { key: key.clone(), url: e.thumb_url.clone() };
            self.library.net(ctx).send(job);
        }
        let installed = self.library.is_installed(e);
        let progress = self.library.downloads.get(&key).copied();
        let p = ui.painter();
        let hovered = resp.hovered();
        p.rect_filled(rect, 0.0, if hovered { theme::WIDGET_HOVER } else { theme::PANEL_DARK });
        let img_rect = Rect::from_min_size(rect.min, vec2(card, card));
        match self.library.thumbs.get(&key) {
            Some(Thumb::Ready(t)) => {
                p.image(t.id(), img_rect, Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), Color32::WHITE);
            }
            _ => {
                p.rect_filled(img_rect.shrink(1.0), 0.0, theme::PANEL);
            }
        }
        if hovered {
            p.rect_stroke(rect, 0.0, egui::Stroke::new(1.0, theme::ACCENT), egui::StrokeKind::Inside);
        }
        let mut job = egui::text::LayoutJob::simple_singleline(e.name.clone(), egui::FontId::proportional(12.5), Color32::WHITE);
        job.wrap = egui::text::TextWrapping { max_width: card - 12.0, max_rows: 1, break_anywhere: true, overflow_character: Some('…') };
        p.galley(rect.left_top() + vec2(6.0, card + 6.0), ui.fonts_mut(|f| f.layout_job(job)), Color32::WHITE);
        let status_y = rect.bottom() - 13.0;
        if let Some((done, total)) = progress {
            let f = if total > 0 { done as f32 / total as f32 } else { 0.0 };
            let bar = Rect::from_min_size(egui::pos2(rect.left() + 6.0, status_y - 3.0), vec2(card - 12.0, 5.0));
            p.rect_filled(bar, 0.0, theme::PANEL);
            p.rect_filled(Rect::from_min_size(bar.min, vec2(bar.width() * f, bar.height())), 0.0, theme::ACCENT);
        } else {
            let (text, color) = if installed {
                ("✓ On your shelf".to_string(), Color32::from_rgb(110, 200, 120))
            } else if hovered {
                (theme::with_icon(icons::DOWNLOAD_SIMPLE, "Download"), theme::ACCENT)
            } else {
                (format!("{} · {}", kind_label(e.kind), e.author), theme::TEXT_DIM)
            };
            p.text(egui::pos2(rect.left() + 6.0, status_y), egui::Align2::LEFT_CENTER, text, egui::FontId::proportional(11.0), color);
        }
        let tip = if e.description.is_empty() { e.name.clone() } else { format!("{}\n\n{}", e.name, e.description) };
        let resp = resp.on_hover_text(tip);
        if resp.clicked() && !installed && progress.is_none() {
            self.library.downloads.insert(key, (0, 0));
            let job = Job::Download { entry: e.clone(), res: self.library.res };
            self.library.net(ctx).send(job);
        }
    }

    /// Adds a downloaded material as a tri-planar fill layer with its maps.
    pub(super) fn add_library_material(&mut self, i: usize, at: usize, parent: Option<u64>) {
        let Some(m) = self.library.local.get(i) else { return };
        let (name, maps) = (m.name.clone(), m.maps.clone());
        let mut fill = FillParams { projection: Projection::Triplanar, tiling: 1.0, ..FillParams::default() };
        // A material without a metal map is not metal.
        let mut enabled = [true, maps[1].is_some(), true, maps[3].is_some(), false, false, false, false];
        enabled[crate::engine::NORMAL] = maps[crate::engine::NORMAL].is_some();
        fill.mat.enabled = enabled;
        fill.mat.metal = 0.0;
        // With a height image the value is its strength, as for the built-in scans.
        fill.mat.height = 0.3;
        self.undo.begin_group(format!("Add {name}"));
        let idx = self.engine.add_fill_layer(at, parent, name.clone(), fill, &mut self.undo);
        for (ch, path) in maps.iter().enumerate() {
            let Some(path) = path else { continue };
            match self.engine.shared_fill_image(path) {
                Ok(img) => self.engine.set_fill_image(idx, ch, Some(img), &mut self.undo),
                Err(e) => self.status = e,
            }
        }
        self.undo.end_group();
        self.selected = idx;
        self.target = StrokeTarget::Content;
        self.status = format!("Added '{name}'. Change its tiling and projection in the layer properties.");
    }

    /// The shelf's context menu on a downloaded material.
    pub(super) fn library_material_menu(&mut self, ui: &mut egui::Ui, i: usize) {
        let Some(m) = self.library.local.get(i) else { return };
        let (url, dir, name) = (m.url.clone(), m.dir.clone(), m.name.clone());
        if !url.is_empty() && ui.button(theme::with_icon(icons::GLOBE, &format!("Open on {}", m.source))).clicked() {
            ui.ctx().open_url(egui::OpenUrl::new_tab(url));
            ui.close();
        }
        if ui.button("Show files").clicked() {
            open_in_file_browser(&dir);
            ui.close();
        }
        if ui.button("Delete download").clicked() {
            let _ = std::fs::remove_dir_all(&dir);
            self.library.reload_local(ui.ctx());
            self.status = format!("Deleted '{name}' from the library.");
            ui.close();
        }
    }

    /// Packs a shelf graph (with its custom nodes) for the community library
    /// and opens the submission page.
    pub(super) fn share_graph(&mut self, ctx: &egui::Context, i: usize) {
        let Some(g) = self.graphs.shelf.get(i) else { return };
        let (path, name) = (g.path.clone(), g.name.clone());
        let graph = match crate::graph::load_file(&path, &self.graphs.lib) {
            Ok(g) => g,
            Err(e) => {
                self.status = e;
                return;
            }
        };
        let user_nodes: Vec<_> = graph
            .nodes
            .iter()
            .filter(|n| n.def.starts_with("user/"))
            .filter_map(|n| self.graphs.lib.user_path(&n.def))
            .filter(|p| p.is_file())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        let thumb = std::fs::read(path.with_extension("png")).ok();
        self.share_package(ctx, Kind::Graph, &path, thumb, &user_nodes, &name);
    }

    /// Packs a user smart material (with any custom nodes its graphs use) for
    /// the community library and opens the submission page.
    pub(super) fn share_smart(&mut self, ctx: &egui::Context, path: &std::path::Path) {
        let name = path.file_stem().map_or(String::new(), |s| s.to_string_lossy().into_owned());
        let nodes = self.smart_user_nodes(path);
        let thumb = Self::smart_thumb_png(path);
        self.share_package(ctx, Kind::Smart, path, thumb, &nodes, &name);
    }

    fn share_package(&mut self, ctx: &egui::Context, kind: Kind, path: &std::path::Path, thumb: Option<Vec<u8>>, user_nodes: &[std::path::PathBuf], name: &str) {
        match library::package(kind, path, thumb, user_nodes, name) {
            Ok((dir, zip)) => {
                open_in_file_browser(dir.parent().unwrap_or(&dir));
                ctx.open_url(egui::OpenUrl::new_tab(library::submit_url(name)));
                self.status = format!(
                    "Packed '{name}' ({} custom node{}). Attach {} to the submission page that just opened.",
                    user_nodes.len(),
                    if user_nodes.len() == 1 { "" } else { "s" },
                    zip.file_name().map_or(String::new(), |f| f.to_string_lossy().into_owned())
                );
            }
            Err(e) => self.status = format!("Couldn't pack '{name}': {e}"),
        }
    }
}

fn kind_label(kind: Kind) -> &'static str {
    match kind {
        Kind::Material => "Material",
        Kind::Graph => "Graph",
        Kind::Node => "Node",
        Kind::Smart => "Smart material",
    }
}
