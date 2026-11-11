//! User preferences (Edit > Preferences), saved per user next to the
//! recent-projects list. Most apply as soon as they change; a few (GPU,
//! MCP port) on the next start.

use super::{PainterApp, labeled, section};
use crate::bake::Quality;
use crate::environment;
use crate::export::{Depth, Format, Padding};
use crate::theme;
use eframe::egui;
use serde_json::{Value, json};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Gpu {
    /// Let the system choose.
    Auto,
    /// The discrete GPU on laptops with two.
    HighPerformance,
    /// The integrated GPU: cooler and longer battery life.
    LowPower,
}

impl Gpu {
    const ALL: [Gpu; 3] = [Gpu::Auto, Gpu::HighPerformance, Gpu::LowPower];
    fn name(self) -> &'static str {
        match self {
            Gpu::Auto => "Automatic",
            Gpu::HighPerformance => "High performance (discrete GPU)",
            Gpu::LowPower => "Low power (integrated GPU)",
        }
    }
    fn key(self) -> &'static str {
        match self {
            Gpu::Auto => "auto",
            Gpu::HighPerformance => "high_performance",
            Gpu::LowPower => "low_power",
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct Preferences {
    // General
    pub autosave: bool,
    /// Edits between autosaves.
    pub autosave_every: u32,
    pub undo_steps: usize,
    pub recent_count: usize,
    /// Texture resolution of new projects.
    pub default_resolution: u32,
    /// Bake mesh maps as soon as a mesh is loaded.
    pub bake_on_load: bool,
    // Interface
    pub ui_scale: f32,
    pub shelf_icon_size: f32,
    // Viewport
    pub default_hdri: String,
    pub key_light: bool,
    pub fov_deg: f32,
    pub orbit_speed: f32,
    pub zoom_speed: f32,
    pub invert_zoom: bool,
    pub rt_max_samples: u32,
    pub rt_bounces: u32,
    // Painting
    pub brush_size: f32,
    pub brush_hardness: f32,
    pub brush_spacing: f32,
    pub swatches: Vec<[u8; 3]>,
    /// Texels of padding polygon fill adds past UV island borders.
    pub poly_padding: f32,
    // Baking
    pub bake_quality: Quality,
    pub bake_match_by_name: bool,
    // Export
    pub export_folder: String,
    pub export_template: String,
    pub export_format: Format,
    pub export_depth: Depth,
    pub export_size: Option<u32>,
    pub export_padding: Padding,
    pub export_open_folder: bool,
    // AI agents
    pub mcp_on_start: bool,
    /// Show the welcome window when the app starts.
    pub show_welcome: bool,
    pub mcp_port: u16,
    // Performance
    pub gpu: Gpu,
}

impl Default for Preferences {
    fn default() -> Self {
        Self {
            autosave: true,
            autosave_every: 10,
            undo_steps: 64,
            recent_count: 10,
            default_resolution: 2048,
            bake_on_load: true,
            ui_scale: 1.0,
            shelf_icon_size: 72.0,
            default_hdri: "Small Studio".into(),
            key_light: true,
            fov_deg: 38.0,
            orbit_speed: 1.0,
            zoom_speed: 1.0,
            invert_zoom: false,
            rt_max_samples: 512,
            rt_bounces: 4,
            brush_size: 70.0,
            brush_hardness: 0.55,
            brush_spacing: 0.08,
            swatches: Vec::new(),
            poly_padding: 2.0,
            bake_quality: Quality::Standard,
            bake_match_by_name: true,
            export_folder: std::env::current_dir().map(|p| p.join("export").display().to_string()).unwrap_or_default(),
            export_template: "Unreal Engine (packed ORM)".into(),
            export_format: Format::Png,
            export_depth: Depth::U8,
            export_size: None,
            export_padding: Padding::Infinite,
            export_open_folder: false,
            // The agent server is opt-in: it accepts local connections.
            mcp_on_start: false,
            show_welcome: true,
            mcp_port: crate::mcp::DEFAULT_PORT,
            gpu: Gpu::Auto,
        }
    }
}

const FILE: &str = "preferences.json";
pub const RESOLUTIONS: [u32; 6] = [256, 512, 1024, 2048, 4096, 8192];
pub const MAX_SWATCHES:usize=32;
fn swatches_from_json(v:&Value)->Vec<[u8;3]>{
let byte=|x:Option<&Value>|x.and_then(Value::as_u64).map(|n|n.min(255) as u8);
v.as_array().map_or_else(Vec::new,|list|list.iter().filter_map(|c|Some([byte(c.get(0))?,byte(c.get(1))?,byte(c.get(2))?])).take(MAX_SWATCHES).collect())
}

impl Preferences {
    pub fn load() -> Preferences {
        super::project_io::data_dir()
            .and_then(|d| std::fs::read_to_string(d.join(FILE)).ok())
            .and_then(|t| serde_json::from_str::<Value>(&t).ok())
            .map_or_else(Preferences::default, |v| Preferences::from_json(&v))
    }

    pub fn save(&self) -> Result<(), String> {
        let dir = super::project_io::data_dir().ok_or("no settings folder")?;
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        std::fs::write(dir.join(FILE), serde_json::to_string_pretty(&self.to_json()).unwrap_or_default()).map_err(|e| e.to_string())
    }

    fn to_json(&self) -> Value {
        json!({
            "autosave": self.autosave,
            "autosave_every": self.autosave_every,
            "undo_steps": self.undo_steps,
            "recent_count": self.recent_count,
            "default_resolution": self.default_resolution,
            "bake_on_load": self.bake_on_load,
            "ui_scale": self.ui_scale,
            "shelf_icon_size": self.shelf_icon_size,
            "default_hdri": self.default_hdri,
            "key_light": self.key_light,
            "fov_deg": self.fov_deg,
            "orbit_speed": self.orbit_speed,
            "zoom_speed": self.zoom_speed,
            "invert_zoom": self.invert_zoom,
            "rt_max_samples": self.rt_max_samples,
            "rt_bounces": self.rt_bounces,
            "brush_size": self.brush_size,
            "brush_hardness": self.brush_hardness,
            "brush_spacing": self.brush_spacing,
            "swatches": self.swatches,
            "poly_padding": self.poly_padding,
            "bake_quality": self.bake_quality.name(),
            "bake_match_by_name": self.bake_match_by_name,
            "export_folder": self.export_folder,
            "export_template": self.export_template,
            "export_format": self.export_format.name(),
            "export_depth": self.export_depth.name(),
            "export_size": self.export_size,
            "export_padding": self.export_padding.key(),
            "export_open_folder": self.export_open_folder,
            "mcp_on_start": self.mcp_on_start,
            "show_welcome": self.show_welcome,
            "mcp_port": self.mcp_port,
            "gpu": self.gpu.key(),
        })
    }

    /// Missing or invalid entries keep their defaults, so files from older
    /// versions load.
    pub(super) fn from_json(v: &Value) -> Preferences {
        let d = Preferences::default();
        let b = |k: &str, def: bool| v[k].as_bool().unwrap_or(def);
        let f = |k: &str, def: f32, lo: f32, hi: f32| v[k].as_f64().map_or(def, |x| (x as f32).clamp(lo, hi));
        let u = |k: &str, def: u64, lo: u64, hi: u64| v[k].as_u64().map_or(def, |x| x.clamp(lo, hi));
        let s = |k: &str, def: &str| v[k].as_str().unwrap_or(def).to_string();
        let res = |k: &str, def: u32| v[k].as_u64().map(|x| x as u32).filter(|x| RESOLUTIONS.contains(x)).unwrap_or(def);
        Preferences {
            autosave: b("autosave", d.autosave),
            autosave_every: u("autosave_every", d.autosave_every as u64, 1, 500) as u32,
            undo_steps: u("undo_steps", d.undo_steps as u64, 8, 512) as usize,
            recent_count: u("recent_count", d.recent_count as u64, 1, 30) as usize,
            default_resolution: res("default_resolution", d.default_resolution),
            bake_on_load: b("bake_on_load", d.bake_on_load),
            ui_scale: f("ui_scale", d.ui_scale, 0.6, 2.5),
            shelf_icon_size: f("shelf_icon_size", d.shelf_icon_size, super::ICON_MIN, super::ICON_MAX),
            default_hdri: s("default_hdri", &d.default_hdri),
            key_light: b("key_light", d.key_light),
            fov_deg: f("fov_deg", d.fov_deg, 10.0, 120.0),
            orbit_speed: f("orbit_speed", d.orbit_speed, 0.1, 5.0),
            zoom_speed: f("zoom_speed", d.zoom_speed, 0.1, 5.0),
            invert_zoom: b("invert_zoom", d.invert_zoom),
            rt_max_samples: u("rt_max_samples", d.rt_max_samples as u64, 16, 16384) as u32,
            rt_bounces: u("rt_bounces", d.rt_bounces as u64, 1, 16) as u32,
            brush_size: f("brush_size", d.brush_size, 1.0, 1000.0),
            brush_hardness: f("brush_hardness", d.brush_hardness, 0.0, 1.0),
            brush_spacing: f("brush_spacing", d.brush_spacing, 0.01, 2.0),
            swatches: swatches_from_json(&v["swatches"]),
            poly_padding: f("poly_padding", d.poly_padding, 0.0, 16.0),
            bake_quality: v["bake_quality"].as_str().and_then(|n| Quality::ALL.into_iter().find(|q| q.name() == n)).unwrap_or(d.bake_quality),
            bake_match_by_name: b("bake_match_by_name", d.bake_match_by_name),
            export_folder: s("export_folder", &d.export_folder),
            export_template: s("export_template", &d.export_template),
            export_format: v["export_format"].as_str().and_then(Format::from_name).unwrap_or(d.export_format),
            export_depth: v["export_depth"]
                .as_str()
                .and_then(|n| [Depth::U8, Depth::U16, Depth::F32].into_iter().find(|x| x.name() == n))
                .unwrap_or(d.export_depth),
            export_size: v["export_size"].as_u64().map(|x| x as u32).filter(|x| RESOLUTIONS.contains(x)),
            export_padding: v["export_padding"].as_str().and_then(Padding::from_key).unwrap_or(d.export_padding),
            export_open_folder: b("export_open_folder", d.export_open_folder),
            mcp_on_start: b("mcp_on_start", d.mcp_on_start),
            show_welcome: b("show_welcome", d.show_welcome),
            mcp_port: u("mcp_port", d.mcp_port as u64, 1024, 65000) as u16,
            gpu: v["gpu"].as_str().and_then(|k| Gpu::ALL.into_iter().find(|g| g.key() == k)).unwrap_or(d.gpu),
        }
    }

    /// The adapter preference, read before the window (and GPU) exist.
    pub fn power_preference() -> eframe::wgpu::PowerPreference {
        match Preferences::load().gpu {
            Gpu::Auto => eframe::wgpu::PowerPreference::from_env().unwrap_or_default(),
            Gpu::HighPerformance => eframe::wgpu::PowerPreference::HighPerformance,
            Gpu::LowPower => eframe::wgpu::PowerPreference::LowPower,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Page {
    General,
    Interface,
    Viewport,
    Painting,
    Baking,
    Export,
    Agents,
    Performance,
}

impl Page {
    const ALL: [Page; 8] = [Page::General, Page::Interface, Page::Viewport, Page::Painting, Page::Baking, Page::Export, Page::Agents, Page::Performance];
    fn name(self) -> &'static str {
        match self {
            Page::General => "General",
            Page::Interface => "Interface",
            Page::Viewport => "Viewport & navigation",
            Page::Painting => "Painting",
            Page::Baking => "Baking",
            Page::Export => "Export",
            Page::Agents => "AI agents (MCP)",
            Page::Performance => "Performance",
        }
    }
}

fn hint(ui: &mut egui::Ui, text: &str) {
    ui.label(egui::RichText::new(text).small().color(theme::TEXT_DIM));
}

fn resolution_combo(ui: &mut egui::Ui, id: &str, v: &mut u32) {
    egui::ComboBox::from_id_salt(id).selected_text(format!("{v} px")).show_ui(ui, |ui| {
        for r in RESOLUTIONS {
            ui.selectable_value(v, r, format!("{r} px"));
        }
    });
}

impl PainterApp {
    pub(super) fn preferences_window(&mut self, ctx: &egui::Context) {
        let mut open = self.prefs_ui.open;
        egui::Window::new("Preferences")
            .open(&mut open)
            .collapsible(false)
            .default_size([640.0, 440.0])
            .min_size([520.0, 320.0])
            .show(ctx, |ui| {
                ui.horizontal_top(|ui| {
                    ui.vertical(|ui| {
                        ui.set_width(160.0);
                        for p in Page::ALL {
                            if ui.add_sized([156.0, 24.0], theme::selectable(self.prefs_ui.page == p, p.name())).clicked() {
                                self.prefs_ui.page = p;
                            }
                        }
                        ui.add_space(12.0);
                        if ui.button("Restore page defaults").clicked() {
                            self.restore_page_defaults(self.prefs_ui.page);
                        }
                        if ui.button("Open settings folder").clicked()
                            && let Some(dir) = super::project_io::data_dir() {
                                let _ = std::fs::create_dir_all(&dir);
                                super::open_in_file_browser(&dir);
                            }
                    });
                    ui.separator();
                    // The scroll area inherits the row's left-to-right layout;
                    // the page itself stacks top to bottom.
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.vertical(|ui| {
                            ui.set_min_width(ui.available_width());
                            self.preferences_page(ui);
                        });
                    });
                });
            });
        self.prefs_ui.open = open;
    }

    fn restore_page_defaults(&mut self, page: Page) {
        let d = Preferences::default();
        let p = &mut self.prefs;
        match page {
            Page::General => {
                (p.autosave, p.autosave_every, p.undo_steps, p.recent_count) = (d.autosave, d.autosave_every, d.undo_steps, d.recent_count);
                (p.default_resolution, p.bake_on_load) = (d.default_resolution, d.bake_on_load);
            }
            Page::Interface => (p.ui_scale, p.shelf_icon_size) = (d.ui_scale, d.shelf_icon_size),
            Page::Viewport => {
                (p.default_hdri, p.key_light, p.fov_deg) = (d.default_hdri, d.key_light, d.fov_deg);
                (p.orbit_speed, p.zoom_speed, p.invert_zoom) = (d.orbit_speed, d.zoom_speed, d.invert_zoom);
                (p.rt_max_samples, p.rt_bounces) = (d.rt_max_samples, d.rt_bounces);
            }
            Page::Painting => {
                (p.brush_size, p.brush_hardness, p.brush_spacing, p.poly_padding) = (d.brush_size, d.brush_hardness, d.brush_spacing, d.poly_padding);
            }
            Page::Baking => (p.bake_quality, p.bake_match_by_name) = (d.bake_quality, d.bake_match_by_name),
            Page::Export => {
                (p.export_folder, p.export_template, p.export_format, p.export_depth) = (d.export_folder, d.export_template, d.export_format, d.export_depth);
                (p.export_size, p.export_padding, p.export_open_folder) = (d.export_size, d.export_padding, d.export_open_folder);
            }
            Page::Agents => (p.mcp_on_start, p.mcp_port) = (d.mcp_on_start, d.mcp_port),
            Page::Performance => p.gpu = d.gpu,
        }
    }

    fn preferences_page(&mut self, ui: &mut egui::Ui) {
        let builtins = self.env_builtins.clone();
        let templates: Vec<String> = self.export.templates.iter().map(|t| t.name.clone()).collect();
        let mcp_url = self.mcp.as_ref().map(|s| s.url());
        let p = &mut self.prefs;
        match self.prefs_ui.page {
            Page::General => {
                section(ui, "Projects");
                labeled(ui, "Resolution", |ui| resolution_combo(ui, "pref res", &mut p.default_resolution));
                hint(ui, "Texture size of new projects. Change the current project's size in the texture set settings.");
                ui.checkbox(&mut p.bake_on_load, "Bake mesh maps when a mesh is loaded");
                ui.horizontal(|ui| {
                    if ui.button("Clear bake cache").clicked() {
                        crate::bake_cache::clear();
                    }
                    hint(ui, "Baked maps are cached, so reopening a mesh with the same settings is instant.");
                });
                labeled(ui, "Recent files", |ui| ui.add(egui::DragValue::new(&mut p.recent_count).range(1..=30)));
                section(ui, "Safety");
                ui.checkbox(&mut p.autosave, "Autosave a recovery copy of unsaved work");
                ui.add_enabled_ui(p.autosave, |ui| {
                    labeled(ui, "Every", |ui| ui.add(egui::DragValue::new(&mut p.autosave_every).range(1..=500).suffix(" edits")));
                });
                labeled(ui, "Undo steps", |ui| ui.add(egui::DragValue::new(&mut p.undo_steps).range(8..=512)));
                hint(ui, "More steps keep more painted texture history in memory.");
            }
            Page::Interface => {
                section(ui, "Scale");
                labeled(ui, "UI scale", |ui| ui.add(egui::Slider::new(&mut p.ui_scale, 0.6..=2.5).max_decimals(2).suffix("x")));
                hint(ui, "Applies when you release the slider. Ctrl + / Ctrl - also zoom the interface.");
                labeled(ui, "Shelf icons", |ui| ui.add(egui::Slider::new(&mut p.shelf_icon_size, super::ICON_MIN..=super::ICON_MAX).max_decimals(0).suffix(" px")));
            }
            Page::Viewport => {
                section(ui, "Lighting");
                labeled(ui, "Environment", |ui| {
                    egui::ComboBox::from_id_salt("pref hdri").selected_text(p.default_hdri.as_str()).show_ui(ui, |ui| {
                        for &i in &builtins {
                            let name = environment::BUILTIN[i].0;
                            ui.selectable_value(&mut p.default_hdri, name.to_string(), name);
                        }
                    });
                });
                ui.checkbox(&mut p.key_light, "Key light with shadows on new projects");
                section(ui, "Camera");
                labeled(ui, "Field of view", |ui| ui.add(egui::Slider::new(&mut p.fov_deg, 10.0..=120.0).max_decimals(0).suffix("°")));
                labeled(ui, "Orbit speed", |ui| ui.add(egui::Slider::new(&mut p.orbit_speed, 0.1..=5.0).max_decimals(2).suffix("x")));
                labeled(ui, "Zoom speed", |ui| ui.add(egui::Slider::new(&mut p.zoom_speed, 0.1..=5.0).max_decimals(2).suffix("x")));
                ui.checkbox(&mut p.invert_zoom, "Invert scroll zoom");
                section(ui, "Ray traced preview");
                labeled(ui, "Max samples", |ui| ui.add(egui::DragValue::new(&mut p.rt_max_samples).range(16..=16384)));
                labeled(ui, "Bounces", |ui| ui.add(egui::DragValue::new(&mut p.rt_bounces).range(1..=16)));
            }
            Page::Painting => {
                section(ui, "New brush defaults");
                labeled(ui, "Size", |ui| ui.add(egui::Slider::new(&mut p.brush_size, 1.0..=400.0).max_decimals(0).suffix(" pt")));
                labeled(ui, "Hardness", |ui| ui.add(egui::Slider::new(&mut p.brush_hardness, 0.0..=1.0).max_decimals(2)));
                labeled(ui, "Spacing", |ui| ui.add(egui::Slider::new(&mut p.brush_spacing, 0.01..=1.0).max_decimals(2)));
                hint(ui, "Used when the app starts.");
                section(ui, "Polygon fill");
                labeled(ui, "Seam padding", |ui| ui.add(egui::Slider::new(&mut p.poly_padding, 0.0..=16.0).max_decimals(1).suffix(" texels")));
                hint(ui, "How far fills reach past UV island borders, so mips and filtering don't show seams.");
            }
            Page::Baking => {
                section(ui, "Defaults");
                labeled(ui, "Quality", |ui| {
                    egui::ComboBox::from_id_salt("pref bake q").selected_text(p.bake_quality.name()).show_ui(ui, |ui| {
                        for q in Quality::ALL {
                            ui.selectable_value(&mut p.bake_quality, q, q.name());
                        }
                    });
                });
                ui.checkbox(&mut p.bake_match_by_name, "Match high and low poly parts by name");
                hint(ui, "Used for new projects; each project keeps its own bake settings.");
            }
            Page::Export => {
                section(ui, "Defaults");
                labeled(ui, "Template", |ui| {
                    egui::ComboBox::from_id_salt("pref tpl").selected_text(p.export_template.as_str()).width(240.0).show_ui(ui, |ui| {
                        for t in &templates {
                            ui.selectable_value(&mut p.export_template, t.clone(), t);
                        }
                    });
                });
                labeled(ui, "Folder", |ui| {
                    ui.add(egui::TextEdit::singleline(&mut p.export_folder).desired_width(240.0));
                    if ui.button("Browse…").clicked()
                        && let Some(dir) = rfd::FileDialog::new().pick_folder() {
                            p.export_folder = dir.display().to_string();
                        }
                });
                super::export_ui::format_controls(ui, p);
                ui.checkbox(&mut p.export_open_folder, "Open the folder after exporting");
                hint(ui, "The export window edits these too: it remembers what you used last.");
            }
            Page::Agents => {
                section(ui, "MCP server");
                ui.checkbox(&mut p.mcp_on_start, "Accept AI agent connections when the app starts");
                labeled(ui, "Port", |ui| ui.add(egui::DragValue::new(&mut p.mcp_port).range(1024..=65000)));
                hint(ui, "Takes effect on the next start. The next free port is used if this one is taken.");
                if let Some(url) = mcp_url {
                    labeled(ui, "Running at", |ui| ui.label(url));
                }
            }
            Page::Performance => {
                section(ui, "Graphics");
                labeled(ui, "GPU", |ui| {
                    egui::ComboBox::from_id_salt("pref gpu").selected_text(p.gpu.name()).width(240.0).show_ui(ui, |ui| {
                        for g in Gpu::ALL {
                            ui.selectable_value(&mut p.gpu, g, g.name());
                        }
                    });
                });
                hint(ui, "Takes effect on the next start.");
                let info = self.rs.adapter.get_info();
                labeled(ui, "Current", |ui| ui.label(format!("{} ({:?})", info.name, info.backend)));
            }
        }
    }

    /// Puts changed preferences into effect and saves them. Runs when no
    /// control is being dragged, so sliders apply once on release.
    pub(super) fn apply_preferences(&mut self, ctx: &egui::Context) {
        if self.prefs == self.prefs_applied || ctx.input(|i| i.pointer.any_down()) {
            return;
        }
        let (old, new) = (self.prefs_applied.clone(), self.prefs.clone());
        if new.ui_scale != old.ui_scale {
            ctx.set_zoom_factor(new.ui_scale);
        }
        if new.shelf_icon_size != old.shelf_icon_size {
            self.shelf.icon_size = new.shelf_icon_size;
        }
        if new.fov_deg != old.fov_deg {
            self.cam.fov_y = new.fov_deg.to_radians();
        }
        // Ray-traced sample count and bounce depth are startup-only:
        // changes are saved here and take effect the next time the app opens.
        self.undo.max_entries = new.undo_steps;
        if new.recent_count != old.recent_count {
            self.recent.truncate(new.recent_count);
        }
        if let Err(e) = new.save() {
            self.status = format!("Couldn't save preferences: {e}");
        }
        self.prefs_applied = new;
    }

    /// Startup: preferences that only take effect on a fresh app.
    pub(super) fn apply_startup_preferences(&mut self, ctx: &egui::Context) {
        let p = self.prefs.clone();
        self.show_welcome = p.show_welcome;
        ctx.set_zoom_factor(p.ui_scale);
        self.shelf.icon_size = p.shelf_icon_size;
        self.cam.fov_y = p.fov_deg.to_radians();
        (self.rt.max_samples, self.rt.bounces) = (p.rt_max_samples, p.rt_bounces);
        self.undo.max_entries = p.undo_steps;
        self.recent.truncate(p.recent_count);
        for brush in [&mut self.brush, &mut self.other_brush] {
            (brush.size, brush.hardness, brush.spacing) = (p.brush_size, p.brush_hardness, p.brush_spacing);
        }
        self.apply_project_defaults();
        self.prefs_applied = p;
    }

    /// Settings a new project starts from.
    pub(super) fn apply_project_defaults(&mut self) {
        let p = &self.prefs;
        self.view.key.on = p.key_light;
        self.bake_settings.apply_quality(p.bake_quality);
        self.bake_settings.match_by_name = p.bake_match_by_name;
    }
}

#[cfg(test)]
mod swatch_tests{
use super::*;
#[test]fn swatches_round_trip(){let p=Preferences{swatches:vec![[255,0,0],[10,20,30]],..Preferences::default()};assert_eq!(Preferences::from_json(&p.to_json()).swatches,p.swatches);}
#[test]fn malformed_swatches_skip(){let v=json!({"swatches":[[1,2,3],[1,2],"red",[300,0,0],null]});assert_eq!(Preferences::from_json(&v).swatches,vec![[1,2,3],[255,0,0]]);}
}
