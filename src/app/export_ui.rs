//! The export window: output templates (built-in engine presets and the
//! user's own, editable channel by channel), format and padding settings,
//! and exports that encode on a worker thread.

use super::prefs::{Preferences, RESOLUTIONS};
use super::{PainterApp, labeled, section};
use crate::export::{self, Chan, Depth, Format, Layout, OutputMap, Padding, Source, Template};
use crate::theme;
use eframe::egui;
use egui::{Color32, vec2};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::mpsc;

pub(super) struct ExportDialog {
    pub open: bool,
    /// Built-in templates first, then the user's.
    pub templates: Vec<Template>,
    /// The user's templates as last saved, to save only on change.
    saved_custom: Vec<Template>,
    /// `$mesh` in file names.
    pub mesh_name: String,
    job: Option<mpsc::Receiver<Result<Vec<PathBuf>, String>>>,
    pub result: Option<Result<String, String>>,
}

const FILE: &str = "export_templates.json";

fn load_custom() -> Vec<Template> {
    super::project_io::data_dir()
        .and_then(|d| std::fs::read_to_string(d.join(FILE)).ok())
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|v| v.as_array().map(|a| a.iter().filter_map(export::template_from).collect()))
        .unwrap_or_default()
}

impl ExportDialog {
    pub fn new(mesh_name: String) -> ExportDialog {
        let custom = load_custom();
        let mut templates = export::builtin_templates();
        templates.extend(custom.iter().cloned());
        ExportDialog { open: false, templates, saved_custom: custom, mesh_name, job: None, result: None }
    }

    pub fn find(&self, name: &str) -> Option<usize> {
        self.templates.iter().position(|t| t.name == name)
    }

    /// A template by name, or a close match for names agents and older
    /// versions use ("Unreal Engine", "Unity URP (MaskMap, …)").
    pub fn find_loose(&self, name: &str) -> Option<usize> {
        let n = name.to_lowercase();
        self.find(name)
            .or_else(|| self.templates.iter().position(|t| t.name.to_lowercase() == n))
            .or_else(|| self.templates.iter().position(|t| t.name.to_lowercase().starts_with(&n)))
            .or_else(|| {
                // Older preset names: match on their leading words.
                let key: String = n.split(['(', ',']).next().unwrap_or("").trim().to_string();
                (!key.is_empty()).then(|| self.templates.iter().position(|t| t.name.to_lowercase().starts_with(&key))).flatten()
            })
    }

    fn unique_name(&self, base: &str) -> String {
        if self.find(base).is_none() {
            return base.to_string();
        }
        (2..).map(|i| format!("{base} {i}")).find(|n| self.find(n).is_none()).expect("unbounded")
    }

    fn save_custom_if_changed(&mut self) -> Result<(), String> {
        let custom: Vec<Template> = self.templates.iter().filter(|t| !t.builtin).cloned().collect();
        if custom == self.saved_custom {
            return Ok(());
        }
        let dir = super::project_io::data_dir().ok_or("no settings folder")?;
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let json = Value::Array(custom.iter().map(export::template_json).collect());
        std::fs::write(dir.join(FILE), serde_json::to_string_pretty(&json).unwrap_or_default()).map_err(|e| e.to_string())?;
        self.saved_custom = custom;
        Ok(())
    }
}

/// Format, bit depth, size and padding (shared with Preferences > Export).
pub(super) fn format_controls(ui: &mut egui::Ui, p: &mut Preferences) {
    labeled(ui, "Format", |ui| {
        egui::ComboBox::from_id_salt("exp fmt").selected_text(p.export_format.name()).show_ui(ui, |ui| {
            for f in Format::ALL {
                ui.selectable_value(&mut p.export_format, f, f.name());
            }
        });
    });
    let depths = p.export_format.depths();
    if !depths.contains(&p.export_depth) {
        p.export_depth = depths[0];
    }
    labeled(ui, "Bit depth", |ui| {
        ui.add_enabled_ui(depths.len() > 1, |ui| {
            egui::ComboBox::from_id_salt("exp depth").selected_text(p.export_depth.name()).show_ui(ui, |ui| {
                for &d in depths {
                    ui.selectable_value(&mut p.export_depth, d, d.name());
                }
            });
        });
    });
    labeled(ui, "Size", |ui| {
        let text = p.export_size.map_or("Document size".to_string(), |s| format!("{s} px"));
        egui::ComboBox::from_id_salt("exp size").selected_text(text).show_ui(ui, |ui| {
            ui.selectable_value(&mut p.export_size, None, "Document size");
            for r in RESOLUTIONS {
                ui.selectable_value(&mut p.export_size, Some(r), format!("{r} px"));
            }
        });
    });
    labeled(ui, "Padding", |ui| {
        egui::ComboBox::from_id_salt("exp pad").selected_text(p.export_padding.name()).width(170.0).show_ui(ui, |ui| {
            for pad in [Padding::Infinite, Padding::Dilate(8), Padding::Transparent, Padding::None] {
                let on = std::mem::discriminant(&p.export_padding) == std::mem::discriminant(&pad);
                if ui.selectable_label(on, pad.name()).clicked() && !on {
                    p.export_padding = pad;
                }
            }
        });
        if let Padding::Dilate(px) = &mut p.export_padding {
            ui.add(egui::DragValue::new(px).range(1..=256).suffix(" px"));
        }
    });
}

/// A channel source picker plus an invert toggle.
fn chan_picker(ui: &mut egui::Ui, id: (usize, usize), label: &str, c: &mut Chan) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.add_sized([14.0, 18.0], egui::Label::new(egui::RichText::new(label).strong().color(theme::TEXT_DIM)));
        egui::ComboBox::from_id_salt(("chan", id)).selected_text(c.label()).width(190.0).height(360.0).show_ui(ui, |ui| {
            for src in Source::all() {
                for comp in 0..src.components() as u8 {
                    let pick = Chan { src, comp, invert: c.invert };
                    if ui.selectable_label(c.src == src && c.comp == comp, Chan { invert: false, ..pick }.label()).clicked() {
                        *c = pick;
                        changed = true;
                    }
                }
            }
        });
        if ui.add(theme::selectable(c.invert, "Invert")).on_hover_text("Write 1 - value (roughness → smoothness)").clicked() {
            c.invert = !c.invert;
            changed = true;
        }
    });
    changed
}

/// Editor for one output map; returns true to delete it.
fn map_editor(ui: &mut egui::Ui, i: usize, m: &mut OutputMap, editable: bool) -> bool {
    let mut delete = false;
    egui::Frame::new().fill(theme::PANEL_DARK).corner_radius(theme::RADIUS).inner_margin(8.0).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.add_enabled_ui(editable, |ui| {
            ui.horizontal(|ui| {
                // The name takes what the type, 16-bit and delete controls leave.
                let w = (ui.available_width() - 230.0).max(90.0);
                ui.add(egui::TextEdit::singleline(&mut m.name).desired_width(w).hint_text("$mesh_Name"));
                egui::ComboBox::from_id_salt(("layout", i)).selected_text(m.layout.name()).width(90.0).show_ui(ui, |ui| {
                    for l in Layout::ALL {
                        ui.selectable_value(&mut m.layout, l, l.name());
                    }
                });
                ui.checkbox(&mut m.force_16, "16-bit").on_hover_text("Write 16 bits per channel where the format allows (height maps band in 8 bits)");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button(crate::icons::TRASH).on_hover_text("Remove this output").clicked() {
                        delete = true;
                    }
                });
            });
            if m.layout == Layout::Gray {
                chan_picker(ui, (i, 0), "L", &mut m.chans[0]);
            } else {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("RGB from").color(theme::TEXT_DIM));
                    egui::ComboBox::from_id_salt(("rgb", i)).selected_text("choose a color map…").width(190.0).show_ui(ui, |ui| {
                        for src in Source::all().into_iter().filter(|s| s.components() == 3) {
                            if ui.selectable_label(false, src.name()).clicked() {
                                for k in 0..3 {
                                    m.chans[k] = Chan::of(src, k as u8);
                                }
                            }
                        }
                    });
                });
                for (k, label) in ["R", "G", "B", "A"].into_iter().enumerate().take(m.layout.channels()) {
                    chan_picker(ui, (i, k), label, &mut m.chans[k]);
                }
            }
        });
    });
    delete
}

pub(super) fn open_in_file_browser(path: &std::path::Path) {
    let program = if cfg!(target_os = "windows") {
        "explorer"
    } else if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    };
    let _ = std::process::Command::new(program).arg(path).spawn();
}

impl PainterApp {
    pub(super) fn export_settings(&self) -> export::Settings {
        let p = &self.prefs;
        export::Settings {
            folder: PathBuf::from(&p.export_folder),
            mesh_name: self.export.mesh_name.clone(),
            project_name: self.project_path.as_ref().and_then(|p| p.file_stem()).map_or("Untitled".into(), |s| s.to_string_lossy().into_owned()),
            set_name: self.mesh.sets.get(self.sets.active).map_or(String::new(), |s| s.name.replace(' ', "")),
            set_count: self.mesh.sets.len(),
            format: p.export_format,
            depth: p.export_depth,
            size: p.export_size,
            padding: p.export_padding,
            normal_strength: self.view.height_strength,
            use_baked_normal: self.view.use_baked_normal,
        }
    }

    /// Every texture set's maps, read back now (after compositing), with
    /// the settings that name its files.
    fn export_snapshots(&mut self, settings: &export::Settings) -> Vec<(export::Snapshot, export::Settings)> {
        if self.engine.dirty {
            self.engine.composite();
        }
        self.composite_parked();
        let engines = self.set_engines();
        self.mesh
            .sets
            .iter()
            .zip(engines)
            .map(|(set, engine)| {
                let snapshot = export::snapshot(engine, &self.rs.device, &self.rs.queue, self.mesh.clone());
                (snapshot, export::Settings { set_name: set.name.replace(' ', ""), ..settings.clone() })
            })
            .collect()
    }

    /// Reads the maps back now and writes the files on a worker thread.
    fn start_export(&mut self, template: Template, ctx: &egui::Context) {
        let settings = self.export_settings();
        let jobs = self.export_snapshots(&settings);
        let (tx, rx) = mpsc::channel();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let mut files = Vec::new();
            let result = jobs.iter().try_for_each(|(snapshot, settings)| {
                files.extend(export::export(snapshot, &template, settings)?);
                Ok::<_, String>(())
            });
            let _ = tx.send(result.map(|_| files));
            ctx.request_repaint();
        });
        self.export.job = Some(rx);
        self.export.result = None;
    }

    /// Exports synchronously (agents wait for the files).
    pub(super) fn export_now(&mut self, template: usize, settings: &export::Settings) -> Result<Vec<PathBuf>, String> {
        let mut files = Vec::new();
        for (snapshot, settings) in self.export_snapshots(settings) {
            files.extend(export::export(&snapshot, &self.export.templates[template], &settings)?);
        }
        Ok(files)
    }

    fn poll_export(&mut self) {
        let Some(rx) = &self.export.job else { return };
        let Ok(result) = rx.try_recv() else { return };
        self.export.job = None;
        self.export.result = Some(result.map(|files| {
            if self.prefs.export_open_folder {
                open_in_file_browser(std::path::Path::new(&self.prefs.export_folder));
            }
            format!("Wrote {} files to {}", files.len(), self.prefs.export_folder)
        }));
        if let Some(Ok(msg)) = &self.export.result {
            self.status = msg.clone();
        }
    }

    pub(super) fn export_window(&mut self, ctx: &egui::Context) {
        self.poll_export();
        if !self.export.open {
            return;
        }
        let mut open = true;
        egui::Window::new("Export Textures")
            .open(&mut open)
            .collapsible(false)
            .default_size([980.0, 600.0])
            .min_size([760.0, 420.0])
            .show(ctx, |ui| {
                let sel = self.export.find(&self.prefs.export_template).unwrap_or(0);
                // Side panels of fixed width and a middle that takes what the
                // window has left, so the window keeps the size it is given.
                egui::Panel::left("export templates")
                    .exact_size(226.0)
                    .frame(egui::Frame::new().inner_margin(egui::Margin { right: 10, ..Default::default() }))
                    .show(ui, |ui| {
                        section(ui, "Output templates");
                        egui::ScrollArea::vertical().id_salt("tpl list").max_height(ui.available_height() - 40.0).show(ui, |ui| {
                            let mut last_builtin = true;
                            for t in &self.export.templates {
                                if last_builtin && !t.builtin {
                                    ui.separator();
                                    ui.label(egui::RichText::new("Your templates").small().color(theme::TEXT_DIM));
                                }
                                last_builtin = t.builtin;
                                if ui.add_sized([204.0, 22.0], theme::selectable(t.name == self.prefs.export_template, t.name.as_str())).clicked() {
                                    self.prefs.export_template = t.name.clone();
                                }
                            }
                        });
                        ui.horizontal(|ui| {
                            if ui.button("New").on_hover_text("An empty template").clicked() {
                                let name = self.export.unique_name("My template");
                                self.export.templates.push(Template { name: name.clone(), maps: Vec::new(), builtin: false });
                                self.prefs.export_template = name;
                            }
                            if ui.button(crate::icons::COPY).on_hover_text("Duplicate (to edit a built-in template)").clicked() {
                                let mut t = self.export.templates[sel].clone();
                                t.name = self.export.unique_name(&format!("{} copy", t.name));
                                t.builtin = false;
                                self.prefs.export_template = t.name.clone();
                                self.export.templates.push(t);
                            }
                            let can_delete = !self.export.templates[sel].builtin;
                            if ui.add_enabled(can_delete, egui::Button::new(crate::icons::TRASH)).on_hover_text("Delete this template").clicked() {
                                self.export.templates.remove(sel);
                                self.prefs.export_template = self.export.templates[sel.min(self.export.templates.len() - 1)].name.clone();
                            }
                        });
                    });
                egui::Panel::right("export settings")
                    .exact_size(330.0)
                    .frame(egui::Frame::new().inner_margin(egui::Margin { left: 12, ..Default::default() }))
                    .show(ui, |ui| {
                    let sel = self.export.find(&self.prefs.export_template).unwrap_or(0);
                        section(ui, "Settings");
                        labeled(ui, "Folder", |ui| {
                            // The rest of the row, less the Browse button.
                            let w = (ui.available_width() - 80.0).max(60.0);
                            ui.add(egui::TextEdit::singleline(&mut self.prefs.export_folder).desired_width(w));
                            if ui.button("Browse…").clicked()
                                && let Some(dir) = rfd::FileDialog::new().pick_folder() {
                                    self.prefs.export_folder = dir.display().to_string();
                                }
                        });
                        labeled(ui, "$mesh", |ui| ui.add(egui::TextEdit::singleline(&mut self.export.mesh_name).desired_width(ui.available_width())));
                        format_controls(ui, &mut self.prefs);
                        ui.checkbox(&mut self.prefs.export_open_folder, "Open folder when done");
                        let doc = self.engine.res;
                        let out = self.prefs.export_size.unwrap_or(doc);
                        if out > doc {
                            ui.label(egui::RichText::new(format!("Upscaled from the {doc} px document.")).small().color(Color32::from_rgb(230, 180, 90)));
                        }

                        section(ui, "Files");
                        let settings = self.export_settings();
                        let t = self.export.templates[sel].clone();
                        egui::ScrollArea::vertical().id_salt("files").max_height(170.0).show(ui, |ui| {
                            for m in &t.maps {
                                let depth = match settings.format.depths() {
                                    d if m.force_16 && d.contains(&Depth::U16) && settings.depth == Depth::U8 => "16-bit",
                                    _ => settings.depth.name(),
                                };
                                ui.add(egui::Label::new(egui::RichText::new(format!("{}  ·  {}  {depth}", settings.file_name(m), m.layout.name())).small()).wrap());
                            }
                        });
                        ui.add_space(8.0);
                        let busy = self.export.job.is_some();
                        let can = !busy && !t.maps.is_empty();
                        ui.horizontal(|ui| {
                            let button = egui::Button::new(egui::RichText::new(theme::with_icon(crate::icons::EXPORT, "Export")).strong()).min_size(vec2(110.0, 28.0));
                            if ui.add_enabled(can, button).clicked() {
                                let template = t.clone();
                                self.start_export(template, ui.ctx());
                            }
                            if busy {
                                ui.spinner();
                                ui.label("Writing…");
                            }
                        });
                        match &self.export.result {
                            Some(Ok(msg)) => {
                                ui.label(egui::RichText::new(msg).color(Color32::from_rgb(120, 200, 120)));
                                if ui.button("Open folder").clicked() {
                                    open_in_file_browser(std::path::Path::new(&self.prefs.export_folder));
                                }
                            }
                            Some(Err(e)) => {
                                ui.label(egui::RichText::new(e).color(Color32::from_rgb(230, 110, 100)));
                            }
                            None => {}
                        }
                    });
                egui::CentralPanel::default().frame(egui::Frame::new()).show(ui, |ui| {
                    let sel = self.export.find(&self.prefs.export_template).unwrap_or(0);
                        let builtin = self.export.templates[sel].builtin;
                        section(ui, "Output maps");
                        if builtin {
                            ui.label(egui::RichText::new("Built-in template. Duplicate it to change maps, names or channel packing.").small().color(theme::TEXT_DIM));
                        } else {
                            let mut name = self.export.templates[sel].name.clone();
                            let r = labeled(ui, "Name", |ui| ui.add(egui::TextEdit::singleline(&mut name).desired_width(240.0)));
                            if r.changed() && !name.trim().is_empty() && self.export.find(&name).is_none() {
                                self.export.templates[sel].name = name.clone();
                                self.prefs.export_template = name;
                            }
                        }
                        ui.label(egui::RichText::new("$mesh, $set (texture set) and $project in file names are replaced.").small().color(theme::TEXT_DIM));
                        egui::ScrollArea::vertical().id_salt("maps").show(ui, |ui| {
                            let t = &mut self.export.templates[sel];
                            let mut remove = None;
                            for (i, m) in t.maps.iter_mut().enumerate() {
                                if map_editor(ui, i, m, !builtin) {
                                    remove = Some(i);
                                }
                                ui.add_space(4.0);
                            }
                            if let Some(i) = remove {
                                t.maps.remove(i);
                            }
                            if !builtin && ui.button(theme::with_icon(crate::icons::SPARKLE, "Add output map")).clicked() {
                                t.maps.push(OutputMap {
                                    name: format!("$mesh_Map{}", t.maps.len() + 1),
                                    layout: Layout::Rgb,
                                    chans: [Chan::of(Source::BaseColor, 0), Chan::of(Source::BaseColor, 1), Chan::of(Source::BaseColor, 2), Chan::of(Source::One, 0)],
                                    force_16: false,
                                });
                            }
                        });
                });
            });
        self.export.open = open;
        if !ctx.input(|i| i.pointer.any_down()) && ctx.memory(|m| m.focused().is_none())
            && let Err(e) = self.export.save_custom_if_changed() {
                self.status = format!("Couldn't save export templates: {e}");
            }
    }
}
