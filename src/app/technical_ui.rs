//! Technical diagnostics panel and user-configurable status-bar metrics.
use super::*;
use egui::RichText;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum TechnicalMetric {
    Fps,
    FrameInterval,
    CpuFrame,
    Bake,
    GpuMemoryEstimate,
    UndoMemory,
    TextureResolution,
    Triangles,
    Vertices,
    MeshParts,
    Objects,
    TextureSets,
    ViewportSize,
    F6Meter,
}

impl TechnicalMetric {
    pub(super) const ALL: [Self; 14] = [
        Self::Fps,
        Self::FrameInterval,
        Self::CpuFrame,
        Self::Bake,
        Self::GpuMemoryEstimate,
        Self::UndoMemory,
        Self::TextureResolution,
        Self::Triangles,
        Self::Vertices,
        Self::MeshParts,
        Self::Objects,
        Self::TextureSets,
        Self::ViewportSize,
        Self::F6Meter,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::Fps => "Viewport FPS",
            Self::FrameInterval => "Frame interval",
            Self::CpuFrame => "CPU frame time",
            Self::Bake => "Bake progress / time",
            Self::GpuMemoryEstimate => "GPU memory estimate",
            Self::UndoMemory => "Undo memory",
            Self::TextureResolution => "Texture resolution",
            Self::Triangles => "Triangles",
            Self::Vertices => "Vertices",
            Self::MeshParts => "Mesh parts",
            Self::Objects => "Imported objects",
            Self::TextureSets => "Texture sets",
            Self::ViewportSize => "Viewport dimensions",
            Self::F6Meter => "F6 paint profiler",
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::Fps => "Estimated throughput from the larger of UI frame work time and egui's frame interval.",
            Self::FrameInterval => "Time between UI frames; useful for detecting stutter.",
            Self::CpuFrame => "Time spent in the application's UI frame, with recent peak.",
            Self::Bake => "Shows live bake progress, then the latest GPU and CPU preparation times.",
            Self::GpuMemoryEstimate => "Application-side estimate of active and parked Engine textures and known buffers; shared Fill images may appear in more than one per-Engine subtotal. Shared Undo staging memory is shown separately.",
            Self::UndoMemory => "Undo history memory and configured limit.",
            Self::TextureResolution => "Resolution of the active texture set.",
            Self::Triangles => "Triangle count in the imported mesh.",
            Self::Vertices => "Vertex count after tangent splitting.",
            Self::MeshParts => "Connected parts reported by the latest bake.",
            Self::Objects => "Imported mesh objects and visible-object count.",
            Self::TextureSets => "Number of material texture sets.",
            Self::ViewportSize => "3D and UV view dimensions in pixels.",
            Self::F6Meter => "Whether detailed paint profiling is enabled.",
        }
    }

    fn short_value(self, app: &PainterApp) -> String {
        let mb = |bytes: u64| bytes as f64 / (1024.0 * 1024.0);
        match self {
            Self::Fps => {
                let interval = app.frame_interval_ms.max(app.frame_ms);
                if interval > 0.01 {
                    format!("{:.0} FPS", 1000.0 / interval)
                } else {
                    "FPS —".into()
                }
            }
            Self::FrameInterval => format!("{:.2} ms/frame", app.frame_interval_ms),
            Self::CpuFrame => format!("CPU {:.1} ms (peak {:.0})", app.frame_ms, app.frame_peak_ms),
            Self::Bake => {
                if let Some((progress, phase)) = app.bake_progress() {
                    format!("{phase} {:.0}%", progress * 100.0)
                } else if let Some(report) = &app.engine.bake_report {
                    if report.from_cache {
                        "Bake: cache".into()
                    } else if let Some(gpu) = report.seconds() {
                        format!("Bake {:.2}s GPU + {:.2}s prep", gpu, report.prep_seconds)
                    } else {
                        "Bake: finished".into()
                    }
                } else {
                    "Bake: —".into()
                }
            }
            Self::GpuMemoryEstimate => {
                let bytes = app.engine.vram_bytes()
                    + app.sets.parked.iter().flatten().map(|parked| parked.engine.vram_bytes()).sum::<u64>();
                format!("GPU est. {:.0} MB", mb(bytes))
            }
            Self::UndoMemory => format!(
                "Undo {:.1}/{:.0} MB ({} steps)",
                mb(app.undo.ram_bytes() as u64),
                mb(app.undo.max_ram_bytes as u64),
                app.undo.undo.len()
            ),
            Self::TextureResolution => format!("{}²", app.engine.res),
            Self::Triangles => format!("{} tris", app.mesh.triangle_count()),
            Self::Vertices => format!("{} verts", app.mesh.vertices.len()),
            Self::MeshParts => app.engine.bake_report.as_ref()
                .map(|r| format!("{} parts", r.parts))
                .unwrap_or_else(|| "Parts —".into()),
            Self::Objects => {
                let visible = app.hidden_objects.iter().filter(|hidden| !**hidden).count();
                format!("{} objects ({visible} visible)", app.mesh.objects.len())
            }
            Self::TextureSets => format!("{} texture sets", app.mesh.sets.len()),
            Self::ViewportSize => format!("3D {:?} · UV {:?}", app.views.view3d.size, app.views.view2d.size),
            Self::F6Meter => if app.meter.on { "F6 profiler ON".into() } else { "F6 profiler OFF".into() },
        }
    }
}

/// Bytes as MiB (1024 * 1024).
fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// One resident Engine in the memory breakdown.
#[derive(Clone, Debug, PartialEq, Eq)]
struct SetMemory {
    name: String,
    parked: bool,
    bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct MemoryTotals {
    active: u64,
    parked: u64,
    total: u64,
}

impl MemoryTotals {
    fn of(entries: &[SetMemory]) -> Self {
        let mut totals = Self::default();
        for entry in entries {
            if entry.parked {
                totals.parked = totals.parked.saturating_add(entry.bytes);
            } else {
                totals.active = totals.active.saturating_add(entry.bytes);
            }
        }
        totals.total = totals.active.saturating_add(totals.parked);
        totals
    }
}

#[derive(Serialize, Deserialize)]
pub(super) struct TechnicalUi {
    pub(super) open: bool,
    pub(super) bottom: Vec<TechnicalMetric>,
    /// Temporary benchmark controls/results; never persisted to user settings.
    #[serde(skip)]
    capture_scenario: usize,
    #[serde(skip)]
    capture_report: Option<String>,
    #[serde(skip)]
    capture_status: Option<String>,
}

impl Default for TechnicalUi {
    fn default() -> Self {
        Self {
            open: false,
            // Keep performance and bake indicators visible by default.
            bottom: vec![TechnicalMetric::Fps, TechnicalMetric::CpuFrame, TechnicalMetric::Bake],
        }
    }
}

impl TechnicalUi {
    fn settings_path() -> Option<std::path::PathBuf> {
        Some(super::project_io::data_dir()?.join("technical-ui.json"))
    }

    pub(super) fn load() -> Self {
        Self::settings_path()
            .and_then(|path| std::fs::read_to_string(path).ok())
            .and_then(|text| serde_json::from_str::<Self>(&text).ok())
            .unwrap_or_default()
            .sanitized()
    }

    fn sanitized(mut self) -> Self {
        let mut unique = Vec::with_capacity(self.bottom.len().min(TechnicalMetric::ALL.len()));
        for metric in self.bottom.drain(..) {
            if TechnicalMetric::ALL.contains(&metric) && !unique.contains(&metric) {
                unique.push(metric);
            }
        }
        self.bottom = if unique.is_empty() {
            vec![TechnicalMetric::Fps, TechnicalMetric::CpuFrame, TechnicalMetric::Bake]
        } else {
            unique
        };
        self
    }

    pub(super) fn save(&self) {
        let Some(path) = Self::settings_path() else { return };
        let Some(dir) = path.parent() else { return };
        if std::fs::create_dir_all(dir).is_err() {
            return;
        }
        if let Ok(text) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(path, text);
        }
    }

    fn add(&mut self, metric: TechnicalMetric) {
        if !self.bottom.contains(&metric) {
            self.bottom.push(metric);
            self.save();
        }
    }

    fn remove(&mut self, metric: TechnicalMetric) {
        self.bottom.retain(|item| *item != metric);
        self.save();
    }

    fn move_by(&mut self, index: usize, delta: isize) {
        let target = index as isize + delta;
        if index >= self.bottom.len() || target < 0 || target >= self.bottom.len() as isize {
            return;
        }
        self.bottom.swap(index, target as usize);
        self.save();
    }
}

impl PainterApp {
    /// The active Engine first, then each parked Engine that is still loaded.
    /// Read-only: it only queries `vram_bytes()`.
    fn gpu_memory_entries(&self) -> Vec<SetMemory> {
        let name = |index: usize| {
            self.mesh.sets.get(index).map_or_else(|| format!("(missing set {index})"), |set| set.name.clone())
        };
        let mut entries = vec![SetMemory {
            name: name(self.sets.active),
            parked: false,
            bytes: self.engine.vram_bytes(),
        }];
        for (index, slot) in self.sets.parked.iter().enumerate() {
            if index == self.sets.active {
                continue;
            }
            if let Some(parked) = slot {
                entries.push(SetMemory { name: name(index), parked: true, bytes: parked.engine.vram_bytes() });
            }
        }
        entries
    }

    pub(super) fn technical_info_panel(&mut self, ui: &mut egui::Ui) {
        if let Some(report) = crate::startup_trace::take_completed_report() {
            self.technical_ui.capture_report = Some(format!(
                "{}\n\nTEST CONTEXT\nbuild_profile={}\n{}",
                report,
                if cfg!(debug_assertions) { "debug (not suitable for FPS comparison)" } else { "release" },
                self.technical_report(),
            ));
            self.technical_ui.capture_status = Some("Capture completed: 1,200 frames recorded.".into());
        }
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Technical Info");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("×").on_hover_text("Close technical panel").clicked() {
                        self.technical_ui.open = false;
                        self.technical_ui.save();
                    }
                });
            });
            ui.horizontal(|ui| {
                if ui.button("Copy complete report").on_hover_text("Copies runtime, GPU, bake, F6, mesh, UV, normals and tangent diagnostics").clicked() {
                    ui.ctx().copy_text(self.technical_report());
                    self.status = "Complete technical report copied to clipboard.".into();
                }
                if ui.button("Save .txt").clicked()
                    && let Some(path) = rfd::FileDialog::new()
                        .set_file_name("MaterialPainter-Technical-Report.txt")
                        .save_file()
                {
                    match std::fs::write(&path, self.technical_report()) {
                        Ok(()) => self.status = format!("Technical report saved to {}.", path.display()),
                        Err(error) => self.status = format!("Could not save technical report: {error}"),
                    }
                }
            });
            ui.label(RichText::new("The copied report is intentionally exhaustive. It includes detailed mesh and UV analysis, not only the values shown below.")
                .small().color(theme::TEXT_DIM));
            ui.separator();

            egui::CollapsingHeader::new("Viewport performance")
                .default_open(true)
                .show(ui, |ui| {
                    ui.label(format!("Viewport FPS (frame interval): {}", TechnicalMetric::Fps.short_value(self)));
                    ui.label(format!("Frame interval: {:.3} ms", self.frame_interval_ms));
                    ui.label(format!("Application CPU frame: {:.2} ms average · {:.2} ms recent peak", self.frame_ms, self.frame_peak_ms));
                    ui.label(format!("3D viewport: {:?} px", self.views.view3d.size));
                    ui.label(format!("UV viewport: {:?} px", self.views.view2d.size));
                    ui.label(format!("Pixels per point: {:.3}", self.ppp));
                    ui.label(format!("Ray-traced preview: {}", if self.rt.enabled { "enabled" } else { "disabled" }));
                    ui.label(RichText::new("FPS is the measured UI/viewport update interval, not a dedicated GPU timestamp query.")
                        .small().color(theme::TEXT_DIM));
                });
            
            egui::CollapsingHeader::new("Frame-time test · 1,200 frames")
                .default_open(true)
                .show(ui, |ui| {
                    ui.label("Run three separate captures using the same project, viewport size and display settings. Each capture ends automatically after 1,200 frames (about 20–60 seconds at 20–60 FPS). Use the Release build for meaningful FPS results.");
                    ui.label(RichText::new("Idle: do not touch the mouse or keyboard. Camera: orbit continuously at a steady pace. Painting: make continuous strokes with the same brush and texture set.")
                        .small().color(theme::TEXT_DIM));
                    let scenarios = ["Idle — no interaction", "Camera — continuous orbit", "Painting — continuous strokes"];
                    if crate::startup_trace::capture_active() {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("CAPTURING").strong().color(theme::ACCENT));
                            ui.label(format!("{} / 1,200 frames", crate::startup_trace::capture_frame_count()));
                            if ui.button("Stop and keep partial result").clicked() {
                                if let Some(report) = crate::startup_trace::stop_capture() {
                                    self.technical_ui.capture_report = Some(format!(
                                        "{}\n\nTEST CONTEXT\nbuild_profile={}\n{}",
                                        report,
                                        if cfg!(debug_assertions) { "debug (not suitable for FPS comparison)" } else { "release" },
                                        self.technical_report(),
                                    ));
                                    self.technical_ui.capture_status = Some("Partial capture stopped and retained.".into());
                                }
                            }
                        });
                        let progress = (crate::startup_trace::capture_frame_count() as f32 / 1200.0).clamp(0.0, 1.0);
                        ui.add(egui::ProgressBar::new(progress).desired_width(ui.available_width()));
                    } else {
                        egui::ComboBox::from_label("Scenario")
                            .selected_text(scenarios[self.technical_ui.capture_scenario.min(2)])
                            .show_ui(ui, |ui| {
                                for (index, name) in scenarios.iter().enumerate() {
                                    ui.selectable_value(&mut self.technical_ui.capture_scenario, index, *name);
                                }
                            });
                        if ui.button("Start 1,200-frame capture").clicked() {
                            self.technical_ui.capture_report = None;
                            let started = crate::startup_trace::begin_capture(1200, scenarios[self.technical_ui.capture_scenario.min(2)]);
                            self.technical_ui.capture_status = Some(if started {
                                "Capture started. Keep the selected activity consistent until it completes.".into()
                            } else {
                                "A capture is already active; finish it before starting another.".into()
                            });
                        }
                    }
                    if let Some(status) = &self.technical_ui.capture_status {
                        ui.label(RichText::new(status).small().color(theme::TEXT_DIM));
                    }
                    if let Some(report) = &self.technical_ui.capture_report {
                        ui.separator();
                        ui.label(RichText::new("Latest capture report is ready.").strong());
                        ui.horizontal(|ui| {
                            if ui.button("Copy test report").clicked() {
                                ui.ctx().copy_text(report.clone());
                                self.status = "Frame-time test report copied to clipboard.".into();
                            }
                            if ui.button("Save test report .txt").clicked()
                                && let Some(path) = rfd::FileDialog::new()
                                    .set_file_name("MaterialPainter-Frame-Time-Test.txt")
                                    .save_file()
                            {
                                match std::fs::write(&path, report) {
                                    Ok(()) => self.status = format!("Frame-time report saved to {}.", path.display()),
                                    Err(error) => self.status = format!("Could not save frame-time report: {error}"),
                                }
                            }
                        });
                    }
                });

            egui::CollapsingHeader::new("Bake information")
                .default_open(true)
                .show(ui, |ui| {
                    if let Some((progress, phase)) = self.bake_progress() {
                        ui.add(egui::ProgressBar::new(progress).desired_width(ui.available_width())
                            .text(format!("{phase} {:.1}%", progress * 100.0)));
                    } else {
                        ui.label("No bake is currently running.");
                    }
                    if let Some(report) = &self.engine.bake_report {
                        ui.label(format!("Source: {}", report.source));
                        ui.label(format!("Maps: {}", report.baked.count_ones()));
                        ui.label(format!("Mesh: {} triangles · {} parts · {} feature edges", report.triangles, report.parts, report.hard_edges));
                        ui.label(format!("Resolution: {}² · {} rays/texel", report.resolution, report.rays_per_texel));
                        if report.from_cache {
                            ui.label("Latest maps loaded from bake cache.");
                        } else if let Some(gpu) = report.seconds() {
                            ui.label(format!("GPU wall time: {:.3} s · CPU preparation: {:.3} s", gpu, report.prep_seconds));
                        } else {
                            ui.label("Bake timing is still being finalized.");
                        }
                        if let Some(missed) = report.missed_percent() {
                            ui.label(format!("Texels missed by all rays: {missed:.4}%"));
                        }
                        if let Some(filled) = report.filled_percent() {
                            ui.label(format!("Texels filled from nearest high-poly surface: {filled:.4}%"));
                        }
                    } else {
                        ui.label("No completed bake report is available.");
                    }
                    ui.label(format!("Configured bake options: {:#?}", self.bake_settings));
                });

            egui::CollapsingHeader::new("Memory and undo")
                .default_open(true)
                .show(ui, |ui| {
                    let mb = |bytes: u64| bytes as f64 / (1024.0 * 1024.0);
                    let entries = self.gpu_memory_entries();
                    let totals = MemoryTotals::of(&entries);
                    ui.label(format!("GPU resource allocation estimate: {:.1} MiB", mib(totals.total)));
                    ui.label(RichText::new("Internal application-side estimate of textures and known buffers for resident Engines; it is not a direct reading of physical VRAM. Driver padding, shared renderer resources and the viewport mesh are excluded.")
                        .small().color(theme::TEXT_DIM));
                    ui.label(RichText::new("Shared Fill image textures can appear in more than one per-Engine subtotal. The document-wide Undo readback cache is shown separately.")
                        .small().color(theme::TEXT_DIM));
                    egui::Grid::new("technical_gpu_memory_by_set").num_columns(3).striped(true).show(ui, |ui| {
                        ui.label(RichText::new("Texture set").strong());
                        ui.label(RichText::new("State").strong());
                        ui.label(RichText::new("Estimate").strong());
                        ui.end_row();
                        for entry in &entries {
                            ui.label(&entry.name);
                            ui.label(if entry.parked { "PARKED" } else { "ACTIVE" });
                            ui.label(format!("{:.1} MiB", mib(entry.bytes)));
                            ui.end_row();
                        }
                    });
                    ui.label(format!("Active Engine subtotal: {:.1} MiB", mib(totals.active)));
                    ui.label(format!("Parked Engines subtotal: {:.1} MiB", mib(totals.parked)));
                    ui.label(format!("Total of known resident Engines: {:.1} MiB", mib(totals.total)));
                    ui.label(format!(
                        "Document-shared Undo readback cache: {:.1} MiB",
                        mib(self.engine.snapshot_buffer_pool_cached_bytes()),
                    ));
                    ui.label(RichText::new("Host-visible staging buffers; physical memory location depends on the graphics backend and driver.")
                        .small().color(theme::TEXT_DIM));
                    ui.label(format!("Undo RAM: {:.1} MB / {:.1} MB", mb(self.undo.ram_bytes() as u64), mb(self.undo.max_ram_bytes as u64)));
                    ui.label(format!("Undo steps retained: {}", self.undo.undo.len()));
                    ui.label(format!("Active texture resolution: {}²", self.engine.res));
                    ui.label(format!("Texture sets with parked engines: {}", self.sets.parked.iter().filter(|item| item.is_some()).count()));
                });

            egui::CollapsingHeader::new("Mesh and texture sets")
                .default_open(true)
                .show(ui, |ui| {
                    ui.label(format!("Mesh: {}", self.mesh.name));
                    ui.label(format!("Vertices after tangent split: {}", self.mesh.vertices.len()));
                    ui.label(format!("Triangles: {}", self.mesh.triangle_count()));
                    ui.label(format!("Indices: {}", self.mesh.indices.len()));
                    ui.label(format!("UV edge segments: {}", self.mesh.uv_edges.len() / 2));
                    ui.label(format!("UV seam segments: {}", self.mesh.seam_edges.len() / 2));
                    ui.label(format!("Texture sets: {}", self.mesh.sets.len()));
                    ui.label(format!("Imported objects: {}", self.mesh.objects.len()));
                    ui.label(format!("Visible imported objects: {}", self.hidden_objects.iter().filter(|hidden| !**hidden).count()));
                    ui.label(format!("Active texture set: {} ({})", self.sets.active, self.mesh.sets.get(self.sets.active).map_or("(missing)", |set| set.name.as_str())));
                    if let Some(report) = &self.engine.bake_report {
                        ui.label(format!("Bake connected parts: {}", report.parts));
                    }
                });

            egui::CollapsingHeader::new("GPU and system")
                .default_open(false)
                .show(ui, |ui| {
                    let info = self.rs.adapter.get_info();
                    ui.label(format!("GPU adapter: {}", info.name));
                    ui.label(format!("Backend: {:?}", info.backend));
                    ui.label(format!("Device type: {:?}", info.device_type));
                    ui.label(format!("Driver: {} · {}", info.driver, info.driver_info));
                    ui.label(format!("OS / architecture: {} / {}", std::env::consts::OS, std::env::consts::ARCH));
                    ui.label(format!("Application version: {}", env!("CARGO_PKG_VERSION")));
                    ui.collapsing("Device limits", |ui| {
                        ui.label(format!("{:#?}", self.rs.device.limits()));
                    });
                });

            egui::CollapsingHeader::new("F6 paint profiler")
                .default_open(false)
                .show(ui, |ui| {
                    ui.label(if self.meter.on { "F6 profiler is enabled." } else { "F6 profiler is disabled. Press F6 to enable it." });
                    if self.meter.on {
                        let details = self.meter.text(
                            self.engine.res,
                            self.engine.last_paint_scissor_pixels,
                            self.engine.last_paint_scissor_rects,
                            self.engine.last_scissor_pixels,
                            self.engine.last_scissor_bbox_pixels,
                            self.engine.last_scissor_rects,
                            self.engine.last_composite_layer_passes,
                            self.engine.last_composite_folder_passes,
                            self.engine.last_mask_cache_hits,
                            self.engine.last_mask_cache_misses,
                            self.engine.last_mask_gpu_passes,
                            self.engine.last_fill_variation_cache_hits,
                            self.engine.last_fill_variation_cache_misses,
                            self.engine.last_fill_variation_cache_gpu_passes,
                        );
                        ui.add(egui::Label::new(RichText::new(details).small()).wrap());
                    }
                });

            ui.separator();
            ui.heading("Bottom status bar");
            ui.label(RichText::new("Use the arrows to add, remove and reorder the values shown along the bottom.")
                .small().color(theme::TEXT_DIM));
            ui.label(RichText::new("Pinned metrics").strong());
            let pinned = self.technical_ui.bottom.clone();
            let mut action: Option<(u8, TechnicalMetric, usize)> = None;
            for (index, metric) in pinned.iter().copied().enumerate() {
                ui.horizontal(|ui| {
                    ui.label(metric.label());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("×").on_hover_text("Remove from bottom bar").clicked() {
                            action = Some((0, metric, index));
                        } else if ui.small_button("↓").on_hover_text("Move right").clicked() {
                            action = Some((2, metric, index));
                        } else if ui.small_button("↑").on_hover_text("Move left").clicked() {
                            action = Some((1, metric, index));
                        }
                    });
                });
            }
            if let Some((kind, metric, index)) = action {
                match kind {
                    0 => self.technical_ui.remove(metric),
                    1 => self.technical_ui.move_by(index, -1),
                    2 => self.technical_ui.move_by(index, 1),
                    _ => {}
                }
            }

            ui.add_space(5.0);
            ui.label(RichText::new("Available metrics").strong());
            let mut add_metric = None;
            for metric in TechnicalMetric::ALL {
                if self.technical_ui.bottom.contains(&metric) {
                    continue;
                }
                ui.horizontal(|ui| {
                    ui.label(metric.label()).on_hover_text(metric.description());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("↓").on_hover_text("Add to bottom bar").clicked() {
                            add_metric = Some(metric);
                        }
                    });
                });
            }
            if let Some(metric) = add_metric {
                self.technical_ui.add(metric);
            }
        });
    }

    pub(super) fn technical_metrics_bar(&mut self, ui: &mut egui::Ui) {
        let metrics = self.technical_ui.bottom.clone();
        egui::ScrollArea::horizontal()
            .id_salt("technical_bottom_metrics")
            .auto_shrink([false, false])
            .max_height(24.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.set_min_height(22.0);
                    for (index, metric) in metrics.into_iter().enumerate() {
                        if index > 0 {
                            ui.separator();
                        }
                        match metric {
                            TechnicalMetric::Bake => {
                                if let Some((progress, phase)) = self.bake_progress() {
                                    ui.add(egui::ProgressBar::new(progress).desired_width(175.0)
                                        .text(format!("Bake · {phase} {:.0}%", progress * 100.0)));
                                } else if let Some(report) = &self.engine.bake_report {
                                    if report.from_cache {
                                        ui.label("Bake · cache");
                                    } else if let Some(gpu) = report.seconds() {
                                        ui.label(format!("Bake · {:.2}s GPU + {:.2}s prep", gpu, report.prep_seconds));
                                    } else {
                                        ui.label("Bake · finished");
                                    }
                                } else {
                                    ui.label("Bake · —");
                                }
                            }
                            _ => {
                                let text = metric.short_value(self);
                                let label = ui.label(RichText::new(text).small());
                                if metric == TechnicalMetric::Fps {
                                    label.on_hover_text("Estimated from the real interval between application/viewport frames; not a GPU timestamp query.");
                                } else {
                                    label.on_hover_text(metric.description());
                                }
                            }
                        }
                    }
                });
            });
    }

    fn technical_report(&self) -> String {
        let info = self.rs.adapter.get_info();
        let gpu_bytes = self.sets.parked.iter().flatten().fold(
            self.engine.vram_bytes(),
            |total, parked| total.saturating_add(parked.engine.vram_bytes()),
        );
        let shared_undo_staging_bytes = self.engine.snapshot_buffer_pool_cached_bytes();
        let undo_bytes = self.undo.ram_bytes() as u64;
        let bake = self.engine.bake_report.as_ref().map_or_else(
            || "No completed bake report.".to_string(),
            |report| format!(
                "source={}\ntriangles={}\nparts={}\nfeature_edges={}\nresolution={}\nrays_per_texel={}\nfrom_cache={}\ngpu_seconds={:?}\nprep_seconds={:.6}\nmissed_percent={:?}\nfilled_percent={:?}\nbaked_map_mask={:#010x}\n",
                report.source,
                report.triangles,
                report.parts,
                report.hard_edges,
                report.resolution,
                report.rays_per_texel,
                report.from_cache,
                report.seconds(),
                report.prep_seconds,
                report.missed_percent(),
                report.filled_percent(),
                report.baked,
            ),
        );
        let f6 = if self.meter.on {
            self.meter.text(
                self.engine.res,
                self.engine.last_paint_scissor_pixels,
                self.engine.last_paint_scissor_rects,
                self.engine.last_scissor_pixels,
                self.engine.last_scissor_bbox_pixels,
                self.engine.last_scissor_rects,
                self.engine.last_composite_layer_passes,
                self.engine.last_composite_folder_passes,
                self.engine.last_mask_cache_hits,
                self.engine.last_mask_cache_misses,
                self.engine.last_mask_gpu_passes,
                self.engine.last_fill_variation_cache_hits,
                self.engine.last_fill_variation_cache_misses,
                self.engine.last_fill_variation_cache_gpu_passes,
            )
        } else {
            "disabled".to_string()
        };

        format!(
            "{}\n\nAPPLICATION / FRAME\nframe_interval_ms={:.6}\nviewport_fps_estimate={:.3}\nui_frame_cpu_ema_ms={:.6}\nui_frame_recent_peak_ms={:.6}\nstatus={}\n\nGPU ADAPTER\n{:#?}\n\nGPU DEVICE LIMITS\n{:#?}\ngpu_resource_allocation_estimate_bytes={}\ndocument_shared_undo_readback_cache_bytes={}\nundo_memory_bytes={}\nundo_memory_limit_bytes={}\nundo_steps={}\n\nBAKE REPORT\n{}\nCURRENT BAKE SETTINGS\n{:#?}\n\nF6 PAINT PROFILER\n{}\n\nMESH / UV / NORMAL / TANGENT DIAGNOSTICS\n{}",
            self.diagnostic_context_report("technical panel copy"),
            self.frame_interval_ms,
            if self.frame_interval_ms > 0.01 { 1000.0 / self.frame_interval_ms } else { 0.0 },
            self.frame_ms,
            self.frame_peak_ms,
            self.status.replace('\n', " | "),
            info,
            self.rs.device.limits(),
            gpu_bytes,
            shared_undo_staging_bytes,
            undo_bytes,
            self.undo.max_ram_bytes as u64,
            self.undo.undo.len(),
            bake,
            self.bake_settings,
            f6,
            self.diagnostic_mesh_report(),
        )
    }
}

#[cfg(test)]
mod memory_breakdown_tests {
    use super::*;

    fn entry(name: &str, parked: bool, bytes: u64) -> SetMemory {
        SetMemory { name: name.into(), parked, bytes }
    }

    #[test]
    fn totals_split_active_and_parked() {
        let totals = MemoryTotals::of(&[
            entry("Body", false, 100),
            entry("Wheels", true, 30),
            entry("Glass", true, 12),
        ]);
        assert_eq!(totals, MemoryTotals { active: 100, parked: 42, total: 142 });
    }

    #[test]
    fn single_set_has_no_parked_bytes() {
        let totals = MemoryTotals::of(&[entry("Body", false, 5)]);
        assert_eq!(totals, MemoryTotals { active: 5, parked: 0, total: 5 });
    }

    #[test]
    fn no_entries_gives_zero() {
        assert_eq!(MemoryTotals::of(&[]), MemoryTotals::default());
    }

    #[test]
    fn mib_uses_binary_units() {
        assert_eq!(mib(1024 * 1024), 1.0);
        assert_eq!(mib(3 * 512 * 1024), 1.5);
    }

    #[test]
    fn totals_saturate_instead_of_wrapping() {
        let totals = MemoryTotals::of(&[
            entry("Body", false, u64::MAX),
            entry("Glass", true, u64::MAX),
        ]);
        assert_eq!(totals, MemoryTotals { active: u64::MAX, parked: u64::MAX, total: u64::MAX });
    }

    #[test]
    fn settings_remove_non_adjacent_duplicate_metrics() {
        let ui = TechnicalUi {
            open: true,
            bottom: vec![
                TechnicalMetric::Fps,
                TechnicalMetric::Bake,
                TechnicalMetric::Fps,
                TechnicalMetric::F6Meter,
                TechnicalMetric::Bake,
            ],
        };
        assert_eq!(
            ui.sanitized().bottom,
            vec![TechnicalMetric::Fps, TechnicalMetric::Bake, TechnicalMetric::F6Meter],
        );
    }
}
