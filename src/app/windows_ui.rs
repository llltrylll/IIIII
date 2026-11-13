//! The About, crash-report and Help windows.
use super::*;

impl PainterApp {
    pub(super) fn about_window(&mut self, ctx: &egui::Context) {
        let mut open = self.show_about;
        egui::Window::new("About").open(&mut open).collapsible(false).resizable(false).show(ctx, |ui| {
            ui.set_width(360.0);
            ui.label(egui::RichText::new("Material Painter").heading().color(Color32::WHITE));
            ui.label(egui::RichText::new(format!("Version {}", env!("CARGO_PKG_VERSION"))).color(theme::TEXT_DIM));
            if let Some(gpu) = crate::crash::context() {
                // Selectable, so it can be copied into bug reports.
                ui.add(egui::Label::new(egui::RichText::new(format!("GPU: {gpu}")).color(theme::TEXT_DIM)).selectable(true));
            }
            ui.add_space(8.0);
            ui.label("A PBR texture painter: layers, smart materials and masks, mesh map baking, material graphs and custom node shaders.");
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new("Free software under the GNU General Public License, version 3 or later. It comes with no warranty.")
                    .small()
                    .color(theme::TEXT_DIM),
            );
            ui.label(
                egui::RichText::new("Bundled HDRIs and textures: Poly Haven (CC0). Icons: Phosphor (MIT).").small().color(theme::TEXT_DIM),
            );
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("Source code").clicked() {
                    ui.ctx().open_url(egui::OpenUrl::new_tab(REPO_URL));
                }
                if ui.button("Third-party licenses").clicked() {
                    match notices_path() {
                        Some(p) => ui.ctx().open_url(egui::OpenUrl::new_tab(format!("file:///{}", p.display().to_string().replace('\\', "/")))),
                        None => ui.ctx().open_url(egui::OpenUrl::new_tab(format!("{REPO_URL}/blob/master/THIRD_PARTY_NOTICES.html"))),
                    }
                }
            });
        });
        self.show_about = open;
    }

    /// Tells about a crash in the last run, once, with its report.
    pub(super) fn crash_window(&mut self, ctx: &egui::Context) {
        let Some(path) = self.crash_report.clone() else { return };
        let mut close = false;
        egui::Window::new("Material Painter closed unexpectedly").collapsible(false).resizable(false).show(ctx, |ui| {
            ui.set_width(420.0);
            ui.label("Last time, the app stopped because of an error. A report was saved; sending it helps fix the problem.");
            ui.add_space(4.0);
            ui.label(egui::RichText::new(path.display().to_string()).small().monospace().color(theme::TEXT_DIM));
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui.button("Open the report").clicked() {
                    ui.ctx().open_url(egui::OpenUrl::new_tab(format!("file:///{}", path.display().to_string().replace('\\', "/"))));
                }
                if ui.button("Report the problem").on_hover_text("Opens a new issue on the project page; attach the report").clicked() {
                    ui.ctx().open_url(egui::OpenUrl::new_tab(format!("{REPO_URL}/issues/new")));
                }
                if ui.button("Close").clicked() {
                    close = true;
                }
            });
        });
        if close {
            self.crash_report = None;
        }
    }

    pub(super) fn help_window(&mut self, ctx: &egui::Context) {
        let mut open = self.show_help;
        egui::Window::new("Controls").open(&mut open).collapsible(false).resizable(false).show(ctx, |ui| {
            egui::Grid::new("help").striped(true).show(ui, |ui| {
                for (k, v) in [
                    ("LMB", "Paint / erase"),
                    ("Alt + LMB  or  RMB", "Orbit (3D) · pan (2D)"),
                    ("MMB", "Pan"),
                    ("Alt + RMB / Wheel", "Zoom"),
                    ("Shift + RMB", "Rotate environment"),
                    ("Ctrl + RMB", "Brush size (→) / hardness (↑)"),
                    ("Alt + double-click", "Set orbit center on the mesh"),
                    ("[  ]", "Brush size"),
                    ("Shift + click", "Straight line from the last stroke"),
                    ("Ctrl+G", "Material Designer (the selected layer's graph)"),
                    ("Ctrl+PageUp / PageDown", "Switch workspace"),
                    ("1 / 2 / I", "Paint / Eraser / Pick material"),
                    ("X", "Toggle X symmetry"),
                    ("F", "Frame mesh"),
                    ("M / C", "Material view / cycle channels"),
                    ("F1 / F2 / F3", "3D+2D / 3D / 2D"),
                    ("Ctrl+F1", "Save viewport diagnostic snapshot"),
                    ("Ctrl+F2", "Start / stop ray-hit diagnostic capture"),
                    ("Ctrl+Z / Ctrl+Y", "Undo / redo"),
                    ("Ctrl+S / Ctrl+Shift+S", "Save / save as"),
                    ("Ctrl+O", "Open project"),
                    ("Ctrl+Shift+E", "Export textures"),
                    ("Double-click (shelf)", "Add to the layer stack"),
                    ("Drag (shelf)", "Drop on a view or in the layer stack"),
                    ("Ctrl + wheel (shelf)", "Thumbnail size"),
                    ("Drag (layers)", "Reorder; drop on a folder's middle to put it inside"),
                ] {
                    ui.label(egui::RichText::new(k).strong());
                    ui.label(v);
                    ui.end_row();
                }
            });
        });
        self.show_help = open;
    }
}
