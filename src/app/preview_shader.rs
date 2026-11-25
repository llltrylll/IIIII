//! Preview shaders: a shading graph (see `graph::shading`) that replaces the
//! 3D view's built-in look while you texture, e.g. toon, clay, unlit, or
//! the look of a target engine. Edited in the Material Designer like any
//! graph; chosen in the Preview panel; saved with the project.

use super::{PainterApp, Workspace, theme};
use crate::graph::{self, shading};
use crate::icons;
use eframe::egui;

/// Inputs that change the generated WGSL. Node positions and parameter values
/// are deliberately excluded: they affect the editor layout or uniforms, not code.
#[derive(Clone, Debug, PartialEq, Eq)]
struct PreviewCompileKey {
    nodes: Vec<(u64, String, Vec<(String, u64)>)>,
    definitions: Vec<(String, Option<String>)>,
}

impl PreviewCompileKey {
    fn new(graph: &graph::Graph, lib: &graph::Library) -> Self {
        let nodes = graph.nodes.iter().map(|node| {
            let mut links: Vec<(String, u64)> = node.links.iter()
                .map(|(input, source)| (input.clone(), *source))
                .collect();
            links.sort_unstable();
            (node.id, node.def.clone(), links)
        }).collect();

        let mut definition_ids: Vec<String> =
            graph.nodes.iter().map(|node| node.def.clone()).collect();
        definition_ids.sort_unstable();
        definition_ids.dedup();

        let definitions = definition_ids.into_iter().map(|id| {
            let source = lib.find(&id).map(|definition| definition.source.clone());
            (id, source)
        }).collect();

        Self { nodes, definitions }
    }
}

/// What the app keeps about the preview shader in use.
#[derive(Default)]
pub(super) struct PreviewShader {
    /// The shading graph in use (a graph id), or the built-in look.
    pub graph: Option<u64>,
    /// The code the view runs now (recompiled only when it changes).
    code: String,
    compiled: Option<shading::Compiled>,
    /// Last graph/definition structure used to generate the compiled WGSL.
    compile_key: Option<PreviewCompileKey>,
    pub error: Option<String>,
}

impl PainterApp {
    /// Compiles the preview shader when its graph changed, and sends its
    /// parameter values (every frame: they're just uniforms).
    pub(super) fn update_preview_shader(&mut self) {
        // Editing a shading graph in the Material Designer previews it.
        if self.workspace == Workspace::Designer
            && let Some(doc) = self.graphs.editor.doc
            && self.graphs.doc(doc).is_some_and(|d| d.shading)
        {
            self.preview.graph = Some(doc);
        }
        let graph = self.preview.graph.and_then(|id| self.graphs.doc(id)).map(|d| &d.graph);
        let Some(graph) = graph else {
            if self.views.has_custom_shader() {
                let _ = self.views.set_custom_shader(&self.rs.device, None);
            }
            self.preview = Default::default();
            return;
        };
        // Compiling walks the graph, allocates strings and rebuilds the WGSL
        // source. Only repeat that work when a node, connection or definition
        // changed. Parameter values are read separately and remain live uniforms.
        let compile_key = PreviewCompileKey::new(graph, &self.graphs.lib);
        if self.preview.compile_key.as_ref() != Some(&compile_key) {
            self.preview.compiled = None;
            self.preview.compile_key = Some(compile_key);
            match shading::compile(graph, &self.graphs.lib) {
                Ok(compiled) => {
                    self.preview.compiled = Some(compiled);
                    self.preview.error = None;
                }
                Err(e) => self.preview.error = Some(e),
            }
        }

        let Some(compiled) = self.preview.compiled.as_ref() else {
            // Keep a compile error cached for this exact graph structure; don't
            // rebuild the same failing WGSL on every UI frame.
            return;
        };
        if compiled.code != self.preview.code || !self.views.has_custom_shader() {
            match self.views.set_custom_shader(&self.rs.device, Some(&compiled.code)) {
                Ok(()) => self.preview.error = None,
                Err(e) => self.preview.error = Some(e),
            }
            self.preview.code = compiled.code.clone();
        }
        let values = shading::param_values(compiled, graph, &self.graphs.lib);
        self.views.set_custom_params(&self.rs.queue, &values);
    }

    /// Starts a preview shader from preset `i` (offscreen UI tests); returns
    /// its graph id.
    #[cfg(test)]
    pub(crate) fn test_preview_preset(&mut self, i: usize) -> u64 {
        self.new_preview_shader(shading::presets().swap_remove(i));
        self.preview.graph.expect("set")
    }

    /// The preview shader's compile error, if any (offscreen UI tests).
    #[cfg(test)]
    pub(crate) fn test_preview_error(&self) -> Option<String> {
        self.preview.error.clone()
    }

    /// Adds a shading graph (a preset, or the starter) and makes it the
    /// preview shader.
    fn new_preview_shader(&mut self, graph: crate::graph::Graph) {
        let id = self.graphs.insert(&self.rs, graph, None);
        self.preview.graph = Some(id);
        self.status = "Added a preview shader. Edit its graph in the Material Designer; the 3D view uses it while you texture.".into();
    }

    /// The Preview panel's section for choosing and editing the shader.
    pub(super) fn preview_shader_section(&mut self, ui: &mut egui::Ui) {
        let shaders: Vec<(u64, String)> = self.graphs.docs.iter().filter(|d| d.shading).map(|d| (d.id, d.graph.name.clone())).collect();
        let current = self.preview.graph.and_then(|id| shaders.iter().find(|(s, _)| *s == id)).map_or("Default (PBR)".to_string(), |(_, n)| n.clone());
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Shader").color(theme::TEXT_DIM));
            egui::ComboBox::from_id_salt("preview shader").selected_text(current).width(ui.available_width() - 66.0).show_ui(ui, |ui| {
                if ui.selectable_label(self.preview.graph.is_none(), "Default (PBR)").clicked() {
                    self.preview.graph = None;
                }
                for (id, name) in &shaders {
                    if ui.selectable_label(self.preview.graph == Some(*id), name).clicked() {
                        self.preview.graph = Some(*id);
                    }
                }
            });
            if ui.add_enabled(self.preview.graph.is_some(), egui::Button::new("Edit")).on_hover_text("Open its graph in the Material Designer").clicked()
                && let Some(id) = self.preview.graph
            {
                self.open_graph(id);
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("New").small().color(theme::TEXT_DIM));
            for g in shading::presets() {
                if ui.small_button(&g.name).on_hover_text("Start a preview shader from this preset").clicked() {
                    self.new_preview_shader(g);
                }
            }
            if ui.small_button(theme::with_icon(icons::FILE_PLUS, "Blank")).on_hover_text("The PBR look as a graph, to change step by step").clicked() {
                self.new_preview_shader(shading::starter("Preview shader"));
            }
        });
        if let Some(e) = &self.preview.error {
            ui.label(egui::RichText::new(format!("The shader doesn't compile: {e}")).small().color(egui::Color32::from_rgb(240, 110, 100)));
        } else {
            ui.label(
                egui::RichText::new("Only changes how the 3D view looks while you work; exports are unaffected. Ray tracing uses the default look.")
                    .small()
                    .color(theme::TEXT_DIM),
            );
        }
    }
}


#[cfg(test)]
mod preview_compile_key_tests {
    use super::PreviewCompileKey;
    use crate::graph::{self, Value};

    fn clay_graph() -> graph::Graph {
        graph::shading::presets().into_iter()
            .find(|graph| graph.name == "Clay")
            .expect("Clay preview preset")
    }

    #[test]
    fn parameter_values_and_canvas_positions_do_not_invalidate_compilation() {
        let lib = graph::Library::load(None);
        let mut graph = clay_graph();
        let before = PreviewCompileKey::new(&graph, &lib);

        graph.name = "Renamed".into();
        graph.res = 4096;
        graph.nodes[0].pos = [1234.0, -987.0];
        graph.nodes[0].params.insert(
            "color".into(),
            Value::Color([0.1, 0.2, 0.3]),
        );

        assert_eq!(before, PreviewCompileKey::new(&graph, &lib));
    }

    #[test]
    fn changed_node_connections_invalidate_compilation() {
        let lib = graph::Library::load(None);
        let mut graph = clay_graph();
        let before = PreviewCompileKey::new(&graph, &lib);

        let output = graph.nodes.iter_mut()
            .find(|node| node.def == graph::shading::OUTPUT)
            .expect("shader output");
        output.links.clear();

        assert_ne!(before, PreviewCompileKey::new(&graph, &lib));
    }

    #[test]
    fn changed_node_definition_source_invalidates_compilation() {
        let mut lib = graph::Library::load(None);
        let graph = clay_graph();
        let before = PreviewCompileKey::new(&graph, &lib);

        let definition = lib.defs.iter_mut()
            .find(|definition| definition.id == "shade_color")
            .expect("built-in shade_color definition");
        definition.source.push_str("\n// cache invalidation test");

        assert_ne!(before, PreviewCompileKey::new(&graph, &lib));
    }
}
