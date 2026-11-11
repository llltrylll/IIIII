//! The Shader Maker workspace: write custom node shaders (WGSL with a
//! header, see `graph::def`) against a live preview, with test values for
//! the parameters and compiler errors pointing at lines of the file.

use super::graph_ui::{param_widget, unique_stem};
use super::{Action, PainterApp, Workspace, labeled, section, theme};
use crate::graph::{self, Graph, Library, Runtime};
use eframe::egui;
use egui::{Color32, Rect, pos2, vec2};

const PREVIEW_DEF: &str = "preview";

/// What feeds the preview's inputs (built-in node ids).
const PREVIEW_INPUTS: [(&str, &str); 5] =
    [("noise", "Noise"), ("gradient", "Gradient"), ("checker", "Checker"), ("cells", "Cells"), ("tiles", "Tiles")];

/// Starting points for a new shader: (file stem, menu label, source).
pub(super) const TEMPLATES: [(&str, &str, &str); 4] = [
    (
        "my_filter",
        "Filter (changes an input)",
        "// @name My Filter
// @category Custom
// @description Tints its input toward a color, keeping its light and dark.
// @output color
// @input input: color = 0.5
// @param tint: color = 0.35, 0.75, 1
// @param amount: float(0, 1) = 0.7

fn node(uv: vec2<f32>) -> vec4<f32> {
    let c = in_input(uv);
    let tinted = params.tint * luma(c.rgb) * 1.6;
    return vec4<f32>(mix(c.rgb, tinted, params.amount), 1.0);
}
",
    ),
    (
        "my_generator",
        "Generator (makes a pattern)",
        "// @name My Generator
// @category Custom
// @description Makes a tileable pattern from nothing: rings of noise around soft cells.
// @output gray
// @param scale: int(1, 64) = 6
// @param rings: float(1, 20) = 6
// @param seed: int(0, 1000) = 0

fn node(uv: vec2<f32>) -> vec4<f32> {
    let c = cells(uv, f32(params.scale), 1.0, f32(params.seed));
    let n = fbm2(uv, f32(params.scale) * 2.0, 4, 0.5, f32(params.seed) + 7.0);
    let r = 0.5 + 0.5 * sin((c.x + n * 0.3) * params.rings * TAU);
    return vec4<f32>(r);
}
",
    ),
    (
        "my_blend",
        "Blend (combines two inputs)",
        "// @name My Blend
// @category Custom
// @description Combines a top and a bottom input through a mask.
// @output color
// @input top: color = 1
// @input bottom: color = 0
// @input mask: gray = 0.5
// @param opacity: float(0, 1) = 1

fn node(uv: vec2<f32>) -> vec4<f32> {
    let m = clamp(in_mask(uv) * params.opacity, 0.0, 1.0);
    return vec4<f32>(mix(in_bottom(uv).rgb, in_top(uv).rgb, m), 1.0);
}
",
    ),
    (
        "my_mask",
        "Mask (from a baked mesh map)",
        "// @name My Mask
// @category Custom
// @description A mask from a baked mesh map, for graph masks.
// @output gray
// @param map: meshmap
// @param threshold: float(0, 1) = 0.5
// @param softness: float(0, 0.5) = 0.1
// @param invert: bool = false

fn node(uv: vec2<f32>) -> vec4<f32> {
    var m = smoothstep(params.threshold - params.softness, params.threshold + params.softness, map(uv));
    if (params.invert) {
        m = 1.0 - m;
    }
    return vec4<f32>(m);
}
",
    ),
];

/// Things to insert at the cursor: (label, what it does, text).
const SNIPPETS: [(&str, &str, &str); 14] = [
    ("@param float", "A slider", "// @param amount: float(0, 1) = 0.5\n"),
    ("@param int", "A whole-number slider", "// @param count: int(1, 16) = 4\n"),
    ("@param color", "A color picker", "// @param tint: color = 1, 1, 1\n"),
    ("@param enum", "A dropdown", "// @param mode: enum(First, Second) = First\n"),
    ("@param bool", "A checkbox", "// @param flip: bool = false\n"),
    ("@input gray", "A grayscale input", "// @input mask: gray = 1\n"),
    ("@input color", "A color input", "// @input input: color = 0.5\n"),
    ("perlin", "Tileable gradient noise, 0..1", "perlin(uv, 8.0, 0.0)"),
    ("fbm2", "Fractal noise: scale, octaves, roughness, seed", "fbm2(uv, 6.0, 5, 0.5, 0.0)"),
    ("value_noise", "Soft tileable value noise", "value_noise(uv, 8.0, 0.0)"),
    ("cells", "Worley cells: x = distance, y = 2nd distance, z = random", "cells(uv, 8.0, 1.0, 0.0)"),
    ("rotate2", "Turns a 2D vector by an angle (radians)", "rotate2(uv - 0.5, 0.5) + 0.5"),
    ("px()", "One texel in uv, for neighbor lookups", "px()"),
    ("luma", "Brightness of a color", "luma(c.rgb)"),
];

// WGSL syntax colors.
const KEYWORDS: [&str; 18] =
    ["fn", "let", "var", "const", "if", "else", "for", "loop", "while", "return", "switch", "case", "default", "break", "continue", "struct", "true", "false"];
const TYPES: [&str; 12] = ["f32", "i32", "u32", "bool", "vec2", "vec3", "vec4", "mat2x2", "mat3x3", "mat4x4", "array", "texture_2d"];
const BUILTINS: [&str; 44] = [
    "sin", "cos", "tan", "atan2", "abs", "floor", "ceil", "fract", "round", "min", "max", "clamp", "mix", "step", "smoothstep", "length",
    "distance", "dot", "cross", "normalize", "pow", "exp", "log", "sqrt", "select", "radians", "degrees", "sign", "textureSampleLevel",
    "perlin", "fbm2", "value_noise", "cells", "rotate2", "px", "res", "luma", "hash1", "hash2", "wrap_cell", "grad_dot", "saturate", "trunc",
    "inverseSqrt",
];

/// Colors WGSL for the code editor.
fn highlight(text: &str, font: egui::FontId) -> egui::text::LayoutJob {
    let mut job = egui::text::LayoutJob::default();
    let fmt = |c: Color32| egui::TextFormat { font_id: font.clone(), color: c, ..Default::default() };
    let (plain, comment, tag, keyword, ty, builtin, number, param) = (
        Color32::from_gray(215),
        Color32::from_rgb(110, 140, 110),
        Color32::from_rgb(230, 190, 100),
        Color32::from_rgb(200, 130, 220),
        Color32::from_rgb(90, 200, 190),
        Color32::from_rgb(110, 170, 250),
        Color32::from_rgb(240, 160, 110),
        Color32::from_rgb(240, 200, 120),
    );
    for line in text.split_inclusive('\n') {
        let (code, rest) = match line.find("//") {
            Some(i) => line.split_at(i),
            None => (line, ""),
        };
        let mut chars = code.char_indices().peekable();
        while let Some((i, c)) = chars.next() {
            let mut end = i + c.len_utf8();
            if c.is_ascii_alphabetic() || c == '_' {
                while let Some(&(j, d)) = chars.peek() {
                    if d.is_ascii_alphanumeric() || d == '_' {
                        end = j + d.len_utf8();
                        chars.next();
                    } else {
                        break;
                    }
                }
                let word = &code[i..end];
                let color = if KEYWORDS.contains(&word) {
                    keyword
                } else if TYPES.contains(&word) {
                    ty
                } else if BUILTINS.contains(&word) {
                    builtin
                } else if word == "params" || word.starts_with("in_") || word.starts_with("img_") {
                    param
                } else {
                    plain
                };
                job.append(word, 0.0, fmt(color));
            } else if c.is_ascii_digit() || (c == '.' && chars.peek().is_some_and(|(_, d)| d.is_ascii_digit())) {
                while let Some(&(j, d)) = chars.peek() {
                    if d.is_ascii_alphanumeric() || d == '.' {
                        end = j + d.len_utf8();
                        chars.next();
                    } else {
                        break;
                    }
                }
                job.append(&code[i..end], 0.0, fmt(number));
            } else {
                job.append(&code[i..end], 0.0, fmt(plain));
            }
        }
        if !rest.is_empty() {
            // Header tags stand out inside comments.
            match rest.find('@') {
                Some(at) => {
                    let tag_end = rest[at..].find(char::is_whitespace).map_or(rest.len(), |e| at + e);
                    job.append(&rest[..at], 0.0, fmt(comment));
                    job.append(&rest[at..tag_end], 0.0, fmt(tag));
                    job.append(&rest[tag_end..], 0.0, fmt(comment));
                }
                None => job.append(rest, 0.0, fmt(comment)),
            }
        }
    }
    job
}

/// The first `line N` an error message points at.
fn error_line(msg: &str) -> Option<usize> {
    let i = msg.find("line ")?;
    msg[i + 5..].chars().take_while(char::is_ascii_digit).collect::<String>().parse().ok()
}

pub(super) struct ShaderMaker {
    /// The node being edited: `user/<stem>`, or a built-in (read-only).
    id: Option<String>,
    text: String,
    /// The text as last saved (or loaded), to tell edits apart.
    saved: String,
    error: Option<String>,
    /// A one-node graph (plus noise feeding its inputs) rendering the text.
    preview: Graph,
    node: u64,
    rt: Option<Runtime>,
    /// The text changed since the preview rendered it.
    stale: bool,
    /// Show the preview 2×2 to check that it tiles.
    tile: bool,
    confirm_delete: bool,
    thumb: Option<(egui::TextureId, usize)>,
    /// What feeds the preview's inputs (`PREVIEW_INPUTS`).
    input_src: usize,
}

impl ShaderMaker {
    pub fn new() -> ShaderMaker {
        ShaderMaker {
            id: None,
            text: String::new(),
            saved: String::new(),
            error: None,
            preview: Graph::new("Preview"),
            node: 0,
            rt: None,
            stale: false,
            tile: false,
            confirm_delete: false,
            thumb: None,
            input_src: 0,
        }
    }

    fn modified(&self) -> bool {
        self.text != self.saved
    }

    fn editable(&self) -> bool {
        self.id.as_deref().is_some_and(|id| id.starts_with("user/"))
    }
}

impl PainterApp {
    /// Opens a node's source in the Shader Maker.
    pub(super) fn open_code(&mut self, id: &str) {
        let lib = &self.graphs.lib;
        let text = lib.find(id).map(|d| d.source.clone()).or_else(|| lib.user_path(id).and_then(|p| std::fs::read_to_string(p).ok()));
        let Some(text) = text else { return };
        let sm = &mut self.graphs.shader;
        if sm.id.as_deref() != Some(id) {
            // A new node gets fresh test values.
            sm.preview = Graph::new("Preview");
            sm.preview.res = 256;
            sm.node = sm.preview.add(PREVIEW_DEF, [0.0, 0.0]);
        }
        sm.id = Some(id.to_string());
        sm.saved = text.clone();
        sm.text = text;
        sm.stale = true;
        sm.confirm_delete = false;
        self.workspace = Workspace::ShaderMaker;
    }

    /// A new user shader from a template, opened for editing.
    fn new_shader(&mut self, base: &str, source: &str) {
        let stem = unique_stem(&self.graphs.lib, base);
        match self.graphs.lib.save_user(&stem, source) {
            Ok(id) => self.open_code(&id),
            Err(e) => self.status = format!("Couldn't create the shader: {e}"),
        }
    }

    /// User node files changed on disk: an unedited open file follows them.
    pub(super) fn shader_files_changed(&mut self) {
        let sm = &self.graphs.shader;
        let Some(id) = sm.id.clone().filter(|_| sm.editable() && !sm.modified()) else { return };
        if let Some(text) = self.graphs.lib.user_path(&id).and_then(|p| std::fs::read_to_string(p).ok()) {
            let sm = &mut self.graphs.shader;
            if text != sm.text {
                sm.saved = text.clone();
                sm.text = text;
                sm.stale = true;
            }
        }
    }

    /// Renders the edited text when it changed. Runs before drawing.
    pub(super) fn update_shader_preview(&mut self) {
        if !self.graphs.shader.stale || self.graphs.shader.id.is_none() {
            return;
        }
        let sm = &mut self.graphs.shader;
        sm.stale = false;
        let def = match graph::def::parse(PREVIEW_DEF, &sm.text) {
            Ok(d) => d,
            Err(e) => {
                sm.error = Some(e);
                return;
            }
        };
        // A shading node makes no texture: it shows in the 3D view as part of
        // a preview shader (Preview panel → Shader).
        if def.shading {
            sm.error = Some("This is a shading node (@shading): use it in a preview shader graph to see it in the 3D view.".into());
            return;
        }
        // Inputs get a built-in pattern, so filters have something to work on.
        let src_id = PREVIEW_INPUTS[sm.input_src.min(PREVIEW_INPUTS.len() - 1)].0;
        let src_def = self.graphs.lib.find(src_id).cloned().expect("built-in preview input");
        let lib = Library { defs: vec![src_def, def.clone()], errors: Vec::new(), user_dir: None };
        let node = sm.node;
        sm.preview.nodes.retain(|n| n.id == node);
        if let Some(n) = sm.preview.node_mut(node) {
            n.links.clear();
            n.params.retain(|k, _| def.params.iter().any(|p| &p.name == k));
        }
        if !def.inputs.is_empty() {
            let src = sm.preview.add(src_id, [-200.0, 0.0]);
            for inp in &def.inputs {
                sm.preview.connect(src, node, &inp.name);
            }
        }
        let rt = sm.rt.get_or_insert_with(|| Runtime::new(&self.rs.device, &self.rs.queue));
        let engine = &self.engine;
        let maps = |m: usize| engine.mesh_map(m);
        let surface = engine.graph_surface();
        rt.evaluate(&self.rs.device, &self.rs.queue, &sm.preview, &lib, &maps, Some(&surface), engine.maps_version, false);
        sm.error = rt.errors.get(&node).map(|e| def.map_error(e));
    }

    // ------------------------------------------------------------------ workspace

    pub(super) fn shader_maker(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().frame(egui::Frame::new().fill(theme::BG).inner_margin(egui::Margin::same(4))).show(ui, |ui| {
            self.dock_ui(ui, Workspace::ShaderMaker);
        });
    }

    /// The Code tab: the editor, or the start page when nothing is open.
    pub(super) fn shader_code_pane(&mut self, ui: &mut egui::Ui) {
        if self.graphs.shader.id.is_none() {
            self.shader_start(ui);
        } else {
            self.shader_editor(ui);
        }
    }

    /// The Preview tab.
    pub(super) fn shader_preview_pane(&mut self, ui: &mut egui::Ui) {
        if self.graphs.shader.id.is_none() {
            ui.label(egui::RichText::new("Open or make a shader to see it here.").color(theme::TEXT_DIM));
        } else {
            self.shader_preview(ui);
        }
    }

    /// The start page: templates, your shaders and the built-ins, as cards.
    fn shader_start(&mut self, ui: &mut egui::Ui) {
        use super::gallery::{card, heading, node_key, template_key};
        self.ensure_gallery();
        let mut open = None;
        let mut new = None;
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            heading(ui, "Make a shader", "Write a node for the Material Designer in WGSL. The preview updates as you type.");
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = vec2(12.0, 12.0);
                for (stem, label, src) in TEMPLATES {
                    let (title, desc) = label.split_once(" (").map_or((label, ""), |(t, d)| (t, d.trim_end_matches(')')));
                    if card(ui, self.gallery_thumb(&template_key(stem)), title, desc, 168.0).clicked() {
                        new = Some((stem, src));
                    }
                }
            });
            let lib = &self.graphs.lib;
            let user: Vec<_> = lib.defs.iter().filter(|d| d.id.starts_with("user/")).collect();
            if !user.is_empty() {
                heading(ui, "Your shaders", "");
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = vec2(12.0, 12.0);
                    for d in user {
                        if card(ui, self.gallery_thumb(&node_key(&d.id)), &d.name, &d.description, 140.0).clicked() {
                            open = Some(d.id.clone());
                        }
                    }
                });
            }
            heading(ui, "Learn from the built-in nodes", "They are written the same way. Open one to read it, then make an editable copy.");
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = vec2(10.0, 10.0);
                for d in lib.defs.iter().filter(|d| !d.id.starts_with("user/") && !d.shading) {
                    if card(ui, self.gallery_thumb(&node_key(&d.id)), &d.name, &d.category, 112.0).on_hover_text(&d.description).clicked() {
                        open = Some(d.id.clone());
                    }
                }
            });
            ui.add_space(20.0);
        });
        if let Some((stem, src)) = new {
            self.new_shader(stem, src);
        }
        if let Some(id) = open {
            self.open_code(&id);
        }
    }

    pub(super) fn shader_list(&mut self, ui: &mut egui::Ui) {
        if ui.add(theme::selectable(self.graphs.shader.id.is_none(), "⌂ Start page")).clicked() {
            self.graphs.shader.id = None;
        }
        ui.label(egui::RichText::new("SHADERS").small().color(theme::TEXT_DIM));
        ui.horizontal(|ui| {
            ui.menu_button("New shader ▾", |ui| {
                for (stem, label, src) in TEMPLATES {
                    if ui.button(label).clicked() {
                        self.new_shader(stem, src);
                        ui.close();
                    }
                }
            });
            if ui.button("Folder").on_hover_text("Your shaders live here; edits from other editors reload automatically").clicked()
                && let Some(dir) = &self.graphs.lib.user_dir {
                    super::open_in_file_browser(dir);
                }
        });
        let open = self.graphs.shader.id.clone();
        let mut pick = None;
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            section(ui, "Your shaders");
            let lib = &self.graphs.lib;
            let mut any = false;
            for d in lib.defs.iter().filter(|d| d.id.starts_with("user/")) {
                any = true;
                if ui.selectable_label(open.as_deref() == Some(d.id.as_str()), &d.name).on_hover_text(&d.id).clicked() {
                    pick = Some(d.id.clone());
                }
            }
            for (path, err) in &lib.errors {
                any = true;
                let id = format!("user/{}", path.file_stem().unwrap_or_default().to_string_lossy());
                let text = egui::RichText::new(format!("⚠ {}", id.trim_start_matches("user/"))).color(Color32::from_rgb(240, 130, 90));
                if ui.selectable_label(open.as_deref() == Some(id.as_str()), text).on_hover_text(err).clicked() {
                    pick = Some(id);
                }
            }
            if !any {
                ui.label(egui::RichText::new("None yet.").color(theme::TEXT_DIM));
            }
            section(ui, "Built-in (read-only)");
            for (cat, ids) in lib.categories() {
                if cat == "Custom" {
                    continue;
                }
                egui::CollapsingHeader::new(&cat).id_salt(("builtin", cat.as_str())).show(ui, |ui| {
                    for id in ids.iter().filter(|id| !id.starts_with("user/")) {
                        let d = lib.find(id).expect("listed");
                        if ui.selectable_label(open.as_deref() == Some(id.as_str()), &d.name).on_hover_text(&d.description).clicked() {
                            pick = Some(id.clone());
                        }
                    }
                });
            }
        });
        if let Some(id) = pick {
            self.open_code(&id);
        }
    }

    fn shader_editor(&mut self, ui: &mut egui::Ui) {
        let editable = self.graphs.shader.editable();
        let mut save = false;
        let mut revert = false;
        let mut delete = false;
        let mut to_graph = false;
        let mut copy = None;
        {
            let sm = &mut self.graphs.shader;
            let id = sm.id.clone().unwrap_or_default();
            ui.horizontal(|ui| {
                let title = format!("{}{}", id.trim_start_matches("user/"), if sm.modified() { " *" } else { "" });
                ui.label(egui::RichText::new(title).heading().color(Color32::WHITE));
                if !editable {
                    ui.label(egui::RichText::new("built-in, read-only").color(theme::TEXT_DIM));
                }
                ui.separator();
                if editable {
                    save = ui.add_enabled(sm.modified(), egui::Button::new("Save")).on_hover_text("Ctrl+S").clicked();
                    revert = ui.add_enabled(sm.modified(), egui::Button::new("Revert")).clicked();
                } else if ui.button("Make an editable copy").clicked() {
                    copy = Some(id.clone());
                }
                to_graph = ui
                    .add_enabled(self.graphs.editor.doc.is_some() && !sm.modified(), egui::Button::new("Add to graph"))
                    .on_hover_text("Puts this node into the graph open in the Material Designer (save first)")
                    .clicked();
                if editable {
                    if sm.confirm_delete {
                        ui.label(egui::RichText::new("Delete the file?").color(Color32::from_rgb(240, 130, 90)));
                        delete = ui.button("Yes, delete").clicked();
                        if ui.button("No").clicked() {
                            sm.confirm_delete = false;
                        }
                    } else if ui.button("Delete…").clicked() {
                        sm.confirm_delete = true;
                    }
                }
            });
            match &sm.error {
                Some(e) => {
                    egui::Frame::new().fill(Color32::from_rgb(60, 28, 24)).inner_margin(egui::Margin::same(6)).show(ui, |ui| {
                        egui::ScrollArea::vertical().id_salt("shader errors").max_height(110.0).show(ui, |ui| {
                            ui.label(egui::RichText::new(e).monospace().small().color(Color32::from_rgb(250, 160, 140)));
                        });
                    });
                }
                None => {
                    ui.label(egui::RichText::new("✔ Compiles. The preview updates as you type.").small().color(Color32::from_rgb(120, 200, 120)));
                }
            }
            ui.add_space(4.0);
            // Line numbers beside the text, in one scroll area so they stay aligned.
            let lines = sm.text.lines().count().max(1) + usize::from(sm.text.ends_with('\n'));
            let font = egui::TextStyle::Monospace.resolve(ui.style());
            let avail = ui.available_height() - 26.0;
            let bad = sm.error.as_deref().and_then(error_line);
            egui::ScrollArea::vertical().id_salt("shader code").max_height(avail).auto_shrink([false, false]).show(ui, |ui| {
                ui.horizontal_top(|ui| {
                    // Line numbers; the one an error points at in red.
                    let mut gutter = egui::text::LayoutJob::default();
                    for n in 1..=lines {
                        let (color, bg) = if Some(n) == bad {
                            (Color32::WHITE, Color32::from_rgb(170, 50, 40))
                        } else {
                            (theme::TEXT_DIM, Color32::TRANSPARENT)
                        };
                        gutter.append(&format!("{n:>4}\n"), 0.0, egui::TextFormat { font_id: font.clone(), color, background: bg, ..Default::default() });
                    }
                    ui.add(egui::Label::new(gutter).selectable(false));
                    let mut layouter = |ui: &egui::Ui, buf: &dyn egui::TextBuffer, wrap: f32| {
                        let mut job = highlight(buf.as_str(), font.clone());
                        job.wrap.max_width = wrap;
                        ui.painter().layout_job(job)
                    };
                    let r = ui.add_enabled(
                        true,
                        egui::TextEdit::multiline(&mut sm.text)
                            .id(egui::Id::unique("shader code edit"))
                            .code_editor()
                            .interactive(editable)
                            .desired_width(f32::INFINITY)
                            .desired_rows(lines.max(30))
                            .lock_focus(true)
                            .layouter(&mut layouter),
                    );
                    if r.changed() {
                        sm.stale = true;
                    }
                    if editable && r.has_focus() && ui.input_mut(|i| i.consume_shortcut(&egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::S))) {
                        save = true;
                    }
                });
            });
            ui.label(
                egui::RichText::new(
                    "Header: @name @category @description, @output color|gray, @input name: color|gray = default, @param name: float(min,max) | int(min,max) | bool | color | enum(A,B) | image | meshmap = default.  Body: fn node(uv: vec2<f32>) -> vec4<f32>, with in_<input>(uv), params.<name>, img_<image>(uv), <meshmap>(uv), res(), px(), perlin, fbm2, value_noise, cells, rotate2, luma.",
                )
                .small()
                .color(theme::TEXT_DIM),
            );
        }
        let sm = &mut self.graphs.shader;
        let id = sm.id.clone().unwrap_or_default();
        if revert {
            sm.text = sm.saved.clone();
            sm.stale = true;
        }
        if save {
            let stem = id.trim_start_matches("user/").to_string();
            let text = sm.text.clone();
            match self.graphs.lib.save_user(&stem, &text) {
                Ok(_) => self.status = format!("Saved {stem}.wgsl."),
                Err(e) => self.status = format!("Saved {stem}.wgsl, but it has errors: {e}"),
            }
            self.graphs.shader.saved = text;
            for d in &mut self.graphs.docs {
                // Graphs using it render with the new code.
                if d.graph.nodes.iter().any(|n| n.def == id) {
                    self.pending.push(Action::GraphRefresh(d.id));
                }
            }
        }
        if delete {
            if let Some(path) = self.graphs.lib.user_path(&id) {
                let _ = std::fs::remove_file(path);
            }
            self.graphs.lib.reload_user();
            self.graphs.shader.id = None;
            self.status = format!("Deleted {}.", id.trim_start_matches("user/"));
        }
        if let Some(from) = copy {
            self.new_custom_node(Some(&from), false);
        }
        if to_graph {
            self.pending.push(Action::GraphAddNode(id));
        }
    }

    fn shader_preview(&mut self, ui: &mut egui::Ui) {
        let custom = self.bake_settings.custom.clone();
        // Keep the preview's egui texture current.
        let sm = &mut self.graphs.shader;
        if let Some(t) = sm.rt.as_ref().and_then(|rt| rt.thumb(sm.node)) {
            let ptr = std::sync::Arc::as_ptr(t) as usize;
            let mut renderer = self.rs.renderer.write();
            match &mut sm.thumb {
                Some((_, p)) if *p == ptr => {}
                Some((tex, p)) => {
                    renderer.update_egui_texture_from_wgpu_texture(&self.rs.device, &t.view, eframe::wgpu::FilterMode::Linear, *tex);
                    *p = ptr;
                }
                None => sm.thumb = Some((renderer.register_native_texture(&self.rs.device, &t.view, eframe::wgpu::FilterMode::Linear), ptr)),
            }
        }
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("PREVIEW").small().color(theme::TEXT_DIM));
            ui.checkbox(&mut sm.tile, "Tile 2×2").on_hover_text("Check that it repeats without seams");
            let before = sm.input_src;
            egui::ComboBox::from_id_salt("preview input")
                .width(90.0)
                .selected_text(PREVIEW_INPUTS[sm.input_src].1)
                .show_ui(ui, |ui| {
                    for (k, (_, name)) in PREVIEW_INPUTS.iter().enumerate() {
                        ui.selectable_value(&mut sm.input_src, k, *name);
                    }
                })
                .response
                .on_hover_text("What feeds the node's inputs in the preview");
            if sm.input_src != before {
                sm.stale = true;
            }
        });
        let size = ui.available_width().min(320.0);
        let (rect, _) = ui.allocate_exact_size(vec2(size, size), egui::Sense::hover());
        let p = ui.painter();
        p.rect_filled(rect, 0.0, Color32::from_gray(20));
        let failed = sm.error.is_some() && sm.rt.as_ref().is_none_or(|rt| rt.output(sm.node).is_none());
        match sm.thumb.filter(|_| !failed) {
            Some((tex, _)) => {
                let full = Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));
                if sm.tile {
                    let h = size * 0.5;
                    for k in 0..4 {
                        let min = rect.min + vec2((k % 2) as f32 * h, (k / 2) as f32 * h);
                        p.image(tex, Rect::from_min_size(min, vec2(h, h)), full, Color32::WHITE);
                    }
                } else {
                    p.image(tex, rect, full, Color32::WHITE);
                }
            }
            None => {
                p.text(rect.center(), egui::Align2::CENTER_CENTER, "No preview: fix the errors", egui::FontId::proportional(13.0), theme::TEXT_DIM);
            }
        }
        ui.add_space(6.0);
        // Insert snippets at the editor's cursor.
        egui::CollapsingHeader::new("Insert").id_salt("shader snippets").show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                for (label, tip, text) in SNIPPETS {
                    if ui.small_button(label).on_hover_text(tip).clicked() && sm.editable() {
                        insert_at_cursor(ui.ctx(), &mut sm.text, text);
                        sm.stale = true;
                    }
                }
            });
        });
        let def = graph::def::parse(PREVIEW_DEF, &sm.text).ok();
        let Some(def) = def else { return };
        if !def.inputs.is_empty() {
            ui.label(egui::RichText::new("Inputs are fed with noise in the preview.").small().color(theme::TEXT_DIM));
        }
        if def.params.is_empty() {
            return;
        }
        section(ui, "Test values");
        let mut changed = false;
        egui::ScrollArea::vertical().id_salt("shader params").show(ui, |ui| {
            let node = sm.node;
            for p in &def.params {
                let Some(n) = sm.preview.node_mut(node) else { break };
                let mut v = n.params.get(&p.name).filter(|v| std::mem::discriminant(*v) == std::mem::discriminant(&p.default)).cloned().unwrap_or_else(|| p.default.clone());
                if labeled(ui, &p.name.replace('_', " "), |ui| param_widget(ui, p, &mut v, 0, &custom)) {
                    n.params.insert(p.name.clone(), v);
                    changed = true;
                }
            }
        });
        if changed {
            sm.stale = true;
        }
    }
}

/// Inserts `text` at the code editor's cursor (or the end) and moves the
/// cursor past it.
fn insert_at_cursor(ctx: &egui::Context, buf: &mut String, text: &str) {
    use egui::text::{CCursor, CCursorRange};
    let id = egui::Id::unique("shader code edit");
    let mut state = egui::text_edit::TextEditState::load(ctx, id);
    let chars = buf.chars().count();
    let at = state.as_ref().and_then(|s| s.cursor.char_range()).map_or(chars, |r| r.primary.index.0.min(chars));
    let byte = buf.char_indices().nth(at).map_or(buf.len(), |(b, _)| b);
    buf.insert_str(byte, text);
    if let Some(s) = &mut state {
        s.cursor.set_char_range(Some(CCursorRange::one(CCursor::new(at + text.chars().count()))));
        s.clone().store(ctx, id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn highlighting_keeps_the_text_and_colors_it() {
        for (_, _, src) in TEMPLATES {
            let job = highlight(src, egui::FontId::monospace(12.0));
            assert_eq!(job.text, src);
            assert!(job.sections.len() > 20);
            // Every template parses as a node.
            graph::def::parse("t", src).unwrap();
        }
        let job = highlight("let x = 1.5; // @param a", egui::FontId::monospace(12.0));
        let color = |word: &str| {
            let at = job.text.find(word).unwrap();
            job.sections.iter().find(|s| (s.byte_range.start.0..s.byte_range.end.0).contains(&at)).unwrap().format.color
        };
        assert_ne!(color("let"), color("x"));
        assert_ne!(color("1.5"), color("x"));
        assert_ne!(color("@param"), color("//"));
    }

    #[test]
    fn finds_the_error_line() {
        assert_eq!(error_line("Shader 'preview' parsing error: no definition ┌─ line 12:5"), Some(12));
        assert_eq!(error_line("line 3: unknown tag '@foo'"), Some(3));
        assert_eq!(error_line("generated code"), None);
    }
}
