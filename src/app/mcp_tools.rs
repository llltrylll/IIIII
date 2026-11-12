//! MCP tool handlers. They run on the UI thread against the live document and
//! record into the same undo history as the UI (see `crate::mcp`).

use super::{Layout, PainterApp, add_smart_material};
use crate::bake::Quality;
use crate::engine::{Blend, BrushPaint, Content, GEN_INPUTS, GenKind, Generator, MAPS, MAX_STAMPS, StampUniform};
use crate::undo::StrokeTarget;
use crate::mcp::{Call, Content as Out, ToolResult};
use crate::mesh::{Mesh, Primitive};
use crate::presets;
use crate::select;
use crate::environment::{self, EnvSource};
use crate::post::Tonemap;
use crate::viewport::{Background, ViewMode, ViewRenderer, ViewSettings};
use eframe::egui;
use serde_json::{Value, json};

/// When a tool's answer is ready.
enum Reply {
    Now(ToolResult),
    /// After the running bake finishes.
    AfterBake,
    /// The window screenshot callback answers it.
    AfterScreenshot,
    /// After the HDRI being loaded is in use.
    AfterEnvironment,
}

fn text(t: impl Into<String>) -> Reply {
    Reply::Now(Ok(vec![Out::Text(t.into())]))
}

fn json_text(v: Value) -> Reply {
    text(serde_json::to_string_pretty(&v).unwrap_or_default())
}

fn png(rgba: &[u8], w: u32, h: u32) -> Result<Vec<u8>, String> {
    use image::ImageEncoder;
    let mut out = Vec::new();
    image::codecs::png::PngEncoder::new(&mut out)
        .write_image(rgba, w, h, image::ExtendedColorType::Rgba8)
        .map_err(|e| e.to_string())?;
    Ok(out)
}

fn eq(a: &str, b: &str) -> bool {
    a.trim().eq_ignore_ascii_case(b.trim())
}

fn f32_arg(args: &Value, key: &str) -> Option<f32> {
    args.get(key).and_then(Value::as_f64).map(|v| v as f32)
}

fn str_arg<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(Value::as_str)
}

fn view_mode(name: &str) -> Result<ViewMode, String> {
    ViewMode::all().into_iter().find(|m| eq(m.name(), name)).ok_or_else(|| format!("unknown view mode '{name}'"))
}

impl PainterApp {
    /// Runs queued MCP calls and completes the ones waiting on a bake.
    pub(super) fn process_mcp(&mut self, ctx: &egui::Context) {
        let Some(server) = &self.mcp else { return };
        let calls: Vec<Call> = server.calls.try_iter().collect();
        for call in calls {
            if self.stroke.is_some() {
                let _ = call.reply.send(Err("Busy: the user is painting a stroke. Try again in a moment.".into()));
                continue;
            }
            match self.run_tool(&call.tool, &call.args, ctx) {
                Reply::Now(r) => {
                    let _ = call.reply.send(r);
                }
                Reply::AfterBake => self.mcp_bake_waiters.push(call.reply),
                Reply::AfterEnvironment => self.mcp_env_waiters.push(call.reply),
                Reply::AfterScreenshot => {
                    let reply = call.reply;
                    let callback = egui::ScreenshotCallback::new(move |image| {
                        let [w, h] = image.size;
                        let _ = reply.send(png(image.as_raw(), w as u32, h as u32).map(|p| vec![Out::Png(p)]));
                    });
                    ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(callback));
                    ctx.request_repaint();
                }
            }
        }

        let finished = self.bake_job.is_none() && self.engine.bake_report.as_ref().is_some_and(|r| r.finished());
        if finished && !self.mcp_bake_waiters.is_empty() {
            let report = self.bake_summary();
            for w in self.mcp_bake_waiters.drain(..) {
                let _ = w.send(Ok(vec![Out::Text(report.to_string())]));
            }
        }
    }

    /// Fails every MCP call waiting on a bake (the bake errored or was dropped).
    pub(super) fn fail_mcp_bake(&mut self, error: &str) {
        for w in self.mcp_bake_waiters.drain(..) {
            let _ = w.send(Err(error.to_string()));
        }
    }

    fn bake_summary(&self) -> Value {
        match &self.engine.bake_report {
            Some(r) => json!({
                "source": r.source,
                "matched_names": r.matched_names,
                "used_cage": r.used_cage,
                "triangles": r.triangles,
                "parts": r.parts,
                "feature_edges": r.hard_edges,
                "resolution": r.resolution,
                "rays_per_texel": r.rays_per_texel,
                "edge_search_distance": r.edge_max_distance,
                "gpu_seconds": r.seconds(),
                "missed_texels_percent": r.missed_percent(),
                "id_regions": r.id_regions.iter().enumerate().map(|(i, g)| json!({
                    "id": i,
                    "color": g.color,
                    "triangles": g.triangles,
                    "min": g.min,
                    "max": g.max,
                })).collect::<Vec<_>>(),
            }),
            None => Value::Null,
        }
    }

    fn layer_index(&self, args: &Value) -> Result<usize, String> {
        let n = self.engine.layers.len();
        match args.get("index").and_then(Value::as_u64) {
            Some(i) if (i as usize) < n => Ok(i as usize),
            Some(i) => Err(format!("layer index {i} out of range (0..{n})")),
            None => Err("missing 'index'".into()),
        }
    }

    fn run_tool(&mut self, tool: &str, args: &Value, ctx: &egui::Context) -> Reply {
        match self.try_tool(tool, args, ctx) {
            Ok(r) => r,
            Err(e) => Reply::Now(Err(e)),
        }
    }

    fn try_tool(&mut self, tool: &str, args: &Value, ctx: &egui::Context) -> Result<Reply, String> {
        Ok(match tool {
            "get_state" => json_text(self.state_json()),
            "list_presets" => json_text(json!({
                "smart_materials": presets::SMART_MATERIALS.iter().map(|s| s.name).collect::<Vec<_>>(),
                "materials": presets::MATERIALS.iter().map(|m| m.name).collect::<Vec<_>>(),
                "generators": GenKind::ALL.iter().map(|g| g.name()).collect::<Vec<_>>(),
                "generator_inputs": GEN_INPUTS.iter().map(|(n, d)| json!({"name": n, "description": d})).collect::<Vec<_>>(),
                "blend_modes": Blend::ALL.iter().map(|b| b.name()).collect::<Vec<_>>(),
                "view_modes": ViewMode::all().iter().map(|m| m.name()).collect::<Vec<_>>(),
                "export_presets": self.export.templates.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
                "bake_qualities": ["Draft", "Standard", "High", "Ultra"],
                "brush_tips": crate::brushes::BRUSH_ALPHAS.iter().map(|b| b.name).collect::<Vec<_>>(),
                "textures": crate::textures::TEXTURES.iter().map(|t| t.name).collect::<Vec<_>>(),
                "poly_fill_modes": crate::polyfill::FillMode::ALL.iter().map(|m| m.key()).collect::<Vec<_>>(),
            })),
            "load_mesh" => {
                let mesh = if let Some(path) = str_arg(args, "path") {
                    Mesh::load_file(std::path::Path::new(path))?
                } else if let Some(name) = str_arg(args, "primitive") {
                    Primitive::ALL.iter().find(|p| eq(p.name(), name)).ok_or(format!("unknown primitive '{name}'"))?.build()
                } else {
                    return Err("give 'path' or 'primitive'".into());
                };
                self.load_mesh(mesh, ctx);
                Reply::AfterBake
            }
            "bake" => {
                if let Some(q) = str_arg(args, "quality") {
                    let q = Quality::ALL
                        .into_iter()
                        .find(|x| x.name().to_ascii_lowercase().starts_with(&q.trim().to_ascii_lowercase()))
                        .ok_or(format!("unknown quality '{q}'"))?;
                    self.bake_settings.apply_quality(q);
                }
                if let Some(hp) = str_arg(args, "high_poly") {
                    self.bake_settings.high_poly = (!hp.is_empty()).then(|| hp.into());
                }
                if let Some(c) = str_arg(args, "cage") {
                    self.bake_settings.cage = (!c.is_empty()).then(|| c.into());
                }
                if let Some(v) = f32_arg(args, "front") {
                    self.bake_settings.front = v.clamp(0.0005, 2.0);
                }
                if let Some(v) = f32_arg(args, "rear") {
                    self.bake_settings.rear = v.clamp(0.0005, 2.0);
                }
                if let Some(m) = args.get("match_by_name").and_then(Value::as_bool) {
                    self.bake_settings.match_by_name = m;
                }
                if let Some(aa) = args.get("antialiasing").and_then(Value::as_u64) {
                    self.bake_settings.aa = aa.clamp(1, 8) as u32;
                }
                match str_arg(args, "id_source") {
                    Some("parts") => self.bake_settings.id_source = crate::bake::IdSource::MeshParts,
                    Some("materials") => self.bake_settings.id_source = crate::bake::IdSource::Materials,
                    Some("vertex_colors") => self.bake_settings.id_source = crate::bake::IdSource::VertexColors,
                    Some(x) => return Err(format!("id_source must be parts, materials or vertex_colors, not '{x}'")),
                    None => {}
                }
                if let Some(r) = args.get("resolution").and_then(Value::as_u64) {
                    let r = r as u32;
                    if !super::RESOLUTIONS.contains(&r) {
                        return Err(format!("resolution must be one of {:?}", super::RESOLUTIONS));
                    }
                    self.engine.set_resolution(r, &self.gpu_mesh, &mut self.undo);
                }
                self.start_bake(ctx);
                Reply::AfterBake
            }
            "add_layer" => {
                let at = args.get("index").and_then(Value::as_u64).map_or(self.engine.layers.len(), |i| i as usize);
                let preset = str_arg(args, "preset");
                let name = str_arg(args, "name").map(str::to_string);
                let at = at.min(self.engine.layers.len());
                let parent = self.engine.parent_for_insert(at);
                let idx = match str_arg(args, "kind").unwrap_or("paint") {
                    "paint" => self.engine.add_paint_layer(at, parent, name, &mut self.undo),
                    "folder" => self.engine.add_folder(at, parent, name.unwrap_or_else(|| "Folder".into()), &mut self.undo),
                    "fill" => {
                        let p = preset.unwrap_or("Iron Raw");
                        let m = presets::MATERIALS.iter().find(|m| eq(m.name, p)).ok_or(format!("unknown material '{p}'"))?;
                        self.engine.add_fill_layer(at, parent, name.unwrap_or_else(|| m.name.to_string()), m.fill(), &mut self.undo)
                    }
                    "texture" => {
                        let p = preset.ok_or("texture layers need 'preset' (a texture name)")?;
                        let t = crate::textures::TEXTURES.iter().position(|t| eq(t.name, p)).ok_or(format!("unknown texture '{p}'"))?;
                        self.add_texture_layer(t, at, parent);
                        self.selected
                    }
                    "smart" => {
                        let p = preset.ok_or("smart layers need 'preset'")?;
                        let s = presets::SMART_MATERIALS.iter().find(|s| eq(s.name, p)).ok_or(format!("unknown smart material '{p}'"))?;
                        add_smart_material(&mut self.engine, &mut self.undo, s, at, parent)
                    }
                    k => return Err(format!("unknown kind '{k}'")),
                };
                self.selected = idx;
                text(format!("Added layer at index {idx}."))
            }
            "update_layer" => {
                let i = self.layer_index(args)?;
                let blend = match str_arg(args, "blend") {
                    Some(b) => Some(
                        Blend::ALL
                            .into_iter()
                            .find(|x| eq(x.name(), b) || eq(x.short(), b) || format!("{x:?}").eq_ignore_ascii_case(b))
                            .ok_or(format!("unknown blend mode '{b}'"))?,
                    ),
                    None => None,
                };
                // Images first: a bad path fails the call before anything changes.
                let mut images = Vec::new();
                if let Some(obj) = args.get("images").and_then(Value::as_object) {
                    for (key, v) in obj {
                        const NAMES: [&str; crate::engine::CHANNELS] = ["color", "roughness", "metallic", "height", "opacity", "emission", "normal", "specular"];
                        let ch = NAMES.iter().position(|c| c == key).ok_or(format!("unknown image channel '{key}' ({})", NAMES.join(", ")))?;
                        let img = match v.as_str() {
                            None => None,
                            Some(path) => {
                                let bytes = std::fs::read(path).map_err(|e| format!("can't read {path}: {e}"))?;
                                let name = std::path::Path::new(path).file_name().map_or("image".into(), |n| n.to_string_lossy().into_owned());
                                Some(self.engine.decode_channel_image(ch, name, std::sync::Arc::new(bytes))?)
                            }
                        };
                        images.push((ch, img));
                    }
                }
                let layer = &mut self.engine.layers[i];
                if let Some(n) = str_arg(args, "name") {
                    layer.name = n.to_string();
                }
                if let Some(v) = args.get("visible").and_then(Value::as_bool) {
                    layer.visible = v;
                }
                if let Some(o) = f32_arg(args, "opacity") {
                    layer.opacity = o.clamp(0.0, 1.0);
                }
                if let Some(b) = blend {
                    layer.blend = b;
                }
                const FILL_KEYS: [&str; 19] = [
                    "images",
                    "emission",
                    "specular",
                    "normal_strength",
                    "projection",
                    "tiling",
                    "rotation",
                    "offset",
                    "triplanar_sharpness",
                    "color",
                    "roughness",
                    "metallic",
                    "height",
                    "color_variation",
                    "roughness_variation",
                    "height_variation",
                    "variation_scale",
                    "variation_seed",
                    "part_variation",
                ];
                let material = FILL_KEYS.iter().any(|k| args.get(k).is_some());
                match &mut layer.content {
                    Content::Fill(f) => {
                        if let Some(c) = args.get("color").and_then(Value::as_array) {
                            for (k, v) in c.iter().take(3).enumerate() {
                                f.mat.color[k] = v.as_f64().unwrap_or(0.0).clamp(0.0, 1.0) as f32;
                            }
                        }
                        if let Some(r) = f32_arg(args, "roughness") {
                            f.mat.rough = r.clamp(0.0, 1.0);
                        }
                        if let Some(m) = f32_arg(args, "metallic") {
                            f.mat.metal = m.clamp(0.0, 1.0);
                        }
                        if let Some(h) = f32_arg(args, "height") {
                            f.mat.height = h.clamp(-1.0, 1.0);
                            f.mat.enabled[3] = true;
                        }
                        if let Some(c) = args.get("emission").and_then(Value::as_array) {
                            for (k, v) in c.iter().take(3).enumerate() {
                                f.mat.emission[k] = v.as_f64().unwrap_or(0.0).clamp(0.0, 1.0) as f32;
                            }
                            f.mat.enabled[crate::engine::EMISSION] = true;
                        }
                        if let Some(v) = f32_arg(args, "specular") {
                            f.mat.specular = v.clamp(0.0, 1.0);
                            f.mat.enabled[crate::engine::SPECULAR] = true;
                        }
                        if let Some(v) = f32_arg(args, "normal_strength") {
                            f.mat.normal = v.clamp(0.0, 4.0);
                        }
                        if let Some(v) = f32_arg(args, "color_variation") {
                            f.color_var = v.clamp(0.0, 1.0);
                        }
                        if let Some(v) = f32_arg(args, "roughness_variation") {
                            f.rough_var = v.clamp(0.0, 1.0);
                        }
                        if let Some(v) = f32_arg(args, "height_variation") {
                            f.height_var = v.clamp(0.0, 1.0);
                            f.mat.enabled[3] = true;
                        }
                        if let Some(v) = f32_arg(args, "variation_scale") {
                            f.var_scale = v.max(0.01);
                        }
                        if let Some(v) = f32_arg(args, "variation_seed") {
                            f.seed = v;
                        }
                        if let Some(v) = f32_arg(args, "part_variation") {
                            f.part_var = v.clamp(0.0, 1.0);
                        }
                        if let Some(p) = str_arg(args, "projection") {
                            f.projection = crate::engine::Projection::from_key(p).ok_or(format!(
                                "projection must be uv, triplanar, planar_x, planar_y, planar_z, spherical or cylindrical, not '{p}'"
                            ))?
                        }
                        if let Some(v) = f32_arg(args, "tiling") {
                            f.tiling = v.max(0.001);
                        }
                        if let Some(v) = f32_arg(args, "rotation") {
                            f.rotation = v;
                        }
                        if let Some(o) = args.get("offset").and_then(Value::as_array) {
                            for (k, v) in o.iter().take(2).enumerate() {
                                f.offset[k] = v.as_f64().unwrap_or(0.0) as f32;
                            }
                        }
                        if let Some(v) = f32_arg(args, "triplanar_sharpness") {
                            f.tri_sharpness = v.clamp(1.0, 64.0);
                        }
                    }
                    Content::Paint(_) | Content::Folder(_) if material => {
                        return Err("material and variation values apply to fill layers only".into());
                    }
                    Content::Paint(_) | Content::Folder(_) => {}
                }
                self.engine.mark_dirty();
                self.engine.commit_props(&mut self.undo);
                for (ch, img) in images {
                    self.engine.set_fill_image(i, ch, img, &mut self.undo);
                }
                text(format!("Updated layer {i}."))
            }
            "delete_layer" => {
                let i = self.layer_index(args)?;
                self.engine.delete_layer(i, &mut self.undo);
                self.after_history_step();
                text(format!("Deleted layer {i}."))
            }
            "duplicate_layer" => {
                let i = self.layer_index(args)?;
                let at = self.engine.duplicate_layer(i, &mut self.undo).ok_or("nothing to duplicate")?;
                self.selected = at;
                text(format!("Duplicated into index {at}."))
            }
            "move_layer" => {
                let from = args.get("from").and_then(Value::as_u64).ok_or("missing 'from'")? as usize;
                let to = args.get("to").and_then(Value::as_u64).ok_or("missing 'to'")? as usize;
                if from >= self.engine.layers.len() {
                    return Err(format!("layer index {from} out of range"));
                }
                self.engine.move_layer(from, to, &mut self.undo);
                self.selected = to.min(self.engine.layers.len() - 1);
                text("Moved.")
            }
            "set_mask" => {
                let i = self.layer_index(args)?;
                if args.get("remove").and_then(Value::as_bool) == Some(true) {
                    self.engine.remove_mask(i, &mut self.undo);
                    self.after_history_step();
                    return Ok(text("Mask removed."));
                }
                if let Some(g) = str_arg(args, "generator") {
                    let kind = GenKind::ALL.into_iter().find(|k| eq(k.name(), g)).ok_or(format!("unknown generator '{g}'"))?;
                    let mut generator = kind.default_generator();
                    self.tune_generator(&mut generator, args)?;
                    match &mut self.engine.layers[i].mask {
                        Some(m) => {
                            m.generator = Some(generator);
                            self.engine.mark_dirty();
                            self.engine.commit_props(&mut self.undo);
                        }
                        None => self.engine.add_mask(i, true, Some(generator), &mut self.undo),
                    }
                    return Ok(text(format!("Layer {i} masked by {}.", kind.name())));
                }
                let white = match str_arg(args, "paint") {
                    Some("black") => false,
                    Some("white") | None => true,
                    Some(p) => return Err(format!("paint must be white or black, not '{p}'")),
                };
                self.engine.add_mask(i, white, None, &mut self.undo);
                text(format!("Layer {i} has a paintable mask."))
            }
            "undo" | "redo" => {
                let steps = args.get("steps").and_then(Value::as_u64).unwrap_or(1).max(1);
                let mut done = Vec::new();
                for _ in 0..steps {
                    let stack = if tool == "undo" { &self.undo.undo } else { &self.undo.redo };
                    let Some(label) = stack.last().map(|e| e.label.clone()) else { break };
                    if tool == "undo" {
                        self.engine.undo(&mut self.undo);
                    } else {
                        self.engine.redo(&mut self.undo);
                    }
                    done.push(label);
                }
                self.after_history_step();
                text(if done.is_empty() { format!("Nothing to {tool}.") } else { format!("{tool}: {}", done.join(", ")) })
            }
            "set_view" => {
                if let Some(m) = str_arg(args, "mode") {
                    self.view.mode = view_mode(m)?;
                }
                self.cam.yaw = f32_arg(args, "yaw").unwrap_or(self.cam.yaw);
                self.cam.pitch = f32_arg(args, "pitch").unwrap_or(self.cam.pitch).clamp(-1.55, 1.55);
                self.cam.dist = f32_arg(args, "distance").unwrap_or(self.cam.dist).max(0.05);
                self.cam.target = self.tool_camera(args).0.target;
                if let Some(l) = str_arg(args, "layout") {
                    self.layout = match l {
                        "split" => Layout::Split,
                        "3d" => Layout::View3d,
                        "2d" => Layout::View2d,
                        _ => return Err(format!("unknown layout '{l}'")),
                    };
                }
                if let Some(rt) = args.get("ray_trace").and_then(Value::as_bool) {
                    self.rt.enabled = rt;
                }
                if let Some(w) = args.get("wireframe").and_then(Value::as_bool) {
                    self.view.show_wire_3d = w;
                }
                text("View updated.")
            }
            "paint_stroke" => {
                let i = self.layer_index(args)?;
                let target = match str_arg(args, "target").unwrap_or("content") {
                    "content" => StrokeTarget::Content,
                    "mask" => StrokeTarget::Mask,
                    t => return Err(format!("target must be content or mask, not '{t}'")),
                };
                let layer = &self.engine.layers[i];
                if target == StrokeTarget::Content && !layer.is_paint() {
                    return Err("only paint layers take paint; use target 'mask' to paint a mask".into());
                }
                if target == StrokeTarget::Mask && layer.mask.is_none() {
                    return Err("the layer has no mask; add one with set_mask first".into());
                }
                let points: Vec<glam::Vec2> = args
                    .get("points")
                    .and_then(Value::as_array)
                    .ok_or("missing 'points'")?
                    .iter()
                    .filter_map(|p| {
                        let a = p.as_array()?;
                        Some(glam::Vec2::new(a.first()?.as_f64()? as f32, a.get(1)?.as_f64()? as f32))
                    })
                    .collect();
                if points.is_empty() {
                    return Err("'points' needs at least one [x, y] pixel position".into());
                }
                // Channels: the ones given are painted, the rest left alone.
                let mut mat = crate::engine::MatValues { enabled: [false; crate::engine::CHANNELS], ..Default::default() };
                if let Some(c) = args.get("color").and_then(Value::as_array) {
                    for (k, v) in c.iter().take(3).enumerate() {
                        mat.color[k] = v.as_f64().unwrap_or(0.0).clamp(0.0, 1.0) as f32;
                    }
                    mat.enabled[0] = true;
                }
                if let Some(c) = args.get("emission").and_then(Value::as_array) {
                    for (k, v) in c.iter().take(3).enumerate() {
                        mat.emission[k] = v.as_f64().unwrap_or(0.0).clamp(0.0, 1.0) as f32;
                    }
                    mat.enabled[crate::engine::EMISSION] = true;
                }
                if let Some(v) = f32_arg(args, "specular") {
                    mat.specular = v.clamp(0.0, 1.0);
                    mat.enabled[crate::engine::SPECULAR] = true;
                }
                for (k, key) in [(1, "roughness"), (2, "metallic"), (3, "height"), (4, "opacity")] {
                    if let Some(v) = f32_arg(args, key) {
                        match k {
                            1 => mat.rough = v.clamp(0.0, 1.0),
                            2 => mat.metal = v.clamp(0.0, 1.0),
                            4 => mat.opacity = v.clamp(0.0, 1.0),
                            _ => mat.height = v.clamp(-1.0, 1.0),
                        }
                        mat.enabled[k] = true;
                    }
                }
                let erase = args.get("erase").and_then(Value::as_bool).unwrap_or(false);
                if target == StrokeTarget::Content && !erase && !mat.enabled.contains(&true) {
                    return Err("give at least one of color, roughness, metallic, height, opacity, emission, specular".into());
                }
                let symmetry = match str_arg(args, "symmetry").unwrap_or("off") {
                    "off" => 0.0,
                    "x" => 1.0,
                    "y" => 2.0,
                    "z" => 3.0,
                    s => return Err(format!("symmetry must be off, x, y or z, not '{s}'")),
                };
                let size = f32_arg(args, "size").unwrap_or(40.0).clamp(1.0, 2000.0);
                let hardness = f32_arg(args, "hardness").unwrap_or(0.6).clamp(0.0, 0.99);
                let flow = f32_arg(args, "flow").unwrap_or(1.0).clamp(0.01, 1.0);
                let spacing = f32_arg(args, "spacing").unwrap_or(0.15).clamp(0.02, 1.0);
                let tip = match str_arg(args, "tip") {
                    Some(t) if !eq(t, "round") => {
                        Some(crate::brushes::BRUSH_ALPHAS.iter().position(|b| eq(b.name, t)).ok_or(format!("unknown brush tip '{t}'"))?)
                    }
                    _ => None,
                };
                let angle = f32_arg(args, "angle").unwrap_or(0.0).to_radians();
                let follow = args.get("follow").and_then(Value::as_bool).unwrap_or(false);
                let projection = match str_arg(args, "projection") {
                    None => super::BrushProjection::Screen,
                    Some(k) => super::BrushProjection::ALL
                        .into_iter()
                        .find(|p| p.key() == k)
                        .ok_or(format!("projection must be screen, surface or uv, not '{k}'"))?,
                };
                let paint_through = args.get("paint_through").and_then(Value::as_bool).unwrap_or(false);

                // Visibility comes from a depth render with the tool's camera.
                let (cam, [w, h]) = self.tool_camera(args);
                if self.engine.dirty {
                    self.engine.composite();
                }
                let views = self.mcp_views.get_or_insert_with(|| ViewRenderer::new(&self.rs));
                let engines = super::texture_sets::engines(&self.engine, &self.sets, self.mesh.sets.len());
                views.render_3d(&self.rs, [w, h], &engines, &self.env, &self.gpu_mesh, &cam, &self.view, Some(&self.overlays));
                if !self.engine.begin_stroke(i, target, mat.enabled, erase) {
                    return Err("nothing to paint: check the channels (or, erasing, that the layer has paint)".into());
                }
                let first_dir = points.get(1).map_or(0.0, |b| (*b - points[0]).to_angle());
                let mut stamps = vec![[points[0].x, points[0].y, 1.0, angle + if follow { first_dir } else { 0.0 }]];
                let step = (spacing * size).max(1.0);
                for pair in points.windows(2) {
                    let (a, b) = (pair[0], pair[1]);
                    let turn = if follow { (b - a).to_angle() } else { 0.0 };
                    let n = ((b - a).length() / step).ceil().max(1.0) as usize;
                    stamps.extend((1..=n).map(|k| {
                        let p = a.lerp(b, k as f32 / n as f32);
                        [p.x, p.y, 1.0, angle + turn]
                    }));
                }
                match tip {
                    Some(t) => {
                        const SIZE: u32 = 256;
                        let px = crate::brushes::BRUSH_ALPHAS[t].render(SIZE);
                        self.engine.set_brush_alpha(Some((&px, SIZE)));
                    }
                    None => self.engine.set_brush_alpha(None),
                }
                // The app's own brush tip goes back on the next frame.
                self.uploaded_tip = None;
                let su = StampUniform {
                    view_proj: cam.view_proj(w as f32 / h as f32).to_cols_array_2d(),
                    cam_pos: cam.eye().extend(1.0).to_array(),
                    view: [w as f32, h as f32, 0.0, 0.0],
                    view2d: [0.0; 4],
                    brush: [size * 0.5, hardness, flow, symmetry],
                    clip: [crate::viewport::NEAR, crate::viewport::FAR, if tip.is_some() { 1.0 } else { 0.0 }, 0.0],
                    proj: [projection as u32 as f32, if paint_through { 1.0 } else { 0.0 }, 85f32.to_radians().cos(), 0.0],
                };
                let raw: Vec<([f32; 4], f32)> = stamps.iter().map(|&s| (s, 1.0)).collect();
                let stamps = super::resolve_stamps(&self.mesh, &self.bvh, &cam, glam::Vec2::new(w as f32, h as f32), projection, symmetry as u8, size * 0.5, &raw);
                let paint = BrushPaint {
                    mat,
                    mask_value: f32_arg(args, "mask_value").unwrap_or(1.0).clamp(0.0, 1.0),
                    opacity: f32_arg(args, "opacity").unwrap_or(1.0).clamp(0.01, 1.0),
                    erase,
                    grain: [0.0; 4],
                };
                let depth = self.mcp_views.as_ref().expect("rendered above").view3d.depth_view();
                for chunk in stamps.chunks(MAX_STAMPS) {
                    self.engine.stroke_step(chunk, su, depth, &paint, None, false);
                }
                if let Some(entry) = self.engine.end_stroke() {
                    self.undo.push("Paint stroke", crate::undo::Edit::Stroke(entry));
                }
                text(format!("Painted {} stamps on layer {i}.", stamps.len()))
            }
            "poly_fill" => {
                let i = self.layer_index(args)?;
                let target = match str_arg(args, "target").unwrap_or("content") {
                    "content" => StrokeTarget::Content,
                    "mask" => StrokeTarget::Mask,
                    t => return Err(format!("target must be content or mask, not '{t}'")),
                };
                let layer = &self.engine.layers[i];
                if target == StrokeTarget::Content && !layer.is_paint() {
                    return Err("only paint layers take paint; use target 'mask' to fill a mask".into());
                }
                if target == StrokeTarget::Mask && layer.mask.is_none() {
                    return Err("the layer has no mask; add one with set_mask first".into());
                }
                let mode_key = str_arg(args, "mode").unwrap_or("polygon");
                let mode = crate::polyfill::FillMode::ALL
                    .into_iter()
                    .find(|m| m.key() == mode_key)
                    .ok_or(format!("mode must be triangle, polygon, part, object or uv_island, not '{mode_key}'"))?;
                let topo = self.poly_topology();
                let tri = if let Some(uv) = args.get("uv").and_then(Value::as_array) {
                    let v = |k: usize| uv.get(k).and_then(Value::as_f64).unwrap_or(0.0) as f32;
                    topo.pick_uv(glam::Vec2::new(v(0), v(1))).ok_or("no triangle at that UV position")?
                } else {
                    let (x, y) = (f32_arg(args, "x").ok_or("give 'x' and 'y', or 'uv'")?, f32_arg(args, "y").ok_or("missing 'y'")?);
                    let (cam, [w, h]) = self.tool_camera(args);
                    let (o, d) = cam.ray(glam::Vec2::new(x, y), glam::Vec2::new(w as f32, h as f32));
                    self.bvh.raycast(o, d).ok_or("no mesh under that pixel")?.tri
                };
                let mut mat = crate::engine::MatValues { enabled: [false; crate::engine::CHANNELS], ..Default::default() };
                if let Some(c) = args.get("color").and_then(Value::as_array) {
                    for (k, v) in c.iter().take(3).enumerate() {
                        mat.color[k] = v.as_f64().unwrap_or(0.0).clamp(0.0, 1.0) as f32;
                    }
                    mat.enabled[0] = true;
                }
                if let Some(c) = args.get("emission").and_then(Value::as_array) {
                    for (k, v) in c.iter().take(3).enumerate() {
                        mat.emission[k] = v.as_f64().unwrap_or(0.0).clamp(0.0, 1.0) as f32;
                    }
                    mat.enabled[crate::engine::EMISSION] = true;
                }
                if let Some(v) = f32_arg(args, "specular") {
                    mat.specular = v.clamp(0.0, 1.0);
                    mat.enabled[crate::engine::SPECULAR] = true;
                }
                for (k, key) in [(1, "roughness"), (2, "metallic"), (3, "height"), (4, "opacity")] {
                    if let Some(v) = f32_arg(args, key) {
                        match k {
                            1 => mat.rough = v.clamp(0.0, 1.0),
                            2 => mat.metal = v.clamp(0.0, 1.0),
                            4 => mat.opacity = v.clamp(0.0, 1.0),
                            _ => mat.height = v.clamp(-1.0, 1.0),
                        }
                        mat.enabled[k] = true;
                    }
                }
                let erase = args.get("erase").and_then(Value::as_bool).unwrap_or(false);
                if target == StrokeTarget::Content && !erase && !mat.enabled.contains(&true) {
                    return Err("give at least one of color, roughness, metallic, height, opacity, emission, specular".into());
                }
                if !self.engine.begin_stroke(i, target, mat.enabled, erase) {
                    return Err("nothing to fill: check the channels (or, erasing, that the layer has paint)".into());
                }
                let tris = topo.select(mode, tri);
                let geometry = topo.uv_geometry(&tris, 2.0 / self.engine.res as f32);
                let paint = BrushPaint {
                    mat,
                    mask_value: f32_arg(args, "mask_value").unwrap_or(1.0).clamp(0.0, 1.0),
                    opacity: f32_arg(args, "opacity").unwrap_or(1.0).clamp(0.01, 1.0),
                    erase,
                    grain: [0.0; 4],
                };
                self.engine.stroke_polys(&geometry, &paint);
                if let Some(entry) = self.engine.end_stroke() {
                    self.undo.push("Polygon fill", crate::undo::Edit::Stroke(entry));
                }
                text(format!("Filled {} of {} triangles ({}) on layer {i}.", tris.len(), topo.triangle_count(), mode.name()))
            }
            "cage_preview" => {
                let show = args.get("show").and_then(Value::as_bool).unwrap_or(true);
                let s = &mut self.bake_settings;
                if let Some(hp) = str_arg(args, "high_poly") {
                    s.high_poly = (!hp.is_empty()).then(|| hp.into());
                }
                if let Some(c) = str_arg(args, "cage") {
                    s.cage = (!c.is_empty()).then(|| c.into());
                }
                if let Some(v) = f32_arg(args, "front") {
                    s.front = v.clamp(0.001, 0.5);
                }
                if let Some(v) = f32_arg(args, "rear") {
                    s.rear = v.clamp(0.001, 0.5);
                }
                if let Some(m) = args.get("match_by_name").and_then(Value::as_bool) {
                    s.match_by_name = m;
                }
                self.show_bake = show;
                self.cage_preview.show = show;
                if show {
                    // Probe right away so the reply carries the result.
                    let key = crate::cage::ProbeKey::new(&self.mesh, &self.bake_settings);
                    if !self.cage_preview.probe.as_ref().is_some_and(|p| p.key == key) {
                        let probe = crate::cage::CageProbe::build(&self.mesh, key)?;
                        self.cage_preview.probe = Some(std::sync::Arc::new(probe));
                        self.cage_preview.job = None;
                        self.cage_preview.failed = None;
                        self.cage_preview.drawn = None;
                    }
                }
                self.update_cage_preview(ctx);
                match (show, self.cage_preview.stats) {
                    (false, _) => text("Cage preview hidden."),
                    (true, None) => text("Cage shown (low poly only: nothing to project, so no coverage report)."),
                    (true, Some(st)) => json_text(json!({
                        "covered": st.ok, "tight": st.tight, "missed": st.missed, "overlapping": st.overlap,
                        "front": self.bake_settings.front, "rear": self.bake_settings.rear,
                        "cage": self.bake_settings.cage.as_ref().map(|p| p.display().to_string()),
                    })),
                }
            }
            "save_project" => {
                let path = match str_arg(args, "path") {
                    Some(p) => std::path::PathBuf::from(p),
                    None => self.project_path.clone().ok_or("the project has no file yet; give 'path'")?,
                };
                self.save_project(&path)?;
                text(self.status.clone())
            }
            "open_project" => {
                let path = str_arg(args, "path").ok_or("missing 'path'")?;
                self.open_project(std::path::Path::new(path), ctx)?;
                Reply::AfterBake
            }
            "preview_settings" => {
                let v = &mut self.view;
                if let Some(f) = f32_arg(args, "fov") {
                    self.cam.fov_y = f.clamp(5.0, 120.0).to_radians();
                }
                if let Some(r) = f32_arg(args, "env_rotation") {
                    v.env_rotation = r.to_radians();
                }
                if let Some(b) = str_arg(args, "background") {
                    v.background = Background::ALL
                        .into_iter()
                        .find(|x| eq(x.name(), b) || format!("{x:?}").eq_ignore_ascii_case(b))
                        .ok_or(format!("unknown background '{b}' (environment, color, transparent)"))?;
                }
                if let Some(c) = args.get("bg_color").and_then(Value::as_array) {
                    for (k, x) in c.iter().take(3).enumerate() {
                        v.bg_color[k] = x.as_f64().unwrap_or(0.0).max(0.0) as f32;
                    }
                }
                if let Some(t) = str_arg(args, "tonemap") {
                    v.post.tonemap = Tonemap::ALL
                        .into_iter()
                        .find(|x| eq(x.name(), t) || format!("{x:?}").eq_ignore_ascii_case(t))
                        .ok_or(format!("unknown tone mapper '{t}'"))?;
                }
                if let Some(on) = args.get("key_light").and_then(Value::as_bool) {
                    v.key.on = on;
                }
                let p = &mut v.post;
                for (key, field, lo, hi) in [
                    ("exposure", &mut v.exposure, 0.01, 16.0),
                    ("env_intensity", &mut v.env_intensity, 0.0, 16.0),
                    ("key_intensity", &mut v.key.intensity, 0.0, 16.0),
                    ("key_azimuth", &mut v.key.azimuth, -180.0, 180.0),
                    ("key_elevation", &mut v.key.elevation, 5.0, 90.0),
                    ("bg_blur", &mut v.bg_blur, 0.0, 1.0),
                    ("bg_intensity", &mut v.bg_intensity, 0.0, 16.0),
                    ("contrast", &mut p.contrast, 0.0, 3.0),
                    ("saturation", &mut p.saturation, 0.0, 3.0),
                    ("temperature", &mut p.temperature, -1.0, 1.0),
                    ("tint", &mut p.tint, -1.0, 1.0),
                    ("bloom", &mut p.bloom, 0.0, 4.0),
                    ("bloom_threshold", &mut p.bloom_threshold, 0.0, 16.0),
                    ("vignette", &mut p.vignette, 0.0, 1.0),
                    ("chromatic_aberration", &mut p.chromatic, 0.0, 1.0),
                    ("sharpen", &mut p.sharpen, 0.0, 2.0),
                    ("grain", &mut p.grain, 0.0, 1.0),
                ] {
                    if let Some(x) = f32_arg(args, key) {
                        *field = x.clamp(lo, hi);
                    }
                }
                if let Some(h) = str_arg(args, "hdri") {
                    let source = EnvSource::parse(h);
                    if source != self.env.source || self.env_load.is_some() {
                        self.load_env(source, ctx);
                        return Ok(Reply::AfterEnvironment);
                    }
                }
                json_text(self.preview_json())
            }
            "render_view" => {
                let (cam, [w, h]) = self.tool_camera(args);
                let mut settings = ViewSettings { mode: self.view.mode, ..self.view };
                if let Some(m) = str_arg(args, "mode") {
                    settings.mode = view_mode(m)?;
                }
                if args.get("transparent").and_then(Value::as_bool) == Some(true) {
                    settings.background = Background::Transparent;
                }
                if self.engine.dirty {
                    self.engine.composite();
                }
                // A separate renderer, so the app's own viewport keeps its size.
                let views = self.mcp_views.get_or_insert_with(|| ViewRenderer::new(&self.rs));
                let engines = super::texture_sets::engines(&self.engine, &self.sets, self.mesh.sets.len());
                views.render_3d(&self.rs, [w, h], &engines, &self.env, &self.gpu_mesh, &cam, &settings, Some(&self.overlays));
                let img = views.view3d.image();
                let rgba = crate::gpu::read_texture(&self.rs.device, &self.rs.queue, img);
                Reply::Now(png(&rgba, img.width, img.height).map(|p| vec![Out::Png(p)]))
            }
            "select_surface" => {
                let i = self.layer_index(args)?;
                let mode = match str_arg(args, "mode").unwrap_or("replace") {
                    "replace" => select::MaskMode::Replace,
                    "add" => select::MaskMode::Add,
                    "subtract" => select::MaskMode::Subtract,
                    m => return Err(format!("mode must be replace, add or subtract, not '{m}'")),
                };
                let selected = if let Some(parts) = args.get("parts").and_then(Value::as_array) {
                    let parts: Vec<u32> = parts.iter().filter_map(Value::as_u64).map(|p| p as u32).collect();
                    let (tri_part, count) = select::mesh_parts(&self.mesh);
                    if let Some(bad) = parts.iter().find(|&&p| p >= count) {
                        return Err(format!("no part {bad} (the mesh has {count} parts)"));
                    }
                    tri_part.iter().map(|p| parts.contains(p)).collect()
                } else {
                    let seed_point = if let Some(p) = args.get("point").and_then(Value::as_array) {
                        let c: Vec<f32> = p.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect();
                        if c.len() != 3 {
                            return Err("'point' needs 3 numbers".into());
                        }
                        glam::Vec3::new(c[0], c[1], c[2])
                    } else {
                        let x = f32_arg(args, "x").ok_or("give 'parts', a pixel 'x' and 'y', or a 'point'")?;
                        let y = f32_arg(args, "y").ok_or("missing 'y'")?;
                        let (cam, [w, h]) = self.tool_camera(args);
                        let (o, d) = cam.ray(glam::Vec2::new(x, y), glam::Vec2::new(w as f32, h as f32));
                        self.bvh.raycast(o, d).ok_or(format!("pixel ({x}, {y}) does not hit the mesh"))?.pos
                    };
                    let seed = select::nearest_triangle(&self.mesh, seed_point).ok_or("the mesh has no triangles")?;
                    let stop = match str_arg(args, "stop_at").unwrap_or("concave") {
                        "concave" => select::StopAt::Concave,
                        "any" => select::StopAt::Any,
                        s => return Err(format!("stop_at must be 'concave' or 'any', not '{s}'")),
                    };
                    let angle = f32_arg(args, "crease_angle").unwrap_or(30.0).clamp(1.0, 179.0);
                    select::grow(&self.mesh, seed, angle, stop)
                };
                let (mut lo, mut hi, mut count) = (glam::Vec3::MAX, glam::Vec3::MIN, 0);
                for (t, _) in self.mesh.indices.chunks_exact(3).zip(&selected).filter(|(_, s)| **s) {
                    count += 1;
                    for &v in t {
                        let p = glam::Vec3::from_array(self.mesh.vertices[v as usize].pos);
                        (lo, hi) = (lo.min(p), hi.max(p));
                    }
                }
                self.engine.mask_selection(i, &self.mesh, &selected, mode, &mut self.undo);
                self.selected = i;
                json_text(json!({
                    "layer": i,
                    "triangles_selected": count,
                    "triangles_total": selected.len(),
                    "bounds_min": lo.to_array(),
                    "bounds_max": hi.to_array(),
                }))
            }
            "screenshot_app" => Reply::AfterScreenshot,
            "export_textures" => {
                let folder = str_arg(args, "folder").ok_or("missing 'folder'")?;
                let template = match str_arg(args, "preset") {
                    Some(p) => self.export.find_loose(p).ok_or_else(|| {
                        let names: Vec<&str> = self.export.templates.iter().map(|t| t.name.as_str()).collect();
                        format!("unknown export template '{p}'; available: {}", names.join(", "))
                    })?,
                    None => self.export.find(&self.prefs.export_template).unwrap_or(0),
                };
                let mut settings = self.export_settings();
                settings.folder = std::path::PathBuf::from(folder);
                if let Some(p) = str_arg(args, "prefix") {
                    settings.mesh_name = p.to_string();
                }
                if let Some(f) = str_arg(args, "format") {
                    settings.format = crate::export::Format::from_name(f).ok_or(format!("unknown format '{f}'"))?;
                }
                use crate::export::Depth;
                settings.depth = match args.get("bit_depth").and_then(Value::as_u64) {
                    Some(8) => Depth::U8,
                    Some(16) => Depth::U16,
                    Some(32) => Depth::F32,
                    Some(d) => return Err(format!("bit_depth must be 8, 16 or 32, not {d}")),
                    None => settings.depth,
                };
                if !settings.format.depths().contains(&settings.depth) {
                    settings.depth = settings.format.depths()[0];
                }
                if let Some(s) = args.get("size").and_then(Value::as_u64) {
                    settings.size = Some((s as u32).clamp(16, 8192));
                }
                if let Some(p) = str_arg(args, "padding") {
                    settings.padding = crate::export::Padding::from_key(p).ok_or(format!("unknown padding '{p}'"))?;
                }
                let files = self.export_now(template, &settings)?;
                json_text(json!({"files": files.iter().map(|f| f.display().to_string()).collect::<Vec<_>>()}))
            }
            _ => return Err(format!("unknown tool '{tool}'")),
        })
    }

    /// The camera and image size a tool's yaw/pitch/distance/target/width/height
    /// describe, defaulting to the app's camera.
    fn preview_json(&self) -> Value {
        let (v, p) = (&self.view, &self.view.post);
        json!({
            "hdri": self.env.source.name(),
            "available_hdris": self.env_builtins.iter().map(|&i| environment::BUILTIN[i].0).collect::<Vec<_>>(),
            "fov": self.cam.fov_y.to_degrees(),
            "env_rotation": v.env_rotation.to_degrees(),
            "env_intensity": v.env_intensity,
            "key_light": v.key.on,
            "key_intensity": v.key.intensity,
            "key_azimuth": v.key.azimuth,
            "key_elevation": v.key.elevation,
            "background": v.background.name(),
            "bg_blur": v.bg_blur,
            "bg_intensity": v.bg_intensity,
            "bg_color": v.bg_color,
            "exposure": v.exposure,
            "tonemap": p.tonemap.name(),
            "tonemappers": Tonemap::ALL.iter().map(|t| t.name()).collect::<Vec<_>>(),
            "contrast": p.contrast,
            "saturation": p.saturation,
            "temperature": p.temperature,
            "tint": p.tint,
            "bloom": p.bloom,
            "bloom_threshold": p.bloom_threshold,
            "vignette": p.vignette,
            "chromatic_aberration": p.chromatic,
            "sharpen": p.sharpen,
            "grain": p.grain,
        })
    }

    fn tool_camera(&self, args: &Value) -> (crate::viewport::OrbitCam, [u32; 2]) {
        let w = args.get("width").and_then(Value::as_u64).unwrap_or(1024).clamp(16, 4096) as u32;
        let h = args.get("height").and_then(Value::as_u64).unwrap_or(768).clamp(16, 4096) as u32;
        let mut cam = self.cam;
        cam.yaw = f32_arg(args, "yaw").unwrap_or(cam.yaw);
        cam.pitch = f32_arg(args, "pitch").unwrap_or(cam.pitch).clamp(-1.55, 1.55);
        cam.dist = f32_arg(args, "distance").unwrap_or(cam.dist).max(0.05);
        if let Some(t) = args.get("target").and_then(Value::as_array) {
            for (k, v) in t.iter().take(3).enumerate() {
                cam.target[k] = v.as_f64().unwrap_or(0.0) as f32;
            }
        }
        (cam, [w, h])
    }

    /// Applies `set_mask`'s optional generator parameters.
    fn tune_generator(&self, g: &mut Generator, args: &Value) -> Result<(), String> {
        for (key, field) in [
            ("level", &mut g.level),
            ("contrast", &mut g.contrast),
            ("scale", &mut g.scale),
            ("noise", &mut g.noise),
            ("seed", &mut g.seed),
            ("edge_width", &mut g.edge_width),
            ("id_tolerance", &mut g.id_tolerance),
        ] {
            if let Some(v) = f32_arg(args, key) {
                *field = v;
            }
        }
        if let Some(v) = args.get("invert").and_then(Value::as_bool) {
            g.invert = v;
        }
        if let Some(c) = args.get("id_color").and_then(Value::as_array) {
            for (k, v) in c.iter().take(3).enumerate() {
                g.id_color[k] = v.as_f64().unwrap_or(0.0) as f32;
            }
        }
        if let Some(id) = args.get("id").and_then(Value::as_u64) {
            let regions = self.engine.bake_report.as_ref().map(|r| r.id_regions.as_slice()).unwrap_or_default();
            let r = regions.get(id as usize).ok_or(format!("no ID region {id} ({} regions)", regions.len()))?;
            g.id_color = r.color;
        }
        if let Some(w) = args.get("weights").and_then(Value::as_object) {
            for (name, v) in w {
                let k = GEN_INPUTS.iter().position(|(n, _)| eq(n, name)).ok_or(format!("unknown generator input '{name}'"))?;
                g.weights[k] = v.as_f64().unwrap_or(0.0) as f32;
            }
        }
        Ok(())
    }

    fn state_json(&self) -> Value {
        let layers: Vec<Value> = self
            .engine
            .layers
            .iter()
            .enumerate()
            .map(|(i, l)| {
                let mask = match &l.mask {
                    None => json!(null),
                    Some(m) => json!({
                        "generator": m.generator.map(|g| json!({
                            "kind": g.kind.name(),
                            "level": g.level,
                            "contrast": g.contrast,
                            "invert": g.invert,
                            "id_color": (g.kind == GenKind::IdMask).then_some(g.id_color),
                        })),
                        "painted": m.paint.is_some(),
                    }),
                };
                let content = match &l.content {
                    Content::Paint(ch) => {
                        let names = ["color", "roughness", "metallic", "height"];
                        let painted: Vec<&str> = names.iter().zip(ch).filter(|(_, t)| t.is_some()).map(|(n, _)| *n).collect();
                        json!({"kind": "paint", "channels": painted})
                    }
                    Content::Fill(f) => json!({
                        "kind": "fill",
                        "color": f.mat.color,
                        "roughness": f.mat.rough,
                        "metallic": f.mat.metal,
                    }),
                    Content::Folder(_) => json!({"kind": "folder", "layers_inside": self.engine.block(i).len() - 1}),
                };
                json!({
                    "index": i,
                    "name": l.name,
                    "visible": l.visible,
                    "opacity": l.opacity,
                    "blend": l.blend.name(),
                    "content": content,
                    "folder": l.parent.and_then(|p| self.engine.index_of(p)),
                    "mask": mask,
                    "selected": i == self.selected,
                })
            })
            .collect();
        json!({
            "mesh": {"name": self.mesh.name, "triangles": self.mesh.triangle_count()},
            "resolution": self.engine.res,
            "layers_bottom_to_top": layers,
            "bake": self.bake_summary(),
            "baking": self.bake_job.is_some(),
            "baked_maps": MAPS.iter().map(|m| m.name).collect::<Vec<_>>(),
            "view": {
                "mode": self.view.mode.name(),
                "yaw": self.cam.yaw,
                "pitch": self.cam.pitch,
                "distance": self.cam.dist,
                "ray_trace": self.rt.enabled,
            },
            "undo": self.undo.undo.iter().rev().take(10).map(|e| e.label.as_str()).collect::<Vec<_>>(),
            "redo": self.undo.redo.iter().rev().take(10).map(|e| e.label.as_str()).collect::<Vec<_>>(),
        })
    }
}
