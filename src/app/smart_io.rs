//! User smart materials: a folder of the layer stack (or one layer) saved as
//! a `.mpsmart` file and dropped onto any mesh later.
//!
//! The file is a plain zip, readable without the app: `smart.json` (the
//! layers, their fills and masks, and the material graphs they use, in the
//! same JSON as project files), the fill images under `images/`, and a
//! `thumb.png`. Hand-painted strokes are tied to one mesh's UVs, so paint
//! layers and painted masks are left out; procedural masks, mask layers,
//! filters and graphs carry over.

use super::project_io::{data_dir, fill_from, fill_json, generator_from, generator_json, mask_filters_from, mask_filters_json, mask_layers_from, mask_layers_json};
use super::{PainterApp, theme};
use crate::engine::{Blend, Content, Folder, Layer};
use crate::undo::StrokeTarget;
use eframe::egui;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const FORMAT: &str = "material-painter-smart-material";
const VERSION: u64 = 1;
pub(super) const EXTENSION: &str = "mpsmart";

/// The folder user smart materials live in.
pub(super) fn smart_dir() -> Option<PathBuf> {
    data_dir().map(|d| d.join("smart"))
}

/// A user smart material on the shelf.
pub(super) struct UserSmart {
    pub path: PathBuf,
    pub name: String,
    pub thumb: Option<egui::TextureHandle>,
}

/// Every saved smart material, by name, with its thumbnail.
pub(super) fn list(ctx: &egui::Context) -> Vec<UserSmart> {
    let Some(dir) = smart_dir() else { return Vec::new() };
    let Ok(read) = std::fs::read_dir(&dir) else { return Vec::new() };
    let mut out: Vec<UserSmart> = read
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == EXTENSION))
        .map(|path| {
            let name = path.file_stem().map_or(String::new(), |s| s.to_string_lossy().into_owned());
            let thumb = read_entry(&path, "thumb.png").ok().and_then(|png| {
                let img = image::load_from_memory(&png).ok()?.into_rgba8();
                let size = [img.width() as usize, img.height() as usize];
                Some(ctx.load_texture(format!("smart {}", path.display()), egui::ColorImage::from_rgba_unmultiplied(size, &img), Default::default()))
            });
            UserSmart { path, name, thumb }
        })
        .collect();
    out.sort_by_key(|a| a.name.to_lowercase());
    out
}

fn read_entry(path: &Path, name: &str) -> Result<Vec<u8>, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut zip = zip::ZipArchive::new(file).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut entry = zip.by_name(name).map_err(|e| format!("{}: {name}: {e}", path.display()))?;
    let mut out = Vec::new();
    entry.read_to_end(&mut out).map_err(|e| e.to_string())?;
    Ok(out)
}

/// A file name that is free in `dir`: `name.ext`, else `name 2.ext`, ...
fn free_path(dir: &Path, name: &str, ext: &str) -> PathBuf {
    let clean: String = name.chars().map(|c| if c.is_alphanumeric() || " -_".contains(c) { c } else { '_' }).collect();
    let clean = clean.trim();
    let clean = if clean.is_empty() { "Smart material" } else { clean };
    (1..)
        .map(|n| if n == 1 { dir.join(format!("{clean}.{ext}")) } else { dir.join(format!("{clean} {n}.{ext}")) })
        .find(|p| !p.exists())
        .expect("some name is free")
}

impl PainterApp {
    /// Saves the selected layer (a folder with everything in it) as a smart
    /// material on the shelf.
    pub(super) fn save_smart_material(&mut self, ctx: &egui::Context) {
        let idx = self.selected;
        let Some(top) = self.engine.layers.get(idx) else { return };
        let name = top.name.clone();
        match self.write_smart(idx, &name) {
            Ok((path, skipped)) => {
                self.reload_user_smarts(ctx);
                self.shelf.cat = super::ShelfCat::SmartMaterials;
                let note = match skipped {
                    0 => String::new(),
                    n => format!(" ({n} hand-painted part{} left out: painting belongs to this mesh's UVs.)", if n == 1 { "" } else { "s" }),
                };
                self.status = format!(
                    "Saved '{}' as a smart material: it's first in the shelf's Smart materials.{note}",
                    path.file_stem().map_or(name, |s| s.to_string_lossy().into_owned())
                );
            }
            Err(e) => self.status = format!("Couldn't save the smart material: {e}"),
        }
    }

    /// Writes the block of layer `idx` as a `.mpsmart`; returns the file and
    /// how many painted parts were left out.
    fn write_smart(&mut self, idx: usize, name: &str) -> Result<(PathBuf, usize), String> {
        let block = self.engine.block(idx);
        let mut skipped = 0;
        let mut layers = Vec::new();
        let mut images: Vec<(String, Arc<Vec<u8>>)> = Vec::new();
        let mut graph_ids: Vec<u64> = Vec::new();
        for i in block {
            let l = &self.engine.layers[i];
            let content = match &l.content {
                Content::Paint(_) => {
                    skipped += 1;
                    continue;
                }
                Content::Fill(p) => {
                    graph_ids.extend(p.graph);
                    json!({"kind": "fill", "fill": fill_json(p)})
                }
                Content::Folder(_) => json!({"kind": "folder"}),
            };
            // A painted mask without anything procedural is dropped (the
            // layer then shows everywhere); a painted base under procedural
            // parts becomes white.
            let mask = l.mask.as_ref().and_then(|m| {
                let procedural = m.generator.is_some() || m.graph.is_some() || !m.layers.is_empty();
                if m.paint.is_some() {
                    skipped += 1;
                }
                procedural.then(|| {
                    graph_ids.extend(m.graph);
                    json!({
                        "generator": m.generator.as_ref().map(generator_json),
                        "graph": m.graph,
                        "filters": mask_filters_json(&m.filters),
                        "layers": mask_layers_json(&m.layers),
                    })
                })
            });
            // Graph-driven images are rendered again from the graph.
            let from_graph = matches!(&l.content, Content::Fill(p) if p.graph.is_some());
            let image_refs: Vec<Value> = l
                .images
                .iter()
                .enumerate()
                .map(|(ch, im)| match im.as_ref().filter(|_| !from_graph) {
                    Some(im) => {
                        let ext = Path::new(&im.name).extension().and_then(|e| e.to_str()).unwrap_or("png").to_lowercase();
                        let file = format!("images/{}_{ch}.{ext}", layers.len());
                        images.push((file.clone(), im.encoded.clone()));
                        json!({"name": im.name, "file": file})
                    }
                    None => Value::Null,
                })
                .collect();
            layers.push(json!({
                "id": l.id,
                "name": l.name,
                "visible": l.visible,
                "opacity": l.opacity,
                "blend": l.blend.name(),
                "parent": l.parent,
                "content": content,
                "mask": mask,
                "images": image_refs,
            }));
        }
        if layers.is_empty() {
            return Err("there's nothing procedural to save (only painted layers)".into());
        }
        graph_ids.sort_unstable();
        graph_ids.dedup();
        let manifest = json!({
            "format": FORMAT,
            "version": VERSION,
            "name": name,
            "app_version": env!("CARGO_PKG_VERSION"),
            "layers_bottom_to_top": layers,
            "graphs": self.graphs.to_json(&graph_ids),
        });

        let dir = smart_dir().ok_or("no settings folder")?;
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let path = free_path(&dir, name, EXTENSION);
        let thumb = self.smart_thumbnail();
        let tmp = path.with_extension("part");
        {
            let mut zip = zip::ZipWriter::new(std::fs::File::create(&tmp).map_err(|e| e.to_string())?);
            let deflate = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
            // Images are already compressed.
            let stored = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
            zip.start_file("smart.json", deflate).map_err(|e| e.to_string())?;
            zip.write_all(serde_json::to_string_pretty(&manifest).expect("json").as_bytes()).map_err(|e| e.to_string())?;
            for (file, data) in &images {
                zip.start_file(file.as_str(), stored).map_err(|e| e.to_string())?;
                zip.write_all(data).map_err(|e| e.to_string())?;
            }
            if let Some(png) = thumb {
                zip.start_file("thumb.png", stored).map_err(|e| e.to_string())?;
                zip.write_all(&png).map_err(|e| e.to_string())?;
            }
            zip.finish().map_err(|e| e.to_string())?;
        }
        std::fs::rename(&tmp, &path).map_err(|e| e.to_string())?;
        Ok((path, skipped))
    }

    /// The 3D view as it looks now (without the brush cursor), square, as PNG.
    fn smart_thumbnail(&mut self) -> Option<Vec<u8>> {
        const SIZE: u32 = 256;
        if self.engine.dirty {
            self.engine.composite();
        }
        let views = self.mcp_views.get_or_insert_with(|| crate::viewport::ViewRenderer::new(&self.rs));
        let engines = super::texture_sets::engines(&self.engine, &self.sets, self.mesh.sets.len());
        views.render_3d(&self.rs, [SIZE, SIZE], &engines, &self.env, &self.gpu_mesh, &self.cam, &self.view, None);
        let img = views.view3d.image();
        let rgba = crate::gpu::read_texture(&self.rs.device, &self.rs.queue, img);
        let buf = image::RgbaImage::from_raw(img.width, img.height, rgba)?;
        let mut png = Vec::new();
        image::DynamicImage::ImageRgba8(buf).write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).ok()?;
        Some(png)
    }

    pub(super) fn reload_user_smarts(&mut self, ctx: &egui::Context) {
        self.user_smarts = list(ctx);
    }

    /// Adds a saved smart material at `at` inside `parent`, as one undo step.
    pub(super) fn apply_smart_material(&mut self, path: &Path, at: usize, parent: Option<u64>) {
        match self.apply_smart(path, at, parent) {
            Ok(name) => self.status = format!("Added smart material '{name}'."),
            Err(e) => self.status = format!("Couldn't add the smart material: {e}"),
        }
    }

    fn apply_smart(&mut self, path: &Path, at: usize, parent: Option<u64>) -> Result<String, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut zip = zip::ZipArchive::new(file).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut read = |name: &str| -> Result<Vec<u8>, String> {
            let mut entry = zip.by_name(name).map_err(|e| format!("{name}: {e}"))?;
            let mut out = Vec::new();
            entry.read_to_end(&mut out).map_err(|e| e.to_string())?;
            Ok(out)
        };
        let manifest: Value = serde_json::from_slice(&read("smart.json")?).map_err(|e| format!("smart.json: {e}"))?;
        if manifest["format"].as_str() != Some(FORMAT) {
            return Err("not a Material Painter smart material".into());
        }
        if manifest["version"].as_u64().unwrap_or(0) > VERSION {
            return Err("made by a newer Material Painter; update to use it".into());
        }
        let name = manifest["name"].as_str().unwrap_or("Smart material").to_string();

        // Graphs first: they get new ids in this project.
        let mut graph_map: HashMap<u64, u64> = HashMap::new();
        for gv in manifest["graphs"].as_array().into_iter().flatten() {
            let Some(old) = gv["id"].as_u64() else { continue };
            let g = crate::graph::Graph::from_json(&gv["graph"], &self.graphs.lib)?;
            graph_map.insert(old, self.graphs.insert(&self.rs, g, None));
        }

        struct MaskPlan {
            generator: Option<crate::engine::Generator>,
            graph: Option<u64>,
            filters: Vec<crate::engine::MaskFilter>,
            layers: Vec<crate::engine::MaskLayer>,
        }
        let mut layers = Vec::new();
        let mut masks: Vec<Option<MaskPlan>> = Vec::new();
        for lv in manifest["layers_bottom_to_top"].as_array().into_iter().flatten() {
            let c = &lv["content"];
            let content = match c["kind"].as_str() {
                Some("fill") => {
                    let mut f = fill_from(&c["fill"]);
                    f.graph = f.graph.and_then(|g| graph_map.get(&g).copied());
                    Content::Fill(f)
                }
                Some("folder") => Content::Folder(Folder { open: true, graph: None }),
                _ => continue,
            };
            let mut images: [Option<crate::engine::FillImage>; crate::engine::CHANNELS] = Default::default();
            for (ch, iv) in lv["images"].as_array().into_iter().flatten().enumerate().take(crate::engine::CHANNELS) {
                let Some(file) = iv["file"].as_str() else { continue };
                let data = Arc::new(read(file)?);
                images[ch] = Some(self.engine.decode_channel_image(ch, iv["name"].as_str().unwrap_or("image").to_string(), data)?);
            }
            let mv = &lv["mask"];
            masks.push((!mv.is_null()).then(|| MaskPlan {
                generator: mv.get("generator").filter(|g| !g.is_null()).and_then(|g| generator_from(g).ok()),
                graph: mv["graph"].as_u64().and_then(|g| graph_map.get(&g).copied()),
                filters: mask_filters_from(&mv["filters"]),
                layers: mask_layers_from(&mv["layers"]),
            }));
            let blend = lv["blend"].as_str().unwrap_or("Normal");
            layers.push(Layer {
                id: lv["id"].as_u64().ok_or("a layer has no id")?,
                name: lv["name"].as_str().unwrap_or("Layer").to_string(),
                visible: lv["visible"].as_bool().unwrap_or(true),
                opacity: lv["opacity"].as_f64().unwrap_or(1.0) as f32,
                blend: Blend::ALL.into_iter().find(|b| b.name() == blend).unwrap_or(Blend::Normal),
                content,
                mask: None,
                parent: lv["parent"].as_u64(),
                images,
                thumb: None,
            });
        }
        if layers.is_empty() {
            return Err("it has no layers".into());
        }
        // A parent outside the saved block (the folder it was saved from was
        // inside another) means the top: it goes into `parent`.
        let saved: HashSet<u64> = layers.iter().map(|l| l.id).collect();
        for l in &mut layers {
            l.parent = l.parent.filter(|p| saved.contains(p));
        }

        self.undo.begin_group(format!("Add {name}"));
        let ids = self.engine.insert_layers(at, parent, layers, format!("Add {name}"), &mut self.undo);
        for (id, plan) in ids.iter().zip(masks) {
            let Some(plan) = plan else { continue };
            let Some(i) = self.engine.layers.iter().position(|l| l.id == *id) else { continue };
            self.engine.add_mask(i, true, plan.generator, &mut self.undo);
            if let Some(m) = self.engine.layers[i].mask.as_mut() {
                m.graph = plan.graph;
                m.filters = plan.filters;
                m.layers = plan.layers;
            }
        }
        self.engine.commit_props(&mut self.undo);
        self.undo.end_group();
        self.engine.mark_dirty();
        self.repush_graphs();
        if let Some(top) = ids.last().and_then(|id| self.engine.layers.iter().position(|l| l.id == *id)) {
            self.selected = top;
        }
        self.target = StrokeTarget::Content;
        Ok(name)
    }

    /// The shelf's context menu on a user smart material.
    pub(super) fn user_smart_menu(&mut self, ui: &mut egui::Ui, i: usize, share: &mut Option<PathBuf>) {
        let Some(s) = self.user_smarts.get(i) else { return };
        let path = s.path.clone();
        if ui.button(theme::with_icon(crate::icons::SHARE_NETWORK, "Share to community…")).clicked() {
            *share = Some(path.clone());
            ui.close();
        }
        if ui.button("Show file").clicked() {
            super::export_ui::open_in_file_browser(path.parent().unwrap_or(&path));
            ui.close();
        }
        if ui.button("Delete").clicked() {
            let _ = std::fs::remove_file(&path);
            self.reload_user_smarts(ui.ctx());
            ui.close();
        }
    }

    /// The custom nodes a smart material's graphs use (for sharing).
    pub(super) fn smart_user_nodes(&self, path: &Path) -> Vec<PathBuf> {
        let Ok(raw) = read_entry(path, "smart.json") else { return Vec::new() };
        let Ok(manifest) = serde_json::from_slice::<Value>(&raw) else { return Vec::new() };
        let mut out: Vec<PathBuf> = manifest["graphs"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|gv| crate::graph::Graph::from_json(&gv["graph"], &self.graphs.lib).ok())
            .flat_map(|g| g.nodes.into_iter().map(|n| n.def))
            .filter(|d| d.starts_with("user/"))
            .filter_map(|d| self.graphs.lib.user_path(&d))
            .filter(|p| p.is_file())
            .collect();
        out.sort();
        out.dedup();
        out
    }

    /// The thumbnail inside a smart material file, as PNG.
    pub(super) fn smart_thumb_png(path: &Path) -> Option<Vec<u8>> {
        read_entry(path, "thumb.png").ok()
    }
}
