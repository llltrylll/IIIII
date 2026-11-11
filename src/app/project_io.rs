//! Project files (.mpp): a JSON manifest plus LZ4-compressed binary blobs
//! (the mesh, painted channels, masks and bake fixes). Baked mesh maps aren't
//! stored, which keeps files small; opening a project bakes them with the
//! saved settings, or loads them from the bake cache (`bake_cache`).
//!
//! Layout: magic `MPPROJ\0` + format version byte, u64 manifest length,
//! manifest (UTF-8 JSON), u64 blob count, then per blob a u64 length and the
//! compressed bytes. Saving writes a temporary file and renames it over the
//! target, so a failed save never damages an existing project.

use super::PainterApp;
use crate::bake::{IdSource, Quality};
use crate::engine::{Blend, Content, Engine, FillParams, Folder, GenKind, Generator, Layer, Levels, Mask, MatValues, Projection};
use crate::environment::EnvSource;
use crate::gpu::Tex;
use crate::mesh::{Mesh, Vertex};
use crate::post::Tonemap;
use crate::viewport::{Background, OrbitCam};
use eframe::{egui, wgpu};
use serde_json::{Value, json};
use std::io::Write;
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"MPPROJ\x00\x01";

/// Per-user app data: recent projects and the autosave.
pub(super) fn data_dir() -> Option<PathBuf> {
    crate::crash::data_dir()
}

/// This instance's recovery copy of unsaved work, written every few edits.
/// Each running instance has its own, guarded by a lock file it holds open
/// for its lifetime; an autosave whose lock is free belongs to a session that
/// ended without closing cleanly.
pub(super) fn autosave_path() -> Option<PathBuf> {
    static LOCK: std::sync::OnceLock<Option<std::fs::File>> = std::sync::OnceLock::new();
    let dir = data_dir()?;
    let pid = std::process::id();
    LOCK.get_or_init(|| {
        std::fs::create_dir_all(&dir).ok()?;
        let file = std::fs::File::create(dir.join(format!("autosave-{pid}.lock"))).ok()?;
        file.try_lock().ok()?;
        Some(file)
    })
    .as_ref()?;
    Some(dir.join(format!("autosave-{pid}.mpp")))
}

pub(super) fn is_autosave(path: &Path) -> bool {
    data_dir().is_some_and(|d| path.parent() == Some(d.as_path()))
        && path.file_name().is_some_and(|n| n.to_string_lossy().starts_with("autosave-"))
}

/// The newest autosave left by a session that didn't close cleanly.
pub(super) fn pending_recovery() -> Option<PathBuf> {
    let own = autosave_path();
    let mut found: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(data_dir()?)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| is_autosave(p) && p.extension().is_some_and(|x| x == EXTENSION) && Some(p) != own.as_ref())
        .filter(|p| {
            // A lock we can take means its owner is gone.
            let lock = p.with_extension("lock");
            std::fs::File::open(&lock).map_or(true, |f| f.try_lock().is_ok())
        })
        .filter_map(|p| Some((std::fs::metadata(&p).ok()?.modified().ok()?, p)))
        .collect();
    found.sort();
    found.pop().map(|(_, p)| p)
}

/// Deletes an autosave and its lock file.
pub(super) fn remove_recovery(path: &Path) {
    let _ = std::fs::remove_file(path);
    let _ = std::fs::remove_file(path.with_extension("lock"));
}

/// Deletes this instance's autosave (the work was saved or discarded).
pub(super) fn remove_autosave() {
    if let Some(p) = autosave_path() {
        let _ = std::fs::remove_file(p);
    }
}

pub(super) fn load_recent() -> Vec<PathBuf> {
    let Some(text) = data_dir().and_then(|d| std::fs::read_to_string(d.join("recent.json")).ok()) else { return Vec::new() };
    let list: Vec<String> = serde_json::from_str(&text).unwrap_or_default();
    list.into_iter().map(PathBuf::from).filter(|p| p.is_file()).collect()
}
pub const EXTENSION: &str = "mpp";

/// Texture formats a project can hold, by name.
const FORMATS: [wgpu::TextureFormat; 6] = [
    wgpu::TextureFormat::Rgba8Unorm,
    wgpu::TextureFormat::Rg8Unorm,
    wgpu::TextureFormat::Rg16Float,
    wgpu::TextureFormat::R8Unorm,
    wgpu::TextureFormat::R16Float,
    wgpu::TextureFormat::Rgba16Float,
];

struct Writer {
    blobs: Vec<Vec<u8>>,
}

impl Writer {
    fn blob(&mut self, raw: &[u8]) -> Value {
        self.blobs.push(lz4_flex::compress_prepend_size(raw));
        json!(self.blobs.len() - 1)
    }
}

struct Reader<'a> {
    // Borrow the compressed data from the project file instead of copying
    // every blob while the entire container is already resident in memory.
    blobs: Vec<&'a [u8]>,
}

impl Reader<'_> {
    fn blob(&self, v: &Value) -> Result<Vec<u8>, String> {
        const MAX_UNCOMPRESSED_BLOB: usize = 1024 * 1024 * 1024;
        let i = v
            .as_u64()
            .and_then(|index| usize::try_from(index).ok())
            .ok_or("missing or invalid blob reference")?;
        let packed = self.blobs.get(i).ok_or(format!("blob {i} is missing"))?;
        let prefix: [u8; 4] = packed
            .get(..4)
            .ok_or_else(|| format!("blob {i} is truncated"))?
            .try_into()
            .map_err(|_| format!("blob {i} has an invalid size prefix"))?;
        let expected = u32::from_le_bytes(prefix) as usize;
        if expected > MAX_UNCOMPRESSED_BLOB {
            return Err(format!("blob {i} declares an unreasonable uncompressed size"));
        }
        lz4_flex::decompress_size_prepended(packed).map_err(|e| format!("blob {i} is damaged: {e}"))
    }
}

fn f(v: &Value, key: &str, default: f32) -> f32 {
    v.get(key).and_then(Value::as_f64).map_or(default, |x| x as f32)
}

fn arr<const N: usize>(v: &Value, key: &str, default: [f32; N]) -> [f32; N] {
    match v.get(key).and_then(Value::as_array) {
        Some(a) if a.len() == N => std::array::from_fn(|i| a[i].as_f64().unwrap_or(default[i] as f64) as f32),
        _ => default,
    }
}

fn b(v: &Value, key: &str, default: bool) -> bool {
    v.get(key).and_then(Value::as_bool).unwrap_or(default)
}

fn s<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(Value::as_str)
}

fn custom_map_json(c: &crate::bake::CustomMap) -> Value {
    json!({
        "id": c.id,
        "name": c.name,
        "kind": format!("{:?}", c.kind),
        "azimuth": c.azimuth, "elevation": c.elevation, "spread": c.spread,
        "distance": c.distance, "strength": c.strength, "wander": c.wander,
        "point": c.point,
    })
}

/// A custom bake map; `None` for an empty slot (older projects had four
/// fixed slots) or an unknown kind. Maps saved without an id (those slots)
/// get `fallback_id`, the slot's number, which older masks refer to.
fn custom_map_from(v: &Value, fallback_id: u32) -> Option<crate::bake::CustomMap> {
    let name = s(v, "kind")?;
    let kind = crate::bake::CustomKind::ALL.into_iter().find(|k| format!("{k:?}") == name)?;
    let d = kind.default_map();
    let mut point = d.point;
    for (dst, x) in point.iter_mut().zip(v["point"].as_array().into_iter().flatten()) {
        *dst = x.as_f64().unwrap_or(0.0) as f32;
    }
    Some(crate::bake::CustomMap {
        id: v["id"].as_u64().map_or(fallback_id, |id| id as u32),
        name: s(v, "name").unwrap_or(kind.name()).to_string(),
        kind,
        azimuth: f(v, "azimuth", d.azimuth),
        elevation: f(v, "elevation", d.elevation),
        spread: f(v, "spread", d.spread),
        distance: f(v, "distance", d.distance),
        strength: f(v, "strength", d.strength),
        wander: f(v, "wander", d.wander),
        point,
    })
}

pub(super) fn fill_json(p: &FillParams) -> Value {
    json!({
        "enabled": p.mat.enabled,
        "color": p.mat.color,
        "roughness": p.mat.rough,
        "metallic": p.mat.metal,
        "height": p.mat.height,
        "opacity": p.mat.opacity,
        "emission": p.mat.emission,
        "normal_strength": p.mat.normal,
        "specular": p.mat.specular,
        "color_variation": p.color_var,
        "variation_scale": p.var_scale,
        "roughness_variation": p.rough_var,
        "height_variation": p.height_var,
        "seed": p.seed,
        "part_variation": p.part_var,
        "projection": format!("{:?}", p.projection),
        "tiling": p.tiling,
        "rotation": p.rotation,
        "offset": p.offset,
        "triplanar_sharpness": p.tri_sharpness,
        "graph": p.graph,
    })
}

pub(super) fn fill_from(v: &Value) -> FillParams {
    let d = FillParams::default();
    // Four channels before opacity existed, five before emission, six before
    // normal and specular (the missing ones stay off).
    let enabled = match v.get("enabled").and_then(Value::as_array) {
        Some(a) if a.len() >= 4 => std::array::from_fn(|i| a.get(i).is_some_and(|x| x.as_bool().unwrap_or(true))),
        _ => d.mat.enabled,
    };
    FillParams {
        mat: MatValues {
            enabled,
            color: arr(v, "color", d.mat.color),
            rough: f(v, "roughness", d.mat.rough),
            metal: f(v, "metallic", d.mat.metal),
            height: f(v, "height", d.mat.height),
            opacity: f(v, "opacity", d.mat.opacity),
            emission: arr(v, "emission", d.mat.emission),
            normal: f(v, "normal_strength", d.mat.normal),
            specular: f(v, "specular", d.mat.specular),
        },
        color_var: f(v, "color_variation", d.color_var),
        var_scale: f(v, "variation_scale", d.var_scale),
        rough_var: f(v, "roughness_variation", d.rough_var),
        height_var: f(v, "height_variation", d.height_var),
        seed: f(v, "seed", d.seed),
        part_var: f(v, "part_variation", d.part_var),
        projection: s(v, "projection").and_then(Projection::from_key).unwrap_or(Projection::Uv),
        tiling: f(v, "tiling", d.tiling),
        rotation: f(v, "rotation", d.rotation),
        offset: arr(v, "offset", d.offset),
        tri_sharpness: f(v, "triplanar_sharpness", d.tri_sharpness),
        graph: v["graph"].as_u64(),
    }
}

/// Per-corner vertex colors from their saved bytes (nine a triangle); `None`
/// when they don't cover exactly `tris` triangles.
fn corner_colors_from_bytes(bytes: &[u8], tris: usize) -> Option<Vec<[[u8; 3]; 3]>> {
    let colors: Vec<[[u8; 3]; 3]> = bytes
        .chunks_exact(9)
        .map(|c| [[c[0], c[1], c[2]], [c[3], c[4], c[5]], [c[6], c[7], c[8]]])
        .collect();
    (colors.len() == tris && bytes.len() == tris * 9).then_some(colors)
}

pub(super) fn generator_json(g: &Generator) -> Value {
    json!({
        "kind": g.kind.name(),
        "weights": g.weights,
        "level": g.level,
        "contrast": g.contrast,
        "scale": g.scale,
        "noise": g.noise,
        "invert": g.invert,
        "seed": g.seed,
        "edge_width": g.edge_width,
        "id_color": g.id_color,
        "id_tolerance": g.id_tolerance,
        "id_colors": g.id_colors(),
        "custom": g.custom.iter().filter(|c| c.id != 0).map(|c| json!({"id": c.id, "weight": c.weight})).collect::<Vec<_>>(),
    })
}

/// A mask's filters, as saved.
pub(super) fn mask_filters_json(filters: &[crate::engine::MaskFilter]) -> Value {
    filters.iter().map(|f| json!({"kind": f.kind.key(), "enabled": f.enabled, "a": f.a, "b": f.b, "c": f.c})).collect()
}

pub(super) fn mask_filters_from(v: &Value) -> Vec<crate::engine::MaskFilter> {
    v.as_array()
        .into_iter()
        .flatten()
        .filter_map(|fv| {
            let kind = crate::engine::FilterKind::ALL.into_iter().find(|k| Some(k.key()) == fv["kind"].as_str())?;
            let mut f = crate::engine::MaskFilter::new(kind);
            f.enabled = fv["enabled"].as_bool().unwrap_or(true);
            f.a = fv["a"].as_f64().map_or(f.a, |v| v as f32);
            f.b = fv["b"].as_f64().map_or(f.b, |v| v as f32);
            f.c = fv["c"].as_f64().map_or(f.c, |v| v as f32);
            Some(f)
        })
        .collect()
}

/// A mask's mask layers, as saved.
pub(super) fn mask_layers_json(layers: &[crate::engine::MaskLayer]) -> Value {
    layers
        .iter()
        .map(|ml| {
            let source = match &ml.source {
                crate::engine::MaskSource::Generator(g) => json!({"generator": generator_json(g)}),
                crate::engine::MaskSource::Fill(v) => json!({"fill": v}),
            };
            json!({"source": source, "blend": ml.blend.key(), "opacity": ml.opacity, "enabled": ml.enabled})
        })
        .collect()
}

pub(super) fn mask_layers_from(v: &Value) -> Vec<crate::engine::MaskLayer> {
    v.as_array()
        .into_iter()
        .flatten()
        .filter_map(|lv| {
            let src = &lv["source"];
            let source = match (src.get("generator"), src["fill"].as_f64()) {
                (Some(g), _) => crate::engine::MaskSource::Generator(generator_from(g).ok()?),
                (None, Some(v)) => crate::engine::MaskSource::Fill(v as f32),
                _ => return None,
            };
            Some(crate::engine::MaskLayer {
                source,
                blend: crate::engine::MaskBlend::ALL.into_iter().find(|b| Some(b.key()) == lv["blend"].as_str()).unwrap_or(crate::engine::MaskBlend::Multiply),
                opacity: lv["opacity"].as_f64().unwrap_or(1.0) as f32,
                enabled: lv["enabled"].as_bool().unwrap_or(true),
            })
        })
        .collect()
}

/// Built-in smart mask inputs when projects stored custom slots after them.
const LEGACY_INPUTS: usize = 19;

pub(super) fn generator_from(v: &Value) -> Result<Generator, String> {
    let name = s(v, "kind").ok_or("generator without a kind")?;
    let kind = GenKind::ALL.into_iter().find(|k| k.name() == name).ok_or(format!("unknown generator '{name}'"))?;
    let d = kind.default_generator();
    let mut weights = [0.0; crate::engine::GEN_INPUT_COUNT];
    let mut custom = [crate::engine::CustomInput::default(); crate::engine::MAX_CUSTOM_INPUTS];
    if let Some(a) = v.get("weights").and_then(Value::as_array) {
        // Projects from the four fixed custom slots (no "custom" list) kept
        // their weights after the 19 built-in inputs of the time: slot k is
        // the map with id k + 1.
        let legacy = v.get("custom").is_none() && a.len() > LEGACY_INPUTS;
        let builtin = if legacy { LEGACY_INPUTS } else { crate::engine::GEN_INPUT_COUNT };
        // Older projects have fewer inputs; new ones default to 0.
        for (w, x) in weights.iter_mut().zip(a.iter().take(builtin)) {
            *w = x.as_f64().unwrap_or(0.0) as f32;
        }
        let old = a.iter().skip(builtin).filter(|_| legacy).map(|x| x.as_f64().unwrap_or(0.0) as f32);
        let used = old.enumerate().filter(|(_, w)| *w != 0.0).take(crate::engine::MAX_CUSTOM_INPUTS);
        for (slot, (k, w)) in custom.iter_mut().zip(used) {
            *slot = crate::engine::CustomInput { id: k as u32 + 1, weight: w };
        }
    }
    for (slot, c) in custom.iter_mut().zip(v["custom"].as_array().into_iter().flatten()) {
        *slot = crate::engine::CustomInput { id: c["id"].as_u64().unwrap_or(0) as u32, weight: c["weight"].as_f64().unwrap_or(0.0) as f32 };
    }
    let mut g = Generator {
        kind,
        weights,
        level: f(v, "level", d.level),
        contrast: f(v, "contrast", d.contrast),
        scale: f(v, "scale", d.scale),
        noise: f(v, "noise", d.noise),
        invert: b(v, "invert", d.invert),
        seed: f(v, "seed", d.seed),
        edge_width: f(v, "edge_width", d.edge_width),
        id_color: arr(v, "id_color", d.id_color),
        id_tolerance: f(v, "id_tolerance", d.id_tolerance),
        id_more: d.id_more,
        id_count: 1,
        custom,
    };
    // Older projects have just `id_color`.
    let colors: Vec<[f32; 3]> = v["id_colors"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|c| std::array::from_fn(|k| c[k].as_f64().unwrap_or(0.0) as f32))
        .collect();
    g.set_id_colors(&colors);
    Ok(g)
}

impl PainterApp {
    fn tex_json(&self, w: &mut Writer, t: &Tex) -> Value {
        let format = FORMATS.iter().find(|&&x| x == t.format).map(|x| format!("{x:?}"));
        json!({
            "blob": w.blob(&self.engine.read_tex(t)),
            "format": format.expect("only project formats are saved"),
            "width": t.width,
            "height": t.height,
        })
    }

    fn tex_from(&self, r: &Reader, v: &Value, label: &str) -> Result<Tex, String> {
        let name = s(v, "format").ok_or("texture without a format")?;
        let format = FORMATS.into_iter().find(|x| format!("{x:?}") == name).ok_or(format!("unknown texture format '{name}'"))?;
        let w = v["width"].as_u64().and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| format!("{label} texture has an invalid width"))?;
        let h = v["height"].as_u64().and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| format!("{label} texture has an invalid height"))?;
        let max_dimension = self.rs.device.limits().max_texture_dimension_2d;
        if w == 0 || h == 0 || w > max_dimension || h > max_dimension {
            return Err(format!("{label} texture dimensions exceed the GPU limit"));
        }
        let bpp = crate::gpu::bytes_per_texel(format) as usize;
        let expected = (w as usize).checked_mul(h as usize)
            .and_then(|pixels| pixels.checked_mul(bpp))
            .ok_or_else(|| format!("{label} texture size overflows"))?;
        let bytes = r.blob(&v["blob"])?;
        if bytes.len() != expected {
            return Err(format!("{label} texture has the wrong size"));
        }
        Ok(self.engine.texture_from_bytes(label, format, w, h, &bytes))
    }

    /// One texture set's part of a project.
    fn set_json(&self, w: &mut Writer, engine: &Engine) -> Value {
        let mut layers = Vec::new();
        for l in &engine.layers {
            let content = match &l.content {
                Content::Paint(ch) => {
                    let channels: Vec<Value> = ch.iter().map(|t| t.as_ref().map_or(Value::Null, |t| self.tex_json(w, t))).collect();
                    json!({"kind": "paint", "channels": channels})
                }
                Content::Fill(p) => json!({"kind": "fill", "fill": fill_json(p)}),
                Content::Folder(fo) => json!({"kind": "folder", "open": fo.open, "graph": fo.graph}),
            };
            let mask = l.mask.as_ref().map(|m| {
                json!({
                    "paint": m.paint.as_ref().map(|t| self.tex_json(w, t)),
                    "generator": m.generator.as_ref().map(generator_json),
                    "graph": m.graph,
                    "filters": mask_filters_json(&m.filters),
                    "layers": mask_layers_json(&m.layers),
                })
            });
            // A graph's outputs are rendered again on load, not stored.
            let from_graph = matches!(&l.content, Content::Fill(p) if p.graph.is_some());
            let images: Vec<Value> = l
                .images
                .iter()
                .map(|im| im.as_ref().filter(|_| !from_graph))
                .map(|im| im.map_or(Value::Null, |im| json!({"name": im.name, "data": w.blob(&im.encoded)})))
                .collect();
            layers.push(json!({
                "images": images,
                "id": l.id,
                "name": l.name,
                "visible": l.visible,
                "opacity": l.opacity,
                "blend": l.blend.name(),
                "parent": l.parent,
                "content": content,
                "mask": mask,
            }));
        }
        let fixes: Vec<Value> = (0..engine.map_fix.len())
            .filter_map(|i| Some(json!({"map": i, "texture": self.tex_json(w, engine.map_fix[i].as_ref()?)})))
            .collect();
        let levels: Vec<Value> = engine.map_levels.iter().map(|l| json!([l.black, l.white, l.gamma])).collect();
        json!({
            "resolution": engine.res,
            "layers_bottom_to_top": layers,
            "map_fixes": fixes,
            "map_levels": levels,
        })
    }

    pub(super) fn save_project(&mut self, path: &Path) -> Result<(), String> {
        let bytes = self.write_project(path)?;
        self.project_path = Some(path.to_path_buf());
        self.saved_serial = self.undo.state_serial();
        self.status = format!("Saved {} ({:.1} MB).", path.display(), bytes as f64 / 1048576.0);
        self.remember_recent(path);
        remove_autosave();
        self.autosaved_serial = self.saved_serial;
        Ok(())
    }

    /// Writes the recovery copy if enough has changed since the last one.
    pub(super) fn autosave_if_due(&mut self) {
        let every = self.prefs.autosave_every.max(1) as u64;
        let serial = self.undo.state_serial();
        if !self.prefs.autosave || self.stroke.is_some() || !self.is_modified() || serial.abs_diff(self.autosaved_serial) < every {
            return;
        }
        let Some(path) = autosave_path() else { return };
        match self.write_project(&path) {
            Ok(_) => self.autosaved_serial = self.undo.state_serial(),
            Err(e) => self.status = format!("Autosave failed: {e}"),
        }
    }

    /// Writes the project file; returns its size. Doesn't change which file
    /// the document is saved to.
    fn write_project(&mut self, path: &Path) -> Result<usize, String> {
        self.engine.commit_props(&mut self.undo);
        let mut w = Writer { blobs: Vec::new() };
        let mesh = &self.mesh;
        let mesh_json = json!({
            "name": mesh.name,
            "center": mesh.center.to_array(),
            "scale": mesh.scale,
            "vertices": w.blob(bytemuck::cast_slice(&mesh.vertices)),
            "indices": w.blob(bytemuck::cast_slice(&mesh.indices)),
            "objects": mesh.objects,
            "tri_object": w.blob(bytemuck::cast_slice(&mesh.tri_object)),
            "tri_color": mesh.tri_color.as_ref().map(|c| w.blob(bytemuck::cast_slice(c))),
            // Per-corner vertex colors: the ID mask palette and the mesh eyedropper read them.
            "tri_vertex_color": mesh.tri_vertex_color.as_ref().map(|c| w.blob(bytemuck::cast_slice(c))),
        });
        // Every texture set: its layers, size and mesh-map corrections.
        let engines = super::texture_sets::engines(&self.engine, &self.sets, self.mesh.sets.len());
        let sets: Vec<Value> = self
            .mesh
            .sets
            .iter()
            .zip(&engines)
            .enumerate()
            .map(|(k, (set, engine))| {
                let selected = match self.sets.parked.get(k).and_then(Option::as_ref) {
                    Some(p) => p.selected,
                    None => self.selected,
                };
                let mut v = self.set_json(&mut w, engine);
                v["name"] = json!(set.name);
                v["tris"] = json!([set.tris.start, set.tris.end]);
                v["selected"] = json!(selected);
                v
            })
            .collect();
        let bs = &self.bake_settings;
        let bake = json!({
            "quality": format!("{:?}", bs.quality),
            "high_poly": bs.high_poly.as_ref().map(|p| p.display().to_string()),
            "id_source": format!("{:?}", bs.id_source),
            "front": bs.front, "rear": bs.rear, "aa": bs.aa,
            "ao_rays": bs.ao_rays, "ao_distance": bs.ao_distance, "cavity_distance": bs.cavity_distance,
            "thickness_rays": bs.thickness_rays, "thickness_distance": bs.thickness_distance,
            "sky_rays": bs.sky_rays, "sky_cone_deg": bs.sky_cone_deg,
            "curvature_radius": bs.curvature_radius, "curvature_scale": bs.curvature_scale,
            "edge_threshold": bs.edge_threshold, "edge_max_distance": bs.edge_max_distance,
            "edge_auto": bs.edge_auto, "hard_edge_angle": bs.hard_edge_angle,
            "match_by_name": bs.match_by_name,
            "cage": bs.cage.as_ref().map(|p| p.display().to_string()),
            "average_normals": bs.average_normals,
            "ignore_backfaces": bs.ignore_backfaces,
            "fill_misses": bs.fill_misses,
            // Rounded: three decimals are finer than a brush stroke.
            "cage_paint": bs.cage_paint.as_ref().map(|p| p.scales().iter().map(|s| (s * 1000.0).round() / 1000.0).collect::<Vec<_>>()),
            "maps": bs.maps,
            "custom": bs.custom.iter().map(custom_map_json).collect::<Vec<_>>(),
        });
        let v = &self.view;
        let p = &v.post;
        let env = match &self.env.source {
            EnvSource::Builtin(i) => json!({"builtin": crate::environment::BUILTIN[*i].0}),
            EnvSource::File(path) => json!({"file": path.display().to_string()}),
        };
        let view = json!({
            "preview_shader": self.preview.graph,
            "environment": env,
            "exposure": v.exposure, "env_rotation": v.env_rotation, "env_intensity": v.env_intensity,
            "background": format!("{:?}", v.background), "bg_blur": v.bg_blur, "bg_intensity": v.bg_intensity,
            "bg_color": v.bg_color, "height_strength": v.height_strength,
            "show_wire": v.show_wire, "use_baked_normal": v.use_baked_normal, "ao_strength": v.ao_strength, "emission": v.emission,
            "key_light": {"on": v.key.on, "azimuth": v.key.azimuth, "elevation": v.key.elevation,
                "intensity": v.key.intensity, "color": v.key.color},
            "tonemap": format!("{:?}", p.tonemap), "contrast": p.contrast, "saturation": p.saturation,
            "temperature": p.temperature, "tint": p.tint, "vignette": p.vignette, "bloom": p.bloom,
            "bloom_threshold": p.bloom_threshold, "chromatic": p.chromatic, "grain": p.grain, "sharpen": p.sharpen,
            "camera": {"yaw": self.cam.yaw, "pitch": self.cam.pitch, "distance": self.cam.dist,
                       "target": self.cam.target.to_array(), "fov": self.cam.fov_y},
        });
        let manifest = json!({
            "app": "Material Painter",
            // 2: map levels gamma brightens above 1 (version 1 darkened).
            // 3: texture sets (layers, size and map fixes per set).
            "version": 3,
            "mesh": mesh_json,
            "texture_sets": sets,
            "active_set": self.sets.active,
            "graphs": self.graphs.to_json(&self.used_graphs()),
            "bake": bake,
            "view": view,
        });
        let text = serde_json::to_vec(&manifest).map_err(|e| e.to_string())?;
        // Stream the container to the temporary file. The compressed blobs
        // already live in Writer; assembling a second project-sized Vec here
        // needlessly doubled peak memory for large texture sets.
        let tmp = path.with_extension("mpp.tmp");
        let mut file = std::fs::File::create(&tmp).map_err(|e| format!("can't write {}: {e}", tmp.display()))?;
        let mut bytes = 0usize;
        let mut write = |chunk: &[u8]| -> Result<(), String> {
            file.write_all(chunk).map_err(|e| format!("can't write {}: {e}", tmp.display()))?;
            bytes = bytes.checked_add(chunk.len()).ok_or_else(|| "project size overflow".to_string())?;
            Ok(())
        };
        let manifest_len = u64::try_from(text.len()).map_err(|_| "manifest is too large".to_string())?;
        let blob_count = u64::try_from(w.blobs.len()).map_err(|_| "too many project blobs".to_string())?;
        write(MAGIC)?;
        write(&manifest_len.to_le_bytes())?;
        write(&text)?;
        write(&blob_count.to_le_bytes())?;
        for blob in &w.blobs {
            let blob_len = u64::try_from(blob.len()).map_err(|_| "project blob is too large".to_string())?;
            write(&blob_len.to_le_bytes())?;
            write(blob)?;
        }
        drop(write);
        file.flush().map_err(|e| format!("can't finish {}: {e}", tmp.display()))?;
        file.sync_all().map_err(|e| format!("can't sync {}: {e}", tmp.display()))?;
        drop(file);
        std::fs::rename(&tmp, path).map_err(|e| format!("can't replace {}: {e}", path.display()))?;
        Ok(bytes)
    }

    fn remember_recent(&mut self, path: &Path) {
        if is_autosave(path) {
            return;
        }
        self.recent.retain(|p| p != path);
        self.recent.insert(0, path.to_path_buf());
        self.recent.truncate(self.prefs.recent_count.max(1));
        if let Some(dir) = data_dir() {
            let list: Vec<String> = self.recent.iter().map(|p| p.display().to_string()).collect();
            let _ = std::fs::create_dir_all(&dir);
            let _ = std::fs::write(dir.join("recent.json"), json!(list).to_string());
        }
    }

    pub(super) fn open_project(&mut self, path: &Path, ctx: &egui::Context) -> Result<(), String> {
        let data = std::fs::read(path).map_err(|e| format!("can't read {}: {e}", path.display()))?;
        let bad = || format!("{} isn't a Material Painter project (or is damaged)", path.display());
        if data.len() < 24 || &data[..8] != MAGIC {
            return Err(bad());
        }
        let mut at = 8usize;
        let mut take = |n: usize| -> Result<&[u8], String> {
            let end = at.checked_add(n).ok_or_else(bad)?;
            let slice = data.get(at..end).ok_or_else(bad)?;
            at = end;
            Ok(slice)
        };
        let u64_at = |b: &[u8]| -> Result<usize, String> {
            let bytes: [u8; 8] = b.try_into().map_err(|_| bad())?;
            usize::try_from(u64::from_le_bytes(bytes)).map_err(|_| bad())
        };
        const MAX_MANIFEST_BYTES: usize = 64 * 1024 * 1024;
        const MAX_PROJECT_BLOBS: usize = 1_000_000;
        let len = u64_at(take(8)?)?;
        if len > MAX_MANIFEST_BYTES {
            return Err(bad());
        }
        let manifest: Value = serde_json::from_slice(take(len)?).map_err(|e| format!("{}: {e}", bad()))?;
        let count = u64_at(take(8)?)?;
        drop(take);
        if count > MAX_PROJECT_BLOBS || count > data.len().saturating_sub(at) / 8 {
            return Err(bad());
        }
        let mut take = |n: usize| -> Result<&[u8], String> {
            let end = at.checked_add(n).ok_or_else(bad)?;
            let slice = data.get(at..end).ok_or_else(bad)?;
            at = end;
            Ok(slice)
        };
        let mut blobs = Vec::with_capacity(count);
        for _ in 0..count {
            let n = u64_at(take(8)?)?;
            blobs.push(take(n)?);
        }
        drop(take);
        if at != data.len() {
            return Err(bad());
        }
        let r = Reader { blobs };

        // Everything is decoded before the open document is touched.
        let m = &manifest["mesh"];
        let vertices: Vec<Vertex> = bytemuck::pod_collect_to_vec(&r.blob(&m["vertices"])?);
        let indices: Vec<u32> = bytemuck::pod_collect_to_vec(&r.blob(&m["indices"])?);
        if indices.iter().any(|&i| i as usize >= vertices.len()) {
            return Err(bad());
        }
        let center = glam::Vec3::from_array(arr(m, "center", [0.0; 3]));
        let tris = indices.len() / 3;
        let mut mesh = Mesh::from_saved(s(m, "name").unwrap_or("Mesh").to_string(), vertices, indices, center, f(m, "scale", 1.0));
        if let Some(names) = m["objects"].as_array() {
            let tri_object: Vec<u32> = bytemuck::pod_collect_to_vec(&r.blob(&m["tri_object"])?);
            if tri_object.len() == tris && tri_object.iter().all(|&o| (o as usize) < names.len()) {
                mesh.objects = names.iter().map(|n| n.as_str().unwrap_or("").to_string()).collect();
                mesh.tri_object = tri_object;
            }
        }
        if !m["tri_color"].is_null() {
            let colors: Vec<[u8; 3]> = r.blob(&m["tri_color"])?.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
            if colors.len() == tris {
                mesh.tri_color = Some(colors);
            }
        }
        // Projects saved before this key existed only have the per-triangle average.
        if !m["tri_vertex_color"].is_null() {
            mesh.tri_vertex_color = corner_colors_from_bytes(&r.blob(&m["tri_vertex_color"])?, tris);
        }
        let bv = &manifest["bake"];
        let mut bs = crate::bake::BakeSettings::default();
        if let Some(q) = s(bv, "quality") {
            bs.quality = Quality::ALL.into_iter().find(|x| format!("{x:?}") == q).unwrap_or(bs.quality);
        }
        bs.high_poly = s(bv, "high_poly").map(PathBuf::from).filter(|p| p.is_file());
        bs.front = f(bv, "front", bs.front);
        bs.rear = f(bv, "rear", bs.rear);
        bs.aa = bv["aa"].as_u64().map_or(bs.aa, |x| x as u32);
        bs.ao_rays = bv["ao_rays"].as_u64().map_or(bs.ao_rays, |x| x as u32);
        bs.ao_distance = f(bv, "ao_distance", bs.ao_distance);
        bs.cavity_distance = f(bv, "cavity_distance", bs.cavity_distance);
        bs.thickness_rays = bv["thickness_rays"].as_u64().map_or(bs.thickness_rays, |x| x as u32);
        bs.thickness_distance = f(bv, "thickness_distance", bs.thickness_distance);
        bs.sky_rays = bv["sky_rays"].as_u64().map_or(bs.sky_rays, |x| x as u32);
        bs.sky_cone_deg = f(bv, "sky_cone_deg", bs.sky_cone_deg);
        bs.curvature_radius = f(bv, "curvature_radius", bs.curvature_radius);
        bs.curvature_scale = f(bv, "curvature_scale", bs.curvature_scale);
        bs.edge_threshold = f(bv, "edge_threshold", bs.edge_threshold);
        bs.edge_max_distance = f(bv, "edge_max_distance", bs.edge_max_distance);
        bs.edge_auto = b(bv, "edge_auto", bs.edge_auto);
        bs.hard_edge_angle = f(bv, "hard_edge_angle", bs.hard_edge_angle);
        bs.match_by_name = b(bv, "match_by_name", bs.match_by_name);
        bs.cage = s(bv, "cage").map(PathBuf::from).filter(|p| p.is_file());
        bs.average_normals = b(bv, "average_normals", bs.average_normals);
        bs.ignore_backfaces = b(bv, "ignore_backfaces", bs.ignore_backfaces);
        bs.fill_misses = b(bv, "fill_misses", bs.fill_misses);
        bs.cage_paint = bv["cage_paint"].as_array().map(|a| {
            std::sync::Arc::new(crate::bake::CagePaint::new(a.iter().map(|x| x.as_f64().unwrap_or(1.0) as f32).collect()))
        });
        // Projects from before map selection baked everything.
        bs.maps = bv["maps"].as_u64().map_or(crate::bake::ALL_MAPS, |m| m as u32);
        bs.custom = bv["custom"]
            .as_array()
            .into_iter()
            .flatten()
            .enumerate()
            .filter_map(|(k, cv)| custom_map_from(cv, k as u32 + 1))
            .collect();
        bs.id_source = match s(bv, "id_source") {
            Some("Materials") => IdSource::Materials,
            Some("VertexColors") => IdSource::VertexColors,
            _ => IdSource::MeshParts,
        };
        // The maps stored in the project were baked with these settings.
        self.baked_id_source = bs.id_source;

        // Texture sets: projects before version 3 have one, at the top level.
        let set_values: Vec<&Value> = match manifest["texture_sets"].as_array() {
            Some(a) if !a.is_empty() => a.iter().collect(),
            _ => vec![&manifest],
        };
        let ranges: Vec<(String, std::ops::Range<u32>)> = set_values
            .iter()
            .filter_map(|sv| {
                let t = sv["tris"].as_array()?;
                let (a, b) = (t.first()?.as_u64()? as u32, t.get(1)?.as_u64()? as u32);
                Some((s(sv, "name").unwrap_or("Material").to_string(), a..b))
            })
            .collect();
        if ranges.len() == set_values.len() {
            mesh.set_ranges(ranges);
        }
        let set_count = mesh.sets.len();

        // Switch mesh first, then each set's size: layers are created at it.
        self.bake_settings = bs;
        let sizes: Vec<u32> = set_values.iter().map(|sv| sv["resolution"].as_u64().unwrap_or(2048) as u32).collect();
        self.load_mesh_sized(mesh, ctx, &sizes);
        self.graphs.clear(&self.rs);
        for gv in manifest["graphs"].as_array().into_iter().flatten() {
            let Some(id) = gv["id"].as_u64() else { continue };
            match crate::graph::Graph::from_json(&gv["graph"], &self.graphs.lib) {
                Ok(g) => {
                    self.graphs.insert(&self.rs, g, Some(id));
                }
                Err(e) => self.status = format!("A material graph didn't load: {e}"),
            }
        }
        for (k, sv) in set_values.iter().enumerate().take(set_count) {
            self.switch_set(k);
            while !self.engine.layers.is_empty() {
                self.engine.delete_layer(0, &mut self.undo);
            }
            let res = sv["resolution"].as_u64().unwrap_or(2048) as u32;
            if res != self.engine.res {
                self.engine.set_resolution(res, &self.gpu_mesh, &mut self.undo);
            }
            let mut layers = Vec::new();
            for lv in sv["layers_bottom_to_top"].as_array().ok_or_else(bad)? {
                let name = s(lv, "name").unwrap_or("Layer").to_string();
                let c = &lv["content"];
                let content = match s(c, "kind") {
                    Some("paint") => {
                        let ch = c["channels"].as_array().ok_or_else(bad)?;
                        let mut out: [Option<Tex>; crate::engine::CHANNELS] = Default::default();
                        for (k, t) in ch.iter().enumerate().take(crate::engine::CHANNELS) {
                            if !t.is_null() {
                                out[k] = Some(self.tex_from(&r, t, "layer")?);
                            }
                        }
                        Content::Paint(out)
                    }
                    Some("fill") => Content::Fill(fill_from(&c["fill"])),
                    Some("folder") => Content::Folder(Folder { open: b(c, "open", true), graph: c["graph"].as_u64() }),
                    _ => return Err(bad()),
                };
                let mask = match lv.get("mask").filter(|m| !m.is_null()) {
                    Some(mv) => Some(Mask {
                        black_base: false,
                        paint: match mv.get("paint").filter(|p| !p.is_null()) {
                            Some(t) => Some(self.tex_from(&r, t, "mask")?),
                            None => None,
                        },
                        generator: match mv.get("generator").filter(|g| !g.is_null()) {
                            Some(g) => Some(generator_from(g)?),
                            None => None,
                        },
                        graph: mv["graph"].as_u64(),
                        filters: mask_filters_from(&mv["filters"]),
                        layers: mask_layers_from(&mv["layers"]),
                    }),
                    None => None,
                };
                let mut images: [Option<crate::engine::FillImage>; crate::engine::CHANNELS] = Default::default();
                for (k, iv) in lv["images"].as_array().into_iter().flatten().enumerate().take(crate::engine::CHANNELS) {
                    if !iv.is_null() {
                        let data = std::sync::Arc::new(r.blob(&iv["data"])?);
                        images[k] = Some(self.engine.decode_channel_image(k, s(iv, "name").unwrap_or("image").to_string(), data)?);
                    }
                }
                let blend_name = s(lv, "blend").unwrap_or("Normal");
                layers.push(Layer {
                    id: lv["id"].as_u64().ok_or_else(bad)?,
                    name,
                    visible: b(lv, "visible", true),
                    opacity: f(lv, "opacity", 1.0),
                    blend: Blend::ALL.into_iter().find(|x| x.name() == blend_name).unwrap_or(Blend::Normal),
                    content,
                    mask,
                    parent: lv["parent"].as_u64(),
                    images,
                    thumb: None,
                });
            }
            self.engine.restore_layers(layers);
            self.engine.map_fix = Default::default();
            for fv in sv["map_fixes"].as_array().into_iter().flatten() {
                let map = fv["map"].as_u64().unwrap_or(u64::MAX) as usize;
                if map < self.engine.map_fix.len() {
                    self.engine.map_fix[map] = Some(self.tex_from(&r, &fv["texture"], "map fix")?);
                }
            }
            for (dst, lv) in self.engine.map_levels.iter_mut().zip(sv["map_levels"].as_array().into_iter().flatten()) {
                let a = lv.as_array().filter(|a| a.len() == 3);
                if let Some(a) = a {
                    let g = |i: usize| a[i].as_f64().unwrap_or(0.0) as f32;
                    // Version 1 projects stored the inverse gamma; keep their look.
                    let gamma = if manifest["version"].as_u64().unwrap_or(1) < 2 { 1.0 / g(2).max(1e-3) } else { g(2) };
                    *dst = Levels { black: g(0), white: g(1), gamma };
                }
            }
            self.engine.map_levels_changed();
            self.selected = sv["selected"].as_u64().map_or(0, |x| x as usize).min(self.engine.layers.len().saturating_sub(1));
        }
        self.switch_set(manifest["active_set"].as_u64().map_or(0, |x| x as usize).min(set_count - 1));
        self.repush_graphs();

        let v = &manifest["view"];
        let d = crate::viewport::ViewSettings::default();
        let vs = &mut self.view;
        vs.exposure = f(v, "exposure", d.exposure);
        vs.env_rotation = f(v, "env_rotation", d.env_rotation);
        vs.env_intensity = f(v, "env_intensity", d.env_intensity);
        vs.ao_strength = f(v, "ao_strength", d.ao_strength);
        vs.emission = f(v, "emission", d.emission);
        vs.background = Background::ALL.into_iter().find(|x| Some(format!("{x:?}").as_str()) == s(v, "background")).unwrap_or(d.background);
        vs.bg_blur = f(v, "bg_blur", d.bg_blur);
        vs.bg_intensity = f(v, "bg_intensity", d.bg_intensity);
        vs.bg_color = arr(v, "bg_color", d.bg_color);
        vs.height_strength = f(v, "height_strength", d.height_strength);
        vs.show_wire = b(v, "show_wire", d.show_wire);
        vs.use_baked_normal = b(v, "use_baked_normal", d.use_baked_normal);
        self.preview.graph = v["preview_shader"].as_u64().filter(|id| self.graphs.doc(*id).is_some_and(|d| d.shading));
        let k = &v["key_light"];
        // Projects saved before the key light existed open as they looked.
        vs.key = if k.is_null() {
            crate::viewport::KeyLight { on: false, ..d.key }
        } else {
            crate::viewport::KeyLight {
                on: b(k, "on", d.key.on),
                azimuth: f(k, "azimuth", d.key.azimuth),
                elevation: f(k, "elevation", d.key.elevation),
                intensity: f(k, "intensity", d.key.intensity),
                color: arr(k, "color", d.key.color),
            }
        };
        let p = &mut vs.post;
        p.tonemap = Tonemap::ALL.into_iter().find(|x| Some(format!("{x:?}").as_str()) == s(v, "tonemap")).unwrap_or(d.post.tonemap);
        p.contrast = f(v, "contrast", d.post.contrast);
        p.saturation = f(v, "saturation", d.post.saturation);
        p.temperature = f(v, "temperature", d.post.temperature);
        p.tint = f(v, "tint", d.post.tint);
        p.vignette = f(v, "vignette", d.post.vignette);
        p.bloom = f(v, "bloom", d.post.bloom);
        p.bloom_threshold = f(v, "bloom_threshold", d.post.bloom_threshold);
        p.chromatic = f(v, "chromatic", d.post.chromatic);
        p.grain = f(v, "grain", d.post.grain);
        p.sharpen = f(v, "sharpen", d.post.sharpen);
        let cam = &v["camera"];
        let dc = OrbitCam::default();
        self.cam = OrbitCam {
            yaw: f(cam, "yaw", dc.yaw),
            pitch: f(cam, "pitch", dc.pitch),
            dist: f(cam, "distance", dc.dist),
            target: glam::Vec3::from_array(arr(cam, "target", dc.target.to_array())),
            fov_y: f(cam, "fov", dc.fov_y),
        };
        let env = &v["environment"];
        let source = match (s(env, "builtin"), s(env, "file")) {
            (Some(name), _) => Some(EnvSource::parse(name)),
            (_, Some(file)) if Path::new(file).is_file() => Some(EnvSource::File(PathBuf::from(file))),
            _ => None,
        };
        if let Some(src) = source.filter(|src| *src != self.env.source) {
            self.load_env(src, ctx);
        }

        self.undo.clear();
        self.saved_serial = self.undo.state_serial();
        self.autosaved_serial = self.saved_serial;
        if is_autosave(path) {
            // Recovered work isn't saved anywhere yet; its file goes once opened.
            remove_recovery(path);
            self.project_path = None;
            self.saved_serial = u64::MAX;
            self.status = "Recovered your unsaved work. Save it to keep it.".into();
        } else {
            self.project_path = Some(path.to_path_buf());
            self.remember_recent(path);
            self.status = format!("Opened {}.", path.display());
        }
        Ok(())
    }

    /// Unsaved edits since the last save or open.
    pub(super) fn is_modified(&self) -> bool {
        self.undo.state_serial() != self.saved_serial || self.engine.has_uncommitted_props()
    }

    pub(super) fn save_dialog(&self) -> Option<PathBuf> {
        let name = self.project_path.as_ref().and_then(|p| p.file_name()).map_or_else(
            || format!("{}.{EXTENSION}", self.mesh.name.replace(' ', "")),
            |n| n.to_string_lossy().into_owned(),
        );
        rfd::FileDialog::new().add_filter("Material Painter project", &[EXTENSION]).set_file_name(name).save_file()
    }

    pub(super) fn open_dialog() -> Option<PathBuf> {
        rfd::FileDialog::new().add_filter("Material Painter project", &[EXTENSION]).pick_file()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Projects from the four fixed custom slots load as maps with ids 1-4
    /// (empty slots skipped), and their masks' extra weights as inputs on
    /// those ids.
    #[test]
    fn old_custom_slots_migrate() {
        let slots = json!([{"kind": "Flow", "elevation": 90.0}, null, {"kind": "Deposit"}, null]);
        let maps: Vec<_> = slots.as_array().unwrap().iter().enumerate().filter_map(|(k, v)| custom_map_from(v, k as u32 + 1)).collect();
        assert_eq!(maps.iter().map(|c| (c.id, c.name.as_str())).collect::<Vec<_>>(), vec![(1, "Fluid Flow"), (3, "Wind Deposits")]);

        let mut weights = vec![0.0; LEGACY_INPUTS];
        weights[crate::engine::gin::OCCLUSION] = 0.4;
        weights.extend([0.0, 0.0, 1.0, 0.0]);
        let g = generator_from(&json!({"kind": "Custom Bake Map", "weights": weights})).unwrap();
        assert_eq!(g.weights[crate::engine::gin::OCCLUSION], 0.4);
        assert_eq!((g.custom[0].id, g.custom[0].weight), (3, 1.0), "old slot 3 reads the map with id 3");
        assert!(g.custom[1..].iter().all(|c| c.id == 0));

        // And the new format round-trips.
        let back = generator_from(&generator_json(&g)).unwrap();
        assert_eq!(back, g);
    }

    #[test]
    fn corner_colors_survive_a_save() {
        let colors: Vec<[[u8; 3]; 3]> = vec![[[255, 0, 0], [0, 255, 0], [0, 0, 255]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]];
        let bytes: Vec<u8> = bytemuck::cast_slice::<[[u8; 3]; 3], u8>(&colors).to_vec();
        assert_eq!(bytes.len(), 18);
        assert_eq!(corner_colors_from_bytes(&bytes, 2), Some(colors));
        // A color list that doesn't match the mesh is dropped, not misaligned.
        assert_eq!(corner_colors_from_bytes(&bytes, 3), None);
        assert_eq!(corner_colors_from_bytes(&bytes[..17], 2), None);
    }
}
