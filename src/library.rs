//! Online asset libraries: free CC0 materials from Poly Haven and ambientCG,
//! and the community library of graphs and nodes people share.
//!
//! All network and disk work runs on a few worker threads ([`Net`]); the UI
//! sends [`Job`]s and polls [`Event`]s. Catalogs and thumbnails are cached in
//! the settings folder; downloaded materials live in `library/<source>-<id>/`
//! as plain image files plus a `meta.json`, so they stay usable (and readable
//! by any other tool) without the app.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use eframe::egui;
use serde_json::{Value, json};

const USER_AGENT: &str = concat!("MaterialPainter/", env!("CARGO_PKG_VERSION"), " (https://github.com/alla2001/material-painter)");
/// The community library: a GitHub repository with an `index.json`.
pub const COMMUNITY_REPO: &str = "alla2001/material-painter-community";
const COMMUNITY_RAW: &str = "https://raw.githubusercontent.com/alla2001/material-painter-community/main/";
/// Catalog listings are refreshed at most this often.
const CATALOG_MAX_AGE: Duration = Duration::from_secs(24 * 3600);
const WORKERS: usize = 6;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Source {
    PolyHaven,
    AmbientCg,
    Community,
}

impl Source {
    pub const ALL: [Source; 3] = [Source::PolyHaven, Source::AmbientCg, Source::Community];
    pub fn name(self) -> &'static str {
        match self {
            Source::PolyHaven => "Poly Haven",
            Source::AmbientCg => "ambientCG",
            Source::Community => "Community",
        }
    }
    fn key(self) -> &'static str {
        match self {
            Source::PolyHaven => "polyhaven",
            Source::AmbientCg => "ambientcg",
            Source::Community => "community",
        }
    }
    pub fn site(self) -> &'static str {
        match self {
            Source::PolyHaven => "https://polyhaven.com/textures",
            Source::AmbientCg => "https://ambientcg.com",
            Source::Community => "https://github.com/alla2001/material-painter-community",
        }
    }
}

/// What a catalog entry is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// A PBR texture set (becomes a fill layer).
    Material,
    /// A material graph (`.mpgraph`, onto the Graphs shelf).
    Graph,
    /// A custom node (`.wgsl`, into the node library).
    Node,
    /// A smart material (`.mpsmart`, onto the Smart materials shelf).
    Smart,
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub source: Source,
    pub id: String,
    pub name: String,
    pub kind: Kind,
    pub author: String,
    pub description: String,
    /// Lowercase search text: name, tags and categories.
    pub search: String,
    pub thumb_url: String,
    /// Community entries: files relative to the repository root.
    pub files: Vec<String>,
}

impl Entry {
    /// Where this entry's thumbnail and download are cached.
    pub fn key(&self) -> String {
        format!("{}-{}", self.source.key(), sanitize(&self.id))
    }
}

/// Texture resolution to download.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Res {
    K1,
    K2,
    K4,
}

impl Res {
    pub const ALL: [Res; 3] = [Res::K1, Res::K2, Res::K4];
    pub fn label(self) -> &'static str {
        match self {
            Res::K1 => "1K",
            Res::K2 => "2K",
            Res::K4 => "4K",
        }
    }
    fn polyhaven(self) -> &'static str {
        match self {
            Res::K1 => "1k",
            Res::K2 => "2k",
            Res::K4 => "4k",
        }
    }
}

pub enum Job {
    Catalog(Source),
    Thumb { key: String, url: String },
    Download { entry: Entry, res: Res },
}

pub enum Event {
    Catalog(Source, Result<Vec<Entry>, String>),
    Thumb { key: String, image: Option<egui::ColorImage> },
    Progress { key: String, done: u64, total: u64 },
    Downloaded { key: String, kind: Kind, result: Result<PathBuf, String> },
}

/// Worker threads for network and disk jobs.
pub struct Net {
    jobs: Sender<Job>,
    pub events: Receiver<Event>,
}

impl Net {
    pub fn new(ctx: &egui::Context) -> Net {
        let (jobs, job_rx) = mpsc::channel::<Job>();
        let (tx, events) = mpsc::channel::<Event>();
        let job_rx = Arc::new(Mutex::new(job_rx));
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .user_agent(USER_AGENT)
            .timeout_connect(Some(Duration::from_secs(15)))
            .timeout_recv_body(Some(Duration::from_secs(120)))
            .build()
            .into();
        for i in 0..WORKERS {
            let (job_rx, tx, ctx, agent) = (job_rx.clone(), tx.clone(), ctx.clone(), agent.clone());
            let _ = std::thread::Builder::new().name(format!("library {i}")).spawn(move || {
                loop {
                    // Hold the lock only to take the next job.
                    let job = match job_rx.lock() {
                        Ok(rx) => rx.recv(),
                        Err(_) => return,
                    };
                    let Ok(job) = job else { return };
                    let event = run(&agent, job, &tx, &ctx);
                    if tx.send(event).is_err() {
                        return;
                    }
                    ctx.request_repaint();
                }
            });
        }
        Net { jobs, events }
    }

    pub fn send(&self, job: Job) {
        let _ = self.jobs.send(job);
    }
}

fn run(agent: &ureq::Agent, job: Job, tx: &Sender<Event>, ctx: &egui::Context) -> Event {
    match job {
        Job::Catalog(source) => Event::Catalog(source, catalog(agent, source)),
        Job::Thumb { key, url } => Event::Thumb { image: thumb(agent, &key, &url), key },
        Job::Download { entry, res } => {
            let key = entry.key();
            let progress = |done: u64, total: u64| {
                let _ = tx.send(Event::Progress { key: key.clone(), done, total });
                ctx.request_repaint();
            };
            let result = match entry.source {
                Source::PolyHaven => download_polyhaven(agent, &entry, res, &progress),
                Source::AmbientCg => download_ambientcg(agent, &entry, res, &progress),
                Source::Community => download_community(agent, &entry, &progress),
            };
            Event::Downloaded { kind: entry.kind, key, result }
        }
    }
}

// ---------------------------------------------------------------- folders

fn data_dir() -> Option<PathBuf> {
    crate::crash::data_dir()
}

/// Downloaded materials.
pub fn library_dir() -> Option<PathBuf> {
    data_dir().map(|d| d.join("library"))
}

fn cache_dir() -> Option<PathBuf> {
    data_dir().map(|d| d.join("library_cache"))
}

/// Keeps ids safe as file names.
fn sanitize(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect()
}

fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("part");
    std::fs::File::create(&tmp)?.write_all(bytes)?;
    std::fs::rename(&tmp, path)
}

// ---------------------------------------------------------------- http

fn get_text(agent: &ureq::Agent, url: &str) -> Result<String, String> {
    let mut resp = agent.get(url).call().map_err(|e| format!("{url}: {e}"))?;
    resp.body_mut().with_config().limit(64 << 20).read_to_string().map_err(|e| format!("{url}: {e}"))
}

fn get_json(agent: &ureq::Agent, url: &str) -> Result<Value, String> {
    serde_json::from_str(&get_text(agent, url)?).map_err(|e| format!("{url}: {e}"))
}

/// Downloads a file, reporting progress, with a hard size cap.
fn get_bytes(agent: &ureq::Agent, url: &str, progress: &dyn Fn(u64, u64)) -> Result<Vec<u8>, String> {
    const MAX_DOWNLOAD_BYTES: u64 = 1 << 30;
    const INITIAL_RESERVE: u64 = 16 << 20;

    let mut resp = agent.get(url).call().map_err(|e| format!("{url}: {e}"))?;
    let total = resp.body().content_length().unwrap_or(0);
    if total > MAX_DOWNLOAD_BYTES {
        return Err(format!("{url}: file exceeds the 1 GiB download limit"));
    }
    let initial = usize::try_from(total.min(INITIAL_RESERVE))
        .map_err(|_| format!("{url}: invalid Content-Length"))?;
    let mut out = Vec::new();
    out.try_reserve_exact(initial).map_err(|e| format!("{url}: can't reserve download buffer: {e}"))?;
    let mut reader = resp.body_mut().with_config().limit(MAX_DOWNLOAD_BYTES).reader();
    let mut buf = vec![0u8; 256 * 1024];
    loop {
        let n = reader.read(&mut buf).map_err(|e| format!("{url}: {e}"))?;
        if n == 0 {
            break;
        }
        let next_len = out
            .len()
            .checked_add(n)
            .ok_or_else(|| format!("{url}: downloaded size overflow"))?;
        if next_len as u64 > MAX_DOWNLOAD_BYTES {
            return Err(format!("{url}: file exceeds the 1 GiB download limit"));
        }
        out.try_reserve(n).map_err(|e| format!("{url}: can't grow download buffer: {e}"))?;
        out.extend_from_slice(&buf[..n]);
        progress(next_len as u64, total);
    }
    Ok(out)
}

// ---------------------------------------------------------------- catalogs

fn catalog(agent: &ureq::Agent, source: Source) -> Result<Vec<Entry>, String> {
    let cache = cache_dir().map(|d| d.join(format!("{}.json", source.key())));
    let fresh = cache.as_ref().and_then(|p| std::fs::metadata(p).ok()).and_then(|m| m.modified().ok()).is_some_and(|t| {
        SystemTime::now().duration_since(t).is_ok_and(|age| age < CATALOG_MAX_AGE)
    });
    if fresh && let Some(raw) = cache.as_ref().and_then(|p| std::fs::read_to_string(p).ok())
        && let Ok(v) = serde_json::from_str::<Value>(&raw) {
            return Ok(parse_catalog(source, &v));
        }
    let fetched = fetch_catalog(agent, source);
    match fetched {
        Ok(v) => {
            if let Some(p) = &cache {
                let _ = write_atomic(p, v.to_string().as_bytes());
            }
            Ok(parse_catalog(source, &v))
        }
        // Offline: a stale catalog beats none.
        Err(e) => match cache.as_ref().and_then(|p| std::fs::read_to_string(p).ok()).and_then(|r| serde_json::from_str(&r).ok()) {
            Some(v) => Ok(parse_catalog(source, &v)),
            None => Err(e),
        },
    }
}

/// The raw catalog as the source serves it (cached as is).
fn fetch_catalog(agent: &ureq::Agent, source: Source) -> Result<Value, String> {
    match source {
        Source::PolyHaven => get_json(agent, "https://api.polyhaven.com/assets?t=textures"),
        Source::AmbientCg => {
            let mut all = Vec::new();
            let mut offset = 0;
            loop {
                let url = format!(
                    "https://ambientcg.com/api/v2/full_json?type=Material&include=tagData,previewData&sort=Popular&limit=250&offset={offset}"
                );
                let page = get_json(agent, &url)?;
                let found = page["foundAssets"].as_array().cloned().unwrap_or_default();
                let total = page["numberOfResults"].as_u64().unwrap_or(0) as usize;
                let n = found.len();
                all.extend(found);
                offset += n;
                if n == 0 || offset >= total {
                    break;
                }
            }
            Ok(Value::Array(all))
        }
        Source::Community => get_json(agent, &format!("{COMMUNITY_RAW}index.json")),
    }
}

fn parse_catalog(source: Source, v: &Value) -> Vec<Entry> {
    let strs = |v: &Value| -> Vec<String> { v.as_array().map(|a| a.iter().filter_map(|s| s.as_str().map(str::to_string)).collect()).unwrap_or_default() };
    let mut out = Vec::new();
    match source {
        Source::PolyHaven => {
            let Some(map) = v.as_object() else { return out };
            for (id, a) in map {
                let name = a["name"].as_str().unwrap_or(id).to_string();
                let tags = [strs(&a["tags"]), strs(&a["categories"])].concat().join(" ");
                let author = a["authors"].as_object().map(|m| m.keys().cloned().collect::<Vec<_>>().join(", ")).unwrap_or_default();
                out.push(Entry {
                    source,
                    search: format!("{} {} {}", name, id, tags).to_lowercase(),
                    thumb_url: format!("https://cdn.polyhaven.com/asset_img/thumbs/{id}.png?width=256&height=256"),
                    id: id.clone(),
                    name,
                    kind: Kind::Material,
                    author,
                    description: a["description"].as_str().unwrap_or("").to_string(),
                    files: Vec::new(),
                });
            }
            // Newest first, like the site.
            out.sort_by_key(|e| std::cmp::Reverse(map[&e.id]["date_published"].as_u64().unwrap_or(0)));
        }
        Source::AmbientCg => {
            for a in v.as_array().into_iter().flatten() {
                let Some(id) = a["assetId"].as_str() else { continue };
                let name = a["displayName"].as_str().unwrap_or(id).to_string();
                let tags = strs(&a["tags"]).join(" ");
                out.push(Entry {
                    source,
                    search: format!("{} {} {} {}", name, id, tags, a["displayCategory"].as_str().unwrap_or("")).to_lowercase(),
                    thumb_url: a["previewImage"]["256-PNG"].as_str().unwrap_or("").to_string(),
                    id: id.to_string(),
                    name,
                    kind: Kind::Material,
                    author: "ambientCG".into(),
                    description: a["description"].as_str().unwrap_or("").to_string(),
                    files: Vec::new(),
                });
            }
        }
        Source::Community => {
            for item in v["items"].as_array().into_iter().flatten() {
                let Some(id) = item["id"].as_str() else { continue };
                let kind = match item["kind"].as_str() {
                    Some("node") => Kind::Node,
                    Some("smart") => Kind::Smart,
                    _ => Kind::Graph,
                };
                let name = item["name"].as_str().unwrap_or(id).to_string();
                let base = item["path"].as_str().unwrap_or("").trim_end_matches('/').to_string();
                let join = |f: &str| if base.is_empty() { f.to_string() } else { format!("{base}/{f}") };
                let thumb = item["thumb"].as_str().map(|t| format!("{COMMUNITY_RAW}{}", join(t))).unwrap_or_default();
                out.push(Entry {
                    source,
                    search: format!("{} {} {}", name, item["author"].as_str().unwrap_or(""), strs(&item["tags"]).join(" ")).to_lowercase(),
                    thumb_url: thumb,
                    id: id.to_string(),
                    name,
                    kind,
                    author: item["author"].as_str().unwrap_or("").to_string(),
                    description: item["description"].as_str().unwrap_or("").to_string(),
                    files: strs(&item["files"]).iter().map(|f| join(f)).collect(),
                });
            }
        }
    }
    out
}

// ---------------------------------------------------------------- thumbnails

fn thumb(agent: &ureq::Agent, key: &str, url: &str) -> Option<egui::ColorImage> {
    let path = cache_dir()?.join("thumbs").join(format!("{key}.png"));
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(_) => {
            if url.is_empty() {
                return None;
            }
            let raw = get_bytes(agent, url, &|_, _| {}).ok()?;
            // Store small: thumbnails are shown at shelf size.
            let img = image::load_from_memory(&raw).ok()?.thumbnail(256, 256).into_rgba8();
            let mut png = Vec::new();
            image::DynamicImage::ImageRgba8(img).write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).ok()?;
            let _ = write_atomic(&path, &png);
            png
        }
    };
    let img = image::load_from_memory(&bytes).ok()?.into_rgba8();
    Some(egui::ColorImage::from_rgba_unmultiplied([img.width() as usize, img.height() as usize], &img))
}

// ---------------------------------------------------------------- downloads

fn finish_material(entry: &Entry, dir: &Path, files: &HashMap<&'static str, String>, url: String, license: &str) -> Result<PathBuf, String> {
    if !files.contains_key("color") {
        return Err(format!("{} has no color map", entry.name));
    }
    // Thumbnail for the shelf, from the catalog cache.
    if let Some(cached) = cache_dir().map(|d| d.join("thumbs").join(format!("{}.png", entry.key())))
        && cached.is_file()
    {
        let _ = std::fs::copy(cached, dir.join("thumb.png"));
    }
    let meta = json!({
        "name": entry.name,
        "source": entry.source.name(),
        "id": entry.id,
        "author": entry.author,
        "url": url,
        "license": license,
        "maps": files,
    });
    write_atomic(&dir.join("meta.json"), serde_json::to_string_pretty(&meta).expect("json").as_bytes()).map_err(|e| e.to_string())?;
    Ok(dir.to_path_buf())
}

fn material_dir(entry: &Entry) -> Result<PathBuf, String> {
    let dir = library_dir().ok_or("no settings folder")?.join(entry.key());
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

fn download_polyhaven(agent: &ureq::Agent, entry: &Entry, res: Res, progress: &dyn Fn(u64, u64)) -> Result<PathBuf, String> {
    let files = get_json(agent, &format!("https://api.polyhaven.com/files/{}", entry.id))?;
    let dir = material_dir(entry)?;
    let r = res.polyhaven();
    // Prefer JPG for color data, PNG for data that banding would hurt.
    let pick = |map: &str, formats: &[&str]| -> Option<String> {
        let by_res = &files[map];
        let at = if by_res[r].is_object() { &by_res[r] } else { by_res.as_object()?.values().next()? };
        formats.iter().find_map(|f| at[*f]["url"].as_str().map(str::to_string))
    };
    let wanted: [(&'static str, &str, &[&str]); 5] = [
        ("color", "Diffuse", &["jpg", "png"]),
        ("rough", "Rough", &["jpg", "png"]),
        ("height", "Displacement", &["png", "jpg"]),
        ("normal", "nor_gl", &["jpg", "png"]),
        ("metal", "Metal", &["jpg", "png"]),
    ];
    let mut out: HashMap<&'static str, String> = HashMap::new();
    let n = wanted.len() as u64;
    for (k, (name, map, formats)) in wanted.iter().enumerate() {
        let Some(url) = pick(map, formats) else { continue };
        let ext = if url.ends_with(".png") { "png" } else { "jpg" };
        let bytes = get_bytes(agent, &url, &|d, t| progress(k as u64 * 1000 + if t > 0 { d * 1000 / t } else { 0 }, n * 1000))?;
        let file = format!("{name}.{ext}");
        write_atomic(&dir.join(&file), &bytes).map_err(|e| e.to_string())?;
        out.insert(name, file);
    }
    // No separate metal map: take metalness from the packed ARM map (blue).
    if !out.contains_key("metal") && let Some(url) = pick("arm", &["png", "jpg"]) {
        let bytes = get_bytes(agent, &url, &|_, _| {})?;
        if let Ok(arm) = image::load_from_memory(&bytes) {
            let rgb = arm.into_rgb8();
            let metal = image::GrayImage::from_fn(rgb.width(), rgb.height(), |x, y| image::Luma([rgb.get_pixel(x, y)[2]]));
            if metal.pixels().any(|p| p[0] > 12) {
                metal.save(dir.join("metal.png")).map_err(|e| e.to_string())?;
                out.insert("metal", "metal.png".into());
            }
        }
    }
    finish_material(entry, &dir, &out, format!("https://polyhaven.com/a/{}", entry.id), "CC0")
}

fn download_ambientcg(agent: &ureq::Agent, entry: &Entry, res: Res, progress: &dyn Fn(u64, u64)) -> Result<PathBuf, String> {
    let url = format!("https://ambientcg.com/get?file={}_{}-JPG.zip", entry.id, res.label());
    let zip_bytes = get_bytes(agent, &url, progress)?;
    let dir = material_dir(entry)?;
    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(zip_bytes)).map_err(|e| format!("{}: {e}", entry.name))?;
    let mut out: HashMap<&'static str, String> = HashMap::new();
    for i in 0..zip.len() {
        let mut f = zip.by_index(i).map_err(|e| e.to_string())?;
        let name = f.name().to_string();
        let lower = name.to_lowercase();
        let map = if lower.contains("_color.") {
            "color"
        } else if lower.contains("_roughness.") {
            "rough"
        } else if lower.contains("_metalness.") {
            "metal"
        } else if lower.contains("_displacement.") {
            "height"
        } else if lower.contains("_normalgl.") {
            "normal"
        } else {
            continue;
        };
        let ext = Path::new(&name).extension().and_then(|e| e.to_str()).unwrap_or("jpg").to_lowercase();
        let mut bytes = Vec::new();
        f.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
        let file = format!("{map}.{ext}");
        write_atomic(&dir.join(&file), &bytes).map_err(|e| e.to_string())?;
        out.insert(map, file);
    }
    finish_material(entry, &dir, &out, format!("https://ambientcg.com/view?id={}", entry.id), "CC0")
}

fn download_community(agent: &ureq::Agent, entry: &Entry, progress: &dyn Fn(u64, u64)) -> Result<PathBuf, String> {
    let data = data_dir().ok_or("no settings folder")?;
    let n = entry.files.len().max(1) as u64;
    let mut main = None;
    for (k, file) in entry.files.iter().enumerate() {
        let bytes = get_bytes(agent, &format!("{COMMUNITY_RAW}{file}"), &|d, t| {
            progress(k as u64 * 1000 + if t > 0 { d * 1000 / t } else { 0 }, n * 1000)
        })?;
        let name = Path::new(file).file_name().and_then(|n| n.to_str()).ok_or("bad file name")?;
        let ext = Path::new(name).extension().and_then(|e| e.to_str()).unwrap_or("");
        // Graphs (with their thumbnail) go on the Graphs shelf; nodes into the node library.
        let target = match ext {
            "mpgraph" => data.join("graphs").join(format!("{}.mpgraph", sanitize(&entry.id))),
            "mpsmart" => data.join("smart").join(format!("{}.mpsmart", entry.name.replace(['/', '\\', ':'], "_"))),
            "wgsl" => data.join("nodes").join(name),
            "png" if entry.kind == Kind::Graph => data.join("graphs").join(format!("{}.png", sanitize(&entry.id))),
            _ => continue,
        };
        write_atomic(&target, &bytes).map_err(|e| e.to_string())?;
        if ext == "mpgraph" || ext == "mpsmart" || (entry.kind == Kind::Node && ext == "wgsl") {
            main = Some(target);
        }
    }
    main.ok_or_else(|| format!("{} has no graph, smart material or node file", entry.name))
}

/// Where a downloaded community entry lives (to show it as installed).
pub fn installed_path(entry: &Entry) -> Option<PathBuf> {
    let data = data_dir()?;
    match entry.kind {
        Kind::Material => Some(library_dir()?.join(entry.key()).join("meta.json")),
        Kind::Graph => Some(data.join("graphs").join(format!("{}.mpgraph", sanitize(&entry.id)))),
        Kind::Smart => Some(data.join("smart").join(format!("{}.mpsmart", entry.name.replace(['/', '\\', ':'], "_")))),
        Kind::Node => {
            let file = entry.files.iter().find(|f| f.ends_with(".wgsl"))?;
            Some(data.join("nodes").join(Path::new(file).file_name()?))
        }
    }
}

// ---------------------------------------------------------------- sharing

/// Packs a creation for the community library: the file itself (a graph or
/// smart material), its thumbnail, the custom nodes it uses and an
/// `info.json`, in a folder and a `.zip` of it (to attach to the
/// submission). Returns (folder, zip).
pub fn package(kind: Kind, main: &Path, thumb_png: Option<Vec<u8>>, user_nodes: &[PathBuf], name: &str) -> Result<(PathBuf, PathBuf), String> {
    let id = sanitize(&name.to_lowercase().replace(' ', "-"));
    let dir = data_dir().ok_or("no settings folder")?.join("share").join(&id);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("nodes")).map_err(|e| e.to_string())?;
    let ext = main.extension().and_then(|e| e.to_str()).unwrap_or("bin");
    let mut files = vec![format!("{id}.{ext}")];
    std::fs::copy(main, dir.join(&files[0])).map_err(|e| e.to_string())?;
    let has_thumb = thumb_png.is_some_and(|png| std::fs::write(dir.join("thumb.png"), png).is_ok());
    for node in user_nodes {
        let Some(file) = node.file_name().and_then(|f| f.to_str()) else { continue };
        std::fs::copy(node, dir.join("nodes").join(file)).map_err(|e| e.to_string())?;
        files.push(format!("nodes/{file}"));
    }
    let info = json!({
        "id": id,
        "name": name,
        "kind": match kind {
            Kind::Smart => "smart",
            Kind::Node => "node",
            _ => "graph",
        },
        "author": "",
        "description": "",
        "tags": [],
        "path": format!("items/{id}"),
        "files": files,
        "thumb": if has_thumb { Value::from("thumb.png") } else { Value::Null },
        "license": "CC0",
    });
    write_atomic(&dir.join("info.json"), serde_json::to_string_pretty(&info).expect("json").as_bytes()).map_err(|e| e.to_string())?;

    let zip_path = dir.with_extension("zip");
    let mut zip = zip::ZipWriter::new(std::fs::File::create(&zip_path).map_err(|e| e.to_string())?);
    let opts = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    let mut all = files.clone();
    all.push("info.json".into());
    if has_thumb {
        all.push("thumb.png".into());
    }
    for f in &all {
        zip.start_file(format!("{id}/{f}"), opts).map_err(|e| e.to_string())?;
        zip.write_all(&std::fs::read(dir.join(f)).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    }
    zip.finish().map_err(|e| e.to_string())?;
    Ok((dir, zip_path))
}

/// The page where people submit a packed creation.
pub fn submit_url(name: &str) -> String {
    let title: String = format!("Submit: {name}").chars().map(|c| if c == ' ' { '+' } else { c }).filter(|c| c.is_ascii_alphanumeric() || "+:-_".contains(*c)).collect();
    format!("https://github.com/{COMMUNITY_REPO}/issues/new?template=submit.yml&title={title}")
}

// ---------------------------------------------------------------- downloaded materials

/// A material downloaded into the library folder.
pub struct LocalMaterial {
    pub key: String,
    pub dir: PathBuf,
    pub name: String,
    pub source: String,
    pub url: String,
    /// Image file per engine channel (color, rough, metal, height, normal).
    pub maps: [Option<PathBuf>; crate::engine::CHANNELS],
}

/// Every downloaded material, by name.
pub fn local_materials() -> Vec<LocalMaterial> {
    let Some(dir) = library_dir() else { return Vec::new() };
    let Ok(read) = std::fs::read_dir(&dir) else { return Vec::new() };
    let mut out: Vec<LocalMaterial> = read
        .flatten()
        .filter_map(|e| {
            let dir = e.path();
            let meta: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).ok()?).ok()?;
            let map = |k: &str| meta["maps"][k].as_str().map(|f| dir.join(f)).filter(|p| p.is_file());
            Some(LocalMaterial {
                key: e.file_name().to_string_lossy().into_owned(),
                name: meta["name"].as_str().unwrap_or("Material").to_string(),
                source: meta["source"].as_str().unwrap_or("").to_string(),
                url: meta["url"].as_str().unwrap_or("").to_string(),
                maps: std::array::from_fn(|ch| match ch {
                    0 => map("color"),
                    1 => map("rough"),
                    2 => map("metal"),
                    3 => map("height"),
                    crate::engine::NORMAL => map("normal"),
                    _ => None,
                }),
                dir,
            })
        })
        .collect();
    out.sort_by_key(|a| a.name.to_lowercase());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_polyhaven_and_community_catalogs() {
        let ph = json!({
            "rusty_metal_02": {"name": "Rusty Metal 02", "tags": ["rust"], "categories": ["metal"], "authors": {"Rob Tuytel": "All"}, "date_published": 2},
            "old_wood": {"name": "Old Wood", "tags": [], "categories": ["wood"], "authors": {}, "date_published": 5},
        });
        let e = parse_catalog(Source::PolyHaven, &ph);
        assert_eq!(e.len(), 2);
        assert_eq!(e[0].id, "old_wood", "newest first");
        assert!(e[1].search.contains("rust") && e[1].search.contains("metal"));
        assert_eq!(e[1].author, "Rob Tuytel");

        let community = json!({"version": 1, "items": [
            {"id": "lava-crust", "name": "Lava Crust", "kind": "graph", "author": "someone", "path": "items/lava-crust",
             "files": ["lava-crust.mpgraph", "nodes/crackle.wgsl"], "thumb": "thumb.png", "tags": ["lava"]},
            {"id": "swirl", "kind": "node", "files": ["swirl.wgsl"]},
        ]});
        let e = parse_catalog(Source::Community, &community);
        assert_eq!(e.len(), 2);
        assert_eq!(e[0].kind, Kind::Graph);
        assert_eq!(e[0].files, vec!["items/lava-crust/lava-crust.mpgraph", "items/lava-crust/nodes/crackle.wgsl"]);
        assert!(e[0].thumb_url.ends_with("items/lava-crust/thumb.png"));
        assert_eq!(e[1].kind, Kind::Node);
        assert_eq!(e[1].name, "swirl");
    }

    #[test]
    fn keys_are_safe_file_names() {
        let e = Entry {
            source: Source::AmbientCg,
            id: "../Metal 001".into(),
            name: String::new(),
            kind: Kind::Material,
            author: String::new(),
            description: String::new(),
            search: String::new(),
            thumb_url: String::new(),
            files: Vec::new(),
        };
        assert_eq!(e.key(), "ambientcg-___Metal_001");
    }

    #[test]
    fn sharing_packs_the_graph_its_thumbnail_and_custom_nodes() {
        let src = std::env::temp_dir().join("mp_share_test");
        let _ = std::fs::remove_dir_all(&src);
        std::fs::create_dir_all(&src).unwrap();
        let graph = src.join("Lava Crust.mpgraph");
        std::fs::write(&graph, "{}").unwrap();
        std::fs::write(src.join("Lava Crust.png"), [137, 80, 78, 71]).unwrap();
        let node = src.join("crackle.wgsl");
        std::fs::write(&node, "// @name Crackle").unwrap();

        let thumb = std::fs::read(src.join("Lava Crust.png")).ok();
        let (dir, zip_path) = package(Kind::Graph, &graph, thumb, &[node], "Lava Crust").unwrap();
        let info: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("info.json")).unwrap()).unwrap();
        assert_eq!(info["id"], "lava-crust");
        assert_eq!(info["files"], json!(["lava-crust.mpgraph", "nodes/crackle.wgsl"]));
        assert_eq!(info["thumb"], "thumb.png");

        let mut zip = zip::ZipArchive::new(std::fs::File::open(&zip_path).unwrap()).unwrap();
        let mut names: Vec<String> = (0..zip.len()).map(|i| zip.by_index(i).unwrap().name().to_string()).collect();
        names.sort();
        assert_eq!(
            names,
            ["lava-crust/info.json", "lava-crust/lava-crust.mpgraph", "lava-crust/nodes/crackle.wgsl", "lava-crust/thumb.png"]
        );
        // The entry the community index will hold parses back into a catalog entry.
        let mut item = info.clone();
        item["path"] = "items/lava-crust".into();
        let e = parse_catalog(Source::Community, &json!({"items": [item]}));
        assert_eq!(e[0].files, ["items/lava-crust/lava-crust.mpgraph", "items/lava-crust/nodes/crackle.wgsl"]);
        assert!(submit_url("Lava Crust").ends_with("template=submit.yml&title=Submit:+Lava+Crust"));
    }

    /// Lists both online catalogs and downloads one small material from
    /// each. Needs the internet: `cargo test library_online -- --ignored`.
    #[test]
    #[ignore = "needs the internet"]
    fn library_online() {
        let ctx = egui::Context::default();
        let net = Net::new(&ctx);
        for source in [Source::PolyHaven, Source::AmbientCg] {
            net.send(Job::Catalog(source));
            let entries = loop {
                if let Event::Catalog(s, r) = net.events.recv().unwrap() {
                    assert_eq!(s, source);
                    break r.expect("catalog");
                }
            };
            println!("  {}: {} materials", source.name(), entries.len());
            assert!(entries.len() > 100);
            let entry = entries[0].clone();
            net.send(Job::Thumb { key: entry.key(), url: entry.thumb_url.clone() });
            net.send(Job::Download { entry: entry.clone(), res: Res::K1 });
            let (mut got_thumb, mut path) = (false, None);
            while !got_thumb || path.is_none() {
                match net.events.recv().unwrap() {
                    Event::Thumb { image, .. } => got_thumb = image.is_some() || panic!("thumbnail"),
                    Event::Downloaded { result, .. } => path = Some(result.expect("download")),
                    _ => {}
                }
            }
            let local = local_materials();
            let m = local.iter().find(|m| m.key == entry.key()).expect("listed after download");
            println!("  downloaded {} -> {:?}", m.name, m.maps.iter().map(|p| p.is_some()).collect::<Vec<_>>());
            assert!(m.maps[0].is_some(), "has a color map");
        }
    }
}
