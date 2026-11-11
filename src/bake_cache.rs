//! Disk cache of baked mesh maps. Opening the same mesh (or a project on it)
//! with the same bake settings loads the maps instead of baking them again.
//!
//! An entry is keyed by everything a bake depends on: the mesh geometry, the
//! bake settings (with the size and date of any high-poly or cage file), the
//! bake resolution and the source of the bake code itself, so a new build
//! with a changed baker never serves stale maps. Entries are LZ4-compressed;
//! the least recently used ones are dropped past [`MAX_BYTES`].

use crate::bake::{BakeReport, BakeSettings, BakedMaps, CUSTOM_FORMAT, CUSTOM_PER_TEXTURE, MAP_FORMATS, MAP_TEXTURES, MAX_BAKE_RES, MAX_CUSTOM_MAPS};
use crate::gpu::{self, LAYER_USAGE, Tex};
use crate::mesh::Mesh;
use eframe::wgpu;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

const MAGIC: &[u8; 4] = b"MPBC";
const FORMAT_VERSION: u32 = 3;
/// Total size the cache may grow to before old entries are evicted.
const MAX_BYTES: u64 = 2 << 30;

/// Identifies one bake (see the module docs).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Key(String);

/// A cached bake: its report, the packed maps and the custom map textures,
/// as uploaded bytes.
pub struct Entry {
    pub report: BakeReport,
    maps: [(u32, Vec<u8>); MAP_TEXTURES],
    custom: Vec<(u32, Vec<u8>)>,
}

/// 64-bit FNV-1a over 8-byte words: stable across runs and builds (unlike
/// `DefaultHasher`), and fast enough to hash a multi-million-vertex mesh.
struct Hasher(u64);

impl Hasher {
    fn new() -> Self {
        Hasher(0xcbf2_9ce4_8422_2325)
    }
    fn word(&mut self, w: u64) {
        self.0 = (self.0 ^ w).wrapping_mul(0x0000_0100_0000_01b3);
    }
    fn bytes(&mut self, b: &[u8]) {
        self.word(b.len() as u64);
        let mut chunks = b.chunks_exact(8);
        for c in &mut chunks {
            self.word(u64::from_le_bytes(c.try_into().expect("8 bytes")));
        }
        let mut tail = [0u8; 8];
        tail[..chunks.remainder().len()].copy_from_slice(chunks.remainder());
        self.word(u64::from_le_bytes(tail));
    }
    fn str(&mut self, s: &str) {
        self.bytes(s.as_bytes());
    }
}

/// Hash of the code that produces a bake; changes whenever any of it does.
fn code_fingerprint() -> u64 {
    static FP: OnceLock<u64> = OnceLock::new();
    *FP.get_or_init(|| {
        let mut h = Hasher::new();
        for src in [
            include_str!("bake.rs"),
            include_str!("bvh.rs"),
            include_str!("cage.rs"),
            include_str!("mesh.rs"),
            include_str!("import.rs"),
            include_str!("shaders/bake.wgsl"),
            include_str!("shaders/bake_uv.wgsl"),
        ] {
            h.str(src);
        }
        h.0
    })
}

/// A source file as part of the key: its path, size and modification time.
fn file_stamp(h: &mut Hasher, path: &Option<PathBuf>) {
    let Some(path) = path else {
        h.word(0);
        return;
    };
    h.str(&path.to_string_lossy());
    match fs::metadata(path) {
        Ok(m) => {
            h.word(m.len());
            let modified = m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok());
            h.word(modified.map_or(0, |d| d.as_nanos() as u64));
        }
        Err(_) => h.word(u64::MAX),
    }
}

/// The key of baking texture set `set` of `mesh` with `settings` at `res`
/// (already clamped to the bake limit).
pub fn key(mesh: &Mesh, settings: &BakeSettings, res: u32, set: usize) -> Key {
    let mut h = Hasher::new();
    h.word(FORMAT_VERSION as u64);
    h.word(code_fingerprint());
    h.word(res as u64);
    // Every set's range: the ID map colors by set.
    h.word(set as u64);
    for s in &mesh.sets {
        h.word(((s.tris.start as u64) << 32) | s.tris.end as u64);
    }
    h.str(&format!("{settings:?}"));
    file_stamp(&mut h, &settings.high_poly);
    file_stamp(&mut h, &settings.cage);
    h.bytes(bytemuck::cast_slice(&mesh.vertices));
    h.bytes(bytemuck::cast_slice(&mesh.indices));
    h.bytes(bytemuck::cast_slice(&mesh.tri_object));
    for name in &mesh.objects {
        h.str(name);
    }
    match &mesh.tri_color {
        Some(c) => h.bytes(bytemuck::cast_slice(c)),
        None => h.word(0),
    }
    // The ID map from vertex colors reads the per-corner colors.
    match &mesh.tri_vertex_color {
        Some(c) => h.bytes(bytemuck::cast_slice(c)),
        None => h.word(0),
    }
    h.bytes(bytemuck::cast_slice(&mesh.center.to_array()));
    h.word(mesh.scale.to_bits() as u64);
    Key(format!("{:016x}", h.0))
}

fn dir() -> Option<PathBuf> {
    crate::crash::data_dir().map(|d| d.join("bake_cache"))
}

fn path(key: &Key) -> Option<PathBuf> {
    dir().map(|d| d.join(format!("{}.bake", key.0)))
}

/// Reads a cached bake, or `None` if there is none (or it is unreadable,
/// in which case it is removed).
pub fn load(key: &Key) -> Option<Entry> {
    let path = path(key)?;
    let metadata = fs::metadata(&path).ok()?;
    if metadata.len() > MAX_BYTES {
        let _ = fs::remove_file(&path);
        return None;
    }
    let bytes = fs::read(&path).ok()?;
    if bytes.len() as u64 > MAX_BYTES {
        let _ = fs::remove_file(&path);
        return None;
    }
    match decode(&bytes) {
        Some(entry) => {
            // Mark it recently used, for eviction.
            if let Ok(f) = fs::File::options().write(true).open(&path) {
                let _ = f.set_modified(std::time::SystemTime::now());
            }
            Some(entry)
        }
        None => {
            let _ = fs::remove_file(&path);
            None
        }
    }
}

fn decode(mut b: &[u8]) -> Option<Entry> {
    fn take<'a>(b: &mut &'a [u8], n: usize) -> Option<&'a [u8]> {
        if b.len() < n {
            return None;
        }
        let (head, rest) = b.split_at(n);
        *b = rest;
        Some(head)
    }
    fn u32_(b: &mut &[u8]) -> Option<u32> {
        Some(u32::from_le_bytes(take(b, 4)?.try_into().ok()?))
    }
    if take(&mut b, 4)? != MAGIC || u32_(&mut b)? != FORMAT_VERSION {
        return None;
    }
    let json_len = u32_(&mut b)? as usize;
    let report: serde_json::Value = serde_json::from_slice(take(&mut b, json_len)?).ok()?;
    let report = BakeReport::from_cache_json(&report)?;
    if report.resolution == 0 || report.resolution > MAX_BAKE_RES {
        return None;
    }
    let texture = |b: &mut &[u8], format: wgpu::TextureFormat| -> Option<(u32, Vec<u8>)> {
        let res = u32_(b)?;
        if res == 0 || res > MAX_BAKE_RES {
            return None;
        }
        let len = u32_(b)? as usize;
        let packed = take(b, len)?;
        let data = lz4_flex::decompress_size_prepended(packed).ok()?;
        let expected = (res as usize)
            .checked_mul(res as usize)?
            .checked_mul(gpu::bytes_per_texel(format) as usize)?;
        (data.len() == expected).then_some((res, data))
    };
    let mut maps: [(u32, Vec<u8>); MAP_TEXTURES] = Default::default();
    for (i, slot) in maps.iter_mut().enumerate() {
        *slot = texture(&mut b, MAP_FORMATS[i])?;
    }
    let resolution = maps[0].0;
    if maps.iter().any(|(res, _)| *res != resolution) {
        return None;
    }
    let count = u32_(&mut b)? as usize;
    let max_custom_textures = MAX_CUSTOM_MAPS.div_ceil(CUSTOM_PER_TEXTURE);
    if count > max_custom_textures {
        return None;
    }
    let expected_custom_textures = report.custom.len().div_ceil(CUSTOM_PER_TEXTURE);
    if count != expected_custom_textures {
        return None;
    }
    let custom = (0..count).map(|_| texture(&mut b, CUSTOM_FORMAT)).collect::<Option<Vec<_>>>()?;
    if report.resolution != resolution
        || custom.iter().any(|(res, _)| *res != resolution)
        || !b.is_empty()
    {
        return None;
    }
    Some(Entry { report, maps, custom })
}

impl Entry {
    /// Uploads the cached maps as the textures a bake would have produced.
    pub fn upload(self, device: &wgpu::Device, queue: &wgpu::Queue) -> (BakedMaps, BakeReport) {
        let usage = LAYER_USAGE | wgpu::TextureUsages::STORAGE_BINDING;
        let make = |format: wgpu::TextureFormat, (res, data): (u32, Vec<u8>)| {
            let tex = Tex::new(device, "mesh map", res, res, format, usage);
            let layout = wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(res * gpu::bytes_per_texel(format)),
                rows_per_image: Some(res),
            };
            queue.write_texture(tex.copy_info(), &data, layout, tex.extent());
            tex
        };
        let mut maps = self.maps.into_iter().enumerate().map(|(i, m)| make(MAP_FORMATS[i], m));
        let maps = std::array::from_fn(|_| maps.next().expect("every map"));
        let custom = self.custom.into_iter().map(|m| make(CUSTOM_FORMAT, m)).collect();
        (BakedMaps { maps, custom }, self.report)
    }
}

/// Saves a finished bake in the background: the maps are read back from the
/// GPU once the bake's work completes, then compressed and written. Failures
/// only cost the cache entry, so they are ignored.
pub fn store(key: Key, baked: &BakedMaps, report: Arc<BakeReport>, device: &wgpu::Device, queue: &wgpu::Queue) {
    let Some(path) = path(&key) else { return };
    let textures: Vec<wgpu::Texture> = baked.maps.iter().map(|m| m.texture.clone()).collect();
    let custom: Vec<wgpu::Texture> = baked.custom.iter().map(|m| m.texture.clone()).collect();
    let (device, queue) = (device.clone(), queue.clone());
    let spawned = std::thread::Builder::new().name("bake cache".into()).spawn(move || {
        let bytes = encode(&textures, &custom, &report, &device, &queue);
        if write_atomic(&path, &bytes).is_ok() {
            evict(MAX_BYTES);
        }
    });
    if let Err(e) = spawned {
        eprintln!("bake cache: {e}");
    }
}

/// Reads the maps back (blocking until the bake's GPU work is done) and
/// serializes them with the report.
fn encode(textures: &[wgpu::Texture], custom: &[wgpu::Texture], report: &BakeReport, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<u8> {
    // Map by map, so only one uncompressed map is in memory at a time.
    let mut body = Vec::new();
    let put = |body: &mut Vec<u8>, tex: &wgpu::Texture| {
        let raw = gpu::read_wgpu_texture(device, queue, tex);
        let packed = lz4_flex::compress_prepend_size(&raw);
        body.extend_from_slice(&tex.width().to_le_bytes());
        body.extend_from_slice(&(packed.len() as u32).to_le_bytes());
        body.extend_from_slice(&packed);
    };
    for tex in textures {
        put(&mut body, tex);
    }
    // The custom map textures, after their count.
    body.extend_from_slice(&(custom.len() as u32).to_le_bytes());
    for tex in custom {
        put(&mut body, tex);
    }
    // The readback waited for the bake to finish, so the report is final.
    let json = serde_json::to_vec(&report.cache_json()).expect("report serializes");
    let mut out = Vec::with_capacity(12 + json.len() + body.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(&json);
    out.extend_from_slice(&body);
    out
}

fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    fs::create_dir_all(path.parent().expect("cache file has a folder"))?;
    let tmp = path.with_extension("tmp");
    fs::File::create(&tmp)?.write_all(bytes)?;
    fs::rename(&tmp, path)
}

fn entries() -> Vec<(PathBuf, u64, std::time::SystemTime)> {
    let Some(dir) = dir() else { return Vec::new() };
    let Ok(read) = fs::read_dir(dir) else { return Vec::new() };
    read.flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "bake"))
        .filter_map(|e| {
            let m = e.metadata().ok()?;
            Some((e.path(), m.len(), m.modified().unwrap_or(std::time::UNIX_EPOCH)))
        })
        .collect()
}

/// Drops the least recently used entries until the cache fits in `max` bytes.
fn evict(max: u64) {
    let mut all = entries();
    all.sort_by_key(|e| std::cmp::Reverse(e.2));
    let mut total = 0;
    for (path, len, _) in all {
        total += len;
        if total > max {
            let _ = fs::remove_file(path);
        }
    }
}

/// Removes every cached bake.
pub fn clear() {
    for (path, ..) in entries() {
        let _ = fs::remove_file(path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Primitive;

    #[test]
    fn key_tracks_mesh_settings_and_resolution() {
        let ball = Primitive::MaterialBall.build();
        let s = BakeSettings::default();
        let k = key(&ball, &s, 2048, 0);
        assert_eq!(k, key(&ball, &s, 2048, 0), "the key is deterministic");
        assert_ne!(k, key(&ball, &s, 1024, 0));
        assert_ne!(k, key(&Primitive::Sphere.build(), &s, 2048, 0));
        let mut s2 = s.clone();
        s2.ao_distance += 0.01;
        assert_ne!(k, key(&ball, &s2, 2048, 0));
    }

    fn test_report_json() -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "source": "test",
            "triangles": 1,
            "parts": 1,
            "hard_edges": 0,
            "resolution": 1,
            "rays_per_texel": 1,
            "prep_seconds": 0.0,
            "edge_max_distance": 1.0,
            "id_regions": [],
            "matched_names": 0,
            "used_cage": false,
            "baked": 0,
            "custom": [],
            "gpu_seconds": 0.0,
            "valid_texels": 0,
            "missed_texels": 0,
            "filled_texels": 0
        })).unwrap()
    }

    fn test_cache_header() -> Vec<u8> {
        let report = test_report_json();
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(report.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&report);
        bytes
    }

    fn append_test_texture(bytes: &mut Vec<u8>, res: u32, format: wgpu::TextureFormat) {
        let raw = vec![0u8; (res as usize).pow(2) * gpu::bytes_per_texel(format) as usize];
        let packed = lz4_flex::compress_prepend_size(&raw);
        bytes.extend_from_slice(&res.to_le_bytes());
        bytes.extend_from_slice(&(packed.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&packed);
    }

    #[test]
    fn corrupt_entries_are_rejected() {
        assert!(decode(b"").is_none());
        assert!(decode(b"MPBC\x01\0\0\0garbage").is_none());
        let mut wrong_version = MAGIC.to_vec();
        wrong_version.extend_from_slice(&99u32.to_le_bytes());
        assert!(decode(&wrong_version).is_none());
    }

    #[test]
    fn cache_rejects_oversized_texture_resolution_before_decompression() {
        let mut bytes = test_cache_header();
        bytes.extend_from_slice(&(MAX_BAKE_RES + 1).to_le_bytes());
        assert!(decode(&bytes).is_none());
    }

    #[test]
    fn cache_rejects_too_many_custom_texture_arrays() {
        let mut bytes = test_cache_header();
        for format in MAP_FORMATS {
            append_test_texture(&mut bytes, 1, format);
        }
        let too_many = (MAX_CUSTOM_MAPS.div_ceil(CUSTOM_PER_TEXTURE) + 1) as u32;
        bytes.extend_from_slice(&too_many.to_le_bytes());
        assert!(decode(&bytes).is_none());
    }

    #[test]
    fn cache_rejects_trailing_bytes() {
        let mut bytes = test_cache_header();
        for format in MAP_FORMATS {
            append_test_texture(&mut bytes, 1, format);
        }
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(b"unexpected");
        assert!(decode(&bytes).is_none());
    }

    #[test]
    fn cache_rejects_report_and_texture_resolution_mismatch() {
        let mut bytes = test_cache_header();
        for format in MAP_FORMATS {
            append_test_texture(&mut bytes, 2, format);
        }
        bytes.extend_from_slice(&0u32.to_le_bytes());
        assert!(decode(&bytes).is_none());
    }

    #[test]
    fn cache_rejects_custom_texture_count_mismatch() {
        let mut bytes = test_cache_header();
        for format in MAP_FORMATS {
            append_test_texture(&mut bytes, 1, format);
        }
        bytes.extend_from_slice(&1u32.to_le_bytes());
        append_test_texture(&mut bytes, 1, CUSTOM_FORMAT);
        assert!(decode(&bytes).is_none());
    }

    /// A real bake survives the trip through the cache format bit for bit,
    /// report included. `cargo test bake_cache_round_trip -- --ignored`
    #[test]
    #[ignore = "needs a GPU"]
    fn bake_cache_round_trip() {
        let (device, queue) = crate::bake::tests::gpu();
        let mesh = Primitive::MaterialBall.build();
        let gpu_mesh = crate::gpu::GpuMesh::new(&device, &mesh);
        let mut settings = BakeSettings::default();
        settings.apply_quality(crate::bake::Quality::Draft);
        // Five custom maps: two custom textures.
        for kind in [crate::bake::CustomKind::Flow, crate::bake::CustomKind::Exposure, crate::bake::CustomKind::Contact, crate::bake::CustomKind::Point, crate::bake::CustomKind::Flow] {
            settings.add_custom(kind);
        }
        let (baked, report, _) = crate::bake::tests::run_bake(&device, &queue, &mesh, &gpu_mesh, &settings, 256);
        assert_eq!(baked.custom.len(), 2);

        let textures: Vec<wgpu::Texture> = baked.maps.iter().map(|m| m.texture.clone()).collect();
        let custom: Vec<wgpu::Texture> = baked.custom.iter().map(|m| m.texture.clone()).collect();
        let t = std::time::Instant::now();
        let bytes = encode(&textures, &custom, &report, &device, &queue);
        let all: Vec<&Tex> = baked.maps.iter().chain(&baked.custom).collect();
        let raw: usize = all.iter().map(|m| (m.width * m.height * gpu::bytes_per_texel(m.format)) as usize).sum();
        println!("  encoded {} KB ({} KB raw) in {:.0} ms", bytes.len() / 1024, raw / 1024, t.elapsed().as_secs_f32() * 1000.0);

        let entry = decode(&bytes).expect("decodes");
        let (restored, r) = entry.upload(&device, &queue);
        assert!(r.from_cache && r.finished());
        assert_eq!((r.resolution, r.triangles, r.parts, r.hard_edges), (report.resolution, report.triangles, report.parts, report.hard_edges));
        assert_eq!(r.id_regions.len(), report.id_regions.len());
        assert_eq!(r.edge_max_distance, report.edge_max_distance);
        assert_eq!(r.missed_percent(), report.missed_percent());
        assert_eq!(r.custom, report.custom);
        assert_eq!(restored.custom.len(), baked.custom.len());
        for (a, b) in all.iter().zip(restored.maps.iter().chain(&restored.custom)) {
            assert_eq!(a.format, b.format);
            assert!(gpu::read_texture(&device, &queue, a) == gpu::read_texture(&device, &queue, b), "map data matches");
        }
    }
}
