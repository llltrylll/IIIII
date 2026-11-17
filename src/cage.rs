//! Bake cage: the directions bake rays search along, and the preview of the
//! projection shell drawn over the 3D view, colored by what the bake's rays
//! will find from points across the low poly.
//!
//! Probing the high poly is the slow part and runs once per scene on a worker
//! thread; recoloring for new front/rear distances or a painted cage is a
//! cheap pass over the stored hits, so the preview follows the sliders and
//! the cage brush live. The same hits tell [`CageProbe::auto_fix`] how far
//! out or in each vertex's cage must reach.

use crate::bake::{BakeSettings, CAGE_SCALE, CagePaint, MAX_CAGE_REACH, base_name, components};
use crate::bvh::Bvh;
use crate::mesh::{Mesh, unique_edges, weld_positions};
use crate::viewport::OverlayVertex;
use glam::Vec3;
use std::path::PathBuf;

/// How far the probe looks out from and into the low poly along each
/// direction: the farthest a cage reaches.
const REACH: f32 = MAX_CAGE_REACH;

/// Per mesh vertex: the direction bake rays search along (outward).
/// Averaged: the angle-weighted normals of every face at the vertex's
/// position, so split copies (hard edges, UV seams) share one direction and
/// the cage stays closed. Otherwise the vertex's own normal.
fn projection_dirs_from_data(positions: &[Vec3], vertex_normals: &[Vec3], indices: &[u32], average: bool) -> Vec<Vec3> {
    let own: Vec<Vec3> = vertex_normals.iter().map(|n| n.normalize_or_zero()).collect();
    if !average {
        return own;
    }
    let weld = weld_positions(positions);
    let mut sum = vec![Vec3::ZERO; positions.len()];
    for t in indices.chunks_exact(3) {
        let p = [0, 1, 2].map(|k| positions[t[k] as usize]);
        let n = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
        for k in 0..3 {
            // Weighted by the corner's angle, so how a face is split into
            // triangles doesn't tilt the direction.
            let (a, b) = ((p[(k + 1) % 3] - p[k]).normalize_or_zero(), (p[(k + 2) % 3] - p[k]).normalize_or_zero());
            sum[weld[t[k] as usize]] += n * a.dot(b).clamp(-1.0, 1.0).acos();
        }
    }
    (0..positions.len())
        .map(|v| {
            let d = sum[weld[v]].normalize_or_zero();
            // A degenerate spot (faces cancelling out) keeps its own normal.
            if d == Vec3::ZERO { own[v] } else { d }
        })
        .collect()
}

pub fn projection_dirs(mesh: &Mesh, average: bool) -> Vec<Vec3> {
    let positions: Vec<Vec3> = mesh.vertices.iter().map(|v| Vec3::from_array(v.pos)).collect();
    let normals: Vec<Vec3> = mesh.vertices.iter().map(|v| Vec3::from_array(v.normal)).collect();
    projection_dirs_from_data(&positions, &normals, &mesh.indices, average)
}

/// The minimum low-poly data required for a cage-probe worker. Unlike cloning
/// Mesh, this does not copy UVs, tangents, display edges, material sets or
/// import provenance that the probe never reads.
pub(crate) struct CageMesh {
    positions: Vec<Vec3>,
    normals: Vec<Vec3>,
    indices: Vec<u32>,
    bounds_min: Vec3,
    bounds_max: Vec3,
    center: Vec3,
    scale: f32,
    tri_object: Vec<u32>,
    objects: Vec<String>,
}

impl CageMesh {
    pub(crate) fn snapshot(mesh: &Mesh, include_object_mapping: bool) -> CageMesh {
        CageMesh {
            positions: mesh.vertices.iter().map(|v| Vec3::from_array(v.pos)).collect(),
            normals: mesh.vertices.iter().map(|v| Vec3::from_array(v.normal)).collect(),
            indices: mesh.indices.clone(),
            bounds_min: mesh.bounds_min,
            bounds_max: mesh.bounds_max,
            center: mesh.center,
            scale: mesh.scale,
            tri_object: if include_object_mapping { mesh.tri_object.clone() } else { Vec::new() },
            objects: if include_object_mapping { mesh.objects.clone() } else { Vec::new() },
        }
    }
}

/// Only the high-poly channels needed by the cage preview. Importer metadata,
/// normals, UVs, materials and vertex colors are dropped immediately.
struct CageHighPoly {
    positions: Vec<Vec3>,
    indices: Vec<u32>,
    tri_object: Vec<u32>,
    objects: Vec<String>,
}

impl CageHighPoly {
    fn load(path: &std::path::Path, center: Vec3, scale: f32, include_object_mapping: bool) -> Result<Self, String> {
        let mut raw = crate::import::load(path)?;
        for position in &mut raw.positions {
            *position = (*position - center) * scale;
        }
        Ok(Self {
            positions: raw.positions,
            indices: raw.indices,
            tri_object: if include_object_mapping { raw.tri_object } else { Vec::new() },
            objects: if include_object_mapping { raw.objects } else { Vec::new() },
        })
    }
}

/// What the bake ray from one low-poly vertex finds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status {
    /// Hits its own surface with room to spare.
    Ok,
    /// Hits, but the high poly is close to the cage or to the rear limit.
    Tight,
    /// Finds nothing: raise front/rear or fix the cage.
    Missed,
    /// Reaches another part's surface before its own: that part bleeds in.
    Overlap,
}

impl Status {
    pub fn color(self) -> [u8; 3] {
        match self {
            Status::Ok => [60, 220, 120],
            Status::Tight => [255, 190, 40],
            Status::Missed => [255, 60, 60],
            Status::Overlap => [230, 70, 230],
        }
    }
}

/// Inputs the probe depends on; a change means probing again.
#[derive(Clone, PartialEq, Debug)]
pub struct ProbeKey {
    pub mesh: String,
    pub vertex_count: usize,
    pub high_poly: Option<PathBuf>,
    pub cage: Option<PathBuf>,
    pub match_by_name: bool,
    pub average_normals: bool,
}

impl ProbeKey {
    pub fn new(mesh: &Mesh, settings: &BakeSettings) -> ProbeKey {
        ProbeKey {
            mesh: mesh.name.clone(),
            vertex_count: mesh.vertices.len(),
            high_poly: settings.high_poly.clone(),
            cage: settings.high_poly.as_ref().and(settings.cage.clone()),
            match_by_name: settings.match_by_name,
            average_normals: settings.average_normals,
        }
    }
}

/// The distances the bake searches with, as the preview needs them.
#[derive(Clone, PartialEq, Debug)]
pub struct Reach {
    pub front: f32,
    pub rear: f32,
    /// Per-vertex multipliers (painted or fixed), when the bake builds its cage.
    pub paint: Option<std::sync::Arc<CagePaint>>,
}

impl Reach {
    pub fn new(settings: &BakeSettings) -> Reach {
        Reach { front: settings.front, rear: settings.rear, paint: settings.cage_paint.clone() }
    }
}

struct Probe {
    /// Low-poly vertex position.
    pos: Vec3,
    /// Unit direction from the low poly toward the cage.
    dir: Vec3,
    /// Distance to the loaded cage along `dir` (replaces the front distance).
    cage_len: Option<f32>,
    /// Low-poly triangle the sample lies on, its corners (mesh vertices) and
    /// the sample's weights of them.
    tri: u32,
    corners: [u32; 3],
    bary: [f32; 3],
    /// High-poly crossings as (offset along `dir` from `pos`, part), filtered
    /// to the parts the bake lets this vertex see.
    hits: Vec<(f32, u32)>,
}

/// Probe results for one scene.
pub struct CageProbe {
    pub key: ProbeKey,
    probes: Vec<Probe>,
    indices: Vec<u32>,
    edges: Vec<u32>,
    pub has_high: bool,
    /// Per mesh vertex: its corner probe.
    vertex_probe: Vec<u32>,
    /// Per mesh vertex: the first vertex at the same position.
    weld: Vec<usize>,
}

/// Share of sample points (spread over the low poly's area) in each state.
#[derive(Clone, Copy, Default, Debug)]
pub struct CageStats {
    pub ok: f32,
    pub tight: f32,
    pub missed: f32,
    pub overlap: f32,
}

/// What [`CageProbe::auto_fix`] did.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct FixReport {
    /// Vertices whose cage moved out (to reach their surface).
    pub raised: usize,
    /// Vertices whose cage moved in (another part was reached first).
    pub lowered: usize,
    /// Vertices with no surface of their own within reach: their texels
    /// take the nearest surface instead (if the bake fills misses).
    pub unreachable: usize,
}

impl CageProbe {
    /// Probes the high poly from sample points across the low poly. Runs on
    /// a worker thread.
    pub fn build(mesh: CageMesh, key: ProbeKey) -> Result<CageProbe, String> {
        let CageMesh {
            positions,
            normals: vertex_normals,
            indices: low_indices,
            bounds_min,
            bounds_max,
            center,
            scale,
            tri_object: low_tri_object,
            objects: low_objects,
        } = mesh;
        // Reuse the compact snapshot's arrays; don't materialize a second set
        // of positions and normals after cloning the full render mesh.
        let normals = projection_dirs_from_data(
            &positions,
            &vertex_normals,
            &low_indices,
            key.average_normals && key.cage.is_none(),
        );

        // A loaded cage gives each low vertex its own ray start.
        let mut cage_pos: Option<Vec<Vec3>> = None;
        if let Some(path) = &key.cage {
            let cage = CageHighPoly::load(path, center, scale, false)?;
            if cage.indices.len() != low_indices.len() {
                return Err(format!(
                    "The cage has {} triangles but the low poly has {}; a cage must be the low poly with its vertices moved.",
                    cage.indices.len() / 3,
                    low_indices.len() / 3
                ));
            }
            let mut c = positions.clone();
            for (&l, &k) in low_indices.iter().zip(&cage.indices) {
                c[l as usize] = cage.positions[k as usize];
            }
            cage_pos = Some(c);
        }

        // Sample points: corners, plus a grid across triangles that are large
        // next to the mesh, so big faces are judged over their area rather
        // than at their corners (where rays tend to skim past edges).
        let tri_count = low_indices.len() / 3;
        let corner = |t: usize, k: usize| low_indices[t * 3 + k] as usize;
        let longest = |t: usize| {
            (0..3).map(|k| positions[corner(t, k)].distance(positions[corner(t, (k + 1) % 3)])).fold(0.0f32, f32::max)
        };
        let mut spacing = (bounds_max - bounds_min).length() / 48.0;
        let grid = |t: usize, spacing: f32| ((longest(t) / spacing).ceil() as usize).clamp(1, 12);
        // Keep the sample count bounded on huge meshes.
        while (0..tri_count).map(|t| grid(t, spacing).pow(2)).sum::<usize>() > 400_000 {
            spacing *= 1.5;
        }
        let probe_at = |t: usize, b: [f32; 3]| {
            let mix = |a: &[Vec3]| a[corner(t, 0)] * b[0] + a[corner(t, 1)] * b[1] + a[corner(t, 2)] * b[2];
            let pos = mix(&positions);
            let (dir, cage_len) = match &cage_pos {
                Some(c) => {
                    let span = mix(c) - pos;
                    let len = span.length();
                    // Where the cage touches the low poly the bake falls
                    // back to the normal and the front distance; so does the probe.
                    if len > 1e-6 { (span / len, Some(len)) } else { (mix(&normals).normalize_or_zero(), None) }
                }
                None => (mix(&normals).normalize_or_zero(), None),
            };
            let corners = [corner(t, 0), corner(t, 1), corner(t, 2)].map(|v| v as u32);
            Probe { pos, dir, cage_len, tri: t as u32, corners, bary: b, hits: Vec::new() }
        };
        let mut probes: Vec<Probe> = Vec::new();
        let mut at_vertex: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();
        let mut indices = Vec::with_capacity(low_indices.len());
        for t in 0..tri_count {
            let k = grid(t, spacing);
            // Grid point (i, j): barycentrics (1 - (i + j)/k, i/k, j/k).
            let mut ids = vec![0u32; (k + 1) * (k + 2) / 2];
            let mut n = 0;
            let mut index = vec![vec![0usize; k + 1]; k + 1];
            for i in 0..=k {
                for j in 0..=k - i {
                    index[i][j] = n;
                    n += 1;
                }
            }
            for i in 0..=k {
                for j in 0..=k - i {
                    let b = [1.0 - (i + j) as f32 / k as f32, i as f32 / k as f32, j as f32 / k as f32];
                    let vertex = match (i, j) {
                        (0, 0) => Some(corner(t, 0)),
                        (i, 0) if i == k => Some(corner(t, 1)),
                        (0, j) if j == k => Some(corner(t, 2)),
                        _ => None,
                    };
                    let id = match vertex {
                        Some(v) => *at_vertex.entry(v).or_insert_with(|| {
                            probes.push(probe_at(t, b));
                            probes.len() as u32 - 1
                        }),
                        None => {
                            probes.push(probe_at(t, b));
                            probes.len() as u32 - 1
                        }
                    };
                    ids[index[i][j]] = id;
                }
            }
            for i in 0..k {
                for j in 0..k - i {
                    indices.extend([ids[index[i][j]], ids[index[i + 1][j]], ids[index[i][j + 1]]]);
                    if i + j + 1 < k {
                        indices.extend([ids[index[i + 1][j]], ids[index[i + 1][j + 1]], ids[index[i][j + 1]]]);
                    }
                }
            }
        }
        // The wireframe follows the low poly's own edges, through its corners.
        let edges: Vec<u32> = unique_edges(&low_indices).into_iter().map(|v| at_vertex[&(v as usize)]).collect();
        let vertex_probe: Vec<u32> = (0..positions.len()).map(|v| at_vertex.get(&v).copied().unwrap_or(u32::MAX)).collect();
        let weld = weld_positions(&positions);

        let has_high = key.high_poly.is_some();
        if let Some(path) = &key.high_poly {
            let high = CageHighPoly::load(path, center, scale, key.match_by_name)?;
            // Build the hit-test BVH from indexed data. This avoids first
            // expanding every source triangle into a temporary Vec.
            let bvh = Bvh::from_indexed(&high.positions, &high.indices);
            let (tri_part, _) = components(&high.positions, &high.indices);

            // Name groups, as the bake pairs them (0 = sees / is seen by all).
            let mut low_tri_group = vec![0u32; tri_count];
            let mut high_tri_group = vec![0u32; high.indices.len() / 3];
            if key.match_by_name && !low_objects.is_empty() && !high.objects.is_empty() {
                let mut groups = std::collections::HashMap::new();
                for name in &low_objects {
                    let n = groups.len() as u32 + 1;
                    groups.entry(base_name(name)).or_insert(n);
                }
                let high_group: Vec<u32> = high.objects.iter().map(|n| groups.get(&base_name(n)).copied().unwrap_or(0)).collect();
                let matched: std::collections::HashSet<u32> = high_group.iter().copied().filter(|&g| g > 0).collect();
                if !matched.is_empty() {
                    for (g, &o) in high_tri_group.iter_mut().zip(&high.tri_object) {
                        *g = high_group[o as usize];
                    }
                    for (t, g) in low_tri_group.iter_mut().enumerate() {
                        let group = low_tri_object.get(t).map_or(0, |&o| groups[&base_name(&low_objects[o as usize])]);
                        *g = if matched.contains(&group) { group } else { 0 };
                    }
                }
            }

            use rayon::prelude::*;
            probes.par_iter_mut().for_each_init(Vec::new, |found, p| {
                let start = p.cage_len.unwrap_or(REACH) + 1e-4;
                let origin = p.pos + p.dir * start;
                bvh.all_hits(origin, -p.dir, start + REACH, found);
                let g = low_tri_group[p.tri as usize];
                p.hits = found
                    .iter()
                    .filter(|&&(_, t)| {
                        let tg = high_tri_group[t as usize];
                        g == 0 || tg == 0 || tg == g
                    })
                    .map(|&(t, tri)| (start - t, tri_part[tri as usize]))
                    .collect();
            });
        }
        Ok(CageProbe { key, probes, indices, edges, has_high, vertex_probe, weld })
    }

    /// The probe's multiplier of front and rear: the painted ones of its
    /// triangle's corners, blended (1 with a cage file, which sets its own).
    fn scale(&self, p: &Probe, paint: Option<&CagePaint>) -> f32 {
        match paint {
            Some(paint) if p.cage_len.is_none() => (0..3).map(|k| paint.at(p.corners[k] as usize) * p.bary[k]).sum(),
            _ => 1.0,
        }
    }

    fn status(&self, p: &Probe, front: f32, rear: f32) -> Status {
        let front = p.cage_len.unwrap_or(front);
        // The bake ray starts on the cage and keeps the first surface it meets.
        let first = p.hits.iter().filter(|h| h.0 <= front && h.0 >= -rear).max_by(|a, b| a.0.total_cmp(&b.0));
        let Some(&(h, part)) = first else {
            return Status::Missed;
        };
        // The surface the vertex really belongs to is the nearest one.
        let own = p.hits.iter().min_by(|a, b| a.0.abs().total_cmp(&b.0.abs())).map(|o| o.1);
        if own != Some(part) {
            Status::Overlap
        } else if h > front * 0.85 || h < -rear * 0.85 {
            Status::Tight
        } else {
            Status::Ok
        }
    }

    /// Overlay geometry for the given distances: a translucent shell and its
    /// wireframe, colored per vertex.
    pub fn overlay(&self, reach: &Reach) -> (Vec<OverlayVertex>, Vec<OverlayVertex>, CageStats) {
        let mut stats = CageStats::default();
        let paint = reach.paint.as_deref().filter(|p| p.fits(self.weld.len()));
        let scales: Vec<f32> = self.probes.iter().map(|p| self.scale(p, paint)).collect();
        let colors: Vec<[u8; 3]> = self
            .probes
            .iter()
            .zip(&scales)
            .map(|(p, &s)| {
                if !self.has_high {
                    return [90, 160, 255];
                }
                let st = self.status(p, (reach.front * s).min(REACH), (reach.rear * s).min(REACH));
                match st {
                    Status::Ok => stats.ok += 1.0,
                    Status::Tight => stats.tight += 1.0,
                    Status::Missed => stats.missed += 1.0,
                    Status::Overlap => stats.overlap += 1.0,
                }
                st.color()
            })
            .collect();
        let n = self.probes.len().max(1) as f32;
        stats = CageStats { ok: stats.ok / n, tight: stats.tight / n, missed: stats.missed / n, overlap: stats.overlap / n };
        let shell: Vec<Vec3> =
            self.probes.iter().zip(&scales).map(|(p, &s)| p.pos + p.dir * p.cage_len.unwrap_or((reach.front * s).min(REACH))).collect();
        let vertex = |i: u32, alpha: u8| {
            let c = colors[i as usize];
            OverlayVertex { pos: shell[i as usize].to_array(), color: [c[0], c[1], c[2], alpha] }
        };
        let tris = self.indices.iter().map(|&i| vertex(i, 46)).collect();
        let lines = self.edges.iter().map(|&i| vertex(i, 120)).collect();
        (tris, lines, stats)
    }

    /// Fits the cage to the high poly: per vertex, the multiplier that lets
    /// its ray reach its own surface (with a margin) without first reaching
    /// another part's. Vertices that are fine keep their multiplier; copies
    /// at one position get one value, so the cage stays closed. `None`
    /// without a high poly, or with a cage file (it sets its own shape).
    pub fn auto_fix(&self, reach: &Reach) -> Option<(Vec<f32>, FixReport)> {
        if !self.has_high || self.key.cage.is_some() {
            return None;
        }
        let n = self.weld.len();
        let (front, rear) = (reach.front.max(1e-6), reach.rear.max(1e-6));
        let (lo, hi) = (*CAGE_SCALE.start(), (*CAGE_SCALE.end()).min(REACH / front));
        let current = |v: usize| reach.paint.as_deref().filter(|p| p.fits(n)).map_or(1.0, |p| p.at(v));
        // Per welded position: the least multiplier that reaches its surface, the most that stays clear of others.
        let mut need = vec![lo; n];
        let mut allow = vec![hi; n];
        let mut found = vec![false; n];
        for v in 0..n {
            let Some(p) = self.probes.get(self.vertex_probe[v] as usize) else { continue };
            let w = self.weld[v];
            let Some(&(o, part)) = p.hits.iter().min_by(|a, b| a.0.abs().total_cmp(&b.0.abs())) else { continue };
            found[w] = true;
            // 15% margin past the surface, so the ray starts clear of it.
            let reach_out = if o > 0.0 { o * 1.15 / front } else { 0.0 };
            let reach_in = if o < 0.0 { -o * 1.15 / rear } else { 0.0 };
            need[w] = need[w].max(reach_out).max(reach_in);
            // Another part outside this surface: the cage must start below it.
            let foreign = p.hits.iter().filter(|h| h.1 != part && h.0 > o).map(|h| h.0).fold(f32::MAX, f32::min);
            if foreign < f32::MAX {
                allow[w] = allow[w].min(((o + foreign) * 0.5).max(0.0) / front);
            }
        }
        let mut out: Vec<f32> = (0..n).map(current).collect();
        let mut report = FixReport::default();
        let mut done = vec![false; n];
        for v in 0..n {
            let w = self.weld[v];
            if !found[w] {
                if !done[w] {
                    report.unreachable += 1;
                }
                done[w] = true;
                continue;
            }
            let cur = current(w);
            // A conflict (the surface lies past another part): stay clear of the other part.
            let s = if need[w] <= allow[w] { cur.clamp(need[w], allow[w]) } else { allow[w].max(lo) };
            if !done[w] {
                if s > cur * 1.001 {
                    report.raised += 1;
                } else if s < cur * 0.999 {
                    report.lowered += 1;
                }
                done[w] = true;
            }
            out[v] = s;
        }
        Some((out, report))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Primitive;

    /// The sphere `low`, scaled by `k` about its center, as an OBJ in its
    /// original units (which `HighPoly::load` normalizes like the low).
    fn scaled_obj(low: &Mesh, k: impl Fn(Vec3) -> f32) -> String {
        let mut obj = String::new();
        for v in &low.vertices {
            let p = Vec3::from_array(v.pos);
            let p = p * k(p) / low.scale + low.center;
            obj.push_str(&format!("v {} {} {}\n", p.x, p.y, p.z));
        }
        for t in low.indices.chunks_exact(3) {
            obj.push_str(&format!("f {} {} {}\n", t[0] + 1, t[1] + 1, t[2] + 1));
        }
        obj
    }

    fn probe_for(low: &Mesh, obj: String, name: &str) -> (CageProbe, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("mp_cage_test_{}_{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("high.obj");
        std::fs::write(&path, obj).unwrap();
        let key = ProbeKey {
            mesh: low.name.clone(),
            vertex_count: low.vertices.len(),
            high_poly: Some(path),
            cage: None,
            match_by_name: true,
            average_normals: true,
        };
        (CageProbe::build(CageMesh::snapshot(low, key.match_by_name), key).unwrap(), dir)
    }

    /// A sphere baked from a slightly larger sphere: covered when the front
    /// distance reaches it, missed when it doesn't.
    #[test]
    fn statuses_follow_the_distances() {
        let low = Primitive::Sphere.build();
        let (probe, dir) = probe_for(&low, scaled_obj(&low, |_| 1.05), "distances");
        // The shell sits ~0.05 outside (the sphere has radius ~1).
        let reach = |front, rear| Reach { front, rear, paint: None };
        let (_, _, near) = probe.overlay(&reach(0.02, 0.02));
        assert!(near.missed > 0.95, "{near:?}");
        let (tris, lines, far) = probe.overlay(&reach(0.2, 0.05));
        assert!(far.ok > 0.95, "{far:?}");
        assert!(tris.len() >= low.indices.len() && tris.len() % 3 == 0);
        assert!(!lines.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The high poly bulges out on one side, past a front distance that
    /// suits the rest: the fix raises the cage only there, and then nothing misses.
    #[test]
    fn auto_fix_raises_the_cage_where_rays_miss() {
        let low = Primitive::Sphere.build();
        // 3% out everywhere, 30% out where x > 0.5.
        let (probe, dir) = probe_for(&low, scaled_obj(&low, |p| if p.x > 0.5 { 1.3 } else { 1.03 }), "fix");
        let reach = Reach { front: 0.08, rear: 0.05, paint: None };
        let (_, _, before) = probe.overlay(&reach);
        assert!(before.missed > 0.05, "the bulge misses at first: {before:?}");
        let (scale, report) = probe.auto_fix(&reach).expect("fixable");
        assert!(report.raised > 0 && report.unreachable == 0, "{report:?}");
        // Raised on the bulge, left alone elsewhere.
        for (v, s) in scale.iter().enumerate() {
            let x = low.vertices[v].pos[0];
            if x < 0.3 {
                assert!((s - 1.0).abs() < 1e-4, "vertex {v} at x {x} moved to {s}");
            }
        }
        let fixed = Reach { paint: Some(std::sync::Arc::new(CagePaint::new(scale))), ..reach };
        let (_, _, after) = probe.overlay(&fixed);
        println!("  missed {:.1}% before the fix, {:.1}% after", before.missed * 100.0, after.missed * 100.0);
        assert!(after.missed < before.missed * 0.2, "{after:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A second part sits just outside the surface (a shell around it):
    /// with a generous front distance rays reach the shell first; the fix
    /// pulls the cage in between the two.
    #[test]
    fn auto_fix_pulls_the_cage_in_under_another_part() {
        let low = Primitive::Sphere.build();
        let own = scaled_obj(&low, |_| 1.02);
        let shell = scaled_obj(&low, |_| 1.07);
        // The shell's faces come after the surface's vertices.
        let n = low.vertices.len();
        let shell = shell
            .lines()
            .map(|l| match l.strip_prefix("f ") {
                Some(f) => format!("f {}
", f.split(' ').map(|i| (i.parse::<usize>().unwrap() + n).to_string()).collect::<Vec<_>>().join(" ")),
                None => format!("{l}
"),
            })
            .collect::<String>();
        let verts: String = own.lines().filter(|l| l.starts_with("v ")).chain(shell.lines().filter(|l| l.starts_with("v "))).map(|l| format!("{l}
")).collect();
        let faces: String = own.lines().filter(|l| l.starts_with("f ")).chain(shell.lines().filter(|l| l.starts_with("f "))).map(|l| format!("{l}
")).collect();
        let (probe, dir) = probe_for(&low, verts + &faces, "overlap");
        let reach = Reach { front: 0.12, rear: 0.05, paint: None };
        let (_, _, before) = probe.overlay(&reach);
        assert!(before.overlap > 0.9, "the shell is reached first: {before:?}");
        let (scale, report) = probe.auto_fix(&reach).expect("fixable");
        assert!(report.lowered > 0 && report.raised == 0, "{report:?}");
        let fixed = Reach { paint: Some(std::sync::Arc::new(CagePaint::new(scale))), ..reach };
        let (_, _, after) = probe.overlay(&fixed);
        println!("  overlapping {:.1}% before the fix, {:.1}% after", before.overlap * 100.0, after.overlap * 100.0);
        assert!(after.overlap < 0.02 && after.missed < 0.02, "{after:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Averaged directions are shared by split copies of a vertex (a cube's
    /// corners have three normals but one direction) and point outward.
    #[test]
    fn averaged_directions_close_hard_edges() {
        let cube = Primitive::RoundedCube.build();
        let dirs = projection_dirs(&cube, true);
        let pos: Vec<Vec3> = cube.vertices.iter().map(|v| Vec3::from_array(v.pos)).collect();
        let weld = weld_positions(&pos);
        for v in 0..pos.len() {
            assert!(dirs[v].distance(dirs[weld[v]]) < 1e-5, "copies of a vertex share a direction");
            assert!(dirs[v].dot(pos[v] - Vec3::from_array([0.0; 3])) > 0.0, "outward");
        }
    }
}
