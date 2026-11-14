//! Mesh data, procedural preview meshes and OBJ import.
//!
//! UVs are stored "texture-down": t = 0 is the top row of every texture, which
//! matches how GPU textures and image files are laid out. OBJ files (v-up) are
//! flipped on import.

use glam::{Mat3, Vec2, Vec3};
use std::collections::HashMap;
use std::f32::consts::{FRAC_PI_2, PI, TAU};
use std::ops::Range;

/// A texture set: the triangles of one material, which get their own layer
/// stack and textures. A set's triangles are contiguous in `Mesh::indices`,
/// and its UV edges and seams contiguous in `uv_edges` and `seam_edges`.
#[derive(Clone, Debug, PartialEq)]
pub struct MeshSet {
    pub name: String,
    /// Triangles (not indices).
    pub tris: Range<u32>,
    /// Index ranges into `uv_edges` and `seam_edges`.
    pub edges: Range<u32>,
    pub seams: Range<u32>,
}

impl MeshSet {
    /// The set's range in `Mesh::indices`.
    pub fn index_range(&self) -> Range<u32> {
        self.tris.start * 3..self.tris.end * 3
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub tangent: [f32; 4],
}

#[derive(Clone)]
pub struct Mesh {
    pub name: String,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    /// Unique UV edges as a line list, used by the 2D view wireframe.
    /// Polygon edges (index pairs): every triangle edge once, without the
    /// diagonals that split quads.
    pub uv_edges: Vec<u32>,
    /// UV seams (index pairs): edges where the two sides use different UVs,
    /// and open borders; drawn on both islands.
    pub seam_edges: Vec<u32>,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    /// Normalization applied on import: `normalized = (original - center) * scale`.
    pub center: Vec3,
    pub scale: f32,
    /// Source object per triangle and object names (empty for primitives):
    /// pairs low- and high-poly parts by name when baking.
    pub tri_object: Vec<u32>,
    pub objects: Vec<String>,
    /// Per-object normal/winding provenance captured at import time.
    pub object_ufbx_generated_normals: Vec<bool>,
    pub object_reversed_winding: Vec<bool>,
    pub object_negative_transform_determinant: Vec<bool>,
    pub object_smooth_faces: Vec<u32>,
    pub object_face_smoothing_entries: Vec<u32>,
    pub object_smooth_edges: Vec<u32>,
    pub object_edge_smoothing_entries: Vec<u32>,
    pub object_fallback_normals_generated: Vec<bool>,
    /// Vertex colors per triangle (sRGB), for color ID maps.
    pub tri_color: Option<Vec<[u8; 3]>>,
    /// Original vertex colors for each triangle corner, in triangle order.
    pub tri_vertex_color: Option<Vec<[[u8; 3]; 3]>>,
    /// Texture sets, one per material, in triangle order; never empty.
    pub sets: Vec<MeshSet>,
}

/// A detail mesh used only as a bake source; it needs no UVs.
pub struct HighPoly {
    pub name: String,
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub indices: Vec<u32>,
    /// Material index per triangle (OBJ `usemtl` groups).
    pub tri_material: Vec<u32>,
    pub material_count: u32,
    pub tri_object: Vec<u32>,
    pub objects: Vec<String>,
    pub tri_color: Option<Vec<[u8; 3]>>,
    pub tri_vertex_color: Option<Vec<[[u8; 3]; 3]>>,
}

impl HighPoly {
    /// Loads an OBJ or FBX in the low poly's original space and applies the
    /// same normalization, so both meshes line up.
    pub fn load(path: &std::path::Path, center: Vec3, scale: f32) -> Result<HighPoly, String> {
        let raw = crate::import::load(path)?;
        let positions: Vec<Vec3> = raw.positions.iter().map(|p| (*p - center) * scale).collect();
        let normals = raw.normals.unwrap_or_else(|| smooth_normals(&positions, &raw.indices));
        Ok(HighPoly {
            name: raw.name,
            positions,
            normals,
            indices: raw.indices,
            tri_material: raw.tri_material,
            material_count: raw.material_count,
            tri_object: raw.tri_object,
            objects: raw.objects,
            tri_color: raw.tri_color,
            tri_vertex_color: raw.tri_vertex_color,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Primitive {
    RoundedCube,
    Sphere,
    Torus,
    Cylinder,
    Plane,
    MaterialBall,
}

impl Primitive {
    pub const ALL: [Primitive; 6] = [
        Primitive::RoundedCube,
        Primitive::Sphere,
        Primitive::Torus,
        Primitive::Cylinder,
        Primitive::Plane,
        Primitive::MaterialBall,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Primitive::RoundedCube => "Rounded Cube",
            Primitive::Sphere => "Sphere",
            Primitive::Torus => "Torus",
            Primitive::Cylinder => "Cylinder",
            Primitive::Plane => "Plane",
            Primitive::MaterialBall => "Material Ball",
        }
    }

    pub fn build(self) -> Mesh {
        let (p, n, uv, idx) = match self {
            Primitive::RoundedCube => rounded_cube(0.2),
            Primitive::Sphere => uv_sphere(96, 48),
            Primitive::Torus => torus(0.7, 0.3, 128, 64),
            Primitive::Cylinder => cylinder(96, 24),
            Primitive::Plane => plane(32),
            Primitive::MaterialBall => material_ball(),
        };
        Mesh::from_parts(self.name().to_string(), p, Some(n), uv, idx)
    }
}

impl Mesh {
    pub fn from_parts(
        name: String,
        mut positions: Vec<Vec3>,
        normals: Option<Vec<Vec3>>,
        uvs: Vec<Vec2>,
        indices: Vec<u32>,
    ) -> Mesh {
        // Normalize into a [-1, 1] box so brush, noise and bake settings behave
        // the same on every mesh.
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for p in &positions {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        let center = (lo + hi) * 0.5;
        let scale = 2.0 / (hi - lo).max_element().max(1e-6);
        for p in &mut positions {
            *p = (*p - center) * scale;
        }
        let bounds_min = (lo - center) * scale;
        let bounds_max = (hi - center) * scale;

        let normals = normals.unwrap_or_else(|| smooth_normals(&positions, &indices));
        let (indices, source, tangents) = match mikk_tangents(&positions, &normals, &uvs, &indices) {
            Some(t) => t,
            None => {
                let t = compute_tangents(&positions, &normals, &uvs, &indices);
                (indices, (0..positions.len()).collect(), t)
            }
        };
        let vertices: Vec<Vertex> = source
            .iter()
            .zip(tangents)
            .map(|(&i, tangent)| Vertex {
                pos: positions[i].to_array(),
                normal: normals[i].normalize_or_zero().to_array(),
                uv: uvs[i].to_array(),
                tangent,
            })
            .collect();
        let mut mesh = Mesh {
            name,
            vertices,
            indices,
            uv_edges: Vec::new(),
            seam_edges: Vec::new(),
            bounds_min,
            bounds_max,
            center,
            scale,
            tri_object: Vec::new(),
            objects: Vec::new(),
            object_ufbx_generated_normals: Vec::new(),
            object_reversed_winding: Vec::new(),
            object_negative_transform_determinant: Vec::new(),
            object_smooth_faces: Vec::new(),
            object_face_smoothing_entries: Vec::new(),
            object_smooth_edges: Vec::new(),
            object_edge_smoothing_entries: Vec::new(),
            object_fallback_normals_generated: Vec::new(),
            tri_color: None,
            tri_vertex_color: None,
            sets: Vec::new(),
        };
        mesh.set_ranges(Vec::new());
        mesh
    }

    /// Rebuilds a mesh saved in a project, already normalized.
    pub fn from_saved(name: String, vertices: Vec<Vertex>, indices: Vec<u32>, center: Vec3, scale: f32) -> Mesh {
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for v in &vertices {
            lo = lo.min(Vec3::from_array(v.pos));
            hi = hi.max(Vec3::from_array(v.pos));
        }
        let mut mesh = Mesh {
            name,
            vertices,
            indices,
            uv_edges: Vec::new(),
            seam_edges: Vec::new(),
            bounds_min: lo,
            bounds_max: hi,
            center,
            scale,
            tri_object: Vec::new(),
            objects: Vec::new(),
            object_ufbx_generated_normals: Vec::new(),
            object_reversed_winding: Vec::new(),
            object_negative_transform_determinant: Vec::new(),
            object_smooth_faces: Vec::new(),
            object_face_smoothing_entries: Vec::new(),
            object_smooth_edges: Vec::new(),
            object_edge_smoothing_entries: Vec::new(),
            object_fallback_normals_generated: Vec::new(),
            tri_color: None,
            tri_vertex_color: None,
            sets: Vec::new(),
        };
        mesh.set_ranges(Vec::new());
        mesh
    }

    /// Splits the mesh into texture sets, one per material: triangles are
    /// reordered (stably) so each material's are contiguous. `tri_material`
    /// has one entry per triangle; `names` names the materials. Materials
    /// without triangles get no set.
    pub fn split_by_material(&mut self, tri_material: &[u32], names: &[String]) {
        let tris = self.triangle_count();
        if tri_material.len() != tris {
            return;
        }
        let mut order: Vec<usize> = (0..tris).collect();
        order.sort_by_key(|&t| tri_material[t]);
        let mut indices = Vec::with_capacity(self.indices.len());
        for &t in &order {
            indices.extend_from_slice(&self.indices[t * 3..t * 3 + 3]);
        }
        self.indices = indices;
        if self.tri_object.len() == tris {
            self.tri_object = order.iter().map(|&t| self.tri_object[t]).collect();
        }
        if let Some(c) = &self.tri_color {
            self.tri_color = Some(order.iter().map(|&t| c[t]).collect());
        }
        if let Some(c) = &self.tri_vertex_color {
            self.tri_vertex_color = Some(order.iter().map(|&t| c[t]).collect());
        }
        let mut ranges: Vec<(String, Range<u32>)> = Vec::new();
        for (k, &t) in order.iter().enumerate() {
            let m = tri_material[t];
            let name = names.get(m as usize).filter(|n| !n.trim().is_empty()).cloned().unwrap_or_else(|| format!("Material {}", m + 1));
            match ranges.last_mut() {
                Some((n, r)) if *n == name && r.end == k as u32 => r.end += 1,
                _ => ranges.push((name, k as u32..k as u32 + 1)),
            }
        }
        // Two materials with one name share a set only when adjacent; keep
        // names unique so sets can be told apart (and exported) by name.
        let mut seen: HashMap<String, usize> = HashMap::new();
        for (n, _) in &mut ranges {
            let count = seen.entry(n.clone()).or_insert(0);
            *count += 1;
            if *count > 1 {
                *n = format!("{n} {count}");
            }
        }
        self.set_ranges(ranges);
    }

    /// Sets the texture sets from `(name, triangle range)` pairs covering
    /// the triangles in order (one set named after the mesh when empty or
    /// invalid), and rebuilds the per-set UV edges and seams.
    pub fn set_ranges(&mut self, ranges: Vec<(String, Range<u32>)>) {
        let tris = self.triangle_count() as u32;
        let valid = !ranges.is_empty()
            && ranges[0].1.start == 0
            && ranges.last().is_some_and(|r| r.1.end == tris)
            && ranges.windows(2).all(|w| w[0].1.end == w[1].1.start)
            && ranges.iter().all(|r| r.1.start < r.1.end);
        let ranges = if valid { ranges } else { vec![(self.name.clone(), 0..tris)] };
        self.uv_edges.clear();
        self.seam_edges.clear();
        self.sets = ranges
            .into_iter()
            .map(|(name, tris)| {
                let idx = &self.indices[tris.start as usize * 3..tris.end as usize * 3];
                let e0 = self.uv_edges.len() as u32;
                self.uv_edges.extend(polygon_edges(&self.vertices, idx));
                let s0 = self.seam_edges.len() as u32;
                self.seam_edges.extend(seam_edges(&self.vertices, idx));
                MeshSet { name, tris, edges: e0..self.uv_edges.len() as u32, seams: s0..self.seam_edges.len() as u32 }
            })
            .collect();
    }

    /// The texture set a triangle belongs to.
    pub fn set_of(&self, tri: u32) -> usize {
        self.sets.iter().position(|s| s.tris.contains(&tri)).unwrap_or(0)
    }

    /// Loads an OBJ or FBX as the paintable mesh. Every object needs UVs.
    pub fn load_file(path: &std::path::Path) -> Result<Mesh, String> {
        let raw = crate::import::load(path)?;
        let Some(uvs) = raw.uvs else {
            let names = raw.missing_uvs.join(", ");
            return Err(format!("These objects have no UVs: {names}. Unwrap the mesh before importing it."));
        };
        let mut mesh = Mesh::from_parts(raw.name, raw.positions, raw.normals, uvs, raw.indices);
        // Triangle order survives tangent generation, so per-triangle data still lines up.
        mesh.tri_object = raw.tri_object;
        mesh.objects = raw.objects;
        mesh.object_ufbx_generated_normals = raw.object_ufbx_generated_normals;
        mesh.object_reversed_winding = raw.object_reversed_winding;
        mesh.object_negative_transform_determinant = raw.object_negative_transform_determinant;
        mesh.object_smooth_faces = raw.object_smooth_faces;
        mesh.object_face_smoothing_entries = raw.object_face_smoothing_entries;
        mesh.object_smooth_edges = raw.object_smooth_edges;
        mesh.object_edge_smoothing_entries = raw.object_edge_smoothing_entries;
        mesh.object_fallback_normals_generated = raw.object_fallback_normals_generated;
        mesh.tri_color = raw.tri_color;
        mesh.tri_vertex_color = raw.tri_vertex_color;
        mesh.split_by_material(&raw.tri_material, &raw.material_names);
        Ok(mesh)
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

pub fn smooth_normals(positions: &[Vec3], indices: &[u32]) -> Vec<Vec3> {
    // Accumulate over welded positions so UV seams don't show up as shading seams.
    let weld = weld_positions(positions);
    let mut acc = vec![Vec3::ZERO; positions.len()];
    for tri in indices.chunks_exact(3) {
        let [a, b, c] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        let n = (positions[b] - positions[a]).cross(positions[c] - positions[a]);
        for v in [a, b, c] {
            acc[weld[v]] += n;
        }
    }
    (0..positions.len()).map(|i| acc[weld[i]].normalize_or_zero()).collect()
}

/// Every triangle edge once, as index pairs.
pub fn unique_edges(indices: &[u32]) -> Vec<u32> {
    let mut seen = std::collections::HashSet::new();
    let mut edges = Vec::new();
    for tri in indices.chunks_exact(3) {
        for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            let key = (a.min(b), a.max(b));
            if seen.insert(key) {
                edges.push(key.0);
                edges.push(key.1);
            }
        }
    }
    edges
}

/// Triangle edges once each, leaving out quad diagonals: an edge shared by
/// two nearly coplanar triangles that is the longest edge of both.
pub fn polygon_edges(vertices: &[Vertex], indices: &[u32]) -> Vec<u32> {
    let pos = |i: u32| Vec3::from_array(vertices[i as usize].pos);
    let mut users: HashMap<(u32, u32), Vec<usize>> = HashMap::with_capacity(indices.len());
    for (t, tri) in indices.chunks_exact(3).enumerate() {
        for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            users.entry((a.min(b), a.max(b))).or_default().push(t);
        }
    }
    let longest = |t: usize| {
        let tri = &indices[t * 3..t * 3 + 3];
        [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
            .into_iter()
            .max_by(|x, y| pos(x.0).distance_squared(pos(x.1)).total_cmp(&pos(y.0).distance_squared(pos(y.1))))
            .map(|(a, b)| (a.min(b), a.max(b)))
            .expect("three edges")
    };
    let normal = |t: usize| {
        let tri = &indices[t * 3..t * 3 + 3];
        (pos(tri[1]) - pos(tri[0])).cross(pos(tri[2]) - pos(tri[0])).normalize_or_zero()
    };
    let mut edges = Vec::new();
    for tri in indices.chunks_exact(3) {
        for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            let key = (a.min(b), a.max(b));
            let Some(ts) = users.get_mut(&key) else { continue };
            let diagonal = ts.len() == 2 && longest(ts[0]) == key && longest(ts[1]) == key && normal(ts[0]).dot(normal(ts[1])) > 0.95;
            if !diagonal {
                edges.push(key.0);
                edges.push(key.1);
            }
            // Each edge once.
            users.remove(&key);
        }
    }
    edges
}

/// UV island borders: edges whose sides use different UVs (or that have one
/// side only), as index pairs of every triangle along them.
pub fn seam_edges(vertices: &[Vertex], indices: &[u32]) -> Vec<u32> {
    let positions: Vec<Vec3> = vertices.iter().map(|v| Vec3::from_array(v.pos)).collect();
    let weld = weld_positions(&positions);
    let uv_key = |i: u32| {
        let uv = vertices[i as usize].uv;
        [(uv[0] * 1e5).round() as i64, (uv[1] * 1e5).round() as i64]
    };
    // Geometric edge (welded ends, low first) → (its UVs at those ends, the index pair) per side.
    let mut sides: HashMap<(usize, usize), Vec<([[i64; 2]; 2], (u32, u32))>> = HashMap::with_capacity(indices.len());
    for tri in indices.chunks_exact(3) {
        for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            let (wa, wb) = (weld[a as usize], weld[b as usize]);
            if wa == wb {
                continue;
            }
            let (lo, hi) = if wa < wb { (a, b) } else { (b, a) };
            sides.entry((wa.min(wb), wa.max(wb))).or_default().push(([uv_key(lo), uv_key(hi)], (a, b)));
        }
    }
    let mut edges = Vec::new();
    for list in sides.values() {
        let seam = list.len() != 2 || list[0].0 != list[1].0;
        if seam {
            for (_, (a, b)) in list {
                edges.push(*a);
                edges.push(*b);
            }
        }
    }
    edges
}

/// Maps every vertex to a canonical vertex sharing (almost) the same position.
pub fn weld_positions(positions: &[Vec3]) -> Vec<usize> {
    let mut map: HashMap<[i32; 3], usize> = HashMap::with_capacity(positions.len());
    positions
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let key = (*p * 1.0e4).round().as_ivec3().to_array();
            *map.entry(key).or_insert(i)
        })
        .collect()
}

/// MikkTSpace tangents, the standard Blender, Unreal, Unity and Substance
/// use, so baked normal maps shade the same in every engine. MikkTSpace works
/// per face corner; vertices whose corners disagree (mirrored or split UVs)
/// are duplicated. Returns the new indices, the source vertex of each output
/// vertex, and the tangents (w: our handedness, see `compute_tangents`).
fn mikk_tangents(positions: &[Vec3], normals: &[Vec3], uvs: &[Vec2], indices: &[u32]) -> Option<(Vec<u32>, Vec<usize>, Vec<[f32; 4]>)> {
    struct Geo<'a> {
        positions: &'a [Vec3],
        normals: &'a [Vec3],
        uvs: &'a [Vec2],
        indices: &'a [u32],
        corners: Vec<[f32; 4]>,
    }
    impl mikktspace::Geometry for Geo<'_> {
        fn num_faces(&self) -> usize {
            self.indices.len() / 3
        }
        fn num_vertices_of_face(&self, _face: usize) -> usize {
            3
        }
        fn position(&self, face: usize, vert: usize) -> [f32; 3] {
            self.positions[self.indices[face * 3 + vert] as usize].to_array()
        }
        fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
            self.normals[self.indices[face * 3 + vert] as usize].normalize_or_zero().to_array()
        }
        fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
            // Engines run MikkTSpace with V pointing up.
            let t = self.uvs[self.indices[face * 3 + vert] as usize];
            [t.x, 1.0 - t.y]
        }
        fn set_tangent_encoded(&mut self, t: [f32; 4], face: usize, vert: usize) {
            // V up flips the handedness relative to our V-down convention.
            self.corners[face * 3 + vert] = [t[0], t[1], t[2], -t[3]];
        }
    }
    let mut geo = Geo { positions, normals, uvs, indices, corners: vec![[0.0; 4]; indices.len()] };
    if indices.is_empty() || !mikktspace::generate_tangents(&mut geo) {
        return None;
    }
    // Degenerate triangles are skipped by MikkTSpace and their corners left
    // unset; they take the simple tangent of their vertex.
    let fallback = compute_tangents(positions, normals, uvs, indices);
    for (corner, t) in geo.corners.iter_mut().enumerate() {
        if t[0] * t[0] + t[1] * t[1] + t[2] * t[2] < 0.25 {
            *t = fallback[indices[corner] as usize];
        }
    }
    let same = |a: &[f32; 4], b: &[f32; 4]| a[3] == b[3] && (0..3).all(|k| (a[k] - b[k]).abs() < 1e-4);
    let mut source: Vec<usize> = Vec::with_capacity(positions.len());
    let mut tangents: Vec<[f32; 4]> = Vec::with_capacity(positions.len());
    // Output vertices made from each input vertex so far.
    let mut made: Vec<Vec<u32>> = vec![Vec::new(); positions.len()];
    let mut out = Vec::with_capacity(indices.len());
    for (corner, &v) in indices.iter().enumerate() {
        let t = geo.corners[corner];
        let v = v as usize;
        let found = made[v].iter().copied().find(|&o| same(&tangents[o as usize], &t));
        let o = found.unwrap_or_else(|| {
            let o = source.len() as u32;
            source.push(v);
            tangents.push(t);
            made[v].push(o);
            o
        });
        out.push(o);
    }
    Some((out, source, tangents))
}

fn compute_tangents(positions: &[Vec3], normals: &[Vec3], uvs: &[Vec2], indices: &[u32]) -> Vec<[f32; 4]> {
    let mut tan = vec![Vec3::ZERO; positions.len()];
    let mut bit = vec![Vec3::ZERO; positions.len()];
    for tri in indices.chunks_exact(3) {
        let [a, b, c] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        let (e1, e2) = (positions[b] - positions[a], positions[c] - positions[a]);
        let (d1, d2) = (uvs[b] - uvs[a], uvs[c] - uvs[a]);
        let det = d1.x * d2.y - d2.x * d1.y;
        if det.abs() < 1e-12 {
            continue;
        }
        let r = 1.0 / det;
        let t = (e1 * d2.y - e2 * d1.y) * r;
        let bt = (e2 * d1.x - e1 * d2.x) * r;
        for v in [a, b, c] {
            tan[v] += t;
            bit[v] += bt;
        }
    }
    (0..positions.len())
        .map(|i| {
            let n = normals[i].normalize_or_zero();
            let mut t = tan[i] - n * n.dot(tan[i]);
            if t.length_squared() < 1e-12 {
                t = n.any_orthonormal_vector();
            }
            let t = t.normalize();
            // w encodes the handedness of the dP/dt direction (t points down the texture).
            let w = if n.cross(t).dot(bit[i]) < 0.0 { -1.0 } else { 1.0 };
            [t.x, t.y, t.z, w]
        })
        .collect()
}

type Parts = (Vec<Vec3>, Vec<Vec3>, Vec<Vec2>, Vec<u32>);

fn grid_indices(idx: &mut Vec<u32>, base: u32, cols: u32, rows: u32) {
    let stride = cols + 1;
    for y in 0..rows {
        for x in 0..cols {
            let i0 = base + y * stride + x;
            let (i1, i2, i3) = (i0 + 1, i0 + stride, i0 + stride + 1);
            idx.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);
        }
    }
}

/// A cube with rounded edges; each face gets its own UV island in a 3x2 atlas.
fn rounded_cube(radius: f32) -> Parts {
    let inner = 1.0 - radius;
    // Dense sampling in the rounded bands, sparse on the flat part.
    let mut coords = Vec::new();
    let band = 8;
    let flat = 10;
    for i in 0..band {
        coords.push(-1.0 + radius * i as f32 / band as f32);
    }
    for i in 0..flat {
        coords.push(-inner + 2.0 * inner * i as f32 / flat as f32);
    }
    for i in 0..=band {
        coords.push(inner + radius * i as f32 / band as f32);
    }
    let n = coords.len() as u32 - 1;
    // UVs follow surface arc length so flat faces and bevels get even texel density.
    let half_len = inner + radius * std::f32::consts::FRAC_PI_4;
    let arc = |x: f32| {
        let s = if x.abs() <= inner { x } else { x.signum() * (inner + radius * ((x.abs() - inner) / radius).atan()) };
        (s + half_len) / (2.0 * half_len)
    };

    // (normal, u axis, v axis) with v pointing down the texture.
    let faces = [
        (Vec3::Z, Vec3::X, -Vec3::Y),
        (Vec3::X, -Vec3::Z, -Vec3::Y),
        (-Vec3::Z, -Vec3::X, -Vec3::Y),
        (-Vec3::X, Vec3::Z, -Vec3::Y),
        (Vec3::Y, Vec3::X, Vec3::Z),
        (-Vec3::Y, Vec3::X, -Vec3::Z),
    ];
    let cell = 1.0 / 3.0;
    let pad = 0.012;
    let (mut p, mut nn, mut uv, mut idx) = (vec![], vec![], vec![], vec![]);
    for (f, (fnrm, ua, va)) in faces.iter().enumerate() {
        let base = p.len() as u32;
        let (cx, cy) = ((f % 3) as f32 * cell, (f / 3) as f32 * cell);
        for &cyv in &coords {
            for &cxv in &coords {
                let q = *fnrm + *ua * cxv + *va * cyv;
                let core = q.clamp(Vec3::splat(-inner), Vec3::splat(inner));
                let dir = (q - core).normalize_or_zero();
                p.push(core + dir * radius);
                nn.push(dir);
                let (a, b) = (arc(cxv), arc(cyv));
                uv.push(Vec2::new(
                    cx + pad + a * (cell - 2.0 * pad),
                    cy + pad + b * (cell - 2.0 * pad),
                ));
            }
        }
        grid_indices(&mut idx, base, n, n);
    }
    // Windings follow the face orientation; flip faces whose basis is left-handed.
    fix_winding(&p, &nn, &mut idx);
    (p, nn, uv, idx)
}

/// Moves a part into place and its UVs into `rect` (u, v, width, height).
fn place(parts: Parts, rot: Mat3, scale: Vec3, offset: Vec3, rect: [f32; 4]) -> Parts {
    let (p, n, uv, idx) = parts;
    let p = p.into_iter().map(|p| rot * (p * scale) + offset).collect();
    let n = n.into_iter().map(|n| (rot * (n / scale)).normalize_or_zero()).collect();
    let uv = uv.into_iter().map(|t| Vec2::new(rect[0] + t.x * rect[2], rect[1] + t.y * rect[3])).collect();
    (p, n, uv, idx)
}

fn merge(parts: impl IntoIterator<Item = Parts>) -> Parts {
    let (mut p, mut nn, mut uv, mut idx) = (vec![], vec![], vec![], vec![]);
    for (pp, pn, puv, pidx) in parts {
        let base = p.len() as u32;
        idx.extend(pidx.into_iter().map(|i| i + base));
        p.extend(pp);
        nn.extend(pn);
        uv.extend(puv);
    }
    (p, nn, uv, idx)
}

/// Preview shape for the shelf: a ball sunk into a bevelled pedestal, with a
/// belt and a cross ring half buried in it and bolts on the pedestal. It has
/// sharp and soft edges, smooth curvature, contact creases and occluded
/// pockets, so every smart mask input shows up on it.
fn material_ball() -> Parts {
    let id = Mat3::IDENTITY;
    let ball_y = 0.12;
    let mut parts = vec![
        place(uv_sphere(96, 48), id, Vec3::splat(0.62), Vec3::Y * ball_y, [0.0, 0.0, 0.62, 0.62]),
        place(rounded_cube(0.25), id, Vec3::new(0.7, 0.24, 0.7), Vec3::Y * -0.68, [0.0, 0.64, 0.54, 0.36]),
        place(torus(0.62, 0.09, 128, 24), id, Vec3::ONE, Vec3::Y * ball_y, [0.56, 0.64, 0.44, 0.10]),
        place(
            torus(0.62, 0.06, 128, 16),
            Mat3::from_rotation_x(FRAC_PI_2),
            Vec3::ONE,
            Vec3::Y * ball_y,
            [0.56, 0.76, 0.44, 0.08],
        ),
    ];
    for (k, (x, z)) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)].into_iter().enumerate() {
        let rect = [0.64 + 0.09 * k as f32, 0.0, 0.08, 0.08];
        parts.push(place(uv_sphere(24, 12), id, Vec3::splat(0.06), Vec3::new(0.5 * x, -0.44, 0.5 * z), rect));
    }
    merge(parts)
}

fn uv_sphere(seg: u32, rings: u32) -> Parts {
    let (mut p, mut nn, mut uv, mut idx) = (vec![], vec![], vec![], vec![]);
    for r in 0..=rings {
        let t = r as f32 / rings as f32;
        let phi = t * PI;
        for s in 0..=seg {
            let u = s as f32 / seg as f32;
            let th = u * TAU;
            let d = Vec3::new(phi.sin() * th.sin(), phi.cos(), phi.sin() * th.cos());
            p.push(d);
            nn.push(d);
            uv.push(Vec2::new(u, t));
        }
    }
    grid_indices(&mut idx, 0, seg, rings);
    fix_winding(&p, &nn, &mut idx);
    (p, nn, uv, idx)
}

fn torus(major: f32, minor: f32, seg: u32, sides: u32) -> Parts {
    let (mut p, mut nn, mut uv, mut idx) = (vec![], vec![], vec![], vec![]);
    for j in 0..=sides {
        let v = j as f32 / sides as f32;
        let b = v * TAU;
        for i in 0..=seg {
            let u = i as f32 / seg as f32;
            let a = u * TAU;
            let ring = Vec3::new(a.sin(), 0.0, a.cos());
            let n = ring * b.cos() + Vec3::Y * b.sin();
            p.push(ring * major + n * minor);
            nn.push(n);
            uv.push(Vec2::new(u, v));
        }
    }
    grid_indices(&mut idx, 0, seg, sides);
    fix_winding(&p, &nn, &mut idx);
    (p, nn, uv, idx)
}

fn cylinder(seg: u32, rows: u32) -> Parts {
    let (mut p, mut nn, mut uv, mut idx) = (vec![], vec![], vec![], vec![]);
    let (r, h) = (0.6, 1.0);
    // Side wall takes the top half of the texture.
    for y in 0..=rows {
        let t = y as f32 / rows as f32;
        for s in 0..=seg {
            let u = s as f32 / seg as f32;
            let a = u * TAU;
            let d = Vec3::new(a.sin(), 0.0, a.cos());
            p.push(d * r + Vec3::Y * (h - 2.0 * h * t));
            nn.push(d);
            uv.push(Vec2::new(0.01 + u * 0.98, 0.01 + t * 0.48));
        }
    }
    grid_indices(&mut idx, 0, seg, rows);
    // Caps as discs in the bottom half.
    for (cap, y, ny, cx) in [(0, h, 1.0f32, 0.25f32), (1, -h, -1.0, 0.75)] {
        let _ = cap;
        let center = p.len() as u32;
        p.push(Vec3::new(0.0, y, 0.0));
        nn.push(Vec3::Y * ny);
        uv.push(Vec2::new(cx, 0.75));
        for s in 0..=seg {
            let a = s as f32 / seg as f32 * TAU;
            p.push(Vec3::new(a.sin() * r, y, a.cos() * r));
            nn.push(Vec3::Y * ny);
            uv.push(Vec2::new(cx + a.sin() * 0.23, 0.75 + a.cos() * 0.23 * ny));
        }
        for s in 0..seg {
            idx.extend_from_slice(&[center, center + 1 + s, center + 2 + s]);
        }
    }
    fix_winding(&p, &nn, &mut idx);
    (p, nn, uv, idx)
}

fn plane(n: u32) -> Parts {
    let (mut p, mut nn, mut uv, mut idx) = (vec![], vec![], vec![], vec![]);
    for y in 0..=n {
        for x in 0..=n {
            let (u, t) = (x as f32 / n as f32, y as f32 / n as f32);
            p.push(Vec3::new(u * 2.0 - 1.0, 0.0, t * 2.0 - 1.0));
            nn.push(Vec3::Y);
            uv.push(Vec2::new(u, t));
        }
    }
    grid_indices(&mut idx, 0, n, n);
    fix_winding(&p, &nn, &mut idx);
    (p, nn, uv, idx)
}

/// Makes every triangle counter-clockwise when seen from the side its normals face.
fn fix_winding(p: &[Vec3], n: &[Vec3], idx: &mut [u32]) {
    for tri in idx.chunks_exact_mut(3) {
        let [a, b, c] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        let face = (p[b] - p[a]).cross(p[c] - p[a]);
        if face.dot(n[a] + n[b] + n[c]) < 0.0 {
            tri.swap(1, 2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MikkTSpace must keep our handedness convention (shaders read w) and
    /// agree with the simple tangents' direction where both are well defined.
    #[test]
    fn mikk_tangents_keep_the_convention() {
        for prim in [Primitive::RoundedCube, Primitive::Sphere, Primitive::Cylinder] {
            let (p, n, uv, idx) = match prim {
                Primitive::RoundedCube => rounded_cube(0.2),
                Primitive::Sphere => uv_sphere(48, 24),
                _ => cylinder(32, 8),
            };
            let simple = compute_tangents(&p, &n, &uv, &idx);
            let (new_idx, source, mikk) = mikk_tangents(&p, &n, &uv, &idx).expect("mikktspace runs");
            assert_eq!(new_idx.len(), idx.len());
            let (mut agree, mut total) = (0, 0);
            for (o, &v) in source.iter().enumerate() {
                let (a, b) = (Vec3::from_slice(&mikk[o][..3]), Vec3::from_slice(&simple[v][..3]));
                assert!((a.length() - 1.0).abs() < 1e-3, "unit tangents");
                assert!(a.dot(n[v].normalize()).abs() < 1e-2, "tangents lie in the surface");
                // Poles and seams are where the two methods may legitimately differ.
                if n[v].y.abs() < 0.9 {
                    total += 1;
                    if mikk[o][3] == simple[v][3] && a.dot(b) > 0.9 {
                        agree += 1;
                    }
                }
            }
            assert!(agree as f32 > total as f32 * 0.98, "{prim:?}: {agree}/{total} agree");
            // Corner splitting must only ever reference real vertices.
            assert!(new_idx.iter().all(|&i| (i as usize) < source.len()));
        }
    }
}

#[cfg(test)]
mod edge_tests {
    use super::*;

    #[test]
    fn quads_lose_their_diagonals_and_islands_get_seams() {
        // A plane is one quad island: 4 border edges, no diagonal, all seams (open borders).
        let plane = Primitive::Plane.build();
        let poly = polygon_edges(&plane.vertices, &plane.indices).len() / 2;
        let all = unique_edges(&plane.indices).len() / 2;
        assert!(poly < all, "diagonals removed: {poly} of {all}");
        // The rounded cube's six islands meet along seams.
        let cube = Primitive::RoundedCube.build();
        assert!(!cube.seam_edges.is_empty());
        assert!(cube.uv_edges.len() < unique_edges(&cube.indices).len());
        // Every seam edge's ends share positions with another island's edge (or a border).
        assert_eq!(cube.seam_edges.len() % 2, 0);
    }
}

#[cfg(test)]
pub(crate) mod set_tests {
    use super::*;

    /// A plane split by material: left half "Left", right half "Right".
    pub(crate) fn split_plane() -> Mesh {
        let mut m = Primitive::Plane.build();
        let side: Vec<u32> = m
            .indices
            .chunks_exact(3)
            .map(|t| {
                let u = t.iter().map(|&i| m.vertices[i as usize].uv[0]).sum::<f32>() / 3.0;
                if u < 0.5 { 0 } else { 1 }
            })
            .collect();
        m.split_by_material(&side, &["Left".into(), "Right".into()]);
        m
    }

    #[test]
    fn materials_become_contiguous_sets() {
        let plain = Primitive::Plane.build();
        let m = split_plane();
        assert_eq!(m.sets.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), ["Left", "Right"]);
        assert_eq!(m.sets[0].tris.start, 0);
        assert_eq!(m.sets[0].tris.end, m.sets[1].tris.start);
        assert_eq!(m.sets[1].tris.end as usize, m.triangle_count());
        // The same triangles, reordered: each set's lie on its side of the UV square.
        let key = |idx: &[u32]| {
            let mut t: Vec<[u32; 3]> = idx.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
            t.sort();
            t
        };
        assert_eq!(key(&plain.indices), key(&m.indices));
        for (k, s) in m.sets.iter().enumerate() {
            for t in m.indices[s.index_range().start as usize..s.index_range().end as usize].chunks_exact(3) {
                let u = t.iter().map(|&i| m.vertices[i as usize].uv[0]).sum::<f32>() / 3.0;
                assert_eq!(u >= 0.5, k == 1);
            }
            assert!(!s.edges.is_empty() && !s.seams.is_empty(), "{} has its own UV edges", s.name);
            assert_eq!(m.set_of(s.tris.start), k);
        }
        assert_eq!(m.sets[1].edges.end as usize, m.uv_edges.len());
    }

    #[test]
    fn set_names_are_unique_and_ranges_validated() {
        let mut m = Primitive::Plane.build();
        let tris = m.triangle_count();
        // Materials 0 and 2 share a name; material 1 has none.
        let mat: Vec<u32> = (0..tris).map(|t| (t % 3) as u32).collect();
        m.split_by_material(&mat, &["Body".into(), String::new(), "Body".into()]);
        let names: Vec<&str> = m.sets.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["Body", "Material 2", "Body 2"]);
        // Gaps fall back to one set.
        let n = m.triangle_count() as u32;
        m.set_ranges(vec![("A".into(), 0..2), ("B".into(), 3..n)]);
        assert_eq!(m.sets.len(), 1);
        assert_eq!(m.sets[0].tris, 0..n);
    }
}
