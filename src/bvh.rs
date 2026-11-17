//! Binned-SAH bounding volume hierarchy over triangles. Built on the CPU,
//! traversed on the CPU (picking) and uploaded to the GPU for baking.

use crate::mesh::Mesh;
use glam::Vec3;

#[derive(Clone, Copy)]
pub struct Node {
    pub min: Vec3,
    pub max: Vec3,
    /// Leaf: first index into `order`. Interior: index of the left child (right = left + 1).
    pub first: u32,
    /// Triangle count; zero for interior nodes.
    pub count: u32,
}

pub struct Bvh {
    pub nodes: Vec<Node>,
    /// Original triangle index for each leaf slot.
    pub order: Vec<u32>,
    /// Triangles in leaf order.
    tris: Vec<[Vec3; 3]>,
}

pub struct Hit {
    pub pos: Vec3,
    /// Index of the hit triangle in the source mesh.
    pub tri: u32,
    /// Barycentric weights of the triangle's three vertices at the hit.
    pub bary: Vec3,
}

const BINS: usize = 12;
const MAX_LEAF: u32 = 4;

/// Keeps normal BVH traversals allocation-free. The overflow vector is only
/// used if an unusually unbalanced tree exceeds the inline traversal depth.
const INLINE_TRAVERSAL_STACK: usize = 64;

struct TraversalStack {
    inline: [u32; INLINE_TRAVERSAL_STACK],
    len: usize,
    overflow: Vec<u32>,
}

impl TraversalStack {
    fn root(root: u32) -> Self {
        let mut inline = [0; INLINE_TRAVERSAL_STACK];
        inline[0] = root;
        Self { inline, len: 1, overflow: Vec::new() }
    }

    fn push(&mut self, node: u32) {
        if self.overflow.is_empty() && self.len < INLINE_TRAVERSAL_STACK {
            self.inline[self.len] = node;
            self.len += 1;
        } else {
            self.overflow.push(node);
        }
    }

    fn pop(&mut self) -> Option<u32> {
        self.overflow.pop().or_else(|| {
            if self.len == 0 {
                None
            } else {
                self.len -= 1;
                Some(self.inline[self.len])
            }
        })
    }
}

fn area(min: Vec3, max: Vec3) -> f32 {
    let e = (max - min).max(Vec3::ZERO);
    2.0 * (e.x * e.y + e.y * e.z + e.z * e.x)
}

/// Per-triangle bounds used by the BVH builder. The centroid is derived
/// from the bounds so the builder does not retain another Vec3 per triangle.
#[derive(Clone, Copy)]
struct TriangleBounds {
    min: Vec3,
    max: Vec3,
}

impl TriangleBounds {
    #[inline]
    fn centroid(self) -> Vec3 {
        (self.min + self.max) * 0.5
    }
}

impl Bvh {
    pub fn from_mesh(mesh: &Mesh) -> Bvh {
        let tris = mesh
            .indices
            .chunks_exact(3)
            .map(|t| [t[0], t[1], t[2]].map(|i| Vec3::from_array(mesh.vertices[i as usize].pos)))
            .collect();
        Bvh::build(tris)
    }

    /// Over the mesh triangles `keep` accepts (hits still report mesh triangle indices).
    pub fn from_mesh_where(mesh: &Mesh, keep: impl Fn(usize) -> bool) -> Bvh {
        let ids: Vec<u32> = (0..mesh.indices.len() / 3).filter(|&t| keep(t)).map(|t| t as u32).collect();
        let tris = ids
            .iter()
            .map(|&t| {
                let k = t as usize * 3;
                [0, 1, 2].map(|c| Vec3::from_array(mesh.vertices[mesh.indices[k + c] as usize].pos))
            })
            .collect();
        let mut bvh = Bvh::build(tris);
        for o in &mut bvh.order {
            *o = ids[*o as usize];
        }
        bvh
    }

    pub fn build(tris: Vec<[Vec3; 3]>) -> Bvh {
        let bounds: Vec<TriangleBounds> = tris
            .iter()
            .map(|t| TriangleBounds { min: t[0].min(t[1]).min(t[2]), max: t[0].max(t[1]).max(t[2]) })
            .collect();
        let (nodes, order) = Self::build_bounds(bounds);
        let ordered_tris = order.iter().map(|&i| tris[i as usize]).collect();
        Bvh { nodes, order, tris: ordered_tris }
    }

    /// Builds only the hierarchy and triangle order from indexed positions.
    /// Baking needs these for GPU packing, not a second CPU copy of every
    /// triangle in BVH order.
    pub fn build_ordered(positions: &[Vec3], indices: &[u32]) -> (Vec<Node>, Vec<u32>) {
        let n = indices.len() / 3;
        let mut bounds = Vec::with_capacity(n);
        for tri in indices.chunks_exact(3) {
            let p0 = positions[tri[0] as usize];
            let p1 = positions[tri[1] as usize];
            let p2 = positions[tri[2] as usize];
            bounds.push(TriangleBounds { min: p0.min(p1).min(p2), max: p0.max(p1).max(p2) });
        }
        Self::build_bounds(bounds)
    }

    /// Builds a traversable BVH directly from indexed positions. Only the
    /// final triangles in BVH order are materialized, not a second source
    /// triangle Vec that remains alive during hierarchy construction.
    pub fn from_indexed(positions: &[Vec3], indices: &[u32]) -> Bvh {
        let (nodes, order) = Self::build_ordered(positions, indices);
        let mut tris = Vec::with_capacity(order.len());
        for &triangle in &order {
            let base = triangle as usize * 3;
            tris.push([0, 1, 2].map(|corner| positions[indices[base + corner] as usize]));
        }
        Bvh { nodes, order, tris }
    }

    fn build_bounds(bounds: Vec<TriangleBounds>) -> (Vec<Node>, Vec<u32>) {
        let n = bounds.len();
        let mut order: Vec<u32> = (0..n as u32).collect();
        let mut nodes = Vec::with_capacity(2 * n / MAX_LEAF as usize + 1);
        nodes.push(Node { min: Vec3::ZERO, max: Vec3::ZERO, first: 0, count: n as u32 });
        let mut stack = vec![0usize];

        while let Some(ni) = stack.pop() {
            let (first, count) = (nodes[ni].first as usize, nodes[ni].count as usize);
            let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            let (mut clo, mut chi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            for &p in &order[first..first + count] {
                let bound = bounds[p as usize];
                lo = lo.min(bound.min);
                hi = hi.max(bound.max);
                let center = bound.centroid();
                clo = clo.min(center);
                chi = chi.max(center);
            }
            nodes[ni].min = lo;
            nodes[ni].max = hi;
            if count as u32 <= MAX_LEAF {
                continue;
            }

            // Binned SAH over all three axes. Centroids are derived from
            // bounds on demand instead of retaining a third Vec3 per triangle.
            let ext = chi - clo;
            let mut best: Option<(f32, usize, usize)> = None;
            for axis in 0..3 {
                if ext[axis] <= 1e-12 {
                    continue;
                }
                let scale = BINS as f32 / ext[axis];
                let mut bcount = [0u32; BINS];
                let mut bb = [(Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)); BINS];
                for &p in &order[first..first + count] {
                    let bound = bounds[p as usize];
                    let center = bound.centroid();
                    let bin = (((center[axis] - clo[axis]) * scale) as usize).min(BINS - 1);
                    bcount[bin] += 1;
                    bb[bin].0 = bb[bin].0.min(bound.min);
                    bb[bin].1 = bb[bin].1.max(bound.max);
                }
                let mut left_area = [0.0f32; BINS];
                let mut left_count = [0u32; BINS];
                let (mut l, mut h, mut c) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN), 0);
                for i in 0..BINS - 1 {
                    c += bcount[i];
                    l = l.min(bb[i].0);
                    h = h.max(bb[i].1);
                    left_count[i] = c;
                    left_area[i] = if c > 0 { area(l, h) } else { 0.0 };
                }
                let (mut l, mut h, mut c) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN), 0);
                for i in (1..BINS).rev() {
                    c += bcount[i];
                    l = l.min(bb[i].0);
                    h = h.max(bb[i].1);
                    if left_count[i - 1] == 0 || c == 0 {
                        continue;
                    }
                    let cost = left_count[i - 1] as f32 * left_area[i - 1] + c as f32 * area(l, h);
                    if best.is_none_or(|b| cost < b.0) {
                        best = Some((cost, axis, i));
                    }
                }
            }

            let leaf_cost = count as f32 * area(lo, hi);
            let mid = match best {
                Some((cost, axis, split_bin)) if cost < leaf_cost || count > 16 => {
                    let scale = BINS as f32 / ext[axis];
                    let (mut i, mut j) = (first, first + count);
                    while i < j {
                        let center = bounds[order[i] as usize].centroid();
                        let bin = (((center[axis] - clo[axis]) * scale) as usize).min(BINS - 1);
                        if bin < split_bin {
                            i += 1;
                        } else {
                            j -= 1;
                            order.swap(i, j);
                        }
                    }
                    i
                }
                // All centroids coincide; split the list so leaves stay small.
                None if count > 16 => first + count / 2,
                _ => continue,
            };

            let left = nodes.len();
            nodes.push(Node { min: lo, max: hi, first: first as u32, count: (mid - first) as u32 });
            nodes.push(Node { min: lo, max: hi, first: mid as u32, count: (first + count - mid) as u32 });
            nodes[ni].first = left as u32;
            nodes[ni].count = 0;
            stack.push(left);
            stack.push(left + 1);
        }

        (nodes, order)
    }

    fn slab(min: Vec3, max: Vec3, o: Vec3, inv: Vec3, tmax: f32) -> bool {
        let t0 = (min - o) * inv;
        let t1 = (max - o) * inv;
        let tn = t0.min(t1).max_element().max(0.0);
        let tf = t0.max(t1).min_element().min(tmax);
        tn <= tf
    }

    /// Ray distance and the (u, v) barycentrics of vertices 1 and 2.
    fn tri_hit(tri: &[Vec3; 3], o: Vec3, d: Vec3, tmax: f32) -> Option<(f32, f32, f32)> {
        let e1 = tri[1] - tri[0];
        let e2 = tri[2] - tri[0];
        let p = d.cross(e2);
        let det = e1.dot(p);
        if det.abs() < 1e-12 {
            return None;
        }
        let inv = 1.0 / det;
        let s = o - tri[0];
        // A hair of tolerance, so rays through shared edges and vertices
        // can't slip between neighboring triangles.
        const EPS: f32 = 1e-5;
        let u = s.dot(p) * inv;
        if !(-EPS..=1.0 + EPS).contains(&u) {
            return None;
        }
        let q = s.cross(e1);
        let v = d.dot(q) * inv;
        if v < -EPS || u + v > 1.0 + EPS {
            return None;
        }
        let t = e2.dot(q) * inv;
        (t > 0.0 && t < tmax).then_some((t, u, v))
    }

    /// Calls `each` with every triangle whose bounds pass `overlaps` (given
    /// min and max corners), skipping subtrees whose bounds fail it.
    pub fn query(&self, overlaps: impl Fn(Vec3, Vec3) -> bool, mut each: impl FnMut(u32)) {
        let mut stack = TraversalStack::root(0);
        while let Some(ni) = stack.pop() {
            let node = &self.nodes[ni as usize];
            if !overlaps(node.min, node.max) {
                continue;
            }
            if node.count > 0 {
                let first = node.first as usize;
                for (k, t) in self.tris[first..first + node.count as usize].iter().enumerate() {
                    if overlaps(t[0].min(t[1]).min(t[2]), t[0].max(t[1]).max(t[2])) {
                        each(self.order[first + k]);
                    }
                }
            } else {
                stack.push(node.first);
                stack.push(node.first + 1);
            }
        }
    }

    pub fn raycast(&self, o: Vec3, d: Vec3) -> Option<Hit> {
        self.raycast_filtered(o, d, None)
    }

    /// Finds the nearest hit whose original mesh-triangle index belongs to
    /// the requested range. The full BVH remains shared; other sets are
    /// skipped before their triangles can become the nearest hit.
    pub fn raycast_range(&self, o: Vec3, d: Vec3, triangles: &std::ops::Range<u32>) -> Option<Hit> {
        self.raycast_filtered(o, d, Some(triangles))
    }

    fn raycast_filtered(
        &self,
        o: Vec3,
        d: Vec3,
        triangles: Option<&std::ops::Range<u32>>,
    ) -> Option<Hit> {
        let inv = d.recip();
        let mut tmax = f32::MAX;
        let mut best = None;
        let mut stack = TraversalStack::root(0);
        while let Some(ni) = stack.pop() {
            let node = &self.nodes[ni as usize];
            if !Self::slab(node.min, node.max, o, inv, tmax) {
                continue;
            }
            if node.count > 0 {
                let first = node.first as usize;
                for (k, tri) in self.tris[first..first + node.count as usize].iter().enumerate() {
                    let mesh_triangle = self.order[first + k];
                    if triangles.is_some_and(|range| !range.contains(&mesh_triangle)) {
                        continue;
                    }
                    if let Some((t, u, v)) = Self::tri_hit(tri, o, d, tmax) {
                        tmax = t;
                        best = Some((t, first + k, u, v));
                    }
                }
            } else {
                stack.push(node.first);
                stack.push(node.first + 1);
            }
        }
        best.map(|(t, slot, u, v)| Hit { pos: o + d * t, tri: self.order[slot], bary: Vec3::new(1.0 - u - v, u, v) })
    }

    /// Every triangle the segment `o + d·t` (0 < t < `tmax`) crosses, as
    /// (t, triangle), in no particular order.
    pub fn all_hits(&self, o: Vec3, d: Vec3, tmax: f32, out: &mut Vec<(f32, u32)>) {
        out.clear();
        let inv = d.recip();
        let mut stack = TraversalStack::root(0);
        while let Some(ni) = stack.pop() {
            let node = &self.nodes[ni as usize];
            if !Self::slab(node.min, node.max, o, inv, tmax) {
                continue;
            }
            if node.count > 0 {
                let first = node.first as usize;
                for (k, tri) in self.tris[first..first + node.count as usize].iter().enumerate() {
                    if let Some((t, _, _)) = Self::tri_hit(tri, o, d, tmax) {
                        out.push((t, self.order[first + k]));
                    }
                }
            } else {
                stack.push(node.first);
                stack.push(node.first + 1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Primitive;

    #[test]
    fn traversal_stack_preserves_lifo_across_inline_overflow() {
        let mut stack = TraversalStack::root(0);
        for node in 1..(INLINE_TRAVERSAL_STACK as u32 + 20) {
            stack.push(node);
        }
        for node in (1..(INLINE_TRAVERSAL_STACK as u32 + 20)).rev() {
            assert_eq!(stack.pop(), Some(node));
        }
        assert_eq!(stack.pop(), Some(0));
        assert_eq!(stack.pop(), None);
    }

    #[test]
    fn indexed_build_matches_triangle_build_without_materializing_triangles() {
        let mesh = Primitive::Torus.build();
        let positions: Vec<Vec3> = mesh.vertices.iter().map(|v| Vec3::from_array(v.pos)).collect();
        let tris: Vec<[Vec3; 3]> = mesh.indices.chunks_exact(3)
            .map(|t| [t[0], t[1], t[2]].map(|i| positions[i as usize]))
            .collect();
        let expected = Bvh::build(tris);
        let (nodes, order) = Bvh::build_ordered(&positions, &mesh.indices);
        assert_eq!(order, expected.order);
        assert_eq!(nodes.len(), expected.nodes.len());
        for (actual, expected) in nodes.iter().zip(&expected.nodes) {
            assert_eq!(actual.min.to_array(), expected.min.to_array());
            assert_eq!(actual.max.to_array(), expected.max.to_array());
            assert_eq!(actual.first, expected.first);
            assert_eq!(actual.count, expected.count);
        }
        let indexed = Bvh::from_indexed(&positions, &mesh.indices);
        assert_eq!(indexed.order, expected.order);
        assert_eq!(indexed.tris, expected.tris);
        assert_eq!(indexed.nodes.len(), expected.nodes.len());
    }

    #[test]
    fn subset_reports_mesh_triangles() {
        let mesh = Primitive::RoundedCube.build();
        let o = Vec3::new(0.13, 0.27, 5.0);
        let d = Vec3::new(0.0, 0.0, -1.0);
        let front = Bvh::from_mesh(&mesh).raycast(o, d).expect("hits the front face").tri;
        // Without the front face's triangle, the ray finds another one, by its mesh index.
        let sub = Bvh::from_mesh_where(&mesh, |t| t as u32 != front);
        let hit = sub.raycast(o, d).expect("hits behind");
        assert_ne!(hit.tri, front);
        assert!((hit.tri as usize) < mesh.indices.len() / 3);
        assert_eq!(sub.order.len(), mesh.indices.len() / 3 - 1);
    }

    #[test]
    fn raycast_reports_triangle_and_barycentrics() {
        let mesh = Primitive::RoundedCube.build();
        let bvh = Bvh::from_mesh(&mesh);
        let hit = bvh.raycast(Vec3::new(0.13, 0.27, 5.0), Vec3::new(0.0, 0.0, -1.0)).expect("hits the front face");
        assert!((hit.bary.x + hit.bary.y + hit.bary.z - 1.0).abs() < 1e-4);
        // The barycentrics rebuild the hit point from the reported triangle.
        let corner = |k: usize| Vec3::from_array(mesh.vertices[mesh.indices[hit.tri as usize * 3 + k] as usize].pos);
        let p = corner(0) * hit.bary.x + corner(1) * hit.bary.y + corner(2) * hit.bary.z;
        assert!(p.distance(hit.pos) < 1e-4, "{p} vs {}", hit.pos);
    }
}
