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
        let n = tris.len();
        let bmin: Vec<Vec3> = tris.iter().map(|t| t[0].min(t[1]).min(t[2])).collect();
        let bmax: Vec<Vec3> = tris.iter().map(|t| t[0].max(t[1]).max(t[2])).collect();
        let cent: Vec<Vec3> = (0..n).map(|i| (bmin[i] + bmax[i]) * 0.5).collect();
        let mut order: Vec<u32> = (0..n as u32).collect();
        let mut nodes = Vec::with_capacity(2 * n / MAX_LEAF as usize + 1);
        nodes.push(Node { min: Vec3::ZERO, max: Vec3::ZERO, first: 0, count: n as u32 });
        let mut stack = vec![0usize];

        while let Some(ni) = stack.pop() {
            let (first, count) = (nodes[ni].first as usize, nodes[ni].count as usize);
            let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            let (mut clo, mut chi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            for &p in &order[first..first + count] {
                let p = p as usize;
                lo = lo.min(bmin[p]);
                hi = hi.max(bmax[p]);
                clo = clo.min(cent[p]);
                chi = chi.max(cent[p]);
            }
            nodes[ni].min = lo;
            nodes[ni].max = hi;
            if count as u32 <= MAX_LEAF {
                continue;
            }

            // Binned SAH over all three axes.
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
                    let p = p as usize;
                    let b = (((cent[p][axis] - clo[axis]) * scale) as usize).min(BINS - 1);
                    bcount[b] += 1;
                    bb[b].0 = bb[b].0.min(bmin[p]);
                    bb[b].1 = bb[b].1.max(bmax[p]);
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
                        let p = order[i] as usize;
                        let b = (((cent[p][axis] - clo[axis]) * scale) as usize).min(BINS - 1);
                        if b < split_bin {
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

        let tris = order.iter().map(|&i| tris[i as usize]).collect();
        Bvh { nodes, order, tris }
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
