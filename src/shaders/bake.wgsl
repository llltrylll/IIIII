// GPU mesh-map baker. Software BVH ray tracing in compute shaders, so it runs
// on any GPU (no RT hardware required).
//
// Passes:
//   cs_surface    project each low-poly texel onto the bake source (high poly or
//                 the low poly itself): hit position/normal, tangent-space
//                 normal, height, part data, ID, missed rays.
//   cs_occlusion  (tiled) AO, cavity, bent normal, thickness, sky exposure.
//   cs_curvature  per-texel mean curvature from the projected surface.
//   cs_jfa_*      jump flooding: distance to high-curvature texels (bevels).
//   cs_edges      3D distance to the bake source's feature edges (BVH over edge
//                 segments, so it works across UV seams), merged with the JFA
//                 for high-poly bakes.
//   cs_denoise    geometry-aware a-trous filter for the ray-traced maps.
//   cs_custom     (tiled) the user's custom maps: fluid flow, directional
//                 exposure, deposits, contact and distance from a point.
//   cs_pack       pack everything into the final map textures.
//
// Binding numbers are unique across the module; each pipeline's layout only
// contains the bindings its entry point uses.

struct BakeU {
    a: vec4<f32>,   // resolution, AA grid size, cage front, cage rear
    b: vec4<f32>,   // AO rays, AO distance, cavity distance, thickness rays
    c: vec4<f32>,   // thickness distance, sky rays, cos(sky cone), curvature radius (texels)
    d: vec4<f32>,   // edge threshold, edge max distance, curvature scale, low-poly-only flag
    e: vec4<f32>,   // tile x, tile y, JFA / filter step, seed
    f: vec4<f32>,   // convex edge BVH root, concave edge BVH root (-1 = none), has cage, low-poly-only edge seed curvature
    g: vec4<f32>,   // height range (largest distance above / below the low poly a ray can find), baked-map bits, random cell size (world), projection flags (1 built cage: rear scales with it, 2 ignore backfaces, 4 fill misses from the nearest surface)
    // Custom map slot k (see `bake::CustomMap`), three vec4 each:
    //   [3k]     kind (0 none), distance, cos(spread), rays
    //   [3k + 1] unit direction the fluid or wind comes from, strength
    //   [3k + 2] point, wander (radians)
    cust: array<vec4<f32>, 12>,
};

@group(0) @binding(0) var<uniform> u: BakeU;
@group(0) @binding(1) var low_pos: texture_2d<f32>;
@group(0) @binding(2) var low_nrm: texture_2d<f32>;
@group(0) @binding(3) var low_tan: texture_2d<f32>;
@group(0) @binding(4) var<storage, read> nodes: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> tri_pos: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> tri_nrm: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> parts: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read> ids: array<vec4<f32>>;
@group(0) @binding(9) var surf_pos_w: texture_storage_2d<rgba32float, write>;
@group(0) @binding(10) var surf_nrm_w: texture_storage_2d<rgba16float, write>;
@group(0) @binding(11) var map_d_w: texture_storage_2d<rgba16float, write>;
@group(0) @binding(12) var map_e_w: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(13) var surf_pos: texture_2d<f32>;
@group(0) @binding(14) var surf_nrm: texture_2d<f32>;
@group(0) @binding(15) var occ_w: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(16) var bent_w: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(17) var curv_w: texture_storage_2d<rgba16float, write>;
@group(0) @binding(18) var curv_t: texture_2d<f32>;
@group(0) @binding(19) var jfa_w: texture_storage_2d<rgba32float, write>;
@group(0) @binding(20) var jfa_t: texture_2d<f32>;
@group(0) @binding(21) var occ_t: texture_2d<f32>;
@group(0) @binding(22) var bent_t: texture_2d<f32>;
@group(0) @binding(23) var map_a_w: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(24) var map_b_w: texture_storage_2d<rgba16float, write>;
@group(0) @binding(25) var map_c_w: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(26) var<storage, read_write> counters: array<atomic<u32>, 3>;
// Geometric (face) normal of the source triangle under each texel, and its part index.
@group(0) @binding(27) var surf_geo_w: texture_storage_2d<rgba32float, write>;
@group(0) @binding(28) var surf_geo: texture_2d<f32>;
@group(0) @binding(29) var<storage, read> seg_nodes: array<vec4<f32>>;
@group(0) @binding(30) var<storage, read> segs: array<vec4<f32>>;
@group(0) @binding(31) var edge_w: texture_storage_2d<rgba16float, write>;
@group(0) @binding(32) var edge_t: texture_2d<f32>;
// Cage position per texel (when f.z is set): rays start there.
@group(0) @binding(33) var cage_pos: texture_2d<f32>;
// A cage the bake builds itself: cage_pos holds the projection direction
// (interpolated, unnormalized) and cage_tan.x how far out rays start.
@group(0) @binding(38) var cage_tan: texture_2d<f32>;
// Per texel: the source mesh's vertex curvature interpolated at the
// projected point, and the mean edge length of the source triangle there.
@group(0) @binding(34) var surf_curv_w: texture_storage_2d<rg32float, write>;
@group(0) @binding(35) var surf_curv: texture_2d<f32>;
// The custom maps, one per channel.
@group(0) @binding(36) var cust_w: texture_storage_2d<rgba16float, write>;
@group(0) @binding(37) var cust_t: texture_2d<f32>;
// Original per-corner vertex colors (or flat ID colors) for interpolation.
@group(0) @binding(39) var<storage, read> tri_vcolor: array<vec4<f32>>;

// Match-by-name group of the texel being projected (0 = hit anything). Only
// projection rays set it; occlusion rays see the whole mesh.
var<private> match_group: u32 = 0u;

// Occlusion rays: hits closer than `fold_reach` on part `fold_part` whose
// face, seen from the ray, turns less than 45 degrees from `fold_n` are the
// neighboring facets of the surface the ray left (see `cs_occlusion`) and
// are ignored. 0 = off.
var<private> fold_reach: f32 = 0.0;
var<private> fold_n: vec3<f32> = vec3<f32>(0.0, 0.0, 1.0);
var<private> fold_part: u32 = 0xffffffffu;
const FOLD_COS: f32 = 0.7071;

// Rays ignore every triangle of this part (contact maps look for the other
// parts). NONE = off.
var<private> skip_part: u32 = 0xffffffffu;

// Projection rays skip faces they reach from behind (their face normal
// points along the ray): the inside of another part, the far side of a
// thin one.
var<private> cull_back: bool = false;

const NONE: u32 = 0xffffffffu;
const STACK: u32 = 64u;

struct Hit {
    t: f32,
    tri: u32,
    b1: f32,
    b2: f32,
};

const MISS: f32 = 3.0e38;
const BARY_EPS: f32 = 1e-5;

// Entry distance of the ray into node `ni`'s box, or MISS.
fn box_entry(ni: u32, o: vec3<f32>, inv: vec3<f32>, tmax: f32) -> f32 {
    let a = nodes[ni * 2u].xyz;
    let b = nodes[ni * 2u + 1u].xyz;
    let t0 = (a - o) * inv;
    let t1 = (b - o) * inv;
    let tmn = min(t0, t1);
    let tmx = max(t0, t1);
    let tn = max(max(tmn.x, tmn.y), max(tmn.z, 0.0));
    let tf = min(min(tmx.x, tmx.y), min(tmx.z, tmax));
    return select(MISS, tn, tn <= tf);
}

fn intersect(o: vec3<f32>, d: vec3<f32>, tmax: f32, any_hit: bool) -> Hit {
    var h: Hit;
    h.t = tmax;
    h.tri = NONE;
    let dd = select(d, vec3<f32>(1e-9), abs(d) < vec3<f32>(1e-9));
    let inv = 1.0 / dd;
    var stack: array<u32, 64>;
    var sp = 1u;
    stack[0] = 0u;
    loop {
        if (sp == 0u) {
            break;
        }
        sp -= 1u;
        let ni = stack[sp];
        let a = nodes[ni * 2u];
        let b = nodes[ni * 2u + 1u];
        let t0 = (a.xyz - o) * inv;
        let t1 = (b.xyz - o) * inv;
        let tmn = min(t0, t1);
        let tmx = max(t0, t1);
        let tn = max(max(tmn.x, tmn.y), max(tmn.z, 0.0));
        let tf = min(min(tmx.x, tmx.y), min(tmx.z, h.t));
        if (tn > tf) {
            continue;
        }
        let first = bitcast<u32>(a.w);
        let count = bitcast<u32>(b.w);
        if (count > 0u) {
            for (var i = 0u; i < count; i++) {
                let ti = first + i;
                if (match_group != 0u) {
                    let tg = bitcast<u32>(tri_pos[ti * 3u + 2u].w);
                    if (tg != 0u && tg != match_group) {
                        continue;
                    }
                }
                if (skip_part != NONE && bitcast<u32>(tri_pos[ti * 3u].w) == skip_part) {
                    continue;
                }
                let v0 = tri_pos[ti * 3u].xyz;
                let e1 = tri_pos[ti * 3u + 1u].xyz - v0;
                let e2 = tri_pos[ti * 3u + 2u].xyz - v0;
                let p = cross(d, e2);
                let det = dot(e1, p);
                if (abs(det) < 1e-14) {
                    continue;
                }
                let idet = 1.0 / det;
                let s = o - v0;
                // A hair of tolerance (as in the CPU BVH), so rays through
                // shared edges and vertices can't slip between neighboring
                // triangles: such a ray would find the surface behind, which
                // shows as speckles in the normal map and light leaks in AO.
                let bu = dot(s, p) * idet;
                if (bu < -BARY_EPS || bu > 1.0 + BARY_EPS) {
                    continue;
                }
                let q = cross(s, e1);
                let bv = dot(d, q) * idet;
                if (bv < -BARY_EPS || bu + bv > 1.0 + BARY_EPS) {
                    continue;
                }
                let t = dot(e2, q) * idet;
                if (cull_back && dot(cross(e1, e2), d) > 0.0) {
                    continue;
                }
                if (t > 0.0 && t < fold_reach && bitcast<u32>(tri_pos[ti * 3u].w) == fold_part) {
                    // The face normal on the side the ray arrives from.
                    let gn = normalize(cross(e1, e2));
                    if (dot(select(gn, -gn, dot(gn, d) > 0.0), fold_n) > FOLD_COS) {
                        continue;
                    }
                }
                if (t > 0.0 && t < h.t) {
                    h.t = t;
                    h.tri = ti;
                    h.b1 = bu;
                    h.b2 = bv;
                    if (any_hit) {
                        return h;
                    }
                }
            }
        } else if (sp + 2u <= STACK) {
            // Ordered traversal: visit the nearer child first so closest-hit
            // queries shrink h.t early and cull more of the tree.
            let tl = box_entry(first, o, inv, h.t);
            let tr = box_entry(first + 1u, o, inv, h.t);
            if (tl < MISS && tr < MISS) {
                let near = select(first + 1u, first, tl <= tr);
                let far = select(first, first + 1u, tl <= tr);
                stack[sp] = far;
                stack[sp + 1u] = near;
                sp += 2u;
            } else if (tl < MISS) {
                stack[sp] = first;
                sp += 1u;
            } else if (tr < MISS) {
                stack[sp] = first + 1u;
                sp += 1u;
            }
        }
    }
    return h;
}

// Barycentric weights (of v0, v1, v2) of the point of triangle v0 v1 v2
// nearest `p` (Ericson, Real-Time Collision Detection, 5.1.5).
fn nearest_on_tri(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        return vec3<f32>(1.0, 0.0, 0.0);
    }
    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        let v = d1 / (d1 - d3);
        return vec3<f32>(1.0 - v, v, 0.0);
    }
    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        let w = d2 / (d2 - d6);
        return vec3<f32>(1.0 - w, 0.0, w);
    }
    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return vec3<f32>(0.0, 1.0 - w, w);
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    return vec3<f32>(1.0 - v - w, v, w);
}

// The surface point nearest `p` within `radius` (of the texel's name group):
// t = its distance, tri = NONE when there's none that close.
fn nearest_surface(p: vec3<f32>, radius: f32) -> Hit {
    var h: Hit;
    h.t = radius;
    h.tri = NONE;
    var stack: array<u32, 64>;
    var sp = 1u;
    stack[0] = 0u;
    loop {
        if (sp == 0u) {
            break;
        }
        sp -= 1u;
        let ni = stack[sp];
        let a = nodes[ni * 2u];
        let b = nodes[ni * 2u + 1u];
        // Distance from p to the node's box.
        if (length(max(max(a.xyz - p, p - b.xyz), vec3<f32>(0.0))) >= h.t) {
            continue;
        }
        let first = bitcast<u32>(a.w);
        let count = bitcast<u32>(b.w);
        if (count > 0u) {
            for (var i = 0u; i < count; i++) {
                let ti = first + i;
                if (match_group != 0u) {
                    let tg = bitcast<u32>(tri_pos[ti * 3u + 2u].w);
                    if (tg != 0u && tg != match_group) {
                        continue;
                    }
                }
                let v0 = tri_pos[ti * 3u].xyz;
                let v1 = tri_pos[ti * 3u + 1u].xyz;
                let v2 = tri_pos[ti * 3u + 2u].xyz;
                let w = nearest_on_tri(p, v0, v1, v2);
                let d = distance(p, v0 * w.x + v1 * w.y + v2 * w.z);
                if (d < h.t) {
                    h.t = d;
                    h.tri = ti;
                    h.b1 = w.y;
                    h.b2 = w.z;
                }
            }
        } else if (sp + 2u <= STACK) {
            stack[sp] = first;
            stack[sp + 1u] = first + 1u;
            sp += 2u;
        }
    }
    return h;
}

fn hit_point(h: Hit) -> vec3<f32> {
    let w0 = 1.0 - h.b1 - h.b2;
    return tri_pos[h.tri * 3u].xyz * w0 + tri_pos[h.tri * 3u + 1u].xyz * h.b1 + tri_pos[h.tri * 3u + 2u].xyz * h.b2;
}

// Interpolated shading normal at a hit along ray direction `dir`. Zero or
// cancelling vertex normals (broken imports) fall back to the face normal
// facing the ray, instead of writing NaN into every map.
fn hit_normal(h: Hit, dir: vec3<f32>) -> vec3<f32> {
    let w0 = 1.0 - h.b1 - h.b2;
    let n = tri_nrm[h.tri * 3u].xyz * w0 + tri_nrm[h.tri * 3u + 1u].xyz * h.b1 + tri_nrm[h.tri * 3u + 2u].xyz * h.b2;
    if (dot(n, n) < 1e-12) {
        return hit_geo_normal(h.tri, -dir);
    }
    return normalize(n);
}

// Face normal of the hit triangle, flipped to the side of `toward`.
fn hit_geo_normal(tri: u32, toward: vec3<f32>) -> vec3<f32> {
    let v0 = tri_pos[tri * 3u].xyz;
    let g = cross(tri_pos[tri * 3u + 1u].xyz - v0, tri_pos[tri * 3u + 2u].xyz - v0);
    if (dot(g, g) < 1e-30) {
        return toward;
    }
    let gn = normalize(g);
    return select(-gn, gn, dot(gn, toward) >= 0.0);
}

fn hash_u(x_in: u32) -> u32 {
    var x = x_in;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

// The random seed for rays traced from world point `p`. It changes per world
// cell of about two texels, not per texel: the texels on both sides of a UV
// seam are the same surface point, so they pick the same random rays and
// come out the same. Per-texel seeds gave each side its own noise, which the
// per-island smoothing then left as a line along every seam.
fn world_seed(p: vec3<f32>) -> u32 {
    let q = vec3<i32>(floor(p / max(u.g.z, 1e-6)));
    return hash_u(bitcast<u32>(q.x) * 73856093u ^ bitcast<u32>(q.y) * 19349663u ^ bitcast<u32>(q.z) * 83492791u);
}

fn rand2(seed: u32) -> vec2<f32> {
    let a = hash_u(seed);
    let b = hash_u(a ^ 0x9e3779b9u);
    return vec2<f32>(f32(a & 0xffffffu), f32(b & 0xffffffu)) / 16777216.0;
}

fn radical_inverse(i_in: u32) -> f32 {
    var i = i_in;
    i = (i << 16u) | (i >> 16u);
    i = ((i & 0x55555555u) << 1u) | ((i & 0xaaaaaaaau) >> 1u);
    i = ((i & 0x33333333u) << 2u) | ((i & 0xccccccccu) >> 2u);
    i = ((i & 0x0f0f0f0fu) << 4u) | ((i & 0xf0f0f0f0u) >> 4u);
    i = ((i & 0x00ff00ffu) << 8u) | ((i & 0xff00ff00u) >> 8u);
    return f32(i) * 2.3283064e-10;
}

// Point i of an n-point Hammersley set, toroidally shifted per texel. Far
// lower variance than independent random rays for the same ray count.
fn hammersley(i: u32, n: u32, shift: vec2<f32>) -> vec2<f32> {
    return fract(vec2<f32>((f32(i) + 0.5) / f32(n), radical_inverse(i)) + shift);
}

// Keeps a ray on the outside of the true (faceted) surface: rays sampled
// around an interpolated normal can point below the triangle they start on.
fn above(d: vec3<f32>, g: vec3<f32>) -> vec3<f32> {
    let k = dot(d, g);
    return select(d, normalize(d - 2.0 * k * g), k < 1e-3);
}

fn onb(n: vec3<f32>) -> mat3x3<f32> {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    let t = vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    let bt = vec3<f32>(b, s + n.y * n.y * a, -n.y);
    return mat3x3<f32>(t, bt, n);
}

fn cosine_dir(n: vec3<f32>, r: vec2<f32>) -> vec3<f32> {
    let rad = sqrt(r.x);
    let phi = 6.2831853 * r.y;
    let l = vec3<f32>(rad * cos(phi), rad * sin(phi), sqrt(max(1.0 - r.x, 0.0)));
    return normalize(onb(n) * l);
}

fn cone_dir(axis: vec3<f32>, cos_max: f32, r: vec2<f32>) -> vec3<f32> {
    let ct = 1.0 - r.x * (1.0 - cos_max);
    let st = sqrt(max(1.0 - ct * ct, 0.0));
    let phi = 6.2831853 * r.y;
    return normalize(onb(axis) * vec3<f32>(st * cos(phi), st * sin(phi), ct));
}

fn res() -> i32 {
    return i32(u.a.x);
}

fn in_bounds(c: vec2<i32>) -> bool {
    return all(c >= vec2<i32>(0)) && all(c < vec2<i32>(res()));
}

// Texel footprint on the low-poly surface, for anti-aliasing jitter.
// Islands can touch in UV space, so a texel's UV neighbor may lie anywhere on
// the mesh. When the two one-sided differences disagree, the shorter one is
// the true surface neighbor. Padding copies its neighbor's position, so a
// zero difference carries no information either.
fn footprint(c: vec2<i32>, dir: vec2<i32>, p: vec3<f32>) -> vec3<f32> {
    let r = res() - 1;
    let q1 = textureLoad(low_pos, clamp(c + dir, vec2<i32>(0), vec2<i32>(r)), 0);
    let q0 = textureLoad(low_pos, clamp(c - dir, vec2<i32>(0), vec2<i32>(r)), 0);
    let d1 = q1.xyz - p;
    let d0 = p - q0.xyz;
    let l1 = length(d1);
    let l0 = length(d0);
    let ok1 = q1.w > 0.25 && l1 > 1e-7;
    let ok0 = q0.w > 0.25 && l0 > 1e-7;
    if (ok1 && ok0) {
        if (l1 < 2.0 * l0 && l0 < 2.0 * l1) {
            return (d1 + d0) * 0.5;
        }
        return select(d0, d1, l1 < l0);
    }
    if (ok1) {
        return d1;
    }
    if (ok0) {
        return d0;
    }
    return vec3<f32>(0.0);
}

// World size of one texel at `c`.
fn texel_size(c: vec2<i32>, p: vec3<f32>) -> f32 {
    let a = length(footprint(c, vec2<i32>(1, 0), p));
    let b = length(footprint(c, vec2<i32>(0, 1), p));
    return max(max(a, b), 1e-6);
}

// Orthonormal frame (rays, sampling) plus the raw interpolated normal and
// tangent the tangent-space encoding uses.
struct LowFrame {
    n: vec3<f32>,
    t: vec3<f32>,
    b: vec3<f32>,
    rn: vec3<f32>,
    rt: vec3<f32>,
};

struct LowTap {
    n: vec3<f32>,
    t: vec3<f32>,
};

// A neighbor's low-poly normal/tangent, or the center's when the neighbor is
// empty, padding, across a hard edge, or from another island touching in UV.
fn low_tap(q: vec2<i32>, center_inside: bool, center: LowTap, p: vec3<f32>, texel: f32) -> LowTap {
    if (!in_bounds(q)) {
        return center;
    }
    let qp = textureLoad(low_pos, q, 0);
    if (qp.w < 0.25 || (center_inside && qp.w < 0.75) || distance(qp.xyz, p) > texel * 3.0) {
        return center;
    }
    let n = textureLoad(low_nrm, q, 0).xyz;
    if (dot(normalize(n), normalize(center.n)) < 0.7) {
        return center;
    }
    return LowTap(n, textureLoad(low_tan, q, 0).xyz);
}

// Low-poly tangent frame at a sub-texel offset `j` (texels, -0.5..0.5),
// bilinearly interpolated like the rasterizer would.
fn low_frame(c: vec2<i32>, j: vec2<f32>, center_inside: bool, center: LowTap, sign: f32, p: vec3<f32>, texel: f32) -> LowFrame {
    let sx = select(-1, 1, j.x >= 0.0);
    let sy = select(-1, 1, j.y >= 0.0);
    let ax = abs(j.x);
    let ay = abs(j.y);
    let b = low_tap(c + vec2<i32>(sx, 0), center_inside, center, p, texel);
    let cc = low_tap(c + vec2<i32>(0, sy), center_inside, center, p, texel);
    let d = low_tap(c + vec2<i32>(sx, sy), center_inside, center, p, texel);
    let rn = mix(mix(center.n, b.n, ax), mix(cc.n, d.n, ax), ay);
    let n = normalize(rn);
    let tr = mix(mix(center.t, b.t, ax), mix(cc.t, d.t, ax), ay);
    let t = normalize(tr - n * dot(n, tr));
    return LowFrame(n, t, cross(n, t) * sign, rn, tr);
}

// World normal `hn` in tangent space, the exact inverse of MikkTSpace
// decoding (`fs_mesh`): normalize(T x + B y + N z) = hn with the raw
// interpolated T and N and B = -sign (N x T), so x, y, z come from the
// inverse of the (generally non-orthogonal) matrix [T B N]. G is +V up
// (OpenGL): the tangent's bitangent points down the texture (+V down, see
// `compute_tangents`), hence the minus. Falls back to the orthonormal frame
// where the raw one degenerates.
fn encode_tangent(hn: vec3<f32>, fr: LowFrame, sign: f32) -> vec3<f32> {
    let b = -cross(fr.rn, fr.rt) * sign;
    let bn = cross(b, fr.rn);
    let det = dot(fr.rt, bn);
    if (abs(det) < 1e-8) {
        return vec3<f32>(dot(hn, fr.t), -dot(hn, fr.b), dot(hn, fr.n));
    }
    return normalize(vec3<f32>(dot(hn, bn), dot(hn, cross(fr.rn, fr.rt)), dot(hn, cross(fr.rt, b))) / det);
}

// ------------------------------------------------------------------ surface

// Point `p` on triangle `ti` (it lies on it up to float error): barycentrics
// of its projection, clamped into the triangle.
fn surface_hit(ti: u32, p: vec3<f32>) -> Hit {
    let v0 = tri_pos[ti * 3u].xyz;
    let e1 = tri_pos[ti * 3u + 1u].xyz - v0;
    let e2 = tri_pos[ti * 3u + 2u].xyz - v0;
    let s = p - v0;
    let d00 = dot(e1, e1);
    let d01 = dot(e1, e2);
    let d11 = dot(e2, e2);
    let d20 = dot(s, e1);
    let d21 = dot(s, e2);
    let den = d00 * d11 - d01 * d01;
    var b = vec2<f32>(1.0 / 3.0);
    if (abs(den) > 1e-20) {
        b = max(vec2<f32>(d11 * d20 - d01 * d21, d00 * d21 - d01 * d20) / den, vec2<f32>(0.0));
        let sum = b.x + b.y;
        if (sum > 1.0) {
            b /= sum;
        }
    }
    var h: Hit;
    h.t = 0.0;
    h.tri = ti;
    h.b1 = b.x;
    h.b2 = b.y;
    return h;
}

@compute @workgroup_size(8, 8)
fn cs_surface(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = vec2<i32>(gid.xy);
    if (!in_bounds(c)) {
        return;
    }
    let lp = textureLoad(low_pos, c, 0);
    if (lp.w < 0.25) {
        textureStore(surf_pos_w, c, vec4<f32>(0.0, 0.0, 0.0, -1.0));
        textureStore(surf_nrm_w, c, vec4<f32>(0.0, 1.0, 0.0, 0.5));
        textureStore(map_d_w, c, vec4<f32>(0.0, 0.0, 0.5, 0.0));
        textureStore(map_e_w, c, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        textureStore(surf_geo_w, c, vec4<f32>(0.0, 1.0, 0.0, -1.0));
        textureStore(surf_curv_w, c, vec4<f32>(0.0));
        return;
    }
    // Only texels inside UV islands count toward the report; padding misses are harmless.
    let inside = lp.w > 0.75;
    if (inside) {
        atomicAdd(&counters[0], 1u);
    }
    let n = normalize(textureLoad(low_nrm, c, 0).xyz);
    let tg = textureLoad(low_tan, c, 0);
    let sign = select(1.0, -1.0, tg.w < 0.0);
    let center = LowTap(textureLoad(low_nrm, c, 0).xyz, tg.xyz);
    let dpx = footprint(c, vec2<i32>(1, 0), lp.xyz);
    let dpy = footprint(c, vec2<i32>(0, 1), lp.xyz);
    let low_only = u.d.w > 0.5;
    let texel = texel_size(c, lp.xyz);
    match_group = select(u32(textureLoad(low_nrm, c, 0).w + 0.5), 0u, low_only);
    // Low-poly-only bakes rasterize each texel's own triangle (BVH slot).
    let own_tri = select(NONE, u32(abs(tg.w) + 0.5) - 2u, low_only && abs(tg.w) > 1.5);
    let use_cage = u.f.z > 0.5 && !low_only;
    let cage_center = textureLoad(cage_pos, c, 0).xyz;
    let flags = u32(u.g.w + 0.5);
    let built_cage = (flags & 1u) != 0u;

    // A low-poly-only bake projects the low poly onto itself: its normal map
    // is flat by definition and one sample is enough. The texel's triangle
    // comes from the raster, so parts that intersect or stack can't capture
    // it. Without it, a ray starting just above the surface finds the
    // triangle; padding texels, which sit off the surface, fall back to the
    // full cage.
    // Padding has no meaningful footprint: every sample would sit on its
    // copied point, so it traces one.
    let aa = select(select(max(u32(u.a.y), 1u), 1u, low_only), 1u, !inside);
    let grid = aa * aa;
    // An even grid has no sample at the texel center, and the nearest one
    // sits a quarter texel off (2x2: at (-0.25, -0.25)); position, ID, part
    // and everything traced from them would shift against the normal map.
    // One extra ray at the center supplies them, without joining the average.
    let samples = grid + select(0u, 1u, aa % 2u == 0u);
    var tsum = vec3<f32>(0.0);
    var nsum = vec3<f32>(0.0);
    var hsum = 0.0;
    var hits = 0.0;
    var center_tri = NONE;
    var center_bary = vec2<f32>(0.0);
    var center_pos = lp.xyz;
    var center_n = n;
    var best_center = 1e9;
    for (var si = 0u; si < samples; si++) {
        let j = select(vec2<f32>(0.0), (vec2<f32>(f32(si % aa), f32(si / aa)) + 0.5) / f32(aa) - 0.5, si < grid);
        // Each sample projects along, and is expressed in, its own
        // interpolated low-poly frame; using the texel center's frame for
        // every sample tilts the averaged normal wherever the low poly curves.
        // Padding keeps the frame it was copied with: interpolating there
        // would blend the frames of two different islands.
        var fr = LowFrame(n, normalize(tg.xyz - n * dot(n, tg.xyz)), vec3<f32>(0.0), center.n, tg.xyz);
        fr.b = cross(fr.n, fr.t) * sign;
        if (inside) {
            fr = low_frame(c, j, inside, center, sign, lp.xyz, texel);
        }
        let o = lp.xyz + dpx * j.x + dpy * j.y;
        var front = u.a.z;
        var rear = u.a.w;
        if (low_only) {
            front = 2e-3;
            rear = 2e-3;
        }
        // Rays start above the surface along the normal, or on the cage.
        var origin = o + fr.n * front;
        var dir = -fr.n;
        var start = front;
        if (use_cage && built_cage) {
            // Along the interpolated projection direction, from the painted
            // distance out; rear follows the same multiplier.
            let out_dir = cage_center;
            let reach = textureLoad(cage_tan, c, 0).x;
            if (dot(out_dir, out_dir) > 1e-12 && reach > 1e-6) {
                dir = -normalize(out_dir);
                origin = o - dir * reach;
                start = reach;
                rear = rear * reach / max(u.a.z, 1e-6);
            }
        } else if (use_cage) {
            let oc = cage_center + dpx * j.x + dpy * j.y;
            let span = o - oc;
            let len = length(span);
            if (len > 1e-6) {
                origin = oc;
                dir = span / len;
                start = len;
            }
        }
        var h: Hit;
        if (own_tri != NONE) {
            h = surface_hit(own_tri, o);
            h.t = start;
        } else {
            cull_back = !low_only && (flags & 2u) != 0u;
            h = intersect(origin, dir, start + rear, false);
            cull_back = false;
            // Nothing seen from the front: flipped faces still count.
            if (h.tri == NONE && !low_only && (flags & 2u) != 0u) {
                h = intersect(origin, dir, start + rear, false);
            }
        }
        if (h.tri == NONE && low_only) {
            front = u.a.z;
            rear = u.a.w;
            start = front;
            // `origin` follows, so the hit point below is on this ray.
            origin = o + fr.n * front;
            h = intersect(origin, dir, front + rear, false);
        }
        if (h.tri != NONE) {
            let hn = hit_normal(h, dir);
            if (si < grid) {
                tsum += encode_tangent(hn, fr, sign);
                nsum += hn;
                hsum += start - h.t;
                hits += 1.0;
            }
            let jd = dot(j, j);
            if (jd < best_center) {
                best_center = jd;
                center_tri = h.tri;
                center_bary = vec2<f32>(h.b1, h.b2);
                center_pos = origin + dir * h.t;
                center_n = hn;
            }
        }
    }

    // Every ray missed: take the high poly's point nearest the texel, so the
    // maps hold its surface instead of a hole.
    if (hits == 0.0 && !low_only && (flags & 4u) != 0u) {
        let near = nearest_surface(lp.xyz, max(u.g.x, max(u.a.z, u.a.w)) * 4.0);
        if (near.tri != NONE) {
            var fr = LowFrame(n, normalize(tg.xyz - n * dot(n, tg.xyz)), vec3<f32>(0.0), center.n, tg.xyz);
            fr.b = cross(fr.n, fr.t) * sign;
            let hp = hit_point(near);
            let hn = hit_normal(near, -n);
            tsum = encode_tangent(hn, fr, sign);
            nsum = hn;
            hsum = dot(hp - lp.xyz, n);
            hits = 1.0;
            center_tri = near.tri;
            center_bary = vec2<f32>(near.b1, near.b2);
            center_pos = hp;
            center_n = hn;
            if (inside) {
                atomicAdd(&counters[2], 1u);
            }
        }
    }

    var nh = n;
    var ts = vec3<f32>(0.0, 0.0, 1.0);
    var height = 0.0;
    var missed = 0.0;
    if (hits > 0.0) {
        nh = normalize(nsum);
        if (!low_only) {
            ts = normalize(tsum);
            height = hsum / hits;
        }
    } else if (inside) {
        missed = 1.0;
        atomicAdd(&counters[1], 1u);
    }

    var prand = 0.5;
    var pgrad = 0.5;
    var idc = vec3<f32>(0.5);
    var geo = vec4<f32>(nh, -1.0);
    if (center_tri != NONE) {
        let pi = bitcast<u32>(tri_pos[center_tri * 3u].w);
        let part = parts[pi];
        prand = part.x;
        pgrad = clamp((center_pos.y - part.y) / max(part.z, 1e-5), 0.0, 1.0);
        let bary = vec3<f32>(1.0 - center_bary.x - center_bary.y, center_bary.x, center_bary.y);
        idc = tri_vcolor[center_tri * 3u].xyz * bary.x
            + tri_vcolor[center_tri * 3u + 1u].xyz * bary.y
            + tri_vcolor[center_tri * 3u + 2u].xyz * bary.z;
        geo = vec4<f32>(hit_geo_normal(center_tri, center_n), f32(pi));
    }
    // Heights span [-range, range] around mid-gray. With a cage, rays start
    // up to the cage's largest offset above the low poly, not `front`.
    let range = max(u.g.x, 1e-5);
    textureStore(surf_pos_w, c, vec4<f32>(center_pos, pgrad));
    textureStore(surf_nrm_w, c, vec4<f32>(nh, prand));
    textureStore(surf_geo_w, c, geo);
    textureStore(map_d_w, c, vec4<f32>(ts.x, ts.y, clamp(0.5 + height / (2.0 * range), 0.0, 1.0), missed));
    textureStore(map_e_w, c, vec4<f32>(idc, 1.0));
    var kv = 0.0;
    var tri_size = 0.0;
    if (center_tri != NONE) {
        let b = vec3<f32>(1.0 - center_bary.x - center_bary.y, center_bary.x, center_bary.y);
        kv = dot(b, vec3<f32>(tri_nrm[center_tri * 3u].w, tri_nrm[center_tri * 3u + 1u].w, tri_nrm[center_tri * 3u + 2u].w));
        let p0 = tri_pos[center_tri * 3u].xyz;
        let p1 = tri_pos[center_tri * 3u + 1u].xyz;
        let p2 = tri_pos[center_tri * 3u + 2u].xyz;
        tri_size = (distance(p0, p1) + distance(p1, p2) + distance(p2, p0)) / 3.0;
    }
    textureStore(surf_curv_w, c, vec4<f32>(kv, tri_size, 0.0, 0.0));
}

// ------------------------------------------------------------------ occlusion

@compute @workgroup_size(8, 8)
fn cs_occlusion(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = vec2<i32>(gid.xy) + vec2<i32>(u.e.xy);
    if (!in_bounds(c)) {
        return;
    }
    let sp = textureLoad(surf_pos, c, 0);
    if (sp.w < 0.0) {
        textureStore(occ_w, c, vec4<f32>(1.0));
        textureStore(bent_w, c, vec4<f32>(0.5, 1.0, 0.5, 1.0));
        return;
    }
    let p = sp.xyz;
    let n = normalize(textureLoad(surf_nrm, c, 0).xyz);
    let g = textureLoad(surf_geo, c, 0).xyz;
    let gs = normalize(g + n);
    let lp = textureLoad(low_pos, c, 0).xyz;
    let texel = texel_size(c, lp);
    // Jitter origins across the texel footprint within the surface's own
    // tangent plane, so they never sink below a surface tilted away from the low poly.
    var dpx = footprint(c, vec2<i32>(1, 0), lp);
    var dpy = footprint(c, vec2<i32>(0, 1), lp);
    dpx -= n * dot(n, dpx);
    dpy -= n * dot(n, dpy);
    // Offset scales with the texel: large enough to clear float error on big
    // texels, small enough not to skip contact occlusion on fine detail.
    let bias = clamp(texel * 0.25, 1e-5, 4e-3);
    // Rays at grazing angles hit the next facet at every concave fold of a
    // faceted mesh, which draws its triangle edges into AO and cavity; the
    // smooth surface the mesh stands for doesn't occlude there. Within reach
    // of the neighboring facets (the source triangle's size, at most a tenth
    // of the AO distance), hits on the same part facing the same way as the
    // surface are such folds and pass through. Everything else still counts:
    // skipping the whole reach hid real contact occlusion (overlapping trims,
    // other parts) up to 0.06 units away and, past the cavity distance, left
    // cavity nothing to see.
    let tri_size = textureLoad(surf_curv, c, 0).g;
    let geo_part = textureLoad(surf_geo, c, 0).w;
    let seed = world_seed(p);
    let shift_ao = rand2(seed);
    let shift_th = rand2(seed ^ 0x1b873593u);
    let shift_sky = rand2(seed ^ 0xcc9e2d51u);

    // Ambient occlusion, cavity and bent normal share one set of rays.
    fold_reach = min(max(texel * 1.5, tri_size), u.b.y * 0.1);
    fold_n = n;
    fold_part = select(NONE, u32(geo_part), geo_part >= 0.0);
    let ao_n = u32(u.b.x);
    var open = 0.0;
    var cav_open = 0.0;
    var bent = vec3<f32>(0.0);
    for (var i = 0u; i < ao_n; i++) {
        let r = hammersley(i, ao_n, shift_ao);
        let j = rand2(seed ^ (i * 0x02e5be93u + 17u)) - 0.5;
        // Kept above a normal halfway to the smooth one: the facet's own
        // normal alone makes the ray set change at every triangle edge.
        let o = p + dpx * j.x + dpy * j.y + gs * bias;
        let d = above(cosine_dir(n, r), gs);
        // Cosine-weighted directions: the plain hit ratio is the
        // cosine-weighted visibility (pdf cos/pi cancels the cosine term).
        let h = intersect(o, d, u.b.y, false);
        if (h.tri == NONE) {
            open += 1.0;
            cav_open += 1.0;
            bent += d;
        } else if (h.t > u.b.z) {
            cav_open += 1.0;
        }
    }
    fold_reach = 0.0;
    var ao = select(1.0, open / f32(ao_n), ao_n > 0u);
    var cavity = select(1.0, cav_open / f32(ao_n), ao_n > 0u);
    // A point buried inside another part (where a ball sinks into its base)
    // is fully occluded, but its AO rays can cross the empty inside without
    // hitting anything within the AO distance and read open. Those texels
    // are hidden, yet bilinear filtering and the denoiser would carry their
    // white onto the visible contact line, which should be the darkest spot.
    if (ao_n > 0u && buried(p + gs * bias, n, seed)) {
        ao = 0.0;
        cavity = 0.0;
    }
    let bent_n = select(n, normalize(bent), dot(bent, bent) > 1e-8);

    // Thickness: how far rays travel through the inside of the mesh.
    let th_n = u32(u.b.w);
    let th_d = max(u.c.x, 1e-4);
    var acc = 0.0;
    for (var i = 0u; i < th_n; i++) {
        let r = hammersley(i, th_n, shift_th);
        let d = above(cosine_dir(-n, r), -gs);
        let h = intersect(p - gs * bias, d, th_d, false);
        acc += select(th_d, h.t, h.tri != NONE);
    }
    let thickness = select(1.0, acc / (f32(th_n) * th_d), th_n > 0u);

    // Sky exposure: visibility of the sky inside a cone around world up.
    let sky_n = u32(u.c.y);
    var sky_open = 0.0;
    for (var i = 0u; i < sky_n; i++) {
        let r = hammersley(i, sky_n, shift_sky);
        let d = cone_dir(vec3<f32>(0.0, 1.0, 0.0), u.c.z, r);
        if (dot(d, n) <= 0.0 || dot(d, gs) <= 0.0) {
            continue;
        }
        let h = intersect(p + gs * bias, d, 1e6, true);
        if (h.tri == NONE) {
            sky_open += 1.0;
        }
    }
    let sky = select(1.0, sky_open / f32(sky_n), sky_n > 0u);

    textureStore(occ_w, c, vec4<f32>(ao, cavity, thickness, sky));
    textureStore(bent_w, c, vec4<f32>(bent_n * 0.5 + 0.5, 1.0));
}

const BURY_RAYS: u32 = 9u;

// Inside a closed part: most rays leaving the surface, however far they go,
// first meet the back of a face. (Only the outward hemisphere: rays into the
// point's own part would meet its front.) Open sheets only catch the rays
// heading their way.
fn buried(o: vec3<f32>, n: vec3<f32>, seed: u32) -> bool {
    let shift = rand2(seed ^ 0x85ebca6bu);
    var back = 0u;
    for (var i = 0u; i < BURY_RAYS; i++) {
        let d = cone_dir(n, 0.0, hammersley(i, BURY_RAYS, shift));
        let h = intersect(o, d, 1e6, false);
        if (h.tri != NONE && dot(hit_normal(h, d), d) > 0.0) {
            back += 1u;
        }
    }
    return back * 3u >= BURY_RAYS * 2u;
}

// ------------------------------------------------------------------ curvature

fn interior(c: vec2<i32>) -> bool {
    return textureLoad(low_pos, c, 0).w > 0.75;
}

@compute @workgroup_size(8, 8)
fn cs_curvature(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = vec2<i32>(gid.xy);
    if (!in_bounds(c)) {
        return;
    }
    let sp = textureLoad(surf_pos, c, 0);
    if (sp.w < 0.0) {
        textureStore(curv_w, c, vec4<f32>(0.0));
        return;
    }
    // The source mesh's vertex curvature at the projected point: smooth
    // across its triangles, unlike differences of interpolated normals,
    // which are constant per triangle and show a dense mesh as facets.
    textureStore(curv_w, c, vec4<f32>(textureLoad(surf_curv, c, 0).r, 0.0, 0.0, 0.0));
}

// ------------------------------------------------------------------ triangle-scale smoothing

// Curvature comes from the source mesh's vertices and is interpolated
// linearly across each triangle, and cavity rays follow each triangle's
// facet; on a coarse mesh both show the triangles as fans and blotches. One
// a-trous iteration (spacing `u.e.z`) of a Gaussian in world space whose
// width follows the local triangle size makes them smooth per pixel. Taps
// only count on the same part, the same UV island and the same side of a
// hard edge.
fn smooth_tap(q: vec2<i32>, off: f32, p0: vec3<f32>, n0: vec3<f32>, part: f32, texel: f32, sigma: f32) -> f32 {
    if (!in_bounds(q)) {
        return 0.0;
    }
    let qp = textureLoad(surf_pos, q, 0);
    if (qp.w < 0.0 || textureLoad(surf_geo, q, 0).w != part) {
        return 0.0;
    }
    let d = distance(qp.xyz, p0);
    if (d > off * texel * 2.0 + texel * 2.0) {
        return 0.0;
    }
    let wn = smoothstep(0.6, 0.85, dot(textureLoad(surf_nrm, q, 0).xyz, n0));
    return exp(-0.5 * d * d / (sigma * sigma)) * wn;
}

struct SmoothCenter {
    ok: bool,
    p: vec3<f32>,
    n: vec3<f32>,
    part: f32,
    texel: f32,
    tri: f32,
};

fn smooth_center(c: vec2<i32>) -> SmoothCenter {
    var s: SmoothCenter;
    let sp = textureLoad(surf_pos, c, 0);
    s.ok = sp.w >= 0.0;
    s.p = sp.xyz;
    s.n = textureLoad(surf_nrm, c, 0).xyz;
    s.part = textureLoad(surf_geo, c, 0).w;
    s.texel = texel_size(c, textureLoad(low_pos, c, 0).xyz);
    s.tri = textureLoad(surf_curv, c, 0).g;
    return s;
}

@compute @workgroup_size(8, 8)
fn cs_curv_smooth(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = vec2<i32>(gid.xy);
    if (!in_bounds(c)) {
        return;
    }
    let s = smooth_center(c);
    let v0 = textureLoad(curv_t, c, 0);
    if (!s.ok) {
        textureStore(curv_w, c, v0);
        return;
    }
    // The curvature radius setting (pixels). The vertex curvature is already
    // smooth across the mesh (see `vertex_curvature`), so this only evens
    // out the interpolation; a wider blur here would smooth each UV island
    // on its own and show its borders on meshes cut into many islands.
    let sigma = max(u.c.w, 1.0) * s.texel;
    let step = i32(u.e.z);
    let kernel = array<f32, 5>(0.0625, 0.25, 0.375, 0.25, 0.0625);
    var acc = 0.0;
    var wsum = 0.0;
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let q = c + vec2<i32>(dx, dy) * step;
            let off = length(vec2<f32>(f32(dx), f32(dy))) * f32(step);
            let w = kernel[dx + 2] * kernel[dy + 2] * select(smooth_tap(q, off, s.p, s.n, s.part, s.texel, sigma), 1.0, dx == 0 && dy == 0);
            acc += textureLoad(curv_t, q, 0).x * w;
            wsum += w;
        }
    }
    textureStore(curv_w, c, vec4<f32>(acc / wsum, v0.yzw));
}

// Ray-traced maps: AO, cavity, thickness and sky (occ) and the bent normal.
// Each smooths over half a triangle, but never wider than a fraction of its
// own ray distance, so contact shadows and cavity lines don't spread.
@compute @workgroup_size(8, 8)
fn cs_occ_smooth(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = vec2<i32>(gid.xy);
    if (!in_bounds(c)) {
        return;
    }
    let s = smooth_center(c);
    let o0 = textureLoad(occ_t, c, 0);
    let b0 = textureLoad(bent_t, c, 0);
    if (!s.ok) {
        textureStore(occ_w, c, o0);
        textureStore(bent_w, c, b0);
        return;
    }
    // Half a triangle (rays follow a coarse mesh's facets), but at most a
    // few texels: the pass smooths each UV island on its own, so a wider
    // blur shows island borders on meshes cut into many small islands.
    let want = min(s.tri * 0.5, 4.0 * s.texel);
    let ao_sigma = clamp(want, s.texel, max(u.b.y * 0.05, s.texel));
    let sigma = vec4<f32>(
        ao_sigma,
        clamp(want, s.texel, max(u.b.z * 0.5, s.texel)),
        clamp(want, s.texel, max(u.c.x * 0.25, s.texel)),
        ao_sigma,
    );
    let wide = max(max(sigma.x, sigma.y), max(sigma.z, sigma.w));
    let step = i32(u.e.z);
    let kernel = array<f32, 5>(0.0625, 0.25, 0.375, 0.25, 0.0625);
    var occ = vec4<f32>(0.0);
    var osum = vec4<f32>(0.0);
    var bent = vec3<f32>(0.0);
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let q = c + vec2<i32>(dx, dy) * step;
            let k = kernel[dx + 2] * kernel[dy + 2];
            if (dx == 0 && dy == 0) {
                occ += o0 * k;
                osum += vec4<f32>(k);
                bent += (b0.xyz * 2.0 - 1.0) * k;
                continue;
            }
            let off = length(vec2<f32>(f32(dx), f32(dy))) * f32(step);
            // The tap's gate (same surface) at the widest sigma, then each
            // map's own Gaussian on top.
            let gate = smooth_tap(q, off, s.p, s.n, s.part, s.texel, wide);
            if (gate <= 0.0) {
                continue;
            }
            let d = distance(textureLoad(surf_pos, q, 0).xyz, s.p);
            let g = exp(-0.5 * d * d / (sigma * sigma) + 0.5 * d * d / (wide * wide));
            let w = gate * g * k;
            occ += textureLoad(occ_t, q, 0) * w;
            osum += w;
            bent += (textureLoad(bent_t, q, 0).xyz * 2.0 - 1.0) * w.x;
        }
    }
    let bn = select(s.n, normalize(bent), dot(bent, bent) > 1e-8);
    textureStore(occ_w, c, occ / osum);
    textureStore(bent_w, c, vec4<f32>(bn * 0.5 + 0.5, b0.w));
}

// ------------------------------------------------------------------ edge / crease distance

@compute @workgroup_size(8, 8)
fn cs_jfa_init(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = vec2<i32>(gid.xy);
    if (!in_bounds(c)) {
        return;
    }
    var k = textureLoad(curv_t, c, 0).x * u.d.z;
    var threshold = u.d.x;
    if (u.d.w > 0.5) {
        // The low poly alone: raw curvature against the bevel reach.
        k = textureLoad(curv_t, c, 0).x;
        threshold = u.f.w;
    }
    // Seeds come only from inside UV islands, never from the padding.
    let valid = textureLoad(surf_pos, c, 0).w >= 0.0 && interior(c);
    let me = vec2<f32>(c);
    let edge = select(vec2<f32>(-1.0), me, valid && k > threshold);
    let crease = select(vec2<f32>(-1.0), me, valid && k < -threshold);
    textureStore(jfa_w, c, vec4<f32>(edge, crease));
}

// Distance to a JFA seed texel, if it lies on the same part: another
// part's edge (a ring sitting on a ball) isn't this part's edge.
fn seed_dist(p: vec3<f32>, s: vec2<f32>, part: f32) -> f32 {
    if (s.x < 0.0) {
        return 1e9;
    }
    let q = vec2<i32>(s);
    if (textureLoad(surf_geo, q, 0).w != part) {
        return 1e9;
    }
    return distance(p, textureLoad(surf_pos, q, 0).xyz);
}

@compute @workgroup_size(8, 8)
fn cs_jfa_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = vec2<i32>(gid.xy);
    if (!in_bounds(c)) {
        return;
    }
    var cur = textureLoad(jfa_t, c, 0);
    let sp = textureLoad(surf_pos, c, 0);
    if (sp.w < 0.0) {
        textureStore(jfa_w, c, cur);
        return;
    }
    let step = i32(u.e.z);
    let part = textureLoad(surf_geo, c, 0).w;
    var de = seed_dist(sp.xyz, cur.xy, part);
    var dc = seed_dist(sp.xyz, cur.zw, part);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let q = c + vec2<i32>(dx, dy) * step;
            if ((dx == 0 && dy == 0) || !in_bounds(q)) {
                continue;
            }
            let cand = textureLoad(jfa_t, q, 0);
            let e = seed_dist(sp.xyz, cand.xy, part);
            if (e < de) {
                de = e;
                cur.x = cand.x;
                cur.y = cand.y;
            }
            let k = seed_dist(sp.xyz, cand.zw, part);
            if (k < dc) {
                dc = k;
                cur.z = cand.z;
                cur.w = cand.w;
            }
        }
    }
    textureStore(jfa_w, c, cur);
}

// ------------------------------------------------------------------ hard edges

// Distance from p to the nearest edge segment in the BVH at `root` that
// belongs to `part` (-1 = any part), capped at `cap`. A segment's second w is
// its band half-width (bevel facets read 0 inside), or negative for a hard
// edge; `hard_only` skips everything else.
fn nearest_segment(p: vec3<f32>, root: i32, part: f32, cap: f32, hard_only: bool) -> f32 {
    var best = cap;
    if (root < 0) {
        return best;
    }
    var stack: array<u32, 64>;
    var sp = 1u;
    stack[0] = u32(root);
    loop {
        if (sp == 0u) {
            break;
        }
        sp -= 1u;
        let ni = stack[sp];
        let a = seg_nodes[ni * 2u];
        let b = seg_nodes[ni * 2u + 1u];
        let dv = max(max(a.xyz - p, p - b.xyz), vec3<f32>(0.0));
        if (dot(dv, dv) >= best * best) {
            continue;
        }
        let first = bitcast<u32>(a.w);
        let count = bitcast<u32>(b.w);
        if (count > 0u) {
            for (var i = 0u; i < count; i++) {
                let s0 = segs[(first + i) * 2u];
                if (part >= 0.0 && s0.w != part) {
                    continue;
                }
                let s1 = segs[(first + i) * 2u + 1u];
                if (hard_only && s1.w >= 0.0) {
                    continue;
                }
                let e = s1.xyz - s0.xyz;
                let t = clamp(dot(p - s0.xyz, e) / max(dot(e, e), 1e-20), 0.0, 1.0);
                best = min(best, max(distance(p, s0.xyz + e * t) - max(s1.w, 0.0), 0.0));
            }
        } else if (sp + 2u <= STACK) {
            stack[sp] = first;
            stack[sp + 1u] = first + 1u;
            sp += 2u;
        }
    }
    return best;
}

@compute @workgroup_size(8, 8)
fn cs_edges(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = vec2<i32>(gid.xy);
    if (!in_bounds(c)) {
        return;
    }
    let sp = textureLoad(surf_pos, c, 0);
    let emax = max(u.d.y, 1e-5);
    if (sp.w < 0.0) {
        textureStore(edge_w, c, vec4<f32>(emax));
        return;
    }
    let part = textureLoad(surf_geo, c, 0).w;
    // Hard edges of the source mesh, measured in 3D: unaffected by UV seams.
    let he = nearest_segment(sp.xyz, i32(u.f.x), part, emax, false);
    let hc = nearest_segment(sp.xyz, i32(u.f.y), part, emax, false);
    // Curvature lines are drawn along hard edges only: bevels already have
    // real curvature, and a line per bevel facet would stripe them.
    let lines = vec2<f32>(
        nearest_segment(sp.xyz, i32(u.f.x), part, emax, true),
        nearest_segment(sp.xyz, i32(u.f.y), part, emax, true),
    );
    // Bevels and detail, traced from the surface curvature.
    let j = textureLoad(jfa_t, c, 0);
    let je = min(seed_dist(sp.xyz, j.xy, part), emax);
    let jc = min(seed_dist(sp.xyz, j.zw, part), emax);
    textureStore(edge_w, c, vec4<f32>(min(he, je), min(hc, jc), lines));
}

// ------------------------------------------------------------------ denoise

// One a-trous iteration (5x5 B3 kernel at spacing `step`) over the ray-traced
// maps. Neighbors only contribute when they lie on the same surface: similar
// normal, close to the tangent plane and at the expected 3D distance, which
// also rejects texels from other UV islands.
@compute @workgroup_size(8, 8)
fn cs_denoise(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = vec2<i32>(gid.xy);
    if (!in_bounds(c)) {
        return;
    }
    let sp = textureLoad(surf_pos, c, 0);
    if (sp.w < 0.0) {
        textureStore(occ_w, c, textureLoad(occ_t, c, 0));
        textureStore(bent_w, c, textureLoad(bent_t, c, 0));
        return;
    }
    let step = i32(u.e.z);
    let p0 = sp.xyz;
    let n0 = textureLoad(surf_nrm, c, 0).xyz;
    let texel = texel_size(c, textureLoad(low_pos, c, 0).xyz);
    let kernel = array<f32, 5>(0.0625, 0.25, 0.375, 0.25, 0.0625);
    var occ = vec4<f32>(0.0);
    var bent = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let q = c + vec2<i32>(dx, dy) * step;
            if (!in_bounds(q)) {
                continue;
            }
            let qp = textureLoad(surf_pos, q, 0);
            if (qp.w < 0.0) {
                continue;
            }
            let expect = length(vec2<f32>(f32(dx), f32(dy))) * f32(step) * texel;
            if (distance(qp.xyz, p0) > expect * 2.0 + texel * 2.0) {
                continue;
            }
            let nq = textureLoad(surf_nrm, q, 0).xyz;
            let wn = pow(max(dot(nq, n0), 0.0), 64.0);
            let plane = abs(dot(qp.xyz - p0, n0)) / (texel * f32(step));
            let wp = exp(-plane * plane * 4.0);
            let w = kernel[dx + 2] * kernel[dy + 2] * wn * wp;
            occ += textureLoad(occ_t, q, 0) * w;
            bent += (textureLoad(bent_t, q, 0).xyz * 2.0 - 1.0) * w;
            wsum += w;
        }
    }
    if (wsum < 1e-6) {
        textureStore(occ_w, c, textureLoad(occ_t, c, 0));
        textureStore(bent_w, c, textureLoad(bent_t, c, 0));
        return;
    }
    let bn = select(n0, normalize(bent), dot(bent, bent) > 1e-8);
    textureStore(occ_w, c, occ / wsum);
    textureStore(bent_w, c, vec4<f32>(bn * 0.5 + 0.5, 1.0));
}

// ------------------------------------------------------------------ custom maps

const CK_FLOW: u32 = 1u;
const CK_EXPOSURE: u32 = 2u;
const CK_DEPOSIT: u32 = 3u;
const CK_CONTACT: u32 = 4u;
const CK_POINT: u32 = 5u;
// Steps of a flow walk; each covers 1/FLOW_STEPS of the flow length.
const FLOW_STEPS: u32 = 32u;

fn lattice(c: vec3<f32>) -> f32 {
    let i = vec3<i32>(c);
    let h = hash_u(bitcast<u32>(i.x) * 73856093u ^ bitcast<u32>(i.y) * 19349663u ^ bitcast<u32>(i.z) * 83492791u);
    return f32(h & 0xffffffu) / 16777216.0;
}

// Smooth value noise in [0, 1): flow paths meander alike at neighboring
// texels, so they read as streaks instead of grain.
fn value_noise(x: vec3<f32>) -> f32 {
    let i = floor(x);
    let f = fract(x);
    let w = f * f * (3.0 - 2.0 * f);
    let a = mix(lattice(i), lattice(i + vec3<f32>(1.0, 0.0, 0.0)), w.x);
    let b = mix(lattice(i + vec3<f32>(0.0, 1.0, 0.0)), lattice(i + vec3<f32>(1.0, 1.0, 0.0)), w.x);
    let c = mix(lattice(i + vec3<f32>(0.0, 0.0, 1.0)), lattice(i + vec3<f32>(1.0, 0.0, 1.0)), w.x);
    let d = mix(lattice(i + vec3<f32>(0.0, 1.0, 1.0)), lattice(i + vec3<f32>(1.0, 1.0, 1.0)), w.x);
    return mix(mix(a, b, w.y), mix(c, d, w.y), w.z);
}

struct Landing {
    ok: bool,
    p: vec3<f32>,
    n: vec3<f32>,
};

// Where a ray from `o` along `d` meets the surface, with the normal on the
// side it arrives from.
fn land(o: vec3<f32>, d: vec3<f32>, h: Hit) -> Landing {
    var r: Landing;
    r.ok = h.tri != NONE;
    if (r.ok) {
        r.p = o + d * h.t;
        let n = hit_normal(h, d);
        r.n = select(n, -n, dot(n, d) > 0.0);
    }
    return r;
}

// One step of length `s` along the surface from `p` (normal `n`) toward the
// tangent direction `d`: up a wall met on the way (a concave corner), on
// along the surface, or around a convex edge onto the face beyond. Fails
// where the surface ends (an open border).
fn surface_step(p: vec3<f32>, n: vec3<f32>, d: vec3<f32>, s: f32) -> Landing {
    let lift = s * 0.25;
    let o = p + n * lift;
    var r = land(o, d, intersect(o, d, s, false));
    if (r.ok) {
        return r;
    }
    let a = o + d * s;
    let down = lift + s * 2.0;
    r = land(a, -n, intersect(a, -n, down, false));
    if (r.ok) {
        return r;
    }
    let b = a - n * down;
    return land(b, -d, intersect(b, -d, s * 2.0, false));
}

// How much of a fluid arriving from `src` lands at a point: facing it, with
// nothing in the way. `r` picks one direction in the cone of `cos_spread`
// around it (rain never falls quite straight), which softens the shelter
// overhangs give.
fn fluid_catch(p: vec3<f32>, n: vec3<f32>, src: vec3<f32>, cos_spread: f32, r: vec2<f32>, bias: f32) -> f32 {
    let d = cone_dir(src, cos_spread, r);
    let f = dot(n, d);
    if (f <= 0.0) {
        return 0.0;
    }
    return select(0.0, f, intersect(p + n * bias, d, 1e6, true).tri == NONE);
}

// Fluid flow: fluid lands on the surfaces facing where it comes from and
// runs along the surface away from it. Each walk follows a texel's flow
// upstream for the flow length, collecting the fluid landing on the way; the
// more surface drains through a texel, the brighter. Under a ledge or a bolt
// head this draws the run-off streaks below it, fading with distance.
fn flow_map(p0: vec3<f32>, n0: vec3<f32>, src: vec3<f32>, cos_spread: f32, len: f32, wander: f32, walks: u32, gain: f32, bias: f32, seed: u32) -> f32 {
    let s = len / f32(FLOW_STEPS);
    let freq = 6.0 / max(len, 1e-4);
    let shift = rand2(seed ^ 0x3c6ef372u);
    var total = 0.0;
    for (var w = 0u; w < walks; w++) {
        var p = p0;
        var n = n0;
        var heading = vec3<f32>(0.0);
        var acc = 0.0;
        let offset = vec3<f32>(f32(w) * 17.13, f32(w) * 5.71, f32(w) * 11.37);
        for (var k = 0u; k < FLOW_STEPS; k++) {
            // The catch directions of all walks and steps form one
            // low-discrepancy set: far less noise than picking each at random.
            let dir = hammersley(w * FLOW_STEPS + k, walks * FLOW_STEPS, shift);
            let wet = fluid_catch(p, n, src, cos_spread, dir, bias);
            acc += wet * s;
            // Upstream: toward the source, along the surface. Where the
            // surface faces the flow head-on (a roof under rain) there is no
            // downhill; fluid spreads, so the walk keeps its heading.
            var t = src - n * dot(src, n);
            if (length(t) < 0.2) {
                t = heading - n * dot(heading, n);
            }
            let tl = length(t);
            if (tl < 1e-4) {
                // Level ground with nowhere to come from: fluid stands here,
                // as wet as a full run.
                acc += wet * s * f32(FLOW_STEPS - 1u - k);
                break;
            }
            var d = t / tl;
            if (wander > 0.0) {
                let a = (value_noise(p * freq + offset) - 0.5) * 2.0 * wander;
                d = normalize(d * cos(a) + cross(n, d) * sin(a));
            }
            heading = d;
            let r = surface_step(p, n, d, s);
            if (!r.ok) {
                break;
            }
            p = r.p;
            n = r.n;
        }
        total += 1.0 - exp(-gain * acc / max(len, 1e-4));
    }
    return total / f32(max(walks, 1u));
}

// The ray maps below depend on the smooth (interpolated) normal only, never
// on the face normal of the triangle under the texel: that one changes at
// every triangle edge, and where a UV seam runs along such an edge the
// step can't be smoothed away (smoothing stays within a UV island), so it
// showed as a line along the seam. Rays leaving a faceted surface around
// its smooth normal can graze the neighboring facets; `fold_on` lets them
// pass, as for ambient occlusion.

// Ignore hits on the texel's own part, facing the same way, within `reach`
// (see `fold_reach`).
fn fold_on(n: vec3<f32>, part: f32, reach: f32) {
    fold_reach = reach;
    fold_n = n;
    fold_part = select(NONE, u32(part), part >= 0.0);
}

fn fold_off() {
    fold_reach = 0.0;
    fold_part = NONE;
}

// `d` mirrored to the outside of the smooth surface.
fn outward(d: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let k = dot(d, n);
    return select(d, normalize(d - 2.0 * k * n), k < 1e-3);
}

// Exposure to a direction (wind, sandblasting, sun bleaching): how much of
// a cone around it the surface faces and sees without anything in the way
// within `dist`.
fn exposure_map(p: vec3<f32>, n: vec3<f32>, src: vec3<f32>, cos_spread: f32, dist: f32, rays: u32, bias: f32, seed: u32) -> f32 {
    let shift = rand2(seed ^ 0x2545f491u);
    var acc = 0.0;
    for (var i = 0u; i < rays; i++) {
        let d = cone_dir(src, cos_spread, hammersley(i, rays, shift));
        let f = dot(d, n);
        if (f <= 0.0) {
            continue;
        }
        if (intersect(p + n * bias, d, dist, true).tri == NONE) {
            acc += f;
        }
    }
    return acc / f32(max(rays, 1u));
}

// Deposits (sand, snow, dust drifts): ground facing up, sheltered from the
// wind by something within `dist` upwind.
fn deposit_map(p: vec3<f32>, n: vec3<f32>, src: vec3<f32>, cos_spread: f32, dist: f32, rays: u32, bias: f32, seed: u32) -> f32 {
    let ground = smoothstep(0.25, 0.75, n.y);
    if (ground <= 0.0) {
        return 0.0;
    }
    let shift = rand2(seed ^ 0x68e31da4u);
    var blocked = 0.0;
    for (var i = 0u; i < rays; i++) {
        let d = outward(cone_dir(src, cos_spread, hammersley(i, rays, shift)), n);
        if (intersect(p + n * bias, d, dist, true).tri != NONE) {
            blocked += 1.0;
        }
    }
    return ground * blocked / f32(max(rays, 1u));
}

// Contact: how close another part of the mesh comes (rays over the whole
// hemisphere, ignoring the texel's own part), 1 touching, 0 at `dist`.
fn contact_map(p: vec3<f32>, n: vec3<f32>, part: f32, dist: f32, rays: u32, bias: f32, seed: u32) -> f32 {
    if (part < 0.0) {
        return 0.0;
    }
    skip_part = u32(part);
    let shift = rand2(seed ^ 0x1b56c4e9u);
    var nearest = dist;
    for (var i = 0u; i < rays; i++) {
        let d = cone_dir(n, 0.0, hammersley(i, rays, shift));
        let h = intersect(p + n * bias, d, nearest, false);
        if (h.tri != NONE) {
            nearest = h.t;
        }
    }
    skip_part = NONE;
    let x = 1.0 - nearest / max(dist, 1e-5);
    return x * x;
}

@compute @workgroup_size(8, 8)
fn cs_custom(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = vec2<i32>(gid.xy) + vec2<i32>(u.e.xy);
    if (!in_bounds(c)) {
        return;
    }
    let sp = textureLoad(surf_pos, c, 0);
    if (sp.w < 0.0) {
        textureStore(cust_w, c, vec4<f32>(0.0));
        return;
    }
    let p = sp.xyz;
    let n = normalize(textureLoad(surf_nrm, c, 0).xyz);
    let geo = textureLoad(surf_geo, c, 0);
    let texel = texel_size(c, textureLoad(low_pos, c, 0).xyz);
    let bias = clamp(texel * 0.25, 1e-5, 4e-3);
    // Neighboring facets within reach (see `fold_on`).
    let reach = max(texel * 1.5, textureLoad(surf_curv, c, 0).g);
    let seed = world_seed(p) ^ 0x51ed27u;
    var out = vec4<f32>(0.0);
    for (var k = 0u; k < 4u; k++) {
        let a = u.cust[k * 3u];
        let b = u.cust[k * 3u + 1u];
        let q = u.cust[k * 3u + 2u];
        let rays = u32(a.w);
        let ks = seed ^ (k * 0x27d4eb2fu);
        var v = 0.0;
        switch u32(a.x) {
            case 1u: { v = flow_map(p, n, b.xyz, a.z, a.y, q.w, rays, b.w, bias, ks); }
            case 2u: {
                fold_on(n, geo.w, min(reach, a.y * 0.1));
                v = exposure_map(p, n, b.xyz, a.z, a.y, rays, bias, ks);
                fold_off();
            }
            case 3u: {
                fold_on(n, geo.w, min(reach, a.y * 0.1));
                v = deposit_map(p, n, b.xyz, a.z, a.y, rays, bias, ks);
                fold_off();
            }
            case 4u: { v = contact_map(p, n, geo.w, a.y, rays, bias, ks); }
            case 5u: { v = 1.0 - smoothstep(0.0, max(a.y, 1e-5), distance(p, q.xyz)); }
            default: {}
        }
        out[k] = clamp(v, 0.0, 1.0);
    }
    textureStore(cust_w, c, out);
}

// Ray-sampled custom maps carry per-texel noise; smooth them over a few
// texels on the same surface (as `cs_curv_smooth` does).
@compute @workgroup_size(8, 8)
fn cs_cust_smooth(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = vec2<i32>(gid.xy);
    if (!in_bounds(c)) {
        return;
    }
    let s = smooth_center(c);
    let v0 = textureLoad(cust_t, c, 0);
    if (!s.ok) {
        textureStore(cust_w, c, v0);
        return;
    }
    let sigma = clamp(s.tri * 0.5, s.texel, 3.0 * s.texel);
    let step = i32(u.e.z);
    let kernel = array<f32, 5>(0.0625, 0.25, 0.375, 0.25, 0.0625);
    var acc = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let q = c + vec2<i32>(dx, dy) * step;
            let off = length(vec2<f32>(f32(dx), f32(dy))) * f32(step);
            let w = kernel[dx + 2] * kernel[dy + 2] * select(smooth_tap(q, off, s.p, s.n, s.part, s.texel, sigma), 1.0, dx == 0 && dy == 0);
            if (w <= 0.0) {
                continue;
            }
            acc += textureLoad(cust_t, q, 0) * w;
            wsum += w;
        }
    }
    textureStore(cust_w, c, acc / wsum);
}

// ------------------------------------------------------------------ pack

@compute @workgroup_size(8, 8)
fn cs_pack(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = vec2<i32>(gid.xy);
    if (!in_bounds(c)) {
        return;
    }
    let sp = textureLoad(surf_pos, c, 0);
    if (sp.w < 0.0) {
        textureStore(map_a_w, c, vec4<f32>(1.0, 0.5, 1.0, 1.0));
        textureStore(map_b_w, c, vec4<f32>(1.0, 1.0, 1.0, 0.5));
        textureStore(map_c_w, c, vec4<f32>(0.5, 1.0, 0.5, 0.5));
        return;
    }
    let occ = textureLoad(occ_t, c, 0);
    let bent = textureLoad(bent_t, c, 0);
    let k = textureLoad(curv_t, c, 0).x;
    let ed = textureLoad(edge_t, c, 0);
    let sn = textureLoad(surf_nrm, c, 0);
    let emax = max(u.d.y, 1e-5);
    // Curvature from the surface, plus thin lines along hard edges that the
    // texel-space estimate can't see (flat faces meeting at a split normal).
    let line = max(u.c.w, 1.0) * texel_size(c, textureLoad(low_pos, c, 0).xyz);
    let convex = 1.0 - smoothstep(0.0, line, ed.z);
    let concave = 1.0 - smoothstep(0.0, line, ed.w);
    let ks = k * u.d.z;
    var curv = 0.5 + 0.5 * clamp(max(max(ks, 0.0), convex) - max(max(-ks, 0.0), concave), -1.0, 1.0);
    var edge = clamp(ed.x / emax, 0.0, 1.0);
    var crease = clamp(ed.y / emax, 0.0, 1.0);
    // Maps left out of the bake are neutral: nothing occluded, flat, far
    // from any edge.
    let baked = u32(u.g.y);
    var o = occ;
    if ((baked & 1u) == 0u) { o.x = 1.0; }
    if ((baked & 2u) == 0u) { curv = 0.5; }
    if ((baked & 4u) == 0u) { o.y = 1.0; }
    if ((baked & 8u) == 0u) { o.z = 1.0; }
    if ((baked & 16u) == 0u) { o.w = 1.0; }
    if ((baked & 32u) == 0u) { edge = 1.0; }
    if ((baked & 64u) == 0u) { crease = 1.0; }
    textureStore(map_a_w, c, vec4<f32>(o.x, curv, o.y, o.z));
    textureStore(map_b_w, c, vec4<f32>(o.w, edge, crease, sn.w));
    textureStore(map_c_w, c, vec4<f32>(bent.xyz, sp.w));
}
