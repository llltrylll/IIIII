// Progressive path tracer for the 3D viewport (the "Iray" preview).
// One sample per pixel per dispatch, accumulated while the view is unchanged.
// Lighting matches the raster viewport: everything comes from the environment
// HDRI, importance-sampled by luminance and combined with BSDF sampling by
// multiple importance sampling. Output is scene-linear; post.wgsl tone maps.

struct RtU {
    inv_view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    frame: vec4<f32>,    // width, height, sample index, max bounces
    light: vec4<f32>,    // env rotation, emission strength, ground y, ground enabled
    params: vec4<f32>,   // use baked normal, height strength, texel size, texture resolution
    env: vec4<f32>,      // env intensity, radiance max lod, background blur, background mode
    bg: vec4<f32>,       // background color (linear), background intensity
    sun: vec4<f32>,      // direction toward the key light, intensity
    sun_c: vec4<f32>,    // key light color (linear), on (1) / off (0)
};

@group(0) @binding(0) var<uniform> u: RtU;
@group(0) @binding(1) var<storage, read_write> accum: array<vec4<f32>>;
@group(0) @binding(2) var out_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<storage, read> nodes: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> tri_pos: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> tri_nrm: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> tri_uv: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> tri_tan: array<vec4<f32>>;
// One layer per texture set; a triangle's set rides in its tri_uv.z.
@group(0) @binding(8) var t_color: texture_2d_array<f32>;
@group(0) @binding(9) var t_rough: texture_2d_array<f32>;
@group(0) @binding(10) var t_metal: texture_2d_array<f32>;
@group(0) @binding(11) var t_height: texture_2d_array<f32>;
@group(0) @binding(12) var t_baked: texture_2d_array<f32>;
@group(0) @binding(13) var samp: sampler;
@group(0) @binding(14) var env_rad: texture_2d<f32>;
@group(0) @binding(15) var env_samp: sampler;
// [total, marginal cdf (H + 1), row cdfs H * (W + 1), func W * H]; see environment.rs.
@group(0) @binding(16) var<storage, read> env_cdf: array<f32>;
@group(0) @binding(17) var t_emission: texture_2d_array<f32>;  // sRGB view: reads linear
@group(0) @binding(18) var t_normal: texture_2d_array<f32>;  // painted normal detail, tangent space, encoded 0..1
@group(0) @binding(19) var t_spec: texture_2d_array<f32>;  // specular level (0.5: 4% reflectance)

const PI: f32 = 3.14159265;
const NONE: u32 = 0xffffffffu;
const MISS: f32 = 3.0e38;

// ---------------------------------------------------------------- random

var<private> rng: u32;

fn next_u() -> u32 {
    rng = rng * 747796405u + 2891336453u;
    var w = ((rng >> ((rng >> 28u) + 4u)) ^ rng) * 277803737u;
    return (w >> 22u) ^ w;
}

fn rand() -> f32 {
    return f32(next_u() >> 8u) / 16777216.0;
}

fn rand2() -> vec2<f32> {
    return vec2<f32>(rand(), rand());
}

// ---------------------------------------------------------------- BVH

struct Hit {
    t: f32,
    tri: u32,
    b1: f32,
    b2: f32,
};

fn box_entry(ni: u32, o: vec3<f32>, inv: vec3<f32>, tmax: f32) -> f32 {
    let t0 = (nodes[ni * 2u].xyz - o) * inv;
    let t1 = (nodes[ni * 2u + 1u].xyz - o) * inv;
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
    if (box_entry(0u, o, inv, h.t) == MISS) {
        return h;
    }
    var stack: array<u32, 64>;
    var sp = 1u;
    stack[0] = 0u;
    loop {
        if (sp == 0u) {
            break;
        }
        sp -= 1u;
        let ni = stack[sp];
        let first = bitcast<u32>(nodes[ni * 2u].w);
        let count = bitcast<u32>(nodes[ni * 2u + 1u].w);
        if (count > 0u) {
            for (var i = 0u; i < count; i++) {
                let ti = first + i;
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
                let bu = dot(s, p) * idet;
                if (bu < 0.0 || bu > 1.0) {
                    continue;
                }
                let q = cross(s, e1);
                let bv = dot(d, q) * idet;
                if (bv < 0.0 || bu + bv > 1.0) {
                    continue;
                }
                let t = dot(e2, q) * idet;
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
        } else if (sp + 2u <= 64u) {
            let tl = box_entry(first, o, inv, h.t);
            let tr = box_entry(first + 1u, o, inv, h.t);
            if (tl < MISS && tr < MISS) {
                stack[sp] = select(first, first + 1u, tl <= tr);
                stack[sp + 1u] = select(first + 1u, first, tl <= tr);
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

// ---------------------------------------------------------------- lighting (matches view.wgsl)

fn rot_y(d: vec3<f32>, a: f32) -> vec3<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec3<f32>(c * d.x + s * d.z, d.y, -s * d.x + c * d.z);
}

// Environment radiance; `blur` 0..1 picks a prefiltered mip (backdrop only).
fn env(d: vec3<f32>, blur: f32) -> vec3<f32> {
    let uv = env_dir_to_uv(rot_y(d, -u.light.x));
    return textureSampleLevel(env_rad, env_samp, uv, blur * u.env.y).rgb * u.env.x;
}

const CDF_W: u32 = 256u;
const CDF_H: u32 = 128u;
const ROWS: u32 = 1u + (CDF_H + 1u);
const FUNC: u32 = ROWS + CDF_H * (CDF_W + 1u);

// Last index i in [0, n) with cdf[base + i] <= x.
fn find_interval(base: u32, n: u32, x: f32) -> u32 {
    var lo = 0u;
    var hi = n;
    while (hi - lo > 1u) {
        let mid = (lo + hi) / 2u;
        if (env_cdf[base + mid] <= x) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// A world direction drawn proportionally to environment luminance, and its
// solid-angle pdf (w).
fn sample_env(r: vec2<f32>) -> vec4<f32> {
    let y = find_interval(1u, CDF_H, r.y);
    let m0 = env_cdf[1u + y];
    let m1 = env_cdf[1u + y + 1u];
    let v = (f32(y) + (r.y - m0) / max(m1 - m0, 1e-12)) / f32(CDF_H);
    let row = ROWS + y * (CDF_W + 1u);
    let x = find_interval(row, CDF_W, r.x);
    let c0 = env_cdf[row + x];
    let c1 = env_cdf[row + x + 1u];
    let uu = (f32(x) + (r.x - c0) / max(c1 - c0, 1e-12)) / f32(CDF_W);
    let pdf_uv = env_cdf[FUNC + y * CDF_W + x] / max(env_cdf[0], 1e-12);
    let sin_t = sin(v * PI);
    let pdf = pdf_uv / max(2.0 * PI * PI * sin_t, 1e-8);
    return vec4<f32>(rot_y(env_uv_to_dir(vec2<f32>(uu, v)), u.light.x), pdf);
}

fn env_pdf(d: vec3<f32>) -> f32 {
    let uv = env_dir_to_uv(rot_y(d, -u.light.x));
    let x = min(u32(uv.x * f32(CDF_W)), CDF_W - 1u);
    let y = min(u32(uv.y * f32(CDF_H)), CDF_H - 1u);
    let pdf_uv = env_cdf[FUNC + y * CDF_W + x] / max(env_cdf[0], 1e-12);
    return pdf_uv / max(2.0 * PI * PI * sin(uv.y * PI), 1e-8);
}

fn power_heuristic(a: f32, b: f32) -> f32 {
    return a * a / max(a * a + b * b, 1e-12);
}

fn background(d: vec3<f32>) -> vec3<f32> {
    if (u.env.w > 0.5 && u.env.w < 1.5) {
        return u.bg.rgb;
    }
    return env(d, u.env.z) * u.bg.w;
}

// ---------------------------------------------------------------- BRDF

fn onb(n: vec3<f32>) -> mat3x3<f32> {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    return mat3x3<f32>(
        vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x),
        vec3<f32>(b, s + n.y * n.y * a, -n.y),
        n
    );
}

fn ggx_d(nh: f32, a: f32) -> f32 {
    let a2 = a * a;
    let d = nh * nh * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

fn smith_g(nv: f32, nl: f32, a: f32) -> f32 {
    let a2 = a * a;
    let gv = nl * sqrt(nv * nv * (1.0 - a2) + a2);
    let gl = nv * sqrt(nl * nl * (1.0 - a2) + a2);
    return 2.0 * nv * nl / max(gv + gl, 1e-6);
}

struct Mat {
    albedo: vec3<f32>,
    rough: f32,
    metal: f32,
    // Opacity (base color alpha): the share of rays the surface stops.
    alpha: f32,
    // Light the surface gives off (linear, times the view's emission strength).
    emission: vec3<f32>,
    // Specular level of non-metals (0.5: 4% reflectance).
    spec: f32,
};

fn brdf(m: Mat, n: vec3<f32>, v: vec3<f32>, l: vec3<f32>) -> vec3<f32> {
    let nl = dot(n, l);
    let nv = dot(n, v);
    if (nl <= 0.0 || nv <= 0.0) {
        return vec3<f32>(0.0);
    }
    let h = normalize(v + l);
    let a = max(m.rough * m.rough, 0.002);
    let f0 = mix(vec3<f32>(0.08 * m.spec), m.albedo, m.metal);
    let f = f0 + (1.0 - f0) * pow(1.0 - max(dot(v, h), 0.0), 5.0);
    let spec = f * ggx_d(max(dot(n, h), 0.0), a) * smith_g(nv, nl, a) / (4.0 * nv * nl);
    let diff = (1.0 - f) * m.albedo * (1.0 - m.metal) / PI;
    return diff + spec;
}

// ---------------------------------------------------------------- surface

struct Surface {
    p: vec3<f32>,
    ng: vec3<f32>,
    n: vec3<f32>,
    m: Mat,
};

fn mesh_surface(h: Hit, o: vec3<f32>, d: vec3<f32>) -> Surface {
    var s: Surface;
    let i = h.tri * 3u;
    let w0 = 1.0 - h.b1 - h.b2;
    let v0 = tri_pos[i].xyz;
    let v1 = tri_pos[i + 1u].xyz;
    let v2 = tri_pos[i + 2u].xyz;
    s.p = o + d * h.t;
    s.ng = normalize(cross(v1 - v0, v2 - v0));
    var rn = tri_nrm[i].xyz * w0 + tri_nrm[i + 1u].xyz * h.b1 + tri_nrm[i + 2u].xyz * h.b2;
    var n = normalize(rn);
    let uv = tri_uv[i].xy * w0 + tri_uv[i + 1u].xy * h.b1 + tri_uv[i + 2u].xy * h.b2;
    let layer = i32(tri_uv[i].z);
    let tg = tri_tan[i] * w0 + tri_tan[i + 1u] * h.b1 + tri_tan[i + 2u] * h.b2;
    // Face toward the incoming ray (open meshes, backfaces).
    let back_face = dot(s.ng, d) > 0.0;
    if (back_face) {
        s.ng = -s.ng;
    }
    if (dot(n, s.ng) < 0.0) {
        n = -n;
        rn = -rn;
    }

    let col = textureSampleLevel(t_color, samp, uv, layer, 0.0);
    s.m.albedo = col.rgb;
    s.m.alpha = col.a;
    s.m.emission = textureSampleLevel(t_emission, samp, uv, layer, 0.0).rgb * u.light.y;
    s.m.rough = clamp(textureSampleLevel(t_rough, samp, uv, layer, 0.0).r, 0.03, 1.0);
    s.m.metal = textureSampleLevel(t_metal, samp, uv, layer, 0.0).r;
    s.m.spec = textureSampleLevel(t_spec, samp, uv, layer, 0.0).r;

    // Same normal composition as the raster view: baked normal, painted
    // normal, painted height.
    let tx = u.params.z;
    let hl = textureSampleLevel(t_height, samp, uv - vec2<f32>(tx, 0.0), layer, 0.0).r;
    let hr = textureSampleLevel(t_height, samp, uv + vec2<f32>(tx, 0.0), layer, 0.0).r;
    let hd = textureSampleLevel(t_height, samp, uv - vec2<f32>(0.0, tx), layer, 0.0).r;
    let hb = textureSampleLevel(t_height, samp, uv + vec2<f32>(0.0, tx), layer, 0.0).r;
    let k = u.params.y * u.params.w / 256.0;
    let hu = (hr - hl) * 0.5 * k;
    let ht = (hb - hd) * 0.5 * k;
    var base = vec3<f32>(0.0, 0.0, 1.0);
    if (u.params.x > 0.5) {
        let b = textureSampleLevel(t_baked, samp, uv, layer, 0.0).xy;
        base = vec3<f32>(b, sqrt(max(1.0 - dot(b, b), 0.0)));
    }
    let pn = normalize(textureSampleLevel(t_normal, samp, uv, layer, 0.0).xyz * 2.0 - 1.0);
    base = normalize(vec3<f32>(base.xy + pn.xy, base.z * pn.z));
    let ts = normalize(vec3<f32>(base.xy + vec2<f32>(-hu, ht), base.z));
    // MikkTSpace decoding, as in the raster view, including its back-face
    // rule: seen from behind the tangent and handedness flip (Blender negates
    // the whole mapped normal there), so the bitangent keeps its direction.
    var tan_dir = tg.xyz;
    var tan_sign = select(1.0, -1.0, tg.w < 0.0);
    if (back_face) {
        tan_dir = -tan_dir;
        tan_sign = -tan_sign;
    }
    let b_up = -cross(rn, tan_dir) * tan_sign;
    s.n = normalize(tan_dir * ts.x + b_up * ts.y + rn * ts.z);
    let normal_view_dot = dot(s.n, -d);
    if (normal_view_dot <= 0.0) {
        // Move the mapped normal just inside the viewer-facing hemisphere.
        // Divide by the geometric normal's view component so this remains
        // correct at grazing angles, where simply adding ng is insufficient.
        let geometric_view_dot = max(dot(s.ng, -d), 1e-4);
        s.n = normalize(s.n + s.ng * ((0.01 - normal_view_dot) / geometric_view_dot));
    }
    return s;
}

// ---------------------------------------------------------------- integrator

fn trace(o_in: vec3<f32>, d_in: vec3<f32>) -> vec3<f32> {
    var o = o_in;
    var d = d_in;
    var throughput = vec3<f32>(1.0);
    var radiance = vec3<f32>(0.0);
    let max_bounces = i32(u.frame.w);
    let ground_y = u.light.z;
    let ground_on = u.light.w > 0.5;
    // Pdf of the BSDF sample that produced the current ray (0 for camera rays).
    var bsdf_pdf = 0.0;
    // See-through surfaces passed so far (each one doesn't count as a bounce).
    var passed = 0;

    for (var bounce = 0; bounce <= max_bounces; bounce++) {
        let h = intersect(o, d, 1e30, false);
        var tg = MISS;
        if (ground_on && d.y < -1e-6) {
            tg = (ground_y - o.y) / d.y;
            if (tg <= 1e-4) {
                tg = MISS;
            }
        }
        if (h.tri == NONE && tg == MISS) {
            if (bounce == 0) {
                radiance += background(d);
            } else {
                // Weighted against the light sample that could also have found it.
                radiance += throughput * env(d, 0.0) * power_heuristic(bsdf_pdf, env_pdf(d));
            }
            break;
        }

        var s: Surface;
        if (h.tri != NONE && h.t < tg) {
            s = mesh_surface(h, o, d);
            // Stochastic transparency: a surface of opacity a stops a share a
            // of the rays; the rest go on through it, unchanged.
            if (s.m.alpha < 0.999 && passed < 32 && rand() >= s.m.alpha) {
                passed += 1;
                o = s.p + d * 1e-3;
                bounce -= 1;
                continue;
            }
        } else {
            s.p = o + d * tg;
            s.ng = vec3<f32>(0.0, 1.0, 0.0);
            s.n = s.ng;
            // Studio floor that fades into the backdrop with distance.
            let fade = clamp(length(s.p.xz) / 12.0, 0.0, 1.0);
            s.m.albedo = mix(vec3<f32>(0.32), vec3<f32>(0.22), fade);
            s.m.rough = 0.9;
            s.m.metal = 0.0;
            s.m.alpha = 1.0;
            s.m.emission = vec3<f32>(0.0);
            s.m.spec = 0.5;
        }
        // Emissive surfaces light what they're seen from (and, through
        // bounces, everything near them).
        radiance += throughput * s.m.emission;
        let v = -d;
        let origin = s.p + s.ng * 1e-3;

        // Lobe selection, shared by light sampling (for MIS) and BSDF sampling.
        let nv = max(dot(s.n, v), 1e-4);
        let f0 = mix(vec3<f32>(0.08 * s.m.spec), s.m.albedo, s.m.metal);
        let fv = f0 + (1.0 - f0) * pow(1.0 - nv, 5.0);
        let spec_w = dot(fv, vec3<f32>(0.333));
        let diff_w = dot(s.m.albedo * (1.0 - s.m.metal) * (1.0 - fv), vec3<f32>(0.333));
        let p_spec = clamp(spec_w / max(spec_w + diff_w, 1e-4), 0.1, 0.95);
        let a = max(s.m.rough * s.m.rough, 0.002);

        // Next-event estimation: one environment sample with a shadow ray.
        let es = sample_env(rand2());
        let el = es.xyz;
        let enl = dot(s.n, el);
        if (es.w > 0.0 && enl > 0.0 && dot(s.ng, el) > 0.0 && !(ground_on && el.y < 0.0)) {
            if (intersect(origin, el, 1e30, true).tri == NONE) {
                let eh = normalize(v + el);
                let enh = max(dot(s.n, eh), 0.0);
                let pdf_b = p_spec * ggx_d(enh, a) * enh / (4.0 * max(dot(v, eh), 1e-4)) + (1.0 - p_spec) * enl / PI;
                let w = power_heuristic(es.w, pdf_b);
                radiance += throughput * brdf(s.m, s.n, v, el) * enl * env(el, 0.0) * w / es.w;
            }
        }

        // Key light: a delta light, so a shadow ray toward it and no MIS.
        if (u.sun_c.w > 0.5) {
            let sl = normalize(u.sun.xyz);
            let snl = dot(s.n, sl);
            if (snl > 0.0 && dot(s.ng, sl) > 0.0 && intersect(origin, sl, 1e30, true).tri == NONE) {
                radiance += throughput * brdf(s.m, s.n, v, sl) * snl * u.sun_c.rgb * u.sun.w;
            }
        }

        // Sample the next direction: GGX lobe or diffuse lobe.
        var l: vec3<f32>;
        let r = rand2();
        if (rand() < p_spec) {
            // GGX half-vector sampling.
            let phi = 2.0 * PI * r.x;
            let ct = sqrt((1.0 - r.y) / (1.0 + (a * a - 1.0) * r.y));
            let st = sqrt(max(1.0 - ct * ct, 0.0));
            let hv = normalize(onb(s.n) * vec3<f32>(st * cos(phi), st * sin(phi), ct));
            l = reflect(-v, hv);
        } else {
            let rad = sqrt(r.x);
            let phi = 2.0 * PI * r.y;
            l = normalize(onb(s.n) * vec3<f32>(rad * cos(phi), rad * sin(phi), sqrt(max(1.0 - r.x, 0.0))));
        }
        let nl = dot(s.n, l);
        if (nl <= 0.0 || dot(s.ng, l) <= 0.0) {
            break;
        }
        // Combined pdf of both lobes, so the estimator stays unbiased.
        let hv = normalize(v + l);
        let nh = max(dot(s.n, hv), 0.0);
        let pdf_spec = ggx_d(nh, a) * nh / (4.0 * max(dot(v, hv), 1e-4));
        let pdf_diff = nl / PI;
        let pdf = p_spec * pdf_spec + (1.0 - p_spec) * pdf_diff;
        throughput *= brdf(s.m, s.n, v, l) * nl / max(pdf, 1e-6);
        bsdf_pdf = pdf;

        if (bounce >= 2) {
            let survive = clamp(max(throughput.x, max(throughput.y, throughput.z)), 0.05, 0.95);
            if (rand() > survive) {
                break;
            }
            throughput /= survive;
        }
        o = origin;
        d = l;
    }
    return radiance;
}

@compute @workgroup_size(8, 8)
fn cs_trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(u.frame.xy);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }
    let sample = u32(u.frame.z);
    rng = (gid.y * size.x + gid.x) * 9781u + sample * 6271u + 1u;
    next_u();

    let px = vec2<f32>(gid.xy) + rand2();
    let ndc = vec2<f32>(px.x / f32(size.x) * 2.0 - 1.0, 1.0 - px.y / f32(size.y) * 2.0);
    let far = u.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let d = normalize(far.xyz / far.w - u.cam_pos.xyz);
    var c = trace(u.cam_pos.xyz, d);
    // Clamp fireflies from rare caustic paths.
    let lum = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    if (lum > 20.0) {
        c *= 20.0 / lum;
    }

    let idx = gid.y * size.x + gid.x;
    var acc = vec4<f32>(c, 1.0);
    if (sample > 0u) {
        acc += accum[idx];
    }
    accum[idx] = acc;
    textureStore(out_tex, vec2<i32>(gid.xy), vec4<f32>(acc.rgb / acc.w, 1.0));
}
