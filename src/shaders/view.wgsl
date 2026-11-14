// Viewport rendering: environment backdrop, image-based-lit PBR mesh, 2D
// texture view and UV wireframe. The 3D view writes scene-linear HDR; post.wgsl
// tone maps it.

struct Globals {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    light: vec4<f32>,    // env rotation (radians), exposure, env intensity, radiance max lod
    params: vec4<f32>,   // view mode, height strength, texel size, texture resolution
    view2d: vec4<f32>,   // zoom (px per uv), pan x, pan y, unused
    size: vec4<f32>,     // viewport width, height, baked AO strength, texture set highlight (0..1)
    extra: vec4<f32>,    // use baked normal, has bake, custom map shown (view mode 22), wireframe opacity
    bmin: vec4<f32>,
    bsize: vec4<f32>,
    bg: vec4<f32>,       // background color (linear), background intensity
    bg2: vec4<f32>,      // background blur, background mode (0 environment, 1 color, 2 transparent), emission strength
    light_vp: mat4x4<f32>, // key light shadow projection
    sun: vec4<f32>,      // direction toward the key light, intensity
    sun_c: vec4<f32>,    // key light color (linear), on (1) / off (0)
};

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var t_color: texture_2d<f32>;
@group(0) @binding(2) var t_rough: texture_2d<f32>;
@group(0) @binding(3) var t_metal: texture_2d<f32>;
@group(0) @binding(4) var t_height: texture_2d<f32>;
@group(0) @binding(5) var map_a: texture_2d<f32>;  // AO, curvature, cavity, thickness
@group(0) @binding(6) var map_b: texture_2d<f32>;  // sky, edge dist, crease dist, part random
@group(0) @binding(7) var map_c: texture_2d<f32>;  // bent normal, part gradient
@group(0) @binding(8) var map_d: texture_2d<f32>;  // baked normal xy, height, missed
@group(0) @binding(9) var map_e: texture_2d<f32>;  // ID
@group(0) @binding(19) var custom_maps: texture_2d_array<f32>;  // custom map i: channel i % 4 of layer i / 4
@group(0) @binding(10) var samp: sampler;
@group(0) @binding(11) var t_wnrm: texture_2d<f32>;
@group(0) @binding(12) var t_wpos: texture_2d<f32>;
@group(0) @binding(13) var env_rad: texture_2d<f32>;  // radiance, mip per roughness
@group(0) @binding(14) var env_samp: sampler;
@group(0) @binding(16) var shadow_map: texture_depth_2d;
@group(0) @binding(17) var shadow_samp: sampler_comparison;
@group(0) @binding(15) var<uniform> env_sh: array<vec4<f32>, 9>;
// Parameters of a custom preview shader (a shading graph), 4 floats each.
@group(0) @binding(18) var<uniform> cp: array<vec4<f32>, 128>;
@group(0) @binding(20) var t_emission: texture_2d<f32>;  // sRGB view: reads linear
@group(0) @binding(21) var t_normal: texture_2d<f32>;  // painted normal detail, tangent space, encoded 0..1
@group(0) @binding(22) var t_spec: texture_2d<f32>;  // specular level (0.5: 4% reflectance)

const PI: f32 = 3.14159265;

fn rot_y(d: vec3<f32>, a: f32) -> vec3<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec3<f32>(c * d.x + s * d.z, d.y, -s * d.x + c * d.z);
}

// Prefiltered environment radiance for a reflection of the given roughness.
fn env(d: vec3<f32>, rough: f32) -> vec3<f32> {
    let uv = env_dir_to_uv(rot_y(d, -g.light.x));
    return textureSampleLevel(env_rad, env_samp, uv, clamp(rough, 0.0, 1.0) * g.light.w).rgb * g.light.z;
}

// Diffuse irradiance arriving at a surface facing n.
fn irradiance(n: vec3<f32>) -> vec3<f32> {
    return sh_irradiance(env_sh, rot_y(n, -g.light.x)) * g.light.z;
}

fn aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Karis' analytic approximation of the split-sum environment BRDF.
fn env_brdf(f0: vec3<f32>, rough: f32, nv: f32) -> vec3<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = rough * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * nv)) * r.x + r.y;
    let ab = vec2<f32>(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + ab.y;
}

struct Surface {
    albedo: vec3<f32>,
    // Opacity (the base color's alpha): 1 opaque.
    alpha: f32,
    // Emitted light (linear, at 1 strength).
    emission: vec3<f32>,
    rough: f32,
    metal: f32,
    height: f32,
    // Specular level of non-metals (0.5: 4% reflectance).
    spec: f32,
    // Final tangent-space normal (OpenGL convention): baked normal, painted
    // normal and painted height detail.
    ts: vec3<f32>,
    a: vec4<f32>,
    b: vec4<f32>,
    c: vec4<f32>,
    d: vec4<f32>,
    e: vec4<f32>,
    f: vec4<f32>,
    wnrm: vec3<f32>,
    wpos: vec3<f32>,
};

fn sample_surface(uv: vec2<f32>) -> Surface {
    var s: Surface;
    let col = textureSampleLevel(t_color, samp, uv, 0.0);
    s.albedo = col.rgb;
    s.alpha = col.a;
    s.emission = textureSampleLevel(t_emission, samp, uv, 0.0).rgb;
    s.rough = textureSampleLevel(t_rough, samp, uv, 0.0).r;
    s.metal = textureSampleLevel(t_metal, samp, uv, 0.0).r;
    s.height = textureSampleLevel(t_height, samp, uv, 0.0).r;
    s.spec = textureSampleLevel(t_spec, samp, uv, 0.0).r;
    s.a = textureSampleLevel(map_a, samp, uv, 0.0);
    s.b = textureSampleLevel(map_b, samp, uv, 0.0);
    s.c = textureSampleLevel(map_c, samp, uv, 0.0);
    s.d = textureSampleLevel(map_d, samp, uv, 0.0);
    s.e = textureSampleLevel(map_e, samp, uv, 0.0);
    let ci = max(i32(g.extra.z), 0);
    s.f = vec4<f32>(textureSampleLevel(custom_maps, samp, uv, ci / 4, 0.0)[ci % 4]);
    s.wnrm = textureSampleLevel(t_wnrm, samp, uv, 0.0).xyz * 2.0 - 1.0;
    s.wpos = textureSampleLevel(t_wpos, samp, uv, 0.0).xyz;
    let tx = g.params.z;
    let hl = textureSampleLevel(t_height, samp, uv - vec2<f32>(tx, 0.0), 0.0).r;
    let hr = textureSampleLevel(t_height, samp, uv + vec2<f32>(tx, 0.0), 0.0).r;
    let hd = textureSampleLevel(t_height, samp, uv - vec2<f32>(0.0, tx), 0.0).r;
    let hb = textureSampleLevel(t_height, samp, uv + vec2<f32>(0.0, tx), 0.0).r;
    // Height slope per texel, scaled so the look is resolution-independent.
    let k = g.params.y * g.params.w / 256.0;
    let hu = (hr - hl) * 0.5 * k;
    let ht = (hb - hd) * 0.5 * k;
    var base = vec3<f32>(0.0, 0.0, 1.0);
    if (g.extra.x > 0.5) {
        let bx = s.d.x;
        let by = s.d.y;
        base = vec3<f32>(bx, by, sqrt(max(1.0 - bx * bx - by * by, 0.0)));
    }
    // Whiteout blends: the painted normal on the baked one, then height detail.
    let pn = normalize(textureSampleLevel(t_normal, samp, uv, 0.0).xyz * 2.0 - 1.0);
    base = normalize(vec3<f32>(base.xy + pn.xy, base.z * pn.z));
    s.ts = normalize(vec3<f32>(base.xy + vec2<f32>(-hu, ht), base.z));
    return s;
}

fn hue(h: f32) -> vec3<f32> {
    let k = vec3<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0);
    let p = abs(fract(vec3<f32>(h) + k) * 6.0 - 3.0);
    return clamp(p - 1.0, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn lin(v: f32) -> vec3<f32> {
    return vec3<f32>(v * v);
}

// How much of the key light reaches world point p (geometric normal gn):
// 3x3 filtered lookups in the shadow map, offset along the normal so
// surfaces don't shadow themselves.
fn key_shadow(p: vec3<f32>, gn: vec3<f32>) -> f32 {
    let size = f32(textureDimensions(shadow_map).x);
    let l = normalize(g.sun.xyz);
    let slope = 1.0 - clamp(dot(gn, l), 0.0, 1.0);
    let q = g.light_vp * vec4<f32>(p + gn * (0.004 + 0.01 * slope), 1.0);
    let ndc = q.xyz / q.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) || ndc.z > 1.0) {
        return 1.0;
    }
    var lit = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let o = vec2<f32>(f32(x), f32(y)) * 1.5 / size;
            lit += textureSampleCompareLevel(shadow_map, shadow_samp, uv + o, ndc.z);
        }
    }
    return lit / 9.0;
}

// Key light: GGX specular and Lambert diffuse from a directional light.
fn key_light(s: Surface, n: vec3<f32>, v: vec3<f32>, shadow: f32) -> vec3<f32> {
    if (g.sun_c.w < 0.5 || shadow <= 0.0) {
        return vec3<f32>(0.0);
    }
    let l = normalize(g.sun.xyz);
    let nl = dot(n, l);
    if (nl <= 0.0) {
        return vec3<f32>(0.0);
    }
    let nv = max(dot(n, v), 1e-4);
    let h = normalize(l + v);
    let nh = max(dot(n, h), 0.0);
    let vh = max(dot(v, h), 0.0);
    let rough = clamp(s.rough, 0.03, 1.0);
    let a = rough * rough;
    let a2 = a * a;
    let dd = nh * nh * (a2 - 1.0) + 1.0;
    let d = a2 / (PI * dd * dd);
    let k = a * 0.5;
    let vis = 1.0 / ((nv * (1.0 - k) + k) * (nl * (1.0 - k) + k) * 4.0);
    let f0 = mix(vec3<f32>(0.08 * s.spec), s.albedo, s.metal);
    let f = f0 + (1.0 - f0) * pow(1.0 - vh, 5.0);
    let diffuse = s.albedo * (1.0 - s.metal) * (1.0 - f) / PI;
    return (diffuse + d * vis * f) * g.sun_c.rgb * g.sun.w * nl * shadow;
}

fn shade(s: Surface, n: vec3<f32>, v: vec3<f32>, shadow: f32) -> vec3<f32> {
    let mode = i32(g.params.x);
    switch mode {
        case 1: { return s.albedo; }
        case 2: { return lin(s.rough); }
        case 3: { return lin(s.metal); }
        case 4: { return lin(clamp(s.height * 0.5 + 0.5, 0.0, 1.0)); }
        case 23: { return lin(s.alpha); }
        case 24: { return s.emission; }
        case 25: { return lin(s.spec); }
        case 5: { let c = s.ts * 0.5 + 0.5; return c * c; }
        case 6: { return lin(s.a.r); }
        case 7: { return lin(s.a.g); }
        case 8: { return lin(s.a.b); }
        case 9: { return lin(s.a.a); }
        case 10: { return lin(s.b.r); }
        case 11: { return lin(s.b.g); }
        case 12: { return lin(s.b.b); }
        case 13: { let c = hue(s.b.a) * 0.8 + 0.1; return c * c; }
        case 14: { return lin(s.c.a); }
        case 15: { return s.c.rgb * s.c.rgb; }
        case 16: {
            let c = vec3<f32>(s.d.x, s.d.y, sqrt(max(1.0 - s.d.x * s.d.x - s.d.y * s.d.y, 0.0))) * 0.5 + 0.5;
            return c * c;
        }
        case 17: { return lin(s.d.z); }
        case 18: { return s.e.rgb * s.e.rgb; }
        case 19: { let c = s.wnrm * 0.5 + 0.5; return c * c; }
        case 20: { let c = (s.wpos - g.bmin.xyz) / max(g.bsize.xyz, vec3<f32>(1e-4)); return c * c; }
        case 21: {
            let base = lin(0.25 + 0.5 * s.a.r);
            return mix(base, vec3<f32>(1.0, 0.02, 0.02), s.d.w);
        }
        case 22: { return lin(s.f.r); }
        default: {}
    }
    return shade_pbr(s, n, v, shadow);
}

// The built-in physically based look: image-based light plus the key light.
fn shade_pbr(s: Surface, n: vec3<f32>, v: vec3<f32>, shadow: f32) -> vec3<f32> {
    let rough = clamp(s.rough, 0.03, 1.0);
    let a = rough * rough;
    let f0 = mix(vec3<f32>(0.08 * s.spec), s.albedo, s.metal);
    let diffuse_c = s.albedo * (1.0 - s.metal);
    let nv = max(dot(n, v), 1e-4);

    // All light comes from the environment: SH irradiance for diffuse, the
    // roughness-prefiltered radiance for specular. The baked AO reaches far
    // (it's sky occlusion), so at full strength it darkens whole crevices,
    // not just contact lines; the view applies it at the user's strength.
    let ao = mix(1.0, s.a.r, clamp(g.size.z, 0.0, 1.0));
    let spec = env_brdf(f0, rough, nv);
    var col = diffuse_c * (1.0 - spec) * irradiance(n) / PI * ao;
    let r = reflect(-v, n);
    let spec_ao = clamp(pow(nv + ao, exp2(-16.0 * a - 1.0)) - 1.0 + ao, 0.0, 1.0);
    col += env(r, rough) * spec * spec_ao;
    // Baked AO darkens the key light only a little: it's mostly sky occlusion.
    col += key_light(s, n, v, shadow) * mix(1.0, ao, 0.35);
    col += s.emission * g.bg2.z;
    return col;
}

// ---------------------------------------------------------------- backdrop

@fragment
fn fs_sky(in: FsOut) -> @location(0) vec4<f32> {
    let ndc = vec2<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    let far = g.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let d = normalize(far.xyz / far.w - g.cam_pos.xyz);
    if (g.bg2.y > 0.5) {
        return vec4<f32>(g.bg.rgb, 1.0);
    }
    return vec4<f32>(env(d, g.bg2.x) * g.bg.w, 1.0);
}

// ---------------------------------------------------------------- mesh

struct VIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
};

struct MeshOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) wpos: vec3<f32>,
    @location(1) n: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) t: vec4<f32>,
};

@vertex
fn vs_mesh(v: VIn) -> MeshOut {
    var out: MeshOut;
    out.clip = g.view_proj * vec4<f32>(v.pos, 1.0);
    out.wpos = v.pos;
    out.n = v.normal;
    out.uv = v.uv;
    out.t = v.tangent;
    return out;
}

// Everything a shader sees at one point of the mesh; custom preview shaders
// (shading graphs) read their inputs from it.
struct Shade {
    s: Surface,
    // Shading normal (baked normal and painted height applied), the
    // geometric normal, and the direction toward the camera.
    n: vec3<f32>,
    gn: vec3<f32>,
    v: vec3<f32>,
    // Toward the key light, and its color times intensity (zero when off).
    l: vec3<f32>,
    light: vec3<f32>,
    shadow: f32,
    uv: vec2<f32>,
    wpos: vec3<f32>,
};

fn mesh_shade_inputs(in: MeshOut, front: bool) -> Shade {
    var c: Shade;
    c.s = sample_surface(in.uv);

    // Face-forward the true geometric normal toward the camera. The raster
    // path previously called the interpolated shading normal "geometric",
    // while the path tracer correctly derives it from triangle positions.
    var gn = cross(dpdx(in.wpos), dpdy(in.wpos));
    if (dot(gn, gn) > 1e-12) {
        gn = normalize(gn);
    } else {
        gn = normalize(in.n);
    }
    c.v = normalize(g.cam_pos.xyz - in.wpos);
    if (dot(gn, c.v) < 0.0) {
        gn = -gn;
    }

    // Match the path tracer: preserve authored shading normals and normal-map
    // tangent-space detail, but orient the shading frame consistently with
    // the actual face. This prevents inverted imported normals from shading
    // the raster viewport differently from the path-traced viewport.
    var rn = in.n;
    let missing_shading_normal = dot(rn, rn) <= 1e-12;
    if (missing_shading_normal) {
        // Malformed or unavailable imported normals must not leave the
        // raster shading frame undefined; use the face-oriented geometric
        // normal as the conservative fallback.
        rn = gn;
    } else {
        if (!front) {
            rn = -rn;
        }
        if (dot(rn, gn) < 0.0) {
            rn = -rn;
        }
    }
    c.gn = gn;

    // Preserve a valid imported tangent frame. If its normal or tangent is
    // missing, reconstruct the tangent basis from the surface's screen-space
    // derivatives. This follows the actual UV orientation; an arbitrary world
    // axis would rotate tangent-space normal detail on malformed imports.
    var tangent = in.t.xyz;
    var handedness = in.t.w;
    if (missing_shading_normal || dot(tangent, tangent) <= 1e-12) {
        let dp_dx = dpdx(in.wpos);
        let dp_dy = dpdy(in.wpos);
        let duv_dx = dpdx(in.uv);
        let duv_dy = dpdy(in.uv);
        let uv_det = duv_dx.x * duv_dy.y - duv_dx.y * duv_dy.x;
        var reconstructed = vec3<f32>(0.0);
        var reconstructed_bitangent = vec3<f32>(0.0);
        if (abs(uv_det) > 1e-10) {
            reconstructed = (dp_dx * duv_dy.y - dp_dy * duv_dx.y) / uv_det;
            reconstructed_bitangent = (dp_dy * duv_dx.x - dp_dx * duv_dy.x) / uv_det;
        }
        reconstructed = reconstructed - rn * dot(rn, reconstructed);
        if (dot(reconstructed, reconstructed) > 1e-12) {
            tangent = normalize(reconstructed);
            handedness = select(1.0, -1.0, dot(reconstructed_bitangent, -cross(rn, tangent)) < 0.0);
        } else {
            // Only degenerate UV derivatives need a deterministic fallback.
            let axis = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(rn.x) > 0.9);
            tangent = normalize(cross(axis, rn));
            handedness = 1.0;
        }
    }

    // MikkTSpace decoding: the interpolated normal and tangent as they are
    // (not normalized or made orthogonal), the bitangent from their cross
    // product, normalized only after combining. The baker inverts exactly
    // this, and so do Unity, Unreal, Godot and Blender.
    let b_up = -cross(rn, tangent) * handedness;
    c.n = normalize(tangent * c.s.ts.x + b_up * c.s.ts.y + rn * c.s.ts.z);
    c.l = normalize(g.sun.xyz);
    c.light = g.sun_c.rgb * g.sun.w * g.sun_c.w;
    c.shadow = 1.0;
    if (g.sun_c.w > 0.5) {
        c.shadow = key_shadow(in.wpos, c.gn);
    }
    c.uv = in.uv;
    c.wpos = in.wpos;
    return c;
}

// A highlighted texture set: tinted toward a bright accent, so it reads
// against any material and lighting.
fn highlighted(col: vec3<f32>) -> vec3<f32> {
    let h = clamp(g.size.w, 0.0, 1.0);
    let accent = vec3<f32>(0.1, 0.55, 1.0);
    // The surface's brightness in the accent's hue, so shading and detail stay readable.
    let l = dot(col, vec3<f32>(0.2126, 0.7152, 0.0722));
    return mix(col, accent * (l * 1.3 + 0.03), h * 0.75);
}

// Opacity from which a surface draws in the opaque pass. Below it, the
// transparent pass blends it over what's behind (still writing depth, so
// brushes and picking find see-through surfaces).
const OPAQUE: f32 = 0.999;

// The surface's opacity as the view shows it: only the Material view is see-through.
fn view_alpha(s: Surface) -> f32 {
    return select(1.0, s.alpha, i32(g.params.x) == 0);
}

@fragment
fn fs_mesh(in: MeshOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let c = mesh_shade_inputs(in, front);
    if (view_alpha(c.s) < OPAQUE) {
        discard;
    }
    return vec4<f32>(highlighted(shade(c.s, c.n, c.v, c.shadow)), 1.0);
}

// The see-through part of the mesh, blended (after everything opaque).
@fragment
fn fs_mesh_t(in: MeshOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let c = mesh_shade_inputs(in, front);
    let a = view_alpha(c.s);
    if (a >= OPAQUE) {
        discard;
    }
    return vec4<f32>(highlighted(shade(c.s, c.n, c.v, c.shadow)), a);
}

// ---------------------------------------------------------------- 2D view

struct QuadOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

fn uv_to_clip(uv: vec2<f32>) -> vec4<f32> {
    let screen = g.size.xy * 0.5 + g.view2d.yz + (uv - 0.5) * g.view2d.x;
    let ndc = screen / g.size.xy * 2.0 - 1.0;
    return vec4<f32>(ndc.x, -ndc.y, 0.0, 1.0);
}

@vertex
fn vs_quad(@builtin(vertex_index) vi: u32) -> QuadOut {
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0)
    );
    var out: QuadOut;
    out.uv = corners[vi];
    out.clip = uv_to_clip(out.uv);
    return out;
}

@fragment
fn fs_quad(in: QuadOut) -> @location(0) vec4<f32> {
    let s = sample_surface(in.uv);
    let pn = s.ts;
    // Lit by the environment with the camera looking straight at the texture.
    // The 2D view has no post pass, so it tone maps here.
    // The 2D view's normals are in texture space: no key light there.
    var lit = shade(s, pn, vec3<f32>(0.0, 0.0, 1.0), 0.0);
    if (i32(g.params.x) == 0) {
        lit = aces(lit * g.light.y);
        // See-through texels over a checkerboard.
        let cell = vec2<i32>(floor(in.clip.xy / 8.0));
        let checker = select(0.18, 0.32, (cell.x + cell.y) % 2 == 0);
        lit = mix(vec3<f32>(checker), lit, s.alpha);
    } else if (i32(g.params.x) == 1) {
        // Base Color is a flat texture view, but it still needs to preview alpha.
        let cell = vec2<i32>(floor(in.clip.xy / 8.0));
        let checker = select(0.18, 0.32, (cell.x + cell.y) % 2 == 0);
        lit = mix(vec3<f32>(checker), s.albedo, s.alpha);
    }
    return vec4<f32>(lit, 1.0);
}

@vertex
fn vs_wire(v: VIn) -> QuadOut {
    var out: QuadOut;
    out.uv = v.uv;
    out.clip = uv_to_clip(v.uv);
    return out;
}

// UV wireframe (opacity in extra.w).
@fragment
fn fs_wire(in: QuadOut) -> @location(0) vec4<f32> {
    return vec4<f32>(0.92, 0.95, 1.0, g.extra.w);
}

// UV seams: island borders, bright orange.
@fragment
fn fs_seam(in: QuadOut) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.62, 0.15, 0.95);
}

// UV seams over the shaded 3D view, a little nearer the eye than the wireframe.
@vertex
fn vs_seam3d(v: VIn) -> OverlayOut {
    var out: OverlayOut;
    out.clip = toward_eye(v.pos, 0.002);
    out.color = vec4<f32>(1.0, 0.62, 0.15, 0.95);
    return out;
}

// ------------------------------------------------------------------ shadow map

@vertex
fn vs_shadow(v: VIn) -> @builtin(position) vec4<f32> {
    return g.light_vp * vec4<f32>(v.pos, 1.0);
}

// ------------------------------------------------------------------ 3D overlays

struct OverlayOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

// Pulls a surface point a little toward the camera so lines and highlights
// lying on the mesh win the depth test against it.
fn toward_eye(p: vec3<f32>, k: f32) -> vec4<f32> {
    return g.view_proj * vec4<f32>(p + (g.cam_pos.xyz - p) * k, 1.0);
}

// Mesh wireframe over the shaded 3D view (opacity in extra.w).
@vertex
fn vs_wire3d(v: VIn) -> OverlayOut {
    var out: OverlayOut;
    out.clip = toward_eye(v.pos, 0.0015);
    out.color = vec4<f32>(0.75, 0.85, 1.0, g.extra.w);
    return out;
}

struct OverlayIn {
    @location(0) pos: vec3<f32>,
    // sRGB color, straight alpha.
    @location(1) color: vec4<f32>,
};

// Colored triangles and lines: bake cage, polygon fill selection.
@vertex
fn vs_overlay(v: OverlayIn) -> OverlayOut {
    var out: OverlayOut;
    out.clip = toward_eye(v.pos, 0.001);
    out.color = vec4<f32>(pow(v.color.rgb, vec3<f32>(2.2)), v.color.a);
    return out;
}

@fragment
fn fs_overlay(in: OverlayOut) -> @location(0) vec4<f32> {
    return in.color;
}
