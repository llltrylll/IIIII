// Rasterizes the mesh in UV space to produce per-texel world position,
// normal and tangent maps, then grows the nearest island texel into its padding.
// This copies one source frame: averaging across nearby islands can invent
// positions and normals that belong to no surface. Alpha decays per pass and
// acts as a distance priority, so later passes prefer the nearest source.
//
// Original island texels stay at alpha 1.0. Dilated texels remain below 0.9
// (polygon-fill coverage threshold) and above 0.5 after all eight passes
// (brush surface threshold).
const DILATE_ALPHA_MAX: f32 = 0.88;
const DILATE_ALPHA_DECAY: f32 = 0.97;
const_assert DILATE_ALPHA_MAX < 0.9;
const_assert DILATE_ALPHA_MAX * DILATE_ALPHA_DECAY * DILATE_ALPHA_DECAY * DILATE_ALPHA_DECAY * DILATE_ALPHA_DECAY * DILATE_ALPHA_DECAY * DILATE_ALPHA_DECAY * DILATE_ALPHA_DECAY * DILATE_ALPHA_DECAY > 0.5;

struct VIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
};

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) wpos: vec3<f32>,
    @location(1) n: vec3<f32>,
    @location(2) t: vec4<f32>,
};

@vertex
fn vs_uv(v: VIn) -> VOut {
    var out: VOut;
    out.clip = vec4<f32>(v.uv.x * 2.0 - 1.0, 1.0 - v.uv.y * 2.0, 0.5, 1.0);
    out.wpos = v.pos;
    out.n = v.normal;
    out.t = v.tangent;
    return out;
}

struct MapsOut {
    @location(0) pos: vec4<f32>,
    @location(1) nrm: vec4<f32>,
    // Tangent (xyz) and bitangent sign (w), encoded 0..1: with the normal,
    // the frame tangent-space normals are in (as the 3D view decodes them).
    @location(2) tan: vec4<f32>,
};

@fragment
fn fs_uv(in: VOut) -> MapsOut {
    var out: MapsOut;
    out.pos = vec4<f32>(in.wpos, 1.0);
    out.nrm = vec4<f32>(normalize(in.n) * 0.5 + 0.5, 1.0);
    out.tan = vec4<f32>(normalize(in.t.xyz) * 0.5 + 0.5, select(0.0, 1.0, in.t.w >= 0.0));
    return out;
}

@group(0) @binding(0) var src_pos: texture_2d<f32>;
@group(0) @binding(1) var src_nrm: texture_2d<f32>;
@group(0) @binding(2) var src_tan: texture_2d<f32>;

@fragment
fn fs_dilate(in: FsOut) -> MapsOut {
    let c = vec2<i32>(in.pos.xy);
    var out: MapsOut;
    let p = textureLoad(src_pos, c, 0);
    if (p.a > 0.5) {
        out.pos = p;
        out.nrm = textureLoad(src_nrm, c, 0);
        out.tan = textureLoad(src_tan, c, 0);
        return out;
    }
    let dims = vec2<i32>(textureDimensions(src_pos));
    var best_alpha = 0.0;
    var best_pos = vec4<f32>(0.0);
    var best_nrm = vec4<f32>(0.0);
    var best_tan = vec4<f32>(0.0);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let q = clamp(c + vec2<i32>(dx, dy), vec2<i32>(0), dims - 1);
            let s = textureLoad(src_pos, q, 0);
            // Original island texels have alpha 1; each padding step lowers
            // it slightly. Selecting the strongest neighbor propagates the
            // closest source without mixing two distinct UV islands.
            if (s.a > best_alpha) {
                best_alpha = s.a;
                best_pos = s;
                best_nrm = textureLoad(src_nrm, q, 0);
                best_tan = textureLoad(src_tan, q, 0);
            }
        }
    }
    if (best_alpha <= 0.5) {
        out.pos = vec4<f32>(0.0);
        out.nrm = vec4<f32>(0.0);
        out.tan = vec4<f32>(0.0);
        return out;
    }
    out.pos = vec4<f32>(best_pos.xyz, min(best_alpha * DILATE_ALPHA_DECAY, DILATE_ALPHA_MAX));
    out.nrm = best_nrm;
    out.tan = best_tan;
    return out;
}
