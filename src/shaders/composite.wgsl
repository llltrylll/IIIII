// Layer stack compositing. One pass per visible layer, ping-ponging between
// two accumulation sets: reads the stack below, blends this layer on top.
// Fill layers and smart masks are evaluated procedurally here from the baked
// mesh maps, so they cost no texture memory.

struct LayerU {
    fill_color: vec4<f32>,  // sRGB rgb, opacity
    fill_rmh: vec4<f32>,    // roughness, metallic, height, opacity channel enable
    ch: vec4<f32>,          // per-channel enable (and allocated, for paint layers): color, roughness, metallic, height
    p0: vec4<f32>,          // opacity, blend mode, kind (0 paint, 1 fill), has paint mask
    var0: vec4<f32>,        // color variation, variation scale, roughness variation, height variation
    var1: vec4<f32>,        // variation seed, per-part variation, projection (engine::Projection order), image channel bits (16: gray color, 32: opacity)
    bmin: vec4<f32>,
    bsize: vec4<f32>,
    gw: array<vec4<f32>, 5>, // smart mask input weights (see engine::GEN_INPUTS)
    gc: vec4<f32>,          // custom map inputs: index among the baked custom maps (-1 none)
    gcw: vec4<f32>,         // custom map inputs: weights
    gp0: vec4<f32>,         // level, contrast, scale, noise
    gp1: vec4<f32>,         // invert, seed, has generator, edge width
    gid: vec4<f32>,         // ID color, tolerance
    gx: vec4<f32>,          // baked edge max distance, unused, tri-planar sharpness
    img: vec4<f32>,         // image tiling, rotation (degrees), offset x, offset y
    emis: vec4<f32>,        // fill emission (sRGB rgb), emission channel enable
    ns: vec4<f32>,          // normal image strength, normal channel enable, fill specular, specular channel enable
    gids: array<vec4<f32>, 7>, // more ID mask colors (rgb), 1 in w when in use
};

@group(0) @binding(0) var<uniform> u: LayerU;
@group(0) @binding(1) var prev_color: texture_2d<f32>;
@group(0) @binding(2) var prev_rough: texture_2d<f32>;
@group(0) @binding(3) var prev_metal: texture_2d<f32>;
@group(0) @binding(4) var prev_height: texture_2d<f32>;
@group(0) @binding(5) var l_color: texture_2d<f32>;
@group(0) @binding(6) var l_rough: texture_2d<f32>;
@group(0) @binding(7) var l_metal: texture_2d<f32>;
@group(0) @binding(8) var l_height: texture_2d<f32>;
@group(0) @binding(9) var l_mask: texture_2d<f32>;
@group(0) @binding(10) var pos_map: texture_2d<f32>;
@group(0) @binding(11) var nrm_map: texture_2d<f32>;
@group(0) @binding(12) var map_a: texture_2d<f32>;  // AO, curvature, cavity, thickness
@group(0) @binding(13) var map_b: texture_2d<f32>;  // sky, edge dist, crease dist, part random
@group(0) @binding(14) var map_c: texture_2d<f32>;  // bent normal, part gradient
@group(0) @binding(15) var custom_maps: texture_2d_array<f32>;  // custom map i: channel i % 4 of layer i / 4
@group(0) @binding(16) var map_e: texture_2d<f32>;  // ID color
@group(0) @binding(17) var samp: sampler;
@group(0) @binding(18) var rsamp: sampler;  // tiling, mipmapped: fill images
@group(0) @binding(19) var l_opacity: texture_2d<f32>;  // the layer's opacity (paint) or opacity image (fill)
@group(0) @binding(20) var prev_emission: texture_2d<f32>;
@group(0) @binding(21) var l_emission: texture_2d<f32>;  // the layer's emission (paint) or emission image (fill); a folder merge: the emission below it
@group(0) @binding(22) var prev_normal: texture_2d<f32>;
@group(0) @binding(23) var l_normal: texture_2d<f32>;  // the layer's normals (paint) or normal-map image (fill)
@group(0) @binding(24) var prev_spec: texture_2d<f32>;
@group(0) @binding(25) var l_spec: texture_2d<f32>;  // the layer's specular (paint) or specular image (fill)
@group(0) @binding(26) var tan_map: texture_2d<f32>;  // tangent (xyz) and bitangent sign (w), encoded 0..1

fn weight(i: i32) -> f32 {
    var w = u.gw;
    return w[i / 4][i % 4];
}

fn rot2(q: vec2<f32>, deg: f32) -> vec2<f32> {
    let a = radians(deg);
    let c = cos(a);
    let s = sin(a);
    return vec2<f32>(q.x * c - q.y * s, q.x * s + q.y * c);
}

// A fill image sampled with the layer's projection. The mip level comes from
// how many image texels one output texel covers (`texel_world` is the world
// size of an output texel), since derivatives aren't available here.
fn fill_image(t: texture_2d<f32>, uv: vec2<f32>, p: vec3<f32>, n: vec3<f32>, texel_world: f32) -> vec4<f32> {
    let size = f32(textureDimensions(t).x);
    let tiling = max(u.img.x, 1e-3);
    let mode = i32(u.var1.z + 0.5);
    if (mode == 0) {
        let q = rot2(uv - 0.5, u.img.y) * tiling + 0.5 + u.img.zw;
        let lod = log2(max(tiling * size / f32(textureDimensions(prev_color).x), 1.0));
        return textureSampleLevel(t, rsamp, q, lod);
    }
    if (mode >= 2) {
        // Planar, spherical and cylindrical projections span the mesh's
        // bounds once at tiling 1.
        let ext = max(max(u.bsize.x, u.bsize.y), max(u.bsize.z, 1e-4));
        let d = (p - (u.bmin.xyz + u.bsize.xyz * 0.5)) / ext;
        var c: vec2<f32>;
        switch mode {
            case 2: { c = vec2<f32>(-d.z, -d.y) + 0.5; }
            case 3: { c = vec2<f32>(d.x, d.z) + 0.5; }
            case 4: { c = vec2<f32>(d.x, -d.y) + 0.5; }
            case 5: {
                // Longitude around the vertical axis, latitude from the top.
                let r = max(length(d), 1e-6);
                c = vec2<f32>(atan2(d.x, d.z) / 6.2831853 + 0.5, acos(clamp(d.y / r, -1.0, 1.0)) / 3.1415927);
            }
            default: { c = vec2<f32>(atan2(d.x, d.z) / 6.2831853 + 0.5, 0.5 - d.y); }
        }
        let q = rot2(c - 0.5, u.img.y) * tiling + 0.5 + u.img.zw;
        let lod = log2(max(texel_world * tiling * size / ext, 1.0));
        return textureSampleLevel(t, rsamp, q, lod);
    }
    let lod = log2(max(texel_world * tiling * size, 1.0));
    var w = pow(abs(n), vec3<f32>(max(u.gx.z, 1.0)));
    w /= max(w.x + w.y + w.z, 1e-4);
    let q = p * tiling;
    return textureSampleLevel(t, rsamp, rot2(vec2<f32>(q.z, -q.y), u.img.y) + u.img.zw, lod) * w.x
        + textureSampleLevel(t, rsamp, rot2(q.xz, u.img.y) + u.img.zw, lod) * w.y
        + textureSampleLevel(t, rsamp, rot2(vec2<f32>(q.x, -q.y), u.img.y) + u.img.zw, lod) * w.z;
}

// A normal-map image's slope: the height gradient in its own (u, v)
// coordinates (v down the image), from an OpenGL-convention normal.
fn normal_slope(t: vec4<f32>) -> vec2<f32> {
    let n = t.xyz * 2.0 - 1.0;
    return vec2<f32>(-n.x, n.y) / max(n.z, 0.2);
}

// A tangent-space normal (OpenGL convention) from a height gradient in UV
// coordinates (v down).
fn slope_normal(g: vec2<f32>) -> vec3<f32> {
    return normalize(vec3<f32>(-g.x, g.y, 1.0));
}

// The fill's normal-map image as a tangent-space normal, following the
// layer's projection. Images projected in world space (planar, tri-planar)
// give world slopes that the tangent frame turns into tangent space.
// Spherical and cylindrical projections are approximated as UV-aligned.
fn fill_normal(c: vec2<i32>, uv: vec2<f32>, p: vec3<f32>, n: vec3<f32>, texel_world: f32) -> vec3<f32> {
    let size = f32(textureDimensions(l_normal).x);
    let tiling = max(u.img.x, 1e-3);
    let mode = i32(u.var1.z + 0.5);
    let k = u.ns.x;
    // Sampling at coordinates rotated by `deg` rotates the slope back the other way.
    let deg = u.img.y;
    if (mode == 0 || mode >= 5) {
        return slope_normal(rot2(normal_slope(fill_image(l_normal, uv, p, n, texel_world)), -deg) * k);
    }
    // World-space slope of each axis's image (its u, v as fill_image lays them).
    var grad = vec3<f32>(0.0);
    if (mode == 1) {
        let lod = log2(max(texel_world * tiling * size, 1.0));
        var w = pow(abs(n), vec3<f32>(max(u.gx.z, 1.0)));
        w /= max(w.x + w.y + w.z, 1e-4);
        let q = p * tiling;
        let gx = rot2(normal_slope(textureSampleLevel(l_normal, rsamp, rot2(vec2<f32>(q.z, -q.y), deg) + u.img.zw, lod)), -deg);
        let gy = rot2(normal_slope(textureSampleLevel(l_normal, rsamp, rot2(q.xz, deg) + u.img.zw, lod)), -deg);
        let gz = rot2(normal_slope(textureSampleLevel(l_normal, rsamp, rot2(vec2<f32>(q.x, -q.y), deg) + u.img.zw, lod)), -deg);
        // u, v of each axis: X (z, -y), Y (x, z), Z (x, -y).
        grad = vec3<f32>(0.0, -gx.y, gx.x) * w.x + vec3<f32>(gy.x, 0.0, gy.y) * w.y + vec3<f32>(gz.x, -gz.y, 0.0) * w.z;
    } else {
        let g = rot2(normal_slope(fill_image(l_normal, uv, p, n, texel_world)), -deg);
        // Planar: X (-z, -y), Y (x, z), Z (x, -y).
        switch mode {
            case 2: { grad = vec3<f32>(0.0, -g.y, -g.x); }
            case 3: { grad = vec3<f32>(g.x, 0.0, g.y); }
            default: { grad = vec3<f32>(g.x, -g.y, 0.0); }
        }
    }
    let wn = normalize(n - (grad - n * dot(grad, n)) * k);
    let t4 = textureLoad(tan_map, c, 0);
    let t = normalize(t4.xyz * 2.0 - 1.0);
    let b_up = -cross(n, t) * (t4.w * 2.0 - 1.0);
    return normalize(vec3<f32>(dot(wn, t), dot(wn, b_up), dot(wn, n)));
}

// Detail normal `d` laid on base normal `b` (whiteout blend), both tangent space.
fn combine_normals(b: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    return normalize(vec3<f32>(b.xy + d.xy, b.z * d.z));
}

// World size of one output texel at `c` (on the surface point `p`).
fn texel_size(c: vec2<i32>, p: vec3<f32>) -> f32 {
    let right = textureLoad(pos_map, c + vec2<i32>(1, 0), 0);
    return select(2.0 / f32(textureDimensions(pos_map).x), distance(right.xyz, p), right.a > 0.5);
}

// Smart mask: weighted sum of mesh-map inputs, noise breakup, then levels
// (the inputs and breakup are in wear.wgsl). Sample only the maps an active
// input needs: a Vertex ID mask, for example, reads just the ID-color map.
fn generator(c: vec2<i32>, uv: vec2<f32>) -> f32 {
    var s: MaskSurface;
    let need_a = weight(0) != 0.0 || weight(1) != 0.0 || weight(2) != 0.0
        || weight(3) != 0.0 || weight(4) != 0.0 || weight(7) != 0.0;
    let need_b = weight(0) != 0.0 || weight(5) != 0.0 || weight(10) != 0.0 || weight(11) != 0.0;
    let need_c = weight(8) != 0.0;
    let need_e = weight(14) != 0.0;
    s.a = vec4<f32>(0.0);
    s.b = vec4<f32>(0.0);
    s.c = vec4<f32>(0.0);
    s.e = vec4<f32>(0.0);
    if (need_a) { s.a = textureSampleLevel(map_a, samp, uv, 0.0); }
    if (need_b) { s.b = textureSampleLevel(map_b, samp, uv, 0.0); }
    if (need_c) { s.c = textureSampleLevel(map_c, samp, uv, 0.0); }
    if (need_e) { s.e = textureSampleLevel(map_e, samp, uv, 0.0); }

    let need_position = u.gp0.w != 0.0 || weight(9) != 0.0 || weight(12) != 0.0
        || weight(13) != 0.0 || weight(17) != 0.0 || weight(19) != 0.0;
    let need_normal = weight(6) != 0.0 || weight(15) != 0.0
        || weight(16) != 0.0 || weight(18) != 0.0;
    var p = vec3<f32>(0.0);
    var n = vec3<f32>(0.0, 0.0, 1.0);
    if (need_position) { p = textureLoad(pos_map, c, 0).xyz; }
    if (need_normal) { n = normalize(textureLoad(nrm_map, c, 0).xyz * 2.0 - 1.0); }
    s.p = p;
    s.n = n;
    s.texel_world = select(0.0, texel_size(c, p), weight(18) != 0.0);
    s.bmin = u.bmin.xyz;
    s.bsize = u.bsize.xyz;
    s.emax = u.gx.x;
    s.scale = u.gp0.z;
    s.seed = u.gp1.y;
    s.width = u.gp1.w;
    s.id_color = u.gid.rgb;
    s.id_tolerance = u.gid.w;
    var gids = u.gids;
    for (var k = 0; k < 7; k++) {
        s.id_more[k] = gids[k];
    }

    var v = 0.0;
    for (var i = 0; i < 20; i++) {
        let w = weight(i);
        if (w != 0.0) {
            v += w * mask_input(i, s);
        }
    }
    for (var k = 0; k < 4; k++) {
        let ci = i32(u.gc[k]);
        if (ci >= 0) {
            let t = textureSampleLevel(custom_maps, samp, uv, ci / 4, 0.0);
            v += u.gcw[k] * t[ci % 4];
        }
    }
    return mask_finish(v, s, u.gp0.w, u.gp0.x, u.gp0.y, u.gp1.x > 0.5);
}

struct AccumOut {
    @location(0) color: vec4<f32>,
    @location(1) rough: vec4<f32>,
    @location(2) metal: vec4<f32>,
    @location(3) height: vec4<f32>,
    @location(4) emission: vec4<f32>,
    @location(5) normal: vec4<f32>,
    @location(6) spec: vec4<f32>,
};

// The layer's mask before opacity: the painted mask (or, with mask filters,
// the filtered mask) times the smart mask.
fn layer_mask(c: vec2<i32>, uv: vec2<f32>) -> f32 {
    var mask = 1.0;
    if (u.p0.w > 0.5) {
        mask *= textureLoad(l_mask, c, 0).r;
    }
    if (u.gp1.z > 0.5) {
        mask *= generator(c, uv);
    }
    return mask;
}

// The layer's mask alone, as the first step of its mask filters.
@fragment
fn fs_mask_source(in: FsOut) -> @location(0) vec4<f32> {
    let c = vec2<i32>(in.pos.xy);
    let uv = in.pos.xy / vec2<f32>(textureDimensions(pos_map));
    return vec4<f32>(layer_mask(c, uv), 0.0, 0.0, 1.0);
}

@fragment
fn fs_layer(in: FsOut) -> AccumOut {
    let c = vec2<i32>(in.pos.xy);
    let dims = vec2<f32>(textureDimensions(prev_color));
    let uv = in.pos.xy / dims;

    let pc4 = textureLoad(prev_color, c, 0);
    let pc = pc4.rgb;
    // Opacity accumulates in the color's alpha.
    let po = pc4.a;
    let pr = textureLoad(prev_rough, c, 0).r;
    let pm = textureLoad(prev_metal, c, 0).r;
    let ph = textureLoad(prev_height, c, 0).r;
    let pe = textureLoad(prev_emission, c, 0).rgb;
    let pn = textureLoad(prev_normal, c, 0).xyz * 2.0 - 1.0;
    let ps = textureLoad(prev_spec, c, 0).r;

    var p = vec3<f32>(0.0);
    var n = vec3<f32>(0.0, 0.0, 1.0);
    // Fill projections use surface-space data. Smart masks fetch only the
    // specific position/normal inputs they need inside generator().
    if (u.p0.z > 0.5) {
        p = textureLoad(pos_map, c, 0).xyz;
        n = normalize(textureLoad(nrm_map, c, 0).xyz * 2.0 - 1.0);
    }

    var sc = u.fill_color.rgb;
    var sr = u.fill_rmh.x;
    var sm = u.fill_rmh.y;
    var sh = u.fill_rmh.z;
    var ac = u.ch.x;
    var ar = u.ch.y;
    var am = u.ch.z;
    var ah = u.ch.w;
    var so = u.fill_color.w;
    var ao = u.fill_rmh.w;
    var se = u.emis.rgb;
    var ae = u.emis.w;
    // A fill's normal is flat (no change) unless it has a normal-map image.
    var sn = vec3<f32>(0.0, 0.0, 1.0);
    var an = u.ns.y;
    var ss = u.ns.z;
    var as_ = u.ns.w;

    if (u.p0.z < 0.5) {
        // Paint layer: per-channel (value, alpha).
        if (u.ch.x > 0.5) {
            let t = textureLoad(l_color, c, 0);
            sc = t.rgb;
            ac = t.a;
        }
        if (u.ch.y > 0.5) {
            let t = textureLoad(l_rough, c, 0).rg;
            sr = t.x;
            ar = t.y;
        }
        if (u.ch.z > 0.5) {
            let t = textureLoad(l_metal, c, 0).rg;
            sm = t.x;
            am = t.y;
        }
        if (u.ch.w > 0.5) {
            let t = textureLoad(l_height, c, 0).rg;
            sh = t.x;
            ah = t.y;
        }
        if (u.fill_rmh.w > 0.5) {
            let t = textureLoad(l_opacity, c, 0).rg;
            so = t.x;
            ao = t.y;
        }
        if (u.emis.w > 0.5) {
            let t = textureLoad(l_emission, c, 0);
            se = t.rgb;
            ae = t.a;
        }
        if (u.ns.y > 0.5) {
            let t = textureLoad(l_normal, c, 0);
            sn = normalize(t.xyz * 2.0 - 1.0);
            an = t.a;
        }
        if (u.ns.w > 0.5) {
            let t = textureLoad(l_spec, c, 0).rg;
            ss = t.x;
            as_ = t.y;
        }
    } else {
        // Fill layer: images replace their channels' values, then procedural
        // and per-part variation apply on top.
        let bits = u32(u.var1.w);
        if (bits != 0u) {
            let texel_world = texel_size(c, p);
            if ((bits & 1u) != 0u) {
                let img = fill_image(l_color, uv, p, n, texel_world);
                sc = select(img.rgb, img.rrr, (bits & 16u) != 0u);
            }
            if ((bits & 2u) != 0u) {
                sr = fill_image(l_rough, uv, p, n, texel_world).r;
            }
            if ((bits & 4u) != 0u) {
                sm = fill_image(l_metal, uv, p, n, texel_world).r;
            }
            if ((bits & 8u) != 0u) {
                // Mid grey is the surface; lighter is raised. The layer's
                // height value scales it (0.5: as is; negative inverts).
                sh = (fill_image(l_height, uv, p, n, texel_world).r - 0.5) * 2.0 * u.fill_rmh.z;
            }
            if ((bits & 32u) != 0u) {
                // An opacity (alpha) map, scaled by the layer's opacity value.
                so = fill_image(l_opacity, uv, p, n, texel_world).r * u.fill_color.w;
            }
            if ((bits & 64u) != 0u) {
                // An emission map, tinted by the layer's emission color.
                se = fill_image(l_emission, uv, p, n, texel_world).rgb * u.emis.rgb;
            }
            if ((bits & 128u) != 0u) {
                sn = fill_normal(c, uv, p, n, texel_world);
            }
            if ((bits & 256u) != 0u) {
                // A specular map, scaled so the layer's 0.5 keeps it as is.
                ss = clamp(fill_image(l_spec, uv, p, n, texel_world).r * u.ns.z * 2.0, 0.0, 1.0);
            }
        }
        // Procedural and per-part variation (wear.wgsl, shared with graphs).
        let part = textureSampleLevel(map_b, samp, uv, 0.0).a;
        let vs = u.var0.y;
        sc = fill_variation(sc, 0, p, u.var0.x, vs, u.var1.x, u.var1.y, part);
        sr = fill_variation(vec3<f32>(sr), 1, p, u.var0.z, vs, u.var1.x, u.var1.y, part).x;
        sh = fill_variation(vec3<f32>(sh), 3, p, u.var0.w, vs, u.var1.x, 0.0, part).x;
    }

    let mask = u.p0.x * layer_mask(c, uv);

    let mode = i32(u.p0.y);
    var out: AccumOut;
    let opacity = clamp(mix(po, blend1(mode, po, so), ao * mask), 0.0, 1.0);
    out.color = vec4<f32>(clamp(mix(pc, blend3(mode, pc, sc), ac * mask), vec3<f32>(0.0), vec3<f32>(1.0)), opacity);
    out.rough = vec4<f32>(clamp(mix(pr, blend1(mode, pr, sr), ar * mask), 0.0, 1.0));
    out.metal = vec4<f32>(clamp(mix(pm, blend1(mode, pm, sm), am * mask), 0.0, 1.0));
    out.height = vec4<f32>(mix(ph, blend1(mode, ph, sh), ah * mask));
    out.emission = vec4<f32>(clamp(mix(pe, blend3(mode, pe, se), ae * mask), vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
    // Normals add the layer's detail to what's below, by its coverage.
    let nb = normalize(pn);
    out.normal = vec4<f32>(normalize(mix(nb, combine_normals(nb, sn), an * mask)) * 0.5 + 0.5, 1.0);
    out.spec = vec4<f32>(clamp(mix(ps, blend1(mode, ps, ss), as_ * mask), 0.0, 1.0));
    return out;
}

// Ends a faded or masked folder: the stack with the folder's layers on it
// (prev_*) is faded over the stack below the folder (in the layer slots, l_*)
// by the folder's opacity and mask, like an isolated group.
@fragment
fn fs_folder_merge(in: FsOut) -> AccumOut {
    let c = vec2<i32>(in.pos.xy);
    let uv = in.pos.xy / vec2<f32>(textureDimensions(prev_color));
    var m = u.p0.x;
    if (u.p0.w > 0.5) {
        m *= textureLoad(l_mask, c, 0).r;
    }
    if (u.gp1.z > 0.5) {
        m *= generator(c, uv);
    }
    var out: AccumOut;
    out.color = mix(textureLoad(l_color, c, 0), textureLoad(prev_color, c, 0), m);
    out.rough = vec4<f32>(mix(textureLoad(l_rough, c, 0).r, textureLoad(prev_rough, c, 0).r, m));
    out.metal = vec4<f32>(mix(textureLoad(l_metal, c, 0).r, textureLoad(prev_metal, c, 0).r, m));
    out.height = vec4<f32>(mix(textureLoad(l_height, c, 0).r, textureLoad(prev_height, c, 0).r, m));
    out.emission = mix(textureLoad(l_emission, c, 0), textureLoad(prev_emission, c, 0), m);
    let nl = textureLoad(l_normal, c, 0).xyz * 2.0 - 1.0;
    let np = textureLoad(prev_normal, c, 0).xyz * 2.0 - 1.0;
    out.normal = vec4<f32>(normalize(mix(nl, np, m)) * 0.5 + 0.5, 1.0);
    out.spec = vec4<f32>(mix(textureLoad(l_spec, c, 0).r, textureLoad(prev_spec, c, 0).r, m));
    return out;
}
