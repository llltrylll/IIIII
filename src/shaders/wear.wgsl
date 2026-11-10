// Smart mask building blocks, shared by layer compositing (composite.wgsl)
// and the material graph's surface nodes, so a smart material looks the
// same as a layer stack and as a graph. Needs common.wgsl (noise, hashes).

// Wood grain: rings and fibers running along the object's long axis
// (`size`: the mesh's extent), or vertically on faces that look down that
// axis (the ends of planks).
fn wood_grain(p: vec3<f32>, n: vec3<f32>, scale: f32, size: vec3<f32>, seed_in: f32) -> f32 {
    var ax = 0;
    if (size.y > size.x && size.y >= size.z) {
        ax = 1;
    } else if (size.z > size.x) {
        ax = 2;
    }
    if (abs(n[ax]) > 0.7) {
        ax = select(1, 0, ax == 1);
    }
    let a = p[ax];
    let b = p[(ax + 1) % 3];
    let c = p[(ax + 2) % 3];
    let seed = vec3<f32>(seed_in * 17.31);
    let warp = fbm(vec3<f32>(a * 0.35, b * 1.5, c * 1.5) * scale + seed);
    let r = (b + c) * scale * 2.5 + warp * 5.0;
    let ring = smoothstep(0.6, 1.0, abs(fract(r) * 2.0 - 1.0));
    let fiber = fbm(vec3<f32>(a * 0.5, b * 30.0, c * 30.0) * scale + seed + vec3<f32>(3.7));
    return clamp(ring * 0.7 + smoothstep(0.35, 0.75, fiber) * 0.45, 0.0, 1.0);
}

// Scratches on a plane: some grid cells hold one stroke, slightly bent,
// tapered at both ends, each with its own width and depth. `aa` is the
// texel footprint in q units: strokes thinner than a texel fade instead
// of breaking into stair-steps.
fn scratch_plane(q: vec2<f32>, seed: f32, aa: f32) -> f32 {
    let cell = floor(q);
    var m = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let c = cell + vec2<f32>(f32(x), f32(y));
            let h = hash33(vec3<f32>(c, seed));
            if (h.z < 0.55) {
                continue;
            }
            let h2 = hash33(vec3<f32>(c + vec2<f32>(17.0, 31.0), seed + 5.0));
            let ang = h2.x * 3.14159265;
            let dir = vec2<f32>(cos(ang), sin(ang));
            let side = vec2<f32>(-dir.y, dir.x);
            let half_len = 0.35 + 1.25 * h.x * h.x;
            let rel = q - (c + h.xy);
            let t = dot(rel, dir);
            if (abs(t) > half_len) {
                continue;
            }
            let s = t / half_len;
            // A gentle arc, so strokes aren't ruler-straight.
            let bend = (h2.y - 0.5) * 0.25 * half_len;
            let d = abs(dot(rel, side) - bend * s * s);
            let width = 0.006 + 0.03 * h2.z * h2.z;
            let taper = 1.0 - smoothstep(0.3, 1.0, abs(s));
            let w = width * taper;
            // The falloff spans at least a texel, so a thin diagonal stroke
            // always reaches a texel center (no dashes); thin strokes are
            // fainter instead of thinner than a texel.
            let edge = max(aa, 1e-4);
            let line = clamp(1.0 - d / (w + edge), 0.0, 1.0);
            let strength = 0.3 + 0.7 * fract(h2.x * 13.7 + h.y * 7.1);
            m = max(m, line * line * strength * min(w / edge + 0.35, 1.0));
        }
    }
    return m;
}

// Scratches projected from the three axes and blended by the normal, so
// every face gets clean strokes: a set of bold ones and finer hairlines.
fn scratches(p: vec3<f32>, n: vec3<f32>, scale: f32, texel_world: f32, seed_in: f32) -> f32 {
    var w = pow(abs(n), vec3<f32>(4.0));
    w /= max(w.x + w.y + w.z, 1e-4);
    let k = scale * 5.0;
    let q = p * k;
    let seed = seed_in * 3.7;
    let aa = texel_world * k;
    var m = 0.0;
    for (var i = 0; i < 3; i++) {
        if (w[i] < 0.01) {
            continue;
        }
        let a = select(select(q.xy, q.xz, i == 1), q.yz, i == 0);
        let bold = scratch_plane(a, seed + f32(i), aa);
        let fine = scratch_plane(a * 2.7 + vec2<f32>(3.1, 7.7), seed + f32(i) + 11.0, aa * 2.7) * 0.55;
        m += w[i] * max(bold, fine);
    }
    return m;
}

// Grunge for breaking up mask edges: broad blotches with crisp, gritty
// detail, so worn and dirty areas end in irregular borders instead of
// soft gradients.
fn grunge(q: vec3<f32>) -> f32 {
    let broad = fbm(q);
    let grit = fbm(q * 5.3 + vec3<f32>(4.1, 1.3, 7.9));
    let speck = vnoise(q * 23.0 + vec3<f32>(9.4, 2.2, 5.6));
    return broad * 0.6 + grit * 0.3 + speck * 0.1;
}

// What a smart mask reads at one surface point.
struct MaskSurface {
    // Baked maps: AO, curvature, cavity, thickness / sky, edge distance,
    // crease distance, part random / bent normal, part gradient / ID color.
    a: vec4<f32>,
    b: vec4<f32>,
    c: vec4<f32>,
    e: vec4<f32>,
    p: vec3<f32>,
    n: vec3<f32>,
    // World size of one texel there.
    texel_world: f32,
    // Mesh bounds.
    bmin: vec3<f32>,
    bsize: vec3<f32>,
    // The baked edge and crease distances' largest value.
    emax: f32,
    // The mask's noise scale, seed and edge width.
    scale: f32,
    seed: f32,
    width: f32,
    id_color: vec3<f32>,
    id_tolerance: f32,
    // More ID colors (rgb), 1 in w when in use.
    id_more: array<vec4<f32>, 7>,
};

// Smart mask input `i` (`engine::GEN_INPUTS` order).
fn mask_input(i: i32, s: MaskSurface) -> f32 {
    // Thin parts (rods, rims, belts) are rounded all the way around, so edge
    // and convexity inputs would wear them completely; damp them there.
    let thick = mix(0.15, 1.0, smoothstep(0.15, 0.75, s.a.a));
    let curv = (s.a.g - 0.5) * 2.0;
    let width = max(s.width, 1e-4);
    let q = s.p * s.scale + vec3<f32>(s.seed * 17.31);
    switch i {
        case 0: { return (1.0 - smoothstep(0.0, width, s.b.g * s.emax)) * thick; }
        case 1: { return max(curv, 0.0) * thick; }
        case 2: { return max(-curv, 0.0); }
        case 3: { return 1.0 - s.a.r; }
        case 4: { return 1.0 - s.a.b; }
        case 5: { return s.b.r; }
        case 6: { return max(s.n.y, 0.0); }
        case 7: { return 1.0 - s.a.a; }
        case 8: { return s.c.a; }
        case 9: { return (s.p.y - s.bmin.y) / max(s.bsize.y, 1e-4); }
        case 10: { return s.b.a; }
        case 11: { return 1.0 - smoothstep(0.0, width, s.b.b * s.emax); }
        case 12: {
            let k = s.scale;
            let v = fbm(vec3<f32>(s.p.x * k * 7.0, s.p.y * k * 0.45, s.p.z * k * 7.0) + vec3<f32>(s.seed));
            return smoothstep(0.45, 0.8, v);
        }
        case 13: { return fbm(q); }
        case 14: {
            let tol = max(s.id_tolerance, 1e-3);
            var m = 1.0 - smoothstep(tol * 0.5, tol, distance(s.e.rgb, s.id_color));
            var more = s.id_more;
            for (var k = 0; k < 7; k++) {
                if (more[k].w > 0.5) {
                    m = max(m, 1.0 - smoothstep(tol * 0.5, tol, distance(s.e.rgb, more[k].xyz)));
                }
            }
            return m;
        }
        case 15: { return max(-s.n.y, 0.0); }
        case 16: { return wood_grain(s.p, s.n, s.scale, s.bsize, s.seed); }
        case 17: { return cellular(q * 4.0); }
        case 18: { return scratches(s.p, s.n, s.scale, s.texel_world, s.seed); }
        case 19: {
            let d = normalize(s.p - (s.bmin + s.bsize * 0.5));
            return acos(clamp(d.z, -1.0, 1.0)) / 3.14159265;
        }
        default: { return 0.0; }
    }
}

// The weighted sum of inputs `v` broken up by noise, then through levels.
fn mask_finish(v_in: f32, s: MaskSurface, noise_amt: f32, level: f32, contrast_in: f32, invert: bool) -> f32 {
    var v = v_in;
    if (noise_amt != 0.0) {
        // Fine grit breaks the mask's border up; a broad, slow variation on
        // top moves it in and out along long edges, so wear and grime don't
        // follow the mesh maps' contours as an even band.
        let q = s.p * s.scale + vec3<f32>(s.seed * 17.31);
        let broad = fbm(q * 0.35 + vec3<f32>(11.7, 3.1, 5.9)) - 0.5;
        v += ((grunge(q * 2.0) - 0.5) * 1.3 + broad * 2.6) * noise_amt;
    }
    let contrast = max(contrast_in, 0.001);
    var m = smoothstep(level - contrast, level + contrast, v);
    if (invert) {
        m = 1.0 - m;
    }
    return m;
}

// A fill layer's value with its procedural and per-part variation:
// `channel` 0 color (multiplied), 1 roughness, 3 height (added); `part`
// is the part random value.
fn fill_variation(v: vec3<f32>, channel: i32, p: vec3<f32>, amount: f32, scale: f32, seed_in: f32, part_amount: f32, part: f32) -> vec3<f32> {
    let seed = vec3<f32>(seed_in * 13.7);
    var out = v;
    if (channel == 0) {
        if (amount != 0.0) {
            out = out * (1.0 + (fbm(p * scale + seed) - 0.5) * 2.0 * amount);
        }
        if (part_amount != 0.0) {
            out = out * (1.0 + (part - 0.5) * 2.0 * part_amount);
        }
        return clamp(out, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    if (channel == 1) {
        if (amount != 0.0) {
            out = out + (fbm(p * scale * 1.7 + seed + vec3<f32>(5.2)) - 0.5) * amount;
        }
        if (part_amount != 0.0) {
            out = out + (fract(part * 7.31) - 0.5) * part_amount * 0.6;
        }
        return clamp(out, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    if (channel == 3 && amount != 0.0) {
        out = out + (fbm(p * scale * 3.1 + seed + vec3<f32>(11.9)) - 0.5) * amount;
    }
    return out;
}

// Layer blend modes (`engine::Blend` order).
fn blend3(mode: i32, a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    switch mode {
        case 1: { return a * b; }
        case 2: { return 1.0 - (1.0 - a) * (1.0 - b); }
        case 3: { return select(1.0 - 2.0 * (1.0 - a) * (1.0 - b), 2.0 * a * b, a < vec3<f32>(0.5)); }
        case 4: { return a + b; }
        case 5: { return a - b; }
        case 6: { return min(a, b); }
        case 7: { return max(a, b); }
        default: { return b; }
    }
}

fn blend1(mode: i32, a: f32, b: f32) -> f32 {
    return blend3(mode, vec3<f32>(a), vec3<f32>(b)).x;
}
