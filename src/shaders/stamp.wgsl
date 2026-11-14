// Brush stamping. Runs once per frame over the texture set in UV space and
// accumulates coverage from all stamps laid down this frame into the stroke
// mask with dst = dst + src * (1 - dst), i.e. flow build-up. How a stamp
// reaches a texel depends on the brush projection:
// - Screen: every texel projects its world position to the screen and tests
//   visibility against the viewport depth buffer (paint from the camera).
// - Surface: the stamp is a disc lying on the surface under the cursor, sized
//   in world units, so it doesn't stretch at grazing angles and wraps around
//   edges the camera can't see.
// - UV: the stamp is a disc in texture space around the cursor's UV.

struct StampU {
    view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    view: vec4<f32>,    // width, height, mode (0 = 3D, 1 = 2D), stamp count
    view2d: vec4<f32>,  // zoom (px per uv), pan x, pan y, unused
    brush: vec4<f32>,   // radius px, hardness, flow, mirror axis (0 none, 1 x, 2 y, 3 z; screen projection only)
    clip: vec4<f32>,    // near, far, brush tip (1 = sample alpha_tex), unused
    proj: vec4<f32>,    // projection (0 screen, 1 surface, 2 uv), paint through, cos of the angle limit, unused
};

struct Stamp {
    screen: vec4<f32>,  // screen xy, radius scale, rotation (radians)
    center: vec4<f32>,  // world position under the stamp, world radius (0: missed the mesh)
    normal: vec4<f32>,  // surface normal there, uv radius
    tangent: vec4<f32>, // the tip's x axis on the surface (the screen's right), unused
    uv: vec4<f32>,      // uv under the stamp, flow scale (pen pressure), unused
};

struct StampUvVertex {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
};

struct StampUvOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

@vertex
fn vs_stamp_uv(v: StampUvVertex) -> StampUvOut {
    var out: StampUvOut;
    out.pos = vec4<f32>(v.uv.x * 2.0 - 1.0, 1.0 - v.uv.y * 2.0, 0.5, 1.0);
    out.world_pos = v.pos;
    out.normal = v.normal;
    return out;
}

@group(0) @binding(0) var<uniform> u: StampU;
@group(0) @binding(1) var<storage, read> stamps: array<Stamp>;
@group(0) @binding(2) var pos_map: texture_2d<f32>;
@group(0) @binding(3) var nrm_map: texture_2d<f32>;
@group(0) @binding(4) var depth_map: texture_depth_multisampled_2d;
@group(0) @binding(5) var alpha_tex: texture_2d<f32>;
@group(0) @binding(6) var alpha_samp: sampler;

fn lin_depth(z: f32) -> f32 {
    let n = u.clip.x;
    let f = u.clip.y;
    return n * f / (f - z * (f - n));
}

// Fades paint out on surfaces turned away from the brush by more than the
// angle limit, so it doesn't smear at grazing angles.
fn facing(cos_angle: f32) -> f32 {
    let lim = u.proj.z;
    return smoothstep(lim, min(lim + 0.15, 1.0), cos_angle);
}

// Where a surface point lands on screen, if the 3D camera sees it (z = how
// squarely it faces the camera, 0 when hidden).
fn project(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let clip = u.view_proj * vec4<f32>(p, 1.0);
    if (clip.w <= 0.0) {
        return vec3<f32>(0.0);
    }
    let ndc = clip.xyz / clip.w;
    if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0) {
        return vec3<f32>(0.0);
    }
    let screen = (ndc.xy * vec2<f32>(0.5, -0.5) + 0.5) * u.view.xy;
    let face = facing(dot(n, normalize(u.cam_pos.xyz - p)));
    if (face <= 0.0) {
        return vec3<f32>(0.0);
    }
    if (u.proj.y < 0.5) {
        let sp = clamp(vec2<i32>(screen), vec2<i32>(0), vec2<i32>(u.view.xy) - 1);
        let scene = textureLoad(depth_map, sp, 0);
        if (lin_depth(ndc.z) > lin_depth(scene) * 1.01 + 0.005) {
            return vec3<f32>(0.0);
        }
    }
    return vec3<f32>(screen, face);
}

// Brush falloff at `off` (in brush radii from the stamp center), with the tip
// image turned by the stamp angle.
fn falloff(off: vec2<f32>, angle: f32) -> f32 {
    let d = length(off);
    if (d >= 1.0) {
        return 0.0;
    }
    if (u.clip.z > 0.5) {
        let cs = vec2<f32>(cos(angle), sin(angle));
        let q = vec2<f32>(off.x * cs.x + off.y * cs.y, -off.x * cs.y + off.y * cs.x);
        return textureSampleLevel(alpha_tex, alpha_samp, q * 0.5 + 0.5, 0.0).r;
    }
    return 1.0 - smoothstep(u.brush.y, 1.0, d);
}

fn flow(s: Stamp) -> f32 {
    return u.brush.z * s.uv.z;
}

// Coverage this frame's stamps give a screen position.
fn coverage_screen(screen: vec2<f32>) -> f32 {
    var keep = 1.0;
    let count = u32(u.view.w);
    for (var i = 0u; i < count; i++) {
        let s = stamps[i];
        let fall = falloff((screen - s.screen.xy) / (u.brush.x * max(s.screen.z, 1e-3)), s.screen.w);
        keep *= 1.0 - clamp(fall * flow(s), 0.0, 1.0);
    }
    return 1.0 - keep;
}

// Coverage of discs lying on the surface: the texel's offset from the stamp
// center, measured in the stamp's tangent plane.
fn coverage_surface(p: vec3<f32>, n: vec3<f32>) -> f32 {
    var keep = 1.0;
    let count = u32(u.view.w);
    for (var i = 0u; i < count; i++) {
        let s = stamps[i];
        let r = s.center.w;
        if (r <= 0.0) {
            continue;
        }
        let d = p - s.center.xyz;
        let nc = s.normal.xyz;
        // A slab one radius thick, so paint can't reach the far side of a thin wall.
        if (abs(dot(d, nc)) > r) {
            continue;
        }
        let face = facing(dot(n, nc));
        if (face <= 0.0) {
            continue;
        }
        let x = s.tangent.xyz;
        let y = cross(x, nc);
        let fall = falloff(vec2<f32>(dot(d, x), dot(d, y)) / r, s.screen.w) * face;
        keep *= 1.0 - clamp(fall * flow(s), 0.0, 1.0);
    }
    return 1.0 - keep;
}

// Coverage of discs in texture space.
fn coverage_uv(uv: vec2<f32>) -> f32 {
    var keep = 1.0;
    let count = u32(u.view.w);
    for (var i = 0u; i < count; i++) {
        let s = stamps[i];
        if (s.center.w <= 0.0) {
            continue;
        }
        let fall = falloff((uv - s.uv.xy) / max(s.normal.w, 1e-6), s.screen.w);
        keep *= 1.0 - clamp(fall * flow(s), 0.0, 1.0);
    }
    return 1.0 - keep;
}

// Shared fullscreen path for surface/UV/2D projection. The screen-only
// padding pass deliberately excludes original island texels: those are now
// rasterized per triangle by fs_stamp_uv, so overlapping UV faces don't
// compete for one baked world-position record.
fn stamp_fullscreen(in: FsOut, padding_only: bool) -> vec4<f32> {
    let c = vec2<i32>(in.pos.xy);
    let p = textureLoad(pos_map, c, 0);
    if (p.a < 0.5 || (padding_only && p.a >= 0.9)) {
        discard;
    }
    let dims = vec2<f32>(textureDimensions(pos_map));
    let uv = in.pos.xy / dims;
    var a = 0.0;
    if (u.view.z > 0.5) {
        // 2D view: the screen is the texture.
        a = coverage_screen(u.view.xy * 0.5 + u.view2d.yz + (uv - 0.5) * u.view2d.x);
    } else if (u.proj.x > 1.5) {
        a = coverage_uv(uv);
    } else {
        let n = normalize(textureLoad(nrm_map, c, 0).xyz * 2.0 - 1.0);
        if (u.proj.x > 0.5) {
            a = coverage_surface(p.xyz, n);
        } else {
            let s = project(p.xyz, n);
            if (s.z > 0.0) {
                a = coverage_screen(s.xy) * s.z;
            }
            let axis = i32(u.brush.w);
            if (axis > 0) {
                let flip = select(vec3<f32>(1.0), vec3<f32>(-1.0), vec3<i32>(1, 2, 3) == vec3<i32>(axis));
                let m = project(p.xyz * flip, n * flip);
                if (m.z > 0.0) {
                    a = max(a, coverage_screen(m.xy) * m.z);
                }
            }
        }
    }
    if (a <= 0.0) {
        discard;
    }
    return vec4<f32>(a, 0.0, 0.0, 1.0);
}

@fragment
fn fs_stamp(in: FsOut) -> @location(0) vec4<f32> {
    return stamp_fullscreen(in, false);
}

@fragment
fn fs_stamp_pad(in: FsOut) -> @location(0) vec4<f32> {
    return stamp_fullscreen(in, true);
}

// Orient a geometric face normal toward the camera. This is deliberately
// independent of the imported shading normal: screen visibility is already
// decided by the viewport depth buffer, and malformed vertex normals must not
// make a visible face unpaintable.
fn screen_face_normal(p: vec3<f32>, shading_normal: vec3<f32>, geometric_normal: vec3<f32>) -> vec3<f32> {
    var n = shading_normal;
    if (dot(geometric_normal, geometric_normal) > 1e-16) {
        n = normalize(geometric_normal);
    } else {
        n = normalize(n);
    }
    if (dot(n, u.cam_pos.xyz - p) < 0.0) {
        n = -n;
    }
    return n;
}

// Screen projection evaluates every triangle at its own UV coordinates. The
// viewport depth test selects the visible world-space surface, rather than
// depending on whichever overlapping face won the UV bake.
@fragment
fn fs_stamp_uv(in: StampUvOut) -> @location(0) vec4<f32> {
    let dpdx_world = dpdx(in.world_pos);
    let dpdy_world = dpdy(in.world_pos);
    let geometric_normal = cross(dpdx_world, dpdy_world);
    let n = screen_face_normal(in.world_pos, in.normal, geometric_normal);

    var a = 0.0;
    let s = project(in.world_pos, n);
    if (s.z > 0.0) {
        a = coverage_screen(s.xy) * s.z;
    }

    let axis = i32(u.brush.w);
    if (axis > 0) {
        let flip = select(vec3<f32>(1.0), vec3<f32>(-1.0), vec3<i32>(1, 2, 3) == vec3<i32>(axis));
        let mirrored_pos = in.world_pos * flip;
        let mirrored_normal = screen_face_normal(mirrored_pos, n * flip, geometric_normal * flip);
        let mirrored = project(mirrored_pos, mirrored_normal);
        if (mirrored.z > 0.0) {
            a = max(a, coverage_screen(mirrored.xy) * mirrored.z);
        }
    }
    if (a <= 0.0) {
        discard;
    }
    return vec4<f32>(a, 0.0, 0.0, 1.0);
}
