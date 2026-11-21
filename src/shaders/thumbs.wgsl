// Layer thumbnails: shrinking the texture set's maps for a thumbnail-sized
// composite, and turning results into displayable images.

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
// x: what to show (0 color, 1 gray from red, 2 height around mid gray),
// y: samples per axis over an output texel, z: output size in texels.
@group(0) @binding(2) var<uniform> show: vec4<f32>;
// The thumbnail-sized position map: its alpha marks the UV islands.
@group(0) @binding(3) var cover: texture_2d<f32>;

// The nearest texel: positions and normals must not blend across UV island
// borders (the blend of two islands' positions is a point on neither).
@fragment
fn fs_nearest(in: FsOut) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(src));
    let c = vec2<i32>(min(in.uv * dims, dims - 1.0));
    return textureLoad(src, c, 0);
}

// The average over an output texel's footprint, so a big texture shrinks
// without aliasing.
fn footprint(uv: vec2<f32>) -> vec4<f32> {
    let n = max(i32(show.y), 1);
    let texel = 1.0 / max(show.z, 1.0);
    var acc = vec4<f32>(0.0);
    for (var y = 0; y < n; y++) {
        for (var x = 0; x < n; x++) {
            let o = (vec2<f32>(f32(x), f32(y)) + 0.5) / f32(n) - 0.5;
            acc += textureSampleLevel(src, samp, uv + o * texel, 0.0);
        }
    }
    return acc / f32(n * n);
}

@fragment
fn fs_box(in: FsOut) -> @location(0) vec4<f32> {
    return footprint(in.uv);
}

// How much of an output texel lies on the UV islands.
fn coverage(uv: vec2<f32>) -> f32 {
    let n = max(i32(show.y), 1);
    let texel = 1.0 / max(show.z, 1.0);
    let dims = vec2<f32>(textureDimensions(cover));
    var acc = 0.0;
    for (var y = 0; y < n; y++) {
        for (var x = 0; x < n; x++) {
            let o = (vec2<f32>(f32(x), f32(y)) + 0.5) / f32(n) - 0.5;
            let c = vec2<i32>(clamp((uv + o * texel) * dims, vec2<f32>(0.0), dims - 1.0));
            acc += select(0.0, 1.0, textureLoad(cover, c, 0).a > 0.5);
        }
    }
    return acc / f32(n * n);
}

// A thumbnail to display (stored sRGB-encoded, like painted color, with
// premultiplied alpha). Off the UV islands a layer's content is clear (the
// panel's checker shows), a mask dark gray.
fn display_linear_if_needed(c: vec3<f32>) -> vec3<f32> {
    if (show.w < 0.5) { return c; }
    let low = c / 12.92;
    let high = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(low, high, c > vec3<f32>(0.04045));
}

@fragment
fn fs_show(in: FsOut) -> @location(0) vec4<f32> {
    let v = footprint(in.uv);
    let a = coverage(in.uv);
    switch i32(show.x) {
        case 1: { return vec4<f32>(display_linear_if_needed(mix(vec3<f32>(0.16), v.rrr, a)), 1.0); }
        case 2: { return vec4<f32>(display_linear_if_needed(vec3<f32>(clamp(v.r * 0.5 + 0.5, 0.0, 1.0)) * a), a); }
        default: { return vec4<f32>(display_linear_if_needed(v.rgb * a), a); }
    }
}
