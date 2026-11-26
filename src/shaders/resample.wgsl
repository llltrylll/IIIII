// Continuous maps use a bilinear copy between texture sizes. The categorical
// ID map has a separate nearest-texel entry point below.

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@fragment
fn fs_resample(in: FsOut) -> @location(0) vec4<f32> {
    return textureSampleLevel(src, samp, in.uv, 0.0);
}

 
// ID colors are labels, not continuous values. Select the nearest source texel
// so a resize never blends neighboring colors into a synthetic third label.
@fragment
fn fs_resample_nearest(in: FsOut) -> @location(0) vec4<f32> {
    let size = vec2<i32>(textureDimensions(src));
    let pixel = clamp(
        vec2<i32>(floor(in.uv * vec2<f32>(size))),
        vec2<i32>(0),
        size - vec2<i32>(1),
    );
    return textureLoad(src, pixel, 0);
}
