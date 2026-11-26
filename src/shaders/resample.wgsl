// Continuous maps use a bilinear copy between texture sizes. The categorical
// ID map has a separate nearest-texel entry point below.

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@fragment
fn fs_resample(in: FsOut) -> @location(0) vec4<f32> {
    return textureSampleLevel(src, samp, in.uv, 0.0);
}
