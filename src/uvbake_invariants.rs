//! Cross-check the Rust UV dilation pass count against its WGSL alpha contract.

const ENGINE: &str = include_str!("engine.rs");
const UVBAKE: &str = include_str!("shaders/uvbake.wgsl");

fn numeric_const(source: &str, declaration: &str) -> f32 {
    let line = source.lines().find(|line| line.trim_start().starts_with(declaration))
        .unwrap_or_else(|| panic!("missing constant declaration: {declaration}"));
    line.split_once('=').expect("constant has a value").1
        .trim().trim_end_matches(';').parse()
        .unwrap_or_else(|_| panic!("invalid numeric constant: {line}"))
}

#[test]
fn uv_dilation_pass_count_matches_shader_alpha_thresholds() {
    let rust_line = ENGINE.lines()
        .find(|line| line.trim_start().starts_with("const DILATE_PASSES: u32 ="))
        .expect("DILATE_PASSES must be declared in engine.rs");
    let passes: u32 = rust_line.split_once('=').expect("pass count has a value").1
        .trim().trim_end_matches(';').parse().expect("pass count must be an integer");
    assert!(passes > 0, "UV dilation must execute at least one pass");

    let max_alpha = numeric_const(UVBAKE, "const DILATE_ALPHA_MAX: f32 =");
    let decay = numeric_const(UVBAKE, "const DILATE_ALPHA_DECAY: f32 =");
    assert!(max_alpha < 0.9, "dilated texels must remain below polygon-fill coverage");
    assert!(max_alpha * decay.powi(passes as i32) > 0.5,
        "dilated alpha must remain above the brush surface threshold after every configured pass");

    let contract = UVBAKE.lines()
        .find(|line| line.trim_start().starts_with("const_assert DILATE_ALPHA_MAX *"))
        .expect("WGSL must const-assert the final dilated alpha threshold");
    assert_eq!(contract.matches("DILATE_ALPHA_DECAY").count(), passes as usize,
        "WGSL alpha const_assert must contain exactly one decay factor per Rust dilation pass");
    assert!(contract.contains("> 0.5"),
        "WGSL must const-assert the brush surface alpha threshold");
}
