// Release builds open without a console window (crashes go to the crash log).
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod bake;
mod bake_cache;
mod brushes;
mod bvh;
mod cage;
mod crash;
mod engine;
mod environment;
mod export;
mod gpu;
mod graph;
mod icons;
mod import;
mod library;
mod mcp;
mod mesh;
mod polyfill;
mod post;
mod presets;
#[cfg(test)]
mod preview_test;
#[cfg(test)]
mod surface_projection_test;
mod raytrace;
mod select;
mod spp;
mod spp_import;
mod startup_trace;
mod textures;
mod theme;
mod undo;
mod viewport;

#[cfg(test)]
mod uvbake_invariants;

use eframe::{egui_wgpu, wgpu};
use std::sync::Arc;

fn main() -> eframe::Result {
    startup_trace::init();
    crash::install();
    let mut options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("Material Painter")
            .with_inner_size([1600.0, 960.0])
            .with_min_inner_size([1100.0, 680.0])
            .with_app_id("material-painter")
            .with_icon(app_icon()),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    // Request the adapter's full limits: high-poly bake sources need large
    // storage buffers, and the compositor binds many textures per pass.
    if let egui_wgpu::WgpuSetup::CreateNew(setup) = &mut options.wgpu_options.wgpu_setup {
        // Apply WGPU_BACKEND to the instance descriptor. Without this, wgpu's
        // default descriptor ignores the launcher environment and may choose Vulkan.
        setup.instance_descriptor.backends = setup.instance_descriptor.backends.with_env();
        setup.power_preference = app::prefs::Preferences::power_preference();
        setup.device_descriptor = Arc::new(|adapter| wgpu::DeviceDescriptor {
            label: Some("material painter device"),
            required_features: adapter.features() & (wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS),
            required_limits: adapter.limits(),
            ..Default::default()
        });
    }
    let result = eframe::run_native(
        "Material Painter",
        options,
        Box::new(|cc| Ok(Box::new(app::PainterApp::new(cc)?))),
    );
    if let Err(error) = &result {
        write_startup_failure(error);
    }
    result
}

/// Persist failures returned by eframe before the native window reaches PainterApp.
/// Release builds have no console, so otherwise adapter/device initialization errors
/// can disappear when the launcher process exits.
fn write_startup_failure(error: &eframe::Error) {
    let Some(dir) = crash::data_dir() else { return };
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let backend = std::env::var("WGPU_BACKEND").unwrap_or_else(|_| "(unset)".into());
    let report = format!(
        "Material Painter {} startup failed\\n\\nWGPU_BACKEND={backend}\\nOS: {} {}\\n\\nError:\\n{error:#?}\\n",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH,
    );
    let _ = std::fs::write(dir.join("startup-error.log"), report);
}

/// The window and taskbar icon (the copper material ball).
fn app_icon() -> eframe::egui::IconData {
    let img = image::load_from_memory(include_bytes!("../assets/ui/icon.png")).expect("bundled icon decodes").into_rgba8();
    eframe::egui::IconData { width: img.width(), height: img.height(), rgba: img.into_raw() }
}
