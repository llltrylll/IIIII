//! Compact HSV picker for the paint brush color.
//!
//! The picker uses a continuous saturation/value field, a hue spectrum and
//! material-oriented swatches. It is intentionally local to the paint brush;
//! other color properties keep egui's standard picker.

use eframe::egui::{self, Color32, CornerRadius, Sense, Stroke, vec2};

const PICKER_WIDTH: f32 = 244.0;
const FIELD_HEIGHT: f32 = 132.0;
const HUE_HEIGHT: f32 = 14.0;

const SWATCHES: [(&str, [u8; 3]); 10] = [
    ("Graphite", [31, 33, 37]),
    ("Gunmetal", [75, 83, 91]),
    ("Steel", [151, 160, 168]),
    ("Ivory", [231, 226, 207]),
    ("Rust", [137,  64, 45]),
    ("Copper", [184, 112, 67]),
    ("Brass", [184, 151, 72]),
    ("Olive", [91, 103, 57]),
    ("Field blue", [56, 91, 125]),
    ("Oxide red", [137, 57, 52]),
];

pub(crate) fn button(ui: &mut egui::Ui, color: &mut [f32; 3]) -> bool {
    let current = to_color32(*color);
    let (rect, response) = ui.allocate_exact_size(vec2(32.0, 21.0), Sense::click());
    let painter = ui.painter();
    painter.rect_filled(rect, CornerRadius::same(2), current);
    painter.rect_stroke(
        rect,
        CornerRadius::same(2),
        Stroke::new(1.0, crate::theme::OUTLINE),
        egui::StrokeKind::Inside,
    );
    let response = response.on_hover_text("Open the HSV color field and material swatches");
    let mut changed = false;

    egui::Popup::menu(&response)
        .width(PICKER_WIDTH)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(|ui| {
            ui.set_min_width(PICKER_WIDTH);
            ui.spacing_mut().item_spacing = vec2(5.0, 6.0);

            let hue_id = response.id.with("brush_color_hue");
            let (mut hue, mut saturation, mut value) = rgb_to_hsv(*color);
            let stored_hue = ui.ctx().data(|data| data.get_temp::<f32>(hue_id));
            if saturation <= 0.001 {
                hue = stored_hue.unwrap_or(0.58);
            } else {
                ui.ctx().data_mut(|data| data.insert_temp(hue_id, hue));
            }

            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("COLOR").strong());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let hex = format!(
                        "#{:02X}{:02X}{:02X}",
                        channel(color[0]),
                        channel(color[1]),
                        channel(color[2])
                    );
                    ui.label(
                        egui::RichText::new(hex)
                            .monospace()
                            .color(crate::theme::TEXT_DIM),
                    );
                });
            });

            let (field, field_response) =
                ui.allocate_exact_size(vec2(ui.available_width(), FIELD_HEIGHT), Sense::click_and_drag());
            paint_sv_field(ui.painter(), field, hue);
            let marker = egui::pos2(
                field.left() + saturation * field.width(),
                field.bottom() - value * field.height(),
            );
            ui.painter().circle_stroke(marker, 6.0, Stroke::new(2.0, Color32::BLACK));
            ui.painter().circle_stroke(marker, 4.5, Stroke::new(1.5, Color32::WHITE));

            if (field_response.clicked() || field_response.dragged())
                && let Some(pointer) = field_response.interact_pointer_pos()
            {
                saturation = ((pointer.x - field.left()) / field.width()).clamp(0.0, 1.0);
                value = (1.0 - (pointer.y - field.top()) / field.height()).clamp(0.0, 1.0);
                *color = hsv_to_rgb(hue, saturation, value);
                changed = true;
            }

            let (spectrum, hue_response) =
                ui.allocate_exact_size(vec2(ui.available_width(), HUE_HEIGHT), Sense::click_and_drag());
            paint_hue_spectrum(ui.painter(), spectrum);
            let hue_x = spectrum.left() + hue * spectrum.width();
            ui.painter().line_segment(
                [
                    egui::pos2(hue_x, spectrum.top() - 2.0),
                    egui::pos2(hue_x, spectrum.bottom() + 2.0),
                ],
                Stroke::new(2.0, Color32::WHITE),
            );
            if (hue_response.clicked() || hue_response.dragged())
                && let Some(pointer) = hue_response.interact_pointer_pos()
            {
                hue = ((pointer.x - spectrum.left()) / spectrum.width()).clamp(0.0, 1.0);
                *color = hsv_to_rgb(hue, saturation, value);
                changed = true;
                ui.ctx().data_mut(|data| data.insert_temp(hue_id, hue));
            }

            ui.label(egui::RichText::new("MATERIAL SWATCHES").small().color(crate::theme::TEXT_DIM));
            ui.horizontal_wrapped(|ui| {
                for (index, (name, rgb)) in SWATCHES.iter().enumerate() {
                    let swatch = Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
                    let (swatch_rect, swatch_response) =
                        ui.allocate_exact_size(vec2(19.0, 19.0), Sense::click());
                    ui.painter().rect_filled(swatch_rect, CornerRadius::same(2), swatch);
                    ui.painter().rect_stroke(
                        swatch_rect,
                        CornerRadius::same(2),
                        Stroke::new(1.0, crate::theme::OUTLINE),
                        egui::StrokeKind::Inside,
                    );
                    if swatch_response.on_hover_text(*name).clicked() {
                        *color = rgb.map(|v| v as f32 / 255.0);
                        let (new_hue, new_saturation, new_value) = rgb_to_hsv(*color);
                        hue = new_hue;
                        saturation = new_saturation;
                        value = new_value;
                        ui.ctx().data_mut(|data| data.insert_temp(hue_id, hue));
                        changed = true;
                    }
                    if index == 4 {
                        ui.add_space(3.0);
                    }
                }
            });

            ui.label(
                egui::RichText::new("Drag the square for saturation/value; drag the strip for hue.")
                    .small()
                    .color(crate::theme::TEXT_DIM),
            );
        });

    changed
}

fn paint_sv_field(painter: &egui::Painter, rect: egui::Rect, hue: f32) {
    painter.rect_filled(rect, CornerRadius::same(2), to_color32(hsv_to_rgb(hue, 1.0, 1.0)));

    paint_gradient_quad(
        painter,
        rect,
        [
            Color32::WHITE,
            Color32::from_rgba_unmultiplied(255, 255, 255, 0),
            Color32::from_rgba_unmultiplied(255, 255, 255, 0),
            Color32::WHITE,
        ],
    );
    paint_gradient_quad(
        painter,
        rect,
        [
            Color32::from_rgba_unmultiplied(0, 0, 0, 0),
            Color32::from_rgba_unmultiplied(0, 0, 0, 0),
            Color32::from_black_alpha(235),
            Color32::from_black_alpha(235),
        ],
    );
    painter.rect_stroke(
        rect,
        CornerRadius::same(2),
        Stroke::new(1.0, crate::theme::OUTLINE),
        egui::StrokeKind::Inside,
    );
}

fn paint_gradient_quad(painter: &egui::Painter, rect: egui::Rect, colors: [Color32; 4]) {
    let mut mesh = egui::epaint::Mesh::default();
    let base = mesh.vertices.len() as u32;
    mesh.colored_vertex(rect.left_top(), colors[0]);
    mesh.colored_vertex(rect.right_top(), colors[1]);
    mesh.colored_vertex(rect.right_bottom(), colors[2]);
    mesh.colored_vertex(rect.left_bottom(), colors[3]);
    mesh.add_triangle(base, base + 1, base + 2);
    mesh.add_triangle(base, base + 2, base + 3);
    painter.add(egui::Shape::mesh(mesh));
}

fn paint_hue_spectrum(painter: &egui::Painter, rect: egui::Rect) {
    const SEGMENTS: usize = 48;
    let mut mesh = egui::epaint::Mesh::default();

    for segment in 0..SEGMENTS {
        let t0 = segment as f32 / SEGMENTS as f32;
        let t1 = (segment + 1) as f32 / SEGMENTS as f32;
        let base = mesh.vertices.len() as u32;
        let c0 = to_color32(hsv_to_rgb(t0, 1.0, 1.0));
        let c1 = to_color32(hsv_to_rgb(t1, 1.0, 1.0));
        mesh.colored_vertex(egui::pos2(rect.left() + rect.width() * t0, rect.top()), c0);
        mesh.colored_vertex(egui::pos2(rect.left() + rect.width() * t1, rect.top()), c1);
        mesh.colored_vertex(egui::pos2(rect.left() + rect.width() * t1, rect.bottom()), c1);
        mesh.colored_vertex(egui::pos2(rect.left() + rect.width() * t0, rect.bottom()), c0);
        mesh.add_triangle(base, base + 1, base + 2);
        mesh.add_triangle(base, base + 2, base + 3);
    }

    painter.add(egui::Shape::mesh(mesh));
    painter.rect_stroke(
        rect,
        CornerRadius::same(2),
        Stroke::new(1.0, crate::theme::OUTLINE),
        egui::StrokeKind::Inside,
    );
}

fn to_color32(rgb: [f32; 3]) -> Color32 {
    Color32::from_rgb(channel(rgb[0]), channel(rgb[1]), channel(rgb[2]))
}

fn channel(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn rgb_to_hsv(rgb: [f32; 3]) -> (f32, f32, f32) {
    let r = rgb[0].clamp(0.0, 1.0);
    let g = rgb[1].clamp(0.0, 1.0);
    let b = rgb[2].clamp(0.0, 1.0);
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let hue = if delta <= f32::EPSILON {
        0.0
    } else if max == r {
        ((g - b) / delta).rem_euclid(6.0) / 6.0
    } else if max == g {
        ((b - r) / delta + 2.0) / 6.0
    } else {
        ((r - g) / delta + 4.0) / 6.0
    };

    let saturation = if max <= f32::EPSILON { 0.0 } else { delta / max };
    (hue, saturation, max)
}

fn hsv_to_rgb(hue: f32, saturation: f32, value: f32) -> [f32; 3] {
    let h = hue.rem_euclid(1.0) * 6.0;
    let s = saturation.clamp(0.0, 1.0);
    let v = value.clamp(0.0, 1.0);
    let c = v * s;
    let x = c * (1.0 - (h.rem_euclid(2.0) - 1.0).abs());
    let m = v - c;

    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    [r + m, g + m, b + m]
}

#[cfg(test)]
mod tests {
    use super::{hsv_to_rgb, rgb_to_hsv};

    #[test]
    fn hsv_primary_colors_match_rgb() {
        let samples = [
            (0.0, [1.0, 0.0, 0.0]),
            (1.0 / 6.0, [1.0, 1.0, 0.0]),
            (1.0 / 3.0, [0.0, 1.0, 0.0]),
            (0.5, [0.0, 1.0, 1.0]),
            (2.0 / 3.0, [0.0, 0.0, 1.0]),
            (5.0 / 6.0, [1.0, 0.0, 1.0]),
        ];
        for (hue, expected) in samples {
            let actual = hsv_to_rgb(hue, 1.0, 1.0);
            for channel in 0..3 {
                assert!((actual[channel] - expected[channel]).abs() < 0.0001);
            }
        }
    }

    #[test]
    fn rgb_hsv_round_trip_keeps_material_colors() {
        let samples = [
            [0.73, 0.21, 0.12],
            [0.18, 0.34, 0.62],
            [0.43, 0.47, 0.29],
            [0.86, 0.79, 0.61],
            [0.08, 0.08, 0.08],
        ];
        for sample in samples {
            let (hue, saturation, value) = rgb_to_hsv(sample);
            let actual = hsv_to_rgb(hue, saturation, value);
            for channel in 0..3 {
                assert!((actual[channel] - sample[channel]).abs() < 0.0001);
            }
        }
    }
}
