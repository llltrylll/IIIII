//! The colour picker: a saturation/value square with a hue rail beside it
//! (the layout InstaMAT and Photoshop use) and the extras the Color panel
//! puts around it: an old|new compare swatch, a mix strip between the two
//! brush colours and a harmony row.
//!
//! The square is drawn exactly, not approximated by a grid of cells: a
//! white-to-hue quad with a transparent-to-black quad over it. The first is
//! linear in x and the second linear in y, so the GPU's own interpolation
//! gives `hsv(h, s, v)` at every pixel (see `layered_square_matches_hsv`).
//! Sizes are fixed rather than taken from the free space, so a popup can
//! never grow by feeding its own width back into the layout.

use super::{hsv_to_rgb, rgb_to_hsv, theme};
use eframe::egui;
use egui::epaint::Mesh;
use egui::{Color32, CursorIcon, Pos2, Rect, Sense, Stroke, StrokeKind, pos2, vec2};

pub(crate) type Rgb = [f32; 3];

const SQUARE: egui::Vec2 = vec2(352.0, 216.0);
const RAIL_W: f32 = 22.0;
const GAP: f32 = 8.0;
/// Width of the picker; the strips below it use the same width.
pub(crate) const WIDTH: f32 = SQUARE.x + GAP + RAIL_W;
/// Harmony offsets in degrees: two triadic, two analogous and the complement.
const HARMONY: [f32; 5] = [-120.0, -30.0, 30.0, 120.0, 180.0];
/// Red, yellow, green, cyan, blue, magenta, red: the hue rail's stops. Each
/// segment between two stops is a straight line in sRGB, which is what HSV
/// is, so six quads draw the rail exactly.
const HUE_STOPS: [Rgb; 7] = [
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
];

pub(crate) fn to_color32(c: Rgb) -> Color32 {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    Color32::from_rgb(q(c[0]), q(c[1]), q(c[2]))
}

pub(crate) fn hex_string(c: Rgb) -> String {
    let c = to_color32(c);
    format!("#{:02X}{:02X}{:02X}", c.r(), c.g(), c.b())
}

/// Saturation and value under `pos`, clamped to the square's edge.
fn sv_at(square: Rect, pos: Pos2) -> (f32, f32) {
    let s = ((pos.x - square.left()) / square.width()).clamp(0.0, 1.0);
    let v = (1.0 - (pos.y - square.top()) / square.height()).clamp(0.0, 1.0);
    (s, v)
}

fn sv_pos(square: Rect, s: f32, v: f32) -> Pos2 {
    pos2(square.left() + s * square.width(), square.bottom() - v * square.height())
}

/// Hue under `pos` on the vertical rail: red at the top, red again at the bottom.
fn hue_at(rail: Rect, pos: Pos2) -> f32 {
    ((pos.y - rail.top()) / rail.height()).clamp(0.0, 1.0)
}

fn mix(a: Rgb, b: Rgb, t: f32) -> Rgb {
    std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t)
}

/// `c` with its hue turned by `degrees`; saturation and value stay.
fn rotate_hue(c: Rgb, degrees: f32) -> Rgb {
    let [h, s, v] = rgb_to_hsv(c);
    hsv_to_rgb([(h + degrees / 360.0).rem_euclid(1.0), s, v])
}

fn quad(mesh: &mut Mesh, r: Rect, top_left: Color32, top_right: Color32, bottom_right: Color32, bottom_left: Color32) {
    let i = mesh.vertices.len() as u32;
    mesh.colored_vertex(r.left_top(), top_left);
    mesh.colored_vertex(r.right_top(), top_right);
    mesh.colored_vertex(r.right_bottom(), bottom_right);
    mesh.colored_vertex(r.left_bottom(), bottom_left);
    mesh.add_triangle(i, i + 1, i + 2);
    mesh.add_triangle(i, i + 2, i + 3);
}

fn draw_square(painter: &egui::Painter, square: Rect, hue: f32) {
    let pure = to_color32(hsv_to_rgb([hue, 1.0, 1.0]));
    let mut across = Mesh::default();
    quad(&mut across, square, Color32::WHITE, pure, pure, Color32::WHITE);
    painter.add(egui::Shape::mesh(across));
    let mut down = Mesh::default();
    quad(&mut down, square, Color32::TRANSPARENT, Color32::TRANSPARENT, Color32::BLACK, Color32::BLACK);
    painter.add(egui::Shape::mesh(down));
    painter.rect_stroke(square, 2.0, Stroke::new(1.0, theme::OUTLINE), StrokeKind::Inside);
}

fn draw_rail(painter: &egui::Painter, rail: Rect) {
    let mut mesh = Mesh::default();
    let step = rail.height() / (HUE_STOPS.len() - 1) as f32;
    for (i, pair) in HUE_STOPS.windows(2).enumerate() {
        let top = rail.top() + step * i as f32;
        let r = Rect::from_min_max(pos2(rail.left(), top), pos2(rail.right(), top + step));
        let (a, b) = (to_color32(pair[0]), to_color32(pair[1]));
        quad(&mut mesh, r, a, a, b, b);
    }
    painter.add(egui::Shape::mesh(mesh));
    painter.rect_stroke(rail, 2.0, Stroke::new(1.0, theme::OUTLINE), StrokeKind::Inside);
}

/// The square and its hue rail. The caller keeps `hsv` between frames so the
/// hue and saturation survive a trip through black or white. True if changed.
pub(crate) fn picker(ui: &mut egui::Ui, hsv: &mut Rgb) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = GAP;
        let (square, square_response) = ui.allocate_exact_size(SQUARE, Sense::click_and_drag());
        let (rail, rail_response) = ui.allocate_exact_size(vec2(RAIL_W, SQUARE.y), Sense::click_and_drag());

        // Handle initial clicks and every frame of a primary-button drag.
        let primary_interacting = |r: &egui::Response| {
            r.is_pointer_button_down_on()
                || r.dragged_by(egui::PointerButton::Primary)
                || r.clicked()
        };
        if primary_interacting(&square_response) {
            if let Some(pos) = square_response.interact_pointer_pos() {
                let (s, v) = sv_at(square, pos);
                (hsv[1], hsv[2]) = (s, v);
                changed = true;
            }
        }
        if primary_interacting(&rail_response) {
            if let Some(pos) = rail_response.interact_pointer_pos() {
                hsv[0] = hue_at(rail, pos);
                changed = true;
            }
        }

        let painter = ui.painter();
        draw_square(painter, square, hsv[0]);
        draw_rail(painter, rail);

        let marker = sv_pos(square, hsv[1], hsv[2]);
        painter.circle_filled(marker, 6.5, to_color32(hsv_to_rgb(*hsv)));
        painter.circle_stroke(marker, 6.5, Stroke::new(2.0, Color32::WHITE));
        painter.circle_stroke(marker, 8.0, Stroke::new(1.0, Color32::from_black_alpha(170)));

        let y = rail.top() + hsv[0] * rail.height();
        let handle = Rect::from_center_size(pos2(rail.center().x, y), vec2(RAIL_W + 6.0, 7.0));
        painter.rect_filled(handle, 2.0, to_color32(hsv_to_rgb([hsv[0], 1.0, 1.0])));
        painter.rect_stroke(handle, 2.0, Stroke::new(2.0, Color32::WHITE), StrokeKind::Outside);
        painter.rect_stroke(handle.expand(2.0), 3.0, Stroke::new(1.0, Color32::from_black_alpha(170)), StrokeKind::Outside);

        square_response.on_hover_cursor(CursorIcon::Crosshair)
            .on_hover_text("Click or drag to set saturation and value");
        rail_response.on_hover_cursor(CursorIcon::ResizeVertical)
            .on_hover_text("Click or drag to set hue");
    });
    changed
}

/// A row of swatches. Click picks the color, Shift+click makes it the second color.
fn swatch_strip(ui: &mut egui::Ui, colors: &[Rgb], height: f32) -> Option<(Rgb, bool)> {
    let gap = 3.0;
    let width = (WIDTH - gap * (colors.len() - 1) as f32) / colors.len() as f32;
    let shift = ui.input(|i| i.modifiers.shift);
    let mut picked = None;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = gap;
        for &c in colors {
            let (rect, response) = ui.allocate_exact_size(vec2(width, height), Sense::click());
            ui.painter().rect_filled(rect, 2.0, to_color32(c));
            let edge = if response.hovered() { Color32::WHITE } else { theme::OUTLINE };
            ui.painter().rect_stroke(rect, 2.0, Stroke::new(1.0, edge), StrokeKind::Inside);
            let response = response.on_hover_text(format!("{}\nClick: brush color\nShift+click: second color", hex_string(c)));
            if response.clicked() {
                picked = Some((c, shift));
            }
        }
    });
    picked
}

fn caption(ui: &mut egui::Ui, text: &str) {
    ui.label(egui::RichText::new(text).small().color(theme::TEXT_DIM));
}

/// Continuous blend between the two active colors. Clicking or dragging
/// samples any point along the gradient; Shift targets the secondary color.
pub(crate) fn mix_strip(ui: &mut egui::Ui, a: Rgb, b: Rgb) -> Option<(Rgb, bool)> {
    caption(ui, "Mix the two active colors");
    let (rect, response) = ui.allocate_exact_size(vec2(WIDTH, 22.0), Sense::click_and_drag());
    let mut mesh = Mesh::default();
    quad(&mut mesh, rect, to_color32(a), to_color32(b), to_color32(b), to_color32(a));
    ui.painter().add(egui::Shape::mesh(mesh));
    ui.painter().rect_stroke(rect, 2.0, Stroke::new(1.0, theme::OUTLINE), StrokeKind::Inside);

    let interacting = response.is_pointer_button_down_on()
        || response.dragged_by(egui::PointerButton::Primary)
        || response.clicked();
    let pointer = response.interact_pointer_pos();
    if response.hovered() || interacting {
        if let Some(pos) = pointer {
            let t = ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
            let x = rect.left() + t * rect.width();
            ui.painter().line_segment(
                [pos2(x, rect.top()), pos2(x, rect.bottom())],
                Stroke::new(2.0, Color32::WHITE),
            );
        }
    }
    response.clone().on_hover_text("Click or drag to mix the two colors. Hold Shift to assign the result to the second color.");
    if interacting {
        if let Some(pos) = pointer {
            let t = ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
            let second = ui.input(|i| i.modifiers.shift);
            return Some((mix(a, b, t), second));
        }
    }
    None
}

/// Colors related to `base` by hue: analogous, triadic and the complement.
pub(crate) fn harmony_row(ui: &mut egui::Ui, base: Rgb) -> Option<(Rgb, bool)> {
    caption(ui, "Harmony (analogous, triadic, complement)");
    let colors = HARMONY.map(|degrees| rotate_hue(base, degrees));
    swatch_strip(ui, &colors, 18.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::parse_color_hex;

    fn close(a: Rgb, b: Rgb) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4)
    }

    #[test]
    fn square_position_maps_to_saturation_and_value() {
        let r = Rect::from_min_size(pos2(10.0, 20.0), vec2(200.0, 100.0));
        assert_eq!(sv_at(r, r.left_top()), (0.0, 1.0));
        assert_eq!(sv_at(r, r.right_bottom()), (1.0, 0.0));
        assert_eq!(sv_at(r, r.center()), (0.5, 0.5));
        // Dragging outside the square stays on its edge.
        assert_eq!(sv_at(r, pos2(-500.0, 900.0)), (0.0, 0.0));
        assert_eq!(sv_at(r, pos2(900.0, -500.0)), (1.0, 1.0));
        for (s, v) in [(0.0, 0.0), (1.0, 1.0), (0.25, 0.8), (0.9, 0.1)] {
            let (s2, v2) = sv_at(r, sv_pos(r, s, v));
            assert!((s - s2).abs() < 1e-5 && (v - v2).abs() < 1e-5);
        }
    }

    #[test]
    fn rail_position_maps_to_hue() {
        let r = Rect::from_min_size(pos2(0.0, 40.0), vec2(22.0, 216.0));
        assert_eq!(hue_at(r, pos2(5.0, 40.0)), 0.0);
        assert_eq!(hue_at(r, pos2(5.0, 40.0 + 108.0)), 0.5);
        assert_eq!(hue_at(r, pos2(5.0, 1000.0)), 1.0);
        assert_eq!(hue_at(r, pos2(5.0, -1000.0)), 0.0);
    }

    /// What the two quads give the GPU: base = white -> hue across, then black
    /// at alpha `1 - v` over it. It has to be exactly the HSV colour.
    #[test]
    fn layered_square_matches_hsv() {
        for h in [0.0, 0.08, 0.33, 0.5, 0.71, 0.95] {
            let pure = hsv_to_rgb([h, 1.0, 1.0]);
            for s in [0.0, 0.2, 0.5, 1.0] {
                for v in [0.0, 0.3, 0.75, 1.0] {
                    let across = mix([1.0; 3], pure, s);
                    let shown = across.map(|c| c * v);
                    assert!(close(shown, hsv_to_rgb([h, s, v])), "h {h} s {s} v {v}");
                }
            }
        }
    }

    #[test]
    fn hue_rail_segments_are_straight_lines_in_hsv() {
        for (i, pair) in HUE_STOPS.windows(2).enumerate() {
            assert!(close(pair[0], hsv_to_rgb([i as f32 / 6.0, 1.0, 1.0])), "stop {i}");
            for t in [0.25, 0.5, 0.75] {
                let h = (i as f32 + t) / 6.0;
                assert!(close(mix(pair[0], pair[1], t), hsv_to_rgb([h, 1.0, 1.0])), "segment {i} at {t}");
            }
        }
        assert!(close(HUE_STOPS[6], HUE_STOPS[0]), "the rail wraps around");
    }

    #[test]
    fn mix_runs_from_one_color_to_the_other() {
        let (a, b) = ([0.0, 0.2, 1.0], [1.0, 0.6, 0.0]);
        assert!(close(mix(a, b, 0.0), a));
        assert!(close(mix(a, b, 1.0), b));
        assert!(close(mix(a, b, 0.5), [0.5, 0.4, 0.5]));
    }

    #[test]
    fn harmony_turns_the_hue_and_keeps_the_rest() {
        let red = [1.0, 0.0, 0.0];
        assert!(close(rotate_hue(red, 120.0), [0.0, 1.0, 0.0]));
        assert!(close(rotate_hue(red, -120.0), [0.0, 0.0, 1.0]));
        assert!(close(rotate_hue(red, 180.0), [0.0, 1.0, 1.0]));
        assert!(close(rotate_hue(red, 360.0), red));
        let muted = [0.6, 0.3, 0.3];
        let [_, s, v] = rgb_to_hsv(muted);
        for degrees in HARMONY {
            let [_, s2, v2] = rgb_to_hsv(rotate_hue(muted, degrees));
            assert!((s - s2).abs() < 1e-4 && (v - v2).abs() < 1e-4, "{degrees}");
        }
        // A gray has no hue to turn.
        assert!(close(rotate_hue([0.4; 3], 120.0), [0.4; 3]));
    }

    #[test]
    fn hex_round_trips() {
        assert_eq!(hex_string([1.0, 0.0, 0.5]), "#FF0080");
        assert_eq!(hex_string([2.0, -1.0, 0.0]), "#FF0000");
        for c in ["#000000", "#FFFFFF", "#3D8EFF", "#C8742A"] {
            assert_eq!(hex_string(parse_color_hex(c).unwrap()), c);
        }
    }

    /// Looks at the picker without a window: runs egui headless and fills the
    /// triangles it produces with a small software rasteriser, then writes a PNG.
    /// `cargo test --locked color_picker_preview -- --ignored`
    #[test]
    #[ignore]
    fn color_picker_preview() {
        use egui::epaint::Primitive;
        const W: usize = 430;
        const H: usize = 330;
        let ctx = egui::Context::default();
        let mut hsv = rgb_to_hsv([0.78, 0.45, 0.16]);
        let second = [0.1, 0.55, 0.5];
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(W as f32, H as f32))),
            ..Default::default()
        };
        let mut output = ctx.run_ui(input, |ui| {
            ui.set_width(WIDTH);
            picker(ui, &mut hsv);
            mix_strip(ui, hsv_to_rgb(hsv), second);
            harmony_row(ui, hsv_to_rgb(hsv));
        });
        output.textures_delta.clear();
        let mut pixels = vec![[0.0f32; 4]; W * H];
        for p in pixels.iter_mut() {
            *p = [24.0, 24.0, 28.0, 255.0];
        }
        for clipped in ctx.tessellate(output.shapes, 1.0) {
            let Primitive::Mesh(mesh) = clipped.primitive else { continue };
            for tri in mesh.indices.chunks_exact(3) {
                let v = [mesh.vertices[tri[0] as usize], mesh.vertices[tri[1] as usize], mesh.vertices[tri[2] as usize]];
                // Text and other textured triangles are skipped: only flat/gradient fills.
                if v[0].uv != v[1].uv || v[0].uv != v[2].uv {
                    continue;
                }
                let (a, b, c) = (v[0].pos, v[1].pos, v[2].pos);
                let area = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
                if area.abs() < 1e-6 {
                    continue;
                }
                let (x0, x1) = (a.x.min(b.x).min(c.x).floor().max(0.0) as usize, (a.x.max(b.x).max(c.x).ceil() as usize).min(W));
                let (y0, y1) = (a.y.min(b.y).min(c.y).floor().max(0.0) as usize, (a.y.max(b.y).max(c.y).ceil() as usize).min(H));
                for y in y0..y1 {
                    for x in x0..x1 {
                        let p = pos2(x as f32 + 0.5, y as f32 + 0.5);
                        let w0 = ((b.x - p.x) * (c.y - p.y) - (c.x - p.x) * (b.y - p.y)) / area;
                        let w1 = ((c.x - p.x) * (a.y - p.y) - (a.x - p.x) * (c.y - p.y)) / area;
                        let w2 = 1.0 - w0 - w1;
                        if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                            continue;
                        }
                        let ch = |f: fn(Color32) -> u8| w0 * f(v[0].color) as f32 + w1 * f(v[1].color) as f32 + w2 * f(v[2].color) as f32;
                        let src = [ch(|c| c.r()), ch(|c| c.g()), ch(|c| c.b()), ch(|c| c.a())];
                        // Premultiplied alpha, blended in gamma space like the app.
                        let dst = &mut pixels[y * W + x];
                        let k = 1.0 - src[3] / 255.0;
                        for i in 0..3 {
                            dst[i] = src[i] + dst[i] * k;
                        }
                    }
                }
            }
        }
        let mut img = image::RgbaImage::new(W as u32, H as u32);
        for (i, p) in pixels.iter().enumerate() {
            img.put_pixel((i % W) as u32, (i / W) as u32, image::Rgba([p[0] as u8, p[1] as u8, p[2] as u8, 255]));
        }
        let out = std::env::var("PREVIEW_OUT").unwrap_or_else(|_| "target/color_picker_preview.png".into());
        img.save(&out).unwrap();
        println!("wrote {out}");
    }
}
