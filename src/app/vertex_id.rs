//! The Vertex Color ID mask: the mesh's vertex color palette, the mask panel
//! in the Properties, the preview on the 3D model, and reading vertex colors
//! from a file for projects saved without them.

use super::{labeled, slider};
use crate::bake::IdSource;
use crate::engine::{Generator, MAX_ID_COLORS};
use crate::icons;
use crate::mesh::Mesh;
use crate::theme;
use crate::viewport::OverlayVertex;
use eframe::egui;
use egui::{Color32, Sense, Stroke, vec2};
use std::collections::HashMap;

/// Colors listed in the palette (the most used ones).
pub const PALETTE_MAX: usize = 48;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PaletteEntry {
    pub rgb: [u8; 3],
    /// Triangle corners painted this color.
    pub corners: u32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Palette {
    pub entries: Vec<PaletteEntry>,
    /// Distinct colors in the mesh (the list holds at most `PALETTE_MAX`).
    pub distinct: usize,
}

/// The distinct vertex colors of a mesh, most used first.
pub fn palette(colors: &[[[u8; 3]; 3]]) -> Palette {
    let mut counts: HashMap<[u8; 3], u32> = HashMap::new();
    for tri in colors {
        for c in tri {
            *counts.entry(*c).or_default() += 1;
        }
    }
    let distinct = counts.len();
    let mut entries: Vec<PaletteEntry> = counts.into_iter().map(|(rgb, corners)| PaletteEntry { rgb, corners }).collect();
    entries.sort_by(|a, b| b.corners.cmp(&a.corners).then(a.rgb.cmp(&b.rgb)));
    entries.truncate(PALETTE_MAX);
    Palette { entries, distinct }
}

pub fn to_f32(rgb: [u8; 3]) -> [f32; 3] {
    rgb.map(|v| v as f32 / 255.0)
}

fn to_u8(c: [f32; 3]) -> [u8; 3] {
    c.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
}

fn hex(rgb: [u8; 3]) -> String {
    format!("#{:02X}{:02X}{:02X}", rgb[0], rgb[1], rgb[2])
}

/// Which of `ids` is this 8-bit color, if any.
pub fn selected_index(ids: &[[f32; 3]], rgb: [u8; 3]) -> Option<usize> {
    ids.iter().position(|id| to_u8(*id) == rgb)
}

fn smoothstep(lo: f32, hi: f32, x: f32) -> f32 {
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// How much `color` is one of `ids` (1 inside half the tolerance, 0 beyond
/// it): the same function the smart mask input runs on the baked ID map.
pub fn match_value(color: [f32; 3], ids: &[[f32; 3]], tolerance: f32) -> f32 {
    let tol = tolerance.max(1e-3);
    ids.iter()
        .map(|id| {
            let d = ((color[0] - id[0]).powi(2) + (color[1] - id[1]).powi(2) + (color[2] - id[2]).powi(2)).sqrt();
            1.0 - smoothstep(tol * 0.5, tol, d)
        })
        .fold(0.0, f32::max)
}

/// What the 3D model shows in place of its material.
#[derive(Clone, Debug, PartialEq)]
pub enum Shown {
    /// The imported vertex colors themselves.
    Colors,
    /// White where the mask selects, dark elsewhere (ID colors and tolerance bits).
    Mask(Vec<[u32; 3]>, u32),
}

/// The overlay triangles for `shown` (three corners a triangle).
pub fn overlay(mesh: &Mesh, shown: &Shown) -> Vec<OverlayVertex> {
    let Some(colors) = &mesh.tri_vertex_color else { return Vec::new() };
    let ids: Vec<[f32; 3]> = match shown {
        Shown::Colors => Vec::new(),
        Shown::Mask(ids, _) => ids.iter().map(|c| c.map(f32::from_bits)).collect(),
    };
    let tol = match shown {
        Shown::Colors => 0.0,
        Shown::Mask(_, t) => f32::from_bits(*t),
    };
    let mut out = Vec::with_capacity(mesh.indices.len());
    for (tri, corners) in mesh.indices.chunks_exact(3).zip(colors) {
        for k in 0..3 {
            let Some(v) = mesh.vertices.get(tri[k] as usize) else { continue };
            let rgb = match shown {
                Shown::Colors => corners[k],
                Shown::Mask(..) => {
                    let m = match_value(to_f32(corners[k]), &ids, tol);
                    let lit = (34.0 + m * (240.0 - 34.0)).round() as u8;
                    [lit, lit, lit]
                }
            };
            out.push(OverlayVertex { pos: v.pos, color: [rgb[0], rgb[1], rgb[2], 255] });
        }
    }
    out
}

/// Whether the mesh has vertex colors for every triangle.
pub fn has_colors(mesh: &Mesh) -> bool {
    mesh.tri_vertex_color.as_ref().is_some_and(|c| c.len() == mesh.indices.len() / 3)
}

/// The vertex colors of `file` for `mesh`, when it is the same mesh.
pub fn colors_from_file(mesh: &Mesh, file: &std::path::Path) -> Result<Vec<[[u8; 3]; 3]>, String> {
    let other = Mesh::load_file(file)?;
    let same = other.indices.len() == mesh.indices.len()
        && other.vertices.len() == mesh.vertices.len()
        && (0..mesh.vertices.len()).step_by((mesh.vertices.len() / 64).max(1)).all(|i| {
            let (a, b) = (mesh.vertices[i].pos, other.vertices[i].pos);
            (0..3).all(|k| (a[k] - b[k]).abs() < 1e-4)
        });
    if !same {
        return Err(format!(
            "'{}' is not the mesh that is loaded ({} triangles there, {} here). Pick the file this project was made from.",
            file.file_name().and_then(|n| n.to_str()).unwrap_or("file"),
            other.triangle_count(),
            mesh.triangle_count()
        ));
    }
    match other.tri_vertex_color {
        Some(colors) if colors.len() == mesh.triangle_count() => Ok(colors),
        _ => Err("That file has no vertex colors. In Blender: File > Export > FBX > Geometries > Vertex Colors = sRGB (the colors must be a Color Attribute on the mesh).".into()),
    }
}

/// What the ID mask panel keeps between frames.
#[derive(Default)]
pub struct VertexIdState {
    /// The loaded mesh's palette and which mesh (pointer) it was made for.
    pub palette: Option<(usize, Palette)>,
    /// The mask preview on the model is switched off.
    pub preview_off: bool,
    /// The panel was drawn this frame, with these ID colors and tolerance.
    pub panel: Option<(Vec<[f32; 3]>, f32)>,
    /// What the model shows now.
    pub shown: Option<Shown>,
    /// Asked for in the panel, done by the app.
    pub want_import: bool,
    pub want_rebake: bool,
    /// The maps were already baked on their own once for this mesh (so it isn't repeated on every frame).
    pub auto_baked: bool,
}

impl VertexIdState {
    /// The palette of `mesh`, made once per loaded mesh.
    pub fn palette_for(&mut self, mesh: &std::sync::Arc<Mesh>) -> &Palette {
        let key = std::sync::Arc::as_ptr(mesh) as usize;
        if self.palette.as_ref().is_none_or(|(k, _)| *k != key) {
            let p = mesh.tri_vertex_color.as_deref().map(palette).unwrap_or_default();
            self.palette = Some((key, p));
            self.auto_baked = false;
        }
        &self.palette.as_ref().expect("just made").1
    }
}

/// What the panel needs to know about the mesh and its bake.
#[derive(Clone, Copy)]
pub struct Info<'a> {
    pub palette: &'a Palette,
    pub has_colors: bool,
    pub baked_source: IdSource,
}

fn swatch(ui: &mut egui::Ui, rgb: [u8; 3], selected: bool, tip: &str) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(vec2(22.0, 22.0), Sense::click());
    let p = ui.painter();
    p.rect_filled(rect, 3.0, Color32::from_rgb(rgb[0], rgb[1], rgb[2]));
    let stroke = if selected {
        Stroke::new(2.0, Color32::WHITE)
    } else if resp.hovered() {
        Stroke::new(1.0, theme::TEXT)
    } else {
        Stroke::new(1.0, theme::OUTLINE)
    };
    p.rect_stroke(rect, 3.0, stroke, egui::StrokeKind::Inside);
    if selected {
        p.rect_stroke(rect.shrink(2.0), 2.0, Stroke::new(1.0, Color32::BLACK), egui::StrokeKind::Inside);
    }
    resp.on_hover_text(tip).on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// The Vertex Color ID mask settings. Returns whether the generator changed.
pub fn panel(
    ui: &mut egui::Ui,
    g: &mut Generator,
    vs: &mut VertexIdState,
    info: Info,
    pick_target: &mut Option<(usize, Option<usize>)>,
    target: (usize, Option<usize>),
) -> bool {
    let mut dirty = false;
    ui.add_space(4.0);
    ui.label(egui::RichText::new("Vertex colors").strong().color(Color32::WHITE));

    if !info.has_colors {
        ui.label(egui::RichText::new("This mesh has no vertex colors, so there is nothing to select.").color(Color32::from_rgb(0xe8, 0x80, 0x5a)));
        ui.label(
            egui::RichText::new(
                "Projects saved by older versions do not keep them. Pick the FBX/OBJ this project was made from and they are read again; the mesh and layers stay as they are.",
            )
            .small()
            .color(theme::TEXT_DIM),
        );
        if ui.button(theme::with_icon(icons::FOLDER_OPEN, "Import vertex colors from file…")).clicked() {
            vs.want_import = true;
        }
        ui.label(
            egui::RichText::new("In Blender: File > Export > FBX > Geometries > Vertex Colors = sRGB. The colors must be a Color Attribute on the mesh.")
                .small()
                .color(theme::TEXT_DIM),
        );
        return false;
    }

    let mut ids = g.id_colors();
    let before = ids.clone();

    let shown_count = info.palette.entries.len();
    let summary = if info.palette.distinct > shown_count {
        format!("{} colors in the mesh (showing the {} most used). Click to add or remove.", info.palette.distinct, shown_count)
    } else {
        format!("{} colors in the mesh. Click to add or remove.", info.palette.distinct)
    };
    ui.label(egui::RichText::new(summary).small().color(theme::TEXT_DIM));

    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = vec2(4.0, 4.0);
        for e in &info.palette.entries {
            let pos = selected_index(&ids, e.rgb);
            let tip = format!("{} · {} corners{}", hex(e.rgb), e.corners, if pos.is_some() { " · selected" } else { "" });
            if swatch(ui, e.rgb, pos.is_some(), &tip).clicked() {
                match pos {
                    Some(i) if ids.len() > 1 => {
                        ids.remove(i);
                    }
                    Some(_) => {}
                    None if ids.len() < MAX_ID_COLORS => ids.push(to_f32(e.rgb)),
                    None => *ids.last_mut().expect("at least one") = to_f32(e.rgb),
                }
            }
        }
    });

    ui.add_space(4.0);
    ui.label(egui::RichText::new(format!("Selected ({} of {})", ids.len(), MAX_ID_COLORS)).small().color(theme::TEXT_DIM));
    let mut remove = None;
    let many = ids.len() > 1;
    for (i, c) in ids.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            super::color_edit(ui, c);
            ui.label(egui::RichText::new(hex(to_u8(*c))).monospace().color(theme::TEXT));
            if many && ui.small_button("×").on_hover_text("Remove this color").clicked() {
                remove = Some(i);
            }
        });
    }
    if let Some(i) = remove {
        ids.remove(i);
    }
    if ids != before {
        g.set_id_colors(&ids);
        dirty = true;
    }

    dirty |= slider(ui, "ID tolerance", &mut g.id_tolerance, 0.01..=1.0);

    let picking = *pick_target == Some(target);
    ui.horizontal(|ui| {
        if ui
            .selectable_label(picking, theme::with_icon(icons::EYEDROPPER, "Sample from mesh"))
            .on_hover_text("Click a point on the 3D model: its vertex color is added to the selection.")
            .clicked()
        {
            *pick_target = if picking { None } else { Some(target) };
        }
    });
    if picking {
        ui.label(egui::RichText::new("Click the model. The vertex colors are shown on it.").small().color(theme::ACCENT));
    }

    labeled(ui, "Preview on model", |ui| {
        let mut on = !vs.preview_off;
        let changed = ui.checkbox(&mut on, "").on_hover_text("White where this mask selects, dark elsewhere.").changed();
        vs.preview_off = !on;
        changed
    });

    if info.baked_source != IdSource::VertexColors {
        // The mask can't see the colors until the maps are baked from them: do it once on its own.
        if info.has_colors && !vs.auto_baked {
            vs.auto_baked = true;
            vs.want_rebake = true;
        }
        ui.add_space(2.0);
        ui.label(
            egui::RichText::new("The baked ID map was made from mesh parts or materials, so the mask on the texture ignores these colors until it is baked from vertex colors.")
                .small()
                .color(Color32::from_rgb(0xe8, 0x80, 0x5a)),
        );
        if ui.button(theme::with_icon(icons::EYEDROPPER, "Bake maps from vertex colors")).on_hover_text("Sets the ID source to Vertex colors and bakes the mesh maps again.").clicked() {
            vs.want_rebake = true;
        }
    }

    vs.panel = Some((g.id_colors(), g.id_tolerance));
    dirty
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn palette_counts_corners_most_used_first() {
        let red = [255, 0, 0];
        let green = [0, 255, 0];
        let colors = vec![[red, red, green], [green, red, red]];
        let p = palette(&colors);
        assert_eq!(p.distinct, 2);
        assert_eq!(p.entries, vec![PaletteEntry { rgb: red, corners: 4 }, PaletteEntry { rgb: green, corners: 2 }]);
    }

    #[test]
    fn palette_is_capped_but_reports_every_color() {
        let colors: Vec<[[u8; 3]; 3]> = (0..100u8).map(|i| [[i, 0, 0]; 3]).collect();
        let p = palette(&colors);
        assert_eq!(p.distinct, 100);
        assert_eq!(p.entries.len(), PALETTE_MAX);
    }

    #[test]
    fn mask_selects_inside_tolerance_only() {
        let ids = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        assert_eq!(match_value([1.0, 0.0, 0.0], &ids, 0.15), 1.0);
        assert_eq!(match_value([0.0, 0.0, 1.0], &ids, 0.15), 1.0);
        assert_eq!(match_value([0.0, 1.0, 0.0], &ids, 0.15), 0.0);
        let near = match_value([0.9, 0.0, 0.0], &ids, 0.15);
        assert!(near > 0.0 && near < 1.0, "{near}");
    }

    #[test]
    fn selection_matches_8_bit_colors() {
        let ids = [[1.0, 0.0, 0.0], to_f32([10, 20, 30])];
        assert_eq!(selected_index(&ids, [10, 20, 30]), Some(1));
        assert_eq!(selected_index(&ids, [10, 20, 31]), None);
    }

    #[test]
    fn generator_keeps_up_to_eight_id_colors() {
        let mut g = crate::engine::GenKind::IdMask.default_generator();
        assert_eq!(g.id_colors(), vec![[1.0, 0.0, 0.0]]);
        let many: Vec<[f32; 3]> = (0..12).map(|i| [i as f32 / 12.0, 0.0, 0.0]).collect();
        g.set_id_colors(&many);
        assert_eq!(g.id_colors(), many[..MAX_ID_COLORS].to_vec());
        g.set_id_colors(&many[..1]);
        assert_eq!(g.id_colors(), vec![many[0]]);
        assert_eq!(g.id_more, [[0.0; 3]; MAX_ID_COLORS - 1], "unused slots are cleared");
    }
}
