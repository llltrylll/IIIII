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
pub const PALETTE_MAX: usize = crate::bake::MAX_VERTEX_COLOR_ID_REGIONS;

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
    let order = |a: &PaletteEntry, b: &PaletteEntry| b.corners.cmp(&a.corners).then(a.rgb.cmp(&b.rgb));
    if entries.len() > PALETTE_MAX {
        entries.select_nth_unstable_by(PALETTE_MAX, order);
        entries.truncate(PALETTE_MAX);
    }
    entries.sort_unstable_by(order);
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

#[cfg(test)]
fn smoothstep(lo: f32, hi: f32, x: f32) -> f32 {
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// CPU reference for the shader's ID matching rule, used by unit tests.
#[cfg(test)]
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

/// The overlay triangles use the original per-corner sRGB values in both
/// display modes. The dedicated ID overlay shader interpolates these values
/// across the triangle first, then evaluates the same mask as the baked map.
pub fn overlay(mesh: &Mesh, _shown: &Shown, hidden_objects: &[bool], hidden_sets: &[bool]) -> Vec<OverlayVertex> {
    let Some(colors) = &mesh.tri_vertex_color else { return Vec::new() };
    let mut out = Vec::with_capacity(mesh.indices.len());
    for (tri_index, (tri, corners)) in mesh.indices.chunks_exact(3).zip(colors).enumerate() {
        let object = mesh.tri_object.get(tri_index).copied().unwrap_or(u32::MAX) as usize;
        let set = mesh.set_of(tri_index as u32);
        if hidden_objects.get(object).copied().unwrap_or(false)
            || hidden_sets.get(set).copied().unwrap_or(false)
        {
            continue;
        }
        for k in 0..3 {
            let Some(v) = mesh.vertices.get(tri[k] as usize) else { continue };
            let rgb = corners[k];
            out.push(OverlayVertex { pos: v.pos, color: [rgb[0], rgb[1], rgb[2], 255] });
        }
    }
    out
}

/// Whether the mesh has vertex colors for every triangle.
pub fn has_colors(mesh: &Mesh) -> bool {
    mesh.tri_vertex_color.as_ref().is_some_and(|c| c.len() == mesh.indices.len() / 3)
}

/// Whether two imports retain the exact vertex and triangle ordering needed
/// to attach per-corner colors safely. Counts and a sparse position sample are
/// insufficient: a reordered face stream can paint colors onto unrelated faces.
fn same_mesh_order(mesh: &Mesh, other: &Mesh) -> bool {
    mesh.indices == other.indices
        && (mesh.tri_object.is_empty() || other.tri_object.is_empty() || mesh.tri_object == other.tri_object)
        && mesh.vertices.len() == other.vertices.len()
        && mesh.vertices.iter().zip(&other.vertices).all(|(a, b)| {
            (0..3).all(|k| a.pos[k].is_finite() && b.pos[k].is_finite() && (a.pos[k] - b.pos[k]).abs() < 1e-4)
        })
}

/// The vertex colors of `file` for `mesh`, when it is the same mesh.
pub fn colors_from_file(mesh: &Mesh, file: &std::path::Path) -> Result<Vec<[[u8; 3]; 3]>, String> {
    let other = Mesh::load_file(file)?;
    if !same_mesh_order(mesh, &other) {
        return Err(format!(
            "'{}' does not match the loaded mesh's vertex positions or triangle order. Choose the exact FBX/OBJ used to create this project.",
            file.file_name().and_then(|n| n.to_str()).unwrap_or("file")
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
    /// The loaded mesh's palette and a weak identity for the mesh it was made for.
    /// Keeping the control block alive prevents allocator address reuse from
    /// making a different mesh appear to be the cached one.
    pub palette: Option<(std::sync::Weak<Mesh>, Palette)>,
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
        let same_mesh = self.palette.as_ref().is_some_and(|(old, _)| {
            old.upgrade().is_some_and(|old_mesh| std::sync::Arc::ptr_eq(&old_mesh, mesh))
        });
        if !same_mesh {
            let p = mesh.tri_vertex_color.as_deref().map(palette).unwrap_or_default();
            self.palette = Some((std::sync::Arc::downgrade(mesh), p));
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
    pub has_bake: bool,
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

    if !info.has_bake || info.baked_source != IdSource::VertexColors {
        // The mask can't read the selected labels until this Texture Set has
        // a baked ID map from vertex colors. An unopened/uncached map also
        // counts as missing, even when its last saved source was VertexColors.
        if info.has_colors && !vs.auto_baked {
            vs.auto_baked = true;
            vs.want_rebake = true;
        }
        ui.add_space(2.0);
        let message = if info.has_bake {
            "The baked ID map was made from mesh parts or materials, so the mask on the texture ignores these colors until it is baked from vertex colors."
        } else {
            "This Texture Set has no baked mesh maps yet. Bake its ID map from the imported vertex colors to use this mask."
        };
        ui.label(
            egui::RichText::new(message)
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
    fn reimport_identity_checks_all_positions_and_triangle_order() {
        use crate::mesh::Primitive;

        let mesh = Primitive::Sphere.build();
        assert!(same_mesh_order(&mesh, &mesh.clone()));

        let mut moved_late_vertex = mesh.clone();
        let late = moved_late_vertex.vertices.len() - 1;
        moved_late_vertex.vertices[late].pos[0] += 0.01;
        assert!(!same_mesh_order(&mesh, &moved_late_vertex), "a mismatch beyond the old 64-vertex sample must be caught");

        let mut reordered = mesh.clone();
        reordered.indices.swap(0, 1);
        assert!(!same_mesh_order(&mesh, &reordered), "triangle corner order determines per-corner color assignment");
    }

    #[test]
    fn palette_cache_rebuilds_when_the_mesh_arc_changes() {
        use crate::mesh::Primitive;
        use std::sync::Arc;

        let mut red = Primitive::Plane.build();
        red.tri_vertex_color = Some(vec![[[255, 0, 0]; 3]; red.triangle_count()]);
        let mut green = Primitive::Plane.build();
        green.tri_vertex_color = Some(vec![[[0, 255, 0]; 3]; green.triangle_count()]);
        let red = Arc::new(red);
        let green = Arc::new(green);

        let mut state = VertexIdState::default();
        assert_eq!(state.palette_for(&red).entries[0].rgb, [255, 0, 0]);
        assert_eq!(state.palette_for(&green).entries[0].rgb, [0, 255, 0]);
    }

    #[test]
    fn mask_overlay_keeps_raw_corner_colors_for_fragment_level_matching() {
        use crate::mesh::Primitive;

        let mut mesh = Primitive::Plane.build();
        let colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]];
        mesh.tri_vertex_color = Some(vec![[colors[0], colors[1], colors[2]]; mesh.triangle_count()]);
        let shown = Shown::Mask(vec![[1.0, 0.0, 0.0].map(f32::to_bits)], 0.15f32.to_bits());
        let overlay = super::overlay(&mesh, &shown, &[], &[]);

        assert_eq!(overlay.len(), mesh.indices.len());
        for (vertex, expected) in overlay.iter().zip(colors.into_iter().cycle()) {
            assert_eq!(&vertex.color[..3], &expected);
        }
    }

    #[test]
    fn vertex_color_overlay_excludes_hidden_imported_objects() {
        use crate::mesh::Primitive;

        let mut mesh = Primitive::Plane.build();
        let triangles = mesh.triangle_count();
        mesh.tri_vertex_color = Some(vec![[[255, 0, 0]; 3]; triangles]);
        mesh.objects = vec!["Hidden".into(), "Visible".into()];
        mesh.tri_object = (0..triangles).map(|tri| (tri % 2) as u32).collect();

        let visible_triangles = mesh.tri_object.iter().filter(|&&object| object == 1).count();
        let overlay = super::overlay(&mesh, &Shown::Colors, &[true, false], &[]);

        assert_eq!(overlay.len(), visible_triangles * 3);
    }

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
        let selected: Vec<u8> = p.entries.iter().map(|entry| entry.rgb[0]).collect();
        assert_eq!(
            selected,
            (0..PALETTE_MAX).map(|index| index as u8).collect::<Vec<_>>(),
            "equal-frequency palette ties must choose the lowest RGB values deterministically",
        );
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
        assert_eq!(g.id_colors(), vec![[0.95, 0.2375, 0.2375]]);
        let many: Vec<[f32; 3]> = (0..12).map(|i| [i as f32 / 12.0, 0.0, 0.0]).collect();
        g.set_id_colors(&many);
        assert_eq!(g.id_colors(), many[..MAX_ID_COLORS].to_vec());
        g.set_id_colors(&many[..1]);
        assert_eq!(g.id_colors(), vec![many[0]]);
        assert_eq!(g.id_more, [[0.0; 3]; MAX_ID_COLORS - 1], "unused slots are cleared");
    }
}
