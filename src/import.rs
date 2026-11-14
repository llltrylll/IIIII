//! Mesh file import (OBJ and FBX) into a flat triangle list.
//!
//! UVs are returned "texture-down" (t = 1 - v), matching the rest of the app.
//! FBX scenes are flattened: every mesh node is transformed to world space and
//! converted to a right-handed, Y-up, meter-based space, so a low poly and a
//! high poly exported from the same scene line up.

use glam::{Vec2, Vec3};
use std::collections::HashMap;
use std::path::Path;

pub struct RawMesh {
    pub name: String,
    pub positions: Vec<Vec3>,
    /// `None` when the file has no normals; callers generate smooth ones.
    pub normals: Option<Vec<Vec3>>,
    /// `None` when any part of the file lacks UVs.
    pub uvs: Option<Vec<Vec2>>,
    pub indices: Vec<u32>,
    /// Material index per triangle.
    pub tri_material: Vec<u32>,
    pub material_count: u32,
    /// Material names by index (may be shorter than `material_count`).
    pub material_names: Vec<String>,
    /// Object (mesh node) index per triangle, into `objects`.
    pub tri_object: Vec<u32>,
    pub objects: Vec<String>,
    /// Per-object provenance reported by ufbx for imported FBX meshes.
    pub object_ufbx_generated_normals: Vec<bool>,
    pub object_reversed_winding: Vec<bool>,
    /// Whether the mesh node's world transform has a negative determinant.
    pub object_negative_transform_determinant: Vec<bool>,
    /// Per-object fallback normal generation performed by this importer.
    pub object_fallback_normals_generated: Vec<bool>,
    /// Average vertex color per triangle (legacy ID report).
    pub tri_color: Option<Vec<[u8; 3]>>,
    /// Original sRGB vertex colors for each triangle corner, in import order.
    pub tri_vertex_color: Option<Vec<[[u8; 3]; 3]>>,
    /// Objects without UVs, for error messages.
    pub missing_uvs: Vec<String>,
}

pub const EXTENSIONS: [&str; 2] = ["obj", "fbx"];

pub fn load(path: &Path) -> Result<RawMesh, String> {
    let ext = path.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
    let mut mesh = match ext.as_str() {
        "obj" => load_obj(path)?,
        "fbx" => load_fbx(path)?,
        _ => return Err(format!("Unsupported mesh format '.{ext}'. Use OBJ or FBX.")),
    };
    if mesh.indices.is_empty() {
        return Err("The file contains no triangles.".into());
    }
    mesh.name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    Ok(mesh)
}

fn load_obj(path: &Path) -> Result<RawMesh, String> {
    let (models, materials) = tobj::load_obj(path, &tobj::LoadOptions { single_index: true, triangulate: true, ..Default::default() })
        .map_err(|e| format!("Failed to read OBJ: {e}"))?;
    let mut out = Builder::default();
    // A missing .mtl leaves the indices (and default names).
    out.material_names = materials.map(|m| m.into_iter().map(|m| m.name).collect()).unwrap_or_default();
    for m in &models {
        let mesh = &m.mesh;
        let count = mesh.positions.len() / 3;
        let has_uv = mesh.texcoords.len() >= count * 2;
        let has_n = mesh.normals.len() >= count * 3;
        if !has_uv {
            out.missing_uvs.push(m.name.clone());
        }
        let material = mesh.material_id.unwrap_or(0) as u32;
        let has_c = mesh.vertex_color.len() >= count * 3;
        out.begin_object(&m.name);
        for &i in &mesh.indices {
            let i = i as usize;
            out.push_corner(Corner {
                pos: Vec3::from_slice(&mesh.positions[i * 3..i * 3 + 3]),
                normal: has_n.then(|| Vec3::from_slice(&mesh.normals[i * 3..i * 3 + 3])),
                uv: has_uv.then(|| Vec2::new(mesh.texcoords[i * 2], 1.0 - mesh.texcoords[i * 2 + 1])),
                color: has_c.then(|| Vec3::from_slice(&mesh.vertex_color[i * 3..i * 3 + 3])),
            });
        }
        out.push_materials(mesh.indices.len() / 3, material);
    }
    Ok(out.finish())
}

fn load_fbx(path: &Path) -> Result<RawMesh, String> {
    let opts = ufbx::LoadOpts {
        ignore_animation: true,
        ignore_embedded: true,
        generate_missing_normals: true,
        target_axes: ufbx::CoordinateAxes {
            right: ufbx::CoordinateAxis::PositiveX,
            up: ufbx::CoordinateAxis::PositiveY,
            front: ufbx::CoordinateAxis::PositiveZ,
        },
        target_unit_meters: 1.0,
        space_conversion: ufbx::SpaceConversion::ModifyGeometry,
        geometry_transform_handling: ufbx::GeometryTransformHandling::ModifyGeometry,
        ..Default::default()
    };
    let path_str = path.to_str().ok_or("The FBX path is not valid UTF-8.")?;
    let scene = ufbx::load_file(path_str, opts).map_err(|e| format!("Failed to read FBX: {e:?}"))?;

    let mut out = Builder::default();
    let material_count = scene.materials.len().max(1) as u32;
    out.material_names = scene.materials.iter().map(|m| m.element.name.to_string()).collect();
    let mut tri = Vec::new();
    for node in &scene.nodes {
        let Some(mesh) = node.mesh.as_ref() else { continue };
        let to_world = node.geometry_to_world;
        let normal_m = ufbx::matrix_for_normals(&to_world);
        let negative_transform_determinant = ufbx::matrix_determinant(&to_world) < 0.0;
        let has_uv = mesh.vertex_uv.exists;
        let has_n = mesh.vertex_normal.exists;
        if !has_uv {
            out.missing_uvs.push(node.element.name.to_string());
        }
        let has_c = mesh.vertex_color.exists;
        out.begin_object(&node.element.name);
        out.set_ufbx_normal_metadata(mesh.generated_normals, mesh.reversed_winding, negative_transform_determinant);
        tri.resize(mesh.max_face_triangles * 3, 0u32);
        for (face_index, &face) in mesh.faces.iter().enumerate() {
            let n = mesh.triangulate_face(&mut tri, face) as usize;
            let material = mesh
                .face_material
                .get(face_index)
                .and_then(|&m| mesh.materials.get(m as usize))
                .map_or(0, |m| m.element.typed_id);
            for &ix in &tri[..n * 3] {
                let ix = ix as usize;
                let p = ufbx::transform_position(&to_world, mesh.vertex_position[ix]);
                let normal = has_n.then(|| {
                    let v = ufbx::transform_direction(&normal_m, mesh.vertex_normal[ix]);
                    Vec3::new(v.x as f32, v.y as f32, v.z as f32).normalize_or_zero()
                });
                let uv = has_uv.then(|| {
                    let t = mesh.vertex_uv[ix];
                    Vec2::new(t.x as f32, 1.0 - t.y as f32)
                });
                let color = has_c.then(|| {
                    let c = mesh.vertex_color[ix];
                    Vec3::new(c.x as f32, c.y as f32, c.z as f32)
                });
                out.push_corner(Corner { pos: Vec3::new(p.x as f32, p.y as f32, p.z as f32), normal, uv, color });
            }
            out.push_materials(n, material);
        }
    }
    let mut mesh = out.finish();
    mesh.material_count = mesh.material_count.max(material_count);
    Ok(mesh)
}

struct Corner {
    pos: Vec3,
    normal: Option<Vec3>,
    uv: Option<Vec2>,
    /// 0-1 vertex color as stored in the file.
    color: Option<Vec3>,
}

/// Collects triangle corners and welds identical ones into shared vertices.
#[derive(Default)]
struct Builder {
    positions: Vec<Vec3>,
    normals: Vec<Vec3>,
    uvs: Vec<Vec2>,
    indices: Vec<u32>,
    tri_material: Vec<u32>,
    all_uvs: bool,
    corners: usize,
    weld: HashMap<[u32; 9], u32>,
    /// Whether each object needs generated normals for at least one corner.
    object_missing_normals: Vec<bool>,
    /// ufbx reports whether it generated normals rather than evaluating source data.
    object_ufbx_generated_normals: Vec<bool>,
    /// ufbx reports when it reversed the mesh face winding.
    object_reversed_winding: Vec<bool>,
    /// Negative determinant of the node's geometry-to-world transform.
    object_negative_transform_determinant: Vec<bool>,
    /// Per-vertex provenance, so authored normals survive a partial fallback.
    vertex_has_normal: Vec<bool>,
    vertex_object: Vec<u32>,
    missing_uvs: Vec<String>,
    max_material: u32,
    material_names: Vec<String>,
    objects: Vec<String>,
    tri_object: Vec<u32>,
    any_colors: bool,
    tri_color: Vec<[u8; 3]>,
    tri_vertex_color: Vec<[[u8; 3]; 3]>,
    color_corners: [[u8; 3]; 3],
    /// Color sum of the triangle being built.
    color_acc: Vec3,
}

impl Builder {
    /// Starts a new object; its triangles are tagged with its index.
    fn begin_object(&mut self, name: &str) {
        self.objects.push(name.to_string());
        self.object_missing_normals.push(false);
        self.object_ufbx_generated_normals.push(false);
        self.object_reversed_winding.push(false);
        self.object_negative_transform_determinant.push(false);
    }

    fn set_ufbx_normal_metadata(&mut self, generated_normals: bool, reversed_winding: bool, negative_transform_determinant: bool) {
        if let Some(value) = self.object_ufbx_generated_normals.last_mut() {
            *value = generated_normals;
        }
        if let Some(value) = self.object_reversed_winding.last_mut() {
            *value = reversed_winding;
        }
        if let Some(value) = self.object_negative_transform_determinant.last_mut() {
            *value = negative_transform_determinant;
        }
    }

    fn push_corner(&mut self, c: Corner) {
        if self.corners == 0 {
            self.all_uvs = true;
        }
        self.corners += 1;
        let object = self.objects.len().saturating_sub(1) as u32;
        if c.normal.is_none()
            && let Some(missing) = self.object_missing_normals.get_mut(object as usize)
        {
            *missing = true;
        }
        self.all_uvs &= c.uv.is_some();
        // Keep vertex colors even when only some objects carry a color layer.
        // Unpainted objects receive white so they remain a selectable ID region.
        self.any_colors |= c.color.is_some();
        let corner = (self.corners - 1) % 3;
        let color = c.color.unwrap_or(Vec3::ONE);
        self.color_acc += color;
        self.color_corners[corner] = color.to_array().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8);
        if self.corners.is_multiple_of(3) {
            // Keep the average for older ID-region summaries, but preserve
            // each corner so the GPU can interpolate the real vertex color.
            let avg = self.color_acc / 3.0;
            self.tri_color.push(avg.to_array().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8));
            self.tri_vertex_color.push(self.color_corners);
            self.color_corners = [[0; 3]; 3];
            self.color_acc = Vec3::ZERO;
        }
        let has_normal = c.normal.is_some();
        let n = c.normal.unwrap_or(Vec3::ZERO);
        let uv = c.uv.unwrap_or(Vec2::ZERO);
        // Objects never share vertices, even where they touch exactly.
        let key = [c.pos.x, c.pos.y, c.pos.z, n.x, n.y, n.z, uv.x, uv.y].map(f32::to_bits);
        let key = [key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7], object];
        let next = self.positions.len() as u32;
        let index = *self.weld.entry(key).or_insert(next);
        if index == next {
            self.positions.push(c.pos);
            self.normals.push(n);
            self.uvs.push(uv);
            self.vertex_has_normal.push(has_normal);
            self.vertex_object.push(object);
        }
        self.indices.push(index);
    }

    fn push_materials(&mut self, triangles: usize, material: u32) {
        self.max_material = self.max_material.max(material);
        self.tri_material.extend(std::iter::repeat_n(material, triangles));
        let object = self.objects.len().saturating_sub(1) as u32;
        self.tri_object.extend(std::iter::repeat_n(object, triangles));
    }

    fn finish(mut self) -> RawMesh {
        // Generate missing normals one object at a time. A scene-wide fallback
        // would discard authored normals from otherwise complete objects and
        // can cancel opposite, coincident shells from separate objects.
        for (object, &missing) in self.object_missing_normals.iter().enumerate() {
            if !missing {
                continue;
            }
            let mut object_indices = Vec::new();
            for (triangle, &owner) in self.tri_object.iter().enumerate() {
                if owner == object as u32 {
                    object_indices.extend_from_slice(&self.indices[triangle * 3..triangle * 3 + 3]);
                }
            }
            let generated = crate::mesh::smooth_normals(&self.positions, &object_indices);
            for vertex in 0..self.normals.len() {
                if self.vertex_object[vertex] == object as u32 && !self.vertex_has_normal[vertex] {
                    self.normals[vertex] = generated[vertex];
                }
            }
        }

        RawMesh {
            name: String::new(),
            normals: (!self.positions.is_empty()).then_some(self.normals),
            positions: self.positions,
            uvs: self.all_uvs.then_some(self.uvs),
            indices: self.indices,
            tri_material: self.tri_material,
            material_count: self.max_material + 1,
            material_names: self.material_names,
            missing_uvs: self.missing_uvs,
            tri_object: self.tri_object,
            objects: self.objects,
            object_ufbx_generated_normals: self.object_ufbx_generated_normals,
            object_reversed_winding: self.object_reversed_winding,
            object_negative_transform_determinant: self.object_negative_transform_determinant,
            object_fallback_normals_generated: self.object_missing_normals,
            tri_color: (self.any_colors && self.corners > 0).then_some(self.tri_color),
            tri_vertex_color: (self.any_colors && self.corners > 0).then_some(self.tri_vertex_color),
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn object_normal_provenance_is_retained_per_imported_object() {
        let mut builder = super::Builder::default();
        builder.begin_object("generated by ufbx");
        builder.set_ufbx_normal_metadata(true, true, true);
        builder.begin_object("source normals");
        let raw = builder.finish();

        assert_eq!(raw.object_ufbx_generated_normals, [true, false]);
        assert_eq!(raw.object_reversed_winding, [true, false]);
        assert_eq!(raw.object_negative_transform_determinant, [true, false]);
        assert_eq!(raw.object_fallback_normals_generated, [false, false]);
    }

    #[test]
    fn missing_normals_are_generated_per_object_without_replacing_authored_normals() {
        let mut builder = super::Builder::default();
        let positions = [
            glam::Vec3::new(-1.0, -1.0, 0.0),
            glam::Vec3::new(1.0, -1.0, 0.0),
            glam::Vec3::new(-1.0, 1.0, 0.0),
        ];
        let uvs = [
            glam::Vec2::new(0.0, 0.0),
            glam::Vec2::new(1.0, 0.0),
            glam::Vec2::new(0.0, 1.0),
        ];

        // Object 0 has valid authored normals and faces +Z.
        builder.begin_object("authored normals");
        for i in 0..3 {
            builder.push_corner(super::Corner {
                pos: positions[i],
                normal: Some(glam::Vec3::X),
                uv: Some(uvs[i]),
                color: None,
            });
        }
        builder.push_materials(1, 0);

        // Object 1 occupies the exact same positions but has no normals and
        // reversed winding. Its fallback must be -Z, not zero from cancellation.
        builder.begin_object("missing normals");
        for i in [0usize, 2, 1] {
            builder.push_corner(super::Corner {
                pos: positions[i],
                normal: None,
                uv: Some(uvs[i]),
                color: None,
            });
        }
        builder.push_materials(1, 0);

        let raw = builder.finish();
        let normals = raw.normals.expect("mixed scene should have usable normals");
        assert_eq!(&normals[..3], &[glam::Vec3::X; 3], "authored normals must be preserved");
        for (i, normal) in normals[3..].iter().enumerate() {
            assert!(
                normal.distance(-glam::Vec3::Z) < 1e-5,
                "generated normal {i} should be -Z, got {normal:?}"
            );
        }
    }

    #[test]
    fn partial_normals_preserve_authored_vertices_and_generate_missing_ones() {
        let mut builder = super::Builder::default();
        let positions = [
            glam::Vec3::new(-1.0, -1.0, 0.0),
            glam::Vec3::new(1.0, -1.0, 0.0),
            glam::Vec3::new(1.0, 1.0, 0.0),
            glam::Vec3::new(-1.0, 1.0, 0.0),
        ];
        let uvs = [
            glam::Vec2::new(0.0, 0.0),
            glam::Vec2::new(1.0, 0.0),
            glam::Vec2::new(1.0, 1.0),
            glam::Vec2::new(0.0, 1.0),
        ];
        builder.begin_object("partially authored normals");

        for i in [0usize, 1, 2] {
            builder.push_corner(super::Corner {
                pos: positions[i],
                normal: Some(glam::Vec3::X),
                uv: Some(uvs[i]),
                color: None,
            });
        }
        builder.push_materials(1, 0);

        // The second triangle shares an edge but has missing authored normals.
        // Position welding during fallback must still smooth across that edge.
        for i in [0usize, 2, 3] {
            builder.push_corner(super::Corner {
                pos: positions[i],
                normal: None,
                uv: Some(uvs[i]),
                color: None,
            });
        }
        builder.push_materials(1, 0);

        let raw = builder.finish();
        let normals = raw.normals.expect("partial normals should be repaired");
        assert_eq!(&normals[..3], &[glam::Vec3::X; 3], "authored normals must not be replaced");
        for (i, normal) in normals[3..].iter().enumerate() {
            assert!(
                normal.distance(glam::Vec3::Z) < 1e-5,
                "missing normal {i} should be generated as +Z, got {normal:?}"
            );
        }
    }

    /// Opt-in smoke test: `MP_IMPORT_TEST="a.fbx;b.obj" cargo test -- --ignored`.
    #[test]
    #[ignore]
    fn import_files_from_env() {
        let list = std::env::var("MP_IMPORT_TEST").unwrap_or_default();
        for path in list.split(';').filter(|p| !p.is_empty()) {
            let m = super::load(std::path::Path::new(path)).unwrap_or_else(|e| panic!("{path}: {e}"));
            println!(
                "{path}: {} verts, {} tris, normals {}, uvs {}, materials {}, missing uvs {:?}",
                m.positions.len(),
                m.indices.len() / 3,
                m.normals.is_some(),
                m.uvs.is_some(),
                m.material_count,
                m.missing_uvs
            );
            assert!(m.indices.iter().all(|&i| (i as usize) < m.positions.len()));
        }
    }
}
