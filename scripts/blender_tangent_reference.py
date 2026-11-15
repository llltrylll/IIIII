"""Per-corner MikkTSpace reference data from Blender.

Builds small meshes that stress the cases the viewport has to get right
(narrow tips, polar fans, mirrored UV islands, thin two-sided blades), with
explicit custom normals and UVs, triangulates them, and dumps for every
triangle corner: position, normal, UV (Blender convention, V up), and the
tangent and bitangent sign Blender computes (MikkTSpace via calc_tangents).

Run:  blender -b --python blender_tangent_fixtures.py -- out.json
"""
import json
import math
import sys

import bmesh
import bpy
from mathutils import Vector

OUT = sys.argv[sys.argv.index("--") + 1]


def make_mesh(name, verts, faces, face_uvs, vertex_normal_fn):
    """faces: list of vertex-index tuples; face_uvs: list of per-corner (u, v);
    vertex_normal_fn(face_index, corner_index, position) -> normal."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    uvl = me.uv_layers.new(name="UVMap")
    li = 0
    for fi, f in enumerate(me.polygons):
        for ci, li_ in enumerate(f.loop_indices):
            uvl.data[li_].uv = face_uvs[fi][ci]
    # Triangulate with a fixed rule so the app sees the same triangles.
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method="BEAUTY", ngon_method="BEAUTY")
    bm.to_mesh(me)
    bm.free()
    me.use_auto_smooth = True
    normals = []
    for poly in me.polygons:
        for li_ in poly.loop_indices:
            p = Vector(me.vertices[me.loops[li_].vertex_index].co)
            n = vertex_normal_fn(poly.index, li_, p)
            normals.append(tuple(Vector(n).normalized()))
    me.normals_split_custom_set(normals)
    me.calc_tangents(uvmap="UVMap")
    tris = []
    for poly in me.polygons:
        corners = []
        for li_ in poly.loop_indices:
            l = me.loops[li_]
            p = me.vertices[l.vertex_index].co
            uv = me.uv_layers["UVMap"].data[li_].uv
            corners.append(
                [round(c, 6) for c in (p.x, p.y, p.z)]
                + [round(c, 6) for c in l.normal]
                + [round(uv[0], 6), round(uv[1], 6)]
                + [round(c, 6) for c in l.tangent]
                + [l.bitangent_sign]
            )
        tris.append(corners)
    return {"name": name, "tris": tris}


fixtures = []

# 1. Cone: a narrow tip. Each side face is a triangle (apex, i, i+1) whose UVs
#    put the apex above the middle of its base edge (a classic fan unwrap).
N = 12
H = 2.0
R = 0.5
verts = [(0.0, 0.0, H)] + [(R * math.cos(2 * math.pi * i / N), R * math.sin(2 * math.pi * i / N), 0.0) for i in range(N)]
faces, fuv = [], []
for i in range(N):
    j = (i + 1) % N
    faces.append((0, 1 + i, 1 + j))
    fuv.append([((i + 0.5) / N, 1.0), (i / N, 0.0), ((i + 1) / N, 0.0)])


def cone_normal(fi, li, p):
    if p.z > H - 1e-6:
        return (0, 0, 1)
    return (p.x, p.y, R / H * 0.5)


fixtures.append(make_mesh("cone_tip", verts, faces, fuv, cone_normal))

# 2. Tapered blade: two-sided thin wedge. Top face upright, bottom face is a
#    mirrored UV island (negative UV determinant), a very narrow tip.
L, W = 3.0, 0.4
verts = [(-W, 0, 0), (W, 0, 0), (0.02, 0, L), (-0.02, 0, L)]
faces = [(0, 1, 2, 3), (0, 3, 2, 1)]
fuv = [
    [(0.1, 0.1), (0.4, 0.1), (0.27, 0.9), (0.23, 0.9)],
    [(0.9, 0.1), (0.9 - 0.0, 0.9), (0.73, 0.9), (0.6, 0.1)],
]


def blade_normal(fi, li, p):
    return (0, 1, 0) if fi == 0 else (0, -1, 0)


fixtures.append(make_mesh("blade_two_sided", verts, faces, fuv, blade_normal))

# 3. UV sphere with polar fans and one mirrored hemisphere of UVs.
SEG, RINGS = 12, 6
verts = []
for r in range(RINGS + 1):
    th = math.pi * r / RINGS
    for s in range(SEG):
        ph = 2 * math.pi * s / SEG
        verts.append((math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)))
faces, fuv = [], []
for r in range(RINGS):
    for s in range(SEG):
        a = r * SEG + s
        b = r * SEG + (s + 1) % SEG
        c = (r + 1) * SEG + (s + 1) % SEG
        d = (r + 1) * SEG + s
        u0, u1 = s / SEG, (s + 1) / SEG
        v0, v1 = 1.0 - r / RINGS, 1.0 - (r + 1) / RINGS
        mirror = verts[a][0] > 0.0
        if mirror:
            u0, u1 = 1.0 - u0, 1.0 - u1
        if r == 0:
            faces.append((a, c, d)); fuv.append([(u0, v0), (u1, v1), (u0, v1)])
        elif r == RINGS - 1:
            faces.append((a, b, d)); fuv.append([(u0, v0), (u1, v0), (u0, v1)])
        else:
            faces.append((a, b, c, d)); fuv.append([(u0, v0), (u1, v0), (u1, v1), (u0, v1)])


def sphere_normal(fi, li, p):
    return tuple(p)


fixtures.append(make_mesh("sphere_polar_mirrored", verts, faces, fuv, sphere_normal))

with open(OUT, "w") as f:
    json.dump({"blender": bpy.app.version_string, "fixtures": fixtures}, f, separators=(",", ":"))
print("WROTE", OUT, [(fx["name"], len(fx["tris"])) for fx in fixtures])
