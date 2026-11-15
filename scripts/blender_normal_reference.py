"""Blender ground truth for tangent-space normal-map decoding.

Renders (Cycles, CPU) the world-space normal produced by a Normal Map node on a
controlled quad, so the result can be compared with the viewport shader.
Run:  blender -b --python blender_oracle.py -- out.json
"""
import json
import math
import sys

import bpy
import mathutils

OUT = sys.argv[sys.argv.index("--") + 1]


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def build_quad(uv_mode, vertex_normal_z):
    """Quad in the XY plane, CCW seen from +Z (geometric normal +Z).

    uv_mode: 'normal'   u -> +x, v -> +y
             'mirror_u' u -> -x, v -> +y
             'flip_v'   u -> +x, v -> -y
             'both'     u -> -x, v -> -y
    vertex_normal_z: +1 / -1, the imported (custom) shading normal.
    """
    verts = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
    mesh = bpy.data.meshes.new("quad")
    mesh.from_pydata(verts, [], [[0, 1, 2, 3]])
    mesh.update()
    uvl = mesh.uv_layers.new(name="UVMap")
    sx = -1.0 if uv_mode in ("mirror_u", "both") else 1.0
    sy = -1.0 if uv_mode in ("flip_v", "both") else 1.0
    for loop in mesh.loops:
        x, y, _ = verts[loop.vertex_index]
        u = 0.5 + 0.4 * sx * x
        v = 0.5 + 0.4 * sy * y
        uvl.data[loop.index].uv = (u, v)
    mesh.use_auto_smooth = True  # Blender 4.0: needed for custom normals
    mesh.normals_split_custom_set([(0.0, 0.0, float(vertex_normal_z))] * len(mesh.loops))
    obj = bpy.data.objects.new("quad", mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def setup_material(obj, tex):
    mat = bpy.data.materials.new("m")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    rgb = nt.nodes.new("ShaderNodeRGB")
    rgb.outputs[0].default_value = (tex[0] * 0.5 + 0.5, tex[1] * 0.5 + 0.5, tex[2] * 0.5 + 0.5, 1.0)
    nm = nt.nodes.new("ShaderNodeNormalMap")
    nm.space = "TANGENT"
    nm.uv_map = "UVMap"
    nm.inputs["Strength"].default_value = 1.0
    ma = nt.nodes.new("ShaderNodeVectorMath")
    ma.operation = "MULTIPLY_ADD"
    ma.inputs[1].default_value = (0.5, 0.5, 0.5)
    ma.inputs[2].default_value = (0.5, 0.5, 0.5)
    em = nt.nodes.new("ShaderNodeEmission")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(rgb.outputs[0], nm.inputs["Color"])
    nt.links.new(nm.outputs["Normal"], ma.inputs[0])
    nt.links.new(ma.outputs[0], em.inputs["Color"])
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    obj.data.materials.append(mat)


def setup_scene(view_from):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 1
    sc.cycles.use_denoising = False
    sc.cycles.filter_width = 0.01
    sc.render.resolution_x = 16
    sc.render.resolution_y = 16
    sc.render.resolution_percentage = 100
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.view_settings.gamma = 1.0
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_depth = "32"
    sc.world = bpy.data.worlds.new("w")
    sc.world.use_nodes = False
    sc.world.color = (0, 0, 0)
    cam = bpy.data.cameras.new("cam")
    cam.type = "ORTHO"
    cam.ortho_scale = 1.0
    co = bpy.data.objects.new("cam", cam)
    bpy.context.scene.collection.objects.link(co)
    if view_from == "front":
        co.location = (0, 0, 5)
        co.rotation_euler = (0, 0, 0)
    else:
        co.location = (0, 0, -5)
        co.rotation_euler = (math.pi, 0, 0)
    sc.camera = co


def render_center(path):
    sc = bpy.context.scene
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(path)
    w, h = img.size
    px = list(img.pixels)
    i = ((h // 2) * w + (w // 2)) * 4
    n = [px[i] * 2.0 - 1.0, px[i + 1] * 2.0 - 1.0, px[i + 2] * 2.0 - 1.0]
    bpy.data.images.remove(img)
    return n


def corner_tangents(obj):
    me = obj.data
    me.calc_tangents(uvmap="UVMap")
    return [
        {
            "t": [round(c, 5) for c in l.tangent],
            "sign": l.bitangent_sign,
            "n": [round(c, 5) for c in l.normal],
        }
        for l in me.loops
    ]


TILTS = {
    "flat": (0.0, 0.0, 1.0),
    "x": (0.6, 0.0, 0.8),
    "y": (0.0, 0.6, 0.8),
}

results = []
for uv_mode in ("normal", "mirror_u", "flip_v", "both"):
    for view_from in ("front", "back"):
        for vn in (1, -1):
            for tilt_name, tilt in TILTS.items():
                reset()
                obj = build_quad(uv_mode, vn)
                tangents = corner_tangents(obj)[0]
                setup_material(obj, tilt)
                setup_scene(view_from)
                n = render_center(f"/tmp/oracle_{uv_mode}_{view_from}_{vn}_{tilt_name}.exr")
                results.append(
                    {
                        "uv": uv_mode,
                        "view": view_from,
                        "vertex_normal_z": vn,
                        "tilt": tilt_name,
                        "tilt_ts": list(tilt),
                        "blender_world_normal": [round(c, 4) for c in n],
                        "mikk_corner0": tangents,
                    }
                )

with open(OUT, "w") as f:
    json.dump(results, f, indent=1)
print("WROTE", OUT, len(results))
