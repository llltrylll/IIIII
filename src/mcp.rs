//! MCP server: lets AI agents (Claude Code, Claude Desktop, …) drive the app.
//!
//! Streamable HTTP transport answering with plain JSON (no SSE stream), on
//! 127.0.0.1 only. Connection threads parse JSON-RPC and hand each tool call
//! to the UI thread through a channel, waking it with a repaint request; the
//! UI thread runs the tool against the live document and replies when the
//! result is ready (bakes and screenshots reply on completion).

use eframe::egui;
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;

pub const DEFAULT_PORT: u16 = 39831;
const PROTOCOL_VERSION: &str = "2025-06-18";

#[derive(Clone)]
pub enum Content {
    Text(String),
    Png(Vec<u8>),
}

pub type ToolResult = Result<Vec<Content>, String>;

/// A tool call waiting for the UI thread.
pub struct Call {
    pub tool: String,
    pub args: Value,
    pub reply: mpsc::Sender<ToolResult>,
}

pub struct McpServer {
    pub calls: mpsc::Receiver<Call>,
    pub port: u16,
    /// Off: requests are refused with 503 (the listener keeps the port).
    pub enabled: Arc<AtomicBool>,
}

impl McpServer {
    /// Binds the first free port from [`DEFAULT_PORT`] and serves in the background.
    /// Listens on `port` or the next free one of the nine after it.
    pub fn start(ctx: &egui::Context, port: u16, enabled: bool) -> Result<McpServer, String> {
        let listener = (0..10)
            .find_map(|i| TcpListener::bind(("127.0.0.1", port.saturating_add(i))).ok())
            .ok_or_else(|| format!("MCP: ports {port}-{} are in use", port.saturating_add(9)))?;
        let port = listener.local_addr().map_err(|e| e.to_string())?.port();
        let (tx, rx) = mpsc::channel();
        let enabled = Arc::new(AtomicBool::new(enabled));
        let (ctx, on) = (ctx.clone(), enabled.clone());
        std::thread::Builder::new()
            .name("mcp listener".into())
            .spawn(move || {
                for stream in listener.incoming().flatten() {
                    let (tx, ctx, on) = (tx.clone(), ctx.clone(), on.clone());
                    let _ = std::thread::Builder::new().name("mcp connection".into()).spawn(move || {
                        let _ = serve_connection(stream, &tx, &ctx, &on);
                    });
                }
            })
            .map_err(|e| e.to_string())?;
        Ok(McpServer { calls: rx, port, enabled })
    }

    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}/mcp", self.port)
    }
}

struct Request {
    method: String,
    path: String,
    origin: Option<String>,
    body: Vec<u8>,
}

fn read_request(reader: &mut BufReader<TcpStream>) -> std::io::Result<Option<Request>> {
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let path = parts.next().unwrap_or_default().to_string();
    const MAX_REQUEST_BYTES: usize = 16 * 1024 * 1024;
    let (mut length, mut origin, mut saw_length) = (0usize, None, false);
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h)? == 0 || h.trim().is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            match k.trim().to_ascii_lowercase().as_str() {
                "content-length" => {
                    if saw_length {
                        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "duplicate Content-Length"));
                    }
                    saw_length = true;
                    length = v.trim().parse().map_err(|_| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid Content-Length")
                    })?;
                }
                "transfer-encoding" => {
                    return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "transfer encoding is not supported"));
                }
                "origin" => origin = Some(v.trim().to_string()),
                _ => {}
            }
        }
    }
    if length > MAX_REQUEST_BYTES {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "request body exceeds 16 MiB"));
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body)?;
    Ok(Some(Request { method, path, origin, body }))
}

fn respond(stream: &mut TcpStream, status: &str, body: Option<&Value>) -> std::io::Result<()> {
    let text = body.map(|b| b.to_string()).unwrap_or_default();
    let ctype = if body.is_some() { "Content-Type: application/json\r\n" } else { "" };
    write!(stream, "HTTP/1.1 {status}\r\n{ctype}Content-Length: {}\r\n\r\n{text}", text.len())?;
    stream.flush()
}

/// Browsers can reach localhost too; refuse any web page origin (DNS rebinding).
fn origin_allowed(origin: &Option<String>) -> bool {
    match origin.as_deref() {
        None | Some("null") => true,
        Some(o) => ["http://localhost", "http://127.0.0.1", "https://localhost", "https://127.0.0.1"]
            .iter()
            .any(|ok| o == *ok || o.starts_with(&format!("{ok}:"))),
    }
}

fn serve_connection(stream: TcpStream, calls: &mpsc::Sender<Call>, ctx: &egui::Context, on: &AtomicBool) -> std::io::Result<()> {
    let mut out = stream.try_clone()?;
    let mut reader = BufReader::new(stream);
    while let Some(req) = read_request(&mut reader)? {
        if !origin_allowed(&req.origin) {
            respond(&mut out, "403 Forbidden", None)?;
            continue;
        }
        if !req.path.starts_with("/mcp") {
            respond(&mut out, "404 Not Found", None)?;
            continue;
        }
        if !on.load(Ordering::Relaxed) {
            respond(&mut out, "503 Service Unavailable", None)?;
            continue;
        }
        match req.method.as_str() {
            "POST" => {
                let Ok(msg) = serde_json::from_slice::<Value>(&req.body) else {
                    let err = json!({"jsonrpc": "2.0", "id": null, "error": {"code": -32700, "message": "parse error"}});
                    respond(&mut out, "400 Bad Request", Some(&err))?;
                    continue;
                };
                let replies: Vec<Value> = match &msg {
                    Value::Array(batch) => batch.iter().filter_map(|m| handle(m, calls, ctx)).collect(),
                    m => handle(m, calls, ctx).into_iter().collect(),
                };
                match (&msg, replies.len()) {
                    (_, 0) => respond(&mut out, "202 Accepted", None)?,
                    (Value::Array(_), _) => respond(&mut out, "200 OK", Some(&Value::Array(replies)))?,
                    _ => respond(&mut out, "200 OK", Some(&replies[0]))?,
                }
            }
            // No server-initiated stream; sessions are stateless.
            "GET" => respond(&mut out, "405 Method Not Allowed", None)?,
            "DELETE" => respond(&mut out, "200 OK", None)?,
            _ => respond(&mut out, "405 Method Not Allowed", None)?,
        }
    }
    Ok(())
}

/// Handles one JSON-RPC message; `None` for notifications.
fn handle(msg: &Value, calls: &mpsc::Sender<Call>, ctx: &egui::Context) -> Option<Value> {
    let id = msg.get("id")?.clone();
    let method = msg.get("method").and_then(Value::as_str).unwrap_or_default();
    let params = msg.get("params").cloned().unwrap_or(Value::Null);
    let result = match method {
        "initialize" => {
            let version = params.get("protocolVersion").and_then(Value::as_str).unwrap_or(PROTOCOL_VERSION);
            Ok(json!({
                "protocolVersion": version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "material-painter", "version": env!("CARGO_PKG_VERSION")},
                "instructions": INSTRUCTIONS,
            }))
        }
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({"tools": tool_list()})),
        "tools/call" => {
            let tool = params.get("name").and_then(Value::as_str).unwrap_or_default().to_string();
            let args = params.get("arguments").cloned().unwrap_or(json!({}));
            Ok(call_tool(tool, args, calls, ctx))
        }
        _ => Err(json!({"code": -32601, "message": format!("unknown method {method}")})),
    };
    Some(match result {
        Ok(r) => json!({"jsonrpc": "2.0", "id": id, "result": r}),
        Err(e) => json!({"jsonrpc": "2.0", "id": id, "error": e}),
    })
}

fn call_tool(tool: String, args: Value, calls: &mpsc::Sender<Call>, ctx: &egui::Context) -> Value {
    let (tx, rx) = mpsc::channel();
    let result = if calls.send(Call { tool, args, reply: tx }).is_ok() {
        ctx.request_repaint();
        rx.recv().unwrap_or_else(|_| Err("the app closed before answering".into()))
    } else {
        Err("the app is shutting down".into())
    };
    match result {
        Ok(content) => {
            let content: Vec<Value> = content
                .into_iter()
                .map(|c| match c {
                    Content::Text(t) => json!({"type": "text", "text": t}),
                    Content::Png(png) => json!({"type": "image", "data": base64(&png), "mimeType": "image/png"}),
                })
                .collect();
            json!({"content": content, "isError": false})
        }
        Err(e) => json!({"content": [{"type": "text", "text": e}], "isError": true}),
    }
}

fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        for (i, shift) in [18, 12, 6, 0].into_iter().enumerate() {
            if i <= chunk.len() {
                out.push(T[(n >> shift & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

const INSTRUCTIONS: &str = "Material Painter is a PBR texture painting app (a Substance Painter alternative). \
The document is one mesh with a layer stack (paint and fill layers, masks with smart generators driven by baked mesh maps). \
Start with get_state. Layer indices are 0 = bottom. Use list_presets for valid names. \
Every edit is undoable (undo/redo tools, or Ctrl+Z in the app). render_view returns an image of the 3D viewport; \
screenshot_app returns the whole window. bake and load_mesh return once baking finishes.";

fn schema(props: Value, required: &[&str]) -> Value {
    json!({"type": "object", "properties": props, "required": required})
}

fn tool(name: &str, description: &str, input: Value) -> Value {
    json!({"name": name, "description": description, "inputSchema": input})
}

pub fn tool_list() -> Vec<Value> {
    let camera = json!({
        "yaw": {"type": "number", "description": "Orbit angle around the vertical axis, radians"},
        "pitch": {"type": "number", "description": "Orbit elevation, radians"},
        "distance": {"type": "number", "description": "Camera distance; the mesh fits a 2-unit box, default 5.2"},
        "target": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "description": "Orbit center; the mesh is centered on the origin"},
    });
    let mut render_props = camera.as_object().unwrap().clone();
    render_props.insert("mode".into(), json!({"type": "string", "description": "View mode name (see list_presets), default: current"}));
    render_props.insert("width".into(), json!({"type": "integer", "description": "Default 1024"}));
    render_props.insert("height".into(), json!({"type": "integer", "description": "Default 768"}));
    render_props.insert("transparent".into(), json!({"type": "boolean", "description": "Transparent background instead of the sky (default false)"}));
    let mut view_props = camera.as_object().unwrap().clone();
    view_props.insert("mode".into(), json!({"type": "string"}));
    view_props.insert("layout".into(), json!({"type": "string", "enum": ["split", "3d", "2d"]}));
    view_props.insert("ray_trace".into(), json!({"type": "boolean"}));
    view_props.insert("wireframe".into(), json!({"type": "boolean", "description": "Mesh wireframe over the 3D view"}));
    let mut paint_props = render_props.clone();
    paint_props.remove("mode");
    paint_props.remove("transparent");
    for (k, v) in [
        ("index", json!({"type": "integer"})),
        ("points", json!({"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}, "description": "[x, y] pixels along the stroke"})),
        ("target", json!({"type": "string", "enum": ["content", "mask"]})),
        ("color", json!({"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "description": "sRGB 0-1"})),
        ("roughness", json!({"type": "number"})),
        ("metallic", json!({"type": "number"})),
        ("height", json!({"type": "number", "description": "-1..1"})),
        ("opacity", json!({"type": "number", "description": "0 (see-through) .. 1 (opaque)"})),
        ("emission", json!({"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "description": "Emitted light, sRGB 0-1 (black: none)"})),
        ("specular", json!({"type": "number", "description": "Specular level of non-metals 0-1 (0.5: the usual 4% reflectance)"})),
        ("mask_value", json!({"type": "number", "description": "Mask strokes: 1 reveals, 0 hides"})),
        ("erase", json!({"type": "boolean"})),
        ("size", json!({"type": "number", "description": "Brush diameter in pixels (default 40)"})),
        ("hardness", json!({"type": "number"})),
        ("flow", json!({"type": "number"})),
        ("opacity", json!({"type": "number"})),
        ("spacing", json!({"type": "number", "description": "Stamp spacing as a fraction of the size (default 0.15)"})),
        ("symmetry", json!({"type": "string", "enum": ["off", "x", "y", "z"], "description": "Mirror across the mesh center"})),
        ("tip", json!({"type": "string", "description": "Brush tip name (list_presets 'brush_tips'); default round"})),
        ("angle", json!({"type": "number", "description": "Tip rotation in degrees"})),
        ("follow", json!({"type": "boolean", "description": "Turn the tip along the stroke direction"})),
        ("projection", json!({"type": "string", "enum": ["screen", "surface", "uv"], "description": "How stamps reach the mesh: screen (from the camera, default), surface (a disc lying on the surface, wraps around edges), uv (a disc in texture space)"})),
        ("paint_through", json!({"type": "boolean", "description": "Screen projection: also paint surfaces hidden behind others"})),
    ] {
        paint_props.insert(k.into(), v);
    }
    let mut poly_props = render_props.clone();
    poly_props.remove("mode");
    poly_props.remove("transparent");
    for (k, v) in [
        ("index", json!({"type": "integer"})),
        ("mode", json!({"type": "string", "enum": ["triangle", "polygon", "loop", "flat", "part", "object", "uv_island", "texture_set"], "description": "What a click selects (default polygon: a quad or triangle; loop: a strip of quads through the edge nearest the point; flat: faces up to a 30 degree edge; part: a loose part; object: a whole imported object; texture_set: the whole material)"})),
        ("x", json!({"type": "number", "description": "Pixel column in the render_view image (same camera arguments)"})),
        ("y", json!({"type": "number", "description": "Pixel row in the render_view image"})),
        ("uv", json!({"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2, "description": "Pick by UV position instead of a pixel"})),
        ("target", json!({"type": "string", "enum": ["content", "mask"]})),
        ("color", json!({"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "description": "sRGB 0-1"})),
        ("roughness", json!({"type": "number"})),
        ("metallic", json!({"type": "number"})),
        ("height", json!({"type": "number", "description": "-1..1"})),
        ("opacity", json!({"type": "number", "description": "0 (see-through) .. 1 (opaque)"})),
        ("emission", json!({"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "description": "Emitted light, sRGB 0-1 (black: none)"})),
        ("specular", json!({"type": "number", "description": "Specular level of non-metals 0-1 (0.5: the usual 4% reflectance)"})),
        ("mask_value", json!({"type": "number"})),
        ("opacity", json!({"type": "number"})),
        ("erase", json!({"type": "boolean"})),
    ] {
        poly_props.insert(k.into(), v);
    }
    let mut select_props = render_props.clone();
    select_props.remove("mode");
    select_props.remove("transparent");
    select_props.insert("index".into(), json!({"type": "integer", "description": "Layer whose mask receives the selection"}));
    select_props.insert("x".into(), json!({"type": "number", "description": "Pixel column in the render_view image"}));
    select_props.insert("y".into(), json!({"type": "number", "description": "Pixel row in the render_view image"}));
    select_props.insert("parts".into(), json!({"type": "array", "items": {"type": "integer"}, "description": "Select whole mesh parts instead of growing from a seed: ids from the bake report's id_regions (when IDs come from mesh parts)"}));
    select_props.insert("point".into(), json!({"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}));
    select_props.insert("crease_angle".into(), json!({"type": "number", "description": "Fold angle in degrees that ends the region, default 30"}));
    select_props.insert("stop_at".into(), json!({"type": "string", "enum": ["concave", "any"], "description": "concave (default): stop only where the surface folds inward, as where one part sits in another; any: also stop at sharp convex edges"}));
    select_props.insert("mode".into(), json!({"type": "string", "enum": ["replace", "add", "subtract"], "description": "replace (default): the mask becomes the selection"}));
    vec![
        tool("get_state", "Document overview: mesh, resolution, layer stack, bake report, view, undo history.", schema(json!({}), &[])),
        tool("list_presets", "Valid names: smart materials, materials, generators, blend modes, view modes, export presets, bake qualities.", schema(json!({}), &[])),
        tool(
            "load_mesh",
            "Load a mesh file (FBX/OBJ with UVs) or a built-in primitive, keeping the layers. Bakes mesh maps and returns when done.",
            schema(json!({
                "path": {"type": "string", "description": "Absolute path to an .fbx or .obj"},
                "primitive": {"type": "string", "enum": ["Rounded Cube", "Sphere", "Torus", "Cylinder", "Plane"]}
            }), &[]),
        ),
        tool(
            "bake",
            "Bake mesh maps (AO, curvature, edges, thickness, ...) and return the report when finished.",
            schema(json!({
                "quality": {"type": "string", "description": "Draft, Standard, High or Ultra"},
                "high_poly": {"type": "string", "description": "Path of a high poly to bake from; empty string for low-poly only"},
                "cage": {"type": "string", "description": "Path of a cage mesh (the low poly with pushed-out vertices, same topology); empty string for none"},
                "front": {"type": "number", "description": "Ray start distance above the low poly (no cage); mesh is ~2 units across"},
                "rear": {"type": "number", "description": "Ray reach below the low poly"},
                "match_by_name": {"type": "boolean", "description": "Pair low- and high-poly parts by name (_low/_high suffixes ignored); default true"},
                "antialiasing": {"type": "integer", "minimum": 1, "maximum": 8, "description": "Supersampling grid per texel (NxN)"},
                "id_source": {"type": "string", "enum": ["parts", "materials", "vertex_colors"]},
                "resolution": {"type": "integer", "enum": [512, 1024, 2048, 4096], "description": "Texture set resolution"}
            }), &[]),
        ),
        tool(
            "add_layer",
            "Add a layer at stack position `index` (0 = bottom, default: top). kind 'paint' (empty), 'fill' (material preset) or 'smart' (smart material: several masked layers).",
            schema(json!({
                "kind": {"type": "string", "enum": ["paint", "fill", "smart", "folder", "texture"], "description": "texture: a fill layer whose height is a shelf texture (list_presets 'textures')"},
                "preset": {"type": "string", "description": "Material name for fill, smart material name for smart"},
                "name": {"type": "string"},
                "index": {"type": "integer"}
            }), &["kind"]),
        ),
        tool(
            "update_layer",
            "Change layer properties. Fill layers also take color/roughness/metallic/height and procedural variation, which breaks up flat fills the way real surfaces vary.",
            schema(json!({
                "index": {"type": "integer"},
                "name": {"type": "string"},
                "visible": {"type": "boolean"},
                "opacity": {"type": "number", "minimum": 0, "maximum": 1},
                "blend": {"type": "string"},
                "color": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "description": "sRGB 0-1"},
                "roughness": {"type": "number"},
                "metallic": {"type": "number"},
                "height": {"type": "number", "description": "Fill height -1..1; enables the layer's height channel (shades as bumps through the mask)"},
                "emission": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "description": "Fill emission, sRGB 0-1 (black: none); enables the emission channel"},
                "specular": {"type": "number", "description": "Specular level of non-metals 0-1 (0.5: the usual 4% reflectance); enables the specular channel"},
                "normal_strength": {"type": "number", "description": "How strongly the layer's normal-map image bends the surface (1: as authored)"},
                "color_variation": {"type": "number", "description": "Fractal brightness variation 0-1"},
                "roughness_variation": {"type": "number", "description": "Fractal roughness variation 0-1"},
                "height_variation": {"type": "number", "description": "Fractal height variation 0-1; enables the height channel"},
                "variation_scale": {"type": "number", "description": "Variation noise frequency (default 4)"},
                "variation_seed": {"type": "number"},
                "part_variation": {"type": "number", "description": "Per-part brightness and roughness variation 0-1 (each plank or bolt differs)"},
                "images": {"type": "object", "description": "Channel -> image file path (null removes): color, roughness, metallic, height (mid grey = flat), opacity, emission, normal (an OpenGL normal map), specular. Images replace the channel's value."},
                "projection": {"type": "string", "enum": ["uv", "triplanar", "planar_x", "planar_y", "planar_z", "spherical", "cylindrical"], "description": "How images map: UV; tri-planar (world space, seamless, no UVs needed); planar along an axis, spherical or cylindrical (span the mesh's bounds)"},
                "tiling": {"type": "number", "description": "Repeats across UV space, or per world unit for tri-planar (mesh is ~2 units across)"},
                "rotation": {"type": "number", "description": "Degrees"},
                "offset": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                "triplanar_sharpness": {"type": "number", "description": "Blend between the three projections, 1 soft .. 16 sharp"}
            }), &["index"]),
        ),
        tool(
            "preview_settings",
            "Read or change the 3D preview: camera FOV, environment HDRI, background and post-processing. Every argument is optional; returns the current settings and the available HDRIs. Changing 'hdri' replies once it is loaded.",
            schema(json!({
                "hdri": {"type": "string", "description": "Built-in HDRI name (see available_hdris) or a path to an equirectangular .hdr/.exr"},
                "fov": {"type": "number", "description": "Vertical field of view in degrees"},
                "env_rotation": {"type": "number", "description": "Degrees"},
                "env_intensity": {"type": "number"},
                "key_light": {"type": "boolean", "description": "Directional key light with shadows on top of the environment"},
                "key_intensity": {"type": "number"},
                "key_azimuth": {"type": "number", "description": "Degrees around the up axis (turns with env_rotation)"},
                "key_elevation": {"type": "number", "description": "Degrees above the horizon"},
                "background": {"type": "string", "enum": ["environment", "color", "transparent"]},
                "bg_blur": {"type": "number", "description": "0 sharp .. 1 fully blurred"},
                "bg_intensity": {"type": "number"},
                "bg_color": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "description": "Linear RGB"},
                "exposure": {"type": "number", "description": "Multiplier (1 = neutral)"},
                "tonemap": {"type": "string", "description": "ACES, AgX-style, Neutral, Filmic or Linear"},
                "contrast": {"type": "number"},
                "saturation": {"type": "number"},
                "temperature": {"type": "number", "description": "-1 cool .. 1 warm"},
                "tint": {"type": "number", "description": "-1 green .. 1 magenta"},
                "bloom": {"type": "number"},
                "bloom_threshold": {"type": "number"},
                "vignette": {"type": "number"},
                "chromatic_aberration": {"type": "number"},
                "sharpen": {"type": "number"},
                "grain": {"type": "number"}
            }), &[]),
        ),
        tool(
            "paint_stroke",
            "Paint a brush stroke through pixel positions of a render_view image (same camera arguments). Paints the channels given (color, roughness, metallic, height) on a paint layer, or with target 'mask' paints its mask. One undo step.",
            schema(Value::Object(paint_props), &["index", "points"]),
        ),
        tool(
            "poly_fill",
            "Polygon fill: paint whole faces picked at a render_view pixel (or a UV position): one triangle, a polygon (quad), a whole mesh part or a UV island. Fills the channels given on a paint layer, or with target 'mask' its mask. One undo step.",
            schema(Value::Object(poly_props), &["index"]),
        ),
        tool(
            "cage_preview",
            "Show or hide the bake cage in the 3D view and report what the bake rays will find from each low-poly vertex: shares covered, tight, missed and overlapping (another part reached first). Optional settings change the bake settings without baking.",
            schema(json!({
                "show": {"type": "boolean", "description": "Default true"},
                "high_poly": {"type": "string", "description": "High poly path; empty for low poly only"},
                "cage": {"type": "string", "description": "Cage mesh path; empty for none"},
                "front": {"type": "number"},
                "rear": {"type": "number"},
                "match_by_name": {"type": "boolean"}
            }), &[]),
        ),
        tool(
            "save_project",
            "Save the project (.mpp: mesh, layers, folders, masks, painting, bake fixes, preview settings). Without 'path', saves over the open project file.",
            schema(json!({"path": {"type": "string", "description": "Absolute path ending in .mpp"}}), &[]),
        ),
        tool(
            "open_project",
            "Open a .mpp project, replacing the current one without asking. Replies once its mesh maps are re-baked.",
            schema(json!({"path": {"type": "string"}}), &["path"]),
        ),
        tool("delete_layer", "Delete a layer.", schema(json!({"index": {"type": "integer"}}), &["index"])),
        tool("duplicate_layer", "Duplicate a layer.", schema(json!({"index": {"type": "integer"}}), &["index"])),
        tool(
            "move_layer",
            "Move a layer to a new index.",
            schema(json!({"from": {"type": "integer"}, "to": {"type": "integer"}}), &["from", "to"]),
        ),
        tool(
            "set_mask",
            "Give a layer a smart generator mask, a plain paintable mask, or remove its mask. Generator masks start from the named preset; the optional values below override it. To isolate one mesh part or material use generator 'ID Mask' with `id` from the bake report's id_regions.",
            schema(json!({
                "index": {"type": "integer"},
                "generator": {"type": "string", "description": "Generator name, e.g. Edge Wear"},
                "id": {"type": "integer", "description": "ID region (bake report id_regions) for the ID Mask generator"},
                "id_color": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                "id_tolerance": {"type": "number"},
                "weights": {"type": "object", "description": "Input name -> weight (see list_presets generator_inputs); replaces those inputs' weights"},
                "level": {"type": "number", "description": "Threshold, 0-1"},
                "contrast": {"type": "number", "description": "Transition width, small = sharp"},
                "scale": {"type": "number", "description": "Noise scale"},
                "noise": {"type": "number", "description": "Noise breakup amount, 0-1"},
                "seed": {"type": "number"},
                "invert": {"type": "boolean"},
                "edge_width": {"type": "number", "description": "World width of the Edges/Creases inputs"},
                "paint": {"type": "string", "enum": ["white", "black"]},
                "remove": {"type": "boolean"}
            }), &["index"]),
        ),
        tool("undo", "Undo the last edits.", schema(json!({"steps": {"type": "integer", "minimum": 1}}), &[])),
        tool("redo", "Redo undone edits.", schema(json!({"steps": {"type": "integer", "minimum": 1}}), &[])),
        tool("set_view", "Change the app's own viewport: camera, view mode, layout, ray tracing.", schema(Value::Object(view_props), &[])),
        tool(
            "render_view",
            "Render the 3D viewport offscreen and return it as a PNG. Camera values default to the app's current camera.",
            schema(Value::Object(render_props), &[]),
        ),
        tool(
            "select_surface",
            "Mask a layer to one region of the surface, like a polygon fill that finds the region itself: grows from a seed across the mesh and stops at creases, so a gem set in a socket or a blade welded to a guard can be isolated even on a single-piece mesh with split UVs. Give the seed as a pixel (x, y) of a render_view image together with the same camera values (yaw, pitch, distance, target, width, height), or as a 3D 'point'. Or select whole parts with 'parts' (several at once, e.g. every rivet). The layer gets a paintable mask if it has none.",
            schema(Value::Object(select_props), &["index"]),
        ),
        tool("screenshot_app", "Screenshot of the whole app window as a PNG.", schema(json!({}), &[])),
        tool(
            "export_textures",
            "Export the texture set with an output template (engine preset or a user template).",
            schema(json!({
                "folder": {"type": "string"},
                "preset": {"type": "string", "description": "Output template name, e.g. 'Unreal Engine (packed ORM)', 'Unity URP (metallic)', 'Unity HDRP', 'Godot 4 (ORM)', 'glTF PBR (metal/rough)', 'Roblox SurfaceAppearance', 'Blender (Principled BSDF)', 'PBR Specular Glossiness', 'Baked mesh maps (all)'; default: the export window's current template"},
                "prefix": {"type": "string", "description": "Replaces $mesh in file names"},
                "format": {"type": "string", "enum": ["png", "tga", "jpg", "tif", "exr"]},
                "bit_depth": {"type": "integer", "enum": [8, 16, 32]},
                "size": {"type": "integer", "description": "Output size in pixels (default: document size)"},
                "padding": {"type": "string", "description": "none, infinite, transparent or dilate:<pixels> (default infinite)"}
            }), &["folder"]),
        ),
    ]
}

#[cfg(test)]
mod tests {
    #[test]
    fn base64_matches_rfc4648() {
        assert_eq!(super::base64(b""), "");
        assert_eq!(super::base64(b"f"), "Zg==");
        assert_eq!(super::base64(b"fo"), "Zm8=");
        assert_eq!(super::base64(b"foo"), "Zm9v");
        assert_eq!(super::base64(b"foobar"), "Zm9vYmFy");
    }

    fn post(port: u16, body: &str, origin: Option<&str>) -> (String, String) {
        use std::io::{Read, Write};
        let mut s = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        let origin = origin.map(|o| format!("Origin: {o}\r\n")).unwrap_or_default();
        write!(
            s,
            "POST /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\nAccept: application/json, text/event-stream\r\n{origin}\
             Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        s.shutdown(std::net::Shutdown::Write).unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).unwrap();
        let (head, body) = out.split_once("\r\n\r\n").unwrap_or((&out, ""));
        (head.lines().next().unwrap_or_default().to_string(), body.to_string())
    }

    /// Full JSON-RPC round trip over HTTP, with this thread playing the UI.
    #[test]
    fn http_round_trip() {
        let ctx = eframe::egui::Context::default();
        let server = super::McpServer::start(&ctx, super::DEFAULT_PORT, true).unwrap();
        let port = server.port;
        let ui = std::thread::spawn(move || {
            let call = server.calls.recv().unwrap();
            assert_eq!(call.tool, "get_state");
            assert_eq!(call.args["verbose"], true);
            call.reply.send(Ok(vec![super::Content::Text("ok".into()), super::Content::Png(vec![1, 2, 3])])).unwrap();
        });

        let (status, body) = post(port, r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"1"}}}"#, None);
        assert!(status.contains("200"), "{status}");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["result"]["serverInfo"]["name"], "material-painter");

        let (status, _) = post(port, r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#, None);
        assert!(status.contains("202"), "{status}");

        let (_, body) = post(port, r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#, None);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        let tools = v["result"]["tools"].as_array().unwrap();
        assert!(tools.iter().all(|t| t["inputSchema"]["type"] == "object"));
        assert!(tools.len() >= 15);

        let (_, body) = post(port, r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"get_state","arguments":{"verbose":true}}}"#, Some("http://localhost:5173"));
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["id"], 3);
        assert_eq!(v["result"]["content"][0]["text"], "ok");
        assert_eq!(v["result"]["content"][1]["data"], "AQID");
        assert_eq!(v["result"]["content"][1]["mimeType"], "image/png");
        ui.join().unwrap();

        let (status, _) = post(port, r#"{"jsonrpc":"2.0","id":4,"method":"ping"}"#, Some("https://evil.example"));
        assert!(status.contains("403"), "{status}");
    }

    #[test]
    fn origins() {
        assert!(super::origin_allowed(&None));
        assert!(super::origin_allowed(&Some("http://localhost:3000".into())));
        assert!(!super::origin_allowed(&Some("http://localhost.evil.com".into())));
        assert!(!super::origin_allowed(&Some("https://example.com".into())));
    }
}
