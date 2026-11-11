//! Material graphs: a node graph that renders tileable textures, node by
//! node, for graph fill layers and graph masks. Nodes are WGSL files (see
//! [`def`]); the built-in library and user nodes share the format.

pub mod builtin;
pub mod def;
pub mod presets;
pub(crate) mod runtime;
pub mod shading;
pub mod smart;

pub use def::{Kind, NodeDef, ParamKind, Value};
pub use runtime::Runtime;

use serde_json::{Value as Json, json};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Channels an Output node can feed (its `usage` parameter, in order; files
/// store the names, so new ones go at the end).
pub const USAGES: [&str; 9] = ["Base Color", "Roughness", "Metallic", "Height", "Mask", "Opacity", "Emission", "Normal", "Specular"];
/// The Mask usage (graph masks).
pub const MASK_USAGE: usize = 4;
/// The usage feeding each paint channel (`engine::CHANNELS` order).
pub const CHANNEL_USAGE: [usize; crate::engine::CHANNELS] = [0, 1, 2, 3, 5, 6, 7, 8];

/// A graph's outputs per paint channel (`engine::CHANNELS` order).
pub fn channel_outputs<T: Clone>(outs: &[Option<T>; USAGES.len()]) -> [Option<T>; crate::engine::CHANNELS] {
    CHANNEL_USAGE.map(|u| outs[u].clone())
}
pub const OUTPUT_DEF: &str = "output";

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub id: u64,
    pub def: String,
    /// Canvas position.
    pub pos: [f32; 2],
    /// Values that differ from the definition's defaults.
    pub params: HashMap<String, Value>,
    /// Input name → the node feeding it.
    pub links: HashMap<String, u64>,
}

/// A graph parameter: one named control in the layer properties that sets
/// node parameters ("Rust amount" moving a mask's level and a blend's
/// opacity at once).
#[derive(Clone, Debug, PartialEq)]
pub struct GraphParam {
    pub name: String,
    /// The heading it shows under in the layer properties ("": none).
    pub group: String,
    /// The node parameters it sets: (node, parameter name). The first one
    /// gives its kind and current value.
    pub binds: Vec<(u64, String)>,
    /// What "reset" goes back to (the value when it was made).
    pub default: Option<Value>,
}

/// What a node of a folder's live graph stands for (see [`smart`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Role {
    /// The layer's smart mask (a Smart Mask node).
    Mask,
    /// One channel of the layer (a Layer node; its `channel` value).
    Channel(usize),
    /// The folder's output for a usage (an Output node).
    Output(usize),
    /// The node feeding the layer's mask when it isn't the smart mask; the
    /// mask graph (an app graph id) renders what it feeds.
    Source(u64),
}

/// Ties a node of a folder's live graph to a layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tag {
    pub node: u64,
    /// The layer (the folder itself for outputs).
    pub layer: u64,
    pub role: Role,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Graph {
    pub name: String,
    /// Texture size every node renders at.
    pub res: u32,
    /// `res` follows the texture set the graph's layer is in.
    pub match_set: bool,
    pub nodes: Vec<Node>,
    pub next_id: u64,
    /// The graph's parameters, in the order the layer properties show them.
    pub params: Vec<GraphParam>,
    /// A folder's live graph: which nodes are its layers.
    pub tags: Vec<Tag>,
}

impl Graph {
    pub fn new(name: &str) -> Graph {
        Graph { name: name.to_string(), res: 1024, match_set: false, nodes: Vec::new(), next_id: 1, params: Vec::new(), tags: Vec::new() }
    }

    /// The node standing for `role` of layer `layer` (any source for `Source`).
    pub fn tagged(&self, layer: u64, role: Role) -> Option<u64> {
        self.tags
            .iter()
            .find(|t| t.layer == layer && (t.role == role || matches!((t.role, role), (Role::Source(_), Role::Source(_)))))
            .map(|t| t.node)
            .filter(|n| self.node(*n).is_some())
    }

    /// What `node` stands for in a folder's live graph.
    pub fn tag_of(&self, node: u64) -> Option<Tag> {
        self.tags.iter().find(|t| t.node == node && !matches!(t.role, Role::Source(_))).copied()
    }

    /// The graph parameter that sets `node`'s parameter `name`.
    pub fn param_of(&self, node: u64, name: &str) -> Option<usize> {
        self.params.iter().position(|p| p.binds.iter().any(|(n, k)| *n == node && k == name))
    }

    /// Whether any graph parameter sets one of `node`'s parameters.
    pub fn is_bound(&self, node: u64) -> bool {
        self.params.iter().any(|p| p.binds.iter().any(|(n, _)| *n == node))
    }

    /// A new graph parameter for `node`'s parameter `name`, named after
    /// them; returns its index. An already bound parameter keeps its own.
    pub fn expose(&mut self, node: u64, name: &str, lib: &Library) -> usize {
        if let Some(i) = self.param_of(node, name) {
            return i;
        }
        let n = self.node(node);
        let def_name = n.and_then(|n| lib.find(&n.def)).map_or("Node".to_string(), |d| d.name.clone());
        let default = n.and_then(|n| lib.value(n, name));
        let mut label = format!("{def_name} {}", name.replace('_', " "));
        let mut k = 2;
        while self.params.iter().any(|p| p.name == label) {
            label = format!("{def_name} {} {k}", name.replace('_', " "));
            k += 1;
        }
        self.params.push(GraphParam { name: label, group: String::new(), binds: vec![(node, name.to_string())], default });
        self.params.len() - 1
    }

    /// Makes graph parameter `index` also set `node`'s parameter `name`
    /// (taking it from any other graph parameter).
    pub fn bind(&mut self, index: usize, node: u64, name: &str, lib: &Library) {
        self.unbind(node, name);
        let Some(p) = self.params.get_mut(index) else { return };
        p.binds.push((node, name.to_string()));
        // The new node follows the parameter's value.
        if let Some((_, v)) = self.param_value(index, lib) {
            self.set_param(index, v, lib);
        }
    }

    /// Stops any graph parameter setting `node`'s parameter `name`; a
    /// parameter left with nothing to set goes away.
    pub fn unbind(&mut self, node: u64, name: &str) {
        for p in &mut self.params {
            p.binds.retain(|(n, k)| !(*n == node && k == name));
        }
        self.params.retain(|p| !p.binds.is_empty());
    }

    /// The definition and current value of graph parameter `index` (from
    /// the first node parameter it sets that still exists).
    pub fn param_value(&self, index: usize, lib: &Library) -> Option<(def::Param, Value)> {
        self.params.get(index)?.binds.iter().find_map(|(nid, name)| {
            let node = self.node(*nid)?;
            let p = lib.find(&node.def)?.params.iter().find(|p| &p.name == name)?.clone();
            let v = lib.value(node, name)?;
            Some((p, v))
        })
    }

    /// Sets graph parameter `index`: every node parameter it drives that
    /// takes this kind of value (numbers clamped to each one's range).
    pub fn set_param(&mut self, index: usize, v: Value, lib: &Library) {
        let Some(p) = self.params.get(index) else { return };
        for (nid, name) in p.binds.clone() {
            let Some(node) = self.node_mut(nid) else { continue };
            let Some(param) = lib.find(&node.def).and_then(|d| d.params.iter().find(|q| q.name == name)) else { continue };
            if std::mem::discriminant(&param.default) != std::mem::discriminant(&v) {
                continue;
            }
            let v = match (&param.kind, &v) {
                (ParamKind::Float { min, max }, Value::Float(f)) => Value::Float(f.clamp(min.min(*max), max.max(*min))),
                (ParamKind::Int { min, max }, Value::Int(i)) => Value::Int(*i.clamp(min.min(max), max.max(min))),
                (ParamKind::Enum(opts), Value::Enum(i)) => Value::Enum((*i).min(opts.len().saturating_sub(1))),
                _ => v.clone(),
            };
            node.params.insert(name, v);
        }
    }

    /// A starter graph: noise through a gradient map into base color, and
    /// the same noise into roughness.
    pub fn starter(name: &str) -> Graph {
        let mut g = Graph::new(name);
        let noise = g.add("noise", [0.0, 0.0]);
        let map = g.add("gradient_map", [220.0, -60.0]);
        let color = g.add("output", [440.0, -60.0]);
        let levels = g.add("levels", [220.0, 100.0]);
        let rough = g.add("output", [440.0, 100.0]);
        g.connect(noise, map, "input");
        g.connect(map, color, "input");
        g.connect(noise, levels, "input");
        g.connect(levels, rough, "input");
        g.node_mut(levels).unwrap().params.insert("out_low".into(), Value::Float(0.35));
        g.node_mut(levels).unwrap().params.insert("out_high".into(), Value::Float(0.75));
        g.node_mut(rough).unwrap().params.insert("usage".into(), Value::Enum(1));
        g
    }

    pub fn add(&mut self, def: &str, pos: [f32; 2]) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.push(Node { id, def: def.to_string(), pos, params: HashMap::new(), links: HashMap::new() });
        id
    }

    pub fn node(&self, id: u64) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }

    pub fn node_mut(&mut self, id: u64) -> Option<&mut Node> {
        self.nodes.iter_mut().find(|n| n.id == id)
    }

    pub fn remove(&mut self, id: u64) {
        self.nodes.retain(|n| n.id != id);
        for p in &mut self.params {
            p.binds.retain(|(n, _)| *n != id);
        }
        self.params.retain(|p| !p.binds.is_empty());
        for n in &mut self.nodes {
            n.links.retain(|_, from| *from != id);
        }
        // A mask source keeps its tag: the mask graph stays the layer's.
        self.tags.retain(|t| t.node != id || matches!(t.role, Role::Source(_)));
    }

    /// Whether `to` feeds `from`, directly or not (a link from `from` to
    /// `to` would close a loop).
    fn feeds(&self, to: u64, from: u64) -> bool {
        let mut stack = vec![from];
        let mut seen = Vec::new();
        while let Some(id) = stack.pop() {
            if id == to {
                return true;
            }
            if seen.contains(&id) {
                continue;
            }
            seen.push(id);
            if let Some(n) = self.node(id) {
                stack.extend(n.links.values().copied());
            }
        }
        false
    }

    /// Links `from`'s output to `to`'s input; refuses loops.
    pub fn connect(&mut self, from: u64, to: u64, input: &str) -> bool {
        if from == to || self.feeds(to, from) {
            return false;
        }
        match self.node_mut(to) {
            Some(n) => {
                n.links.insert(input.to_string(), from);
                true
            }
            None => false,
        }
    }

    /// Nodes ordered so every node comes after the nodes feeding it.
    pub fn order(&self) -> Vec<u64> {
        fn visit(g: &Graph, id: u64, done: &mut Vec<u64>, active: &mut Vec<u64>) {
            if done.contains(&id) || active.contains(&id) {
                return;
            }
            active.push(id);
            if let Some(n) = g.node(id) {
                let mut inputs: Vec<u64> = n.links.values().copied().collect();
                inputs.sort_unstable();
                for i in inputs {
                    visit(g, i, done, active);
                }
            }
            active.pop();
            done.push(id);
        }
        let mut done = Vec::new();
        for n in &self.nodes {
            visit(self, n.id, &mut done, &mut Vec::new());
        }
        done
    }

    /// The Output node feeding each usage (the first one, if several do).
    pub fn outputs(&self, lib: &Library) -> [Option<u64>; USAGES.len()] {
        let mut out = [None; USAGES.len()];
        for n in self.nodes.iter().filter(|n| n.def == OUTPUT_DEF) {
            let usage = match lib.value(n, "usage") {
                Some(Value::Enum(i)) => i,
                _ => 0,
            };
            if usage < out.len() && out[usage].is_none() {
                out[usage] = Some(n.id);
            }
        }
        out
    }

    pub fn to_json(&self, lib: &Library) -> Json {
        let nodes: Vec<Json> = self
            .nodes
            .iter()
            .map(|n| {
                let def = lib.find(&n.def);
                let params: serde_json::Map<String, Json> = n
                    .params
                    .iter()
                    .map(|(k, v)| {
                        let kind = def.and_then(|d| d.params.iter().find(|p| &p.name == k)).map(|p| &p.kind);
                        (k.clone(), value_json(v, kind))
                    })
                    .collect();
                let links: serde_json::Map<String, Json> = n.links.iter().map(|(k, v)| (k.clone(), json!(v))).collect();
                json!({"id": n.id, "def": n.def, "pos": n.pos, "params": params, "links": links})
            })
            .collect();
        let params: Vec<Json> = self
            .params
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let binds: Vec<Json> = p.binds.iter().map(|(n, k)| json!([n, k])).collect();
                let kind = self.param_value(i, lib).map(|(d, _)| d.kind);
                let default = p.default.as_ref().map_or(Json::Null, |v| value_json(v, kind.as_ref()));
                json!({"name": p.name, "group": p.group, "binds": binds, "default": default})
            })
            .collect();
        let tags: Vec<Json> = self
            .tags
            .iter()
            .map(|t| {
                let (role, i) = match t.role {
                    Role::Mask => ("mask", 0),
                    Role::Channel(c) => ("channel", c as u64),
                    Role::Output(u) => ("output", u as u64),
                    Role::Source(d) => ("source", d),
                };
                json!({"node": t.node, "layer": t.layer, "role": role, "i": i})
            })
            .collect();
        let mut out = json!({"name": self.name, "res": self.res, "match_set": self.match_set, "next_id": self.next_id, "nodes": nodes, "params": params});
        if !tags.is_empty() {
            out["tags"] = json!(tags);
        }
        out
    }

    /// Reads a graph; parameters are typed by the definitions in `lib`
    /// (values of unknown nodes or parameters are dropped).
    pub fn from_json(v: &Json, lib: &Library) -> Result<Graph, String> {
        let mut g = Graph::new(v["name"].as_str().unwrap_or("Graph"));
        g.res = v["res"].as_u64().map_or(1024, |r| (r as u32).clamp(64, 4096));
        g.match_set = v["match_set"].as_bool().unwrap_or(false);
        for n in v["nodes"].as_array().ok_or("graph has no nodes list")? {
            let id = n["id"].as_u64().ok_or("node without id")?;
            let def = n["def"].as_str().ok_or("node without def")?.to_string();
            let pos = [n["pos"][0].as_f64().unwrap_or(0.0) as f32, n["pos"][1].as_f64().unwrap_or(0.0) as f32];
            let mut params = HashMap::new();
            if let (Some(d), Some(map)) = (lib.find(&def), n["params"].as_object()) {
                for (k, jv) in map {
                    if let Some(p) = d.params.iter().find(|p| &p.name == k)
                        && let Some(val) = json_value(jv, &p.kind) {
                            params.insert(k.clone(), val);
                        }
                }
            }
            let links = n["links"].as_object().map_or_else(HashMap::new, |m| {
                m.iter().filter_map(|(k, v)| Some((k.clone(), v.as_u64()?))).collect()
            });
            g.next_id = g.next_id.max(id + 1);
            g.nodes.push(Node { id, def, pos, params, links });
        }
        g.next_id = g.next_id.max(v["next_id"].as_u64().unwrap_or(1));
        let pair = |e: &Json| Some((e[0].as_u64()?, e[1].as_str()?.to_string()));
        for p in v["params"].as_array().into_iter().flatten() {
            let binds: Vec<(u64, String)> = p["binds"].as_array().into_iter().flatten().filter_map(pair).collect();
            if binds.is_empty() {
                continue;
            }
            let kind = binds.iter().find_map(|(n, k)| {
                let node = g.node(*n)?;
                Some(lib.find(&node.def)?.params.iter().find(|q| &q.name == k)?.kind.clone())
            });
            let default = kind.and_then(|k| json_value(&p["default"], &k));
            g.params.push(GraphParam {
                name: p["name"].as_str().unwrap_or("Parameter").to_string(),
                group: p["group"].as_str().unwrap_or("").to_string(),
                binds,
                default,
            });
        }
        for t in v["tags"].as_array().into_iter().flatten() {
            let (Some(node), Some(layer), i) = (t["node"].as_u64(), t["layer"].as_u64(), t["i"].as_u64().unwrap_or(0)) else { continue };
            let role = match t["role"].as_str() {
                Some("mask") => Role::Mask,
                Some("channel") => Role::Channel(i as usize),
                Some("output") => Role::Output(i as usize),
                Some("source") => Role::Source(i),
                _ => continue,
            };
            g.tags.push(Tag { node, layer, role });
        }
        // Graphs from before named parameters: one per exposed node parameter.
        for e in v["exposed"].as_array().into_iter().flatten() {
            if let Some((n, k)) = pair(e) {
                g.expose(n, &k, lib);
            }
        }
        Ok(g)
    }
}

fn value_json(v: &Value, kind: Option<&ParamKind>) -> Json {
    match v {
        Value::Float(f) => json!(f),
        Value::Int(i) => json!(i),
        Value::Bool(b) => json!(b),
        Value::Color(c) => json!(c),
        Value::Enum(i) => match kind {
            Some(ParamKind::Enum(opts)) => json!(opts.get(*i).cloned().unwrap_or_default()),
            _ => json!(i),
        },
        Value::Image(p) => json!(p.as_ref().map(|p| p.display().to_string())),
        // Custom bake maps by id ("custom:3"), built-in ones by name.
        Value::MeshMap(i) if *i >= crate::engine::CUSTOM_MAP_VALUE => json!(format!("custom:{}", i - crate::engine::CUSTOM_MAP_VALUE)),
        Value::MeshMap(i) => json!(crate::engine::MAPS.get(*i).map_or("", |m| m.name)),
    }
}

fn json_value(v: &Json, kind: &ParamKind) -> Option<Value> {
    Some(match kind {
        ParamKind::Float { .. } => Value::Float(v.as_f64()? as f32),
        ParamKind::Int { .. } => Value::Int(v.as_i64()? as i32),
        ParamKind::Bool => Value::Bool(v.as_bool()?),
        ParamKind::Color => {
            let a = v.as_array()?;
            Value::Color([0, 1, 2].map(|i| a.get(i).and_then(Json::as_f64).unwrap_or(0.0) as f32))
        }
        ParamKind::Enum(opts) => match v {
            Json::String(s) => Value::Enum(opts.iter().position(|o| o == s)?),
            _ => Value::Enum((v.as_u64()? as usize).min(opts.len().saturating_sub(1))),
        },
        ParamKind::Image => Value::Image(v.as_str().map(PathBuf::from)),
        ParamKind::MeshMap => {
            let name = v.as_str()?;
            match name.strip_prefix("custom:") {
                Some(id) => Value::MeshMap(crate::engine::CUSTOM_MAP_VALUE + id.parse::<usize>().ok()?),
                None => Value::MeshMap(crate::engine::MAPS.iter().position(|m| m.name == name)?),
            }
        }
    })
}

/// Every node definition: the built-ins, then user nodes from a folder
/// (`user/<file stem>`), with the errors of user files that didn't parse.
pub struct Library {
    pub defs: Vec<NodeDef>,
    /// User node file → parse error.
    pub errors: Vec<(PathBuf, String)>,
    pub user_dir: Option<PathBuf>,
}

impl Library {
    pub fn load(user_dir: Option<PathBuf>) -> Library {
        let mut lib = Library { defs: Vec::new(), errors: Vec::new(), user_dir };
        for (id, src) in builtin::BUILTIN {
            lib.defs.push(def::parse(id, src).unwrap_or_else(|e| panic!("built-in node {id}: {e}")));
        }
        lib.reload_user();
        lib
    }

    /// Rereads the user folder's `.wgsl` files.
    pub fn reload_user(&mut self) {
        self.defs.retain(|d| !d.id.starts_with("user/"));
        self.errors.clear();
        let Some(dir) = &self.user_dir else { return };
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        let mut files: Vec<PathBuf> =
            entries.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "wgsl")).collect();
        files.sort();
        for path in files {
            let id = format!("user/{}", path.file_stem().unwrap_or_default().to_string_lossy());
            match std::fs::read_to_string(&path).map_err(|e| e.to_string()).and_then(|src| def::parse(&id, &src)) {
                Ok(d) => self.defs.push(d),
                Err(e) => self.errors.push((path, e)),
            }
        }
    }

    pub fn find(&self, id: &str) -> Option<&NodeDef> {
        self.defs.iter().find(|d| d.id == id)
    }

    /// The file a user node lives in.
    pub fn user_path(&self, id: &str) -> Option<PathBuf> {
        Some(self.user_dir.as_ref()?.join(format!("{}.wgsl", id.strip_prefix("user/")?)))
    }

    /// Writes a user node file and reloads the folder. The file is written
    /// even when it doesn't parse, so work isn't lost; the error comes back.
    pub fn save_user(&mut self, stem: &str, source: &str) -> Result<String, String> {
        let dir = self.user_dir.clone().ok_or("no user node folder")?;
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let id = format!("user/{stem}");
        std::fs::write(dir.join(format!("{stem}.wgsl")), source).map_err(|e| e.to_string())?;
        self.reload_user();
        def::parse(&id, source).map(|_| id)
    }

    /// A node's value for a parameter: its own if set and of the right
    /// kind, else the definition's default.
    pub fn value(&self, node: &Node, name: &str) -> Option<Value> {
        let p = self.find(&node.def)?.params.iter().find(|p| p.name == name)?;
        let own = node.params.get(name).filter(|v| std::mem::discriminant(*v) == std::mem::discriminant(&p.default));
        Some(own.cloned().unwrap_or_else(|| p.default.clone()))
    }

    /// Categories in display order, each with its node ids.
    pub fn categories(&self) -> Vec<(String, Vec<String>)> {
        let order = ["Generators", "Material", "Adjust", "Blend", "Filters", "Inputs", "Outputs"];
        let mut cats: Vec<(String, Vec<String>)> = Vec::new();
        for d in &self.defs {
            match cats.iter_mut().find(|(c, _)| *c == d.category) {
                Some((_, ids)) => ids.push(d.id.clone()),
                None => cats.push((d.category.clone(), vec![d.id.clone()])),
            }
        }
        cats.sort_by_key(|(c, _)| order.iter().position(|o| o == c).unwrap_or(order.len()));
        cats
    }
}

/// A new user node's starting source.
pub fn template(name: &str) -> String {
    format!(
        "// @name {name}\n// @category Custom\n// @description What this node does.\n// @output color\n// @input input: color = 0.5\n// @param amount: float(0, 1) = 0.5\n\nfn node(uv: vec2<f32>) -> vec4<f32> {{\n    let c = in_input(uv);\n    return vec4<f32>(c.rgb * params.amount, 1.0);\n}}\n"
    )
}

/// Reads a graph file (`.mpgraph`, JSON).
pub fn load_file(path: &Path, lib: &Library) -> Result<Graph, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let v: Json = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    Graph::from_json(&v, lib)
}

pub fn save_file(path: &Path, graph: &Graph, lib: &Library) -> Result<(), String> {
    let text = serde_json::to_string_pretty(&graph.to_json(lib)).map_err(|e| e.to_string())?;
    std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_builtin_parses_and_the_starter_round_trips() {
        let lib = Library::load(None);
        assert_eq!(lib.defs.len(), builtin::BUILTIN.len());
        let g = Graph::starter("Test");
        let outs = g.outputs(&lib);
        assert!(outs[0].is_some() && outs[1].is_some(), "{outs:?}");
        let back = Graph::from_json(&g.to_json(&lib), &lib).unwrap();
        assert_eq!(back.nodes.len(), g.nodes.len());
        assert_eq!(back.outputs(&lib), outs);
        let levels = back.nodes.iter().find(|n| n.def == "levels").unwrap();
        assert_eq!(lib.value(levels, "out_low"), Some(Value::Float(0.35)));
        // Topological order: every node after its inputs.
        let order = back.order();
        for n in &back.nodes {
            let at = order.iter().position(|&i| i == n.id).unwrap();
            for from in n.links.values() {
                assert!(order.iter().position(|i| i == from).unwrap() < at);
            }
        }
    }

    #[test]
    fn graph_parameters_drive_several_nodes_and_round_trip() {
        let lib = Library::load(None);
        let mut g = Graph::new("Params");
        let a = g.add("levels", [0.0; 2]);
        let b = g.add("histogram_scan", [0.0; 2]);
        let i = g.expose(a, "out_low", &lib);
        assert_eq!(g.params[i].name, "Levels out low");
        assert_eq!(g.expose(a, "out_low", &lib), i, "already exposed");
        g.bind(i, b, "position", &lib);
        g.params[i].name = "Wear".into();
        g.params[i].group = "Wear".into();
        g.set_param(i, Value::Float(0.3), &lib);
        assert_eq!(lib.value(g.node(a).unwrap(), "out_low"), Some(Value::Float(0.3)));
        assert_eq!(lib.value(g.node(b).unwrap(), "position"), Some(Value::Float(0.3)));
        assert_eq!(g.param_of(b, "position"), Some(i));
        // A different kind of value leaves other kinds alone.
        let c = g.add("noise", [0.0; 2]);
        g.bind(i, c, "seed", &lib);
        g.set_param(i, Value::Float(0.6), &lib);
        assert_ne!(lib.value(g.node(c).unwrap(), "seed"), Some(Value::Float(0.6)));

        let back = Graph::from_json(&g.to_json(&lib), &lib).unwrap();
        assert_eq!(back.params, g.params);
        // Removing the nodes removes the parameter.
        let mut h = back.clone();
        h.remove(a);
        h.remove(b);
        h.remove(c);
        assert!(h.params.is_empty());

        // An old graph's exposed parameters become graph parameters.
        let mut old = g.to_json(&lib);
        old["params"] = json!([]);
        old["exposed"] = json!([[a, "out_low"]]);
        let o = Graph::from_json(&old, &lib).unwrap();
        assert_eq!(o.params.len(), 1);
        assert_eq!(o.params[0].binds, vec![(a, "out_low".to_string())]);
    }

    #[test]
    fn connect_refuses_loops() {
        let mut g = Graph::new("Loop");
        let a = g.add("invert", [0.0; 2]);
        let b = g.add("invert", [0.0; 2]);
        assert!(g.connect(a, b, "input"));
        assert!(!g.connect(b, a, "input"));
        assert!(!g.connect(a, a, "input"));
    }
}
