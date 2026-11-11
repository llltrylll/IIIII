//! Smart materials as material graphs.
//!
//! A smart material lives in the layer stack as a folder of fill layers with
//! smart masks; its live graph shows the same layers as nodes: a Smart Mask
//! node per masked layer and a Layer node per channel it writes, chained
//! bottom to top into Output nodes. Tags tie the nodes to their layers, so
//! an edit on either side reaches the other: [`update_view`] writes the
//! layers into the graph, [`read_view`] the artist's graph edits into the
//! layers. Nodes of the artist's own feeding a layer's mask become that
//! layer's mask graph ([`mask_subgraph`]).
//!
//! The same builder makes a smart material as one graph fill layer
//! ([`graph`]), with what an artist tunes (a layer's opacity, color,
//! roughness, its mask's level, a scan's tiling) as material parameters
//! grouped under the layer's name.

use super::{Graph, Library, MASK_USAGE, Node, Role, Tag, Value};
use crate::engine::{Blend, CHANNELS, FillParams, Generator};
use crate::presets::{MICRO_DETAIL, Scan, SmartLayer, SmartMaterial};
use std::path::PathBuf;

/// Smart mask inputs a Smart Mask node mixes.
pub const MASK_INPUTS: usize = 4;

/// Columns and rows of the node layout.
const COLUMN: f32 = 260.0;
const ROW: f32 = 190.0;

/// A Layer node's `channel` value → the paint channel it stands for.
pub const NODE_CHANNEL: [usize; 5] = [0, 1, 2, 3, 5];
/// A Layer node's `channel` value → the Output usage it feeds.
const NODE_USAGE: [usize; 5] = [0, 1, 2, 3, 6];

/// A fill layer as its live graph sees it.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerSpec {
    pub id: u64,
    pub name: String,
    pub opacity: f32,
    pub blend: Blend,
    pub fill: FillParams,
    pub generator: Option<Generator>,
    /// Images per paint channel, as graph image paths (scans in the assets
    /// folder; other images don't show in the graph).
    pub images: [Option<PathBuf>; CHANNELS],
    /// The mask comes from a node of the artist's (the layer's `Source`
    /// tag), rendered by a mask graph.
    pub custom_mask: bool,
}

impl LayerSpec {
    /// A smart material's layer (`id`: any id unique in the material).
    pub fn of_smart(id: u64, l: &SmartLayer) -> LayerSpec {
        let mut images: [Option<PathBuf>; CHANNELS] = Default::default();
        if let Some(scan) = l.scan {
            for &(ch, file) in Scan::FILES.iter() {
                if scan_file_exists(scan, file) {
                    // Inside the assets folder, with '/' so project files move between systems.
                    images[ch] = Some(format!("pbr/{}/{file}", scan.id).into());
                }
            }
        }
        LayerSpec {
            id,
            name: l.name.to_string(),
            opacity: l.opacity,
            blend: l.blend,
            fill: l.fill(),
            generator: l.generator(),
            images,
            custom_mask: false,
        }
    }
}

/// The graph of a smart material as one graph fill layer (its layers plus
/// the fine detail every smart material gets on top).
pub fn graph(sm: &SmartMaterial, lib: &Library) -> Graph {
    graph_of(sm.name, sm.layers.iter().chain(MICRO_DETAIL.iter()), lib)
}

/// A graph from a stack of smart layers, bottom first, with their settings
/// as material parameters.
pub fn graph_of<'a>(name: &str, layers: impl Iterator<Item = &'a SmartLayer>, lib: &Library) -> Graph {
    let layers: Vec<&SmartLayer> = layers.collect();
    let specs: Vec<LayerSpec> = layers.iter().enumerate().map(|(k, l)| LayerSpec::of_smart(k as u64 + 1, l)).collect();
    let mut g = Graph::new(name);
    // Smart materials cover whole models: masks need the detail.
    g.res = 2048;
    g.match_set = true;
    update_view(&mut g, 0, &specs, lib);

    let mut groups: Vec<String> = Vec::new();
    for (l, s) in layers.iter().zip(&specs) {
        let nodes: [Option<u64>; 5] = std::array::from_fn(|k| g.tagged(s.id, Role::Channel(k)));
        let mask = g.tagged(s.id, Role::Mask);
        // The layer's parameters, under its name (numbered when repeated).
        let mut group = l.name.to_string();
        let mut k = 2;
        while groups.contains(&group) {
            group = format!("{} {k}", l.name);
            k += 1;
        }
        groups.push(group.clone());
        let param = |g: &mut Graph, binds: &[(u64, &str)], label: &str| {
            let Some(&(first, key)) = binds.first() else { return };
            let i = g.expose(first, key, lib);
            for &(n, key) in &binds[1..] {
                g.bind(i, n, key, lib);
            }
            g.params[i].name = label.to_string();
            g.params[i].group = group.clone();
        };
        let all: Vec<(u64, &str)> = nodes.iter().flatten().map(|&n| (n, "opacity")).collect();
        param(&mut g, &all, "Opacity");
        if l.scan.is_none() {
            if let Some(n) = nodes[0] {
                param(&mut g, &[(n, "color")], "Color");
            }
            if let Some(n) = nodes[1] {
                param(&mut g, &[(n, "value")], "Roughness");
            }
            if let Some(n) = nodes[2].filter(|_| l.metal.is_some()) {
                param(&mut g, &[(n, "value")], "Metallic");
            }
        } else {
            let tiled: Vec<(u64, &str)> = nodes.iter().flatten().map(|&n| (n, "tiling")).collect();
            param(&mut g, &tiled, "Tiling");
        }
        if let Some(n) = nodes[3].filter(|_| l.height.is_some() || l.scan.is_some()) {
            param(&mut g, &[(n, "value")], "Height");
        }
        if let Some(m) = mask {
            param(&mut g, &[(m, "level")], "Mask level");
            param(&mut g, &[(m, "scale")], "Mask scale");
        }
    }
    // One graph, not tied to layers.
    g.tags.clear();
    g
}

/// Brings a folder's live graph in line with its layers (`specs`, bottom
/// first): nodes for new layers and channels, values and links from the
/// layers, nodes of removed ones gone. Nodes already showing what their
/// layer holds are left alone (an artist's slot order, positions), and the
/// artist's own nodes stay.
pub fn update_view(g: &mut Graph, folder: u64, specs: &[LayerSpec], lib: &Library) {
    let ids: Vec<u64> = specs.iter().map(|s| s.id).collect();
    let gone: Vec<u64> = g
        .tags
        .iter()
        .filter(|t| match t.role {
            Role::Mask | Role::Channel(_) => !ids.contains(&t.layer),
            Role::Output(_) => t.layer != folder,
            Role::Source(_) => false,
        })
        .map(|t| t.node)
        .collect();
    for n in gone {
        g.remove(n);
    }
    // Tags of nodes deleted some other way.
    let live: Vec<u64> = g.nodes.iter().map(|n| n.id).collect();
    g.tags.retain(|t| matches!(t.role, Role::Source(_)) || live.contains(&t.node));

    let mut top: [Option<u64>; 5] = [None; 5];
    for (k, s) in specs.iter().enumerate() {
        let x = k as f32 * COLUMN;
        let mask = match &s.generator {
            Some(mg) if !s.custom_mask => {
                let id = ensure(g, s.id, Role::Mask, "smart_mask", [x, -ROW * 1.2]);
                rewrite(g, lib, id, |n| write_mask(lib, n, mg));
                Some(id)
            }
            _ => {
                if let Some(id) = g.tagged(s.id, Role::Mask) {
                    g.remove(id);
                }
                None
            }
        };
        let source = if s.custom_mask { g.tagged(s.id, Role::Source(0)) } else { mask };
        for (nk, &ch) in NODE_CHANNEL.iter().enumerate() {
            if !s.fill.mat.enabled[ch] {
                if let Some(id) = g.tagged(s.id, Role::Channel(nk)) {
                    g.remove(id);
                }
                continue;
            }
            let id = ensure(g, s.id, Role::Channel(nk), "layer", [x, nk as f32 * ROW]);
            rewrite(g, lib, id, |n| write_channel(n, s, nk));
            link(g, source, id, "mask");
            link(g, top[nk], id, "below");
            top[nk] = Some(id);
        }
    }
    let end = specs.len() as f32 * COLUMN;
    for (nk, &usage) in NODE_USAGE.iter().enumerate() {
        let role = Role::Output(usage);
        match top[nk] {
            Some(from) => {
                let o = ensure(g, folder, role, "output", [end, nk as f32 * ROW]);
                rewrite(g, lib, o, |n| {
                    n.params.insert("usage".into(), Value::Enum(usage));
                });
                link(g, Some(from), o, "input");
            }
            None => {
                if let Some(o) = g.tagged(folder, role) {
                    g.remove(o);
                }
            }
        }
    }
}

/// Writes the artist's edits in a folder's live graph into the layers'
/// specs; `seen` is the graph as last brought in line. A Layer node's mask
/// wired to a node of the artist's makes that node the layer's mask source
/// (its `Source` tag, with graph id 0 until the app gives it a mask graph).
pub fn read_view(g: &mut Graph, seen: &Graph, specs: &mut [LayerSpec], lib: &Library) {
    let tags = g.tags.clone();
    for t in tags {
        let Some(s) = specs.iter_mut().find(|s| s.id == t.layer) else { continue };
        let (Some(n), Some(before)) = (g.node(t.node), seen.node(t.node)) else { continue };
        match t.role {
            Role::Mask => {
                if n.params != before.params
                    && let Some(mg) = s.generator.as_mut() {
                        read_mask(lib, n, Some(before), mg);
                    }
            }
            Role::Channel(k) => {
                if n.params != before.params {
                    read_channel(lib, n, s, k);
                }
                let (now, was) = (n.links.get("mask").copied(), before.links.get("mask").copied());
                if now == was {
                    continue;
                }
                let own = g.tagged(s.id, Role::Mask);
                match now {
                    None => {
                        s.generator = None;
                        s.custom_mask = false;
                    }
                    Some(from) if Some(from) == own => s.custom_mask = false,
                    Some(from) => {
                        s.custom_mask = true;
                        s.generator = None;
                        // The layer's smart mask node stays, as one of the
                        // artist's (the new mask may read it).
                        g.tags.retain(|x| !(x.layer == s.id && x.role == Role::Mask));
                        match g.tags.iter_mut().find(|x| x.layer == s.id && matches!(x.role, Role::Source(_))) {
                            Some(x) => x.node = from,
                            None => g.tags.push(Tag { node: from, layer: s.id, role: Role::Source(0) }),
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

/// The nodes feeding `source`, plus an Output (Mask) it feeds: a layer's
/// mask graph cut from its folder's live graph. Node ids stay, so an edit
/// re-renders only what it changes.
pub fn mask_subgraph(g: &Graph, source: u64, name: &str) -> Graph {
    /// The Output node's id, above any the live graph hands out.
    const OUTPUT: u64 = 1 << 40;
    let mut keep: Vec<u64> = Vec::new();
    let mut stack = vec![source];
    while let Some(id) = stack.pop() {
        if keep.contains(&id) {
            continue;
        }
        if let Some(n) = g.node(id) {
            keep.push(id);
            stack.extend(n.links.values().copied());
        }
    }
    let mut m = Graph::new(name);
    m.res = g.res;
    m.match_set = g.match_set;
    for n in g.nodes.iter().filter(|n| keep.contains(&n.id)) {
        let mut c = n.clone();
        c.links.retain(|_, from| keep.contains(from));
        m.nodes.push(c);
    }
    let at = g.node(source).map_or([0.0, 0.0], |n| [n.pos[0] + COLUMN, n.pos[1]]);
    let mut out = Node { id: OUTPUT, def: super::OUTPUT_DEF.into(), pos: at, params: Default::default(), links: Default::default() };
    out.params.insert("usage".into(), Value::Enum(MASK_USAGE));
    out.links.insert("input".into(), source);
    m.nodes.push(out);
    m.next_id = OUTPUT + 1;
    m
}

/// The node for `role` of `layer`, added at `pos` when there's none.
fn ensure(g: &mut Graph, layer: u64, role: Role, def: &str, pos: [f32; 2]) -> u64 {
    if let Some(id) = g.tagged(layer, role) {
        return id;
    }
    let id = g.add(def, pos);
    g.tags.push(Tag { node: id, layer, role });
    id
}

/// Applies `write` to node `id` when that changes what it does.
fn rewrite(g: &mut Graph, lib: &Library, id: u64, write: impl FnOnce(&mut Node)) {
    let Some(n) = g.node(id) else { return };
    let mut c = n.clone();
    write(&mut c);
    if c.params.iter().any(|(k, v)| lib.value(n, k).as_ref() != Some(v)) {
        *g.node_mut(id).expect("exists") = c;
    }
}

/// Feeds `to`'s `input` from `from` (none: unlinked).
fn link(g: &mut Graph, from: Option<u64>, to: u64, input: &str) {
    let current = g.node(to).and_then(|n| n.links.get(input).copied());
    if current == from {
        return;
    }
    if from.is_some_and(|f| g.connect(f, to, input)) {
        return;
    }
    if let Some(n) = g.node_mut(to) {
        n.links.remove(input);
    }
}

fn float(lib: &Library, n: &Node, key: &str) -> Option<f32> {
    match lib.value(n, key)? {
        Value::Float(f) => Some(f),
        _ => None,
    }
}

fn color(lib: &Library, n: &Node, key: &str) -> Option<[f32; 3]> {
    match lib.value(n, key)? {
        Value::Color(c) => Some(c),
        _ => None,
    }
}

/// A Layer node showing channel `k` (its `channel` value) of a layer.
fn write_channel(n: &mut Node, s: &LayerSpec, k: usize) {
    let f = &s.fill;
    let (variation, part) = match k {
        0 => (f.color_var, f.part_var),
        1 => (f.rough_var, f.part_var),
        3 => (f.height_var, 0.0),
        _ => (0.0, 0.0),
    };
    let value = match k {
        1 => f.mat.rough,
        2 => f.mat.metal,
        3 => f.mat.height,
        _ => 0.5,
    };
    let color = if k == 4 { f.mat.emission } else { f.mat.color };
    for (key, v) in [
        ("channel", Value::Enum(k)),
        ("color", Value::Color(color)),
        ("value", Value::Float(value)),
        ("variation", Value::Float(variation)),
        ("variation_scale", Value::Float(f.var_scale)),
        ("part_variation", Value::Float(part)),
        ("seed", Value::Float(f.seed)),
        ("blend", Value::Enum(s.blend as usize)),
        ("opacity", Value::Float(s.opacity)),
        ("image", Value::Image(s.images[NODE_CHANNEL[k]].clone())),
        ("tiling", Value::Float(f.tiling)),
        ("sharpness", Value::Float(f.tri_sharpness)),
    ] {
        n.params.insert(key.into(), v);
    }
}

/// What a Layer node showing channel `k` sets on its layer.
fn read_channel(lib: &Library, n: &Node, s: &mut LayerSpec, k: usize) {
    let fl = |key| float(lib, n, key);
    if let Some(v) = fl("opacity") {
        s.opacity = v;
    }
    if let Some(Value::Enum(b)) = lib.value(n, "blend") {
        s.blend = Blend::ALL.get(b).copied().unwrap_or(s.blend);
    }
    let f = &mut s.fill;
    f.var_scale = fl("variation_scale").unwrap_or(f.var_scale);
    f.seed = fl("seed").unwrap_or(f.seed);
    f.tiling = fl("tiling").unwrap_or(f.tiling);
    f.tri_sharpness = fl("sharpness").unwrap_or(f.tri_sharpness);
    let (variation, part, value) = (fl("variation"), fl("part_variation"), fl("value"));
    match k {
        0 => {
            f.mat.color = color(lib, n, "color").unwrap_or(f.mat.color);
            f.color_var = variation.unwrap_or(f.color_var);
            f.part_var = part.unwrap_or(f.part_var);
        }
        1 => {
            f.mat.rough = value.unwrap_or(f.mat.rough);
            f.rough_var = variation.unwrap_or(f.rough_var);
            f.part_var = part.unwrap_or(f.part_var);
        }
        2 => f.mat.metal = value.unwrap_or(f.mat.metal),
        3 => {
            f.mat.height = value.unwrap_or(f.mat.height);
            f.height_var = variation.unwrap_or(f.height_var);
        }
        _ => f.mat.emission = color(lib, n, "color").unwrap_or(f.mat.emission),
    }
    if let Some(Value::Image(p)) = lib.value(n, "image") {
        s.images[NODE_CHANNEL[k]] = p;
    }
}

/// A Smart Mask node's inputs: (input + 1, or 0 for none; weight) per slot.
fn slots(lib: &Library, n: &Node) -> [(usize, f32); MASK_INPUTS] {
    std::array::from_fn(|k| {
        let input = match lib.value(n, &format!("input_{}", k + 1)) {
            Some(Value::Enum(i)) => i,
            _ => 0,
        };
        (input, float(lib, n, &format!("weight_{}", k + 1)).unwrap_or(1.0))
    })
}

/// A Smart Mask node showing a generator: its strongest inputs, each kept
/// in the slot it already has.
fn write_mask(lib: &Library, n: &mut Node, mg: &Generator) {
    let mut top: Vec<(usize, f32)> = mg.weights.iter().enumerate().filter(|(_, w)| w.abs() > 1e-4).map(|(i, w)| (i + 1, *w)).collect();
    top.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
    top.truncate(MASK_INPUTS);
    let mut out: [Option<(usize, f32)>; MASK_INPUTS] = [None; MASK_INPUTS];
    for (slot, (input, _)) in slots(lib, n).iter().enumerate() {
        if let Some(p) = top.iter().position(|t| t.0 == *input) {
            out[slot] = Some(top.remove(p));
        }
    }
    for slot in out.iter_mut().filter(|s| s.is_none()) {
        if !top.is_empty() {
            *slot = Some(top.remove(0));
        }
    }
    for (k, slot) in out.iter().enumerate() {
        let (input, weight) = slot.unwrap_or((0, 1.0));
        n.params.insert(format!("input_{}", k + 1), Value::Enum(input));
        n.params.insert(format!("weight_{}", k + 1), Value::Float(weight));
    }
    for (key, v) in [
        ("level", mg.level),
        ("contrast", mg.contrast),
        ("scale", mg.scale),
        ("noise", mg.noise),
        ("edge_width", mg.edge_width),
        ("seed", mg.seed),
    ] {
        n.params.insert(key.into(), Value::Float(v));
    }
    n.params.insert("invert".into(), Value::Bool(mg.invert));
}

/// What a Smart Mask node sets on a generator. Inputs it doesn't show
/// (past its slots, custom maps) stay; one taken out of a slot since
/// `before` is cleared.
fn read_mask(lib: &Library, n: &Node, before: Option<&Node>, mg: &mut Generator) {
    let now = slots(lib, n);
    if let Some(b) = before {
        for (input, _) in slots(lib, b) {
            if input > 0 && !now.iter().any(|s| s.0 == input)
                && let Some(w) = mg.weights.get_mut(input - 1) {
                    *w = 0.0;
                }
        }
    }
    for (input, weight) in now {
        if let Some(w) = input.checked_sub(1).and_then(|i| mg.weights.get_mut(i)) {
            *w = weight;
        }
    }
    let fl = |key| float(lib, n, key);
    mg.level = fl("level").unwrap_or(mg.level);
    mg.contrast = fl("contrast").unwrap_or(mg.contrast);
    mg.scale = fl("scale").unwrap_or(mg.scale);
    mg.noise = fl("noise").unwrap_or(mg.noise);
    mg.edge_width = fl("edge_width").unwrap_or(mg.edge_width);
    mg.seed = fl("seed").unwrap_or(mg.seed);
    if let Some(Value::Bool(b)) = lib.value(n, "invert") {
        mg.invert = b;
    }
}

/// Whether a scanned set has this map (sets without metallic leave it out).
fn scan_file_exists(scan: Scan, file: &str) -> bool {
    crate::environment::asset_dir("pbr").is_some_and(|d| d.join(scan.id).join(file).is_file())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presets::SMART_MATERIALS;

    #[test]
    fn every_smart_material_converts() {
        let lib = Library::load(None);
        for sm in SMART_MATERIALS {
            // Masks fit the Smart Mask node's inputs.
            for l in sm.layers {
                if let Some(mg) = l.generator() {
                    let used = mg.weights.iter().filter(|w| w.abs() > 1e-4).count();
                    assert!(used <= MASK_INPUTS, "{} / {}: a mask mixes {used} inputs", sm.name, l.name);
                    assert!(mg.custom_ids().next().is_none(), "{} / {}: custom map inputs", sm.name, l.name);
                }
            }
            let g = graph(sm, &lib);
            let outs = g.outputs(&lib);
            assert!(outs[0].is_some() && outs[1].is_some(), "{}: base color and roughness outputs", sm.name);
            for n in &g.nodes {
                let def = lib.find(&n.def).unwrap_or_else(|| panic!("{}: unknown node {}", sm.name, n.def));
                for (k, v) in &n.params {
                    let p = def.params.iter().find(|p| &p.name == k).unwrap_or_else(|| panic!("{}: {} has no {k}", sm.name, n.def));
                    assert_eq!(std::mem::discriminant(&p.default), std::mem::discriminant(v), "{}: {}.{k}", sm.name, n.def);
                }
            }
            assert!(!g.params.is_empty(), "{} has parameters", sm.name);
            for (i, p) in g.params.iter().enumerate() {
                assert!(g.param_value(i, &lib).is_some(), "{}: parameter {} sets nothing", sm.name, p.name);
                assert!(!p.group.is_empty(), "{}: {} has a group", sm.name, p.name);
            }
            // Through a file and back.
            let back = Graph::from_json(&g.to_json(&lib), &lib).unwrap();
            assert_eq!(back.params, g.params, "{}", sm.name);
            assert_eq!(back.nodes.len(), g.nodes.len());
        }
    }

    fn specs_of(sm: &SmartMaterial) -> Vec<LayerSpec> {
        sm.layers.iter().enumerate().map(|(k, l)| LayerSpec::of_smart(k as u64 + 10, l)).collect()
    }

    /// A folder's live graph: built from the layers, steady when nothing
    /// changes, and edits going both ways.
    #[test]
    fn folder_graph_follows_layers_and_edits() {
        let lib = Library::load(None);
        let sm = SMART_MATERIALS.iter().find(|s| s.layers.len() >= 3 && s.layers.iter().filter(|l| l.generator().is_some()).count() >= 2).unwrap();
        let mut specs = specs_of(sm);
        let mut g = Graph::new(sm.name);
        update_view(&mut g, 1, &specs, &lib);
        let built = g.clone();
        update_view(&mut g, 1, &specs, &lib);
        assert_eq!(g, built, "nothing changed, nothing moves");

        // The layer's color edited in the graph reaches its spec; the
        // opacity on one channel's node reaches the layer and its other nodes.
        let top = specs.last().unwrap().id;
        let color_node = g.tagged(top, Role::Channel(0)).expect("base color node");
        let seen = g.clone();
        g.node_mut(color_node).unwrap().params.insert("color".into(), Value::Color([0.1, 0.2, 0.3]));
        g.node_mut(color_node).unwrap().params.insert("opacity".into(), Value::Float(0.25));
        read_view(&mut g, &seen, &mut specs, &lib);
        let s = specs.iter().find(|s| s.id == top).unwrap();
        assert_eq!(s.fill.mat.color, [0.1, 0.2, 0.3]);
        assert_eq!(s.opacity, 0.25);
        update_view(&mut g, 1, &specs, &lib);
        let rough = g.tagged(top, Role::Channel(1)).expect("roughness node");
        assert_eq!(lib.value(g.node(rough).unwrap(), "opacity"), Some(Value::Float(0.25)), "siblings follow");

        // A layer edited outside the graph shows in it.
        let masked = specs.iter().position(|s| s.generator.is_some()).unwrap();
        specs[masked].generator.as_mut().unwrap().level = 0.123;
        update_view(&mut g, 1, &specs, &lib);
        let m = g.tagged(specs[masked].id, Role::Mask).unwrap();
        assert_eq!(lib.value(g.node(m).unwrap(), "level"), Some(Value::Float(0.123)));

        // A slot moved in the graph keeps its place when the layers come back.
        let seen = g.clone();
        let slot1 = lib.value(g.node(m).unwrap(), "input_1");
        let n = g.node_mut(m).unwrap();
        n.params.insert("input_4".into(), slot1.clone().unwrap());
        n.params.insert("input_1".into(), Value::Enum(0));
        read_view(&mut g, &seen, &mut specs, &lib);
        update_view(&mut g, 1, &specs, &lib);
        assert_eq!(lib.value(g.node(m).unwrap(), "input_4"), slot1, "slot order kept");

        // A layer removed takes its nodes along; the artist's nodes stay.
        let noise = g.add("noise", [0.0, 600.0]);
        let removed = specs.remove(0).id;
        update_view(&mut g, 1, &specs, &lib);
        assert!(g.tagged(removed, Role::Channel(0)).is_none() && g.tagged(removed, Role::Mask).is_none());
        assert!(g.node(noise).is_some(), "own nodes stay");
        for n in &g.nodes {
            if let Some(t) = g.tag_of(n.id) {
                assert!(t.layer == 1 || specs.iter().any(|s| s.id == t.layer), "no tags of removed layers");
            }
        }
    }

    /// A node of the artist's wired into a Layer node's mask becomes the
    /// layer's mask source; the mask graph holds what feeds it.
    #[test]
    fn own_nodes_into_a_mask_make_a_mask_graph() {
        let lib = Library::load(None);
        let sm = SMART_MATERIALS.iter().find(|s| s.layers.iter().any(|l| l.generator().is_some())).unwrap();
        let mut specs = specs_of(sm);
        let mut g = Graph::new(sm.name);
        update_view(&mut g, 1, &specs, &lib);
        let k = specs.iter().position(|s| s.generator.is_some()).unwrap();
        let id = specs[k].id;
        let smart = g.tagged(id, Role::Mask).unwrap();
        let seen = g.clone();
        let blur = g.add("blur", [0.0, -500.0]);
        assert!(g.connect(smart, blur, "input"));
        let target = g.tagged(id, Role::Channel(0)).unwrap();
        assert!(g.connect(blur, target, "mask"));
        read_view(&mut g, &seen, &mut specs, &lib);
        assert!(specs[k].custom_mask && specs[k].generator.is_none());
        assert_eq!(g.tagged(id, Role::Source(0)), Some(blur));
        assert!(g.tagged(id, Role::Mask).is_none() && g.node(smart).is_some(), "the smart mask node stays as the artist's");
        update_view(&mut g, 1, &specs, &lib);
        for nk in 0..5 {
            if let Some(n) = g.tagged(id, Role::Channel(nk)) {
                assert_eq!(g.node(n).unwrap().links.get("mask"), Some(&blur), "every channel reads the new mask");
            }
        }
        let sub = mask_subgraph(&g, blur, "mask");
        assert_eq!(sub.nodes.len(), 3, "smart mask, blur, output");
        assert_eq!(sub.outputs(&lib)[MASK_USAGE].and_then(|o| sub.node(o)).and_then(|o| o.links.get("input")), Some(&blur));

        // Wired back to a (new) smart mask: a generator again.
        specs[k].custom_mask = false;
        specs[k].generator = sm.layers[k].generator();
        update_view(&mut g, 1, &specs, &lib);
        let again = g.tagged(id, Role::Mask).expect("a smart mask node again");
        assert_eq!(g.node(target).unwrap().links.get("mask"), Some(&again));
    }
}
