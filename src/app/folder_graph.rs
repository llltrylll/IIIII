//! A folder's live graph: its fill layers as Layer and Smart Mask nodes in
//! the Material Designer (see `graph::smart`). The layers stay what the
//! material is: the graph follows them while it shows, and the artist's
//! edits in it go back into them (as layer edits, so undo works the same).
//! Nodes of the artist's own wired into a layer's mask become that layer's
//! mask graph.

use super::PainterApp;
use crate::engine::{CHANNELS, Content, Projection};
use crate::graph::smart::{self, LayerSpec};
use crate::graph::{Graph, Role, Value};
use std::path::{Path, PathBuf};

impl PainterApp {
    /// Shows the live graph of folder `idx` in the Material Designer,
    /// making it on first use.
    pub(super) fn open_folder_graph(&mut self, idx: usize) {
        let Some(l) = self.engine.layers.get(idx) else { return };
        let Content::Folder(f) = &l.content else { return };
        let (folder, name) = (l.id, l.name.clone());
        let id = match f.graph.filter(|g| self.graphs.doc(*g).is_some()) {
            Some(id) => id,
            None => {
                let mut g = Graph::new(&name);
                g.match_set = true;
                g.res = self.engine.res;
                let id = self.graphs.insert(&self.rs, g, None);
                if let Content::Folder(f) = &mut self.engine.layers[idx].content {
                    f.graph = Some(id);
                }
                id
            }
        };
        self.sync_folder_graph(id, folder);
        self.open_graph(id);
        self.status = format!(
            "The graph of '{name}': its layers as nodes. Edits here change the layers; wire your own nodes into a Layer node's mask."
        );
    }

    /// Keeps the live graph the Material Designer shows in step with its folder.
    pub(super) fn sync_folder_graphs(&mut self) {
        if self.workspace != super::Workspace::Designer {
            return;
        }
        let Some(doc) = self.graphs.editor.doc else { return };
        let folder = self.engine.layers.iter().find(|l| matches!(&l.content, Content::Folder(f) if f.graph == Some(doc)));
        if let Some(folder) = folder.map(|l| l.id) {
            self.sync_folder_graph(doc, folder);
        }
    }

    /// The artist's edits since the last sync go into the layers, then the
    /// graph takes on whatever else changed in them.
    fn sync_folder_graph(&mut self, doc: u64, folder: u64) {
        let Some(di) = self.graphs.docs.iter().position(|d| d.id == doc) else { return };
        let entries = self.folder_specs(folder, &self.graphs.docs[di].graph);
        let mut specs: Vec<LayerSpec> = entries.iter().map(|(_, s)| s.clone()).collect();
        let before = self.graphs.docs[di].graph.clone();
        if let Some(seen) = self.graphs.docs[di].seen.take().filter(|s| *s != before) {
            smart::read_view(&mut self.graphs.docs[di].graph, &seen, &mut specs, &self.graphs.lib);
            for ((idx, old), new) in entries.iter().zip(&specs) {
                if new != old {
                    self.apply_spec(*idx, old, new, di);
                }
            }
        }
        let d = &mut self.graphs.docs[di];
        smart::update_view(&mut d.graph, folder, &specs, &self.graphs.lib);
        if d.graph != before {
            d.dirty = true;
        }
        d.seen = Some(d.graph.clone());
        self.update_mask_graphs(di, &specs);
    }

    /// The visible fill layers directly inside `folder`, bottom first, with
    /// their indices.
    fn folder_specs(&self, folder: u64, g: &Graph) -> Vec<(usize, LayerSpec)> {
        self.engine
            .layers
            .iter()
            .enumerate()
            .filter(|(_, l)| l.parent == Some(folder) && l.visible)
            .filter_map(|(i, l)| {
                let Content::Fill(f) = &l.content else { return None };
                let images = std::array::from_fn(|ch| l.images[ch].as_ref().and_then(|im| image_path(g, l.id, ch, &im.name)));
                let mask = l.mask.as_ref();
                let custom = mask.and_then(|m| m.graph).is_some_and(|d| g.tags.iter().any(|t| t.layer == l.id && t.role == Role::Source(d)));
                let spec = LayerSpec {
                    id: l.id,
                    name: l.name.clone(),
                    opacity: l.opacity,
                    blend: l.blend,
                    fill: *f,
                    generator: mask.and_then(|m| m.generator),
                    images,
                    custom_mask: custom,
                };
                Some((i, spec))
            })
            .collect()
    }

    /// Writes a layer's spec, changed in its folder's graph (doc `di`), into the layer.
    fn apply_spec(&mut self, idx: usize, old: &LayerSpec, new: &LayerSpec, di: usize) {
        {
            let l = &mut self.engine.layers[idx];
            l.opacity = new.opacity;
            l.blend = new.blend;
            if let Content::Fill(f) = &mut l.content {
                let graph = f.graph;
                *f = new.fill;
                f.graph = graph;
            }
        }
        for ch in 0..CHANNELS {
            if new.images[ch] == old.images[ch] {
                continue;
            }
            match &new.images[ch] {
                Some(p) => match self.engine.shared_fill_image(&crate::graph::runtime::asset_path(p)) {
                    Ok(img) => {
                        self.engine.set_fill_image(idx, ch, Some(img), &mut self.undo);
                        // Layer nodes lay images on tri-planar.
                        if let Content::Fill(f) = &mut self.engine.layers[idx].content {
                            f.projection = Projection::Triplanar;
                        }
                    }
                    Err(e) => self.status = e,
                },
                None => self.engine.set_fill_image(idx, ch, None, &mut self.undo),
            }
        }
        let mode = |s: &LayerSpec| (s.custom_mask, s.generator.is_some());
        if mode(new) != mode(old) {
            let graph = if new.custom_mask { self.mask_graph_for(di, new.id, &new.name) } else { None };
            let generator = if new.custom_mask { None } else { new.generator };
            self.engine.set_mask_source(idx, generator, graph, &mut self.undo);
        } else if let Some(m) = self.engine.layers[idx].mask.as_mut() {
            m.generator = new.generator;
        }
        self.engine.mark_dirty();
    }

    /// The mask graph of `layer`, whose mask comes from a node of the
    /// artist's in folder graph `di`; made (and tagged) on first use.
    fn mask_graph_for(&mut self, di: usize, layer: u64, name: &str) -> Option<u64> {
        let view = &self.graphs.docs[di].graph;
        let tag = view.tags.iter().position(|t| t.layer == layer && matches!(t.role, Role::Source(_)))?;
        let t = view.tags[tag];
        if let Role::Source(id) = t.role
            && id != 0 && self.graphs.doc(id).is_some() {
                return Some(id);
            }
        let sub = smart::mask_subgraph(view, t.node, &format!("{name} mask"));
        let id = self.graphs.insert(&self.rs, sub, None);
        // During a sync: the graph is seen again once it's done.
        self.graphs.docs[di].graph.tags[tag].role = Role::Source(id);
        Some(id)
    }

    /// Mask graphs cut from folder graph `di` take on its edits.
    fn update_mask_graphs(&mut self, di: usize, specs: &[LayerSpec]) {
        for s in specs.iter().filter(|s| s.custom_mask) {
            let view = &self.graphs.docs[di].graph;
            let Some(t) = view.tags.iter().find(|t| t.layer == s.id && matches!(t.role, Role::Source(_))) else { continue };
            let Role::Source(id) = t.role else { continue };
            if view.node(t.node).is_none() {
                continue;
            }
            let sub = smart::mask_subgraph(view, t.node, &format!("{} mask", s.name));
            let Some(d) = self.graphs.docs.iter_mut().find(|d| d.id == id) else { continue };
            if d.graph.nodes != sub.nodes || d.graph.res != sub.res {
                d.graph.nodes = sub.nodes;
                d.graph.next_id = sub.next_id;
                d.graph.res = sub.res;
                d.graph.match_set = sub.match_set;
                d.dirty = true;
            }
        }
    }
}

/// Where a layer's image is, for its folder graph: the path its Layer node
/// already has when that's the same file, else a scan in the assets folder
/// by that name (`name`: the image's folder and file name).
fn image_path(g: &Graph, layer: u64, ch: usize, name: &str) -> Option<PathBuf> {
    let k = smart::NODE_CHANNEL.iter().position(|&c| c == ch)?;
    let tail = |p: &Path| -> Option<String> {
        let dir = p.parent().and_then(|d| d.file_name()).map_or(String::new(), |d| format!("{}/", d.to_string_lossy()));
        Some(format!("{dir}{}", p.file_name()?.to_string_lossy()))
    };
    if let Some(node) = g.tagged(layer, Role::Channel(k)).and_then(|n| g.node(n))
        && let Some(Value::Image(Some(p))) = node.params.get("image")
            && tail(p).as_deref() == Some(name) {
                return Some(p.clone());
            }
    let rel = PathBuf::from(format!("pbr/{name}"));
    crate::graph::runtime::asset_path(&rel).is_file().then_some(rel)
}

#[cfg(test)]
impl PainterApp {
    /// Adds smart material `k` as a layer folder on top; returns its index (offscreen tests).
    pub(crate) fn test_add_smart_folder(&mut self, k: usize) -> usize {
        let at = self.engine.layers.len();
        self.selected = super::add_smart_material(&mut self.engine, &mut self.undo, &crate::presets::SMART_MATERIALS[k], at, None);
        self.selected
    }

    /// Opens folder `idx`'s live graph; returns its id (offscreen tests).
    pub(crate) fn test_open_folder_graph(&mut self, idx: usize) -> Option<u64> {
        self.open_folder_graph(idx);
        match &self.engine.layers[idx].content {
            Content::Folder(f) => f.graph,
            _ => None,
        }
    }

    /// Graph `doc`, to edit as the artist would (offscreen tests).
    pub(crate) fn test_graph_mut(&mut self, doc: u64) -> Option<&mut Graph> {
        self.graphs.docs.iter_mut().find(|d| d.id == doc).map(|d| &mut d.graph)
    }

    pub(crate) fn test_engine_mut(&mut self) -> &mut crate::engine::Engine {
        &mut self.engine
    }

    /// Undoes the last edit (offscreen tests).
    pub(crate) fn test_undo_step(&mut self) {
        self.engine.undo(&mut self.undo);
    }
}
