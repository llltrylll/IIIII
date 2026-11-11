//! Undo/redo for every document edit.
//!
//! Strokes store before/after states of the painted textures, read back
//! asynchronously, LZ4-compressed on a worker thread and kept in system RAM,
//! so stroke history costs almost no VRAM. Structural edits (layers, masks,
//! clears) are swaps that keep the removed data; property edits store the
//! previous non-pixel state.

use crate::engine::{Blend, Content, FillParams, Generator, Layer, Levels, Mask};
use crate::gpu::{Tex, bytes_per_texel};
use eframe::wgpu;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;

#[derive(Clone, Copy)]
pub struct TexInfo {
    pub width: u32,
    pub height: u32,
    pub bpp: u32,
    pub format: wgpu::TextureFormat,
}

enum SnapState {
    Mapping {
        buffers: Vec<wgpu::Buffer>,
        remaining: Arc<AtomicUsize>,
        failure: Arc<Mutex<Option<String>>>,
    },
    Compressing(mpsc::Receiver<Vec<Vec<u8>>>),
    Ready(Vec<Vec<u8>>),
    Failed(String),
}

pub struct Snapshot {
    infos: Vec<TexInfo>,
    state: SnapState,
}

impl Snapshot {
    /// Encodes copies of `textures` into fresh staging buffers. Call
    /// [`Snapshot::request_map`] after the encoder has been submitted.
    pub fn encode(device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, textures: &[&Tex]) -> Snapshot {
        let mut infos = Vec::new();
        let mut buffers = Vec::new();
        for tex in textures {
            let bpp = bytes_per_texel(tex.format);
            let row = tex.width * bpp;
            debug_assert!(row.is_multiple_of(256), "texture rows must be 256-byte aligned for readback");
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("undo staging"),
                size: (row * tex.height) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_texture_to_buffer(
                tex.copy_info(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(row),
                        rows_per_image: Some(tex.height),
                    },
                },
                tex.extent(),
            );
            infos.push(TexInfo { width: tex.width, height: tex.height, bpp, format: tex.format });
            buffers.push(buffer);
        }
        let remaining = Arc::new(AtomicUsize::new(buffers.len()));
        let failure = Arc::new(Mutex::new(None));
        Snapshot { infos, state: SnapState::Mapping { buffers, remaining, failure } }
    }

    pub fn request_map(&self) {
        if let SnapState::Mapping { buffers, remaining, failure } = &self.state {
            for b in buffers {
                let remaining = remaining.clone();
                let failure = failure.clone();
                b.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                    if let Err(error) = result {
                        let mut slot = failure.lock().unwrap_or_else(|e| e.into_inner());
                        if slot.is_none() {
                            *slot = Some(format!("GPU undo snapshot readback failed: {error}"));
                        }
                    }
                    remaining.fetch_sub(1, Ordering::AcqRel);
                });
            }
        }
    }

    /// Advances the state machine without blocking.
    pub fn update(&mut self) {
        match &mut self.state {
            SnapState::Mapping { buffers, remaining, failure } => {
                if remaining.load(Ordering::Acquire) != 0 {
                    return;
                }

                let mut failed = failure.lock().unwrap_or_else(|e| e.into_inner()).clone();
                let mut raw = Vec::with_capacity(buffers.len());
                if failed.is_none() {
                    for buffer in buffers.iter() {
                        match buffer.slice(..).get_mapped_range() {
                            Ok(data) => raw.push(data.to_vec()),
                            Err(error) => {
                                failed = Some(format!("GPU undo snapshot could not be read: {error}"));
                                break;
                            }
                        }
                    }
                }
                // Release every mapping, including successful mappings when another failed.
                for buffer in buffers {
                    buffer.unmap();
                }
                if let Some(error) = failed {
                    self.state = SnapState::Failed(error);
                    return;
                }

                let (tx, rx) = mpsc::channel();
                rayon::spawn(move || {
                    let packed = raw.iter().map(|data| lz4_flex::compress_prepend_size(data)).collect();
                    let _ = tx.send(packed);
                });
                self.state = SnapState::Compressing(rx);
            }
            SnapState::Compressing(rx) => match rx.try_recv() {
                Ok(data) => self.state = SnapState::Ready(data),
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.state = SnapState::Failed("Undo snapshot compression worker stopped unexpectedly.".into());
                }
            },
            SnapState::Ready(_) | SnapState::Failed(_) => {}
        }
    }

    /// Blocks until the data is available. Used when the user undoes a stroke
    /// whose readback is still in flight.
    pub fn wait(&mut self, device: &wgpu::Device) -> Result<(), String> {
        loop {
            match &mut self.state {
                SnapState::Mapping { remaining, .. } => {
                    if remaining.load(Ordering::Acquire) != 0
                        && let Err(error) = device.poll(wgpu::PollType::wait_indefinitely())
                    {
                        let message = format!("GPU did not finish the undo snapshot readback: {error:?}");
                        self.state = SnapState::Failed(message.clone());
                        return Err(message);
                    }
                    self.update();
                }
                SnapState::Compressing(rx) => match rx.recv() {
                    Ok(data) => self.state = SnapState::Ready(data),
                    Err(_) => {
                        let message = "Undo snapshot compression worker stopped unexpectedly.".to_string();
                        self.state = SnapState::Failed(message.clone());
                        return Err(message);
                    }
                },
                SnapState::Ready(_) => return Ok(()),
                SnapState::Failed(error) => return Err(error.clone()),
            }
        }
    }

    pub fn compressed_bytes(&self) -> usize {
        match &self.state {
            SnapState::Ready(d) => d.iter().map(|v| v.len()).sum(),
            _ => 0,
        }
    }

    /// Uploads the snapshot into `textures` (same order and sizes as captured).
    pub fn restore(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, textures: &[&Tex]) -> Result<(), String> {
        self.wait(device)?;
        let SnapState::Ready(data) = &self.state else {
            return Err("Undo snapshot is not ready.".into());
        };
        if textures.len() != self.infos.len() || data.len() != self.infos.len() {
            return Err("Undo snapshot texture count no longer matches its target.".into());
        }

        // Validate the complete snapshot before writing any part of it.
        let mut decoded = Vec::with_capacity(data.len());
        for ((tex, packed), info) in textures.iter().zip(data).zip(&self.infos) {
            if tex.width != info.width || tex.height != info.height || tex.format != info.format {
                return Err("Undo snapshot dimensions or format no longer match its target.".into());
            }
            let raw = lz4_flex::decompress_size_prepended(packed)
                .map_err(|error| format!("Undo snapshot decompression failed: {error}"))?;
            let expected = info.width as usize * info.height as usize * info.bpp as usize;
            if raw.len() != expected {
                return Err("Undo snapshot data has an unexpected size.".into());
            }
            decoded.push(raw);
        }

        for ((tex, raw), info) in textures.iter().zip(&decoded).zip(&self.infos) {
            queue.write_texture(
                tex.copy_info(),
                raw,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(info.width * info.bpp),
                    rows_per_image: Some(info.height),
                },
                tex.extent(),
            );
        }
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StrokeTarget {
    /// Layer content; the channels painted are listed in the entry.
    Content,
    Mask,
    /// A painted correction on baked mesh map `n` (see `engine::MAPS`).
    BakeFix(usize),
}

/// One painted stroke: before/after snapshots of the textures it touched.
pub struct UndoEntry {
    pub layer_id: u64,
    pub target: StrokeTarget,
    pub channels: Vec<usize>,
    pub before: Snapshot,
    pub after: Snapshot,
}

impl UndoEntry {
    fn bytes(&self) -> usize {
        self.before.compressed_bytes() + self.after.compressed_bytes()
    }
}

/// Non-pixel state of one layer: everything the properties panel edits.
#[derive(Clone, PartialEq)]
pub struct LayerProps {
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    pub blend: Blend,
    pub fill: Option<FillParams>,
    pub generator: Option<Generator>,
    pub mask_filters: Vec<crate::engine::MaskFilter>,
    pub mask_layers: Vec<crate::engine::MaskLayer>,
}

/// Non-pixel document state, diffed to record property edits.
#[derive(Clone, PartialEq, Default)]
pub struct DocProps {
    pub layers: Vec<(u64, LayerProps)>,
    pub levels: Vec<Levels>,
}

/// A reversible edit. Every variant except `Stroke` is a swap: applying it
/// exchanges the stored state with the engine's, so the same operation both
/// undoes and redoes it (see `Engine::apply_edit`).
pub enum Edit {
    Stroke(UndoEntry),
    /// `Some(layer)` is inserted at `index`; `None` removes layer `id` and stores it.
    Layer { index: usize, id: u64, layer: Option<Layer> },
    /// Layer order, by id.
    Order(Vec<u64>),
    Content { id: u64, content: Content },
    /// The folder a layer is in.
    Parent { id: u64, parent: Option<u64> },
    /// A fill layer's channel images.
    Images { id: u64, images: [Option<crate::engine::FillImage>; crate::engine::CHANNELS] },
    Mask { id: u64, mask: Option<Mask> },
    Props(DocProps),
    /// Painted correction texture of a baked map.
    Fix { map: usize, tex: Option<Tex> },
    /// Several edits as one step: applied in order to redo, reversed to undo.
    Group(Vec<Edit>),
}

impl Edit {
    fn for_each_stroke(&mut self, f: &mut impl FnMut(&mut UndoEntry)) {
        match self {
            Edit::Stroke(e) => f(e),
            Edit::Group(edits) => edits.iter_mut().for_each(|e| e.for_each_stroke(f)),
            _ => {}
        }
    }

    fn bytes(&self) -> usize {
        match self {
            Edit::Stroke(e) => e.bytes(),
            Edit::Group(edits) => edits.iter().map(|e| e.bytes()).sum(),
            _ => 0,
        }
    }
}

pub struct Entry {
    pub label: String,
    pub edit: Edit,
    /// Unique per recorded step; the top serial identifies the document
    /// state, so "unsaved changes" survives undo and redo correctly.
    pub serial: u64,
    /// The texture set the step edits (see `app::texture_sets`).
    pub set: usize,
}

/// Linear history. Removed layers, masks and cleared paint live here (still
/// on the GPU) until their entry falls off the end.
pub struct UndoStack {
    pub undo: Vec<Entry>,
    pub redo: Vec<Entry>,
    group: Option<Entry>,
    /// Nesting of begin_group calls; the group is pushed when it returns to 0.
    group_depth: usize,
    next_serial: u64,
    /// Steps kept; older ones are dropped (Preferences > General).
    pub max_entries: usize,
    /// The texture set being edited: new steps belong to it.
    pub set: usize,
}

impl Default for UndoStack {
    fn default() -> Self {
        Self { undo: Vec::new(), redo: Vec::new(), group: None, group_depth: 0, next_serial: 0, max_entries: 64, set: 0 }
    }
}

impl UndoStack {
    pub fn push(&mut self, label: impl Into<String>, edit: Edit) {
        if let Some(Entry { edit: Edit::Group(edits), .. }) = &mut self.group {
            edits.push(edit);
            return;
        }
        self.redo.clear();
        self.next_serial += 1;
        self.undo.push(Entry { label: label.into(), edit, serial: self.next_serial, set: self.set });
        if self.undo.len() > self.max_entries.max(1) {
            let excess = self.undo.len() - self.max_entries.max(1);
            self.undo.drain(..excess);
        }
    }

    /// Collects nested edits into one history step until [`UndoStack::end_group`]
    /// closes the outermost group.
    pub fn begin_group(&mut self, label: impl Into<String>) {
        if self.group_depth == 0 {
            self.group = Some(Entry { label: label.into(), edit: Edit::Group(Vec::new()), serial: 0, set: self.set });
        }
        self.group_depth += 1;
    }

    pub fn end_group(&mut self) {
        self.group_depth = self.group_depth.saturating_sub(1);
        if self.group_depth > 0 {
            return;
        }
        if let Some(entry) = self.group.take()
            && matches!(&entry.edit, Edit::Group(edits) if !edits.is_empty()) {
                self.push(entry.label, entry.edit);
            }
    }

    pub fn update(&mut self) {
        for e in self.undo.iter_mut().chain(self.redo.iter_mut()) {
            e.edit.for_each_stroke(&mut |s| {
                s.before.update();
                s.after.update();
            });
        }
    }

    /// Identifies the current document state (0 = nothing recorded).
    pub fn state_serial(&self) -> u64 {
        self.undo.last().map_or(0, |e| e.serial)
    }

    /// Discards history and any group that has not yet been committed.
    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
        self.group = None;
        self.group_depth = 0;
    }

    pub fn ram_bytes(&self) -> usize {
        self.undo.iter().chain(self.redo.iter()).map(|e| e.edit.bytes()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::{Edit, UndoStack};

    #[test]
    fn clearing_history_also_resets_open_groups() {
        let mut history = UndoStack::default();
        history.begin_group("discarded group");
        history.push("discarded edit", Edit::Order(Vec::new()));

        history.clear();

        history.begin_group("kept group");
        history.push("kept edit", Edit::Order(Vec::new()));
        history.end_group();

        assert_eq!(history.undo.len(), 1);
        assert_eq!(history.undo[0].label, "kept group");
        assert!(matches!(&history.undo[0].edit, Edit::Group(edits) if edits.len() == 1));
    }
}
