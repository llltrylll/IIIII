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
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;

#[derive(Clone, Copy)]
pub struct TexInfo {
    pub width: u32,
    pub height: u32,
    pub bpp: u32,
}

enum SnapState {
    Mapping { buffers: Vec<wgpu::Buffer>, remaining: Arc<AtomicUsize> },
    Compressing {
        receiver: mpsc::Receiver<Vec<u8>>,
        packed: Vec<Vec<u8>>,
        pending_raw_bytes: usize,
        total: usize,
    },
    Ready(Vec<Vec<u8>>),
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
            infos.push(TexInfo { width: tex.width, height: tex.height, bpp });
            buffers.push(buffer);
        }
        let remaining = Arc::new(AtomicUsize::new(buffers.len()));
        Snapshot { infos, state: SnapState::Mapping { buffers, remaining } }
    }

    pub fn request_map(&self) {
        if let SnapState::Mapping { buffers, remaining } = &self.state {
            for b in buffers {
                let remaining = remaining.clone();
                b.slice(..).map_async(wgpu::MapMode::Read, move |_| {
                    remaining.fetch_sub(1, Ordering::AcqRel);
                });
            }
        }
    }

    fn uncompressed_bytes(&self) -> usize {
        self.infos.iter().fold(0usize, |sum, info| {
            sum.saturating_add(info.width as usize * info.height as usize * info.bpp as usize)
        })
    }

    /// Advances the state machine without blocking. Compression runs one
    /// texture at a time and uses a one-item channel: the raw image can be
    /// released before the next image is copied/compressed, avoiding a second
    /// full-size collection of uncompressed snapshots in system RAM.
    pub fn update(&mut self) {
        let mut mapped_buffers = None;
        if let SnapState::Mapping { buffers, remaining } = &mut self.state
            && remaining.load(Ordering::Acquire) == 0 {
                mapped_buffers = Some(std::mem::take(buffers));
            }

        if let Some(buffers) = mapped_buffers {
            let total = buffers.len();
            let pending_raw_bytes = self.uncompressed_bytes();
            let (tx, receiver) = mpsc::sync_channel(1);
            rayon::spawn(move || {
                for buffer in buffers {
                    let raw = {
                        let mapped = buffer.slice(..).get_mapped_range().expect("staging buffer is mapped");
                        mapped.to_vec()
                    };
                    buffer.unmap();
                    let compressed = lz4_flex::compress_prepend_size(&raw);
                    drop(raw);
                    if tx.send(compressed).is_err() {
                        break;
                    }
                }
            });
            self.state = SnapState::Compressing {
                receiver,
                packed: Vec::with_capacity(total),
                pending_raw_bytes,
                total,
            };
            return;
        }

        let mut ready = None;
        if let SnapState::Compressing { receiver, packed, pending_raw_bytes, total } = &mut self.state {
            loop {
                match receiver.try_recv() {
                    Ok(data) => {
                        if let Some(info) = self.infos.get(packed.len()) {
                            let raw_bytes = info.width as usize * info.height as usize * info.bpp as usize;
                            *pending_raw_bytes = pending_raw_bytes.saturating_sub(raw_bytes);
                        }
                        packed.push(data);
                    }
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        ready = Some(std::mem::take(packed));
                        break;
                    }
                }
            }
            if packed.len() >= *total {
                ready = Some(std::mem::take(packed));
            }
        }
        if let Some(data) = ready {
            self.state = SnapState::Ready(data);
        }
    }

    /// Blocks until the data is available. Used when the user undoes a stroke
    /// whose readback is still in flight.
    pub fn wait(&mut self, device: &wgpu::Device) {
        loop {
            self.update();
            let mut ready = None;
            match &mut self.state {
                SnapState::Mapping { remaining, .. } => {
                    if remaining.load(Ordering::Acquire) != 0 {
                        let _ = device.poll(wgpu::PollType::wait_indefinitely());
                    }
                }
                SnapState::Compressing { receiver, packed, pending_raw_bytes, total } => {
                    match receiver.recv() {
                        Ok(data) => {
                            let index = packed.len();
                            if let Some(info) = self.infos.get(index) {
                                let raw_bytes = info.width as usize * info.height as usize * info.bpp as usize;
                                *pending_raw_bytes = pending_raw_bytes.saturating_sub(raw_bytes);
                            }
                            packed.push(data);
                            if packed.len() >= *total {
                                ready = Some(std::mem::take(packed));
                            }
                        }
                        Err(_) => ready = Some(std::mem::take(packed)),
                    }
                }
                SnapState::Ready(_) => return,
            }
            if let Some(data) = ready {
                self.state = SnapState::Ready(data);
            }
        }
    }

    /// Current Undo-history footprint. Pending snapshots use their remaining
    /// uncompressed size as a conservative estimate; ready snapshots use the
    /// actual compressed bytes retained in RAM.
    pub fn ram_bytes(&self) -> usize {
        match &self.state {
            SnapState::Mapping { .. } => self.uncompressed_bytes(),
            SnapState::Compressing { packed, pending_raw_bytes, .. } => {
                packed.iter().map(|v| v.len()).sum::<usize>().saturating_add(*pending_raw_bytes)
            }
            SnapState::Ready(d) => d.iter().map(|v| v.len()).sum(),
        }
    }

    /// Uploads the snapshot into `textures` (same order and sizes as captured).
    pub fn restore(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, textures: &[&Tex]) {
        self.wait(device);
        let SnapState::Ready(data) = &self.state else { return };
        for ((tex, packed), info) in textures.iter().zip(data).zip(&self.infos) {
            if tex.width != info.width || tex.height != info.height {
                continue;
            }
            let Ok(raw) = lz4_flex::decompress_size_prepended(packed) else { continue };
            queue.write_texture(
                tex.copy_info(),
                &raw,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(info.width * info.bpp),
                    rows_per_image: Some(info.height),
                },
                tex.extent(),
            );
        }
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
        self.before.ram_bytes().saturating_add(self.after.ram_bytes())
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
    /// RAM budget for painted before/after snapshots. One newest entry may
    /// exceed the budget by itself; it is kept so the latest action remains undoable.
    pub max_ram_bytes: usize,
    /// The texture set being edited: new steps belong to it.
    pub set: usize,
}

impl Default for UndoStack {
    fn default() -> Self {
        Self {
            undo: Vec::new(),
            redo: Vec::new(),
            group: None,
            group_depth: 0,
            next_serial: 0,
            max_entries: 64,
            max_ram_bytes: 512 * 1024 * 1024,
            set: 0,
        }
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
        self.enforce_limits();
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
        self.enforce_limits();
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

    pub fn set_limits(&mut self, max_entries: usize, max_ram_bytes: usize) {
        self.max_entries = max_entries.max(1);
        self.max_ram_bytes = max_ram_bytes.max(1);
        self.enforce_limits();
    }

    fn enforce_limits(&mut self) {
        if self.undo.len() > self.max_entries.max(1) {
            let excess = self.undo.len() - self.max_entries.max(1);
            self.undo.drain(..excess);
        }
        while self.ram_bytes() > self.max_ram_bytes {
            // Redo entries are the first history to discard; pop removes the
            // oldest redo step. Otherwise discard the oldest undo step, but
            // retain the newest one even when a single stroke exceeds budget.
            if !self.redo.is_empty() {
                self.redo.pop();
            } else if self.undo.len() > 1 {
                self.undo.remove(0);
            } else {
                break;
            }
        }
    }

    pub fn ram_bytes(&self) -> usize {
        self.undo.iter().chain(self.redo.iter()).map(|e| e.edit.bytes()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::{Edit, SnapState, Snapshot, StrokeTarget, TexInfo, UndoEntry, UndoStack};
    use std::sync::mpsc;

    fn stroke_with_snapshot_bytes(bytes: usize) -> Edit {
        let snapshot = || Snapshot { infos: Vec::new(), state: SnapState::Ready(vec![vec![0; bytes]]) };
        Edit::Stroke(UndoEntry {
            layer_id: 1,
            target: StrokeTarget::Content,
            channels: vec![0],
            before: snapshot(),
            after: snapshot(),
        })
    }

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

    #[test]
    fn memory_budget_discards_oldest_history_first() {
        let mut history = UndoStack::default();
        history.set_limits(16, 20);
        history.push("first", stroke_with_snapshot_bytes(6));
        history.push("second", stroke_with_snapshot_bytes(6));
        history.push("third", stroke_with_snapshot_bytes(6));

        assert_eq!(history.undo.iter().map(|entry| entry.label.as_str()).collect::<Vec<_>>(), vec!["second", "third"]);
        assert_eq!(history.ram_bytes(), 24);
        assert!(history.ram_bytes() > history.max_ram_bytes, "newest entry is preserved even when it alone exceeds the cap");
    }

    #[test]
    fn memory_budget_never_discards_the_only_newest_undo_step() {
        let mut history = UndoStack::default();
        history.set_limits(16, 4);
        history.push("latest", stroke_with_snapshot_bytes(6));

        assert_eq!(history.undo.len(), 1);
        assert_eq!(history.undo[0].label, "latest");
        assert_eq!(history.ram_bytes(), 12);
    }

    #[test]
    fn pending_snapshot_memory_is_counted_before_compression_finishes() {
        let (_tx, receiver) = mpsc::sync_channel(1);
        let snapshot = Snapshot {
            infos: vec![TexInfo { width: 4, height: 4, bpp: 4 }],
            state: SnapState::Compressing {
                receiver,
                packed: vec![vec![1, 2, 3]],
                pending_raw_bytes: 64,
                total: 2,
            },
        };
        assert_eq!(snapshot.ram_bytes(), 67);
    }
}
