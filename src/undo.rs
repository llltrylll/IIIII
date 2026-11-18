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
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;

#[derive(Clone, Copy)]
pub struct TexInfo {
    pub width: u32,
    pub height: u32,
    pub bpp: u32,
}

/// Reuses completed Undo readback buffers. The cache is deliberately small:
/// it is shared across a document's Texture Sets and cannot retain more than
/// 32 MiB of staging allocations after the compression worker releases them.
#[derive(Default)]
pub struct SnapshotBufferPool {
    state: Mutex<SnapshotBufferPoolState>,
}

#[derive(Default)]
struct SnapshotBufferPoolState {
    buffers: HashMap<u64, Vec<wgpu::Buffer>>,
    cached_bytes: u64,
}

impl SnapshotBufferPool {
    const MAX_PER_SIZE: usize = 2;
    const MAX_CACHED_BYTES: u64 = 32 * 1024 * 1024;

    fn can_recycle(size: u64, per_size: usize, cached_bytes: u64) -> bool {
        per_size < Self::MAX_PER_SIZE
            && size <= Self::MAX_CACHED_BYTES.saturating_sub(cached_bytes)
    }

    fn take(&self, device: &wgpu::Device, size: u64) -> wgpu::Buffer {
        let cached = {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            let buffer = state.buffers.get_mut(&size).and_then(Vec::pop);
            if buffer.is_some() {
                state.cached_bytes = state.cached_bytes.saturating_sub(size);
                if state.buffers.get(&size).is_some_and(Vec::is_empty) {
                    state.buffers.remove(&size);
                }
            }
            buffer
        };
        cached.unwrap_or_else(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("undo staging"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        })
    }

    /// The caller must unmap the buffer before returning it to the pool.
    fn recycle(&self, buffer: wgpu::Buffer) {
        let size = buffer.size();
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let per_size = state.buffers.get(&size).map_or(0, Vec::len);
        if Self::can_recycle(size, per_size, state.cached_bytes) {
            state.buffers.entry(size).or_default().push(buffer);
            state.cached_bytes += size;
        }
    }
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
    pool: Arc<SnapshotBufferPool>,
    /// Last byte count applied to the owning UndoStack's cached total.
    accounted_bytes: usize,
}

impl Snapshot {
    /// Encodes copies of `textures` into fresh staging buffers. Call
    /// [`Snapshot::request_map`] after the encoder has been submitted.
    pub fn encode(
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        textures: &[&Tex],
        pool: Arc<SnapshotBufferPool>,
    ) -> Snapshot {
        let mut infos = Vec::new();
        let mut buffers = Vec::new();
        for tex in textures {
            let bpp = bytes_per_texel(tex.format);
            let row = tex.width * bpp;
            debug_assert!(row.is_multiple_of(256), "texture rows must be 256-byte aligned for readback");
            let buffer = pool.take(device, (row * tex.height) as u64);
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
        let accounted_bytes = infos.iter().fold(0usize, |sum, info| {
            sum.saturating_add(info.width as usize * info.height as usize * info.bpp as usize)
        });
        Snapshot { infos, state: SnapState::Mapping { buffers, remaining }, pool, accounted_bytes }
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
        if matches!(self.state, SnapState::Ready(_)) {
            return;
        }
        let mut mapped_buffers = None;
        if let SnapState::Mapping { buffers, remaining } = &mut self.state
            && remaining.load(Ordering::Acquire) == 0 {
                mapped_buffers = Some(std::mem::take(buffers));
            }

        if let Some(buffers) = mapped_buffers {
            let total = buffers.len();
            let pending_raw_bytes = self.uncompressed_bytes();
            let (tx, receiver) = mpsc::sync_channel(1);
            let pool = self.pool.clone();
            rayon::spawn(move || {
                for buffer in buffers {
                    // Compress directly from the mapped staging memory. A
                    // separate raw Vec would duplicate an entire texture
                    // while the mapped buffer still owns the same pixels.
                    let compressed = {
                        let mapped = buffer.slice(..).get_mapped_range().expect("staging buffer is mapped");
                        lz4_flex::compress_prepend_size(&mapped)
                    };
                    buffer.unmap();
                    pool.recycle(buffer);
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

    /// The byte count last included in UndoStack::cached_ram_bytes.
    fn accounted_ram_bytes(&self) -> usize {
        self.accounted_bytes
    }

    /// Returns the change since the last byte count applied to UndoStack.
    /// This also catches state transitions completed by wait() during Undo/Redo.
    fn take_ram_delta(&mut self) -> i128 {
        let current = self.ram_bytes();
        let delta = current as i128 - self.accounted_bytes as i128;
        self.accounted_bytes = current;
        delta
    }

    fn is_pending(&self) -> bool {
        !matches!(self.state, SnapState::Ready(_))
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
        self.before.accounted_ram_bytes().saturating_add(self.after.accounted_ram_bytes())
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
    /// Cached sum of accounted before/after snapshot sizes; UI reads this in O(1).
    cached_ram_bytes: usize,
    /// True only while at least one retained snapshot needs asynchronous progress.
    pending_snapshots: bool,
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
            cached_ram_bytes: 0,
            pending_snapshots: false,
        }
    }
}

impl UndoStack {
    pub fn push(&mut self, label: impl Into<String>, mut edit: Edit) {
        let mut pending = false;
        edit.for_each_stroke(&mut |stroke| {
            pending |= stroke.before.is_pending() || stroke.after.is_pending();
        });

        if let Some(Entry { edit: Edit::Group(edits), .. }) = &mut self.group {
            // Group contents are not part of the history budget until the group closes.
            self.pending_snapshots |= pending;
            edits.push(edit);
            return;
        }

        let bytes = edit.bytes();
        let redo_bytes = self.redo.iter().map(|entry| entry.edit.bytes()).sum::<usize>();
        self.cached_ram_bytes = self.cached_ram_bytes.saturating_sub(redo_bytes).saturating_add(bytes);
        self.redo.clear();
        self.pending_snapshots |= pending;
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
        if !self.pending_snapshots {
            return;
        }

        let mut delta = 0i128;
        let mut pending = false;
        for entry in self.undo.iter_mut().chain(self.redo.iter_mut()) {
            entry.edit.for_each_stroke(&mut |stroke| {
                stroke.before.update();
                stroke.after.update();
                delta += stroke.before.take_ram_delta();
                delta += stroke.after.take_ram_delta();
                pending |= stroke.before.is_pending() || stroke.after.is_pending();
            });
        }

        // An open group is not yet charged to the RAM budget, but its snapshots
        // still need compression and their accounted sizes must be current when committed.
        if let Some(group) = &mut self.group {
            group.edit.for_each_stroke(&mut |stroke| {
                stroke.before.update();
                stroke.after.update();
                let _ = stroke.before.take_ram_delta();
                let _ = stroke.after.take_ram_delta();
                pending |= stroke.before.is_pending() || stroke.after.is_pending();
            });
        }

        if delta >= 0 {
            self.cached_ram_bytes = self.cached_ram_bytes.saturating_add(delta as usize);
        } else {
            self.cached_ram_bytes = self.cached_ram_bytes.saturating_sub((-delta) as usize);
        }
        self.pending_snapshots = pending;
        // Compression normally shrinks snapshots. If a block expands slightly,
        // enforce the budget after applying that positive delta.
        if delta > 0 {
            self.enforce_limits();
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
        self.cached_ram_bytes = 0;
        self.pending_snapshots = false;
    }

    pub fn set_limits(&mut self, max_entries: usize, max_ram_bytes: usize) {
        self.max_entries = max_entries.max(1);
        self.max_ram_bytes = max_ram_bytes.max(1);
        self.enforce_limits();
    }

    fn enforce_limits(&mut self) {
        if self.undo.len() > self.max_entries.max(1) {
            let excess = self.undo.len() - self.max_entries.max(1);
            for entry in self.undo.drain(..excess) {
                self.cached_ram_bytes = self.cached_ram_bytes.saturating_sub(entry.edit.bytes());
            }
        }
        while self.cached_ram_bytes > self.max_ram_bytes {
            // Redo entries are the first history to discard; pop removes the
            // oldest redo step. Otherwise discard the oldest undo step, but
            // retain the newest one even when a single stroke exceeds budget.
            if let Some(entry) = self.redo.pop() {
                self.cached_ram_bytes = self.cached_ram_bytes.saturating_sub(entry.edit.bytes());
            } else if self.undo.len() > 1 {
                let entry = self.undo.remove(0);
                self.cached_ram_bytes = self.cached_ram_bytes.saturating_sub(entry.edit.bytes());
            } else {
                break;
            }
        }
    }

    /// Cached Undo-history footprint. This is O(1) for the per-frame status display.
    pub fn ram_bytes(&self) -> usize {
        self.cached_ram_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::{Edit, SnapState, Snapshot, SnapshotBufferPool, StrokeTarget, TexInfo, UndoEntry, UndoStack};
    use std::sync::mpsc;
    use std::sync::Arc;

    fn stroke_with_snapshot_bytes(bytes: usize) -> Edit {
        let snapshot = || Snapshot {
            infos: Vec::new(),
            state: SnapState::Ready(vec![vec![0; bytes]]),
            pool: Arc::new(SnapshotBufferPool::default()),
            accounted_bytes: bytes,
        };
        Edit::Stroke(UndoEntry {
            layer_id: 1,
            target: StrokeTarget::Content,
            channels: vec![0],
            before: snapshot(),
            after: snapshot(),
        })
    }

    #[test]
    fn staging_buffer_pool_retention_is_bounded() {
        assert!(SnapshotBufferPool::can_recycle(1024, 0, 0));
        assert!(!SnapshotBufferPool::can_recycle(1024, SnapshotBufferPool::MAX_PER_SIZE, 0));
        assert!(!SnapshotBufferPool::can_recycle(1024, 0, SnapshotBufferPool::MAX_CACHED_BYTES));
        assert!(!SnapshotBufferPool::can_recycle(SnapshotBufferPool::MAX_CACHED_BYTES + 1, 0, 0));
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
        history.set_limits(16, 24);
        history.push("first", stroke_with_snapshot_bytes(6));
        history.push("second", stroke_with_snapshot_bytes(6));
        history.push("third", stroke_with_snapshot_bytes(6));

        assert_eq!(history.undo.iter().map(|entry| entry.label.as_str()).collect::<Vec<_>>(), vec!["second", "third"]);
        assert_eq!(history.ram_bytes(), 24);
        assert_eq!(history.ram_bytes(), history.max_ram_bytes);
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
            pool: Arc::new(SnapshotBufferPool::default()),
            accounted_bytes: 67,
        };
        assert_eq!(snapshot.ram_bytes(), 67);
    }
}
