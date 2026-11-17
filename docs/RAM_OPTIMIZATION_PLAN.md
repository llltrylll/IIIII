# RAM Optimization Plan

Scope: reduce **system RAM and transient CPU-memory peaks** in the existing application. GPU memory is tracked separately; this plan does not reopen brush, projection or normal/shading work.

## Goals and constraints

- Reduce avoidable copies of project data and image data.
- Bound memory retained by Undo history and asynchronous decode queues.
- Keep large allocations proportional to the work currently being processed, not the entire project or shelf.
- Preserve project format/version compatibility, atomic replacement, crash recovery, Undo/Redo ordering and existing image semantics.
- Do not trade RAM for unbounded GPU allocations or silently discard the current document.

## Priority order

### P0 — Project I/O peak memory
1. **Read compressed project blobs on demand.** The parser now stores file offsets and lengths instead of loading the entire project into RAM or copying all compressed blobs. The Reader seeks to and decodes one blob when requested. Mesh vertex/index/object arrays are decompressed directly into their final typed allocations.
2. **Spool compressed blobs to disk during save.** The Writer compresses and writes each blob to a temporary spool, then copies that spool into the final buffered `.mpp.tmp` before the existing atomic rename. It no longer retains all compressed blobs or builds a second full-size output Vec<u8>.
3. Later: split CPU parsing/decompression from GPU resource creation, with explicit peak-memory accounting. This is a separate change because loading must remain transactional and preserve the current document on failure.

### P0 — Undo history RAM
1. Compress snapshots one texture at a time through a one-item bounded channel, releasing each uncompressed staging copy before moving to the next texture.
2. Count snapshots while mapping/compressing, not only after compression has completed.
3. Add a configurable memory budget for painted snapshot history, alongside the existing step-count limit. Evict the oldest history entries first; never evict the newest entry while older entries remain. Keep the limit visible and adjustable in Preferences.
4. Keep structural Undo edits and GPU-resident layer ownership semantics unchanged.

### P1 — Release unused source-image bytes
- Engine::image_cache must not keep original encoded image bytes alive after the last FillImage referencing them has been dropped. Cache weak references to both the GPU texture and its encoded CPU bytes.

### P1 — Bound asynchronous CPU image queues
- Bound CPU shelf-preview results and graph-image decode results.
- Use a bounded number of worker threads and back-pressure so a large shelf or graph cannot queue an arbitrary number of decoded images in RAM.
- Upload to the GPU incrementally on the owning thread.

### P2 — Graph runtime transient allocations
- Replace per-node debug-string hashing with deterministic typed hashing.
- Reuse evaluation scratch storage only where it does not retain large graph-specific allocations indefinitely.
- Measure graph evaluation peak memory before introducing persistent plans or caches.

### P2 — Whole-document loading pipeline
- Parse and decompress CPU payloads on a worker, retaining only the data needed for GPU upload.
- Keep GPU upload on the device-owning thread.
- Use a transactional load result so a failed or cancelled open cannot partially replace the current document.
- Preserve old project versions and recovery files.

## Measurement protocol

Use the same representative project for before/after runs: 50k–100k triangles, about six Texture Sets, realistic paint/fill layers, nested masks and filters, several fill images, and normal history activity.

Record separately:
- Peak process RAM / private working set while opening and saving the project.
- Steady-state RAM after opening and after deleting a fill layer and letting history expire.
- Undo snapshot bytes, including snapshots still mapping or compressing.
- Peak RAM while a multi-channel stroke snapshot is compressed.
- Peak RAM while generating/loading the Smart Material shelf and graph images.
- GPU estimated bytes and VRAM separately; do not mix GPU allocation counts with process RAM.

A candidate is not considered a measured win until this protocol is run on the owner's Windows machine. CI proves correctness/buildability, not owner-machine RAM savings.

## Status

- [x] Source audit of project blobs, Undo snapshots, fill-image cache and asynchronous preview/decode queues.
- [x] File-backed project blob index and on-demand blob reads.
- [x] Stream compressed blobs through a disk spool and buffered atomic project save.
- [x] Incremental Undo compression and pending-memory accounting.
- [x] Configurable Undo snapshot RAM budget, including Preferences and status-bar usage.
- [x] Weak encoded-image cache for Fill source bytes.
- [x] Bounded CPU shelf-preview and graph-image decode queues.
- [ ] Verify project spool/file-backed typed readback tests and Undo budget tests on the latest HEAD; finish CI/GPU/Windows package workflows.
- [ ] Capture before/after owner-machine measurements.