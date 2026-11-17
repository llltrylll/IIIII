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
3. **Stream the JSON manifest.** Opening parses the declared manifest range directly from the file instead of allocating a same-sized byte Vec. Saving counts serialized bytes, then writes the manifest directly to the buffered output instead of retaining the complete serialized JSON copy.
4. Later: split CPU parsing/decompression from GPU resource creation, with explicit peak-memory accounting. This is a separate change because loading must remain transactional and preserve the current document on failure.
- Triangle ID colors and per-corner vertex colors decompress directly into their final typed vectors when counts match the mesh, without an intermediate full-size byte vector.

### P0 — Undo history RAM
1. Compress snapshots one texture at a time through a one-item bounded channel, releasing each uncompressed staging copy before moving to the next texture.
2. Count snapshots while mapping/compressing, not only after compression has completed.
3. Add a configurable memory budget for painted snapshot history, alongside the existing step-count limit. Evict the oldest history entries first; never evict the newest entry while older entries remain. Keep the limit visible and adjustable in Preferences.
4. Keep structural Undo edits and GPU-resident layer ownership semantics unchanged.

### P1 — Release and deduplicate source-image bytes
- `Engine::image_cache` uses weak references to both texture and encoded bytes, and is shared by sibling Texture Set Engines. A source image remains cached only while a layer/history reference owns it.
- Project serialization deduplicates Fill images by exact `Arc` pointer identity; opening repeated references uses one weakly cached `Arc<Vec<u8>>` per blob.

### P1 — Release CPU Bake source data during GPU upload
- The full Bake path moves each large prepared array into its GPU buffer one at a time; each CPU vector is freed immediately after its upload completes.
- Low-poly corner vertices, name-group indices and cage vertices use the same consuming upload path.
- Live preview deliberately keeps the borrowed upload path because its prepared scene is cached and reused across preview resolutions.
- The scene metadata remains available for reporting and dispatch setup; only data already copied to GPU buffers is released.

### P1 — Avoid flat per-corner Bake colors
- For Mesh Parts and Materials ID sources, keep only the small triangle ID palette and read the flat color from the ID index already carried in triangle metadata.
- Allocate and upload per-corner colors only for Vertex Colors mode, where barycentric interpolation is required.
- This removes the per-corner color array and its GPU storage buffer from the common Bake modes; the dedicated Vulkan GPU test checks the empty-buffer palette path.

### P1 — Bake CPU preparation peak
- Bake preparation now receives a cancellation token. Starting a newer Bake cancels an older CPU worker; the worker checks between import/cage, component, ID, BVH, curvature, edge and packing phases, plus periodically in the long preparation loops.
- Build the bake BVH directly from indexed positions, returning only node bounds and triangle order. The bake path no longer materializes a temporary triangle array or retains a reordered CPU triangle copy.
- Build viewport/picking BVHs directly from mesh indices too; do not construct a full source-order triangle Vec before producing the final BVH leaf-order triangles.
- Normalize imported high-poly normals in place instead of allocating a second full normal array.
- Allocate low-poly material IDs and copy vertex-color payloads only when the selected ID source needs them.
- Keep source-object groups unallocated unless match-by-name finds actual matches.
- Borrow the per-triangle ID array in the default Mesh Parts mode instead of cloning it.
- Remaining large allocations: expanded per-corner position/normal/color arrays. Further compaction requires shader/layout changes and separate parity tests.
- Borrow the low-poly index slice for low-only bakes instead of cloning the complete index array; high-poly imports remain owned.
- Derive triangle centroids from compact bounds during BVH construction, removing the third per-triangle bounds vector (12 bytes per triangle when Vec3 is 12 bytes).
- Transform imported high-poly positions in place, avoiding a second full position vector during source loading.
- Cage preview workers share the editor's existing immutable Arc<Mesh> and do not clone the mesh or its index/UV payload. The worker creates only the position/normal scratch it needs and reuses one welded-vertex map for averaged directions and the retained probe.
- Cage preview high-poly inputs retain only positions, indices and optional object-name mapping; the probe builds its traversable BVH directly from indexed data instead of retaining an expanded source-triangle Vec.
- A superseded or hidden cage-preview job is cancelled at bounded work checkpoints so abandoned mesh/BVH scratch is released rather than continuing to compete with the latest preview.
- Store cage-probe hit lists as optional boxed slices; misses no longer carry a 24-byte empty Vec header for every sampled point.
- Build per-probe shell positions and colors directly into one compact overlay-vertex array instead of retaining separate scale, color and shell vectors during UI updates.
- Drop object mapping, source IDs/colors, curvature, and source geometry arrays immediately after their final CPU use in preparation, before later edge packing.

### P1 — Schedule thumbnail rebuilds incrementally
- Rebuild at most two stale layer/mask thumbnails per UI update. The rest remain invalidated and are processed on subsequent frames, avoiding a single large post-edit command-encoding and scratch-allocation batch.
- Deleted-layer thumbnails are still released immediately; the limit applies to rendering replacements, not to retained/stale resource cleanup.

### P1 — Bound asynchronous CPU image queues
- Bound CPU shelf-preview results and graph-image decode results.
- Use a bounded number of worker threads and back-pressure so a large shelf or graph cannot queue an arbitrary number of decoded images in RAM.
- Upload to the GPU incrementally on the owning thread.

### P2 — Graph runtime transient allocations
- Replace per-node debug-string hashing with deterministic typed hashing; node parameter values are borrowed from the graph/definition rather than cloned on each evaluation.
- Store bounded parameter/input slots and texture bindings in fixed-size arrays, eliminating small per-node Vec allocations; pre-size the remaining bind-group entries, plans, dependency sets and reader counts from graph size.
- Do not keep parameter-reference arrays for every planned node. Re-resolve parameters only for nodes actually rendered, reducing each pending plan's retained payload.
- No large persistent graph scratch cache is retained. Remaining: measure graph evaluation peak memory before considering reusable scratch storage.

### P2 — Whole-document loading pipeline
- Parse the manifest, validate the blob table, decompress blobs sequentially to a temporary disk spool, rebuild mesh/UV edges and build the picking BVH on a worker.
- Keep the decompressed project payload on disk; UI reads and uploads only one blob at a time, avoiding an all-textures CPU staging copy.
- Keep GPU resource creation and engine replacement on the device-owning thread.
- Do not touch the current document until worker preparation (including every blob's LZ4 integrity check, texture references/sizes, image decoding and mask generators) succeeds. If the current document changes while preparation runs, cancel the pending replacement.
- Serialize project-open workers so repeated/cancelled open requests cannot run multiple large blob-decompression peaks concurrently.
- Preserve old project versions and recovery files; remove the temporary spool after application or failure.

## Measurement protocol

Use the same representative project for before/after runs: 50k–100k triangles, about six Texture Sets, realistic paint/fill layers, nested masks and filters, several fill images, and normal history activity.

Record separately:
- Peak process RAM / private working set while opening and saving the project.
- Steady-state RAM after opening and after deleting a fill layer and letting history expire.
- Undo snapshot bytes, including snapshots still mapping or compressing.
- Peak RAM while a multi-channel stroke snapshot is compressed.
- Peak RAM while generating/loading the Smart Material shelf and graph images.
- Peak RAM during low-poly-only and high-to-low Bake preparation using fixed representative assets and settings.
- GPU estimated bytes and VRAM separately; do not mix GPU allocation counts with process RAM.

A candidate is not considered a measured win until this protocol is run on the owner's Windows machine. CI proves correctness/buildability, not owner-machine RAM savings.

## Status

- [x] Source audit of project blobs, Undo snapshots, fill-image cache and asynchronous preview/decode queues.
- [x] Stream project manifest parsing and serialization without a full-size intermediate byte Vec.
- [x] File-backed project blob index and on-demand blob reads.
- [x] Stream compressed blobs through a disk spool and buffered atomic project save.
- [x] Incremental Undo compression and pending-memory accounting.
- [x] Configurable Undo snapshot RAM budget, including Preferences and status-bar usage.
- [x] Weak encoded-image cache shared across sibling Texture Sets.
- [x] Deduplicate shared Fill image blobs in project save/open.
- [x] Bound the HDRI thumbnail decode queue to two pending CPU images.
- [x] Bounded CPU shelf-preview and graph-image decode queues.
- [x] Limit stale layer/mask thumbnail rendering to two per UI update after broad changes.
- [x] Decompress project triangle ID and per-corner vertex-color arrays directly into final typed vectors.
- [x] Cancel superseded Bake preparation workers and release their large scratch arrays at bounded checkpoints.
- [x] Remove redundant Bake BVH triangle arrays and avoid unused ID/material/object copies during CPU preparation.
- [x] Build mesh picking BVHs directly from indexed mesh vertices, including filtered object visibility BVHs, without an extra source-order triangle array.
- [x] Borrow low-poly indices during low-only Bake preparation and remove the per-triangle BVH centroid scratch vector.
- [x] Consolidate cage overlay scale/color/position scratch into one per-probe vertex buffer.
- [x] Store optional cage-probe hit arrays without per-sample Vec headers for empty/missed probes.
- [x] Cancel superseded cage-preview work at bounded CPU checkpoints to avoid concurrent obsolete mesh/BVH allocations.
- [x] Load only the high-poly data used by cage preview and build its BVH directly from indexed positions.
- [x] Keep cage preview workers on the shared Arc<Mesh> and avoid duplicate own-normal/result/weld scratch arrays.
- [x] Transform imported high-poly positions in place and release Bake preparation scratch payloads at their last use.
- [x] Normalize imported high-poly normals in place and verify indexed BVH output against the existing builder.
- [x] Remove flat per-corner color storage for Mesh Parts/Materials modes; add a Vulkan GPU regression test for palette lookup.
- [x] Consume and release prepared Bake CPU arrays incrementally during full-bake GPU upload; retain borrowed uploads for cached live previews.
- [x] Replace per-node graph debug-string hash allocations with typed hashing and borrowed parameter values; use fixed-size per-node scratch arrays and pre-size transient evaluation maps.
- [x] Keep per-node parameter reference arrays out of the retained evaluation plan; resolve them only when that node is rendered.
- [x] Verify project spool/file-backed typed reads, shared-blob reuse and Undo RAM-budget tests; latest CI, unit tests, GPU regression and Windows portable package workflows pass.
- [x] Move project manifest/blob and UI resource preflight, sequential LZ4 decompression, mesh/UV-edge construction and picking-BVH build to a serialized, cancellable worker; keep raw payloads in a temporary disk spool and upload them one at a time on the UI thread.
- [ ] Capture before/after owner-machine measurements.
