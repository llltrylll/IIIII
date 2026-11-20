# Parked Engine GPU residency — experimental audit

Date: 2026-11-20  
Implementation lane: `test/agent-lab-b`  
Review: [PR #25 — Experimental: Evict parked world-space UV maps](https://github.com/llltrylll/IIIII/pull/25)  
Prototype head at report time: `ffb57ca05543ef253920be2063dff72aa5e18f83`

## Baseline from the owner's diagnostic snapshot

The diagnostic captured seven 2048² Texture Sets: one active Engine and six parked Engines. Application-side GPU allocation estimates were approximately 2.11 GiB total. Each parked Engine retained approximately:

| Resource category | Approx. allocation | Residency decision |
|---|---:|---|
| World-space surface maps (position, normal, tangent) | 64 MiB | Evict while parked unless the active view/operation needs them |
| Five packed baked mesh-map textures and fixes | 112 MiB | Keep resident in this prototype |
| Final material/accumulation output | 68 MiB | Keep resident to preserve simultaneous material display |
| Other edit-time working resources | Mostly already released by the existing parked-working-set path | Leave existing behavior unchanged |

The 64 MiB surface-map figure follows the current formats at 2048²: position RGBA16F (8 bytes/texel), normal RGBA8 (4), tangent RGBA8 (4). Six parked Engines therefore hold roughly 384 MiB in these three maps alone.

These are application allocation estimates, not physical VRAM telemetry. They do not prove that the 25 FPS observation was caused by VRAM pressure.

## Prototype: mode-aware surface-map residency

The experimental change keeps all final material channels and all five packed baked mesh-map textures resident. It only replaces the parked Engine's position, normal and tangent textures with 1×1 placeholders when the current view does not sample them.

The maps are restored for:
- Position and World Normal 3D view modes, for visible Texture Sets;
- custom Material preview shaders, which can read the full Surface input;
- reactivation of a Texture Set;
- parked-set recomposition, because procedural layers can sample position, normal and tangent inputs.

The ordinary built-in Material shader does not sample those per-Engine world-space surface maps. It uses the final material channels and selected baked mesh maps instead. Consequently, their eviction should not alter the Material image.

The implementation also:
- retains the UV bake identity while its textures are nonresident;
- regenerates unchanged maps without marking the material composite dirty;
- invalidates viewport bind groups when resources are replaced;
- explicitly clears cached bind groups for hidden sets, since the renderer skips refreshing hidden-set bindings;
- avoids restoring surface maps for hidden sets.

## Expected saving

At 2048², the estimated reduction is about 64 MiB per parked Engine, or 384 MiB across six parked Engines. Applied to the observed application allocation estimate, that would move the estimate from approximately 2.11 GiB to approximately 1.74 GiB while the built-in Material view is active and all six parked sets are visible.

This is a theoretical resource-allocation delta, not a measured physical-memory reduction and not an FPS claim. The owner's Vega 8 runtime must verify the actual effect.

## Correctness gate

Added ignored GPU regression test:

`preview_test::parked_surface_maps_release_and_restore_without_visual_changes`

The test:
1. creates two Texture Sets and renders both together;
2. records the multi-set Material and Position images;
3. evicts one set's surface maps and verifies its resource estimate falls;
4. verifies the Material image remains byte-identical;
5. regenerates the same Position/Normal maps and compares their readback bytes;
6. verifies the Position image remains byte-identical;
7. exercises the hidden-set cached-bind-group case and verifies eviction releases the stale binding.

The test is added as a dedicated step to `.github/workflows/gpu-regression.yml`, which runs on Vulkan software rendering. Windows CI and the GPU regression workflow are the required automated gates. Neither substitutes for the owner's Vega 8 runtime test.

## Not changed

No changes were made to brush stamping, picking, BVH, brush projection, stroke spacing, Undo snapshot encoding, stroke-end behavior, or paint-layer composition logic. No production change was made to the active development branch; the work remains in the experimental test lane and a draft PR.

## Next residency opportunity — not implemented

The five baked maps are packed as A–E. The shader's built-in Material path normally needs map A for AO when AO is enabled, and map D when baked normal is enabled. Groups B, C and E are not normally sampled by that path:
- B: sky exposure, edge distance, crease distance, part random;
- C: part gradient and bent normal;
- E: ID.

B (RGBA16F), C (RGBA8) and E (RGBA8) account for another 16 bytes/texel, or approximately 64 MiB per 2048² Engine. Across six parked Engines this is another theoretical 384 MiB.

Do not evict these maps by simply replacing them with placeholders. They are consumed by diagnostic map views, custom material graphs, generators and mesh-map masks. A safe next phase needs a backing and restoration policy, likely using the existing bake-cache records with asynchronous loading, per-set cache identity, and correct reapplication of map levels/fixes. A mode switch must continue to preserve all visible Texture Sets, even while those less-common maps are being made resident.

## Decision

The surface-map prototype is a bounded, low-risk first step with an explicit byte-for-byte visual regression. Keep it experimental until Windows CI, software-Vulkan GPU regression, and the owner's Vega 8 tests pass. Do not merge PR #25 or port it to the active branch without review and measured runtime evidence.
