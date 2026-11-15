# Material Painter — Bugfix Foundation

Date: 2026-11-13  
Repository: \`llltrylll/IIIII\`  
Working branch: \`gpt/bugfix-foundation-2026-11-13\`  
Base commit: \`6c57b40bc0938ed5e2b084bd9caf9419f71c2085\`  
Baseline: diagnostic hotkey fix branch; no changes to PR #4 or upstream.

## Scope and current state

This note consolidates the previous diagnostic reports with a deeper source review. It is an implementation handoff, not a claim that the defects have been fixed.

No product code has been changed in this branch yet. The next changes should be small, independently testable, and committed separately by bug. Preserve the diagnostic hotkeys and the current working branch as the baseline for reproductions.

Confidence labels:
- **Confirmed in source**: directly visible in the current code path.
- **Strong mechanism**: code can produce the reported symptom, but a controlled repro is still required.
- **Unproven**: plausible branch that must not be treated as the cause yet.

---

# 1. UV viewport looks wrong / Material mode becomes very dark

## Source findings

### 1.1 The 3D and 2D views share one display mode

**Confirmed in source.**

- \`ViewSettings.mode\` is a single \`ViewMode\` value in \`src/viewport.rs\`.
- \`ViewRenderer::base_globals()\` writes \`settings.mode.index()\` into the same shader uniform for both render paths.
- \`render_3d()\` and \`render_2d()\` both use that shared setting.
- \`fs_quad()\` in \`src/shaders/view.wgsl\` calls the same \`shade()\` function used by the mesh renderer.

Consequently, switching the 3D viewport to Base Color, Roughness, Normal, a bake map, or Material also changes the UV texture display. There is no independent UV display state.

### 1.2 Material mode in the UV view is not equivalent to the 3D material preview

**Confirmed in source; strong explanation for the very dark UV result.**

In \`fs_quad()\`:

- The UV quad samples the final material channels.
- Its normal is \`s.ts\`, a tangent-space normal, used directly as the 2D view's shading normal.
- It calls \`shade(..., vec3(0, 0, 1), 0.0)\`.
- The final argument is shadow visibility. \`key_light()\` returns zero when shadow is zero, so the UV quad receives no direct key-light contribution.
- The Material result therefore depends on environment lighting / irradiance, environment reflection, material parameters and baked AO.
- A highly metallic surface with a dark environment reflection can become very dark, even while the 3D view remains readable under its direct key light.

This is an approximate 2D material preview, not a physically equivalent render of the 3D model: tangent-space normals have no single world-space orientation in the UV plane.

### 1.3 Base Color transparency is not composited in the Base Color view

**Confirmed in source.**

- In Material mode (\`mode == 0\`), \`fs_quad()\` blends the result over a checkerboard using \`s.alpha\`.
- In Base Color mode (\`mode == 1\`), \`shade()\` returns \`s.albedo\` directly.
- The fragment then returns alpha 1.0.
- Therefore transparent Base Color texels do not show the checkerboard in Base Color mode; their RGB is shown as if opaque.

This can make the texture tile look filled or visually inconsistent with other painting applications. It is separate from the Material-mode lighting problem.

### 1.4 UV scale is not stretched by the panel aspect ratio

**Confirmed in source; previous panel-resize hypothesis rejected.**

\`uv_to_clip()\` uses one scalar \`g.view2d.x\` for both U and V. The camera zoom is a single value. That preserves equal pixel scale on both axes; a non-square panel changes the visible crop/empty margins, not the UV square's aspect ratio. The user's panel-resize test also produced no visible change.

Do not compensate by changing the mesh UVs or applying a non-uniform scale.

### 1.5 Overlay density is a separate presentation factor

The default view settings enable the UV wire and seam overlay. The 2D seam fragment is bright orange with high alpha; the polygon wire uses a pale line and the configured opacity. This may contribute to the impression that the UV area is unusually busy, but it does not explain the Material-mode darkness.

The 2D renderer draws the texture quad first, then the polygon edges and seams. The selected texture set controls which edge/seam ranges are drawn.

## Fix direction

Separate UV presentation from the 3D viewport mode. The UV view needs its own display state and rendering policy.

Suggested behavior:
1. Default UV display to Base Color (or a clearly named flat texture mode), independent of the 3D viewport mode.
2. Keep explicit channel views available in the UV panel without changing the 3D panel.
3. Composite Base Color alpha over a checkerboard in Base Color mode, not only in Material mode.
4. If a UV Material preview remains available, use a documented neutral 2D lighting model. Do not reuse the 3D key-light/shadow contract or pretend tangent-space normals are world-space normals.
5. Keep UV scale isotropic. Provide independent wire/seam visibility and opacity only if the UI redesign is in scope; do not mix that with the first rendering fix.

## Regression cases

- Change 3D mode from Material to Base Color, Normal and a bake map: UV display must remain unchanged.
- Change only the UV display mode: 3D display must remain unchanged.
- Base Color texture with alpha 1, 0.5 and 0: transparent regions must show the checkerboard; color values must remain correct.
- A fully metallic Base Color texture under a dark HDR environment must remain readable in the flat UV display.
- A square UV tile must remain square after changing the panel's width/height.
- Wire and seams must be independently toggleable if the UI exposes them separately.

---

# 2. Painting fails on one side of overlapping UVs

## Source findings

### 2.1 The UV bake stores one surface frame per texel

**Confirmed in source; principal structural limitation.**

- \`Engine::bake_uv()\` rasterizes every triangle in the active texture set into the same position, normal and tangent attachments.
- The \`uvbake\` render pipeline has no depth attachment and no per-texel surface list.
- \`fs_uv()\` writes one world position, one interpolated normal and one tangent frame into each covered texel.
- If multiple triangles overlap in UV space, their fragments compete for the same output texel. The map cannot retain every overlapping surface. Do not rely on a specific winning primitive/order as a contract.
- Dilation copies one neighboring source frame; it cannot restore an overlapping surface that was already lost in the initial bake.

The map is therefore a single-surface lookup, not a multi-layer UV-to-world mapping.

### 2.2 Screen projection depends on that single stored surface

**Confirmed in source; strong mechanism for one-sided painting.**

In \`stamp.wgsl\`:
- Each texture texel loads its stored position from \`pos_map\`.
- Screen projection projects that position through the camera.
- It checks the viewport depth buffer unless Paint Through is enabled.
- It also applies the brush angle/facing test using the stored normal.

If the stored surface at a UV texel is on the opposite side of the object, the projected point may be behind the depth buffer or face away from the camera. The texel is rejected even though another triangle sharing that UV coordinate is currently visible and should be paintable.

Paint Through only bypasses the depth-buffer comparison in \`project()\`. It does not add missing surface records and does not bypass the facing test.

### 2.3 Surface projection has the same single-surface dependency

**Confirmed in source; strong mechanism.**

Surface coverage compares the texel's stored world position and normal with the resolved brush disc:
- distance to the disc's tangent slab,
- angle/facing between stored normal and stamp normal,
- disc falloff.

A texel whose stored surface is not the surface under the cursor will not match, even when another overlapping UV triangle would match.

### 2.4 UV projection is an important control case

**Confirmed in source.**

\`coverage_uv()\` computes coverage from the texel UV and stamp UV center/radius. It does not compare the texel's stored world position or normal; the stamp pass only uses position-map alpha as a validity mask before entering the projection branch.

Therefore, with exactly coincident UVs in the **same texture set**, a UV-projection stroke should affect the shared texels regardless of which overlapping surface won the position bake. The texture is shared, so the result will appear on all geometry using those same texels. It cannot produce independent paint on two faces that share the same UV coordinates.

If UV projection also fails, the cause is likely elsewhere in stamp generation, active texture-set selection, scissor regions, channel/layer state or the actual UV correspondence. Do not attribute that failure to the one-surface position map without isolating it.

### 2.5 Cross-Texture-Set raycasts in Surface/UV stamp resolution

**Confirmed in source; implementation added in PR #8, CI validation pending.**

Previously, `resolve_stamps()` used the full-mesh BVH for Surface and UV projections. The first world-space hit could therefore belong to a different material/texture set than the Engine currently receiving the stroke. The UV and surface frame from that foreign hit were then used for the active set.

Implementation on `gpt/fix-active-texture-set-raycast`:
- `Bvh::raycast_range()` filters candidate hits by their original mesh-triangle index while preserving the shared full-mesh BVH.
- `resolve_stamps()` receives the active Texture Set's global triangle range.
- Both the primary Surface/UV hit and the mirrored UV lookup use that range.
- The normal UI and MCP painting path both pass the active Engine's range.
- Screen projection is unchanged; its visibility path remains independent.

Regression coverage:
- Two overlapping quads in separate material sets with distinct UV islands.
- Surface and UV resolution are checked against the world-space hit and UV island belonging to each selected set.
- Existing fixtures pass the full mesh range explicitly.

The build has passed on the latest PR revision. Unit and GPU checks are still running, and the actual user's asset/owner-machine behavior remains unverified.

### 2.6 Region/scissor filtering must be kept in the investigation

The app derives conservative UV rectangles from the active texture-set triangle range and the brush stamps. The engine expands these rectangles by the UV-bake dilation margin and applies scissors during painting/compositing.

This is an optimization path, not yet proven to be defective. It must be tested with overlapping triangles because a correct stamp mask can still appear to fail if the changed texels were excluded from the scissor region.

### 2.7 Existing diagnostic evidence

The Airfox capture recorded:
- 57,943 triangles total.
- 8,560 exact UV-overlap groups.
- 35,155 triangles included in those groups.
- Maximum exact-overlap stack depth: 558.
- No triangle outside the 0–1 UV tile.
- Multiple ray samples intersected several world surfaces; some were in the active \`chassi\` texture set.

These figures establish that overlapping geometry is common in this asset. They do not, by themselves, prove which projection branch caused a particular failed stroke or identify which modeling operation created an overlap. Groups may contain more than two triangles.

## Fix direction

Do not solve this by asking the user to unwrap the model or remove intentional UV overlaps. The target is to make painting work with supported overlapping UVs.

Potential implementation paths to evaluate:
1. **Screen-to-UV visibility pass:** render the visible 3D surface into a screen-space UV correspondence, then transfer brush-covered screen pixels into the UV stroke mask. This follows the actually visible surface and avoids relying on one arbitrary UV-to-world record. The design must handle coverage continuity, filtering and partial UV overlaps.
2. **Multi-surface UV lookup:** preserve more than one world-space surface record per UV texel. This is more direct but memory and cost scale with overlap depth; the observed maximum depth of 558 makes a fixed small layer count unsafe without overflow/fallback behavior.
3. **Projection-specific hybrid:** retain the current path for ordinary UVs and route only detected-overlap regions through a robust alternative. This needs a reliable overlap map and careful boundary handling.

No architecture should be selected until the minimal repro demonstrates the exact failure mode.

## Regression fixture and acceptance criteria

Build a minimal mesh containing two spatially separated quads with exactly identical UV coordinates and one shared material/texture set. Give them distinguishable world positions and normals. Include variants with partial overlap and a third material/texture set.

For each fixture:
- Verify both faces share the expected texel coordinates.
- Verify UV projection marks the shared region and displays the result on both faces.
- Verify Screen projection from each side reaches the shared texels when that side is visible.
- Verify Surface projection follows the intended surface and does not jump to the other face.
- Repeat with Paint Through on/off; it may change occlusion only, not surface-frame availability.
- Verify the active-set range and scissor rectangles include the affected UV region.
- Verify a stack deeper than the chosen supported limit has defined fallback behavior.

The test must compare both directions (A then B, B then A) because the UV bake's winning overlap fragment may depend on rasterization/order.

---

# 3. Inverted normals / geometry looks fine but painting is rejected

## Source findings

### 3.1 Imported normals can be retained without winding correction

**Confirmed in source.**

The FBX importer transforms normals using \`ufbx::matrix_for_normals()\` and \`transform_direction()\`, then normalizes them. The imported path does not call the procedural \`fix_winding()\` helper.

\`Mesh::from_parts()\` generates MikkTSpace tangents and can split vertices, but does not globally reconcile imported face winding with the supplied normals. The procedural \`fix_winding()\` helper is used by procedural mesh generation, not by the generic imported-mesh path.

This does not prove the user's FBX is wrong. It means an imported mismatch can survive unless the source importer/ufbx conversion has already resolved it.

### 3.2 Missing normals invoke a potentially risky positional weld

**Confirmed in source; strong mechanism for specific malformed/coincident meshes.**

- The import builder tracks whether all corners across the imported scene have normals. If the full data set is incomplete, the resulting \`RawMesh\` omits the normal array and \`Mesh::from_parts()\` generates all normals.
- \`smooth_normals()\` calls \`weld_positions()\`, which groups positions after rounding each coordinate to a 1e-4 grid.
- It sums signed triangle cross products for every face at each welded position and normalizes the result.
- The weld key does not include object, material, UV seam, smoothing group or connected component.

For coincident shells/faces with opposite winding, contributions can cancel. Disconnected surfaces that happen to touch or occupy the same quantized position can also be averaged together. A near-zero accumulated vector becomes a zero normal through \`normalize_or_zero()\`.

This path only applies when normals are absent/incomplete as defined by the importer. If the user's model contains complete authored FBX normals, this fallback is not responsible for its result.

### 3.3 Display shaders can conceal orientation problems

**Confirmed in source.**

Raster viewport:
- \`mesh_shade_inputs()\` receives \`front_facing\`.
- For a back-facing fragment it negates the interpolated vertex normal before shading.
- It does not independently compare the vertex normal with a geometric face normal; the value called \`gn\` in that function is derived from the interpolated vertex normal.

Path tracer:
- It derives a geometric normal from the triangle positions.
- It flips that geometric normal toward the incoming ray.
- It flips the interpolated shading normal to agree with the oriented geometric normal.
- It further biases the final shading normal toward the viewer if it points away.

These are view-oriented shading decisions. They can make a mesh render acceptably from a camera angle while leaving imported winding/normals inconsistent for other operations.

### 3.4 Screen painting consumes the uncorrected bake normal

**Confirmed in source; strong mechanism for visible-but-unpaintable faces.**

- \`uvbake.wgsl::fs_uv()\` stores the interpolated mesh normal into the UV normal map without the raster viewport's \`front_facing\` adjustment.
- \`stamp.wgsl::project()\` applies \`facing(dot(n, camera - p))\` using that baked normal.
- With the default 85-degree angle limit, a normal pointing away from the camera fails the facing smoothstep and rejects screen coverage.
- Paint Through only skips the depth comparison; it does not bypass the facing test.
- UV projection bypasses this normal-facing test; Surface projection compares texel normal to the stamp normal. If both came from the same consistently inverted shading-normal field, that pairwise comparison may still pass.

This can produce a visible discrepancy: the rendering path reorients normals for display, but the Screen painting path uses the original orientation.

### 3.5 The diagnostic counter is a clue, not a verdict

The current mesh report compares per-triangle geometric face normals with vertex shading normals and counts aligned/opposed/mixed cases. The ray trace records both \`geomN\` and \`shadeN\`, plus their dot product.

An opposed count is not sufficient to auto-flip triangles. It can be caused by reversed winding, deliberately inverted shading normals, open/double-sided geometry, or a source import convention. The correct action depends on separating those cases.

## Fix direction

First separate **geometric orientation** from **shading-normal orientation** throughout the paint path.

Potential correction:
1. Carry or derive a geometric face normal for visibility/facing decisions; do not use a shading normal as the sole camera-facing gate.
2. Preserve authored shading normals for lighting and normal-map tangent basis.
3. Diagnose import winding and geometric-normal alignment explicitly.
4. Decide whether missing normals should be generated per connected/smoothing component rather than by a global position-only weld.
5. Never auto-flip all normals based on one global statistic. Preserve valid double-sided/open surfaces and intentional normal edits.

## Regression fixture and acceptance criteria

Create four separate variants of the same simple geometry:
- correct winding + outward normals,
- correct winding + inward shading normals,
- reversed winding + outward normals,
- reversed winding + inward normals.

Add missing-normal variants with two coincident, oppositely oriented shells and with two independent components sharing positions.

Verify:
- import report classifies geometry winding and shading-normal orientation independently;
- raster and path-traced display behavior is documented;
- Screen projection can paint the visible front surface without accidentally accepting the far side;
- Paint Through changes occlusion only;
- Surface projection remains consistent across smooth and hard edges;
- UV projection remains a valid control case;
- zero/near-zero generated normals are detected rather than silently treated as ordinary directions.

---

# Implementation order

## Phase A — lock in reproductions (no behavior changes)

1. Add a deterministic stacked-UV fixture and projection tests.
2. Add a normal/winding fixture matrix and an import fallback test for opposing coincident shells.
3. Add a UV renderer test or reproducible capture with known color/alpha values.
4. Record the expected behavior for each projection and each viewport mode.

## Phase B — implement the low-risk UV viewer separation

Create separate 2D display state, fix Base Color alpha preview, and make the flat UV view independent of 3D PBR/key-light state. Keep this as a separate commit.

## Phase C — Screen brush visibility follows imported normal orientation

For Screen projection, the paint-only depth pass includes only surfaces whose imported vertex normals face the camera. An inward-facing foreground shell therefore does not block the brush, allowing a correctly oriented surface behind it to receive paint. The viewport's shading normal remains a separate concern: raster shading still face-forwards the geometric normal while preserving tangent-space details. Surface and UV projection are not changed by this Screen-only rule.

## Phase D — repair stacked UV painting

Choose between screen-to-UV mapping, layered surface records, or a hybrid after the repro and memory/performance measurements. Keep this separate from normal changes so each regression can be attributed.

## Guardrails

- Do not edit PR #4, merge it, or touch the original upstream repository.
- Do not remove intentional UV overlaps or demand unique UVs as a workaround.
- Do not globally flip imported normals or winding.
- Do not let the 2D UV view change the 3D display mode.
- Do not use Paint Through as a substitute for correct multi-surface handling.
- Do not treat the diagnostic overlap counts as proof of a particular stroke failure.
- Run the available Rust tests and Windows packaging workflow after each isolated code change. A successful compile alone is not proof that GPU painting behavior is correct.


---

# Additional cross-cutting findings from the second source pass

These findings refine the three investigations and should be included in the implementation review.

## A. Imported normal completeness is now tracked per object

**Resolved in the integrated branch and covered by importer tests.**

The FBX/OBJ builder records missing or invalid normals per object. Fallback generation is limited to the affected object and uses logical source-vertex identity, preserving authored normals on other objects and across UV splits. Zero-length and non-finite imported normals are treated as missing.

Regression coverage includes `missing_normals_are_generated_per_object_without_replacing_authored_normals`, `partial_normals_preserve_authored_vertices_and_generate_missing_ones`, and the coincident-shell OBJ fixture. The earlier global-`all_normals` concern is no longer the current implementation.

## B. Raster and path-traced viewport geometric normals now use geometry

**Corrected in the integrated branch and covered by GPU regression tests.**

In `mesh_shade_inputs()`, the raster path derives the geometric normal from `cross(dpdx(world_position), dpdy(world_position))`, with a safe fallback for degenerate derivatives. The path tracer derives it from triangle edges. Both face-forward this geometric normal toward the camera/ray and keep imported shading normals separate for tangent-space shading.

Do not use this face-forwarded display normal as the Screen brush's culling rule: Screen painting now uses the original imported normal direction in its separate paint-depth pass.

## C. Surface and UV stamp resolution are not scoped to the active texture set

**Confirmed in source; likely secondary failure mode for Bug 2.**

`resolve_stamps()` receives the full mesh and full BVH. For Surface and UV projections it calls `bvh.raycast()`, which returns the nearest hit across all triangles. It does not receive the active texture-set triangle range and does not reject a hit from another set before computing the stamp's world center, normal, UV and UV radius.

The engine later paints only the active texture set's texture. This can misassociate a Surface/UV stamp with the wrong face when material boundaries overlap or when the ray first hits a different texture set.

This is distinct from the one-surface-per-texel UV bake limitation. Add a fixture where two coincident or near-coincident surfaces use different material sets and have different UV coordinates. The active-set behavior must be explicit: either resolve to the active set's first valid hit, or provide a UI rule that selects the hit set. Do not permit an accidental cross-set UV stamp.

## D. Diagnostic interpretation guardrails

- The 8,560 exact-overlap groups describe coordinate overlap only; they do not identify the modeling operation that produced each overlap.
- The reported maximum depth of 558 is an observed stack depth. It does not dictate a fixed layered representation; any multi-layer design must handle overflow.
- A ray with multiple world intersections only proves geometric ray overlap. It does not prove a stroke was written or composited.
- A `normals_opposed` diagnostic compares geometric face orientation with vertex shading normals. It does not identify which convention is authoritative for every open or double-sided surface.
- A zero normal can arise from fallback-normal cancellation; it must be reported explicitly rather than treated as an ordinary vector.

## E. Additional regression fixtures

1. Multi-object import: object A has authored normals, object B has none. Confirm whether A's authored normals survive.
2. Coincident opposite shells with no imported normals: inspect the generated vectors at their shared positions and report zero/near-zero results.
3. Same world-space overlap, two different texture sets, distinct UV coordinates: run Screen, Surface and UV projections while changing the active set.
4. Raster vs path tracer: use deliberately inverted shading normals with unchanged winding, then reversed winding with unchanged normals. Compare actual geometric orientation, displayed shading and brush acceptance.

---

# 2026-11-14 implementation update

## Screen painting through inward-facing foreground geometry

The normal-facing depth pass was added as a separate MSAA depth target. It is independent of the viewport's ordinary depth buffer, which continues to render all geometry. The pass writes depth only when the imported interpolated normal faces the camera:

`dot(imported_normal, camera_position - world_position) > 0`.

The cache key includes camera view-projection, viewport dimensions, mesh index count, and visible texture-set ranges. Mesh reload explicitly invalidates it. Material edits do not rebuild it.

The Screen stamp shader still projects each triangle into its own UVs and compares against this depth target. As a result, an inward-facing foreground shell is skipped and the nearest correctly oriented surface behind it becomes the paint target. This does not change the displayed material shading or the Surface/UV brush projections. Paint Through continues to bypass depth testing.

GPU regression results on Mesa Lavapipe:
- Inward-facing front chart: 0 painted texels.
- Correctly oriented background chart: 1,976 painted texels.
- Partially overlapped normal-facing foreground chart: 942 painted texels; hidden counterpart: 0.
- Thin foreground occluder: 0 background texels under the bar.
- Existing stacked-UV, sloped-surface, brush projection, mask and composite parity tests passed in the same workflow.

The user screenshots' remaining red speckling has not been conclusively identified as a separate tangent/normal-map defect. The new test confirms the wrong-surface selection path, not every possible visual artifact.

## Performance and startup follow-up

- The 2048² key-light shadow map is cached until light transform, mesh/visible-set inputs, or explicit mesh invalidation changes.
- Secondary shelf GPU work is throttled to one step every 80 ms and pauses while a mouse button is held. CPU scan-result polling continues.
- This reduces background contention after the main window appears. It does not prove that synchronous work inside `PainterApp::new` is no longer a startup bottleneck. The existing `MP_STARTUP_TRACE=1` report records `ViewRenderer::new`, `GraphStore::new`, `smart_io::list`, environment decode/upload, and shelf stages separately.
