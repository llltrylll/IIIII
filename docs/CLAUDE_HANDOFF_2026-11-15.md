# Claude Handoff — 2026-11-15

Repository: llltrylll/IIIII  
Working branch: gpt/integrated-three-fronts  
Code revision to inspect: a9c8879cb3c7f2a359ae311c88551b5d02160075  
Audit: docs/CODEBASE_AUDIT_2026-11-15.md

## One assignment only: prove and fix the remaining Material viewport normal-map mismatch

The owner has confirmed that Screen painting now reaches the correct surface, but Material viewport shading is still wrong. The latest screenshot still shows red slivers along narrow geometry tips. Do not treat the earlier generic normal changes as a successful fix.

Relevant code:
- src/shaders/view.wgsl — mesh_shade_inputs, tangent frame decode, final shading normal
- src/shaders/pathtrace.wgsl — mesh_surface and path-tracing shading frame
- src/mesh.rs — mikk_tangents and tangent handedness
- src/import.rs — FBX/OBJ normal provenance and winding conversion
- src/shaders/uvbake.wgsl — baked position/normal/tangent convention

Current state:
- Raster face-forwarding, tangent reconstruction and viewer-hemisphere correction are already present.
- Path tracer has a related correction.
- MikkTSpace conversion explicitly flips UV V and tangent handedness.
- FBX import accounts for negative transform determinant XOR ufbx reversed winding.
- Synthetic GPU tests pass, but the real asset still fails the owner's visual comparison.

Important validation gap:
- The grazing-angle normal test compares shader output against an analytically pre-corrected normal map built from the same formula. It does not prove Blender/InstaMAT parity.
- The inverted-normal test uses a synthetic rounded cube, not the user's Airfox FBX.
- The path-tracer test only creates/compiles the pipeline.

Required result:
1. Identify the exact frame/sign/convention divergence, supported by captured per-corner data or a minimal fixture.
2. Add a regression that would fail before the fix and pass after it.
3. Apply the smallest correction and run GPU regression + CI.
4. Report what is still unverified.

Constraints:
- Do not globally flip normals or the normal-map green channel.
- Do not modify Screen paint selection, UV scissors, brush, startup, camera navigation or PR #3/#4.
- Do not refactor adjacent code.
- Keep this as the sole Claude task; GPT will handle the wider performance audit and follow-up optimization.
