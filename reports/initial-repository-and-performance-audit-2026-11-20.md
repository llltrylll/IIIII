# Material Painter — engineering handoff (2026-11-20)

## Repository operating model

The repository uses ten long-lived branches, as defined in AGENTS.md:
- Four code/recovery branches.
- Three diagnostics/report branches.
- Two isolated agent test lanes.
- One backup of useful test work.

No agent should create additional branches. Production-ready changes integrate into the active branch. Experimental changes live only in the two test lanes until reviewed. Do not edit the same subsystem concurrently.

## Project priority

The immediate priority remains the production foundation:
1. 3D brush latency and frame-time spikes.
2. 2D/UV correctness and parity.
3. Vertex ID and painted masks.
4. Smart masks, folders, filters and cache invalidation.
5. Correctness across realistic layer stacks, channels, undo/redo and save/load.
6. Validate on the owner's representative production workload.

Broad viewport/memory/startup optimization follows only after the production foundation is validated.

## Performance observations

The latest owner-provided profile has F5 = 221 ms, GPU composite = 193.568 ms, and approximately 16.965 derived FPS. Micro Roughness = 46.5458 ms; Silver = 42.1528 ms; Scratches = 32.5890 ms.

The previous reference had 8 layers; the latest has 10. This comparison is not controlled, so the overall regression cannot be attributed to a specific code change. The next useful measurement is a controlled F6-enabled capture with the same layer stack and viewport.

## Known validation gap

The latest capture had F6 disabled. The detailed new roughness-FBM cache counters were therefore not collected in that run. CI results do not establish performance on the owner's real machine.

## Suggested next engineering work

- Obtain equivalent A/B profiles with identical project state and F6 enabled.
- Investigate Silver and Scratches stages after confirming the cache-hit behavior for roughness FBM.
- Capture painting-time undo/readback costs, not only idle/composite F5.
- Keep UV paint parity and rendered output in regression coverage.
- Avoid HUD redesign and speculative scale targets outside the owner's stated production envelope.
