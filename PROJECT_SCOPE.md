# Project Scope: Personal Production Fork

## Mission

Maintain this fork of Material Painter as a dependable, responsive production tool for one creator. Its primary job is texturing and preparing assets for Roblox on the owner's own machine. This is not a general-purpose benchmark project and is not being prepared for public distribution.

The owner explicitly authorizes substantial, risky, and breaking internal refactors when they are the best route to a reliable, fast result. Do not preserve weak architecture merely to keep diffs small. Prefer a coherent replacement over piling workarounds onto a fundamentally poor design. Protect user projects and validate the behavior carefully.

## Target workload

| Profile | Geometry | Mesh parts | UV maps |
|---|---:|---:|---:|
| Representative production asset | 50,000–100,000 triangles | 6 | 6 |
| Maximum planned asset | Up to 200,000 triangles | Up to the same 6-part workflow | Up to 12 |

The maximum planned workload is approximately twice the representative asset. The application should remain comfortable with realistic production layer stacks, masks, filters, and texture work inside this envelope.

The owner's actual machine is the performance target. Do not claim gains from compilation or CI results alone; use profiling, GPU measurements, and real application testing. Do not optimize hypothetical workloads or hardware without a concrete reason tied to this target.

## Mandatory work order

Work in these phases, in order. Do not skip ahead because a different optimization looks interesting.

### Phase 1 — Finish brush and mask systems

This is the immediate priority. Make painting and masking reliable and responsive before starting a broad optimization pass.

Scope includes:
- 3D brush stroke responsiveness and latency, including BVH/picking, stroke-region calculation, stamping, incremental recomposition, and frame-time spikes.
- 2D/UV painting parity and correctness; fixes must not improve 3D while breaking UV workflows.
- Vertex ID masks, painted masks, mask layers, smart-mask generators, folder masks, and mask filters.
- Correct invalidation and reuse of mask/composite caches.
- Correct results across layer stacks, nested folders, channels, undo/redo, save/load, and repeated strokes.
- CPU and GPU cost of the brush/mask path, including excessive passes, texture reads, allocations, readbacks, and synchronization.

Phase 1 is not complete merely because the code compiles or the brush feels faster on a nearly empty document. It must pass relevant unit and GPU regression tests, preserve rendered output, and be tested in the app on the owner's machine with a representative painted project and meaningful mask/filter use. Capture comparable F6/performance measurements and report remaining bottlenecks honestly. Where a GPU test is ignored or unavailable in CI, do not present it as validated; run it on a capable adapter or call out the remaining validation explicitly.

### Phase 2 — General foundation optimization

Only after Phase 1 is validated, continue a wider audit and optimization of the application inside the target workload. Consider memory ownership and peaks, resource lifetime, CPU allocations, GPU resource reuse, viewport rendering, baking, import, project load/save, export, background work, and subsystem boundaries. Prioritize based on evidence and user impact, not code aesthetics alone.

### Later / deferred

A major HUD redesign and work for projects with around 100 items are distant possibilities, not current deliverables. They must not displace the core work above.

## Explicit non-goals

Unless the owner changes scope, do not spend engineering time preparing for:
- Projects containing around 100 assets/items or similarly oversized projects.
- Workloads substantially beyond 12 UV maps or 200,000 triangles for the target asset.
- Universal optimization across every possible GPU or computer configuration.
- Public releases, distribution infrastructure, or broad-market product polish.
- Major HUD redesigns or cosmetic UI work before the production foundation is solid.

## Engineering latitude and safeguards

Large or risky changes are allowed. Rewriting a subsystem, replacing a data flow, or changing internal APIs can be preferable to preserving a fragile implementation.

Risk tolerance does not mean skipping safeguards:
- Identify behavior and invariants that must remain true before replacing code.
- Protect project data and maintain old project compatibility unless a deliberate migration is designed and tested.
- Add regression tests for correctness-sensitive behavior.
- Run unit tests and relevant GPU tests; inspect rendered output after graphics changes.
- Build and run the app. Distinguish CI/build evidence from real GPU/runtime evidence.
- Measure before and after when claiming performance improvements.
- Do not hide failures by disabling tests, broadening ignored paths, or silently degrading output.
- Keep commits and PR notes understandable so another agent can continue safely.

## Shared GPT / Claude operating rules

This is one shared codebase used by GPT and Claude. Both agents must follow this file and `AGENTS.md`; Claude must also follow `CLAUDE.md`.

At the start of every task:
1. Read the current scope and agent instructions.
2. Inspect the current branch, HEAD SHA, open PRs, changed files, CI state, and relevant working diff.
3. Check recent commits/PR comments for work performed by the other agent.
4. State the intended files/subsystem and task boundary before making a substantial change.

Coordination rules:
- Never assume the other agent's context or local state is current.
- Do not concurrently edit the same files or subsystem. Agree on ownership through the active PR/comment before splitting work.
- Commit completed work in coherent units and leave a concise handoff note: changed behavior, invariants, tests run, unresolved risks, and next useful task.
- Before every write, re-fetch the latest file/branch state and use the latest SHA. Never overwrite a newer commit or another agent's changes.
- If another agent has an active uncommitted or in-progress change that cannot be inspected, do not reset or replace it; choose a separate task or ask for coordination.
- Organize code for the next contributor, regardless of whether that contributor is GPT or Claude. Avoid agent-specific abstractions or style.

## Repository boundary

All collaboration work belongs only in `llltrylll/IIIII`. Never edit or push to upstream `alla2001/material-painter`. Do not push tags. Do not merge PRs or publish releases without explicit owner instruction.
