# Project Scope: Personal Production Fork

## Mission

Build and maintain a dependable, responsive Material Painter production tool for one creator's Roblox asset workflow. This fork is a private-in-practice production tool for the owner's own machine, not a public product, general-purpose benchmark, or distribution project.

The immediate goal is to establish an exceptionally solid foundation: painting, masks, project integrity, and the core production workflow must work reliably and comfortably on the owner's actual computer. Once that foundation is solid, the owner can use the application for real production. A sound architecture also makes future fixes and improvements easier for both GPT and Claude.

The owner explicitly authorizes substantial, risky, and breaking internal refactors when they are the best route to a reliable, fast result. Do not preserve weak architecture merely to keep diffs small. Prefer a coherent replacement over piling workarounds onto a fundamentally poor design. Make engineering decisions that leave the codebase understandable and comfortable for either agent to maintain.

## Target workload — representative Roblox assets

Air Fox is an example of the kind of asset the owner creates, not a special benchmark or a required test model. The limits below describe the owner's normal asset workload in general.

| Profile | Geometry | Mesh parts | UV maps |
|---|---:|---:|---:|
| Normal production workload | 50,000–100,000 triangles | Around 6 | Around 6 |
| Absolute planned ceiling | Up to 200,000 triangles | Around 6 | Up to 12 |

The ceiling is approximately twice the normal workload. The application should remain comfortable with realistic production layer stacks, masks, filters, and texture work within this envelope. Do not spend time preparing for workloads beyond it unless the owner explicitly changes the scope.

The owner's actual machine is the performance target. Profile and validate on that machine; do not optimize for hypothetical hardware or claim gains from compilation/CI results alone. Correctness and responsiveness for the owner's real production workflow matter more than universal compatibility or benchmark records.

## Mandatory work order

Work in these phases, in order. Do not skip ahead because a different optimization looks interesting.

### Phase 1 — Establish and finish the production foundation

This is the immediate priority. Make the core painting and masking workflow reliable, correct, and responsive before broad optimization.

Scope includes:
- 3D brush stroke responsiveness and latency, including BVH/picking, stroke-region calculation, stamping, incremental recomposition, and frame-time spikes.
- 2D/UV painting parity and correctness; fixes must not improve 3D while breaking UV workflows.
- Vertex ID masks, painted masks, mask layers, smart-mask generators, folder masks, and mask filters.
- Correct invalidation and reuse of mask/composite caches.
- Correct results across realistic layer stacks, nested folders, channels, undo/redo, save/load, and repeated strokes.
- CPU and GPU cost of the brush/mask path, including excessive passes, texture reads, allocations, readbacks, and synchronization.
- Stability and integrity of normal production workflows on the owner's machine.

Phase 1 is not complete merely because the code compiles or the brush feels faster on a nearly empty document. It must pass relevant unit and GPU regression tests, preserve rendered output, and be tested in the app on the owner's machine with a representative production asset/project within the workload limits above and meaningful mask/filter use. Air Fox may be used as one convenient example, but is not mandatory. Capture comparable F6/performance measurements and report remaining bottlenecks honestly. Where a GPU test is ignored or unavailable in CI, do not present it as validated; run it on a capable adapter or call out the remaining validation explicitly.

### Phase 2 — Optimize the production foundation

After Phase 1 is validated and the application is usable for real production, audit and optimize the rest of the core application within the defined production workload envelope. Consider memory ownership and peaks, resource lifetime, CPU allocations, GPU resource reuse, viewport rendering, baking, import, project load/save, export, background work, and subsystem boundaries.

Prioritize by measured user impact and risk to production reliability, not code aesthetics alone. Do not optimize subsystems merely because they are interesting; establish whether they materially affect the target workflow.

### Later / explicitly deferred

- A major HUD redesign may be considered months from now, only after the production foundation is solid and the owner has time and suitable paid-agent capacity. It is not a current deliverable.
- Projects containing around 100 items/assets are a very distant possibility and are not a target for current engineering.
- The owner may request other changes later, but do not anticipate them by expanding the present scope.

## Explicit non-goals

Unless the owner changes scope, do not spend engineering time preparing for:
- Projects containing around 100 assets/items or similarly oversized projects.
- Workloads beyond 12 UV maps, 200,000 triangles, or substantially beyond the owner's normal approximately 6-part asset workflow.
- Universal optimization across every possible GPU or computer configuration.
- Public releases, distribution infrastructure, broad-market product polish, or preparing this private production tool for external users.
- Major HUD redesigns, cosmetic UI work, or speculative future features before the production foundation is solid.

## Engineering latitude and safeguards

Large, risky, and breaking internal changes are allowed. Rewriting a subsystem, replacing a data flow, changing internal APIs, or removing a weak implementation can be preferable to preserving it. The desired end state is a program that works well for production, not a small diff or superficial compatibility.

Risk tolerance does not mean skipping safeguards:
- Identify behavior and invariants that must remain true before replacing code.
- Protect user projects and existing project data. Preserve old project compatibility unless a deliberate migration is designed and tested.
- Add regression tests for correctness-sensitive behavior.
- Run unit and relevant GPU tests; inspect rendered output after graphics changes.
- Build and run the app. Distinguish CI/build evidence from real GPU/runtime evidence.
- Measure before and after when claiming performance improvements.
- Do not hide failures by disabling tests, broadening ignored paths, or silently degrading output.
- Keep commits and handoff notes understandable so either GPT or Claude can continue the work.

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
