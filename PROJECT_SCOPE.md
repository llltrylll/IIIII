# Project Scope: Personal Production Fork

## Mission

Maintain this fork of Material Painter as a dependable, responsive production tool for one creator. The main use is texturing and preparing assets for Roblox. The goal is not to build a general-purpose benchmark for extreme scenes or a product intended for public distribution.

The owner explicitly authorizes substantial refactors, replacements, and other high-impact changes when they are the soundest way to reach the desired result. Do not preserve weak architecture merely to keep diffs small. Make invasive changes deliberately: understand the existing behavior, protect document data, add or update tests, and verify the result.

## Target workload

Use this as the actual engineering envelope:

| Profile | Geometry | Mesh parts | UV maps |
|---|---:|---:|---:|
| Representative production asset | 50,000–100,000 triangles | 6 | 6 |
| Maximum planned asset | Up to 200,000 triangles | Not above the representative workflow by design | Up to 12 |

The maximum planned workload is approximately twice the representative asset. The application should remain comfortable with realistic production layer stacks, masks, filters, and texture work inside this envelope.

The owner's actual machine is the performance target. Do not claim performance improvements from compilation or CI results alone; use profiling, GPU measurements, and real application testing. Do not optimize around hardware or workloads that are not part of this use case without a concrete reason.

## Priority order

1. **Correctness and data integrity.** Painting, masks, filters, channels, 2D/3D views, undo/redo, save/load, baking, and export must produce correct and predictable results. Never trade silent corruption or lost work for a speed improvement.
2. **Stability.** Avoid crashes, hangs, invalid GPU state, stale caches, broken projects, and regressions between editing sessions.
3. **Responsiveness on the owner's machine.** Prioritize interactive painting, layer recomposition, mask/filter evaluation, viewport responsiveness, memory use, and project load/save within the target workload.
4. **Maintainability for multiple AI-assisted contributors.** Keep subsystem boundaries understandable, invariants explicit, tests useful, and changes easy for another agent to review and continue.

## Explicit non-goals

Do not spend engineering time preparing for these hypothetical requirements unless the owner changes scope:

- Projects containing around 100 assets/items or similarly oversized production scenes.
- Workloads substantially beyond 12 UV maps or 200,000 triangles per target asset.
- Universal optimization across every possible GPU and machine configuration.
- Public releases, distribution infrastructure, or polishing the application for a broad customer base.
- Major HUD redesigns or cosmetic UI work ahead of core reliability and performance.

These are not promises that such projects could never work. They are simply not the current engineering target.

## Engineering latitude and safeguards

Large or risky changes are allowed. A full subsystem rewrite can be preferable to accumulating patches around a poor design. Choose the implementation that produces the best reliable program, not the one with the smallest diff.

Risk tolerance does not mean skipping validation. For consequential changes:

- Identify the behavior and invariants that must remain true.
- Prefer a coherent replacement over layered compatibility hacks when the old design is the problem.
- Preserve old project compatibility unless a deliberate migration is designed and tested.
- Add regression tests for correctness-sensitive behavior.
- Run the relevant unit and GPU tests; inspect rendered output after graphics changes.
- Build and run the application. Separate CI evidence from real GPU/runtime evidence.
- Measure before and after when claiming performance gains.
- Keep commits understandable and avoid unrelated churn.

## Working across GPT and Claude

Both agents may work on this fork. At the start of a task, read this file and `AGENTS.md`; Claude should also follow `CLAUDE.md`. Inspect the current branch, HEAD, worktree/diff, and open PR before editing. Do not assume a previous agent's in-memory context describes the current repository state.

Coordinate through committed code, tests, and concise notes in the relevant PR. Do not overwrite another agent's work or reset shared changes without checking the current state. The codebase should be organized for whichever agent takes the next task; do not add abstractions solely to make one agent comfortable.

## Repository boundary

All work for this collaboration belongs in `llltrylll/IIIII`. Never edit or push to the upstream repository `alla2001/material-painter`. Do not push tags. Do not merge a PR or publish a release without the owner's explicit instruction.
