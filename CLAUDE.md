# Claude instructions

Read [PROJECT_SCOPE.md](PROJECT_SCOPE.md) first. It is the source of truth for the product objective, target workload, required sequence (Phase 1 brush/masks, then Phase 2 general optimization), refactor authorization, shared-agent coordination, and repository boundaries. Then read [AGENTS.md](AGENTS.md) for architecture, commands, invariants, and technical conventions.

## Immediate assignment order

Do not start broad application optimization yet. Finish and validate the brush and mask systems first: 3D brush latency, UV painting correctness, Vertex ID and painted masks, mask generators/layers/folders/filters, cache invalidation, and incremental compositing. Use meaningful layer/mask/filter stacks and GPU regression tests. The owner must still validate the final behavior/performance on his machine. Only after Phase 1 is validated should work move to general optimization.

The owner authorizes large, risky, or breaking internal refactors when these are the best engineering solution. Do not protect weak design for the sake of small diffs. Protect projects, preserve compatibility deliberately, add tests, and report any validation gaps.

## Shared repository protocol

GPT and Claude share this repository. Before every task, inspect current branch, HEAD, open PRs, changed files/diff, CI, and the other agent's latest commits/comments. Re-fetch files and latest SHAs before writing. Never assume earlier conversation context is current.

Do not edit the same files/subsystem concurrently. Establish task ownership in the active PR or a handoff comment. Do not overwrite the other agent's work. Commit coherent changes and leave a handoff with changed behavior, tests, unresolved risks, and next task.

All changes belong only in `llltrylll/IIIII`. Never edit or push to `alla2001/material-painter`; do not push tags, merge PRs, or publish releases without the owner's explicit instruction.
