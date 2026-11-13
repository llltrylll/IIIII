# Claude instructions

Read [PROJECT_SCOPE.md](PROJECT_SCOPE.md) first. It is the source of truth for the production objective, defined production workload ceiling, required sequence (first establish and validate the painting/mask foundation, then optimize the rest), refactor authorization, shared-agent coordination, and repository boundaries. Then read [AGENTS.md](AGENTS.md) for architecture, commands, invariants, and technical conventions.

## Immediate assignment order

Do not start broad application optimization yet. Establish and validate the core production foundation: 3D brush latency, UV painting correctness, Vertex ID and painted masks, mask generators/layers/folders/filters, cache invalidation, and incremental compositing. Use realistic layer/mask/filter stacks and representative assets within the defined workload limits. Air Fox is an example, not a required benchmark. The owner must validate final behavior and performance on his actual machine. Only after the foundation is solid should work move to general optimization within the defined workload ceiling.

The normal production workload is 50,000–100,000 triangles, around 6 MeshParts, and around 6 UV maps. The absolute ceiling is 200,000 triangles, around 6 MeshParts, and up to 12 UV maps. Air Fox is only an example asset, not a mandatory benchmark. Projects around 100 items are a distant future possibility, not current scope. A major HUD redesign is deferred for months or until the owner explicitly reprioritizes it.

The owner authorizes large, risky, or breaking internal refactors when these are the best engineering solution. Do not protect weak design for the sake of small diffs. The goal is a reliable production tool for the owner's own machine, not a public release or universal product. Protect projects, preserve compatibility deliberately, add tests, and report validation gaps.

## Shared repository protocol

GPT and Claude share this repository. Before every task, inspect current branch, HEAD, open PRs, changed files/diff, CI, and the other agent's latest commits/comments. Re-fetch files and latest SHAs before writing. Never assume earlier conversation context is current.

Do not edit the same files/subsystem concurrently. Establish task ownership in the active PR or a handoff comment. Do not overwrite the other agent's work. Commit coherent changes and leave a handoff with changed behavior, tests, unresolved risks, and next task.

All changes belong only in `llltrylll/IIIII`. Never edit or push to `alla2001/material-painter`; do not push tags, merge PRs, or publish releases without the owner's explicit instruction.
