# Branch cleanup plan — 2026-11-20

The target is exactly 10 long-lived branches. The four code/recovery branches and six new lanes have been created. The remote currently still contains the original 57 branches plus the six newly created ones: 63 total. The owner must delete surplus branches through GitHub's branch-management UI.

## Canonical ten branches — KEEP

1. `perf/vulkan-fill-color-fbm-cache-20261120` — active
2. `perf/vulkan-cache-claude-ui-20261119` — backup
3. `feat/composite-uv-diagnostics-20261118` — backup-2
4. `master` — fork baseline
5. `diagnostics/user-benchmarks`
6. `diagnostics/app-runtime`
7. `reports/engineering`
8. `test/agent-lab-a`
9. `test/agent-lab-b`
10. `backup/test-lab`

## APAGA — no unique commits against the active branch and no open PR dependency

The following 16 branches compare as having zero commits ahead of the active branch. They are not used as heads or bases of currently open PRs. Based on the current comparison, they are the first deletion candidates:

- `backup/perf-general-frame-optimization-2026-11-18`
- `backup/pre-diagnostic-2026-11-13-cbb451e`
- `claude/color-polish`
- `claude/startup-async-trace`
- `diagnostics/gpu-stroke-timestamps`
- `feature/color-panel-final-fork-only`
- `fix/mask-paint-restore`
- `fix/stroke-end-readback-reuse-20261117`
- `gpt/bugfix-foundation-2026-11-13`
- `gpt/diagnostic-hotkeys`
- `gpt/diagnostic-hotkeys-fix`
- `gpt/integrated-three-fronts`
- `gpt/project-settings`
- `gpt/stacked-uv-visible-pass`
- `gpt/startup-async-load`
- `ui/technical-status-row-camera-click-20261118`

These have zero unique commits according to GitHub's compare endpoint against the active branch. This is a strong deletion signal, but review their UI/commit history before final deletion if a particular feature is still needed.

## SEGURA — referenced by open pull requests

Do not delete these yet. They are referenced as PR heads or PR bases:

- `claude/mask-paint-gpu-gap` — base of PR #3.
- `gpt/fix-3d-stroke-region-behind-camera` — head of PR #3.
- `gpt/fix-active-texture-set-raycast` — head of PR #8 and base of PR #9.
- `gpt/fix-surface-projection-overlapped-uvs` — head of PR #9 and base of PRs #18–#20.
- `gpt/multi-scissor-uv-regions` — head of PR #4 and base of PRs #12–#13.
- `gpt/project-sidebar-map-indicators` — base of PRs #7, #8, #10 and #11.
- `perf/general-frame-optimization-20261116` — head of PR #20.

These may have no unique commits versus the active branch, but deleting them now would disrupt open PR references. Decide whether to close, retarget or preserve each PR first.

## REVISAR / PRESERVAR — unique commits not yet accounted for

These 30 branches contain commits that GitHub's compare endpoint reports as ahead of the active branch. Do not delete until the useful changes are inspected and either ported to the active branch, preserved in one of the test lanes, or explicitly rejected:

- `build/uv-bake-test-20261118` — 1 unique commit; temporary GPU/build workflow.
- `claude/fbx-mirror-shading` — 1; FBX mirrored winding fix.
- `claude/mask-equivalence-tests` — 2; mask/render equivalence tests.
- `claude/normal-shading-parity` — 2; normal/winding parity report.
- `claude/startup-trace` — 2; startup tracing.
- `color-tray-eraser` — 2; color tray/eraser work.
- `copilot/polish-ui-shortcuts` — 4; UI/shortcut and undo-state changes.
- `feat/technical-info-drawer-20261118` — 1; technical diagnostics UI.
- `feature/artist-workflow-improvements` — 2; PR #1 merged into master, but not represented by the active branch compare.
- `feature/codebase-review-optimization` — 66; substantial audit/optimization line.
- `feature/color-interaction-vertex-id-final` — 6; Vertex ID interaction/palette work.
- `feature/color-panel-enhanced-clone` — 5; enhanced color-panel work.
- `feature/color-panel-merged-master` — 1; color-panel integration.
- `feature/custom-hsv-color-picker` — 5; superseded PR #2, but has unique commits.
- `fix/mask-paint-scissor` — 1; mask-paint region optimization.
- `fix/screen-msaa-paint-coverage-20261119` — 3; open PR #21, Screen/MSAA correctness fix.
- `fix/technical-panel-package-20261118` — 1; package workflow.
- `fix/uv-bake-package-20261118` — 1; UV bake package workflow.
- `gpt/empty-visible-bvh-guard` — 2; open PR #13.
- `gpt/perf-composite-pass-allocation` — 1; composite pass allocation work.
- `gpt/phase1-thumbnail-regression` — 4; open PR #7; PR #14 also merged this head into a test line.
- `gpt/preflight-project-open-validation` — 4; open PR #11.
- `gpt/reuse-stroke-stamp-buffers` — 2; open PR #12.
- `gpt/test-mask-cache-material-invalidation` — 4; open PR #10.
- `perf/vulkan-viewport-cache-20261119` — 9; PR #22 closed unmerged, contains unique viewport-cache work.
- `test/claude-normal-shading` — 3; normal-shading test line.
- `test/gpt-camera-pivot-fix` — 13; open PR #18.
- `test/gpt-combined` — 9; closed, unmerged test PR #17.
- `test/gpt-combined-package` — 17; test/package line.
- `test/perf-recovery-20261116` — 17; open PR #19.

## Important PR status

Currently open PRs requiring review/closure/retargeting before deleting their head/base branches:
- #3, #4, #7, #8, #9, #10, #11, #12, #13, #18, #19, #20, #21.

Closed unmerged PRs:
- #2 (superseded HSV picker)
- #16 and #17 (test-only)
- #22 (viewport cache)

Merged PRs:
- #1, #5, #6, #14.

Do not assume that a closed or merged PR means every related feature is already present in the active branch. The branch compare results above are against the active branch.

## Management

Manage branches here: https://github.com/llltrylll/IIIII/branches

GitHub branch refs cannot be renamed through the available integration used for this task. The word **APAGA** is therefore used as a classification in this report, not as a literal prefix on the remote branch name.

After deletion/cleanup, verify exactly the ten canonical names remain.
