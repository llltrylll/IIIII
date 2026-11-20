# Test lane B

Use this branch for a second independent agent experiment or regression investigation.

Rules:
- Record the task owner, exact base commit, hypothesis, files/subsystem and success criteria before editing.
- Keep experiments isolated and bounded.
- Do not edit the same files/subsystem as the owner of test/agent-lab-a.
- Run relevant unit and GPU tests; record ignored or unavailable tests honestly.
- Do not treat CI as validation on the owner's GPU.
- Before replacing this lane's current experiment, preserve useful work in backup/test-lab.
- Port reviewed production-ready changes to the active branch; do not turn this into a permanent feature branch.
