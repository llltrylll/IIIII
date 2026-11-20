# Test-lab backup

This branch preserves the last known-good experimental state from test/agent-lab-a or test/agent-lab-b.

Update only before replacing/resetting a test lane or when the owner explicitly requests a checkpoint. Every checkpoint must record the source test branch, exact source commit SHA, task description, tests run, outcome and limitations.

This is a recovery snapshot, not a third active experiment lane.
