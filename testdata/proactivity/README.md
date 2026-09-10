These five policy fixtures were generated on 2026-09-09 by calling
`agentProactivityInstructions` in `server/agent_proactivity.go` for
0, 25, 50, 75, and 100. Source SHA-256: `20fc21c6dabf87448b57d9bb05d5b4af79ad5da7ee35540ae82c6cc1c5fbd313`.

They are immutable inputs to a core-only behavioral evaluation, not a numeric
policy implementation. Regenerate them from server when its policy changes;
do not maintain a separate mapping in core. Core receives the text only.

All levels share the same simulated world, tools, permissions, and model.
The matrix varies only the policy fixture and the situation. Both the old and
updated prompt builds use these same fixtures.
