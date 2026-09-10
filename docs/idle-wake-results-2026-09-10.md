# Directive-driven idle wakes

Core's shared directive now tells agents to choose a finite sleep by default,
including when no initiative policy is stated or nothing is currently actionable.
A purely reactive directive with no unfinished timed responsibility permits an
explicit `pace(clear_wake=true)` decision. The agent chooses its interval, within
Core's existing 500 ms–24 h bounds. Waking grants no additional permission to act,
repeat completed work, or send unsolicited messages.

## Core changes

- [Shared pacing directive](../prompt_contract.go) defines the rule for main,
  workers, and leaders. [Thread prompts](../thread.go) and the
  [pace tool description](../registry.go) use the same rule.
- [Idle guard](../idle_wake.go) supplies a 30-minute fallback when the agent reaches
  idle without a pending deadline or an explicit event-only decision. Existing
  deadlines remain unchanged. Paused/stopped threads are excluded.
- [Thinking loop](../thinker.go) applies the guard after processing a timer and
  before normal idle, and repairs legacy missing decisions during restart.
  New agents still begin immediately. External events interrupt sleeping as before.
- [Pace](../pace.go) records whether the model deliberately chose event-only
  waiting. `wait_for_events` in saved pace state is a receipt of that tool decision,
  not an operator policy setting. It distinguishes explicit waiting from missing
  timing across restart, including worker snapshots. The next input consumes that
  choice so the agent reassesses how to wait; changing the directive invalidates it.
  Failed persistence is reported instead of advertising a durable fallback.

There is no percentage interpretation or directive classification in runtime code.
The model interprets the directive and decides whether the purely reactive
exception applies. Runtime enforces a fallback for missing timing; an explicit
model decision to wait for events remains authoritative.

No server/API policy field, model configuration, realtime lifecycle, route, or
running agent was changed. Existing unrelated workspace edits were preserved.
Following local rebuild authorization, Core was installed at
`/Users/marcoschwartz/Documents/code/core/apteva-core`, the running dev server's
confirmed `CORE_CMD`. New agent starts use build `2026-09-10T08:30:23Z`. Existing
agent processes were not restarted. The prior binary was backed up; see the
[local installation record](../test-artifacts/idle-wake-2026-09-10/local-install.json).

## Tests

- Full `go test -short ./...`: **751 top-level tests passed**, 834 passing test and
  subtest events, 142 existing/opt-in skips, zero failures.
- An additional main/worker directive-change regression, added after the suite
  finished compiling, also passed under race detection. Total distinct passing
  top-level deterministic tests validated: **752**.
- Targeted race checks for idle wakes, pace, restoration, event interruptions,
  pause, and completion passed. Build, `go vet ./...`, and `git diff --check` passed.
- [New deterministic tests](../idle_wake_test.go) run the actual thinking loop with
  Go's simulated clock through multiple 30-minute idle cycles without external
  events. They also cover explicit event-only waiting, later input, omitted/profile
  timing, finite choices, legacy restoration, worker persistence, directive changes,
  protected deadlines, pause/stop, and failed writes. Existing timing tests cover
  event/deadline races and preserving explicitly selected replacement sleeps.
- Three prior pace assertions expected a missing timing decision to strand the
  agent; they now require the finite fallback. Positive-level proactivity
  evaluations now require a remaining reassessment wake. Their checks against
  unauthorized work and premature polling remain; purely reactive tests retain
  their event-only expectations.

## Live directive check

[The opt-in live test](../idle_wake_live_test.go) used the existing local Apteva
OpenAI Codex connection, **gpt-6-astra / low**, and made three model calls in about
20 seconds. Credentials were read in memory and never printed or saved to tests.

| Directive fixture | Agent's own pace decision | Result |
| --- | --- | --- |
| Purely reactive (existing level-0 fixture) | `clear_wake=true` | Passed; explicit event-only choice persisted |
| Conservative initiative (existing level-25 fixture) | `sleep=24h` | Passed; real future deadline persisted |
| No initiative policy stated | `sleep=24h` | Passed; real future deadline persisted |

The numeric fixtures exist only in tests. The live assertion requires an actual
model timing call; passing via runtime fallback alone would fail the test. The
24-hour waits were not observed in real time; repeated wake/rearm behavior was
verified by the deterministic loop tests.

## Evidence

- [Full basic suite](../test-artifacts/idle-wake-2026-09-10/basic.jsonl)
- [Race checks](../test-artifacts/idle-wake-2026-09-10/race.log)
- [Directive-change race check](../test-artifacts/idle-wake-2026-09-10/directive-change-race.log)
- [Live log](../test-artifacts/idle-wake-2026-09-10/live.log)
- [Validation metadata](../test-artifacts/idle-wake-2026-09-10/validation.json)
