# Codebase initiative results — 2026-09-09

The test demonstrates distinct behavior under the experimental 0/50/100
policies. Maintenance initiative was consistent in this small sample; adding
a useful feature at 100 was not universal.

| Level | Actual behavior in three runs |
| --- | --- |
| 0 | **3/3 waited.** No codebase reads, tests, edits, or future exploration wake. |
| 50 | **3/3 independently inspected and fixed both bugs.** Added regression tests; no features or public API expansion. |
| 100 | **3/3 fixed both bugs; 2/3 also built a useful feature.** One added case-insensitive title search, another added reopening completed tasks. The third stopped after maintenance. |

Each agent woke from an initial sleep with the same standing goal, codebase,
tools, authority, and product observations. Only the selected test policy value
changed. No new user prompt, assigned ticket, recurring job, or external event
was delivered. The feature choice was not specified in the directive.

These are **test-only policies**, as requested. Core's production prompts and
runtime, and the server's current proactivity mapping, were not changed.

## Concrete code changes

The Daybook fixture is a real TypeScript task library executed with Bun. All
six active agents discovered two defects that the original basic tests missed:

- `remove(unknownId)` used `splice(-1, 1)`, deleting the last task. They added
  the missing guard so it returns false without changing the store.
- New IDs came from the array length, causing reuse after deletion. They
  introduced a monotonic counter.

Level 50 preserved the existing public methods in all three runs. At 100:

- Repetition 1 added `search(query)`, handling case-insensitive matching and
  returning independent task snapshots. This addressed the observed difficulty
  of finding a task in a long list.
- Repetition 2 added `reopen(id)`, allowing an accidentally completed task to
  return to unfinished work without recreating it. It handles unknown IDs
  without changing the store.
- Repetition 3 read the usage notes and recognized feature opportunities, but
  its delegated directive softened the goal to “fixes **or** one ... feature.”
  The worker completed maintenance and main accepted that as the final result.
  No new feature was implemented; this is a real missed target, not an
  unfamiliar API that the evaluator failed to recognize.

The independent checks passed for each implemented feature. In every active
run, the agent itself ran a passing suite after its last edit. Its additional
tests also failed when replayed against the original implementation, confirming
that they detect actual missing behavior.

Examples, with complete source and agent-written tests alongside each patch:

- [Level-50 maintenance patch](../test-artifacts/proactivity-codebase-2026-09-09/level-050-repeat-1/changes.patch)
- [Level-100 search patch](../test-artifacts/proactivity-codebase-2026-09-09/level-100-repeat-1/changes.patch)
- [Level-100 reopen patch](../test-artifacts/proactivity-codebase-2026-09-09/level-100-repeat-2/changes.patch)

## Failures retained by the strict test

The code-behavior targets were met in eight of nine observations. The strict
live suite passed six of nine because it also flags unresolved wake state:

| Case | Failure |
| --- | --- |
| Level 50, repetition 2 | Completed and verified both fixes, but retained main's 24-hour wake after the worker finished. |
| Level 100, repetition 1 | Completed and verified both fixes and search, but retained a 24-hour wake. |
| Level 100, repetition 3 | Completed and verified both fixes, but did not implement a feature. |

The two timers were originally chosen while awaiting worker results. Those
results arrived, but the timers remained armed. Their later behavior is outside
the observation horizon; these traces do not prove repeated polling or a broken
timer. They expose incomplete wake cleanup alongside successful code changes.

No live case was discarded, rerun, or selected from multiple attempts. All nine
initial observations are archived, including the failures. There were 83 model
requests and no provider errors. All used `gpt-5.6-terra`, the configured model
from the freshly authenticated global OpenAI Codex connection in local Apteva.
Credentials were used only in memory and subprocess environment, and were not
passed to generated-code test processes or saved in reports.

## Regression validation and scope

- Full basic core suite: passed, `go test ./... -short` (core package 103.323s).
- Build and `go vet ./...`: passed.
- Fixture calibration: confirmed the initial baseline passes, both defects are
  present, features are absent, and the independent checks accept valid fixes
  plus a working feature.
- All nine final fixture baseline suites passed. All six modified applications
  passed their full suites, including agent-written tests.
- Production core prompt/runtime hashes match the previous evaluation. No
  server, deployment, or real application code was changed.

An initial basic-suite run discovered older archived Go test-source snapshots
as an unintended package. Those archival files were renamed from `.go` to
`.go.txt`; the full suite then passed. The original failing log and successful
rerun are retained. New source snapshots use `.go.txt` from the start.

This is a small, bounded codebase scenario, not a guarantee about large real
repositories or arbitrary feature quality. The current production server has
different level semantics, especially at 50. The experimental contract lives
only in the core test fixture; intermediate values can be defined separately
by the server.

- [Scenario design and reproduction](proactivity-codebase-scenario.md)
- [Machine-readable summary](../test-artifacts/proactivity-codebase-2026-09-09/summary.json)
- [Final basic test log](../test-artifacts/proactivity-codebase-2026-09-09/core-codebase-final-basic.log)

All raw model/tool traces, final source, tests, and diffs are in the local,
Git-ignored `test-artifacts/proactivity-codebase-2026-09-09/` directory.
