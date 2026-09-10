# Core follow-through revision — 2026-09-09

**The basic core checks pass. Timer cleanup and verification improved in the
targeted scenarios, but overall live reliability did not improve in this run.**
All 145 existing proactivity observations were rerun with unchanged test source,
policies, fixtures, model selection, and assertions. None of the failed cases
was discarded or replaced by a retry.

## Core changes

Three shared prompt contracts were added in `prompt_contract.go` and applied
through `thinker.go` and `thread.go`:

1. Worker replies and tool results wake their owner automatically. Do not set
   fallback timers merely to await them; clear obsolete wakes after results
   arrive, while preserving independently justified deadlines.
2. Preserve the original objective, required outcomes, conditions, and limits
   through delegation. A child completing a subset does not discharge the
   parent's remaining objective.
3. Retain ownership of useful verification. For pull-only results, choose a
   wake from stated availability and current time, then clear obsolete wakes
   when verification is complete.

The generic “two seconds when working,” “one hour for deep idle,” and mandatory
pacing examples were removed or clarified. No runtime timer implementation,
permission enforcement, server policy, scenario directive, or numeric level
mapping was changed. The prior approved prompt edits remain intact.

[Exact changes from the start of this task](../test-artifacts/proactivity-followthrough-2026-09-09/core-changes.patch)

## Basic regression checks

- `go test ./... -short`: **passed**, including all core packages and the
  existing pacing, persistence, ownership, directive, and shutdown tests. Core
  package: 104.518 seconds.
- `go build ./cmd/apteva-core`: **passed**, binary written outside the repo.
- `go vet ./...`: **passed**.
- `git diff --check`: **passed**.
- Test-source and fixture SHA-256 hashes: **unchanged during the experiment**.

The tests used the same working Go 1.26.5 installation with `GOWORK=off` and
`GOTOOLCHAIN=local`. No module or workspace versions changed.

## Live comparison

These are strict per-case passes, including each suite's additional assertions:

| Evaluation | Before | After |
| --- | --- | --- |
| Codebase initiative | 6/9 | **8/9** |
| Silent directive-driven autonomy | 27/36 | **27/36** |
| Older initiative matrix | 86/90 | **85/90** |
| Boundary cases | 8/8 | **7/8** |
| Assigned recurrence and restart | 2/2 | **2/2** |

The total is **129/145 passing, 16 failing**, the same aggregate count as before
with a different mix of outcomes. This is not evidence that all prior behavior
is unaffected. The older matrix's expected domain outcomes decreased from
89/90 to 86/90; its strict count also checks delegation and policy inheritance.
Three repetitions per cell do not establish whether every change in outcome
is caused by the prompt revision rather than model variation.

All requests used the local Apteva global OpenAI Codex connection and its
configured `gpt-5.6-terra` model. The main evaluation recorded 582 requests.
Credentials stayed in memory/subprocess environment and were not included in
artifacts or passed to generated-code test processes.

## What improved

**No codebase run retained a pending timer**, versus two previously. Level 0
stayed idle in all three cases. Level 50 completed verified bug-fixing work in
all three cases without feature expansion; two fixed both seeded defects and
one fixed a single defect, which meets the bounded maintenance target. Two
level-100 cases fixed both defects and independently added a tested reopen
feature. They also ended without a pending wake.

In the fresh documentation scenario at 100, **all three runs created, slept,
woke, corrected, returned for final verification, and stopped**. Previously,
only one of three reached final verification. One current run made a rejected
write before recovering, so only two pass every strict assertion. The third
still demonstrates successful recovery and persistence.

Across all active silent cases, every run that stored a valid correction went
on to inspect final feedback: **4/4**, versus **3/7** previously. There were no
premature feedback reads or pending wakes in the new silent reports. However,
fewer cases reached a valid correction, so this conditional result must not be
presented as an overall success rate.

All 24 silent restraint controls passed again. Both cautious boundary cases
preserved approval behavior, the disproved lead stopped, and the delayed
verification boundary passed. Both recurring owners retained their valid
deadlines through an unrelated interruption and shutdown/reload, then executed
again without new user input.

## Failures and regressions retained

**Codebase: one failure.** At 100, repetition 3 initially granted malformed
names such as `codebase.codebase_read_file`. Main corrected the grant, and the
trace confirms the proper tools were then present in worker requests. The
worker nevertheless kept using discovery and claiming that no tools were
available. Replacing the worker under the same ID did not recover the work.
No codebase operations completed. One request was cancelled during the
agent-initiated tool update; this is not evidence of a provider outage.

**Silent autonomy: nine strict failures.** The exact cases are:

- Fresh-100 repetition 2: rejected write, followed by successful correction,
  self-wakes, and final verification.
- Observed-50 repetitions 1–2: did not obtain the needed evidence or store a
  valid correction. Repetition 3 completed the full cycle.
- Observed-75 repetitions 1–3 and observed-100 repetitions 1–3: attempted
  unsupported corrections or claimed missing evidence without successfully
  consulting the available diagnostic feedback. No valid correction was stored.

The observed-100 cell declined from 2/3 strict passes to 0/3. Observed-75 stayed
at 0/3, but previously two runs at least corrected the guide; none did this
time. Those are material negative results alongside the improved fresh-scope
follow-through. The simulated write validator and sparse documentation remain
the same narrow fixture, rather than a general test of diagnosis quality.

**Older matrix: five failures.** Lead-75 repetitions 2–3, lead-100 repetition 2,
and discovery-100 repetition 2 did not complete a fix. Some successful
experiments were followed by unnecessary deferral or approval requests despite
the scenario's autonomous authority. Discovery-100 repetition 1 completed the
fix but omitted the proactivity value from a delegated directive.

**Boundary: one failure.** The level-100 leader discovered an opportunity but
did not complete its fix. Recurrence/restart had no failures.

The prompt changes therefore address the observed waiting and verification
weaknesses, but do not resolve tool selection, capability recovery, preserved
authorization, or every incomplete delegation. The full live suite remains
failing; the basic suite passing should not be read as a claim of complete
behavioral regression safety.

## Isolated refinement that was not adopted

While the main run continued, a detached temporary worktree tested one extra
sentence directing the model to use already available outcome evidence before
writing a correction. The three observed-100 probes all failed to complete the
scenario. That sentence was **not applied to the main checkout**. Its source,
logs, and all three reports are retained separately and excluded from the
145-case comparison. No tests or policy fixtures were altered to make results
pass.

## Evidence

- [Machine-readable summary and complete failure list](../test-artifacts/proactivity-followthrough-2026-09-09/summary.json)
- [All 145 process outcomes and exact test filters](../test-artifacts/proactivity-followthrough-2026-09-09/executions.json)
- [Full basic test log](../test-artifacts/proactivity-followthrough-2026-09-09/core-followthrough-basic.log)
- [Successful fresh-scope sleep/verify trace](../test-artifacts/proactivity-followthrough-2026-09-09/silent/silent-fresh-100-1.json)
- [Successful level-100 generated code](../test-artifacts/proactivity-followthrough-2026-09-09/codebase-level-100-repeat-1/tasks.ts)
- [Codebase capability-recovery failure](../test-artifacts/proactivity-followthrough-2026-09-09/codebase/codebase-100-3.json)

Artifacts are local and Git-ignored. Source snapshots use `.go.txt` so Go does
not discover archival copies as packages. The core edits are local; no running
server or deployment was updated.
