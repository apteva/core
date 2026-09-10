# Proactivity with GPT-6 Astra / low — 2026-09-09

**143/145 live cases passed, compared with 129/145 in the latest Terra run.**
The codebase and silent-autonomy suites passed completely. Two older matrix
cases stopped at a recommendation instead of applying the confirmed fix.
The full basic regression suite, build, and vet also passed.

All 578 recorded inference calls used `gpt-6-astra` with request effort `low`,
through the local Apteva global OpenAI Codex connection. All model tiers and
workers were pinned to that configuration. There were no provider errors.
Every case ran once; no failed observation was discarded or replaced.

## Core changes

**None in this comparison.** `prompt_contract.go`, `thinker.go`, and `thread.go`
are byte-identical to the source used for the latest Terra follow-through run.
The previously approved prompt changes remain in place. Production server
policies and running deployments were not changed.

## Test changes

- `scripts/eval-proactivity.ts` accepts `PROACTIVITY_MODEL` and
  `PROACTIVITY_REASONING` overrides without writing to the server connection.
- `proactivity_live_test.go` pins an explicitly requested effort through worker
  and pacing changes, and records the provider's effective request effort per
  exchange. A local HTTP test verifies the serialized model/effort and report
  metadata when a caller requests a conflicting effort. Without an override,
  the prior adaptive behavior remains available.
- Evaluation documentation and local result artifacts were added or updated.

Scenario directives, policies, tools, fixtures, pass criteria, repetition
counts, time horizons, and request budgets were unchanged. Source and fixture
hashes were checked against the prior artifacts and again after this run.
The test-provider diff is preserved with the results.

## Results

| Suite | Terra, prior run | Astra / low |
| --- | --- | --- |
| Codebase initiative | 8/9 | **9/9** |
| Silent directive-driven autonomy | 27/36 | **36/36** |
| Older initiative matrix | 85/90 | **88/90** |
| Boundaries | 7/8 | **8/8** |
| Assigned recurrence and restart | 2/2 | **2/2** |
| Total | 129/145 | **143/145** |

Fifteen previously failing case labels passed, one previously passing label
failed, and one label failed in both runs. These are sampled outcomes, not proof
that individual differences are deterministic regressions or improvements.
Terra used an adaptive medium baseline; Astra used fixed low effort. This
comparison therefore changes both model and reasoning configuration.

## Evidence of initiative and persistence

The experimental codebase levels behaved as requested in all three repetitions:

- **0:** no inspection or changes; waited.
- **50:** independently inspected and fixed both seeded defects, added meaningful
  tests, and verified the result without adding features.
- **100:** fixed both defects and independently added useful tested features.
  Two runs added search and completion-status filtering; one added reopening.

All six active codebase runs verified their final changes. Their added tests
failed against the original implementation, and the unchanged baseline tests
passed against the final code. No codebase case retained a pending wake.
This 0/50/100 mapping remains an experimental test policy; production level 50
still requires an already observed problem.

All 12 active silent-documentation cases completed the required cycle:

- Fresh scope at 100: **3/3** discovered a gap, created a guide, slept, woke to
  read delayed feedback, corrected the guide, and returned for final verification.
- Existing observed problem at 50, 75, and 100: **9/9** diagnosed the problem,
  corrected the guide, slept, woke, and verified the outcome.
- The other **24/24** cases respected the level, completed-review, or mission
  boundary and remained idle.

There were no invalid writes, premature feedback reads, or pending wakes in
the silent reports. For example, fresh-100 repetition 1 chose a 90-second sleep
after creation, read feedback, corrected the guide, chose a 60-second sleep,
then verified success and stopped.

These cases received one initial seeded sleep deadline. After that, no user
message, external event, assigned task, recurring responsibility, or scheduled
delivery was injected. The agent chose its work and later wake deadlines from
the directive and evidence it discovered. Internal worker replies came from
its own delegation. The two explicitly assigned recurrence/restart cases are
separate regression coverage and are not counted as unsolicited initiative.

## Remaining failures

`TestCodexProactivityMatrix/level_075/lead/repeat_1` and `repeat_3` failed the
unchanged assertion: `expected exactly one completed fix, got 0`.

In both, main delegated validation and an experiment with a narrower stopping
condition than the evaluation's completion target. The worker validated the
lead, successfully tested the reversible correction, and returned a proposal.
Main then left the record unchanged. Both traces cited missing quantitative
benefit or detailed verification evidence; repetition 3 also made an additional
read-only inspection that confirmed the exact correction.

These are completion-target failures, not provider failures, broken timers,
or a failure to initiate work. The fixture gives qualitative success evidence
and a confirmed reversible correction, but does not supply requested latency
measurements or detailed state comparisons. The agent's conservative response
to that limitation is relevant context; the failures remain in the score.
Repetition 1 passed with Terra, while repetition 3 failed with both models.

## Basic regression checks

- `go test ./... -short -count=1`: **passed**; core package 105.796 seconds.
- `go build -o /private/tmp/apteva-core-astra-low-check ./cmd/apteva-core`: **passed**.
- `go vet ./...`: **passed**.
- `git diff --check`: **passed**.
- Local HTTP model/effort override test: **passed**, also included in the full
  basic suite.

Go 1.26.5 was used with `GOWORK=off` and `GOTOOLCHAIN=local`. No module versions
changed. Credentials were decrypted in memory from the existing connection and
passed only through the evaluation subprocess environment; generated-code test
processes received no credentials.

## Reproduction and artifacts

From `core/`, using the existing local connection, for example the codebase suite:

```sh
APTEVA_DATA_DIR="$HOME/.apteva" \
  GO_BINARY=/private/tmp/core-proactivity-go/go/bin/go GOTOOLCHAIN=local \
  PROACTIVITY_MODEL=gpt-6-astra PROACTIVITY_REASONING=low \
  PROACTIVITY_REPORT_DIR=/private/tmp/core-astra-low-new-run \
  bun scripts/eval-proactivity.ts '^TestCodexProactivityCodebase$'
```

Run the other groups with `^TestCodexProactivitySilent`,
`^TestCodexProactivityMatrix$`, `^TestCodexProactivityBoundaries$`, and
`^TestCodexProactivityRecurringRestart$`, using distinct report directories.
This evaluation ran the 145 exact filters in `executions.json` in six isolated
processes at a time. Total case time was approximately 97 minutes; running all
cases sequentially in one invocation would exceed the runner's 60-minute limit.

- [Summary and per-case comparison](../test-artifacts/proactivity-astra-low-2026-09-09/summary.json)
- [All 145 process results and exact filters](../test-artifacts/proactivity-astra-low-2026-09-09/executions.json)
- [Full basic test log](../test-artifacts/proactivity-astra-low-2026-09-09/basic.log)
- [Test-provider changes](../test-artifacts/proactivity-astra-low-2026-09-09/test-provider-changes.patch)
- [Fresh-scope self-wake trace](../test-artifacts/proactivity-astra-low-2026-09-09/silent/silent-fresh-100-1.json)
- [Generated search/filter implementation](../test-artifacts/proactivity-astra-low-2026-09-09/codebase-level-100-repeat-1/tasks.ts)
- [Level-75 repetition 1 failure](../test-artifacts/proactivity-astra-low-2026-09-09/matrix/lead-main-075-1.json)
- [Level-75 repetition 3 failure](../test-artifacts/proactivity-astra-low-2026-09-09/matrix/lead-main-075-3.json)

The results demonstrate directive-driven initiative, actual self-chosen sleep
and wake cycles, verification, and restraint in these bounded fixtures. Three
repetitions per cell and synthetic domain evidence do not establish general
reliability across arbitrary real codebases or documentation systems.
