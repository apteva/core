# Realtime connection recovery: implementation and validation

Core now keeps an unused realtime thread waiting when its provider connection
ends. New text, audio, or an audio-bridge connection wakes it. Active conversations
still recover automatically. This behavior is shared across realtime providers;
there are no agent IDs, phone-route assumptions, or fixed Google idle timeouts in
the implementation.

**Core changes**

- [Shared recovery state](/Users/marcoschwartz/Documents/code/core/realtime_recovery.go)
  distinguishes a waiting thread from a failed active conversation. It retains
  triggering input, bounds buffered recovery audio to five seconds, and reports
  overflow. Repeated failures back off; after six failures, attempts use a
  one-minute cooldown. Opening a socket alone does not reset that failure state.
- [Realtime thinker](/Users/marcoschwartz/Documents/code/core/realtime_thinker.go)
  preserves tool ownership and completed results across connection loss. It waits
  for unresolved operations before reopening, ignores stale duplicate completions,
  restores verified tool results with conversation history, and prevents automatic
  re-execution of tools from the interrupted turn until a new caller instruction.
  Stop now cancels an in-progress provider open. Planned renewal waits for a safe
  turn/playback boundary, subject to the provider's deadline.
- [Provider capabilities](/Users/marcoschwartz/Documents/code/core/provider_realtime.go)
  provide optional resumption and audio-input-end interfaces, explicit initial
  history intent, and lifecycle events. Existing providers do not need to implement
  unsupported capabilities.
- [Google adapter](/Users/marcoschwartz/Documents/code/core/provider_google_realtime.go)
  handles `GoAway`, records structured close code/reason, and seeds history only
  when restoring an actual conversation. Fresh sessions no longer send an empty
  completed-history turn. Response IDs are unique across connections. Explicit
  input pause/bridge disconnect can send `audioStreamEnd`; dial errors redact the
  credential-bearing URL.

**Google native resumption is intentionally disabled.** Live preflights accepted
resumption handles but sometimes repeated the previous reply or availability
lookup. Consequently, Google's production recovery path opens a fresh connection
and restores Core's verified conversation and tool receipts. The optional shared
interface remains available for adapters with reliable resumption semantics.
Enabling an API option was not treated as proof of continuity.

**Tests, separate from implementation**

The final local validation passed:

| Check | Result |
| --- | --- |
| Complete `go test -short ./...` | 744 top-level tests passed; 818 passing test/subtest events; zero failures |
| Short-mode / opt-in skips | 141 top-level tests skipped by their existing gates |
| Targeted realtime lifecycle race checks | Passed |
| Core executable build | Passed |
| `go vet ./...` | Passed |
| `git diff --check` | Passed |

Ten new deterministic tests cover idle wakeup by text/audio/bridge, first-input
retention, optional resumption and rejection fallback, disconnects before/during/
after a tool operation, duplicate-action prevention, later authorized tool use,
planned expiry, close metadata, history setup, retry cooldown, and cancellation
during a provider open. Existing realtime tests also pass. Their expectation
updates make the renewal test explicitly represent an active call, allow the new
backoff interval, and expect fresh Google sessions to omit empty history.

- [Shared recovery tests](/Users/marcoschwartz/Documents/code/core/realtime_recovery_test.go)
- [Google protocol tests](/Users/marcoschwartz/Documents/code/core/provider_google_realtime_recovery_test.go)
- [Live audio recovery scenario](/Users/marcoschwartz/Documents/code/core/realtime_recovery_live_test.go)

The final live soak passed in 720.3 seconds: 16 caller transcripts, 17 assistant
turns, one greeting, and exactly two external tool calls (one availability lookup
and one booking). It verified spoken context retention after planned renewal,
booking-result retention after a forced disconnect, PCM barge-in, and an idle
thread waiting over six minutes before waking and answering “42”.

Google also sent a real `GoAway` after about nine minutes on the third connection,
with 50 seconds remaining. Core renewed the connection and continued the
conversation with the booking context intact. Every replacement connection used
verified history restoration; none used native Google resumption.

The live scenario uses `gemini-3.1-flash-live-preview`, `Kore`, synthetic spoken
PCM and continuous silence, and a local fake MCP booking service. The two early
renewals are deliberately forced and labeled in the transcript. This is not a
microphone-device or PSTN/Telnyx test.

**Evidence and scope**

- [Final local test results](../test-artifacts/google-realtime-fix-2026-09-10/basic-tests-safe.jsonl)
- [Race checks](../test-artifacts/google-realtime-fix-2026-09-10/race-tests-safe.log)
- [Live log](../test-artifacts/google-realtime-fix-2026-09-10/full-safe/google-live.log)
- [Live recovery summary](../test-artifacts/google-realtime-fix-2026-09-10/full-safe/recovery-summary.json)
- [Validation metadata](../test-artifacts/google-realtime-fix-2026-09-10/validation.json)
- [Validated source hashes](../test-artifacts/google-realtime-fix-2026-09-10/validated-source-hashes.json)

These changes handle provider connection loss within a running Core thread. They
do not establish the cause of the earlier computer crash or guarantee external
operation idempotency across a process/machine crash. Dashboard mute-state
propagation and phone-route configuration were not changed. No agent or server
was restarted; the verified build is `/private/tmp/core-realtime-fix-build`.
Existing proactivity edits and unrelated temporary files were preserved.
