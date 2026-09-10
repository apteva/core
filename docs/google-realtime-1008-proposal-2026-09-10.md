# Google realtime 1008: findings and proposed fix

The new evidence strongly points to a provider input-idle timeout. Five isolated
cases with no ongoing input closed after approximately 150–159 seconds. Periodic
text input and continuous silent PCM both kept the connection open until the
210-second test deadline. This reproduces the readiness session's symptom without
the phone route or audio bridge. Google's internal reason is still not exposed by
the generic “The operation was aborted” close message.

This is a proposal. No product source, running agent, configuration, route, or
deployment was changed. The probes used Go overlays around the production Google
adapter and the project's existing Gemini credential. They used synthetic input,
no external tools, and no phone calls. Credential and resumption-handle values are
absent from the saved results.

**Measurements**

All cases used `google-realtime`, `gemini-3.1-flash-live-preview`, and `Kore`.

| Isolated case | Observed result | Additional evidence |
| --- | --- | --- |
| Current setup, empty history, no input | 1008 at 158.9 s | Two unsolicited assistant turns |
| Same, explicitly enable session resumption | 1008 at 158.0 s | No resumption updates; two unsolicited turns |
| Current setup, text every 40 seconds | Open until deadline, 209.9 s | Six inputs, five completed outputs; valid resumption updates |
| Omit history configuration and initial empty history message; no input | 1008 at 150.2 s | No assistant output |
| Current setup plus `audioStreamEnd: true`; no caller input | 1008 at 157.9 s | Unsolicited output at 54.6 s and 107.9 s |
| One real text turn, then no further input | 1008 at 151.9 s | One completed reply; valid resumption handle available |
| Current setup, continuous synthetic silent PCM | Open until deadline, 209.9 s | 2,097 audio chunks, no assistant output, 226 resumption updates |

None of these seven cases received `GoAway`. The two surviving cases were closed
by the test deadline, not by Google. The silent PCM case models a stream that is
still sending audio; it is not a speech-recognition or phone-call test.

The probe's Go `PASS` means it completed its observation, not that the observed
connection was healthy. Each variant ran once. There is no measured idle policy
guarantee for other models, accounts, or longer calls.

Google's [session-management guide](https://ai.google.dev/gemini-api/docs/live-api/session-management)
documents connection expiration, `GoAway`, and session resumption. Its approximate
10-minute connection lifetime does not explain these 150-second idle closes.
The [protocol reference](https://ai.google.dev/api/live) says resuming from an old
handle while the session is not resumable can lose data, including during model
generation or function calls. It also specifies that initial history should not
trigger generation. Our empty-history/no-input observations warrant a regression
test against that intended behavior.

**Proposed core changes**

1. **Separate the logical realtime thread from an idle provider connection.**
   Keep the thread, transcript, and pending input alive when an unused provider
   connection closes. If there is no connected audio bridge and no pending input,
   output, or tool work, wait for demand instead of immediately reopening forever.
   Reopen on bridge connection or actual text/audio input, retain that triggering
   input, and deliver it once the session is ready. A text-only realtime thread
   must continue to work. For an active call, preserve automatic recovery and
   distinguish quiet audio arriving normally from a missing stream.

2. **Use native Google resumption when a safe checkpoint is available.**
   Enable and retain `SessionResumptionUpdate` state in memory, scoped to the
   logical thread and compatible model/configuration. On a reconnect, use a valid
   handle and avoid reseeding the same transcript or replaying the greeting.
   Handle `GoAway` as a planned renewal when it actually arrives; these observed
   unannounced 1008 closes remain a separate path. If no safe handle exists or it
   is rejected, open a fresh session and restore bounded context explicitly.
   Expose this as an optional provider capability so other realtime adapters keep
   their existing behavior. Simply enabling resumption did not prevent the idle
   close in our probe; this change is for continuity across connections.

3. **Make renewal safe around tool execution and playback.**
   Defer planned renewal until the turn and tool batch reach a safe checkpoint,
   within the provider's deadline. Keep operation outcomes independent of the old
   socket so a late result is not lost. Do not automatically rerun an external
   mutation whose completion is uncertain. Reconcile it using its recorded result
   or the tool's idempotency/status support before allowing another execution.
   Preserve acknowledged playback and avoid replaying already-heard audio. Current
   session replacement clears tool-batch ownership, so this needs explicit work.

4. **Avoid unnecessary empty-history initialization.**
   A brand-new session with no history should wait for actual input or the one
   explicit greeting requested after bridge connection. Configure history seeding
   when restoring actual history; do not send the empty completed-history message
   unconditionally. In the no-history probe this removed unsolicited greetings,
   although it did not prevent the idle close. Verify real input and nonempty
   history restoration before adopting the change.

5. **Record enough lifecycle state to diagnose the next failure.**
   Record close code/reason, session age, time since last input/output, audio-byte
   counts, bridge state, whether `GoAway` arrived, and whether recovery resumed or
   restored a fresh session. Never log credentials or handle values. Apply bounded
   backoff to repeated unexpected established-session failures, not just failed
   opens. Reset the failure budget on demonstrated recovery or a healthy interval;
   a successful socket handshake alone is insufficient. A healthy idle thread
   should wait without spending a reconnect budget.

Core can also support `audioStreamEnd` on explicit input pause/end and bridge
disconnect. It is protocol hygiene, not a proven prevention for this 1008:
the explicit-end control still closed. The dashboard currently stops sending PCM
when muted without sending a mute/end control. Full mute-state propagation would
therefore need a small companion dashboard/bridge change; it cannot be attributed
to Core alone. Preserve actual microphone silence while capture is active. The
synthetic-silence probe is evidence, not a recommendation to send fake user input
to keep an unused session alive.

**Tests, separate from core implementation**

- Local adapter tests: `GoAway` parsing/deadline, valid and invalidated resumption
  handles, rejected-handle fallback, structured 1008 errors, fresh empty history,
  nonempty restoration, and explicit audio-stream end/restart.
- Local thinker tests: idle close waits without reopening; text-only input and
  bridge connection wake it; the first queued input survives reopening; active
  calls reconnect; greetings and heard audio do not replay; stop/cancellation works
  during every state. Use a fake clock/provider for retry and idle transitions.
- Tool-boundary tests: force a disconnect before a tool starts, during execution,
  after its external action succeeds, and while its result is being delivered.
  Assert one external mutation, retained outcomes, and no continuation on the
  obsolete socket. Test stale checkpoint and configuration-change cases too.
- Live Google acceptance: repeat the idle cases beyond two observed timeout
  windows, then supply a new input and verify recovery. Run a 12-minute PCM speech
  conversation with silent intervals, interruptions, and a local fake booking
  tool. Exercise both native resumption and fresh-history fallback. If no natural
  renewal occurs, inject one separately and label it as forced. Measure the audio
  gap and verify caller speech, context, greeting count, and tool execution count
  across the boundary. A 22-second text-input smoke is insufficient for this gate.
- Regression gate after implementation: run the complete short Core suite, build,
  and vet; run the existing realtime provider, audio bridge, shutdown, greeting,
  tool batching, and configuration tests, with race checks on affected lifecycle
  tests. If the optional dashboard mute change is included, run its existing tests
  and build too. Phone validation remains a separate gate once the route is ready.

**Source and evidence**

- [Google setup and unconditional history mode](/Users/marcoschwartz/Documents/code/core/provider_google_realtime.go:174)
- [Parsed lifecycle fields, currently unhandled](/Users/marcoschwartz/Documents/code/core/provider_google_realtime.go:256)
- [Current transcript restoration](/Users/marcoschwartz/Documents/code/core/provider_google_realtime.go:780)
- [Session replacement and tool ownership](/Users/marcoschwartz/Documents/code/core/realtime_thinker.go:244)
- [Current reopen/restore path](/Users/marcoschwartz/Documents/code/core/realtime_thinker.go:354)
- [Current renewal loop](/Users/marcoschwartz/Documents/code/core/realtime_thinker.go:933)
- [Dashboard capture/mute behavior](/Users/marcoschwartz/Documents/code/dashboard/src/realtime/audio.ts:64)
- [Machine-readable findings](../test-artifacts/google-realtime-1008-2026-09-10/findings.json)
- [First three live observations](../test-artifacts/google-realtime-1008-2026-09-10/google-live.log)
- [Four recovered control observations](../test-artifacts/google-realtime-1008-2026-09-10/post-crash-controls/google-live.log)

Two intermediate follow-up builds did not complete; one recorded “no space left
on device” before the user-reported computer crash. This does not establish the
cause of the crash. After restart, approximately 30 GB was free. The temporary Go
toolchain was restored and the four controls completed through one build limited
to two CPU workers. The Google adapter's SHA-256 still matches its pre-probe hash.
