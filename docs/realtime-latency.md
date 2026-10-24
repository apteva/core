# Realtime settings and latency

Realtime threads accept additive, provider-neutral output configuration through
`POST /threads/<id>`, `SpawnOpts.RealtimeOutput`, and persistent thread state:

```json
{
  "realtime": true,
  "turn_detection": {
    "profile": "telephony",
    "silence_duration_ms": 500
  },
  "realtime_output": {
    "tool_mode": "async",
    "speech_guard": {
      "mode": "transcript_prefix",
      "prefix_bytes": 32,
      "max_buffered_audio_ms": 5000
    }
  }
}
```

Omitting these options retains existing behavior. Telephony still defaults to
750 ms of silence; 500 ms is an explicit override within Google's recommended
500–800 ms range. A shorter threshold can split callers' pauses, so changing the
default requires representative conversation measurements.

`tool_mode` accepts `default`, `blocking`, or `async`. Gemini 3.8 standard supports
either tool mode, but its turns always finish on `turnComplete`. Extended
Thinking requires async tools and finishes the interaction on `interactionStatus:
IDLE`; `turnComplete` only ends one utterance. Standard async function responses
explicitly request `WHEN_IDLE` scheduling as a sibling of `id`, `name`, and
`response` on `FunctionResponse`, matching Google's SDK wire shape. This lets
each ready background result prompt speech without interrupting current speech.
Extended Thinking retains model-managed delivery and omits the scheduling
override. Legacy Gemini retains blocking
tools. Core keeps reporting outstanding tool work and does not settle the caller
event merely because an intermediate spoken turn completed. Unsupported explicit tool/guard modes fail setup rather than being
silently ignored. OpenAI/xAI retain their current transcript checks and stream
audio without Google's pre-audio prefix guard.

No guard deadline or mandatory wait is introduced. Prefix bytes, PCM capacity,
and measured elapsed waiting time remain separate. The default prefix is
32 UTF-8 bytes or sentence punctuation, with a five-second PCM capacity. Google
transcription is not chunk-aligned with audio. Core queues checked transcript
events before released audio and keeps the existing leakage recovery policy.

## Telemetry

`realtime.session_opened` includes the session generation, provider,
`connection_to_open_ms`, requested settings, and a sanitized adapter report of
applied settings. Reports state their evidence: `accepted_setup` means Google
accepted the setup; `sent_session_update` means the compatible adapter sent it.
Neither claims remote behavior beyond the available acknowledgement. Ignored
fields are explicitly identified, including standard Gemini 3.8 reasoning and
unsupported VAD sensitivities. Prompts, schemas, transcripts and credentials
are excluded from settings/timing payloads.

`realtime.session_ready` measures connection start to the local receipt of the
provider's setup acknowledgement. Compatible adapters wait for `session.updated`
to report readiness, without delaying their existing `Open` behavior.

One `realtime.utterance_timing` summary is emitted after generation completion
and outstanding Core output frames are written. Blocked, interrupted, replaced
and closed sessions also have terminal summaries. Fields include:

| Field | Local timing boundary |
| --- | --- |
| `speech_end_to_provider_audio_ms` | Available speech-end event receipt to first raw provider audio |
| `provider_audio_to_guard_release_ms` | First provider audio receipt, before guard filtering, to release |
| `guard_release_to_output_enqueue_ms` | Guard release to Core output enqueue |
| `output_enqueue_to_bridge_write_ms` | Core enqueue to successful metadata/audio websocket write |
| `provider_audio_to_bridge_write_ms` | First raw provider audio receipt to successful bridge write |

Durations use local monotonic clocks. Missing boundaries serialize as `null`,
not zero. Gemini's automatic VAD does not expose a reliable normalized speech-end
event; its speech-end metric is unavailable unless the client reports one.
Compatible providers label their native VAD event as
`provider_vad_event_received`. Clients may send `{"type":"input.speech_stopped"}`;
its local receipt is labelled `client_reported_signal_received` and does not
change provider turn detection. Neither source reconstructs a remote clock or
subtracts a configured silence threshold.

Each summary correlates generation, response and audio item, and reports peak
buffered audio duration, maximum Core output queue depth, observed dropped audio
bytes, guard outcome and completion status. State is bounded per session. Frames
retain their original tracker across reconnects. Instrumentation adds no model
calls or per-chunk telemetry events.

The existing `realtime.first_audio` event remains compatible: it records the
first successful Core output enqueue, not websocket delivery or playback.
Optional `playback.progress` produces one `realtime.playback_timing` receipt per
item, labelled `client_reported_progress_received`. This may follow the utterance
summary; it is not carrier playback or an independently measured playback time.

## Verification

The deterministic tests cover audio-first and fragmented transcripts,
same-envelope leakage, PCM overflow, interruption, failed bridge writes,
generation isolation, unavailable measurements, settings persistence,
nonblocking audio observations and the separate Gemini tool/lifecycle modes.
Existing tool-error and interruption/recovery tests remain applicable.

Paid local tests require `GOOGLE_API_KEY` at runtime:

```sh
RUN_GOOGLE_REALTIME_TIMING_LIVE=1 go test -v -run '^TestGoogleRealtimeTimingLive$' .
RUN_GOOGLE_38_LIVE_SMOKE=1 go test -v -run '^TestGoogle38Live' .
```

The timing test synthesizes one caller recording with Gemini, streams it in
20 ms frames through Core's actual local audio websocket bridge, and compares
750 ms and 500 ms with two samples each. The second sample includes an explicitly
labelled client report of the finite recording's end. It verifies accepted
settings, transcripts, PCM delivery, stage measurements and zero dropped audio.
Small live samples validate the path and reveal variation; they do not prove
a statistically significant latency improvement. The tool smoke verifies
delayed parallel results in standard blocking/async and Extended Thinking modes,
plus real Core threads calling a local MCP fixture.

### Live evaluation finding (2026-10-09)

Local bridge timing samples at both 750 ms and 500 ms passed, as did Core/MCP
calls in standard blocking, standard async and Extended Thinking modes. The
full short suite, realtime race checks and Core build passed.

Repeated delayed parallel-tool evaluation exposed a model-quality limitation
in standard async mode. Before the SDK wire-format correction below, two of three
runs completed with both actual results, but one repeated the slow tool call.
The successful runs also initially guessed a marker before correcting it from
the delayed result. The strict evaluation is retained and these failures are
not suppressed. Standard async remains an explicit opt-in; it is not suitable
as a new default on this evidence. Use blocking mode or Extended Thinking for
workflows that need to wait for slow, coordinated tool results.

The timing measurements also isolated approximately 0.9 seconds of guard wait
in a short reply without early sentence punctuation. Other replies cleared the
guard in fractions of a millisecond. This variability is now visible separately
from provider audio latency. The small samples do not justify weakening the
guard or automatically changing the telephony silence default.

### Follow-up: cancellation, replay and strict async evaluation

Core now consumes a provider-neutral `tool_call_cancelled` event. Gemini maps
its cancellation IDs to that event and removes any cancelled responses still
waiting in its blocking batch. The individual execution context is cancelled,
including calls waiting for an execution slot. Other calls and the owner remain
active. Cancellation requests cannot undo side effects already completed; late
outcomes are archived once without sending a stale provider result. A generic
cancellation error also handles the race where a bus result reaches Core before
the provider cancellation event, avoiding a spurious connection recovery.

Completed/cancelled call IDs are remembered in a bounded ledger (256 terminal
records) scoped to their provider session. Pending calls are not evicted. A replay
of the same ID does not re-execute or start another response. New IDs remain
distinct operations, including calls with identical arguments. No general
argument-based deduplication or result cache is introduced.

Async setup adds a shared instruction contract: progress speech is allowed;
result-dependent answers must use actual returned results; pending calls must
not be retried merely because they are slow. Blocking instructions remain
unchanged. These instructions do not delay, mute, or semantically filter audio.

The probe evaluation now checks intermediate transcripts against delivered
results and fails on a wrong claim even if the model later corrects it. With the
shared async instructions applied, three blocking and three Extended Thinking
comparison runs passed. Standard async still produced premature result claims
or repeated the slow call. A final three-run standard-async evaluation recorded
the same incorrect `ALPHA and BETA` answer followed by the correct
`ALPHA and BRAVO` in all three runs; all three failed the stricter evaluation.
The lifecycle fixes are useful independently, but the instructions did not
solve this Gemini behavior. Keep standard async opt-in; blocking or Extended
Thinking remains the supported recommendation for coordinated slow tools.

### SDK wire-format recheck

Rechecking both Google SDKs found a discrepancy with the Live tool guide's
example: the guide nests `scheduling` in `response`, but the SDK defines and
serializes it as a field on `FunctionResponse` itself. Core now follows the SDK,
keeping scheduling metadata outside tool data. Tests assert the actual JSON
structure for success and error results; blocking and Extended Thinking retain
their existing omission of the scheduling override.

The earlier nested field was therefore insufficient evidence of a correctly
applied protocol policy. The SDK also documents `WHEN_IDLE` as the default, so
correcting the placement alone does not establish the cause of premature speech.

After correction, the three-run comparison again passed in standard blocking
and Extended Thinking. A subsequent strict comparison failed all three standard
async runs with the full tool surface, and all three minimal controls using just
two tools and the guide's finite `clientContent` input flow. The controls still
reissued the slow call or spoke guessed markers before correcting them. This
reduces the likelihood that the full Core manifest or live text input caused the
failure; it does not prove the remaining issue is exclusively provider-side.
The evaluation also now rejects wrong marker pairs such as `ALPHA and OMEGA`,
which the earlier BETA-only check missed.

Core's Google realtime defaults, the server fallback, and the integration
catalog now select `gemini-3.8-live`. Explicit model selections retain priority,
including existing agents pinned to `gemini-3.1-flash-live-preview` or Extended
Thinking. The tests explicitly cover standard 3.8 and Extended Thinking.

References: [Google Live capabilities](https://ai.google.dev/gemini-api/docs/live-api/capabilities),
[thinking lifecycle](https://ai.google.dev/gemini-api/docs/live-api/thinking),
[tool response scheduling](https://ai.google.dev/gemini-api/docs/live-api/tools),
[SDK FunctionResponse fields](https://googleapis.github.io/js-genai/release_docs/classes/types.FunctionResponse.html),
[best practices](https://ai.google.dev/gemini-api/docs/live-api/best-practices).

## Input admission and provider recovery

Core drains external audio independently of provider connection work and
transcript persistence. A replacement connection opens, resumes and restores in
an asynchronous attempt. Only the runtime loop may install it, after checking
cancellation and the expected session generation. Superseded or late sessions
are closed. Each connection attempt has a 30-second budget; retries retain the
existing exponential backoff and circuit cooldown. Initial thread creation still
opens synchronously so invalid configuration fails before exposing the thread.

`RealtimeSession.SendAudio` acknowledges **local admission**, not remote receipt.
The shared `RealtimeAdapterError` classes are `pressure`, `closed`, `transport`
and `protocol`. Pressure and closed errors guarantee that the frame was not
accepted. Temporary pressure retries the same frame at a 10 ms interval without
reconnecting; one second of sustained pressure initiates session recovery. An
ambiguous failed frame is reported as `delivery_uncertain` and is not replayed.
Previously admitted frames are never replayed solely because a session failed.

The default input policy allows five seconds of buffered audio and a maximum
frame age of five seconds, using the declared encoding and sample rate. Overflow
rejects excess incoming audio; expired audio is discarded before delivery. Both
produce `realtime.audio_drop` events with bytes, duration, generation, reason and
`discontinuity: true`. Remaining input is reported when the thread stops. Receipt
accounting happens once, rather than again during recovery delivery.

Input is split into at most 20 ms frames and drained incrementally. Continuous
conversation models receive buffered input at its recorded sample rate instead
of a reconnect burst. Turn-based providers drain bounded batches and yield to
lifecycle events between batches. Providers can supply `RealtimeInputPolicy`
overrides for buffering, freshness, frame size, pressure tolerance and pacing.

`realtime.error` retains the underlying error and classification, operation,
generation and queued input duration. `realtime.reconnect` distinguishes total
recovery duration from connection-attempt duration. Audio bytes and credentials
are not included in these diagnostics.

Clients negotiating `audio_protocol=1` receive `provider_state` in the
`X-Apteva-Audio-Session` handshake and `session.state` controls containing `state`
(`recovering`, `ready`, `failed`) and `session_generation`. These describe the
provider session, independently of the external socket. State updates coalesce
to the latest value on a separate lane, so a full playback-control queue cannot
hide readiness. `failed` describes the last attempt; Core may retry after
backoff. These notifications do not change the PCM framing or replace playback
acknowledgements. Legacy clients receive no additional control types.

Deterministic coverage lives in `realtime_input_test.go` and
`realtime_recovery_test.go`. The paid recovery acceptance test runs with:

```sh
RUN_OPENAI_LIVE_RECOVERY=1 GOWORK=off go test -race \
  -run '^TestOpenAILiveRecoveryWithContinuousIntakeLive$' -count=1 -v .
```

Set `OPENAI_API_KEY` in the runtime environment. This test deliberately injects
a local send failure and a slow reconnect, while both provider sessions and the
French audio/transcription use the real API. It checks continued external intake
and a meaningful spoken reply after recovery; it does not establish the cause
of an earlier production send failure.

## Local validation — 24 October 2026

Build: `2026-10-24-realtime-latency-123813Z`.

- Full Core short suite: passed (120.3 s); realtime/managed-work race suite:
  passed (13.1 s).
- Real `gpt-6.1-sol` verified-result streaming: first eligible sentence reached
  the voice context adapter 489 ms before backend completion. This measures
  context delivery, not the caller hearing that sentence.
- Real `gpt-live-1` / Brise interruption and slow-reconnect tests: passed,
  including recognition of the French caller and a meaningful reply.
- Matched two-turn deterministic replay: 14 turns across direct PCM, Twilio
  inbound/outbound, PCMA inbound/outbound and G.722 inbound/outbound. All passed.
  A 400 ms sender stall recovered in 2.015–2.297 s through the real local
  Core/Server/Telephony paths. Queue age settled near 40 ms; PCM ordering is
  separately asserted byte-for-byte in the deterministic Core tests.
- Matched two-turn real voice replay: eight turns across direct PCM, Twilio
  inbound/outbound and PCM round-tripped through the phone codec. All passed.
  Expected-answer transcript onset was 909–1346 ms after caller speech ended;
  first post-question audible output was 1048–1554 ms. Backchannels and earlier
  output are reported separately. These replay questions ask the model to name
  the day heard in the identical French recording; backend streaming is tested
  separately with verified pricing evidence.

The original injected-stall characterization asserted a 150 ms *peak* ceiling
and still failed, as expected, during its deliberate 400 ms stall. The reusable
Core replay helper preserves that peak in the report and additionally checks
that the final two flow windows recovered below 150 ms. Healthy runs retain
the original peak ceiling. This distinguishes recovery from the unavoidable
peak without deleting the recorded evidence.

Logs/replay artifacts for this local run:

- `/tmp/core-latency-release-suite.log`
- `/tmp/core-latency-release-race.log`
- `/tmp/core-latency-release-real.log`
- `/tmp/core-latency-release-transport-stall/`
- `/tmp/core-latency-release-transport-live/`

These small local samples do not establish production latency percentiles or
real carrier/handset playback. Model-generated silent PCM remains untouched.
Incremental results intentionally wait for final delivery when they do not meet
the conservative complete-sentence/evidence gate.

## Request, audio and playback timing (October 24 follow-up)

Core emits separate, provider-neutral observations rather than treating a
caption or successful backend operation as audible completion:

- `realtime.request_timing`: `requested`, `started`, `result_available`,
  `submitted`, `context_acknowledged`, and `backend_completed` milestones.
  Includes session generation, request ID, revision, delegation ID and stage
  durations. `result_available` is when Core's voice runtime receives the
  worker result, not when an external service committed an operation. Native
  voice tool calls use `tool/<call_id>` as their request identity.
- `realtime.utterance_timing`: existing guard, output queue, renderer and bridge
  durations, plus first-audio-to-non-silent-audio and output association fields.
  Non-silent means decoded audio exceeds an amplitude threshold; it does not
  identify meaningful speech. All PCM, including silence, remains unchanged.
- `realtime.caption_timing`: first caption receipt for a playback item, including
  captions arriving after its audio summary. Captions never establish playback.
- `realtime.playback_timing`: first positive renderer progress report and, when
  explicitly associated, the first report covering the answer's non-silent
  frame. Includes `evidence` (`carrier_mark`, `estimated`, `rtp_transmitted`, or
  `legacy`), playback/item ID and generation. The timestamp is **Core receipt of
  the report**, not the handset's actual playback time. Reports are bounded by
  bytes written; stale generations, unwritten audio and cleared items cannot
  produce successful playback timings. Older playback items remain trackable
  after a newer item starts.

Provider acknowledgements measure context injection, not speech. OpenAI Live
correlates the first append acknowledgement using `ClientEventID`; split appends
can take longer to inject in full. Other providers leave acknowledgement latency
unavailable unless their adapter supplies it. Successful `AppendContext` means
local submission/admission, not network delivery or provider consumption.

`OutputWork` and `OutputPurpose` are optional **adapter-supplied evidence** for
output attribution, never model-authored tool arguments. Without an explicit
request/revision association and `answer` purpose, useful-answer latency is null
and `output_association` is `unknown`. In particular, GPT Live currently does not
identify which output audio answers a client delegation; Core does not guess
from proximity to a result. `backchannel` output is measured separately and
never counted as a useful answer. Superseded revisions cannot inherit a later
receipt as a successful answer.

The caller-speech-end source is reported: provider VAD receipt or a renderer's
`input.speech_stopped` signal. If neither exists (including typed-input tests),
caller-speech-end metrics are null. Source timestamps are not mixed across
unsynchronized clocks. Unknown or negative durations are null, never zero.

Failed, timed-out and superseded requests emit explicit stages. Session closure
emits `tracking_closed` with the close reason and known backend outcome. Tracking
is bounded to 64 requests and 128 context-event IDs per generation; eviction is
reported and late callbacks cannot reopen closed session tracking. Audio-lane
observation only updates bounded metadata, without storage or model calls.

Renderer wire messages remain backward compatible. Consumers should negotiate
`audio_protocol=1`, send their existing scoped `playback.progress` messages with
`item_id`, `session_generation`, `audio_end_ms`, and the actual `evidence` kind.
Core preserves that evidence in telemetry. Neither RTP transmission nor an
estimated playback clock proves that the caller heard the answer.

Validation includes deterministic stages, silent leading frames, backchannels,
explicit and unknown associations, late captions/receipts, superseded work,
interruption after output drain, bounds, repeated shutdown, native tool lifecycle,
concurrent sessions and the actual renderer WebSocket control reader. Paid Live
checks record first output, first non-silent output, confirmation-caption arrival,
and request stages independently; their sink does not simulate carrier playback.
