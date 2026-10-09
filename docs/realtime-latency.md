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
