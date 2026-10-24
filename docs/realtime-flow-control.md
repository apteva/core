# Realtime audio flow control

Core applies the same policies to the first audio segment and every subsequent
segment, across provider adapters. Capture devices, local VAD and actual playback
remain client responsibilities.

## Independent audio lane

The provider-event/audio lane and microphone intake run separately from the
conversation actor. Transcript storage, backend thread creation/results and
telemetry forwarding cannot stop microphone sending or renderer delivery.
Ordered non-audio events remain on a bounded actor lane; exhaustion produces an
explicit recovery failure. Audio observations are coalesced, not transcript text,
tool calls, or finalization events. Technical-speech checks run before audio
admission even if transcript persistence is stalled. Backend tools retain their
normal thread ownership and are not cancelled by playback interruption.

## Input admission and recovery

Managed WebSocket bridges admit audio directly into Core's duration queue rather
than waiting on a packet-count channel. The default combined Core/adapter budget
is **5 seconds**, with a **5-second maximum age**, 20 ms internal frames and a
1-second sustained adapter-pressure timeout. Adapter policy may override these
values. Excess or stale input is explicitly reported as a discontinuity; it is
never silently replayed later.

`RealtimeAudioInputSession.SendAudioFrame` carries the original arrival time,
maximum age and a settlement callback through the adapter queue. Reservations
remain charged until transport write or discard, including a write in progress.
OpenAI Realtime, GPT Live, compatible providers and Google Live implement this
contract. Third-party adapters should implement it if they buffer input. A nil
admission error confirms local admission, not provider receipt or processing.
Pressure/closed errors guarantee no admission. Other send failures may be
ambiguous and must not be blindly replayed. Adapter Close settles outstanding
reservations exactly once. Transport deadlines also respect the remaining age
budget. Continuous providers retain sample-rate catch-up pacing.

Connection/restoration/Close work remains asynchronous, cancellable and guarded
by session generation. `session.state` distinguishes provider recovering, ready
and failed from the external WebSocket's own connection state.

## Output bursts and renderer capacity

Core buffers up to **15 seconds** of pending canonical PCM16/24 kHz output,
including the public output channel and in-flight bridge frame. A full channel
alone does not cancel speech. Deferred output remains ordered. The same bound
also limits queue age; sustained exhaustion produces an explicit failure and
session recovery, without cancelling backend work. This is not a guarantee that
an arbitrarily slow renderer can play unlimited generated speech.

Existing clients retain their framing. Updated renderers opt in using
`?audio_protocol=1`. The `X-Apteva-Audio-Session` contract includes
`playback_flow_control: true`, scoped cancellation, provider state events and
input/output budget information. Before expecting output, send:

```json
{"type":"playback.ready","capacity_ms":2000,"evidence":"estimated"}
```

`capacity_ms` enables socket-local flow control; accepted values are clamped to
20–15000 ms. Omit it to retain legacy delivery behavior. When enabled, Core
splits provider bursts into at most 20 ms transport frames and sends only within
the advertised outstanding playback window. Core continues writing interruption
and provider-state controls while waiting for playback progress. An unresponsive
renderer has a bounded wait; Core does not wait indefinitely for credit.

Each `audio.frame` metadata message carries `item_id`, `session_generation`,
`audio_end_ms`, `response_id` and `silence`, immediately followed by its binary
PCM frame. Renderers report cumulative progress using the exact item/generation:

```json
{"type":"playback.progress","item_id":"speech-1","session_generation":3,"audio_end_ms":240,"evidence":"estimated"}
```

Only matching receipts release that playback's credit. Stale/future IDs or
another generation cannot release capacity. A renderer must actually honor
`interrupt` for the specified item/generation, clear its queued playback and
acknowledge the completed clear:

```json
{"type":"playback.cleared","item_id":"speech-1","session_generation":3,"scope":"local"}
```

A local clear releases that scoped capacity; it does not prove remote playback.
Negotiated scoped clears are available with all realtime models, independently
of continuous-conversation capabilities. Legacy clients retain existing control
framing. Send `input.speech_started` promptly for local interruption where the
provider does not supply it. Keep reading controls while playback is paused or
waiting for credit. Telephony/carrier buffering remains a separate client change.

## Evidence and timing

`realtime.input_flow` reports frame count, mean/max Core input residence, adapter
queue residence, transport-write duration, combined queued duration and budget.
`realtime.utterance_timing` adds mean/max output queue residence and output frame
count to the existing provider/guard/bridge stages, plus renderer-credit wait
and renderer transport-write time for every transport segment. Playback progress and clear
receipts retain their evidence source. Partial burst writes settle the delivered
bytes and the discarded remainder separately; delivery settlement is idempotent.
Audio telemetry uses a bounded nonblocking lane; loss of telemetry itself is
reported as `realtime.telemetry_drop` rather than blocking audio.

Transport writes, estimated playback, RTP transmission and carrier acknowledgments
remain different evidence. These timings do not claim provider-internal inference
time or prove that a human heard the audio. No raw audio or credentials are logged.

## Verification

`realtime_flow_test.go` covers stalled transcript storage and telemetry, input
budget shared with each adapter, transport settlement, stale input, first/later
output bursts, actor-delayed interruption, bounded overflow, scoped renderer
credits and partial receipts. Existing recovery, completion, guard, backend and
interruption tests also run under the race detector. Opt-in paid acceptance tests
are `TestOpenAILivePacedInterruptionPlaybackLive` and
`TestOpenAILiveRecoveryWithContinuousIntakeLive`.
