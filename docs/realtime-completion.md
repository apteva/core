# Realtime completion and transcript evidence

Core treats the realtime `done` tool as a completion request. Normal workers
retain their immediate-stop behavior. A finishing realtime worker consumes the
current provider utterance, drains queued audio including in-flight bridge
writes, and waits for renderer playback progress when the renderer has reported
support through `playback.progress` or an optional `{"type":"playback.ready"}`
bridge control. Renderers should send `playback.ready` on connection to allow
Core to wait for acknowledgement of the very first utterance. It accepts no new caller input, tools,
configuration refresh, reconnection, or model response requests during finishing.

Finishing has a five-second bound. After the utterance and output settle, a
100 ms quiet period allows trailing provider-final transcripts to arrive.
Explicit stops, context cancellation, and an absent/disconnected audio bridge
end finishing promptly. Core does not wait for the caller's goodbye. A text-only
worker can finish without an audio bridge.

Each saved transcript has `realtime_transcript` metadata in its SessionEntry:

- `id`: stable turn identity scoped to the logical runtime, provider session
  generation, role, and item/utterance ID (response ID if the provider omits an
  item ID).
- `generation`, `response_id`, `item_id`: provider event correlation.
- `status`: `provider_final`, `interrupted`, or `blocked`.
- `finalization_reason`: provider finalization or the boundary that retained
  incomplete text, such as `bridge_disconnected`, `audio_disconnected`, or
  `server_shutdown`.

Provider adapters explicitly distinguish transcript increments from complete
snapshots. Disconnects, session replacement, and worker shutdown save pending
text as interrupted. A late provider-final event revises the same turn and
advances its history cursor. Live history readers should replace rows by the
stable transcript ID when receiving revisions; `seq` remains the polling cursor.
History order remains chronological, while cursor pagination sorts revisions by
sequence to avoid skipping another entry. Providers receive only ordinary
message text, without the internal transcript metadata.

Assistant text still passes the technical-speech guard. If later evidence blocks
a previously retained partial, its saved turn is revised to empty blocked
content. Transcript finalization and successful socket writes do not establish
that the caller heard speech. Realtime history survives completion and ephemeral
thread cleanup; retention/deletion is a separate caller-owned history policy.

Before cleanup, `realtime.thread_ended` enters the durable telemetry outbox with
the terminal trigger, initiator, completion tool call ID where applicable,
current response/item IDs, session generation, pending transcript count, queued,
pending, bridge-written, dropped, and renderer-acknowledged audio durations, and
graceful completion outcome (`completed`, `timed_out`, `disconnected`, or
`forced_stop`). `realtime.completion_requested` records the start of finishing.
Repeated cleanup emits one terminal receipt. The generic bridge-disconnect event
remains a separate report of the transport boundary.

## Continuous conversation completion review

For continuous phone sessions (`turn_detection.profile: telephony`), Core also
checks settled conversational transcripts when `done` is authorized. This
covers a mutual closing that the voice model speaks without delegating an end
operation. It uses the configured execution provider's small model, with no
tools, a 12-message/12 KB context bound, one in-flight review, an eight-second
timeout, and at most two attempts per transcript revision. Audio continues on
its independent lane. Each review incurs a small additional inference request.

`realtime_output.completion_mode` accepts `auto` (the phone default above),
`semantic` (explicit opt-in), or `delegation` (only existing explicit completion).
A review requires both speakers, one second of settled input/output, no pending
backend/tools, at most 250 ms of paced input backlog, and drained speech receipts. The classifier must confirm a mutual
ending; silence, an assistant-only goodbye, quotations and acknowledgements do
not qualify. Invalid responses and errors leave the conversation open. Semantic
classification is probabilistic; the opt-in live regression corpus tests both
accepted and rejected endings in several languages.

A positive review is discarded after new speech, transcripts, generation or
instruction changes, or when permission/readiness no longer holds. Accepted
reviews use the existing `done` execution gate and bounded graceful completion,
without requesting another farewell. Telemetry records `completion_review` as
the terminal trigger and `application_review` as initiator. Review telemetry
separates the classifier's `completed` suggestion from `accepted` application
action and reports `stale`, failure, duration and token usage. This is not proof
of audible carrier playback; the existing receipt evidence remains authoritative.

Repeatable tests: `go test -short -run 'TestCompletionReview|TestRealtimeCompletion' .`.
Paid semantic tests: `RUN_COMPLETION_REVIEW_LIVE=1 go test -run '^TestCompletionReviewLive$' -count=1 -v .`
with `OPENAI_API_KEY` supplied at runtime.

On thread cleanup, a completion request that reached `completed` or `timed_out`
uses WebSocket close code 1000 with the exact reason `realtime conversation
completed` or `realtime completion timed out`. These reasons are lifecycle
instructions on the authenticated thread socket. Ordinary cleanup retains
`audio bridge unregistered`; it does not authorize a carrier hangup. Receivers
must scope the signal to the current call owner and preserve the completion
outcome in diagnostics. A completed model session alone never emits this signal.
