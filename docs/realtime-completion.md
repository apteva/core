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
