# Realtime speech output

Realtime agents receive a dedicated conversational prompt. They retain tool
argument rules and structured tool access, but do not inherit worker thought,
timer, model-selection, or one-shot completion instructions. Ordinary text
workers keep those instructions.

Google Live uses model-specific thinking settings: legacy models request
`includeThoughts=false`, standard Gemini 3.8 omits `thinkingConfig`, and Extended
Thinking uses its supported reasoning levels. Parts marked `thought` are
discarded. This alone cannot prevent a model from generating instruction
narration as ordinary speech, which is how the recorded incident reached Core.

The Google adapter buffers opening PCM until it has a short output transcript
prefix (32 bytes or sentence punctuation), or the complete transcript for a
short response. Guard-bearing transcript events precede released audio and
cannot be dropped by the audio backpressure path. The buffer is capped at five
seconds of PCM (240,000 bytes); missing transcription drops the audio and emits
an output-blocked event instead of releasing unchecked speech.

These are the default settings. `realtime_output.speech_guard` can explicitly
select `transcript_prefix`, change `prefix_bytes`, or change
`max_buffered_audio_ms`. The five-second capacity measures generated audio,
which may arrive faster than playback. It is neither a five-second wait nor a
wall-clock deadline. Missing transcript/audio ordering can therefore delay a
short reply until its completed transcript. The measured waiting time is
reported separately; see [realtime latency](realtime-latency.md).

The deterministic guard recognizes specific self-directed instruction-narration
patterns, including the recorded "Private reasoning processed..." case. It
handles fragmented transcripts and audio arriving first. A blocked response's
pending and subsequent audio is discarded, and its final text is not persisted
as an assistant reply. Known thought parts remain excluded. Clean responses
stream after the prefix check rather than waiting for their entire turn.

Core also applies the narration detector to realtime output transcripts. Its
existing interruption/recovery path suppresses the response, drops queued
output, and allows at most one recovery request until a clean final response
or a new user turn. Recovery asks for the verified result and explicitly says
not to repeat completed actions. It preserves conversation and tool results.
`realtime.output_blocked` records the reason and recovery outcome without the
leaked content. Serialized tool-call leakage keeps its existing telemetry.

## Scope

This is a targeted defense, not a general semantic guarantee against every
possible wording in every language. Already played audio cannot be retracted.
The opening-prefix buffer protects the reproduced audio-first incident;
late-detected narration stops subsequent output. Google retains this prefix
buffer by default. Continuous providers use the monitoring policy described
below; other realtime providers retain the shared transcript detector and
interruption path. There is no additional model call on the clean path.

Tests cover the recorded incident with zero released PCM, fragmented and late
transcripts, missing-transcript bounds, thought parts, clean streaming, later
leak suppression, recovery limits, preserved results, and normal conversation.
The paid Google Live MCP smoke now rejects internal narration even when its
answer is otherwise correct.

## Continuous streams and playback identity

For providers advertising `continuous_conversation`, the default guard is
`monitor`: PCM is delivered immediately while transcripts are monitored for
specific tool syntax and internal narration. Short replies, delayed captions,
and uncaptioned sounds do not require a 32-byte transcript prefix. Applications
that require checks before playback can explicitly select `transcript_prefix`;
its bounded audio capacity still fails closed. Missing captions are reported as
incomplete transcription (or strict-policy withholding), never as proof of a
leak, and do not request the internal-material recovery prompt. Other providers
retain their defaults. Google also accepts explicit `monitor` mode.

Audio segments use session-unique application playback IDs. These are observed
silence/gap boundaries, not provider turns. Transcript display grouping retains
its own timestamps and identity; the monitoring text window survives audio
segmentation. Original PCM, including silence, retains delivery order. Silence is marked
separately and does not keep speaking state active after voiced playback drains. Primary
Live audio has no authoritative timestamp alignment to late captions, so an
application segment is not proof of a complete answer or exact text/audio
correspondence.

Cancellation invalidates pending guard releases and queued/in-flight bridge
frames for that playback. Late frames remain invalid after subsequent speech
starts. Automatic provider yielding and clearing the player's backlog are
separate: a renderer's `input.speech_started` clears Core playback even when the
provider handles its own yielding. This does not cancel backend inference or
tools. Confirmed content suppression can still request corrective instructions.
Recovery attempts are limited per playback incident for continuous streams.

Continuous speaking state waits for Core/bridge delivery and, when supported,
client playback acknowledgements. Old provider-arrival timestamps do not make
newly released buffered audio immediately idle. Without client receipts, Core
can observe bridge delivery and quiet, but cannot confirm that audio was heard.

### Bridge contract and diagnostics

The existing `audio.frame` metadata keeps `item_id` and cumulative
`audio_end_ms`. Renderers advertise receipt support with `playback.ready` and
send `playback.progress` only as audio is consumed. The `interrupt` control now
includes `item_id` and `reason` for continuous streams. Players should remove
that playback's scheduled audio and ignore its late frames, while preserving
other playback IDs. The bridge sends clear controls before later audio.

Clients may report `{"type":"playback.cleared","item_id":"..."}` after
clearing their playback. This is optional and additive. Core emits separate
`realtime.playback_clear_sent` and `realtime.playback_clear_acknowledged` events,
with bridge-write and client-reported clear latency. These are distinct from
provider behavior and do not prove audible delivery. Interruption telemetry
records the actual provider command (for Live, `session.instructions.append`),
queueing success, and that cancellation remains unconfirmed. Backlog telemetry
includes pending, written, acknowledged, and awaiting-playback duration.

The regression suite includes paced playback and actual in-flight delegated
inference. `RUN_OPENAI_LIVE_PLAYBACK=1 go test -run
'^TestOpenAILivePacedInterruptionPlaybackLive$' -v -timeout 2m .` enables the paid
Live/caller-audio test with simulated playback receipts. It requires an
`OPENAI_API_KEY`; it does not test browser hardware or carrier playback.
