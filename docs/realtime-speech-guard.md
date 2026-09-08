# Realtime speech output

Realtime agents receive a dedicated conversational prompt. They retain tool
argument rules and structured tool access, but do not inherit worker thought,
timer, model-selection, or one-shot completion instructions. Ordinary text
workers keep those instructions.

Google Live explicitly requests `includeThoughts=false`. Parts marked `thought`
are discarded. This alone cannot prevent a model from generating instruction
narration as ordinary speech, which is how the recorded incident reached Core.

The Google adapter buffers opening PCM until it has a short output transcript
prefix (32 bytes or sentence punctuation), or the complete transcript for a
short response. Guard-bearing transcript events precede released audio and
cannot be dropped by the audio backpressure path. The buffer is capped at five
seconds of PCM (240,000 bytes); missing transcription drops the audio and emits
an output-blocked event instead of releasing unchecked speech.

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
late-detected narration stops subsequent output. Google receives this prefix
buffer; other realtime providers receive the shared transcript detector and
interruption path. There is no additional model call on the clean path.

Tests cover the recorded incident with zero released PCM, fragmented and late
transcripts, missing-transcript bounds, thought parts, clean streaming, later
leak suppression, recovery limits, preserved results, and normal conversation.
The paid Google Live MCP smoke now rejects internal narration even when its
answer is otherwise correct.
