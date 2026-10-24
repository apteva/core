# Realtime media transport selection

`realtime_output.media` is immutable per Core session:

```json
{"media":{"transport":"websocket","format":"g711_ulaw"}}
```

Transport values: `websocket`, `webrtc`. Format values: `pcm16` (mono 24 kHz),
`g711_ulaw`, `g711_alaw` (mono 8 kHz). Omission retains the existing PCM/WebSocket
path. Unsupported explicit selections fail; there is no silent transport or
codec fallback. The first adapter supporting these alternatives is OpenAI Live.
Other adapters retain their existing behavior and reject explicit unsupported
media selection. This does not change the carrier's SIP/WebSocket transport.

## Lifetime and streaming

One provider session serves the entire call, until an actual failure/recovery or
termination. WebSocket sends periodic pings. WebRTC maintains one peer connection
with continuous SRTP audio and `oai-events` data-channel control. Session creation
uses the documented HTTP SDP exchange, then waits for `session.started`. It does
not send a WebSocket-style `session.start` over the data channel. Caller audio is
paced continuously, including silence, while output audio and backend work run
independently. Ordinary replies do not reconnect. Queue admission, playback
credits, cancellation and timing all use the selected format's byte rate.

PCM WebRTC uses native libopus (24 kHz encoder/decoder; 48 kHz RTP clock). Linux
and macOS need libopus, or `APTEVA_OPUS_LIBRARY` pointing to a compatible library.
On Alpine, install `gcompat` and `opus`; official Docker images include both.
Other platforms fail explicit PCM/WebRTC setup. G.711 WebRTC requires no Opus.
Provider format support is checked during SDP negotiation. A small bounded RTP
reordering window activates only after a missing sequence; healthy packets are
released immediately. It cannot reconstruct genuinely lost speech.

## App contract

SDK `RealtimeSpawnRequest.realtime_media` and runtime equivalent carry the intent.
Server forwards it to Core and preserves the authenticated app's reachable gateway
origin for the audio URL, avoiding an unnecessary public tunnel when co-located.
Remote apps retain the gateway address they actually reached.

The renderer opts into `audio_protocol=1` and verifies the handshake's
`audio_format`, `sample_rate`, and `channels`. Native G.711 is rejected for legacy
PCM-only renderers. Recording and monitoring taps may still decode/resample audio;
G.711 pass-through describes the matching encoded media path, not all CPU work.

## Repeatable verification

- `GOWORK=off go test -short ./...`
- `GOWORK=off go test -race -run 'TestOpenAILiveWebRTC|TestRealtimeRTP|TestRealtimeMediaContract|TestRealtimeNativeOpus' .`
- Paid, explicit recorded-speech comparison:
  `RUN_REALTIME_MEDIA_COMPARE=1 REALTIME_MEDIA_REPLAY_PCM=/path/phrase.pcm16 REALTIME_MEDIA_REPORT_DIR=/path/results GOWORK=off go test -run '^TestRealtimeMediaTransportLiveComparison$' -v .`
  Supply `OPENAI_API_KEY` through the environment. The fixture is raw mono 24 kHz
  PCM16. Reports measure last voiced input frame to first audible output, not
  carrier playback. Repeat and alternate transport order before choosing defaults.

All alternatives remain opt-in. Initial local measurements do not establish a
consistent latency winner, and persistent connections do not eliminate provider
response-generation latency. SDK v0.107.0 provides the optional media fields for independently built apps;
servers must also support media selection before an app opts in.

### Duplex overlap policy

Provider-owned full-duplex interruption is advertised as `provider_interruption`
in the renderer handshake. It does not require local speech-triggered clears.
Automatic renderers let that provider decide how to handle greetings, short
acknowledgements and genuine interruption; microphone input remains continuous.
An explicit local-interruption override retains scoped hard cancellation.
Cancellation identities and speech-guard blocking are unchanged. This avoids
an energy/VAD signal cancelling an entire continuous application speech segment.

Effective turn-detection telemetry for continuous providers identifies provider
ownership and reports `vad_overrides_applied: false`. It omits the traditional
750 ms silence threshold. Requested settings remain separate from applied
provider settings.
