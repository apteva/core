# Realtime conversations and managed execution threads

Core keeps the existing realtime provider integration and credentials. Selecting
`gpt-live-1` on the OpenAI integration uses the Live wire protocol. Existing
OpenAI Realtime, Gemini Live, and xAI models keep their direct tool-call path.
The public API model tested here is `gpt-live-1`; this does not establish API
availability for GPT-Live 1.1.

## Execution ownership

GPT-Live separates continuous conversation from backend reasoning and tools.
`realtime_execution.mode` selects who supplies backend inference:

- `thread` (default): an ordinary managed child runs Core's normal `Thinker.Run`
  loop with the selected text provider/model, normal context recovery, tool
  discovery, memory, retries, and execution controls.
- `provider`: the same managed child receives inference through the voice
  provider's optional execution-session adapter. OpenAI uses Live Responses
  delegation. Core executes its custom function tools, returns every matching
  result, and continues the remote response once. It does not issue a duplicate
  standalone model request or silently fall back to another executor.

Neither mode requires a particular text model. Provider mode requires a backend
model from the same API integration. Providers advertise this capability;
unsupported combinations fail at startup. Direct-tool voice models do not
create execution children.

Example provider configuration:

```json
{
  "realtime_enabled": true,
  "providers": [{
    "name": "openai",
    "default": true,
    "models": {"large": "gpt-6.1-sol"},
    "realtime_models": {"large": "gpt-live-1"},
    "realtime_voice": "brise",
    "realtime_execution": {
      "mode": "thread",
      "provider": "openai",
      "model": "large"
    }
  }]
}
```

`POST /threads/<id>` can override the provider default for that conversation:

```json
{
  "realtime": true,
  "provider": "openai",
  "model": "large",
  "voice": "sillage",
  "directive": "Speak French and help the caller using verified results.",
  "realtime_execution": {"mode": "provider", "model": "large"}
}
```

## Thread ownership and assignments

A conversation remains one managed realtime thread with its own parent, audio
bridge, transcript, and completion lifecycle. Its execution child has an ordinary
thread identity, session history, trace, and shared event bus. It receives the
parent's delegable capability ceiling, subject to normal depth and `no_spawn`
restrictions. Creating that child does not invalidate unrelated main inference.

The child accepts durable inbox assignments with runtime-owned `ThreadWork`
metadata: request ID, revision, origin, kind, status, and reporting worker. These
fields do not enter model requests. They survive inbox persistence and session
history, and results return through the parent's durable inbox. An assignment
completion keeps the worker available for the next assignment; `done` still
terminates that worker and does not itself stop its parent conversation.

Plain final assistant text completes the assignment. `send(id="parent", ... )`
can deliver progress; `status="completed"` explicitly returns an assignment
result. `status="completion_requested"` requests graceful completion of the
parent conversation, which checks its own completion permission and execution
control before entering the existing finishing phase. Ordinary sends to other
threads retain their existing behavior.

Provider delegation IDs are scoped to connections and are distinct from Core
request IDs. A provider notification for already-running typed work attaches to
that assignment. Corrections advance its revision without cancelling inference
just because someone speaks. Core checks revisions before dispatch and again
before queued external tools start. Actions already admitted can finish; their
actual outcomes and generated file references remain evidence and cannot be
undone by discarding a stale model decision.

Voice reconnection keeps the execution child and its history. Connection IDs are
cleared and results route to the current voice session. Explicit stop cancels
child inference through normal thread shutdown. Parent cleanup cascades to the
child. The execution journal remains available for recovery; operation-specific
idempotency is still required for uncertain external side effects.

## Continuous voice behavior

The Live adapter streams PCM16 mono at 24 kHz through the existing Core audio
bridge. Brise and Sillage are supported French voices; authorized custom voice
IDs use the documented voice object. Conversation instructions stay separate
from detailed execution instructions. Startup text is conservatively bounded.

Caller and assistant transcript fragments may overlap and have no provider-final
turn event. Core groups them for display and persists revisions with
`application_grouped` metadata. It does not label them provider-final. Generated
audio, bridge writes, and acknowledged playback remain separate evidence.

Screenshots go to the execution child, with attachment ownership transferred
only after inbox acceptance. Later screenshots cannot be consumed by an older
handoff. Generated files use the existing reference contract. The speech guard,
bounded graceful completion, forced stop, and final usage collection remain in
the voice runtime. Voice duration is billed separately from backend tokens.

`realtime.execution_started`, `realtime.execution_completed`,
`realtime.execution_superseded`, and `realtime.backend_event` describe routing and
remote response lifecycle. Tool and inference telemetry belongs to the execution
child; audio and spoken transcripts belong to the conversation.

## Verification

Deterministic and race tests cover ordinary thread execution, durable inbox
correlation, duplicate delivery, revisions, queued stale tools, capabilities,
parent ownership, reconnect routing, completion requests, preserved artifacts,
vision handoff, and the managed Responses function-result continuation.

Paid tests are opt-in and use runtime credentials only:

```sh
RUN_OPENAI_LIVE_SMOKE=1 go test \
  -run '^TestOpenAILiveFrenchCoreExecutionLive$' -count=1 -v -timeout 6m .
```

The matrix checks both execution modes and both French voices. It requires one
actual fixture tool execution, a model-supplied `_reason`, assignment completion,
and audio/transcript containing the verified result. Set
`RUN_OPENAI_LIVE_AUDIO_INPUT=1` to generate caller PCM with an independent
`gpt-realtime-2.1-mini` session and stream it at microphone cadence. Otherwise,
the test exercises exact typed input and real generated speech. These synthetic
calls do not replace carrier/browser playback testing or human listening.

## Server and UI follow-up

Core supplies this functionality; server and UI changes are separate:

- Forward and preserve `realtime_models` and `realtime_execution` configuration,
  including the per-thread override.
- Offer GPT-Live, French voices, backend model, and execution ownership through
  the existing OpenAI integration, using the same API key.
- Use `realtime_providers` capability metadata from `GET /config` to avoid
  offering unsupported VAD, truncation, or managed-execution controls.
- Show execution children and their work separately from spoken conversation.
  Replace transcript rows by stable transcript ID when revisions arrive.
- Display voice-duration usage separately from backend token usage, and report
  playback acknowledgments through the existing bridge where supported.

References: [Live delegation](https://developers.openai.com/api/docs/guides/live-delegation),
[migration](https://developers.openai.com/api/docs/guides/live-migration), and
[session lifecycle](https://developers.openai.com/api/docs/guides/live-conversations).
