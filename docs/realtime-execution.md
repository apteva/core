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

## Caller evidence and incremental results

Delegation offsets do not imply that a transcript must cover intervening
silence. Stable caller evidence can start execution once partial captions have
settled for 250 ms; provider-final captions are immediately eligible. New
speech without captions and entirely missing caller evidence retain a bounded
10-second failure path. Late captions still advance the assignment revision
and invalidate stale tool decisions. An incomplete or ambiguous request must
be clarified before an action. These admission rules do not infer semantic
completeness from punctuation.

Thread execution may now publish `ThreadWork.kind=result_chunk` through the
ordinary durable parent inbox. Runtime metadata carries request ID, revision,
delegation ID and a deterministic content hash. Only **complete public answer
sentences matching successful, unshortened tool evidence verbatim** are eligible.
JSON string values in results are supported. This conservative lexical check
is not semantic verification: paraphrases, unsupported conclusions, large
sentences and incomplete text wait for the ordinary final result. It does not
stream reasoning or partial tool arguments, or infer success from a pending
operation. No additional model call is used to approve chunks.

The collector is bounded to 16 KiB per response and sentences to 480 bytes.
An eligible sentence is emitted once its following whitespace arrives, before
worker completion. Failed tool results and preview-only archives cannot justify
an early sentence. Revisions are checked before publishing and consuming.
Provider-managed delegation retains its existing provider delivery path.

Delivered sentences are recorded with their work metadata. Duplicate chunks,
replayed inbox entries and chunks arriving after the final result are ignored.
The final result removes only complete previously delivered sentences; it
preserves changed wording and corrections. Speech interruption keeps backend
work running. While new caller speech awaits captions, chunks supply quiet
context instead of requesting speech; they remain eligible for the later final
spoken result. Context admission is not proof of model speech or playback.

`realtime.execution_chunk` records correlation and explicitly reports
`speech_confirmed=false`. Use output transcript, audio and playback timing for
meaningful answer onset, separately from acknowledgements and backchannels.

## Latency regression commands

```sh
GOWORK=off go test -race -short -run 'TestRealtime|TestManagedAssignment|TestQueuedToolRejects|TestLive' .
RUN_REALTIME_LATENCY_LIVE=1 GOWORK=off go test -race -run '^TestRealtimeVerifiedStreamSolLive$' -v .
```

`realtime_latency_test.go` covers repeated 400 ms stalls, continuous ordered PCM,
jitter, concurrent session isolation, strict-pacing controls, silence, late
captions, revisions, duplicates, unsafe output, interruption and inbox replay.
The paid Sol test measures arrival of the first eligible result at the voice
context adapter before backend completion; it does not measure spoken onset.

For matched multi-turn tests using an existing compatible Telephony checkout:

```sh
python3 scripts/realtime_transport_replay.py \
  --telephony /path/to/apps/mcp/telephony --server /path/to/server \
  --output /tmp/new-replay-results --pause-ms 400
```

Add `--live` with `OPENAI_API_KEY` in the environment for GPT Live. The helper
uses Go overlays, leaving Telephony and Server sources untouched. It exercises
direct PCM and actual loopback carrier handlers through the Server/Core audio
connection. It retains peak delay but separately checks post-stall recovery.
Two turns share a session by default. The existing live 2-second acceptance
threshold remains visible; provider latency can fail it. It does not place
PSTN calls, run a browser, measure handset acoustics, or strip model-generated
silent PCM.

## Voice execution latency policy

When delegated OpenAI Live execution has no explicit provider/model override,
Core uses the existing OpenAI API integration with `gpt-6-luna` and reasoning
`none`. This is an execution-worker default; it does not change main-thread
models, direct-tool voice models, or the voice model. Existing explicit model
choices remain authoritative. Other adapters may supply their own defaults.

`realtime_execution` is accepted in the provider configuration and as a full
per-thread override. Additional optional fields:

```json
{
  "mode": "thread",
  "model_id": "gpt-6-luna",
  "reasoning": "none",
  "preload_tools": ["appointments_availability"],
  "result_tools": [{
    "tool": "appointments_book",
    "success_field": "status",
    "success_value": "confirmed",
    "public_fields": ["status", "date", "reference"]
  }]
}
```

`model` still selects a configured Core tier; `model_id` selects an isolated
exact model and is mutually exclusive with `model`. Exact-model overrides
currently have an implementation on the OpenAI Responses adapter. Other text
providers retain tier selection. `reasoning: auto` preserves model defaults.
The reasoning setting is also sent in provider-managed Live delegation.

Preload and result-tool configuration accepts at most 20 combined entries.
Preloading has a 64 KB schema/description budget, is stable across assignments,
and never grants permission: parent capabilities and `no_spawn` still apply.
Result tools are also preloaded. Normal discovery remains available.

Direct delivery is opt-in through `result_tools`, not inferred from arbitrary
tool JSON. The success field must be a JSON string exactly matching the
configured value. Every public field must be a present, nonempty scalar at the
JSON top level; nested values, null, failures, images and other media use normal
continuation. Public JSON is capped at 400 bytes, conservatively below the Live
500-token append limit; larger receipts use normal inference. Tool results and
attachments retain the normal history/archive behavior.

Eligible schemas require the Core-only `_complete_assignment` boolean. The
worker must explicitly choose true or false, and sets it true only when that single call will finish **all** required
work. Core strips the hint before dispatch. A declared success does not finish
an assignment unless the successful receipt matches its host-configured
contract, the request/revision is current, the latest batch contains one call,
and no other work or input is pending. For example, “book and then email me”
must continue after the booking. A reserved-name conflict disables the shortcut.

Core persists the actual tool result, publishes public facts through the
existing idempotent parent inbox, journals completion and waits for the next
assignment. The parent retains revision checks and delivery deduplication.
`thread.work_direct_result` records the request/revision, delegation and call ID,
and the one skipped narration request. It does not prove audible playback.
`thread.work_direct_result_evaluated` records eligibility and specifically
identifies `completion_decision_missing` or `assignment_not_complete`. Missing
flags remain a safe ordinary continuation, never an inferred completion or a
retry of the already executed action. The required schema and description are
worker-local; app schemas and dispatched arguments remain unchanged.
No raw result is promoted to instructions. The voice adapter receives these
facts through its ordinary context/commentary interface and phrases them.

This shortcut applies to Core-managed `thread` execution. `provider` mode keeps
its required function-output/response-continuation protocol and does not also
send a direct receipt. Tool preloading and model/reasoning configuration apply
to both modes.

Paid validation: `RUN_VOICE_EXECUTION_LIVE=1 go test . -run
'TestVoiceExecution(LunaBookingAccuracyLive|LiveLatencyComparison)' -v` with
`OPENAI_API_KEY` supplied at runtime. The comparison uses real GPT Live/Brise
sessions, identical typed caller input, and a paced PCM silence clock; it
measures backend delivery and verified confirmation-caption arrival, not PSTN
playback or the exact onset of meaningful speech. All booking/email actions
are local fixtures. Test candidates on the application's own tools and
procedures before treating these bounded cases as production accuracy evidence.

### Local validation, 2026-10-24

Real OpenAI API checks passed for the exact booking details, a missing time
(no booking), an unavailable slot (no false success), and booking followed by
email (both steps completed). The confirmation-only shortcut took one inference
request; the email workflow retained all three requests and both tool actions.

Nine matched GPT Live/Brise sessions, three per variant with reversed order in
the middle trial, produced these medians:

| Execution variant | Backend result delivery | Verified confirmation caption | Backend requests |
| --- | ---: | ---: | ---: |
| GPT-6.1 Sol, automatic reasoning, normal continuation | 5.150 s | 13.340 s | 2 |
| GPT-6 Luna, no reasoning, normal continuation | 4.210 s | 12.090 s | 2 |
| GPT-6 Luna, no reasoning, direct receipt | 2.182 s | 8.701 s | 1 |

The receipt variant reduced median backend delivery by 58% against Sol and
48% against Luna with normal continuation in this small local sample. Caption
arrival improved by 35% against Sol; it is not a playback/onset measurement.
One additional real provider-managed Luna/none session also completed correctly
with one booking and the normal two backend requests. No deployment or actual
booking/email action was performed by these tests.


### Required-decision regression validation

The follow-up real `gpt-6-luna` / `none` checks ran three two-turn
availability-then-booking conversations: all six assignments used a direct
receipt and one model request each, with no missing decisions or duplicate
bookings. Missing time, unavailable slots and booking-then-email checks also
passed. Three real GPT Live/Brise booking sessions each used one backend request
and one direct receipt. These are bounded fixture results, not a guarantee that
any model will always comply with its schema.

Run the repeated worker test with `RUN_VOICE_EXECUTION_LIVE=1 GOWORK=off go test
-count=1 -run TestVoiceExecutionLunaMultiTurnReceiptsLive -v .`. Run the Live
receipt path with `-run TestVoiceExecutionLiveLatencyComparison/luna_receipt`.
Provide `OPENAI_API_KEY` only at runtime. Telephony's remote two-agent/carrier
workflow still requires its own replay; local fake tool timings are not a
before/after comparison against the remote application's service latency.
