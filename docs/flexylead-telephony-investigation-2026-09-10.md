# Flexylead 1102: text model selection and realtime investigation

The reported symptoms are real. The text switch is caused by inconsistent
Google tier configuration, rather than a valid saved Gemini override being
lost after every tool call. A fresh isolated realtime conversation passed
today with the same live model and Kore voice. The readiness session's repeated
1008 closes remain unexplained by the available evidence.

No product code, agent configuration, running service, phone route, or carrier
configuration was changed. Investigation files are local. Credentials were
used only in memory and subprocess environment; this report contains none.

## Text model: root cause reproduced

The agent's server metadata selects `google` and contains no `model_override`.
The generated Core config requests `antigravity-preview-05-2026` for all three
Google tiers. That model is absent from Core's static `geminiModels` catalog.

`applyModelOverrides` calls `GoogleProvider.SetModel` for the large tier.
`SetModel` silently ignores names outside that catalog. Medium and small are
then assigned directly, without the same check. The resulting runtime map is:

| Tier | Configured | Actually resolved |
| --- | --- | --- |
| Large | antigravity-preview-05-2026 | gemini-3.6-flash |
| Medium | antigravity-preview-05-2026 | antigravity-preview-05-2026 |
| Small | antigravity-preview-05-2026 | antigravity-preview-05-2026 |

Main starts on large. Processing an external event selects medium. The log's
first switch coincides with the operator setup event: lines 94–108 show the
event being drained while the asynchronous tool is completing. There is no
intentional provider fallback between the two Google requests.

Four isolated probes exercised the real provider-config builder, thinking loop,
asynchronous tool dispatch, and event handling, with a local fake LLM:

| Configuration and continuation | Observed model sequence |
| --- | --- |
| Unknown configured model, tool result only | Gemini → Gemini |
| Unknown configured model, tool result plus operator event | Gemini → Antigravity |
| Gemini configured for all tiers, tool result only | Gemini → Gemini |
| Gemini configured for all tiers, tool result plus operator event | Gemini → Gemini |

Thus a correctly resolved Gemini selection survives these continuations. The
successful initial request in the reported incident was an accidental default,
not evidence that an agent-specific Gemini override had been saved.

The subsequent `opencode-go` fallback also fails: its requests receive
`MissingSessionID` because `x-opencode-session` is absent. That is separate
from Google's multi-turn rejection and is not a successful recovery.

Relevant source:

- [Google override application](/Users/marcoschwartz/Documents/code/core/provider.go:468)
- [Catalog-gated SetModel](/Users/marcoschwartz/Documents/code/core/provider_google.go:120)
- [External-event tier selection](/Users/marcoschwartz/Documents/code/core/thinker.go:2628)
- Server `instances.go`: `applyAgentModelOverride` maps a persisted agent
  selection to every tier; `configuredAgentModelOverride` reads its metadata.

The appropriate agent configuration correction is to save
`model_override = {"provider":"google","model":"gemini-3.6-flash"}` through
the server's model-selection path. Separately, Core should apply or reject
configured Google model IDs consistently across tiers. Correcting the map alone
would not make Antigravity accept multi-turn input; the agent also needs a
conversation-capable configured model.

## Realtime: renewal works; the close cause is unresolved

The readiness session uses `google-realtime`,
`gemini-3.1-flash-live-preview`, and `Kore`, independently of the text provider's
large/medium/small map. Its saved history contains only assistant utterances.
Telemetry has no caller transcript events. Provider usage includes audio-input
tokens, so absence of transcripts should not be interpreted as proof that no
audio bytes reached Google.

Observed sessions lasted approximately 153–162 seconds before closing with
1008 and “The operation was aborted.” Telemetry records successful unplanned
reconnections and new session generations with `restored=true`. This establishes
that Core renews the connection; it does not establish why Google closes it,
that the user heard the greetings, or that caller speech was understood.

Lifecycle review:

- `RealtimeThinker.openSession(true)` opens the same model/voice and restores
  bounded text transcript history.
- The run loop remains alive when a provider session closes. Failed reopen
  attempts use exponential backoff. Successful reopens reset that backoff;
  repeated established-session failures currently have no cumulative retry cap.
- The Google adapter recreates sessions and seeds transcript history. Its
  parsed `goAway` and session-resumption-update fields are not acted on; it
  does not use a native resumption handle here.
- The explicit initial-message path avoids replaying the greeting after audio
  has been emitted. That behavior passes its local regression test. Repeated
  assistant greetings in the saved history do not by themselves prove that
  this particular path resent an initial message.

The available close reason is generic. Idle input, provider behavior, and
client/protocol interactions remain hypotheses. A longer instrumented audio
session spanning one of these closes is needed to attribute the cause and
validate continuity across it. The short test below does not do that.

## Fresh verification

The focused local run passed 18 top-level tests, including the four model
selection probes, Google setup/tool-response serialization, session shutdown,
reconnect/history restoration, and greeting replay protection. Three opt-in
live tests were skipped in that local run.

Separately, `TestRealtimeLiveReceptionistMCP/google-realtime/fr/Kore` ran live
with the project's existing Gemini connection and **passed in 22.20 seconds**:

- Four assistant response turns in French.
- Availability lookup and booking through two successful local fake MCP calls.
- Correct continuation after both asynchronous tool results.
- 706,172 PCM audio bytes emitted.

Caller turns were supplied as synthetic text to the realtime thread; the model
produced real speech audio. This verifies current multi-turn realtime/tool
behavior, and does not claim microphone, PSTN, or long-session reliability.
Earlier saved receptionist artifacts also contain multi-turn caller/tool
interactions; the greeting-only readiness session does not contradict them.

## Phone route

The local Telephony database confirms route
`route-25f4e4990915d394978b625447fa9d70` targets agent 1102, is enabled, and uses
`realtime_immediate` with `programmable_websocket`. Its saved transport
configuration is empty, including no saved Telnyx application ID. Carrier
provisioning and inbound delivery remain unverified. No phone call was placed
and the remote Telnyx configuration was not changed or independently queried.

## Evidence

- [Redacted findings and source hashes](../test-artifacts/flexylead-investigation-2026-09-10/summary.json)
- [Local reproduction and lifecycle tests](../test-artifacts/flexylead-investigation-2026-09-10/local-tests.log)
- [Overlay-only reproduction source](../test-artifacts/flexylead-investigation-2026-09-10/model-probe.go.txt)
- [Fresh live test log](../test-artifacts/flexylead-investigation-2026-09-10/google-live.log)
- [Fresh live conversation transcript](../test-artifacts/flexylead-investigation-2026-09-10/live/google-realtime_fr_Kore_transcript.txt)

The original instance log and runtime configuration contain sensitive material;
they were inspected locally and were not copied into these artifacts.
