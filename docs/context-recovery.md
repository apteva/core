# Context budgeting and overflow recovery

Core checks a request after selecting its native tools, then checks the exact
serialized body inside the Responses, OpenAI-compatible, Anthropic, and Gemini
adapters before HTTP submission. Requests include system instructions, history,
structured calls/results, provider response state, images, and tool schemas.

`llm.request_budget` records provider/model, preparation stage, SHA-256 request
fingerprint, serialized byte length, token estimates, output reservation,
context window, safe input budget, and whether submission was blocked. At the
`serialized` stage the fingerprint and bytes describe the actual HTTP body.
Prompts, media, and credentials are not included in these diagnostic events.

Token counts are estimates, not tokenizer measurements. Image bytes are not
counted as ordinary text; opaque provider state is conservatively accounted
for. The input budget subtracts a 10% margin and the adapter's output limit,
or a 16,384-token reservation when no explicit output limit is sent. Unknown
models use the existing 512 KiB text fallback expressed as 131,072 tokens.

## Recovery

A preflight overflow or recognized provider context-limit error triggers
recovery instead of an unchanged transient retry or immediate fallback:

1. Archive the complete rejected request before reducing it.
2. Project oversized machine payloads and older opaque response state while
   preserving ordinary messages, instructions, and in-flight native calls.
   Structured JSON keeps small fields and exact numeric identifiers. Compact
   structured state notes survive subsequent tool-result aging and restart.
3. If necessary, summarize older history in bounded batches of complete
   messages. Preserve current instructions, the latest external request,
   existing continuation summaries, request-only context, pending calls, and
   tool call/result boundaries. Empty/failed summaries never trigger blind
   deletion.
4. Require at least a 5% reduction (and at least 128 estimated tokens) before
   retrying. Every candidate is budget-checked again before submission. At
   most three recovery passes run within the existing five-minute request
   deadline.
5. Archive the complete previous session journal, including history outside
   the live tail, and atomically checkpoint the reduced durable context.
   Consumed event IDs survive the checkpoint. Session revisions invalidate
   obsolete background summaries without waiting for their model calls.

Archives use the existing private, content-addressed tool-result archive and
its size/quota limits. Request and journal references are recorded in recovery
diagnostics; payload receipts and state notes retain references in context.
Archive/checkpoint failure leaves the original live context intact.

If no safe reduction fits, the execution fails with `context_management_failed`
and the underlying cause. Core does not retry an identical rejected request
or clear the thread. Summaries remain model-generated: original archives are
retained for inspecting details omitted from the working context.

## Provider fallback

Primary and fallback errors retain separate identities and are separately
observable through `llm.provider_error`; combined errors preserve both causes.
A permanent fallback failure is attempted once per retry cycle and does not
prematurely stop recovery of a transient primary failure. A context-limit
failure is handled on the affected provider first. Core cannot repair an
invalid provider credential; configuration must be corrected separately.

## Regression checks

`context_recovery_test.go` covers wire-level context rejection, blocked oversized
requests/tools/instructions, state and journal preservation, restart, structured
JSON and 64-bit IDs, in-flight work, background compaction coordination, and
fallback authentication failures.

```sh
go test -short ./...
go test -race -short ./...
RUN_CODEX_CONTEXT_RECOVERY=1 go test -run '^TestCodexContextRecoveryPreservesReleaseState$' -v -timeout 7m .
```

The opt-in test uses saved Codex authentication and `gpt-5.6-terra`. It injects
the first context rejection; subsequent summarization and continuation calls
are real Codex requests using synthetic local state, with no external actions.
It checks bulky tool output, semantic history reduction, and structured release
state beside large image/document fields.

## September 8 Lily incident

Read-only production telemetry for agent 13 showed that September 7 compaction
reduced 144 messages to 23, but its own token estimate only fell from 487,155 to
399,102 against a configured 272,000-token window. Core submitted that oversized
result anyway. September 8 attempts at 07:40:01 and 07:40:09 UTC reused the
oversized retained history without another compaction event; both primary
requests failed with `context_length_exceeded` and Anthropic fallback failed
with invalid-key HTTP 401.

The previous preflight required enough removable old messages and enough
estimated benefit from a fixed 20-message tail. It did not enforce that the
result actually fit, omitted schemas and opaque Responses state from its
estimate, and did not classify context rejection as requiring a changed
request. Tool-result aging also advanced only after successful responses, so
rejection did not make the oversized retained results eligible for aging.
Historical logs did not contain exact serialized request sizes/fingerprints;
those are now recorded for future diagnoses.

No production agent state, credentials, or server source was changed by this
implementation. Deploying the updated core and repairing fallback credentials
are separate operational steps.
