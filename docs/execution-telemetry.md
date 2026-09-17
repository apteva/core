# Execution telemetry

Detailed timing events extend the existing telemetry contract. No provider
payload, scheduling policy, retry policy, tool routing, or event delivery
semantics change. Existing `llm.start`, `llm.done`, `llm.error`, `tool.call`,
`tool.result`, `thread.spawn`, and `thread.done` events remain available.
Inline `tool.result.duration_ms` now measures elapsed time instead of always
reporting zero.

## Correlation

The original envelope still carries agent (`instance_id`), thread, event ID,
and wall-clock time. Additional IDs are inside `data`, so older servers retain
them without a schema migration:

- `worker_run_id`: one runtime incarnation; distinct from the reusable thread ID.
- `turn_id`: one inference turn, shared across its retries and fallbacks.
- `request_id`: one provider `Chat` attempt; `attempt` counts attempts within
  the turn and `role` identifies primary/fallback selection.
- `http_attempt_id`: one physical HTTP submission, including adapter-local
  authentication/options retries within a `Chat` attempt.
- `tool_span_id`: one tool invocation; the existing `id` remains the original
  provider tool-call ID. Reused `call_0` values therefore remain distinguishable.
- `execution_ids`: tracked incoming workflow executions, when present.
- Child runs carry `parent_worker_run_id`, `parent_request_id`, and, when
  created by a tool, `parent_tool_span_id`.

Async tools capture their origin at admission. Later model turns cannot
relabel their results. IDs that do not apply, such as a request ID before
the worker's first inference, are empty; an untracked event does not acquire
a fabricated workflow execution ID.

## Events and timing

| Event | Meaning |
| --- | --- |
| `llm.request.queued` | Provider attempt entered preparation. |
| `llm.request.started` | Inference capacity acquired; entering provider Chat. |
| `llm.request.first_output` | First nonempty reasoning/text/tool callback, once per output kind. |
| `llm.request.finished` | Attempt returned, failed, timed out, or was cancelled, including pre-dispatch failures. |
| `llm.http.started` / `headers` / `finished` | Physical submission, response headers, and body closure or transport failure. |
| `llm.retry.scheduled` / `finished` | Planned backoff and actual wait, including interrupted waits. |
| `tool.execution.queued` / `started` / `finished` | Admission, execution after capacity wait, and terminal outcome. |
| `worker.created` | Creation accepted, before setup and launching the loop. |
| `worker.ready` | Runtime loop entered; deferred workers have not reached this milestone. |
| `worker.first_inference` | First provider Chat dispatch. |
| `worker.first_action` | First tool execution attempt, attributed to its originating request and tool span. |
| `worker.finished` | Completed, failed, or cancelled; emitted once per run. |

Model `preparation_ms`, `queue_ms`, and `duration_ms` separate request
preparation, inference-capacity wait, and provider Chat time. `total_ms`
covers the entire attempt. First-output offsets are relative to Chat dispatch,
not the first SSE frame: empty metadata frames are not model output.
Tool `queue_ms` includes admission and capacity wait. Worker `elapsed_ms`
starts at creation. All durations use monotonic elapsed time.

Request completion includes provider/model, input/output/cache usage, and
existing provider timing diagnostics. `usage_reported=false` means core did
not receive nonzero usage; zero counts on failed calls are not evidence of
zero provider billing. A transport `closed` outcome only means the HTTP body
closed; use `llm.request.finished` for the parsed model outcome. HTTP records
contain status and provider request IDs, never headers or bodies.

Streaming callbacks define observable output. Providers that return output
without streaming callbacks have no invented first-output timestamp.
Realtime sessions retain their existing `realtime.*` session/usage events;
the request events cover request/response inference, while shared worker and
tool instrumentation also applies to realtime workers.

## Compatibility and aggregation

Continue calculating existing usage/cost/call totals from `llm.done` and
existing realtime usage events. Do not count detailed request or HTTP records
as additional billed calls. Legacy `llm.done.duration_ms` remains the turn
duration, including retries; its emission still follows tool handling.

Do not add nested durations (`total_ms` plus its component durations), HTTP
attempts plus their containing Chat duration, or parallel worker/tool times
to calculate wall-clock latency. Correlate intervals to determine the critical
path. Raw event counts and storage volume naturally increase; the existing
dashboard work timeline ignores the new diagnostic event types.

New timing events use the existing buffered telemetry forwarding path.
They have the same delivery guarantees as ordinary telemetry, not an fsync
per model chunk. A crash can leave an open interval. Existing durable terminal
tool receipts retain their previous durability behavior.

## Verification

- `go test -short ./...`
- `go test -race -short -run 'TestExecutionTelemetry|TestThreadProfile|TestCallLLMWithRetry|TestTelemetry' .`
- `RUN_CODEX_EXECUTION_TELEMETRY=1 go test -run '^TestCodexExecutionTelemetry$' -v .`
- Server: `go test -short -run 'TestExecutionTelemetryAdditiveCompatibility|TestTelemetry' .`
- Dashboard: `bun test src/components/AgentView.executionTelemetry.test.ts`
