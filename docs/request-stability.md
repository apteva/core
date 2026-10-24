# Stable provider requests

Core retains bounded automatic tool selections across follow-up messages. It
revalidates live catalog identities, grants, `no_spawn`, and schema budgets on
every request. Small acknowledgements do not trigger unrelated discovery.
Meaningful requests can add schemas within the automatic budget; explicit
`search_tools` remains available even when that budget is full. Explicitly
activated tools retain their separate bounded LRU. Batch eviction protects
pending calls and required execution prerequisites.

Provider cache identity is independent of diagnostic epochs, instructions and
tool hashes for scoped Thinker requests. It remains partitioned by provider,
model, agent/thread, deployment namespace, and the Codex account when available.
Unscoped direct provider callers retain content-based partitioning. Exact
provider prefix matching remains responsible for cache correctness. A changed
key version causes one migration boundary; it does not invalidate or alter an
agent's history, actions, or permissions.

Subscription adapters can declare a stable session-routing header. Codex uses
`session_id`, with the existing per-thread/purpose session identity; no shared
provider instance is mutated. Public OpenAI and other adapters do not receive
that header. Account credentials never enter telemetry. Serialized request
budget events include only a hash of the actual cache key.

## Context observations

Request-only context updates preserve their prior positions while new messages
and tool results append. Memory, roster, wake state and pacing are independently
versioned sections. Unchanged sections are not repeated. Empty or omitted
sections explicitly supersede their previous values; observations cannot grant
capabilities or act as user instructions. The latest clock and section value
apply. These updates are never written into durable user history.

The ledger checkpoints at 16 updates or 128 KiB, retaining a full current
snapshot. An individual current snapshot is never truncated to meet the ledger
budget. Existing history/pressure compaction remains active. Memory-store
generation changes clear the request-only ledger so withdrawn or changed
memory is not retained just for caching. Manual resets, directive changes and
history recovery retain their existing invalidation boundaries. Current images
and tool/result pairing continue through the existing projection path.

`llm.prompt_cache_reset` is a local diagnostic boundary, not a provider cache
deletion command. It reports `provider_identity_rotated`; the default is false.
Cache misses never trigger tool re-execution or an extra inference warm-up.

## Rollback switches

The fixes are enabled by default in this Core build. Set any independent switch
to `0` to restore that component's earlier behavior:

- `APTEVA_STABLE_TOOL_SELECTION`
- `APTEVA_STABLE_CACHE_IDENTITY`
- `APTEVA_APPEND_REQUEST_CONTEXT`
- `APTEVA_PROVIDER_SESSION_ROUTING`

These are Core process settings; this work does not deploy or restart agents.

## Verification

`request_stability_test.go` covers short follow-ups in multiple languages, new
automatic and explicit discovery, full budgets, revocation, catalog changes,
disconnection, worker restrictions, cache isolation, context updates/retractions,
memory withdrawal, byte/count checkpoints, screenshots/results, pending-tool
protection, rollback, and concurrent provider sessions. Existing recovery,
vision, tool lifecycle and authorization suites remain release checks.

Paid tests use **gpt-6.1-sol** and existing runtime credentials:

```sh
RUN_SOL_CACHE_STABILITY=1 go test -v -count=1 -timeout 8m \
  -run '^TestIntegration_SolCache' .
RUN_CODEX_CACHE_SMOKE=1 CODEX_CACHE_SMOKE_MODEL=gpt-6.1-sol go test \
  -run '^TestIntegration_(CodexPromptCacheSmoke|CacheHitRatio_OpenAICodex)$' -v .
```

The synthetic endpoint control is reported separately from actual managed
workers. The workers use unpadded Core prompts, the normal Run loop, two isolated
threads, ten external turns per thread, explicit discovery, exact verified
results, model reasons, and sixteen uniquely executed fixture reads. Each
unchanged-manifest/epoch window must preserve the exact previous Responses
input items. The required token-weighted warmed cache rate is at least 80% for
each worker. All-call rates include cold starts, discovery and compaction and
are also reported. Unexplained misses remain in the warmed denominator.
Benchmarks start at most one request every five seconds per thread/key, below
the provider's documented cache-routing overflow guidance. This test-only
cadence is disclosed; the production thinker and provider adapter are not
throttled by it. Cache reuse under burst traffic can be lower.

The correction test verifies that real Sol follows updated/cleared context
without taking actions. Provider cache percentages are measured outcomes,
not guarantees for arbitrary production traffic. They do not replace checking
task completion, duplicate operations, latency and total input usage.

For an optional diagnostic run, add `RUN_SOL_CACHE_DIAGNOSTICS=1` to the managed
thread test. This asks the endpoint to compare consecutive response IDs, logs
only hashed routing/cache identities and the diagnostic result, and buffers
responses in the test transport. It does not change production requests.
The Codex endpoint accepted this field but returned `unavailable` in the local
verification; that result cannot explain a miss or establish a cache hit.

## Local verification results (October 23, 2026)

The latest ordinary paced release battery measured **99.14%** warmed reuse in
the fixed-prefix control and **94.74%** in the four-turn direct conversation.
The two managed workers measured **82.70%** and **78.84%** warmed reuse; including
cold starts, tool additions and the history checkpoint, their all-call rates
were **75.18%** and **71.85%**. Thus the **80% per-worker gate still failed**.
An earlier ordinary run measured **95.29% / 84.24%** warmed and
**86.03% / 80.34%** across all calls. These are individual runs, not a production
guarantee. Earlier failing runs remain part of the evidence.

The optional diagnostic run passed the warmed gate at **84.70% / 83.61%**
(all calls **76.87% / 75.95%**), but this does not establish repeatability.
An isolated test of relaying the endpoint's opaque turn-state response header
did not consistently improve reuse and was not adopted in production code.
No cache performance threshold was removed, and misses remain counted.

All managed runs completed the fixture operations and preserved the checked
prefixes. The full short suite, focused race suite, real context correction
test and real screenshot retention test passed. The cache performance gate
remains a release limitation independent of those correctness checks.

Raw logs and machine-readable results are saved locally under the ignored
`test-artifacts/cache-stability/` directory. No services were deployed or
restarted by this work.
