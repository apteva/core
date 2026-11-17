# Agent memory

Core retains the existing append-only memory journal and cached, indexed
automatic recall. The background unconscious now consolidates eligible retained
thread histories, not just the last 50 lines of main.

## Learning from experience

The existing memory thread also consolidates conditional working lessons:
when an approach applies, what was attempted and observed, which alternative
has support, what remains uncertain, and when the evidence was confirmed.
These are ordinary memory content and free-form tags, using the same
`review_history`, search, remember, supersede and drop tools. There is no
additional learning thread, journal, evaluator, foreground reflection call,
model training, or automatic publication of private lessons.

Main, worker, managed-execution and realtime prompts tell the agent to use
relevant experience, adapt to observed failures within its existing authority,
and keep uncertain patterns revisable. Inferred lessons must not be retained
by calling `evolve` or `update`; directive edits still require authoritative
owner/parent instructions. The editable directive itself is unchanged.

Consolidation keeps bounded result metadata alongside retained text: call ID,
tool name, execution error status, and whether the result was previewed.
This metadata is limited to 4 KiB per entry and charged against the review
batch budget. Existing source hashes remain stable across this projection
upgrade. Tool arguments, reasoning, provider state and media remain excluded.
A tool execution completing is not proof of an appointment, sale or other
business outcome. Pending feedback remains unknown, and retries, duplicate
delivery, assistant claims and repeated recall are not independent evidence.

A new technical tool failure can refresh the existing automatic recall cycle
using a bounded query from up to three error results. Identical failures on
retries keep the selected block stable. External requests clear that failure
query; ordinary successful tool continuations retain the current memory block.
Business rejection in a successful tool result is interpreted from historical
evidence by the memory thread, rather than a new hardcoded domain classifier.

Background learning requires the existing `unconscious` setting. It is limited
to retained histories and existing source-thread memory scopes. A cold context
for the same owner can recall its durable lessons; an unrelated conversation
cannot inherit them automatically. Existing explicit management publication,
exclusions, forgetting and provenance rules apply unchanged.

Real Sol 6.1 tests use the ordinary Thinker, native tool dispatcher, automatic
recall and auto-spawned memory thread. Six training episodes produce actual
model-selected attempts against a controlled service fixture with two hidden
profile-specific preferences. Twelve paired cold-context assessments compare
the persisted model-written memories against an empty memory store. Assessment
directories contain no training transcript, and directive/tool definitions are
identical in both arms. The gate requires at least 10/12 learned successes and
at least four additional successes over the control. Separate live cases cover
contradictory evidence, pending/duplicate outcomes, source isolation and
forgetting. This measures behavioral reuse of experience in a synthetic
environment; it does not establish real-world business conversion gains.

```sh
GOWORK=off RUN_CODEX_EXPERIENCE_LIVE=1 go test -v -count=1 \
  -run '^TestIntegration_SolExperience' -timeout 25m .
```

Local Sol 6.1 verification on October 23, 2026 completed two independent
training/consolidation/comparison runs:

| Run | Empty memory control | Model-written experience memory |
| --- | --- | --- |
| First | 2/12 accepted requests | 12/12 accepted requests |
| Repeat | 3/12 accepted requests | 12/12 accepted requests |

Across completed runs this is 5/24 versus 24/24. Each assessment starts with
fresh history for the same permitted owner, and the treatment reloads only the
durable memory journal. This comparison isolates learned memory within the new
Core build; it is not an old-binary/new-binary comparison or a real-world
conversion-rate estimate. An intermediate repeat was interrupted by upstream
overload. The harness now lets Core's existing bounded overload retry finish
instead of treating each attempt's `llm.error` as terminal; thresholds and
operation limits remain unchanged. The interrupted run is preserved with the
other evidence under ignored `test-artifacts/memory-experience/`.

The final four-case experience battery passed, including revision, outcome
grounding, isolation and forgetting. The full short suite, focused race checks
and Core build also passed. Running the older memory suite on Sol 6.1 exposed a
progress-only answer: recall found the correct fact, but the agent idled before
delivering it. The common memory-use contract now requires the requested fact
in the specified response channel; the unchanged three-turn live answer test
passed on recheck. The other four older live memory cases passed as well.

## Consolidation and recovery

When `unconscious` is enabled, `review_history` returns one thread's batch at a
time: at most 50 entries and 48 KiB of projected text, with 8 KiB previews per
entry. Threads are visited round-robin after each committed batch.

The protocol is read -> grounded memory writes -> commit the returned
`batch_id`. Commit even if there are no worthwhile facts, but never commit
in parallel with writes. Reads and restarts replay a persisted pending batch
until commit. A failed checkpoint write does not advance the in-memory cursor.
Writes use deterministic identities to avoid duplicate remembers on replay.

`memory-checkpoints.json` is atomically replaced with mode 0600. Seen-entry
hashes survive history rewrites; new compacted summaries are distinct secondary
evidence. Partial trailing JSONL records wait for completion. Malformed complete
records/checkpoints produce visible errors rather than silently skipping input.
The built-in old unconscious directive is upgraded at restore; custom
directives must use the batch protocol.

Only user/assistant text, retained tool-result text, and compaction summaries
are projected. System messages, reasoning, provider state, binary attachments,
and tool arguments are excluded. Historical content is evidence, not new
instructions or authorization. User assertions, assistant claims, and tool
results retain their source roles.

## Scope and provenance

New background-learned records have an exact thread-ID `scope` and `sources`
containing source thread, entry hash, sequence (when available), and role.
Optional comma-separated `source_ids` on writes selects evidence from the
pending batch; otherwise all its entries are attached. Core, not model-supplied
scope arguments, determines ownership.

Automatic recall filters scope before ranking, deduplication, and top-k.
The background thread can read matching shared records but cannot mutate them
or another thread's records. Supersedes preserve accumulated source lineage.

Existing records with no scope remain agent-wide for backward compatibility.
They cannot be retroactively attributed to a source. Authenticated management
clients can use `POST /memory` with `scope: "chat-id"` for private records,
or `scope: ""` to explicitly share. Omitting scope preserves an upsert's
existing scope; new unscoped API inserts remain shared. GET exposes scope and
provenance without embeddings. Existing index/ID update and delete routes remain.

## Explicit history fallback

Ordinary threads receive the read-only `history_search(query)` Core tool.
Core binds caller identity at dispatch; the tool cannot read a sibling or
parent chat by passing an ID. It searches all retained projected text in the
caller thread and returns at most five bounded, source-referenced snippets.
It makes no embedding or LLM request and supports cancellation.

Search does not recover originals removed by compaction or hydrate archived
tool payloads. It is lexical, not semantic. Memory consolidation is also limited
to retained history; these changes do not add a full transcript archive.

## Exclusions and forgetting

Persisted config (also GET/PUT /config):

```json
{"memory_policy":{"exclude_threads":["private-chat"]}}
```

System threads and the unconscious are excluded automatically. Exclusions
affect future consolidation/search, not memories already learned. Policies
are applied to pending batches too.

Authenticated `POST /memory/forget-source`:

```json
{"thread_id":"chat-id","reason":"user requested forgetting"}
```

This logically removes active records attributable to that scope or source,
including shared derivatives, and persists a deny marker preventing future
consolidation, history search, and memory writes from that source. Repeated
calls are safe. A memory reset preserves these deny markers and existing
review checkpoints. Resetting facts alone is not a ban on future learning.

This is not secure erasure: source transcripts, historical journal content,
checkpoints, backups, and responses already sent to models are not purged.
Records without provenance require explicit management deletion.

## Cost and bounds

Automatic recall retains its generation-aware cache and indexed top-k path;
there is no new foreground inference call. History search is an explicit
on-demand file scan. Background review scans retained files and keeps durable
seen-entry hashes; checkpoint size grows with reviewed entries. Initial
consolidation can process a larger backlog than the old last-50 implementation,
so background inference volume is not guaranteed to stay unchanged.

## Live Codex verification

The opt-in suite pins every model tier to `gpt-6.1-sol` and uses the existing
Codex test credential (environment or local auth file), without printing it.
All test facts, histories, journals, and checkpoints are synthetic and temporary.

```sh
RUN_CODEX_MEMORY_V3_LIVE=1 go test -v -count=1 \
  -run '^TestIntegration_CodexGPT6SolMemory$' -timeout 15m .
```

It covers lexical recall, three-turn ephemeral recall through the actual
thinker, LLM-driven two-thread consolidation with a simulated restart between
write and commit, scoped answers from the learned memories, on-demand history
search, and source forgetting in a fresh model context. A separate case uses
the production auto-spawned unconscious loop and native tool dispatcher to
verify both source threads are learned and checkpointed.

The restart case uses a bounded tool-loop harness with the real provider,
schemas, handlers, and Responses continuation state. It simulates restart by
reloading the journal/checkpoint; it does not kill a live operating-system
process. These are live smoke tests, not exhaustive model-quality evaluations.
