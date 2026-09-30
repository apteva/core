# Agent memory

Core retains the existing append-only memory journal and cached, indexed
automatic recall. The background unconscious now consolidates eligible retained
thread histories, not just the last 50 lines of main.

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

The opt-in suite pins every model tier to `gpt-6-sol` and uses the existing
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
