# Realtime MCP tool visibility

Text requests and realtime sessions use `Thinker.visibleNativeToolSnapshot`.
The resolver receives the complete exact-tool grants and MCP scopes, filters
activation and retention against those grants, and merges the loading baseline:

| Loading policy | Visible without discovery |
| --- | --- |
| `always` | Yes, within the authorized scope |
| `auto` | When the global catalog uses eager loading |
| `deferred` | No; requires activation or an explicit exact-tool grant |

An attached MCP does not itself grant a worker access. Whole-MCP scopes, exact
grants, `no_spawn`, platform bypass and system-only restrictions still apply.
Baseline tools do not enter the activation LRU. Required execution tools and
unexpired discovery retention use the same authorization checks.

Realtime session setup, reopening, refresh and configuration previews share
this resolver. A preview uses the proposed grants and scopes before committing
an update. Immutable providers still require an explicit restart for a parent
update during a connected call.

ToolIndex revisions broadcast to realtime loops through a shared channel,
without polling or a background goroutine per subscriber. Catalog changes
refresh mutable providers. An idle immutable provider is reopened immediately;
an active provider response keeps its original tool manifest until it finishes,
then the session refreshes or reopens.
Reopening rebuilds tool schemas from current policy and permissions. Successful
live updates also update the local schema list used by realtime output guards.

## Regression tests

`realtime_tool_visibility_test.go` covers loading modes, exact and server grants,
unauthorized/stale activation, no_spawn and platform bypass, main-thread policy
overrides, discovery retention, HTTP MCP registration and calls, proposed
permission previews, revocation, schema replacement, session refresh/reopening,
live catalog notifications and serialized Google/OpenAI callable names.

Run the live Google integration explicitly:

```sh
RUN_GOOGLE_REALTIME_MCP_SMOKE=1 GOOGLE_API_KEY=... \
  go test -count=1 -v -run '^TestGoogleRealtimeLiveMCPThread$' -timeout 3m .
```

It connects a local mock HTTP MCP to a real Google Live session, forces discovery
mode, grants only the MCP scope, and sets `tool_loading.default=always`. It
requires exactly one correct MCP call, a successful result, the spoken marker
in the output transcript, and nonempty PCM audio. The fixture has no production
booking side effects. An exact tool grant would mask the original bug and is
explicitly rejected by this test.
