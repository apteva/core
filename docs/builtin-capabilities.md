# Provider-hosted builtins

Core supports one opt-in configuration for capabilities executed by the model
provider. Each provider adapter translates the capability name and options into
its native API tool declaration. Once enabled, the model decides when to use it
from the conversation; Core does not execute a local MCP/function tool for it.

Configure the provider in Core's config file or through `PUT /config`:

```json
{
  "provider": {
    "name": "openai-codex",
    "builtins": {
      "image_generation": {
        "enabled": true,
        "options": {
          "model": "gpt-image-2.5-flare",
          "quality": "low",
          "output_format": "png"
        }
      }
    }
  }
}
```

In the config file, providers can also be entries in the `providers` array.
New capabilities are disabled by default. Set `enabled: false` to disable an
entry. A single-provider API update merges the named capabilities and replaces
each supplied capability's entire options object; it preserves unspecified
capabilities. Supplying the full `providers` array replaces the provider config.

## Current adapters

| Provider | Capability | Allowed option keys |
| --- | --- | --- |
| `openai`, `openai-codex` | `web_search` | `search_context_size`, `user_location`, `filters`, `external_web_access` |
| `openai`, `openai-codex` | `code_execution` | `container` (defaults to `{"type":"auto"}`) |
| `openai`, `openai-codex` | `file_search` | `vector_store_ids` (required), `max_num_results`, `filters`, `ranking_options` |
| `openai`, `openai-codex` | `image_generation` (experimental) | `model`, `size`, `quality`, `output_format` |
| `anthropic` | `web_search` | `max_uses`, `allowed_domains`, `blocked_domains`, `user_location` |
| `anthropic` | `code_execution` | None |
| `google` | `web_search`, `code_execution` | None |

For example, OpenAI API web search can be enabled with:

```json
{"provider":{"name":"openai","builtins":{"web_search":{"enabled":true,"options":{"search_context_size":"low"}}}}}
```

Options follow the selected provider's tool contract; identical names do not
imply identical options. Core rejects unknown capabilities/options, duplicate
aliases, non-JSON values, and missing file-search vector stores before applying
a config update. The provider validates other option values and availability
for the selected model/account. Adapter support does not guarantee access to
every capability on every model or Codex subscription route. There is no
automatic fallback that strips a rejected tool or changes billing routes.

`GET /config` includes `builtin_capabilities`, keyed by configured live provider
name. Entries expose the capability name, API tool type, effective enable flag,
allowed option keys, and experimental status. Compatible-chat, Grok Build, and
realtime adapters do not support the new configuration in this rollout.

## Compatibility and child threads

The existing `builtin_tools` list continues to work. Generic entries override
matching legacy entries, including an explicit disable. `code_interpreter`
aliases `code_execution`; `web_search_preview` and `google_search` alias
`web_search`. Legacy OpenAI `web_search_preview` retains its preview wire type;
the generic entry uses `web_search`.

The existing `image_generation` config remains accepted. Its flag is applied
first; a generic `builtins.image_generation` entry takes precedence. Listing
the image tool in legacy `builtin_tools` alone still cannot enable it.

Provider and reasoning clones preserve options without sharing mutable JSON
objects. A child with no builtin override inherits them. An explicit empty list
disables them; a selected subset preserves the parent's options. Explicitly
disabled generic entries cannot be enabled by a child's name list, and image
generation always requires the provider-level enable flag. The legacy name-list
override semantics for other tools remain compatible.

Image generation still uses the Server gateway and unified `blobref://` file
handles. Core retains only references and metadata; the new config does not
introduce image byte fetching or storage. See
[native-image-generation.md](native-image-generation.md).

Server and Dashboard also support the generic map. In Providers settings,
Built-in capabilities configures defaults on the existing connection. Agent
creation and editing offer Inherit / On / Off and provider-specific options.
Nothing is enabled automatically by installing this support.

## Server defaults and agent overrides

Provider defaults live in `connection.runtime_config.builtins`. Update through
`PATCH /api/connections/:id/runtime-config`, for example:

```json
{"builtins":{"web_search":{"enabled":true}}}
```

Agent overrides are server-owned metadata, keyed by provider. Update through
`PUT /api/agents/:id/config`:

```json
{
  "builtin_overrides": {
    "openai-codex": {
      "web_search": {"enabled": false},
      "image_generation": {"enabled": true, "options": {"quality":"low"}}
    }
  }
}
```

An omitted capability preserves its setting. A null capability removes the
agent override and restores inheritance; a null provider removes all its
agent overrides. Explicit false remains off even if the provider default is
on. Omitted agent options inherit provider options; an explicit options object
replaces them completely, including an empty object. On provider-default
patches, each supplied capability replaces its previous options, matching the
Core single-provider update contract. Settings never transfer across providers.

Server resolves defaults and overrides at creation, startup, live edits, and
stopped-config reads, and forwards the result as Core `providers[].builtins`.
Legacy direct provider `builtins` are treated as overrides and migrated on edit;
legacy `builtin_tools` and `image_generation` remain accepted. Enabled capability
summaries on agent cards identify the originating provider; they do not certify
account or model access.

Provider default saves apply to running agents, including Helper. The response
and connection runtime configuration expose `builtin_sync.pending` and `errors`
when a runtime cannot accept the update. Settings offers retry; saved defaults
also apply on the next restart. Existing app/MCP attachments are preserved.

## Credential-free discovery

The installed Core binary supports two side-effect-free commands:

- `apteva-core --version --builtin-capabilities`: versioned adapter descriptors
  with option types, enums, required fields, defaults and experimental flags.
- `apteva-core --version --validate-builtins`: reads a JSON object containing
  `provider` and `builtins` on stdin and returns canonical `builtins` or `error`.

Both commands return before Core runtime initialization. The leading `--version`
means older binaries safely print their version instead of starting an agent.
Server caches discovery by binary identity, exposes it on runtime connection
summaries, and delegates configuration validation to the same adapter code.
Live `/config` descriptors include the same option metadata. Older installed
or running Core versions produce an upgrade/restart message rather than a
successful response that silently drops the settings. Deployment needs the
updated Server, Dashboard assets, and Core binary together.

## Verification

The short suite covers native request shapes, options, enable/disable precedence,
cloning and child selection, config persistence, discovery, invalid-update
rollback, and Gemini hosted code results. The opt-in image live test now enables
images using `builtins.image_generation` and verifies real generation, reference
persistence, and an actual model passing the unchanged handle to a file tool.

Run from `server/`:

```sh
RUN_NATIVE_IMAGE_GENERATION_LIVE=1 go test -v -count=1 \
  -run '^TestNativeImageGenerationLive$' -timeout 6m .
```

On 2026-10-03, the generic configuration passed the real Codex + GPT Image
2.5 Flare test: it generated a 790,579-byte PNG and completed the unchanged-ref
file-tool handoff and continuation. Core's full short suite and the targeted
builtin race tests also passed.
