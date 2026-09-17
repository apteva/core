# OpenCode Go session routing

Core sends `x-opencode-session` and `User-Agent: apteva-core/<version>` on
OpenCode Go requests. This fixes the provider's HTTP 400 `MissingSessionID`
response, including when switching to `kimi-k3`.

The opaque session ID is derived from the instance working directory, server
URL, agent identity, thread identity and request purpose. It survives process
restarts at the same instance location, model changes, retries and prompt-cache
resets. Separate threads, deployments and compaction purposes get separate IDs.
Moving an instance directory or changing its server URL starts a new routing
identity; this does not alter its saved history.

Standalone provider callers without a Thinker context receive a random ID for
the lifetime of that provider instance, shared by its reasoning/builtin clones.
Callers representing multiple conversations should supply separate session
contexts. The shared HTTP transport does not add these headers to other providers.

No new configuration, credential storage or Server API is required. Rebuild
Core and use the new binary to activate the fix. Existing running binaries do
not acquire it automatically. HTTP 402 billing failures are also classified as
non-retryable within an inference attempt; this does not restore provider credit.

Validation:

```sh
go test -short -run 'Test(OpenCodeGo|ProviderSession|ProviderBilling)' .
go test -race -short -run 'Test(OpenCodeGo|ProviderSession|ProviderBilling)' .
```

With `OPENCODE_GO_API_KEY` supplied securely in the environment:

```sh
RUN_LLM_INTEGRATION_TESTS=1 go test -run '^TestIntegration_OpenCodeGo_KimiK3SessionToolRoundTrip$' -count=1 -timeout 4m -v .
```

The live test reviews a Go function, receives an `inspect_source` tool call,
returns fixture source as a structured result, and verifies the corrected
function on the next request. It does not execute model-generated code.
