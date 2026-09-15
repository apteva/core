# Kimi routing and pinned-model experiments

OpenCode Go now requires `x-opencode-session` on chat requests. The routing
identity is stable per deployment, agent, conversation and purpose, independent
of model switches and prompt-cache epochs. Auxiliary compaction conversations
use separate identities. Standalone provider instances receive a random identity
shared by their clones. Required headers survive optional-parameter retries.

Bounded experiments may set `disable_provider_fallback: true` using Core's
`PUT /config`. The setting persists and is shared with workers. Core's config
response reports its effective value. The default remains false, preserving
existing provider fallback behavior. A runner should verify the flag and model
pin before releasing its first inference checkpoint.

This flag disables provider switching, not same-provider transport retries.
Use an execution gate plus an external wall-clock timeout to bound a review.

Validation includes mock HTTP session routing/retry tests, worker and auxiliary
identity isolation, existing fallback behavior, explicit disabled fallback,
execution checkpoint tests, focused race tests and the full short suite.
