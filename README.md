# Apteva Core

The continuous thinking engine. Runs autonomous AI agents that observe, reason, act, and evolve — around the clock.

Core is a standalone Go binary. It can run headless, with its own TUI, or managed by [apteva-server](https://github.com/apteva/server).

## Architecture

```
┌─────────────────────────────────────────────┐
│  Main Thread (coordinator)                  │
│  Observes events, spawns/kills threads      │
└──────────┬──────────────────────────────────┘
           │
     ┌─────┴─────┐
     │  EventBus  │ ← never blocks, pub/sub
     └─────┬─────┘
           │
    ┌──────┼──────┐
    ▼      ▼      ▼
 Thread  Thread  Thread   ← permanent or temporary workers
    │      │      │
    ▼      ▼      ▼
  MCP    MCP    MCP       ← external tools (stdio or HTTP)
```

## Quick Start

```bash
# Set your API key
echo "FIREWORKS_API_KEY=your-key" > .env

# Build and run with TUI
go build -o apteva-core . && ./apteva-core

# Or run headless (API only)
./apteva-core --headless
```

Or use the [apteva CLI](https://github.com/apteva/apteva) which manages everything:

```bash
cd ../apteva && ./apteva   # spawns server + core + TUI
```

## API

Default port: `3210` (set with `API_PORT` env var)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/status` | GET | Uptime, iteration, rate, model, mode, threads, memory |
| `/threads` | GET | List all threads with state |
| `/threads/{id}` | DELETE | Kill a thread |
| `/events` | GET | SSE stream of telemetry events |
| `/event` | POST | Inject a message to a thread |
| `/config` | GET/PUT | Read/update config (directive, mode, MCP servers, computer) |
| `/pause` | POST | Toggle pause/resume |

## Core Tools

Always available to all threads:

| Tool | Description |
|------|-------------|
| `pace` | Set thinking speed and model size |
| `send` | Send message to another thread |
| `done` | Terminate this thread |
| `evolve` | Rewrite own directive |
| `remember` | Store to persistent memory |

Coordinator-only:

| Tool | Description |
|------|-------------|
| `spawn` | Create new thread with directive and tools |
| `kill` | Stop a thread |
| `update` | Change a thread's directive/tools |
| `connect` | Attach an MCP server at runtime |
| `disconnect` | Detach an MCP server |

Discoverable (RAG-retrieved when relevant):

| Tool | Description |
|------|-------------|
| `web` | Fetch a URL |
| `exec` | Run a shell command |

All tools support `_reason` — an optional observability field for explaining why the tool is being called.

## Safety Modes

No forced approval gates. The agent decides, learns, asks when unsure.

| Mode | Behavior |
|------|----------|
| `autonomous` | Acts freely. Learns from feedback. |
| `cautious` | Asks before risky actions. Learns from answers. |
| `learn` | Asks about every new tool type. Builds safety profile. |

Set via `PUT /config {"mode": "cautious"}` or the CLI `/mode` command.

## Session Persistence

Conversation history persists across restarts:

- JSONL files per thread: `history/main.jsonl`, `history/<thread>.jsonl`
- Last 50 messages loaded on startup
- Auto-compaction at 500 messages (keeps 100 recent + summaries)
- Thread history deleted on `[[done]]` or `[[kill]]`

## Providers

| Provider | Env Var | Native Tool Calling |
|----------|---------|-------------------|
| Fireworks (Kimi K2.5) | `FIREWORKS_API_KEY` | Yes |
| Anthropic (Claude) | `ANTHROPIC_API_KEY` | Yes |
| OpenAI (GPT-4) | `OPENAI_API_KEY` | Yes |
| Google (Gemini) | `GOOGLE_API_KEY` | Yes |

## Configuration

```json
{
  "directive": "Your mission here",
  "mode": "autonomous",
  "provider": {
    "name": "fireworks",
    "models": { "large": "accounts/fireworks/models/kimi-k2p5", "small": "accounts/fireworks/models/kimi-k2p5" }
  },
  "mcp_servers": [
    { "name": "myservice", "command": "./my-server", "main_access": true }
  ]
}
```

## Testing

```bash
go test ./... -short              # unit tests
go test -v -run TestScenario      # full agent scenarios
```

## License

MIT — see [LICENSE](LICENSE)
