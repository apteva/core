# Cogito

A continuous thinking engine that runs autonomous AI agent teams. Agents coordinate through an event bus, use external tool servers, and manage themselves via natural language directives.

*Cogito ergo sum* — I think, therefore I am.

## Architecture

```
┌─────────────────────────────────────────┐
│  Main Thread (coordinator)              │
│  Observes events, spawns/kills threads  │
└──────────┬──────────────────────────────┘
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
 Server  Server  Server   ← external tools (any stdio JSON-RPC server)
```

## Quick Start

```bash
# Set your API key
echo "FIREWORKS_API_KEY=your-key" > .env

# Build and run with TUI
go build -o cogito . && ./cogito

# Or run headless (API only)
./cogito --headless
```

## API

Default port: `3210` (set with `API_PORT` env var)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/status` | GET | Uptime, iteration count, threads, memory |
| `/threads` | GET | List all threads with state |
| `/events` | GET | SSE stream of all events |
| `/event` | POST | Inject a console command |
| `/config` | GET/PUT | Read/update directive |

## TUI Keys

| Key | Action |
|-----|--------|
| `i` | Chat input |
| `c` | Console command |
| `e` | Edit directive |
| `b` | Event bus viewer |
| `t` | Thread panel |
| `m` | Memory panel |
| `o` | Tools panel |
| `]` / `[` | Switch tabs |
| `j` / `k` | Scroll |
| `space` | Pause/resume |
| `q` | Quit |

## Core Tools

Always available to all threads:

| Tool | Description |
|------|-------------|
| `pace` | Set thinking speed (fast/normal/slow/sleep) and model size |
| `send` | Send message to another thread |
| `done` | Terminate this thread |
| `evolve` | Rewrite own directive |
| `remember` | Store to persistent memory |
| `spawn` | Create new thread (coordinator only) |
| `kill` | Stop a thread (coordinator only) |

Additional tools are provided by external servers and discovered automatically.

## Configuration

```json
{
  "directive": "Your mission here",
  "mcp_servers": [
    {
      "name": "myservice",
      "command": "./my-server-binary",
      "env": {"API_KEY": "..."}
    }
  ]
}
```

## Tool Servers

External tools connect via [MCP](https://modelcontextprotocol.io/) (stdio JSON-RPC). The `mcps/` directory contains examples:

| Server | Purpose |
|--------|---------|
| `helpdesk` | Support ticket management |
| `chat` | User conversations |
| `orders` | Order management |
| `inventory` | Stock tracking |
| `schedule` | Content calendar |
| `creative` | AI content generation |
| `social` | Social media publishing |
| `sensors` | Robot sensor readings |
| `motors` | Robot motor control |
| `pushover` | Push notifications |

## Scenarios

Integration tests that validate full agent team behavior:

```bash
# Run all scenarios
go test -v -run TestScenario -timeout=600s
```

| Scenario | What it tests |
|----------|---------------|
| Helpdesk | Ticket monitoring, KB lookup, reply/close |
| Chat | Multi-user conversation, factual Q&A |
| Bakery | Multi-service coordination, stock management, failure handling |
| Social Team | 3-stage content pipeline with 3 permanent workers |
| Robot | Sensor-motor loop, obstacle avoidance, camera scanning |

## License

MIT — see [LICENSE](LICENSE)
