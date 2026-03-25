# Cogito

A continuous thinking engine that runs autonomous AI agent teams. Agents coordinate through an event bus, use MCP servers as tools, and manage themselves via directives.

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
  MCP    MCP    MCP       ← external tools via Model Context Protocol
```

## Quick Start

```bash
# Set your API key
echo "FIREWORKS_API_KEY=your-key" > .env

# Build and run with TUI
go build -o cogito . && ./cogito

# Or run headless (API only)
./cogito --headless
# or
NO_TUI=1 ./cogito
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
| `[[pace rate="..." model="..."]]` | Set thinking speed (fast/normal/slow/sleep) and model (large/small) |
| `[[send id="..." message="..."]]` | Send message to another thread |
| `[[done message="..."]]` | Terminate this thread |
| `[[evolve directive="..."]]` | Rewrite own directive |
| `[[remember text="..."]]` | Store to persistent memory |
| `[[spawn id="..." directive="..." tools="..."]]` | Create new thread (main only) |
| `[[kill id="..."]]` | Stop a thread (main only) |

## MCP Servers

External tools are connected via [Model Context Protocol](https://modelcontextprotocol.io/) servers. Configure in `config.json`:

```json
{
  "directive": "Your mission here",
  "mcp_servers": [
    {
      "name": "pushover",
      "command": "./mcp-pushover-server",
      "env": {"PUSHOVER_API_TOKEN": "..."}
    }
  ]
}
```

Included MCP servers in `mcps/`:

| Server | Tools | Purpose |
|--------|-------|---------|
| `pushover` | send_notification | Push notifications |
| `helpdesk` | list_tickets, reply_ticket, close_ticket, lookup_kb | Support desk |
| `chat` | get_messages, send_reply | User conversations |
| `orders` | get_orders, update_order | Order management |
| `inventory` | check_stock, use_stock, list_stock | Inventory tracking |
| `schedule` | get_schedule, update_slot | Content calendar |
| `creative` | generate_post, generate_image | AI content generation |
| `social` | post, get_channels, get_posts | Social media publishing |
| `sensors` | read_sensors, read_camera | Robot sensor readings |
| `motors` | move, turn, stop, get_status | Robot motor control |

## Scenarios

Integration tests that validate full agent team behavior:

```bash
# Run all scenarios
go test -v -run TestScenario -timeout=600s

# Run specific scenario
go test -v -run TestScenario_Chat -timeout=300s
```

| Scenario | Threads | MCPs | What it tests |
|----------|---------|------|---------------|
| Helpdesk | 1 worker | helpdesk | Ticket monitoring, KB lookup, reply/close |
| Chat | per-user | chat | Multi-user conversation, factual Q&A |
| Bakery | 2 workers | orders + inventory | Multi-MCP coordination, stock checks, failure handling |
| Social Team | 3 workers | schedule + creative + social | 3-stage content pipeline, team coordination |
| Robot | 1 pilot | sensors + motors | Sensor loop, obstacle avoidance, camera scanning |

## EventBus

All communication flows through a single event bus:

- **Targeted events** (`To: "thread-id"`) — delivered to specific thread, wakes it
- **Broadcasts** (`To: ""`) — only delivered to observers (TUI, tests), never wake threads
- **Publishing never blocks** — if a subscriber is slow, events are dropped
- Subscribers: `Subscribe(id)` for threads, `SubscribeAll(id)` for observers

## License

MIT
