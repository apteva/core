package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// ANSI color codes
const (
	ansiReset   = "\033[0m"
	ansiDim     = "\033[2m"
	ansiBold    = "\033[1m"
	ansiRed     = "\033[31m"
	ansiGreen   = "\033[32m"
	ansiYellow  = "\033[33m"
	ansiBlue    = "\033[34m"
	ansiMagenta = "\033[35m"
	ansiCyan    = "\033[36m"
	ansiWhite   = "\033[37m"
)

// ConsoleLogger renders live telemetry events to stderr with colors.
// Used in headless mode to provide human-readable output without the TUI.
type ConsoleLogger struct {
	telemetry  *Telemetry
	lastThread string
	cursor     int
}

func NewConsoleLogger(t *Telemetry) *ConsoleLogger {
	return &ConsoleLogger{telemetry: t}
}

func (c *ConsoleLogger) Run() {
	for {
		// Wait for new events
		<-c.telemetry.notify

		events, next := c.telemetry.Events(c.cursor)
		c.cursor = next
		for _, ev := range events {
			c.render(ev)
		}
	}
}

func (c *ConsoleLogger) render(ev TelemetryEvent) {
	switch ev.Type {
	case "llm.done", "llm.error", "tool.call", "tool.result", "tool.pending", "tool.approved", "tool.rejected",
		"thread.spawn", "thread.done", "event.received", "instance.paused", "instance.resumed", "mode.changed":
		// Render these
	default:
		return
	}

	threadLabel := ev.ThreadID
	if threadLabel == "" {
		threadLabel = "main"
	}

	ts := ev.Time.Format("15:04:05")

	// Print thread header when switching context
	if threadLabel != c.lastThread {
		c.lastThread = threadLabel
		fmt.Fprintf(os.Stderr, "\n  %s┊%s %s%s%s  %s%s%s\n", ansiDim, ansiReset, ansiCyan, threadLabel, ansiReset, ansiDim, ts, ansiReset)
	}

	var data map[string]any
	if ev.Data != nil {
		json.Unmarshal(ev.Data, &data)
	}
	if data == nil {
		data = map[string]any{}
	}

	switch ev.Type {
	case "llm.done":
		msg := conTruncate(conGetString(data, "message"), 60)
		dur := conGetFloat(data, "duration_ms")
		tokIn := conGetInt(data, "tokens_in")
		tokOut := conGetInt(data, "tokens_out")
		cost := conGetFloat(data, "cost_usd")
		rate := conGetString(data, "rate")
		model := conGetString(data, "model")
		iter := conGetInt(data, "iteration")
		meta := fmt.Sprintf("%.1fs  ↑%d ↓%d  $%.4f", dur/1000, tokIn, tokOut, cost)
		if model != "" {
			meta += "  " + model
		}
		if rate != "" {
			meta += "  ⏱" + rate
		}
		if iter > 0 {
			meta += fmt.Sprintf("  #%d", iter)
		}
		fmt.Fprintf(os.Stderr, "  %s%s │%s %s✓%s %s%-50s%s  %s%s%s\n",
			ansiDim, ts, ansiReset,
			ansiGreen, ansiReset,
			ansiWhite, msg, ansiReset,
			ansiDim, meta, ansiReset,
		)

	case "llm.error":
		errMsg := conTruncate(conGetString(data, "error"), 70)
		fmt.Fprintf(os.Stderr, "  %s%s │%s %s✗ %s%s\n",
			ansiDim, ts, ansiReset,
			ansiRed, errMsg, ansiReset,
		)

	case "tool.call":
		toolName := conGetString(data, "name")
		args := conFormatToolArgs(data)
		if args != "" {
			fmt.Fprintf(os.Stderr, "  %s%s │%s %s⚡%s %s%s%s(%s)%s\n",
				ansiDim, ts, ansiReset,
				ansiYellow, ansiReset,
				ansiWhite, toolName, ansiDim, args, ansiReset,
			)
		} else {
			fmt.Fprintf(os.Stderr, "  %s%s │%s %s⚡%s %s%s%s\n",
				ansiDim, ts, ansiReset,
				ansiYellow, ansiReset,
				ansiWhite, toolName, ansiReset,
			)
		}

	case "tool.result":
		toolName := conGetString(data, "name")
		dur := conGetFloat(data, "duration_ms")
		success := conGetBool(data, "success")
		if !success {
			result := conTruncate(conGetString(data, "result"), 50)
			fmt.Fprintf(os.Stderr, "  %s%s │%s %s✗ %s failed: %s%s  %s%.1fs%s\n",
				ansiDim, ts, ansiReset,
				ansiRed, toolName, result, ansiReset,
				ansiDim, dur/1000, ansiReset,
			)
		} else {
			result := conTruncate(conGetString(data, "result"), 60)
			fmt.Fprintf(os.Stderr, "  %s%s │%s %s↳%s %s%s%s  %s%.1fs%s",
				ansiDim, ts, ansiReset,
				ansiGreen, ansiReset,
				ansiWhite, toolName, ansiReset,
				ansiDim, dur/1000, ansiReset,
			)
			if result != "" {
				fmt.Fprintf(os.Stderr, "  %s%s%s", ansiDim, result, ansiReset)
			}
			fmt.Fprintln(os.Stderr)
		}

	case "tool.pending":
		toolName := conGetString(data, "name")
		fmt.Fprintf(os.Stderr, "  %s%s │%s %s⏳ %s awaiting approval%s\n",
			ansiDim, ts, ansiReset,
			ansiMagenta, toolName, ansiReset,
		)

	case "tool.approved":
		toolName := conGetString(data, "name")
		fmt.Fprintf(os.Stderr, "  %s%s │%s %s✓ %s approved%s\n",
			ansiDim, ts, ansiReset,
			ansiGreen, toolName, ansiReset,
		)

	case "tool.rejected":
		toolName := conGetString(data, "name")
		fmt.Fprintf(os.Stderr, "  %s%s │%s %s✗ %s rejected%s\n",
			ansiDim, ts, ansiReset,
			ansiRed, toolName, ansiReset,
		)

	case "thread.spawn":
		childID := conGetString(data, "id")
		if childID == "" {
			childID = ev.ThreadID
		}
		directive := conTruncate(conGetString(data, "directive"), 50)
		fmt.Fprintf(os.Stderr, "  %s%s │%s %s⚙%s  thread %s%s%s spawned",
			ansiDim, ts, ansiReset,
			ansiCyan, ansiReset,
			ansiBold, childID, ansiReset,
		)
		if directive != "" {
			fmt.Fprintf(os.Stderr, "  %s— %s%s", ansiDim, directive, ansiReset)
		}
		fmt.Fprintln(os.Stderr)

	case "thread.done":
		childID := conGetString(data, "id")
		if childID == "" {
			childID = ev.ThreadID
		}
		result := conTruncate(conGetString(data, "result"), 50)
		fmt.Fprintf(os.Stderr, "  %s%s │%s %s✓%s  thread %s%s%s done",
			ansiDim, ts, ansiReset,
			ansiCyan, ansiReset,
			ansiBold, childID, ansiReset,
		)
		if result != "" {
			fmt.Fprintf(os.Stderr, "  %s— %s%s", ansiDim, result, ansiReset)
		}
		fmt.Fprintln(os.Stderr)

	case "instance.paused":
		fmt.Fprintf(os.Stderr, "  %s%s │%s %s⏸  paused%s\n",
			ansiDim, ts, ansiReset,
			ansiYellow, ansiReset,
		)

	case "instance.resumed":
		fmt.Fprintf(os.Stderr, "  %s%s │%s %s▶  resumed%s\n",
			ansiDim, ts, ansiReset,
			ansiGreen, ansiReset,
		)

	case "event.received":
		source := conGetString(data, "source")
		msg := conTruncate(conGetString(data, "message"), 60)
		icon := "▶"
		if source == "thread" {
			icon = "⇄"
		} else if source == "webhook" {
			icon = "⚑"
		}
		fmt.Fprintf(os.Stderr, "  %s%s │%s %s%s%s %s[%s]%s %s%s\n",
			ansiDim, ts, ansiReset,
			ansiCyan, icon, ansiReset,
			ansiDim, source, ansiReset,
			msg, ansiReset,
		)

	case "mode.changed":
		mode := conGetString(data, "mode")
		fmt.Fprintf(os.Stderr, "  %s%s │%s %s◆%s  mode → %s%s%s\n",
			ansiDim, ts, ansiReset,
			ansiBlue, ansiReset,
			ansiBold, mode, ansiReset,
		)
	}
}

// --- helpers ---

func conTruncate(s string, max int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.TrimSpace(s)
	if len(s) > max {
		return s[:max-1] + "…"
	}
	return s
}

func conGetString(data map[string]any, key string) string {
	if v, ok := data[key].(string); ok {
		return v
	}
	return ""
}

func conGetFloat(data map[string]any, key string) float64 {
	if v, ok := data[key].(float64); ok {
		return v
	}
	return 0
}

func conGetInt(data map[string]any, key string) int {
	if v, ok := data[key].(float64); ok {
		return int(v)
	}
	return 0
}

func conGetBool(data map[string]any, key string) bool {
	if v, ok := data[key].(bool); ok {
		return v
	}
	return false
}

func conFormatToolArgs(data map[string]any) string {
	args, ok := data["args"]
	if !ok {
		return ""
	}
	if m, ok := args.(map[string]any); ok {
		parts := make([]string, 0, len(m))
		for k, v := range m {
			val := fmt.Sprintf("%v", v)
			if len(val) > 40 {
				val = val[:40] + "…"
			}
			parts = append(parts, k+"="+val)
		}
		result := strings.Join(parts, ", ")
		if len(result) > 80 {
			return result[:80] + "…"
		}
		return result
	}
	if s, ok := args.(string); ok {
		if len(s) > 80 {
			return s[:80] + "…"
		}
		return s
	}
	return ""
}
