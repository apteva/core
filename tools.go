package main

import (
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"
	"unicode/utf8"
)

const maxToolResultLen = 4000

type toolCall struct {
	Name string
	Args map[string]string
	Raw  string // original matched text
}

// [[tool_name key="val" key2="val2"]] — values can span multiple lines, escaped quotes allowed
var toolCallRe = regexp.MustCompile(`(?s)\[\[([\w-]+)((?:\s+\w+="(?:[^"\\]|\\.)*")*)\]\]`)
var argRe = regexp.MustCompile(`(?s)(\w+)="((?:[^"\\]|\\.)*)"`)

// stripToolCalls removes [[...]] tool call syntax from text for display
func stripToolCalls(text string) string {
	cleaned := toolCallRe.ReplaceAllString(text, "")
	return collapseWhitespace(cleaned)
}

func parseToolCalls(text string) []toolCall {
	matches := toolCallRe.FindAllStringSubmatch(text, -1)
	var calls []toolCall
	for _, m := range matches {
		name := m[1]
		args := make(map[string]string)
		for _, a := range argRe.FindAllStringSubmatch(m[2], -1) {
			// Unescape \" in values
			val := strings.ReplaceAll(a[2], `\"`, `"`)
			args[a[1]] = val
		}
		calls = append(calls, toolCall{Name: name, Args: args, Raw: m[0]})
	}
	return calls
}

func executeTool(t *Thinker, call toolCall) {
	// Telemetry: tool.call
	if t.telemetry != nil {
		argsSummary := ""
		for k, v := range call.Args {
			if len(argsSummary) > 0 {
				argsSummary += ", "
			}
			val := v
			if len(val) > 50 {
				val = val[:50] + "..."
			}
			argsSummary += k + "=" + val
		}
		t.telemetry.Emit("tool.call", t.threadID, ToolCallData{
			Name: call.Name, Args: argsSummary,
		})
	}

	go func() {
		logMsg("TOOL", fmt.Sprintf("dispatch %s args=%v", call.Name, call.Args))
		start := time.Now()
		defer func() {
			if r := recover(); r != nil {
				logMsg("TOOL", fmt.Sprintf("PANIC %s: %v", call.Name, r))
				t.Inject(fmt.Sprintf("[tool:%s] error: panic: %v", call.Name, r))
				if t.telemetry != nil {
					t.telemetry.Emit("tool.result", t.threadID, ToolResultData{
						Name: call.Name, DurationMs: time.Since(start).Milliseconds(),
						Success: false, Result: fmt.Sprintf("panic: %v", r),
					})
				}
			}
		}()
		var result string
		if t.registry != nil {
			if res, ok := t.registry.Dispatch(call.Name, call.Args); ok {
				result = res
			} else {
				result = fmt.Sprintf("unknown tool %q", call.Name)
			}
		} else {
			// Fallback for tests without registry
			switch call.Name {
			case "web":
				result = webTool(call.Args)
			case "write_file":
				result = writeFileTool(call.Args)
			case "read_file":
				result = readFileTool(call.Args)
			case "list_files":
				result = listFilesTool(call.Args)
			default:
				result = fmt.Sprintf("unknown tool %q", call.Name)
			}
		}

		resultPreview := result
		if len(resultPreview) > 200 {
			resultPreview = resultPreview[:200] + "..."
		}
		logMsg("TOOL", fmt.Sprintf("result %s (%dms): %s", call.Name, time.Since(start).Milliseconds(), resultPreview))

		// Telemetry: tool.result
		if t.telemetry != nil {
			resultSummary := result
			if len(resultSummary) > 200 {
				resultSummary = resultSummary[:200] + "..."
			}
			t.telemetry.Emit("tool.result", t.threadID, ToolResultData{
				Name: call.Name, DurationMs: time.Since(start).Milliseconds(),
				Success: !strings.HasPrefix(result, "error") && !strings.HasPrefix(result, "unknown"),
				Result: resultSummary,
			})
		}

		t.Inject(fmt.Sprintf("[tool:%s] %s", call.Name, result))
	}()
}

func webTool(args map[string]string) string {
	url := args["url"]
	if url == "" {
		return "error: missing url argument"
	}

	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return fmt.Sprintf("error: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Sprintf("error: HTTP %d", resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 100_000))
	if err != nil {
		return fmt.Sprintf("error reading body: %v", err)
	}

	text := stripHTML(string(body))
	text = collapseWhitespace(text)

	if utf8.RuneCountInString(text) > maxToolResultLen {
		runes := []rune(text)
		text = string(runes[:maxToolResultLen]) + "\n[truncated]"
	}

	return text
}

var reScript = regexp.MustCompile(`(?is)<script[^>]*>.*?</script>`)
var reStyle = regexp.MustCompile(`(?is)<style[^>]*>.*?</style>`)
var reTag = regexp.MustCompile(`<[^>]+>`)

func stripHTML(s string) string {
	s = reScript.ReplaceAllString(s, "")
	s = reStyle.ReplaceAllString(s, "")
	s = reTag.ReplaceAllString(s, " ")
	// Decode common entities
	s = strings.ReplaceAll(s, "&amp;", "&")
	s = strings.ReplaceAll(s, "&lt;", "<")
	s = strings.ReplaceAll(s, "&gt;", ">")
	s = strings.ReplaceAll(s, "&quot;", `"`)
	s = strings.ReplaceAll(s, "&#39;", "'")
	s = strings.ReplaceAll(s, "&nbsp;", " ")
	return s
}

func collapseWhitespace(s string) string {
	lines := strings.Split(s, "\n")
	var out []string
	blank := 0
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			blank++
			if blank <= 1 {
				out = append(out, "")
			}
			continue
		}
		blank = 0
		out = append(out, trimmed)
	}
	return strings.Join(out, "\n")
}
