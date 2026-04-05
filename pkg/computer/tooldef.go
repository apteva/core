package computer

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"
)

// ToolDefinition describes a tool for non-Anthropic providers.
type ToolDefinition struct {
	Name        string
	Description string
	Syntax      string
	Rules       string
	Parameters  map[string]any
}

// GetComputerToolDef returns the computer_use tool definition (screen interaction only, no navigate).
func GetComputerToolDef(display DisplaySize) ToolDefinition {
	return ToolDefinition{
		Name: "computer_use",
		Description: fmt.Sprintf(
			"Interact with a browser screen (%dx%d). See what's on screen, click elements, type text, press keys, scroll. "+
				"Every action returns a screenshot of the current screen state. "+
				"Use browser_session to open URLs first, then use this tool to interact with the page.",
			display.Width, display.Height,
		),
		Syntax: `[[computer_use action="screenshot"]]`,
		Rules: "Actions: screenshot, click (coordinate=\"x,y\"), double_click (coordinate=\"x,y\"), " +
			"type (text=\"...\"), key (key=\"Enter\"/\"Escape\"/\"ctrl+c\"), " +
			"scroll (direction=\"up\"/\"down\", amount=3), mouse_move (coordinate=\"x,y\"), wait (duration=1000ms). " +
			"Always take a screenshot first to see the current state before interacting.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"action": map[string]any{
					"type":        "string",
					"description": "Action: screenshot, click, double_click, type, key, scroll, mouse_move, wait",
				},
				"coordinate": map[string]any{
					"type":        "string",
					"description": "Position as \"x,y\" for click/double_click/scroll/mouse_move",
				},
				"text": map[string]any{
					"type":        "string",
					"description": "Text to type (for type action)",
				},
				"key": map[string]any{
					"type":        "string",
					"description": "Key to press (for key action, e.g. Enter, Escape, ctrl+s)",
				},
				"direction": map[string]any{
					"type":        "string",
					"description": "Scroll direction: up, down",
				},
				"amount": map[string]any{
					"type":        "string",
					"description": "Scroll amount (default 3)",
				},
				"duration": map[string]any{
					"type":        "string",
					"description": "Wait duration in milliseconds (default 1000)",
				},
			},
			"required": []string{"action"},
		},
	}
}

// GetSessionToolDef returns the browser_session tool definition.
func GetSessionToolDef() ToolDefinition {
	return ToolDefinition{
		Name: "browser_session",
		Description: "Manage browser sessions. Open a URL to navigate the browser, check status, or close the session. " +
			"This tool does NOT return screenshots. After opening a URL, use computer_use with action=screenshot to see the page.",
		Syntax: `[[browser_session action="open" url="https://example.com"]]`,
		Rules: "Actions: open (url — navigates browser to URL, returns screenshot), " +
			"close (ends browser session), " +
			"resume (session_id — reconnect to a Browserbase session), " +
			"status (returns current URL, session type, session ID).",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"action": map[string]any{
					"type":        "string",
					"description": "Action: open, close, resume, status",
				},
				"url": map[string]any{
					"type":        "string",
					"description": "URL to navigate to (for open action)",
				},
				"session_id": map[string]any{
					"type":        "string",
					"description": "Session ID to resume (for resume action, Browserbase only)",
				},
			},
			"required": []string{"action"},
		},
	}
}

// AnthropicToolSpec is the native Claude computer use tool format.
type AnthropicToolSpec struct {
	Type            string `json:"type"`
	Name            string `json:"name"`
	DisplayWidthPx  int    `json:"display_width_px"`
	DisplayHeightPx int    `json:"display_height_px"`
}

// GetAnthropicToolSpec returns the native Anthropic computer use tool spec.
func GetAnthropicToolSpec(display DisplaySize, toolVersion string) AnthropicToolSpec {
	return AnthropicToolSpec{
		Type:            "computer_" + toolVersion,
		Name:            "computer",
		DisplayWidthPx:  display.Width,
		DisplayHeightPx: display.Height,
	}
}

// AnthropicBetaHeader returns the appropriate beta header for computer use.
func AnthropicBetaHeader(toolVersion string) string {
	switch toolVersion {
	case "20251124":
		return "computer-use-2025-11-24"
	default:
		return "computer-use-2025-01-24"
	}
}

// HandleComputerAction executes a screen interaction action (no navigate).
func HandleComputerAction(comp Computer, args map[string]string) (text string, screenshot []byte, err error) {
	actionType := args["action"]
	if actionType == "" {
		return "", nil, fmt.Errorf("missing action argument")
	}

	// Reject navigate — use browser_session for that
	if actionType == "navigate" {
		return "", nil, fmt.Errorf("use browser_session to navigate to URLs, not computer_use")
	}

	action := Action{Type: actionType}
	parseCoordinate(args["coordinate"], &action)
	action.Text = args["text"]
	action.Key = args["key"]
	action.Direction = args["direction"]
	if amt := args["amount"]; amt != "" {
		action.Amount, _ = strconv.Atoi(amt)
	}
	if dur := args["duration"]; dur != "" {
		action.Duration, _ = strconv.Atoi(dur)
	}

	start := time.Now()
	if actionType == "screenshot" {
		screenshot, err = comp.Screenshot()
	} else {
		screenshot, err = comp.Execute(action)
	}
	duration := time.Since(start)

	if err != nil {
		return fmt.Sprintf("Error: %v", err), nil, err
	}

	text = fmt.Sprintf("Success: %s action completed. Screenshot attached (%d bytes, %dms).",
		actionType, len(screenshot), duration.Milliseconds())
	return text, screenshot, nil
}

// HandleSessionAction manages browser session lifecycle.
func HandleSessionAction(comp Computer, args map[string]string) (text string, screenshot []byte, err error) {
	actionType := args["action"]
	if actionType == "" {
		return "", nil, fmt.Errorf("missing action argument")
	}

	switch actionType {
	case "open":
		url := args["url"]
		if url == "" {
			return "", nil, fmt.Errorf("url required for open action")
		}
		start := time.Now()
		_, err = comp.Execute(Action{Type: "navigate", URL: url})
		duration := time.Since(start)
		if err != nil {
			return fmt.Sprintf("Error navigating to %s: %v", url, err), nil, err
		}
		text = fmt.Sprintf("Navigated to %s (%dms). Use computer_use with action=screenshot to see the page.",
			url, duration.Milliseconds())
		return text, nil, nil

	case "close":
		if err := comp.Close(); err != nil {
			return fmt.Sprintf("Error closing session: %v", err), nil, err
		}
		return "Session closed.", nil, nil

	case "status":
		display := comp.DisplaySize()
		info := fmt.Sprintf("Browser active. Display: %dx%d.", display.Width, display.Height)
		// Check for optional session info
		if s, ok := comp.(SessionInfo); ok {
			info += fmt.Sprintf(" Type: %s.", s.SessionType())
			if id := s.SessionID(); id != "" {
				info += fmt.Sprintf(" Session ID: %s.", id)
			}
			if url := s.CurrentURL(); url != "" {
				info += fmt.Sprintf(" URL: %s.", url)
			}
		}
		return info, nil, nil

	case "resume":
		sessionID := args["session_id"]
		if sessionID == "" {
			return "", nil, fmt.Errorf("session_id required for resume action")
		}
		if r, ok := comp.(Resumable); ok {
			if err := r.Resume(sessionID); err != nil {
				return fmt.Sprintf("Error resuming session: %v", err), nil, err
			}
			return fmt.Sprintf("Resumed session %s. Use computer_use with action=screenshot to see the page.", sessionID), nil, nil
		}
		return "Resume not supported for this browser type.", nil, nil

	default:
		return "", nil, fmt.Errorf("unknown action: %s (use open, close, status, resume)", actionType)
	}
}

// SessionInfo is an optional interface for computers that can report session details.
type SessionInfo interface {
	SessionType() string // "local", "browserbase", "service"
	SessionID() string   // empty for local
	CurrentURL() string  // current page URL
}

// Resumable is an optional interface for computers that support session resumption.
type Resumable interface {
	Resume(sessionID string) error
}

// parseCoordinate parses "x,y" or [x,y] format into action X/Y fields.
func parseCoordinate(coord string, action *Action) {
	if coord == "" {
		return
	}
	// Try JSON array [x, y]
	if strings.HasPrefix(coord, "[") {
		var arr []int
		if json.Unmarshal([]byte(coord), &arr) == nil && len(arr) == 2 {
			action.X = arr[0]
			action.Y = arr[1]
			return
		}
	}
	// Try "x,y"
	parts := strings.SplitN(coord, ",", 2)
	if len(parts) == 2 {
		action.X, _ = strconv.Atoi(strings.TrimSpace(parts[0]))
		action.Y, _ = strconv.Atoi(strings.TrimSpace(parts[1]))
	}
}
