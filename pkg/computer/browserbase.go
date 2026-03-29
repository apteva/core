package computer

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

type browserbaseComputer struct {
	apiKey    string
	projectID string
	sessionID string
	display   DisplaySize
	debugURL  string // CDP debug URL for the session
	client    *http.Client
}

func newBrowserbase(cfg Config) (Computer, error) {
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("browserbase: api_key is required")
	}

	c := &browserbaseComputer{
		apiKey:    cfg.APIKey,
		projectID: cfg.ProjectID,
		display:   cfg.Display,
		client:    &http.Client{Timeout: 30 * time.Second},
	}

	// Create a session
	if err := c.createSession(); err != nil {
		return nil, fmt.Errorf("browserbase: failed to create session: %w", err)
	}

	return c, nil
}

func (c *browserbaseComputer) createSession() error {
	body := map[string]any{
		"projectId": c.projectID,
		"browserSettings": map[string]any{
			"viewport": map[string]int{
				"width":  c.display.Width,
				"height": c.display.Height,
			},
		},
	}
	data, _ := json.Marshal(body)

	req, err := http.NewRequest("POST", "https://api.browserbase.com/v1/sessions", bytes.NewReader(data))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-bb-api-key", c.apiKey)

	resp, err := c.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 && resp.StatusCode != 201 {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	var result struct {
		ID       string `json:"id"`
		DebugURL string `json:"debugUrl"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return err
	}
	c.sessionID = result.ID
	c.debugURL = result.DebugURL
	return nil
}

func (c *browserbaseComputer) Execute(action Action) ([]byte, error) {
	// Send action to the session
	body := map[string]any{
		"action": action.Type,
	}
	switch action.Type {
	case "click", "double_click":
		body["x"] = action.X
		body["y"] = action.Y
	case "type":
		body["text"] = action.Text
	case "key":
		body["key"] = action.Key
	case "scroll":
		body["x"] = action.X
		body["y"] = action.Y
		body["direction"] = action.Direction
		body["amount"] = action.Amount
	case "navigate":
		body["url"] = action.URL
	case "wait":
		dur := action.Duration
		if dur <= 0 {
			dur = 1000
		}
		time.Sleep(time.Duration(dur) * time.Millisecond)
		return c.Screenshot()
	case "screenshot":
		return c.Screenshot()
	}

	data, _ := json.Marshal(body)
	url := fmt.Sprintf("https://api.browserbase.com/v1/sessions/%s/actions", c.sessionID)
	req, err := http.NewRequest("POST", url, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-bb-api-key", c.apiKey)

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("action %s failed: %w", action.Type, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("action %s: HTTP %d: %s", action.Type, resp.StatusCode, string(respBody))
	}

	// Small delay for action to take effect, then screenshot
	time.Sleep(200 * time.Millisecond)
	return c.Screenshot()
}

func (c *browserbaseComputer) Screenshot() ([]byte, error) {
	url := fmt.Sprintf("https://api.browserbase.com/v1/sessions/%s/screenshot", c.sessionID)
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("x-bb-api-key", c.apiKey)

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("screenshot failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("screenshot: HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	// Response may be base64 or raw PNG depending on API version
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	// Try to decode as base64 first
	if decoded, err := base64.StdEncoding.DecodeString(string(respBody)); err == nil {
		return decoded, nil
	}
	// Otherwise assume raw bytes
	return respBody, nil
}

func (c *browserbaseComputer) DisplaySize() DisplaySize {
	return c.display
}

func (c *browserbaseComputer) Close() error {
	if c.sessionID == "" {
		return nil
	}
	url := fmt.Sprintf("https://api.browserbase.com/v1/sessions/%s", c.sessionID)
	req, err := http.NewRequest("DELETE", url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("x-bb-api-key", c.apiKey)
	resp, err := c.client.Do(req)
	if err != nil {
		return err
	}
	resp.Body.Close()
	c.sessionID = ""
	return nil
}
