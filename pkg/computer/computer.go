// Package computer defines the Computer interface for screen-based environments.
// Implementations connect to browser services (Browserbase, Playwright, etc.)
// and execute actions (click, type, scroll) returning screenshots.
package computer

import "fmt"

// Action represents a normalized computer use action.
type Action struct {
	Type      string `json:"type"`                // "click", "double_click", "type", "key", "scroll", "screenshot", "navigate", "wait"
	X         int    `json:"x,omitempty"`         // click/scroll coordinate
	Y         int    `json:"y,omitempty"`         // click/scroll coordinate
	Text      string `json:"text,omitempty"`      // for "type" action
	Key       string `json:"key,omitempty"`       // for "key" action (e.g. "Enter", "Escape")
	Direction string `json:"direction,omitempty"` // for "scroll": "up", "down", "left", "right"
	Amount    int    `json:"amount,omitempty"`    // scroll amount
	URL       string `json:"url,omitempty"`       // for "navigate"
	Duration  int    `json:"duration,omitempty"`  // for "wait" (milliseconds)
}

// DisplaySize holds screen dimensions.
type DisplaySize struct {
	Width  int `json:"width"`
	Height int `json:"height"`
}

// Computer is the interface for screen-based environments.
// All methods return a screenshot of the current state after the action.
type Computer interface {
	// Execute performs an action and returns a screenshot.
	Execute(action Action) (screenshot []byte, err error)

	// Screenshot takes a screenshot without performing any action.
	Screenshot() ([]byte, error)

	// DisplaySize returns the screen dimensions.
	DisplaySize() DisplaySize

	// Close terminates the session and releases resources.
	Close() error
}

// Config holds the configuration for creating a Computer.
type Config struct {
	Type      string      `json:"type"`                // "browserbase", "service", "playwright"
	URL       string      `json:"url,omitempty"`       // for "service" type
	APIKey    string      `json:"api_key,omitempty"`   // for "browserbase"
	ProjectID string      `json:"project_id,omitempty"` // for "browserbase"
	Display   DisplaySize `json:"display"`
}

// New creates a Computer from config. Returns nil if type is empty (no computer use).
func New(cfg Config) (Computer, error) {
	if cfg.Type == "" {
		return nil, nil
	}
	if cfg.Display.Width == 0 {
		cfg.Display.Width = 1280
	}
	if cfg.Display.Height == 0 {
		cfg.Display.Height = 800
	}

	switch cfg.Type {
	case "browserbase":
		return newBrowserbase(cfg)
	case "service":
		return newService(cfg)
	default:
		return nil, fmt.Errorf("unknown computer type: %s", cfg.Type)
	}
}
