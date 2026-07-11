package core

import "fmt"

const maxThreadIDLength = 128

// validateThreadID keeps one identifier safe for maps, event routing, URLs,
// telemetry, and history filenames. Thread IDs are machine identifiers, not
// display names; Thread.Name carries human-readable text.
func validateThreadID(id string) error {
	if id == "" {
		return fmt.Errorf("thread id required")
	}
	if len(id) > maxThreadIDLength {
		return fmt.Errorf("thread id exceeds %d bytes", maxThreadIDLength)
	}
	for i := 0; i < len(id); i++ {
		c := id[i]
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
			(c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' {
			continue
		}
		return fmt.Errorf("thread id %q contains invalid character %q", id, c)
	}
	if id == "." || id == ".." || id[0] == '.' {
		return fmt.Errorf("thread id %q must not start with a dot", id)
	}
	return nil
}
