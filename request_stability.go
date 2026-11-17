package core

import (
	"os"
	"strings"
)

// Each change can be rolled back independently without changing the durable
// conversation, grants, or tool execution contract. Unset enables the fix.
func requestStabilityEnabled(setting string) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(setting))) {
	case "0", "false", "off", "no", "disabled":
		return false
	default:
		return true
	}
}
