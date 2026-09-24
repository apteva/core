package core

import (
	"encoding/json"
	"time"
)

const maxConnectorCallTimeout = 10 * time.Minute

// Connector schemas carry their default upstream timeout as a platform
// extension. Leave ordinary tools at three minutes; give slower connectors
// enough room for the upstream call and local dispatch overhead.
func toolCallTimeout(call toolCall) time.Duration {
	timeout := 3 * time.Minute
	if call.definition == nil || !call.definition.MCP {
		return timeout
	}
	if millis, ok := connectorTimeoutNumber(call.definition.InputSchema["x-apteva-timeout-ms"]); ok && millis > 0 {
		if candidate := time.Duration(millis)*time.Millisecond + 30*time.Second; candidate > timeout {
			timeout = candidate
		}
	}
	if raw := call.Args["_apteva"]; raw != "" {
		var options map[string]any
		if json.Unmarshal([]byte(raw), &options) == nil {
			if millis, ok := connectorTimeoutNumber(options["timeout_ms"]); ok && millis > 0 && millis <= 600_000 {
				if candidate := time.Duration(millis)*time.Millisecond + 30*time.Second; candidate > timeout {
					timeout = candidate
				}
			}
		}
	}
	if timeout > maxConnectorCallTimeout {
		return maxConnectorCallTimeout
	}
	return timeout
}

func connectorTimeoutNumber(value any) (int64, bool) {
	switch n := value.(type) {
	case int:
		return int64(n), true
	case int64:
		return n, true
	case float64:
		return int64(n), n == float64(int64(n))
	case json.Number:
		v, err := n.Int64()
		return v, err == nil
	default:
		return 0, false
	}
}
