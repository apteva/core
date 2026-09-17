package core

import (
	"fmt"
	"testing"
)

func TestProviderBillingErrorIsNotRetried(t *testing.T) {
	for _, tc := range []struct {
		code      int
		permanent bool
	}{{400, true}, {401, true}, {402, true}, {403, true}, {429, false}, {500, false}, {503, false}} {
		err := fmt.Errorf("API error %d: provider response", tc.code)
		if got := permanentProviderError(err); got != tc.permanent {
			t.Errorf("HTTP %d permanent=%v, want %v", tc.code, got, tc.permanent)
		}
	}
}
