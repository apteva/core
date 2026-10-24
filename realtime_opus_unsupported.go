//go:build !darwin && !linux

package core

import "fmt"

type realtimeOpus struct{}

func newRealtimeOpus() (*realtimeOpus, error) {
	return nil, fmt.Errorf("native Opus unavailable on this platform")
}
func (*realtimeOpus) Encode([]byte) ([]byte, error) {
	return nil, fmt.Errorf("native Opus unavailable")
}
func (*realtimeOpus) Decode([]byte) ([]byte, error) {
	return nil, fmt.Errorf("native Opus unavailable")
}
func (*realtimeOpus) Close() {}
