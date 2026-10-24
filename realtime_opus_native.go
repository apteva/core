//go:build darwin || linux

package core

import (
	"encoding/binary"
	"fmt"
	"github.com/ebitengine/purego"
	"os"
	"runtime"
	"sync"
)

type realtimeOpus struct {
	encodeMu, decodeMu             sync.Mutex
	encoder, decoder               uintptr
	encode                         func(uintptr, *int16, int, *byte, int32) int32
	decode                         func(uintptr, *byte, int32, *int16, int, int) int
	destroyEncoder, destroyDecoder func(uintptr)
}

var realtimeOpusLibrary struct {
	sync.Once
	handle uintptr
	err    error
}

func newRealtimeOpus() (*realtimeOpus, error) {
	realtimeOpusLibrary.Do(func() {
		names := []string{"libopus.so.0", "libopus.so"}
		if runtime.GOOS == "darwin" {
			names = []string{"libopus.0.dylib", "/opt/homebrew/lib/libopus.0.dylib", "/usr/local/lib/libopus.0.dylib"}
		}
		if p := os.Getenv("APTEVA_OPUS_LIBRARY"); p != "" {
			names = []string{p}
		}
		for _, name := range names {
			h, err := purego.Dlopen(name, purego.RTLD_NOW|purego.RTLD_LOCAL)
			if err == nil {
				realtimeOpusLibrary.handle = h
				return
			}
		}
		realtimeOpusLibrary.err = fmt.Errorf("WebRTC PCM audio requires libopus; install libopus or configure APTEVA_OPUS_LIBRARY")
	})
	if realtimeOpusLibrary.err != nil {
		return nil, realtimeOpusLibrary.err
	}
	h := realtimeOpusLibrary.handle
	var encCreate func(int32, int32, int32, *int32) uintptr
	var decCreate func(int32, int32, *int32) uintptr
	c := &realtimeOpus{}
	// Resolve symbols before registering: an incompatible library must fail setup.
	for _, name := range []string{"opus_encoder_create", "opus_decoder_create", "opus_encode", "opus_decode", "opus_encoder_destroy", "opus_decoder_destroy"} {
		if _, err := purego.Dlsym(h, name); err != nil {
			return nil, fmt.Errorf("libopus missing %s", name)
		}
	}
	purego.RegisterLibFunc(&encCreate, h, "opus_encoder_create")
	purego.RegisterLibFunc(&decCreate, h, "opus_decoder_create")
	purego.RegisterLibFunc(&c.encode, h, "opus_encode")
	purego.RegisterLibFunc(&c.decode, h, "opus_decode")
	purego.RegisterLibFunc(&c.destroyEncoder, h, "opus_encoder_destroy")
	purego.RegisterLibFunc(&c.destroyDecoder, h, "opus_decoder_destroy")
	var code int32
	c.encoder = encCreate(24000, 1, 2048, &code)
	if code != 0 || c.encoder == 0 {
		return nil, fmt.Errorf("opus encoder initialization failed: %d", code)
	}
	c.decoder = decCreate(24000, 1, &code)
	if code != 0 || c.decoder == 0 {
		c.destroyEncoder(c.encoder)
		return nil, fmt.Errorf("opus decoder initialization failed: %d", code)
	}
	return c, nil
}
func (c *realtimeOpus) Encode(pcm []byte) ([]byte, error) {
	c.encodeMu.Lock()
	defer c.encodeMu.Unlock()
	if c.encoder == 0 {
		return nil, fmt.Errorf("opus closed")
	}
	if len(pcm) != 960 {
		return nil, fmt.Errorf("Opus expects 20 ms PCM24")
	}
	var in [480]int16
	for i := range in {
		in[i] = int16(binary.LittleEndian.Uint16(pcm[2*i:]))
	}
	out := make([]byte, 1275)
	n := c.encode(c.encoder, &in[0], 480, &out[0], int32(len(out)))
	if n < 0 {
		return nil, fmt.Errorf("opus encode failed: %d", n)
	}
	return out[:n], nil
}
func (c *realtimeOpus) Decode(packet []byte) ([]byte, error) {
	c.decodeMu.Lock()
	defer c.decodeMu.Unlock()
	if c.decoder == 0 {
		return nil, fmt.Errorf("opus closed")
	}
	samples := make([]int16, 2880)
	var in *byte
	if len(packet) > 0 {
		in = &packet[0]
	}
	n := c.decode(c.decoder, in, int32(len(packet)), &samples[0], len(samples), 0)
	if n < 0 {
		return nil, fmt.Errorf("opus decode failed: %d", n)
	}
	out := make([]byte, 2*n)
	for i, v := range samples[:n] {
		binary.LittleEndian.PutUint16(out[2*i:], uint16(v))
	}
	return out, nil
}
func (c *realtimeOpus) Close() {
	c.encodeMu.Lock()
	defer c.encodeMu.Unlock()
	c.decodeMu.Lock()
	defer c.decodeMu.Unlock()
	if c.encoder != 0 {
		c.destroyEncoder(c.encoder)
		c.encoder = 0
	}
	if c.decoder != 0 {
		c.destroyDecoder(c.decoder)
		c.decoder = 0
	}
}
