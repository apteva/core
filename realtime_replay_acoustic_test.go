package core

import (
	"testing"
	"time"
)

// This replay accepts one known caller utterance per fixture. Compute its final
// endpoint before streaming so early greetings/backchannels cannot become a
// reply simply because the most recently sent input frame happened to be quiet.
func replayAcousticEndpoint(pcm []byte) time.Duration {
	var d realtimeAcousticDetector
	var end time.Duration
	for off := 0; off < len(pcm); off += 960 {
		next := min(off+960, len(pcm))
		if d.observe(pcm[off:next], AudioPCM16, 24000, 20) {
			end = time.Duration(next) * time.Second / 48000
		}
	}
	return end
}

type realtimeReplayAcousticTiming struct {
	endpoint, firstPacket, firstAcoustic, priorActive time.Time
	overlap                                           bool
	before, after                                     realtimeAcousticDetector
}

func (r *realtimeReplayAcousticTiming) audio(pcm []byte, at time.Time) {
	if at.Before(r.endpoint) {
		if r.before.observe(pcm, AudioPCM16, 24000, 100) {
			r.priorActive = at
		}
		return
	}
	if r.firstPacket.IsZero() {
		r.firstPacket = at
		r.overlap = !r.priorActive.IsZero() && r.endpoint.Sub(r.priorActive) < 300*time.Millisecond
	}
	if r.after.observe(pcm, AudioPCM16, 24000, 100) && r.firstAcoustic.IsZero() {
		r.firstAcoustic = at
	}
}

func TestReplayAcousticTimingExcludesEarlyOutputAndFlagsContinuation(t *testing.T) {
	base := time.Now()
	r := realtimeReplayAcousticTiming{endpoint: base.Add(time.Second)}
	r.audio(acousticPCM(100, 500), base.Add(100*time.Millisecond))
	if !r.firstAcoustic.IsZero() {
		t.Fatal("early backchannel counted as reply")
	}
	r.audio(acousticPCM(100, 0), base.Add(1100*time.Millisecond))
	r.audio(acousticPCM(100, 500), base.Add(1300*time.Millisecond))
	if r.overlap || timingMS(r.endpoint, r.firstAcoustic) != float64(300) {
		t.Fatalf("wrong reply boundaries: %+v", r)
	}
	r = realtimeReplayAcousticTiming{endpoint: base.Add(time.Second)}
	r.audio(acousticPCM(100, 500), base.Add(950*time.Millisecond))
	r.audio(acousticPCM(100, 500), base.Add(1050*time.Millisecond))
	if !r.overlap {
		t.Fatal("continuing greeting presented as eligible latency sample")
	}
}

func TestReplayAcousticEndpointIncludesQuietSpeechAndTrailingSilence(t *testing.T) {
	pcm := append(acousticPCM(200, 40), acousticPCM(500, 0)...)
	if got := replayAcousticEndpoint(pcm); got != 200*time.Millisecond {
		t.Fatalf("endpoint=%v", got)
	}
	if replayAcousticEndpoint(acousticPCM(500, 0)) != 0 {
		t.Fatal("silence became caller speech")
	}
}
