package core

import (
	"github.com/pion/rtp"
	"testing"
	"time"
)

func TestRealtimeRTPReorderRecovery(t *testing.T) {
	q := &realtimeRTPOrder{}
	now := time.Unix(1, 0)
	push := func(seq uint16, ms int) []*rtp.Packet {
		return q.push(&rtp.Packet{Header: rtp.Header{SequenceNumber: seq}, Payload: []byte{byte(seq)}}, now.Add(time.Duration(ms)*time.Millisecond))
	}
	if len(push(65535, 0)) != 1 || len(push(0, 20)) != 1 {
		t.Fatal("healthy media delayed or wrap failed")
	}
	if len(push(2, 40)) != 0 {
		t.Fatal("reordered packet released early")
	}
	got := push(1, 50)
	if len(got) != 2 || got[0].SequenceNumber != 1 || got[1].SequenceNumber != 2 {
		t.Fatal("reordering lost speech")
	}
	if len(push(2, 60)) != 0 {
		t.Fatal("duplicate replayed")
	}
	if len(push(4, 80)) != 0 {
		t.Fatal("loss wait missing")
	}
	got = q.pop(now.Add(120*time.Millisecond), false)
	if len(got) != 1 || q.lost != 1 {
		t.Fatal("loss did not recover within 40ms")
	}
	if len(push(3, 130)) != 0 {
		t.Fatal("late speech replayed")
	}
	if len(push(5, 140)) != 1 {
		t.Fatal("fresh speech not immediate")
	}
	if len(push(500, 160)) != 1 || len(q.pending) != 0 {
		t.Fatal("discontinuity retained obsolete backlog")
	}
}
