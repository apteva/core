package core

import (
	"github.com/pion/rtp"
	"time"
)

// Healthy packets incur no extra reserve. Only a missing sequence starts a
// bounded reordering wait. Wraparound, duplicates and late arrivals cannot
// replay speech. Large discontinuities discard the obsolete reorder window.
type realtimeRTPOrder struct {
	initialized bool
	next        uint16
	pending     map[uint16]*rtp.Packet
	waiting     time.Time
	lost, late  uint64
}

const realtimeRTPReorderWait = 40 * time.Millisecond

func (q *realtimeRTPOrder) push(p *rtp.Packet, now time.Time) []*rtp.Packet {
	if !q.initialized {
		q.initialized = true
		q.next = p.SequenceNumber
		q.pending = map[uint16]*rtp.Packet{}
	}
	distance := int16(p.SequenceNumber - q.next)
	if distance < 0 {
		q.late++
		return nil
	}
	if distance > 128 {
		q.lost += uint64(distance)
		q.pending = map[uint16]*rtp.Packet{}
		q.next = p.SequenceNumber
		q.waiting = time.Time{}
	}
	if _, ok := q.pending[p.SequenceNumber]; ok {
		q.late++
		return nil
	}
	q.pending[p.SequenceNumber] = p
	return q.pop(now, len(q.pending) >= 8)
}
func (q *realtimeRTPOrder) pop(now time.Time, force bool) []*rtp.Packet {
	var result []*rtp.Packet
	for len(q.pending) > 0 {
		if p := q.pending[q.next]; p != nil {
			delete(q.pending, q.next)
			q.next++
			result = append(result, p)
			q.waiting = time.Time{}
			continue
		}
		if q.waiting.IsZero() {
			q.waiting = now
		}
		if !force && now.Sub(q.waiting) < realtimeRTPReorderWait {
			break
		}
		distance := uint16(65535)
		for seq := range q.pending {
			if d := seq - q.next; d < distance {
				distance = d
			}
		}
		q.lost += uint64(distance)
		q.next += distance
		q.waiting = time.Time{}
		force = false
	}
	return result
}
