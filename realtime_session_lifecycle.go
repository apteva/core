package core

import "sync"

// realtimeSessionLifecycle closes a provider session's event stream only after
// every goroutine that can publish to it has stopped. Shutdown remains
// asynchronous: session Close methods signal their loops and close the
// connection, while this lifecycle reaps them in the background.
type realtimeSessionLifecycle struct {
	startOnce sync.Once
	loops     sync.WaitGroup
	stopped   chan struct{}
}

func (l *realtimeSessionLifecycle) start(events chan RealtimeEvent, runs ...func()) {
	l.startOnce.Do(func() {
		l.stopped = make(chan struct{})
		l.loops.Add(len(runs))
		for _, run := range runs {
			go func() {
				defer l.loops.Done()
				run()
			}()
		}
		go func() {
			l.loops.Wait()
			close(events)
			close(l.stopped)
		}()
	})
}
