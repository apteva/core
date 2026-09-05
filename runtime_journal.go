package core

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"sort"
)

// Runtime deltas are committed independently of the comparatively large static
// thread/provider configuration. The snapshot's sequence is a replay checkpoint.
type runtimeDelta struct {
	ExecutionOrder []string                   `json:"execution_order,omitempty"`
	OutboxOrder    []string                   `json:"outbox_order,omitempty"`
	ThreadEvents   map[string]json.RawMessage `json:"thread_events,omitempty"`
	Sequence       uint64                     `json:"seq"`
	Executions     map[string]json.RawMessage `json:"executions,omitempty"`
	Outbox         map[string]json.RawMessage `json:"outbox,omitempty"`
	MainEvents     *[]PersistentThreadEvent   `json:"main_events,omitempty"`
}
type runtimeState struct {
	ThreadEvents map[string][]PersistentThreadEvent
	Executions   []PersistentEventExecution
	Outbox       []EventLifecycleTransition
	MainEvents   []PersistentThreadEvent
}

func (c *Config) runtimeStateLocked() runtimeState {
	events := map[string][]PersistentThreadEvent{}
	for _, thread := range c.Threads {
		if len(thread.Events) > 0 {
			events[thread.ID] = thread.Events
		}
	}
	return runtimeState{ThreadEvents: events, Executions: c.EventExecutions, Outbox: c.EventLifecycleOutbox, MainEvents: c.MainEvents}
}
func keyedJSON[T any](values []T, key func(T) string) map[string]json.RawMessage {
	out := map[string]json.RawMessage{}
	for _, v := range values {
		out[key(v)], _ = json.Marshal(v)
	}
	return out
}
func jsonDelta(before, after map[string]json.RawMessage) map[string]json.RawMessage {
	out := map[string]json.RawMessage{}
	for k, v := range after {
		if !bytes.Equal(v, before[k]) {
			out[k] = v
		}
	}
	for k := range before {
		if _, ok := after[k]; !ok {
			out[k] = nil
		}
	}
	return out
}
func applyJSONDelta[T any](values []T, delta map[string]json.RawMessage, key func(T) string, order []string) ([]T, error) {
	out := make([]T, 0, len(values)+len(delta))
	seen := map[string]bool{}
	for _, v := range values {
		k := key(v)
		seen[k] = true
		raw, ok := delta[k]
		if !ok {
			out = append(out, v)
		} else if len(raw) > 0 && string(raw) != "null" {
			var next T
			if err := json.Unmarshal(raw, &next); err != nil {
				return nil, err
			}
			out = append(out, next)
		}
	}
	if len(order) == 0 {
		for k := range delta {
			order = append(order, k)
		}
		sort.Strings(order)
	}
	for _, k := range order {
		raw := delta[k]
		if !seen[k] && len(raw) > 0 && string(raw) != "null" {
			var next T
			if err := json.Unmarshal(raw, &next); err != nil {
				return nil, err
			}
			out = append(out, next)
		}
	}
	return out, nil
}
func (c *Config) applyRuntimeDeltaLocked(d runtimeDelta) error {
	var err error
	c.EventExecutions, err = applyJSONDelta(c.EventExecutions, d.Executions, func(v PersistentEventExecution) string { return v.ExecutionID }, d.ExecutionOrder)
	if err != nil {
		return err
	}
	c.EventLifecycleOutbox, err = applyJSONDelta(c.EventLifecycleOutbox, d.Outbox, func(v EventLifecycleTransition) string { return v.ID }, d.OutboxOrder)
	if err != nil {
		return err
	}
	for i := range c.Threads {
		if raw, ok := d.ThreadEvents[c.Threads[i].ID]; ok {
			var events []PersistentThreadEvent
			if len(raw) > 0 {
				if err := json.Unmarshal(raw, &events); err != nil {
					return err
				}
			}
			c.Threads[i].Events = events
		}
	}
	if d.MainEvents != nil {
		c.MainEvents = *d.MainEvents
	}
	c.RuntimeSequence = d.Sequence
	return nil
}
func (c *Config) loadRuntimeJournalLocked() error {
	f, err := os.Open(c.path + ".runtime.jsonl")
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	defer f.Close()
	scan := bufio.NewScanner(f)
	scan.Buffer(make([]byte, 64<<10), 32<<20)
	var offset int64
	for scan.Scan() {
		line := scan.Bytes()
		var d runtimeDelta
		if err := json.Unmarshal(line, &d); err != nil { // Only a torn final append is recoverable.
			if scan.Scan() {
				return fmt.Errorf("corrupt interior runtime journal record")
			}
			if err := scan.Err(); err != nil {
				return err
			}
			return os.Truncate(c.path+".runtime.jsonl", offset)
		}
		offset += int64(len(line) + 1)
		if d.Sequence <= c.RuntimeSequence {
			continue
		}
		if d.Sequence != c.RuntimeSequence+1 {
			return fmt.Errorf("runtime journal sequence gap")
		}
		if err := c.applyRuntimeDeltaLocked(d); err != nil {
			return err
		}
	}
	if err := scan.Err(); err != nil {
		return err
	}
	// A crash can leave a complete JSON record without its final newline.
	// Keep the committed record, but restore the separator before another append.
	info, err := f.Stat()
	if err != nil {
		return err
	}
	if offset == info.Size()+1 {
		out, err := os.OpenFile(c.path+".runtime.jsonl", os.O_APPEND|os.O_WRONLY, 0600)
		if err != nil {
			return err
		}
		_, err = out.WriteString("\n")
		if err == nil {
			err = out.Sync()
		}
		closeErr := out.Close()
		if err != nil {
			return err
		}
		return closeErr
	}
	return nil
}
func (c *Config) updateRuntime(fn func()) error {
	c.saveMu.Lock()
	defer c.saveMu.Unlock()
	c.mu.Lock()
	defer c.mu.Unlock()
	beforeBytes, err := json.Marshal(c.runtimeStateLocked())
	if err != nil {
		return err
	}
	var before runtimeState
	if err = json.Unmarshal(beforeBytes, &before); err != nil {
		return err
	}
	fn()
	rollback := func() {
		c.EventExecutions = before.Executions
		c.EventLifecycleOutbox = before.Outbox
		c.MainEvents = before.MainEvents
		for i := range c.Threads {
			c.Threads[i].Events = before.ThreadEvents[c.Threads[i].ID]
		}
	}
	after := c.runtimeStateLocked()
	oldEvents, newEvents := map[string]json.RawMessage{}, map[string]json.RawMessage{}
	for id, events := range before.ThreadEvents {
		oldEvents[id], _ = json.Marshal(events)
	}
	for id, events := range after.ThreadEvents {
		newEvents[id], _ = json.Marshal(events)
	}
	d := runtimeDelta{Sequence: c.RuntimeSequence + 1, ThreadEvents: jsonDelta(oldEvents, newEvents),
		Executions: jsonDelta(keyedJSON(before.Executions, func(v PersistentEventExecution) string { return v.ExecutionID }), keyedJSON(after.Executions, func(v PersistentEventExecution) string { return v.ExecutionID })),
		Outbox:     jsonDelta(keyedJSON(before.Outbox, func(v EventLifecycleTransition) string { return v.ID }), keyedJSON(after.Outbox, func(v EventLifecycleTransition) string { return v.ID }))}
	a, _ := json.Marshal(before.MainEvents)
	b, _ := json.Marshal(after.MainEvents)
	if !bytes.Equal(a, b) {
		d.MainEvents = &after.MainEvents
	}
	if len(d.Executions) == 0 && len(d.Outbox) == 0 && d.MainEvents == nil && len(d.ThreadEvents) == 0 {
		return nil
	}
	// JSON object iteration cannot carry insertion order. Preserve it explicitly
	// for newly replayed records, especially claimed -> active transitions.
	for _, v := range after.Executions {
		if _, ok := d.Executions[v.ExecutionID]; ok {
			d.ExecutionOrder = append(d.ExecutionOrder, v.ExecutionID)
		}
	}
	for _, v := range after.Outbox {
		if _, ok := d.Outbox[v.ID]; ok {
			d.OutboxOrder = append(d.OutboxOrder, v.ID)
		}
	}
	if c.path != "" {
		data, err := json.Marshal(d)
		if err != nil {
			rollback()
			return err
		}
		data = append(data, '\n')
		// Establish a snapshot before the first journal, including existing definitions.
		if _, err := os.Stat(c.path); os.IsNotExist(err) {
			rollback()
			snapshot, _ := json.MarshalIndent(c, "", "  ")
			if err := atomicWriteFile(c.path, snapshot, 0600); err != nil {
				return err
			}
			c.EventExecutions = after.Executions
			c.EventLifecycleOutbox = after.Outbox
			c.MainEvents = after.MainEvents
			for i := range c.Threads {
				c.Threads[i].Events = after.ThreadEvents[c.Threads[i].ID]
			}
		}
		f, err := os.OpenFile(c.path+".runtime.jsonl", os.O_CREATE|os.O_RDWR|os.O_APPEND, 0600)
		if err != nil {
			rollback()
			return err
		}
		info, err := f.Stat()
		if err != nil {
			f.Close()
			rollback()
			return err
		}
		_, err = f.Write(data)
		if err == nil {
			err = f.Sync()
		}
		if err != nil {
			_ = f.Truncate(info.Size())
			_ = f.Sync()
			f.Close()
			rollback()
			return err
		}
		if err = f.Close(); err != nil {
			rollback()
			return err
		}
	}
	c.RuntimeSequence = d.Sequence
	if c.path != "" && c.RuntimeSequence%256 == 0 {
		snapshot, err := json.MarshalIndent(c, "", "  ")
		if err == nil {
			err = atomicWriteFile(c.path, snapshot, 0600)
		}
		if err == nil {
			_ = os.Truncate(c.path+".runtime.jsonl", 0)
		}
	}
	return nil
}
