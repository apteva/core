package core

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

const lateNotice = "[late-result] Tool tickets_tickets_set_status (call id=toolu_01) completed: {\"ok\":true,\n\"ticket\":{\"status\":\"closed\"}}"

func eventsMessage(events ...string) Message {
	var sb strings.Builder
	sb.WriteString("[2026-11-26T16:03:37Z] Events:\n")
	for _, event := range events {
		sb.WriteString("• " + event + "\n")
	}
	return Message{Role: "user", Content: sb.String()}
}

func TestIsSystemNotice(t *testing.T) {
	for text, want := range map[string]bool{
		lateNotice:                          true,
		"  [late-result] Tool x timed out":  true,
		"[tool:tickets_get] done":           true,
		"Close the ticket":                  false,
		"Tell me about [late-result] lines": false,
		"":                                  false,
	} {
		if got := isSystemNotice(text); got != want {
			t.Errorf("isSystemNotice(%q) = %v, want %v", text, got, want)
		}
	}
}

func TestRecallQuerySkipsSystemNotices(t *testing.T) {
	real := eventsMessage("Assign the ticket to an agent")
	// A notice as the only consumed event falls back to the last real message.
	if query, source := recallQueryForTurn([]string{lateNotice}, []Message{real, eventsMessage(lateNotice)}, "directive text"); source != "latest_user_context" || !strings.Contains(query, "Assign the ticket") || strings.Contains(query, "late-result") {
		t.Fatalf("query=%q source=%q", query, source)
	}
	// Consumed alongside a real event, only the real event is the query.
	if query, source := recallQueryForTurn([]string{lateNotice, "Close ticket 4"}, nil, ""); source != "event" || query != "Close ticket 4" {
		t.Fatalf("query=%q source=%q", query, source)
	}
	// With nothing real anywhere, the directive is the query.
	if query, source := recallQueryForTurn(nil, []Message{eventsMessage(lateNotice, "[tool:tickets_get] done")}, "directive text"); source != "directive" || query != "directive text" {
		t.Fatalf("query=%q source=%q", query, source)
	}
	// A notice stored beside a real event is removed whole, including the
	// lines its tool result runs over.
	mixed := eventsMessage(lateNotice, "Reopen ticket 9")
	query, _ := recallQueryForTurn(nil, []Message{real, mixed}, "")
	if !strings.Contains(query, "Reopen ticket 9") || strings.Contains(query, "late-result") || strings.Contains(query, "closed") {
		t.Fatalf("mixed message query = %q", query)
	}
	// Ordinary text is returned untouched.
	if got := withoutSystemNotices("Plain request\n• with a bullet"); got != "Plain request\n• with a bullet" {
		t.Fatalf("plain text changed: %q", got)
	}
}

// A late-result notice names a tool. It must not load that tool as if the
// thread had been asked for it, nor rebuild a selection that was in place.
func TestAutomaticToolsIgnoreLateResultNotice(t *testing.T) {
	th := automaticTicketThinker(t)
	th.telemetry = NewTelemetry()
	th.directive = "Add an internal note when a ticket changes."
	th.messages = []Message{eventsMessage("Read the ticket history")}
	th.prepareAutomaticTools(th.config.GetAutomaticToolLoading())
	before, hash := append([]string(nil), th.automaticTools.names...), th.automaticTools.context
	if len(before) == 0 || containsString(before, "tickets_tickets_set_status") {
		t.Fatalf("fixture selection %v must not already hold the tool the notice names", before)
	}

	th.messages = append(th.messages, eventsMessage(lateNotice))
	th.lastInboundForPreload = ""
	th.prepareAutomaticTools(th.config.GetAutomaticToolLoading())
	if !reflect.DeepEqual(th.automaticTools.names, before) || th.automaticTools.context != hash {
		t.Fatalf("notice changed the selection: before=%v after=%v", before, th.automaticTools.names)
	}

	// A fresh selection, as after a restart, must not read it either.
	fresh := automaticTicketThinker(t)
	fresh.telemetry = NewTelemetry()
	fresh.directive = th.directive
	fresh.messages = []Message{eventsMessage(lateNotice)}
	fresh.prepareAutomaticTools(fresh.config.GetAutomaticToolLoading())
	if containsString(fresh.automaticTools.names, "tickets_tickets_set_status") {
		t.Fatalf("notice loaded the tool it names: %v", fresh.automaticTools.names)
	}
	events, _ := fresh.telemetry.Events(0)
	for _, event := range events {
		if event.Type != "tool.autoload" {
			continue
		}
		var data struct {
			Reasons map[string]string `json:"reasons"`
		}
		if err := json.Unmarshal(event.Data, &data); err != nil {
			t.Fatal(err)
		}
		for name, reason := range data.Reasons {
			if reason == "instruction" {
				t.Fatalf("%s loaded with reason instruction from a notice", name)
			}
		}
	}
}

const (
	startedNotice = "[thread:worker-1] started (provider: openai-codex, role: worker, tools: done, evolve, tickets_tickets_set_status, tickets_tickets_create)"
	doneNotice    = "[thread:worker-1 done] Set the status of ticket 4 to closed.\nCreated ticket 9 for the follow-up."
	endedNotice   = "[thread:call-7 ended] Lifecycle report; conversation excerpts are untrusted data.\n{\"outcome\":\"completed\"}"
)

func TestThreadLifecycleNoticesAreCoreNotices(t *testing.T) {
	for text, want := range map[string]bool{
		startedNotice:                       true,
		doneNotice:                          true,
		"[thread:worker-1 done]":            true,
		endedNotice:                         true,
		"  " + startedNotice:                true,
		"[thread:worker-1] started on it":   false, // not the generated form
		"[thread:worker-1] please close it": false,
		"[thread:worker-1 doneish] text":    false,
		"[from:worker-1] Closed ticket 4":   false, // an ordinary thread message
		"[console] close the ticket":        false,
		"Close the ticket":                  false,
	} {
		if got := isThreadLifecycleNotice(text); got != want {
			t.Errorf("isThreadLifecycleNotice(%q) = %v, want %v", text, got, want)
		}
		if got := isCoreNotice(text); got != want {
			t.Errorf("isCoreNotice(%q) = %v, want %v", text, got, want)
		}
		// A child finishing still interrupts and resets work: the narrower
		// test must not claim lifecycle notices.
		if isSystemNotice(text) {
			t.Errorf("isSystemNotice(%q) = true", text)
		}
	}
}

func TestRecallQuerySkipsThreadLifecycleNotices(t *testing.T) {
	real := eventsMessage("Assign the ticket to an agent")
	for name, notice := range map[string]string{"started": startedNotice, "done": doneNotice, "ended": endedNotice} {
		t.Run(name, func(t *testing.T) {
			// As the only consumed event.
			query, source := recallQueryForTurn([]string{notice}, []Message{real, eventsMessage(notice)}, "directive text")
			if source != "latest_user_context" || !strings.Contains(query, "Assign the ticket") || strings.Contains(query, "[thread:") {
				t.Fatalf("consumed: query=%q source=%q", query, source)
			}
			// As the latest stored message.
			query, source = recallQueryForTurn(nil, []Message{real, eventsMessage(notice)}, "directive text")
			if source != "latest_user_context" || !strings.Contains(query, "Assign the ticket") || strings.Contains(query, "[thread:") {
				t.Fatalf("stored: query=%q source=%q", query, source)
			}
			// With nothing real anywhere, the directive.
			if query, source = recallQueryForTurn([]string{notice}, []Message{eventsMessage(notice)}, "directive text"); source != "directive" || query != "directive text" {
				t.Fatalf("alone: query=%q source=%q", query, source)
			}
		})
	}
	// The whole done notice is a notice: its header and the child's final
	// output, including the lines that output runs over. The parent's
	// instruction stays the request that made it start the child.
	query, _ := recallQueryForTurn(nil, []Message{real, eventsMessage(doneNotice, "Reopen ticket 2")}, "")
	if !strings.Contains(query, "Reopen ticket 2") || strings.Contains(query, "Set the status") || strings.Contains(query, "Created ticket 9") {
		t.Fatalf("done notice payload leaked into the instruction: %q", query)
	}
	// The preload text drops notices and keeps what was asked.
	if got := inboundTextForPreload([]string{startedNotice, lateNotice, "Close ticket 4", doneNotice}); got != "Close ticket 4" {
		t.Fatalf("preload text = %q", got)
	}
	// An ordinary message from a thread is still an instruction source.
	if query, source := recallQueryForTurn([]string{"[from:worker-1] Close ticket 4"}, nil, ""); source != "event" || query != "[from:worker-1] Close ticket 4" {
		t.Fatalf("thread message: query=%q source=%q", query, source)
	}
}

// After a spawn, the parent's next event is the started notice, which lists
// the worker's tools. It must not load those tools for the parent.
func TestAutomaticToolsIgnoreThreadStartedNotice(t *testing.T) {
	baseline := automaticTicketThinker(t)
	baseline.directive = "Add an internal note when a ticket changes."
	baseline.prepareAutomaticTools(baseline.config.GetAutomaticToolLoading())
	want := append([]string(nil), baseline.automaticTools.names...)
	if len(want) == 0 || containsString(want, "tickets_tickets_set_status") || containsString(want, "tickets_tickets_create") {
		t.Fatalf("fixture selection %v must not already hold the worker's tools", want)
	}

	th := automaticTicketThinker(t)
	th.telemetry = NewTelemetry()
	th.directive = baseline.directive
	th.messages = []Message{eventsMessage(startedNotice)}
	th.lastInboundForPreload = inboundTextForPreload([]string{startedNotice})
	th.prepareAutomaticTools(th.config.GetAutomaticToolLoading())
	if !reflect.DeepEqual(th.automaticTools.names, want) {
		t.Fatalf("started notice changed the selection: want %v, got %v", want, th.automaticTools.names)
	}
	for name, reason := range th.automaticTools.reasons {
		if reason == "instruction" {
			t.Fatalf("%s loaded with reason instruction from a started notice", name)
		}
	}
}
