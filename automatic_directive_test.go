package core

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// operatorDirective is shaped like a real triage directive: the word
// "conversation" recurs for CRM email threads, while two sentences ask for
// actions in the Chat app.
const operatorDirective = `# Role
You triage incoming customer email in the CRM app for the operator.

# Rules
- Every hour, check the CRM inbox for open email conversations.
- Review each open conversation and decide whether the conversation is important.
- Close every conversation after triaging it. Never delete a conversation.
- When a conversation is important, alert the operator in Chat with the sender and subject.
- Every day at 18:00, file one daily summary report for the operator covering all conversations.

# Tone
- Be concise.`

func operatorDirectiveThinker(t *testing.T, maxTools int) *Thinker {
	t.Helper()
	t.Setenv("APTEVA_TOOL_SEARCH", "on")
	wide := map[string]any{"type": "object", "properties": map[string]any{}}
	for _, field := range []string{"conversation_id", "conversation_status", "conversation_subject", "conversation_sender", "conversation_summary", "open", "important", "operator", "email", "customer"} {
		wide["properties"].(map[string]any)[field] = map[string]any{"type": "string", "description": "The " + strings.ReplaceAll(field, "_", " ") + " of the conversation."}
	}
	chat := []mcpToolDef{
		mkTool("alert", "Raise an alert. Set severity explicitly."),
		mkTool("report", "File a substantive report with a title and a summary."),
		{Name: "keep_thread", Description: "Keep this conversation thread awake while the conversation still has open work for the operator.", InputSchema: wide},
		{Name: "subscribe", Description: "Subscribe a conversation to events about open and important conversations for the operator.", InputSchema: wide},
	}
	crm := []mcpToolDef{
		mkTool("conversations_inbox", "Triage queue of open email conversations, newest first."),
		{Name: "contacts_update", Description: "Update a customer contact and the conversation fields of each open conversation.", InputSchema: wide},
		mkTool("contacts_set_conversation_status", "Close or reopen a conversation."),
	}
	ix, r := NewToolIndex(), NewToolRegistry("")
	ix.Add("chat", chat, false)
	ix.Add("crm", crm, false)
	registerTestMCPTools(r, "chat", chat)
	registerTestMCPTools(r, "crm", crm)
	th := &Thinker{threadID: "main", toolIndex: ix, registry: r, activeTools: map[string]bool{}, directive: operatorDirective, telemetry: NewTelemetry()}
	th.config = &Config{AutomaticToolLoading: &AutomaticToolLoadingConfig{Enabled: true, MaxTools: maxTools, MaxSchemaTokens: 6000}}
	return th
}

func TestDirectiveUnits(t *testing.T) {
	got := directiveUnits("# Rules\n- Check the inbox at 18:00 every day. Close it; then report v1.2 results!\n1. Be brief.\n\n* Never delete a conversation")
	want := []string{"Check the inbox at 18:00 every day", "then report v1.2 results", "Never delete a conversation"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("units = %q, want %q", got, want)
	}
	if units := directiveUnits(strings.Repeat("Check the inbox now.\n", 500)); len(units) != directiveUnitLimit {
		t.Fatalf("units = %d, want the limit %d", len(units), directiveUnitLimit)
	}
}

// One search over the whole directive ranks the tools whose large schemas
// repeat its commonest words first. Matching sentence by sentence must load
// the tools the action sentences ask for, within a tight tool budget.
func TestDirectiveSentencesLoadTheToolsTheyAskFor(t *testing.T) {
	th := operatorDirectiveThinker(t, 4)
	whole := names(th.toolIndex.Search(th.automaticQueryNormalizer()(operatorDirective), 3, true))
	if containsString(whole, "chat_alert") {
		t.Fatalf("fixture no longer shows the problem: whole-directive search already ranks chat_alert in %v", whole)
	}
	selected := th.prepareAutomaticTools(th.config.GetAutomaticToolLoading())
	for _, name := range []string{"chat_alert", "chat_report", "crm_conversations_inbox"} {
		if !selected[name] {
			t.Fatalf("%s not loaded; selection = %v", name, th.automaticTools.names)
		}
		if th.automaticTools.reasons[name] != "directive" {
			t.Fatalf("%s reason = %q", name, th.automaticTools.reasons[name])
		}
	}
	// The sentence names the tool: its server and every word of its name.
	order := th.directiveToolOrder(operatorDirective, th.automaticQueryNormalizer(), []string{"chat", "crm"}, true)
	firstSearched := len(order)
	alert := -1
	for i, entry := range order {
		if entry.Match != "named" && i < firstSearched {
			firstSearched = i
		}
		if entry.Name == "chat_alert" && entry.Match == "named" {
			alert = i
		}
	}
	if alert < 0 || alert > firstSearched {
		t.Fatalf("named tool does not lead the searched matches: %v", names(order))
	}
}

func TestDirectiveSentencesRespectGrants(t *testing.T) {
	th := operatorDirectiveThinker(t, 4)
	th.toolAllowlist = map[string]bool{"chat_report": true, "crm_conversations_inbox": true}
	selected := th.prepareAutomaticTools(th.config.GetAutomaticToolLoading())
	if selected["chat_alert"] || !selected["chat_report"] {
		t.Fatalf("selection ignored grants: %v", th.automaticTools.names)
	}
}

// With stable selection on, an earlier selection is carried forward. Tools
// picked for an old directive must be matched again when the directive changes.
func TestDirectiveChangeRematchesDirectiveTools(t *testing.T) {
	th := operatorDirectiveThinker(t, 2)
	th.directive = "Keep the conversation thread awake. Subscribe the conversation to events."
	th.prepareAutomaticTools(th.config.GetAutomaticToolLoading())
	before := append([]string(nil), th.automaticTools.names...)
	if len(before) == 0 || containsString(before, "chat_alert") {
		t.Fatalf("unexpected first selection %v", before)
	}
	th.directive = "When something is important, alert the operator in Chat."
	selected := th.prepareAutomaticTools(th.config.GetAutomaticToolLoading())
	if !selected["chat_alert"] {
		t.Fatalf("old directive's tools kept the budget: before=%v after=%v", before, th.automaticTools.names)
	}
}

func TestAutoloadLogsCandidatesLeftOut(t *testing.T) {
	th := operatorDirectiveThinker(t, 2)
	th.prepareAutomaticTools(th.config.GetAutomaticToolLoading())
	events, _ := th.telemetry.Events(0)
	var data struct {
		Selected  []string `json:"selected"`
		NotLoaded []struct {
			Name   string  `json:"name"`
			Source string  `json:"source"`
			Rank   int     `json:"rank"`
			Score  float64 `json:"score"`
			Reason string  `json:"reason"`
		} `json:"not_loaded"`
	}
	found := false
	for _, event := range events {
		if event.Type == "tool.autoload" {
			found = true
			if err := json.Unmarshal(event.Data, &data); err != nil {
				t.Fatal(err)
			}
		}
	}
	if !found || len(data.Selected) != 2 || len(data.NotLoaded) == 0 {
		t.Fatalf("tool.autoload selected=%v not_loaded=%v", data.Selected, data.NotLoaded)
	}
	for _, left := range data.NotLoaded {
		if left.Name == "" || left.Source != "directive" || left.Rank < 1 || left.Reason != "tool_count" {
			t.Fatalf("not_loaded entry lacks its name, source, rank or reason: %+v", left)
		}
		if containsString(data.Selected, left.Name) {
			t.Fatalf("%s is listed as both loaded and left out", left.Name)
		}
	}
}
