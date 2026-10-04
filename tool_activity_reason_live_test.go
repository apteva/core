package core

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

const toolReasonLiveModel = "gpt-6.1-sol"

// Real Codex calls, pinned to Sol 6.1. Billing results are local fixtures;
// no production billing tool, agent, or conversation is invoked.
//
//	RUN_CODEX_TOOL_REASON_LIVE=1 go test -v -count=1 \
//	  -run '^TestIntegration_CodexGPT61SolToolReasons' -timeout 12m .
//
// Every raw call is checked before Core's fallback. An omission fails the
// sample immediately; samples never retry to hide model-contract failures.
func TestIntegration_CodexGPT61SolToolReasonsBillingWorkflow(t *testing.T) {
	token := toolReasonLiveCredential(t)
	for _, role := range []string{"main", "worker", "leader"} {
		for sample := 1; sample <= 2; sample++ {
			t.Run(fmt.Sprintf("%s/sample_%d", role, sample), func(t *testing.T) {
				registry := toolReasonBillingFixture()
				terminal := "done"
				if role == "main" {
					terminal = "conversations_send"
				}
				tools := []NativeTool{registry.Get("billing_customers_search").native, registry.Get("billing_customers_get_context").native, registry.Get(terminal).native}
				prompt := "In conversation live-reason-fixture, check the billing accounts for Acme Test and Beta Test. Make two separate customer searches in parallel, one for each exact name, then fetch billing context for both returned customer IDs. After both contexts arrive, report the findings using " + terminal + " exactly once. Perform the work yourself; do not delegate."
				messages := []Message{{Role: "system", Content: toolReasonLivePrompt(role, registry)}, {Role: "user", Content: prompt}}
				provider := toolReasonLiveProvider(t, token)
				ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
				defer cancel()
				searched := map[string]bool{}
				contexts := map[string]bool{}
				parallel := false
				calls := 0
				for turn := 1; turn <= 6; turn++ {
					response, err := provider.Chat(ctx, messages, toolReasonLiveModel, tools, nil, nil, nil)
					if err != nil {
						t.Fatalf("live %s turn=%d: %v", toolReasonLiveModel, turn, err)
					}
					if len(response.ToolCalls) == 0 {
						t.Fatalf("workflow stopped before its reporting tool, turn=%d checked_calls=%d", turn, calls)
					}
					parallel = parallel || len(response.ToolCalls) > 1
					messages = append(messages, Message{Role: "assistant", Content: response.Text, Reasoning: response.Reasoning, ToolCalls: response.ToolCalls, ProviderState: response.ProviderState})
					var results []ToolResult
					finished := false
					for _, call := range response.ToolCalls {
						assertLiveModelToolReason(t, registry, call)
						calls++
						var result string
						switch call.Name {
						case "billing_customers_search":
							name := strings.TrimSpace(call.Args["q"])
							id := "101"
							if name == "Beta Test" {
								id = "102"
							}
							if (name != "Acme Test" && name != "Beta Test") || searched[name] {
								t.Fatalf("expected one search per named customer; got q=%q", name)
							}
							searched[name] = true
							result = fmt.Sprintf(`{"count":1,"customers":[{"id":%s,"name":%q}],"has_more":false}`, id, name)
						case "billing_customers_get_context":
							id := call.Args["id"]
							name := "Acme Test"
							if id == "102" {
								name = "Beta Test"
							}
							if (id != "101" && id != "102") || !searched[name] || contexts[id] {
								t.Fatalf("unexpected customer context id=%q", id)
							}
							contexts[id] = true
							result = fmt.Sprintf(`{"customer":{"id":%s,"name":%q},"open_invoices":[],"recent_payments":[],"totals":{"currency":"USD","outstanding_cents":0}}`, id, name)
						case terminal:
							if len(contexts) != 2 || len(response.ToolCalls) != 1 {
								t.Fatal("report must be alone after both billing context results")
							}
							key := "message"
							if terminal == "conversations_send" {
								key = "text"
								if call.Args["conversation_id"] != "live-reason-fixture" {
									t.Fatal("report addressed a different conversation")
								}
							}
							if strings.TrimSpace(call.Args[key]) == "" {
								t.Fatal("report is empty")
							}
							finished = true
							result = `{"accepted":true}`
						default:
							t.Fatalf("unexpected tool %q", call.Name)
						}
						results = append(results, ToolResult{CallID: call.ID, ToolName: call.Name, Content: result})
					}
					messages = append(messages, Message{Role: "tool", ToolResults: results})
					if finished {
						if calls != 5 || !parallel {
							t.Fatalf("checked_calls=%d parallel=%v, want five calls and a parallel batch", calls, parallel)
						}
						t.Logf("model=%s verified %d raw model reasons across %d turns, including parallel calls and result continuation", toolReasonLiveModel, calls, turn)
						return
					}
				}
				t.Fatal("billing workflow exceeded six model turns")
			})
		}
	}
}

func TestIntegration_CodexGPT61SolToolReasonsArgumentShapes(t *testing.T) {
	token := toolReasonLiveCredential(t)
	for _, role := range []string{"main", "worker", "leader"} {
		for _, shape := range []string{"no_business_arguments", "optional_false", "invoice_search"} {
			t.Run(role+"/"+shape, func(t *testing.T) {
				registry := toolReasonBillingFixture()
				name, prompt := "billing_health", "Check billing service health using billing_health now. Perform only this one operation; do not report or delegate yet."
				if shape == "optional_false" {
					name = "billing_customers_search"
					prompt = "Look up the customer named Acme Test using billing_customers_search, with include_context explicitly false. Leave all other optional overrides unset. Perform only this one operation; do not report or delegate yet."
				}
				if shape == "invoice_search" {
					name = "billing_invoices_search"
					prompt = "Search for open invoices with billing_invoices_search, sorted by due_date and limited to 200 entries. Leave every other optional filter unset. Perform only this one operation; do not report or delegate yet."
				}
				tool := registry.Get(name).native
				ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
				defer cancel()
				response, err := toolReasonLiveProvider(t, token).Chat(ctx, []Message{{Role: "system", Content: toolReasonLivePrompt(role, registry)}, {Role: "user", Content: prompt}}, toolReasonLiveModel, []NativeTool{tool}, nil, nil, nil)
				if err != nil {
					t.Fatalf("live %s: %v", toolReasonLiveModel, err)
				}
				if len(response.ToolCalls) != 1 || response.ToolCalls[0].Name != name {
					t.Fatalf("expected exactly one %s call, got %d", name, len(response.ToolCalls))
				}
				call := response.ToolCalls[0]
				assertLiveModelToolReason(t, registry, call)
				if shape == "no_business_arguments" && len(call.Args) != 1 {
					t.Fatal("zero-argument operation should contain only _reason")
				}
				if shape == "optional_false" {
					if call.Args["include_context"] != "false" || call.Args["q"] != "Acme Test" || len(call.Args) != 3 {
						t.Fatal("required reason must coexist with explicit false and omission of other optional arguments")
					}
				}
				if shape == "invoice_search" {
					if call.Args["status"] != "open" || call.Args["sort"] != "due_date" || call.Args["limit"] != "200" || len(call.Args) != 4 {
						t.Fatal("invoice lookup must include _reason and the three requested filters only")
					}
				}
			})
		}
	}
}

func assertLiveModelToolReason(t *testing.T, registry *ToolRegistry, call NativeToolCall) {
	t.Helper()
	var raw map[string]any
	if call.RawArgs == "" || json.Unmarshal([]byte(call.RawArgs), &raw) != nil {
		t.Fatalf("tool %s has no inspectable raw model JSON", call.Name)
	}
	reason, ok := raw["_reason"].(string)
	if !ok || strings.TrimSpace(reason) == "" {
		t.Fatalf("model=%s tool=%s call=%s omitted a nonempty string _reason in its raw arguments", toolReasonLiveModel, call.Name, call.ID)
	}
	if call.Args["_reason"] != reason {
		t.Fatalf("tool %s reason changed between raw and parsed arguments", call.Name)
	}
	var canonical map[string]any
	if json.Unmarshal(call.CanonicalArgs, &canonical) != nil || canonical["_reason"] != reason {
		t.Fatalf("tool %s replay arguments lost the model reason", call.Name)
	}
	args := make(map[string]string, len(call.Args))
	for key, value := range call.Args {
		args[key] = value
	}
	data := (&Thinker{registry: registry}).prepareToolCallData(&toolCall{Name: call.Name, NativeID: call.ID, Args: args}, nil)
	if data.ReasonSource != "model" || data.Reason != strings.TrimSpace(reason) {
		t.Fatalf("tool %s used fallback despite a real model reason", call.Name)
	}
	if _, exists := data.Args["_reason"]; exists {
		t.Fatalf("tool %s dispatch retained _reason", call.Name)
	}
	t.Logf("model=%s tool=%s reason_source=model reason=%q", toolReasonLiveModel, call.Name, reason)
}

func toolReasonLiveCredential(t *testing.T) string {
	t.Helper()
	if testing.Short() || os.Getenv("RUN_CODEX_TOOL_REASON_LIVE") != "1" {
		t.Skip("set RUN_CODEX_TOOL_REASON_LIVE=1 without -short")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Fatal("live reason test requested but no valid Codex credential is available")
	}
	return token
}

func toolReasonLiveProvider(t *testing.T, token string) *OpenAINativeProvider {
	t.Helper()
	p := NewOpenAICodexProvider(token).(*OpenAINativeProvider)
	p.runtimeTokenURL, p.serverAPIKey = "", ""
	p.models = map[ModelTier]string{ModelLarge: toolReasonLiveModel, ModelMedium: toolReasonLiveModel, ModelSmall: toolReasonLiveModel}
	if p.accountID == "" {
		home, _ := os.UserHomeDir()
		data, _ := os.ReadFile(filepath.Join(home, ".codex", "auth.json"))
		var auth struct {
			Tokens struct {
				AccountID string `json:"account_id"`
			} `json:"tokens"`
		}
		if json.Unmarshal(data, &auth) == nil {
			p.accountID = auth.Tokens.AccountID
		}
	}
	return p
}

func toolReasonLivePrompt(role string, registry *ToolRegistry) string {
	directive := "Carry out the assigned billing operation with the available tools. All billing data and tools in this isolated test are local fixtures."
	if role == "main" {
		return buildSystemPrompt(directive, registry, "", nil, nil, nil, nil)
	}
	return formatThreadBasePrompt(role == "leader", false, "billing-"+role, "main") + "\n\n[DIRECTIVE]\n" + directive
}

func toolReasonBillingFixture() *ToolRegistry {
	registry := NewToolRegistry("")
	registry.Register(&ToolDef{Name: "billing_invoices_search", Description: "Filter invoices. q searches invoice number, notes, or exact invoice ID; it does not search customer names. For a named customer, use customers_search first (include_context=true for balances), then customer_id here if individual invoices are needed. Args: q, customer_id, status (draft|open|paid|void|uncollectible), provider (local|stripe), currency, since (RFC3339), until (RFC3339), min_total_cents, max_total_cents, sort (due_date), limit (default 50, max 200), offset.", InputSchema: map[string]any{
		"type": "object", "properties": map[string]any{
			"customer_id": map[string]any{"type": "integer"}, "status": map[string]any{"type": "string"},
			"provider": map[string]any{"type": "string"}, "currency": map[string]any{"type": "string"},
			"since": map[string]any{"type": "string"}, "until": map[string]any{"type": "string"},
			"min_total_cents": map[string]any{"type": "integer"}, "max_total_cents": map[string]any{"type": "integer"},
			"sort": map[string]any{"type": "string"}, "limit": map[string]any{"type": "integer"}, "offset": map[string]any{"type": "integer"},
		},
	}})
	registry.Register(&ToolDef{Name: "billing_customers_search", Description: "Find billing customers by name or email; start here for customer/account/balance checks. q searches the literal name/email phrase. Args: q, email, include_context, payments_limit, limit, offset.", InputSchema: map[string]any{
		"type": "object", "properties": map[string]any{
			"q": map[string]any{"type": "string"}, "email": map[string]any{"type": "string"},
			"include_context": map[string]any{"type": "boolean"}, "payments_limit": map[string]any{"type": "integer"},
			"limit": map[string]any{"type": "integer"}, "offset": map[string]any{"type": "integer"},
		},
	}})
	registry.Register(&ToolDef{Name: "billing_customers_get_context", Description: "Snapshot + open invoices + recent payments + lifetime totals. Args: id OR email, payments_limit (default 10).", InputSchema: map[string]any{
		"type": "object", "properties": map[string]any{"id": map[string]any{"type": "integer"}, "email": map[string]any{"type": "string"}, "payments_limit": map[string]any{"type": "integer"}},
	}})
	registry.Register(&ToolDef{Name: "billing_health", Description: "Check billing service health.", InputSchema: map[string]any{"type": "object", "properties": map[string]any{}, "additionalProperties": false}})
	registry.Register(&ToolDef{Name: "conversations_send", Description: "Send the final outcome into a conversation you participate in.", InputSchema: map[string]any{
		"type": "object", "properties": map[string]any{"conversation_id": map[string]any{"type": "string"}, "text": map[string]any{"type": "string"}}, "required": []string{"conversation_id", "text"},
	}})
	return registry
}
