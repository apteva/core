package core

import (
	"bufio"
	"encoding/json"
	"io"
	"strings"
	"testing"
)

// The managed MCP runner mirrors structuredContent in a text block, as the MCP
// spec recommends. These tests cover decoding that shape and its direct receipt.

const bookingJSON = `{"status":"confirmed","date":"Tuesday","reference":"R42"}`

func structuredPayload(structured string, content ...map[string]any) map[string]any {
	payload := map[string]any{"content": content}
	if structured != "" {
		payload["structuredContent"] = json.RawMessage(structured)
	}
	return payload
}

func textBlock(text string) map[string]any { return map[string]any{"type": "text", "text": text} }

func decodeWirePayload(t *testing.T, payload map[string]any) ToolResponse {
	t.Helper()
	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	var result mcpCallResult
	if err := json.Unmarshal(raw, &result); err != nil {
		t.Fatal(err)
	}
	resp, err := decodeMCPResult(result)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func receiptThinker(t *testing.T) (*Thinker, ThreadWork) {
	t.Helper()
	th := newTestThinkerFull()
	t.Cleanup(th.Stop)
	th.telemetry = &Telemetry{notify: make(chan struct{}, 1), quit: make(chan struct{})}
	th.work = &threadWorkState{}
	th.executionPolicy = &RealtimeExecutionConfig{ResultTools: []DirectToolResultPolicy{bookingResultPolicy()}}
	w := ThreadWork{ID: "request", Revision: 1, Origin: "main", Kind: "request"}
	th.work.observe(&w)
	return th, w
}

func lastReceiptReason(t *testing.T, th *Thinker) string {
	t.Helper()
	events, _ := th.telemetry.StoredEvents(0)
	for i := len(events) - 1; i >= 0; i-- {
		if events[i].Type == "thread.work_direct_result_evaluated" {
			var data struct{ Reason string }
			_ = json.Unmarshal(events[i].Data, &data)
			return data.Reason
		}
	}
	t.Fatal("no receipt evaluation recorded")
	return ""
}

func TestDecodeMCPResultDeduplicatesMirroredStructuredContent(t *testing.T) {
	reordered := "{\n  \"reference\": \"R42\", \"date\": \"Tuesday\",\n  \"status\": \"confirmed\"\n}"
	cases := map[string]struct {
		payload map[string]any
		want    string
	}{
		"identical":         {structuredPayload(bookingJSON, textBlock(bookingJSON)), bookingJSON},
		"reordered_spacing": {structuredPayload(bookingJSON, textBlock(reordered)), reordered},
		"structured_only":   {structuredPayload(bookingJSON), bookingJSON},
		"distinct_text":     {structuredPayload(bookingJSON, textBlock("Booked."), textBlock(bookingJSON)), "Booked.\n" + bookingJSON},
		"conflicting":       {structuredPayload(bookingJSON, textBlock(`{"status":"pending"}`)), `{"status":"pending"}` + "\n" + bookingJSON},
		"number_precision":  {structuredPayload(`{"n":12345678901234567890}`, textBlock(`{"n":12345678901234567891}`)), `{"n":12345678901234567891}` + "\n" + `{"n":12345678901234567890}`},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			resp := decodeWirePayload(t, tc.payload)
			if resp.Text != tc.want {
				t.Fatalf("text=%q want %q", resp.Text, tc.want)
			}
			if !equivalentJSON(resp.StructuredContent, tc.payload["structuredContent"].(json.RawMessage)) {
				t.Fatalf("structured content not preserved: %s", resp.StructuredContent)
			}
		})
	}
}

func TestStructuredJSONRejectsAmbiguousKeys(t *testing.T) {
	for _, raw := range []string{
		`{"status":"failed","status":"confirmed"}`,
		`{"status":"confirmed","status":"confirmed"}`,
		`{"status":"failed","\u0073tatus":"confirmed"}`,
		`{"nested":{"error":true,"error":false}}`,
		`[{"status":"failed","status":"confirmed"}]`,
		`{} {}`, `{} trailing`, `{"x":1,}`, `[1,]`, `{"x":`,
	} {
		t.Run(raw, func(t *testing.T) {
			if _, err := decodeSingleJSON([]byte(raw)); err == nil {
				t.Fatal("ambiguous or malformed JSON accepted")
			}
			if equivalentJSON([]byte(raw), []byte(raw)) {
				t.Fatal("ambiguous JSON eligible for deduplication")
			}
		})
	}
	// Repeated names in different objects are unambiguous. Preserve exact
	// large numbers, nulls, empty containers and escaped JSON strings.
	a := `{"items":[{"id":12345678901234567890},{"id":2}],"empty":[],"nil":null,"s":"a\\b"}`
	b := `{"s":"a\\b","nil":null,"empty":[],"items":[{"id":12345678901234567890},{"id":2}]}`
	if !equivalentJSON([]byte(a), []byte(b)) {
		t.Fatal("valid reordered structured result no longer deduplicates")
	}
	deep := strings.Repeat("[", 1002) + "0" + strings.Repeat("]", 1002)
	if _, err := decodeSingleJSON([]byte(deep)); err == nil {
		t.Fatal("eligibility parser has no nesting bound")
	}
}

func TestDirectReceiptRejectsDuplicateKeysWithoutLosingEvidence(t *testing.T) {
	for _, raw := range []string{
		`{"status":"failed","status":"confirmed","date":"Tuesday","reference":"R42"}`,
		`{"status":"failed","\u0073tatus":"confirmed","date":"Tuesday","reference":"R42"}`,
		`{"status":"confirmed","date":"Tuesday","reference":"R42","detail":{"failed":true,"failed":false}}`,
	} {
		for _, shape := range []string{"text_only", "structured_only", "conflicting_mirror", "identical_mirror"} {
			t.Run(shape+raw, func(t *testing.T) {
				var payload map[string]any
				switch shape {
				case "text_only":
					payload = structuredPayload("", textBlock(raw))
				case "structured_only":
					payload = structuredPayload(raw)
				case "conflicting_mirror":
					payload = structuredPayload(bookingJSON, textBlock(raw))
				case "identical_mirror":
					payload = structuredPayload(raw, textBlock(raw))
				}
				resp := decodeWirePayload(t, payload)
				if !strings.Contains(resp.Text, raw) {
					t.Fatal("ambiguous evidence removed from normal continuation")
				}
				if shape == "conflicting_mirror" && resp.Text != raw+"\n"+bookingJSON {
					t.Fatal("conflicting representations incorrectly merged")
				}
				th, w := receiptThinker(t)
				call := toolCall{Name: "book", NativeID: "call", Args: map[string]string{completeAssignmentArg: "true"}, work: &w}
				if receipt := th.directToolReceipt(call, resp); receipt != nil {
					t.Fatalf("ambiguous result produced direct confirmation: %s", receipt.Content)
				}
			})
		}
	}
}

func TestDecodeMCPResultKeepsMediaWithMirroredStructuredContent(t *testing.T) {
	resp := decodeWirePayload(t, structuredPayload(bookingJSON,
		textBlock(bookingJSON),
		mediaImageBlock("image/png", []byte{0x89, 0x50}),
		map[string]any{"type": "audio", "mimeType": "audio/wav", "data": "AQIDBA=="},
		map[string]any{"type": "resource_link", "uri": "https://example.com/r.pdf"},
	))
	if len(resp.Image) == 0 || len(resp.Parts) != 1 || resp.Parts[0].InputAudio == nil {
		t.Fatalf("media lost: image=%d parts=%+v", len(resp.Image), resp.Parts)
	}
	if resp.Text != bookingJSON+"\nhttps://example.com/r.pdf" {
		t.Fatalf("text=%q", resp.Text)
	}
}

func TestDirectReceiptStructuredContent(t *testing.T) {
	cases := map[string]struct {
		payload map[string]any
		reason  string // "eligible" or the expected rejection reason
	}{
		"mirrored":                {structuredPayload(bookingJSON, textBlock(bookingJSON)), "eligible"},
		"mirrored_reordered":      {structuredPayload(bookingJSON, textBlock(`{"reference":"R42","status":"confirmed","date":"Tuesday"}`)), "eligible"},
		"structured_only":         {structuredPayload(bookingJSON), "eligible"},
		"text_only":               {structuredPayload("", textBlock(bookingJSON)), "eligible"},
		"text_only_whitespace":    {structuredPayload("", textBlock("\n  "+bookingJSON+"\n")), "eligible"},
		"conflicting":             {structuredPayload(bookingJSON, textBlock(`{"status":"failed","error":"slot taken"}`)), "conflicting_representations"},
		"concatenated_text":       {structuredPayload("", textBlock(bookingJSON), textBlock(`{"status":"failed"}`)), "malformed_result"},
		"text_with_trailing_text": {structuredPayload("", textBlock(bookingJSON+" but the SMS failed")), "malformed_result"},
		"explanatory_text":        {structuredPayload(bookingJSON, textBlock("Note: payment still pending."), textBlock(bookingJSON)), "conflicting_representations"},
		"structured_not_object":   {structuredPayload(`["confirmed"]`, textBlock(`["confirmed"]`)), "malformed_result"},
		"structured_failed":       {structuredPayload(`{"status":"blocked","date":"Tuesday","reference":"R42"}`), "success_status_failed"},
		"structured_missing":      {structuredPayload(`{"status":"confirmed"}`, textBlock(`{"status":"confirmed"}`)), "result_not_eligible"},
		"media":                   {structuredPayload(bookingJSON, textBlock(bookingJSON), mediaImageBlock("image/png", []byte{1})), "result_not_eligible"},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			th, w := receiptThinker(t)
			resp := decodeWirePayload(t, tc.payload)
			args := map[string]string{completeAssignmentArg: "true"}
			receipt := th.directToolReceipt(toolCall{Name: "book", NativeID: "call", Args: args, work: &w}, resp)
			if (receipt != nil) != (tc.reason == "eligible") {
				t.Fatalf("receipt=%+v want %s", receipt, tc.reason)
			}
			if got := lastReceiptReason(t, th); got != tc.reason {
				t.Fatalf("reason=%q want %q", got, tc.reason)
			}
			if receipt != nil && receipt.Content != `{"date":"Tuesday","reference":"R42","status":"confirmed"}` {
				t.Fatalf("receipt content=%s", receipt.Content)
			}
		})
	}
}

func TestDirectReceiptStructuredErrorAndGuards(t *testing.T) {
	for _, name := range []string{"is_error", "missing_decision", "not_terminal", "stale"} {
		t.Run(name, func(t *testing.T) {
			th, w := receiptThinker(t)
			payload := structuredPayload(bookingJSON, textBlock(bookingJSON))
			args := map[string]string{completeAssignmentArg: "true"}
			switch name {
			case "is_error":
				payload["isError"] = true
			case "missing_decision":
				delete(args, completeAssignmentArg)
			case "not_terminal":
				args[completeAssignmentArg] = "false"
			case "stale":
				newer := w
				newer.Revision++
				th.work.advance(newer)
			}
			if receipt := th.directToolReceipt(toolCall{Name: "book", NativeID: "call", Args: args, work: &w}, decodeWirePayload(t, payload)); receipt != nil {
				t.Fatal("guard bypassed by structured content")
			}
		})
	}
}

func TestDirectReceiptMirroredResultCompletesOnce(t *testing.T) {
	th, w := receiptThinker(t)
	resp := decodeWirePayload(t, structuredPayload(bookingJSON, textBlock(bookingJSON)))
	args := map[string]string{completeAssignmentArg: "true"}
	receipt := th.directToolReceipt(toolCall{Name: "book", NativeID: "call", Args: args, work: &w}, resp)
	if receipt == nil {
		t.Fatal("no receipt for mirrored result")
	}
	th.messages = append(th.messages, Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "call", Name: "book", Args: args}}})
	result := ToolResult{CallID: "call", ToolName: "book", Content: resp.Text, DirectReceipt: receipt}
	for i, want := range []bool{true, false} {
		done, err := th.completeDirectToolResult([]ToolResult{result}, false)
		if err != nil || done != want {
			t.Fatalf("delivery %d: done=%v err=%v", i+1, done, err)
		}
	}
	events, _ := th.telemetry.StoredEvents(0)
	direct := 0
	for _, e := range events {
		if e.Type == "thread.work_direct_result" {
			direct++
			if !strings.Contains(string(e.Data), `"narration_requests_skipped":1`) {
				t.Fatalf("event data=%s", e.Data)
			}
		}
	}
	if direct != 1 {
		t.Fatalf("direct result events=%d", direct)
	}
}

// stdioResultMCP serves one fixed tools/call result over an in-memory stdio pipe.
func stdioResultMCP(t *testing.T, payload any) *MCPServer {
	t.Helper()
	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	serverReader, clientWriter := io.Pipe()
	clientReader, serverWriter := io.Pipe()
	t.Cleanup(func() { clientWriter.Close(); serverWriter.Close() })
	go func() {
		scanner := bufio.NewScanner(serverReader)
		for scanner.Scan() {
			var req jsonRPCRequest
			if json.Unmarshal(scanner.Bytes(), &req) != nil || req.Method != "tools/call" {
				continue
			}
			data, _ := json.Marshal(jsonRPCResponse{JSONRPC: "2.0", ID: req.ID, Result: raw})
			serverWriter.Write(append(data, '\n'))
		}
	}()
	srv := &MCPServer{Name: "booking", stdin: clientWriter, scanner: bufio.NewScanner(clientReader), pending: make(map[int64]chan jsonRPCResponse)}
	go srv.readLoop()
	return srv
}

func TestDirectReceiptMirroredResultOverTransports(t *testing.T) {
	payload := structuredPayload(bookingJSON, textBlock(bookingJSON))
	for name, conn := range map[string]MCPConn{"http": mediaResultMCP(t, payload), "stdio": stdioResultMCP(t, payload)} {
		t.Run(name, func(t *testing.T) {
			resp := mcpProxyHandler(conn, "book")(map[string]string{})
			if resp.Text != bookingJSON || !equivalentJSON(resp.StructuredContent, []byte(bookingJSON)) {
				t.Fatalf("decoded text=%q structured=%s", resp.Text, resp.StructuredContent)
			}
			th, w := receiptThinker(t)
			if th.directToolReceipt(toolCall{Name: "book", NativeID: "call", Args: map[string]string{completeAssignmentArg: "true"}, work: &w}, resp) == nil {
				t.Fatalf("no receipt over %s (reason %s)", name, lastReceiptReason(t, th))
			}
		})
	}
}
