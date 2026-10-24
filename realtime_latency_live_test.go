package core

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"
)

// Paid: checks the actual Sol text stream against successful fixture evidence,
// including delivery to the realtime context adapter before Chat returns.
func TestRealtimeVerifiedStreamSolLive(t *testing.T) {
	if testing.Short() || os.Getenv("RUN_REALTIME_LATENCY_LIVE") != "1" {
		t.Skip("set RUN_REALTIME_LATENCY_LIVE=1")
	}
	loadIntegrationEnv()
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		t.Fatal("OPENAI_API_KEY required")
	}
	rt, voice, worker, w := streamLatencyFixture(t)
	sentence := "Le tarif confirmé est de 42 euros."
	provider := NewOpenAINativeProvider(key)
	applyModelOverrides(provider, map[string]string{"large": "gpt-6.1-sol", "medium": "gpt-6.1-sol", "small": "gpt-6.1-sol"})
	worker.model = ModelLarge
	messages := []Message{
		{Role: "system", Content: "Répondez en français. Commencez par recopier exactement la phrase answer du résultat vérifié, puis expliquez en trois phrases courtes que le tarif est informatif et qu'aucune réservation n'a été créée. Aucun nouvel outil n'est nécessaire."},
		{Role: "user", Content: "Quel est le tarif ? Est-ce déjà réservé ?"},
		{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "price", Name: "lookup", Args: map[string]string{}}}},
		{Role: "user", ToolResults: []ToolResult{{CallID: "price", ToolName: "lookup", Content: `{"answer":"` + sentence + `","booking_created":false}`}}},
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	type result struct {
		response ChatResponse
		err      error
		at       time.Time
	}
	done := make(chan result, 1)
	started := time.Now()
	go func() {
		r, e := worker.thinkWithProviderMessages(ctx, provider, messages)
		done <- result{r, e, time.Now()}
	}()
	var first time.Time
	for {
		select {
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		case <-rt.sub.Wake:
			for _, e := range rt.sub.DrainTargeted() {
				rt.consumeThreadWork(e)
				if e.Work != nil && e.Work.Kind == "result_chunk" && first.IsZero() {
					first = time.Now()
				}
			}
		case r := <-done:
			if r.err != nil {
				t.Fatal(r.err)
			}
			if first.IsZero() || !first.Before(r.at) {
				t.Fatal("no verified chunk delivered before completion")
			}
			if len(voice.texts) != 1 || voice.texts[0].Content != sentence {
				t.Fatalf("unexpected early result: %+v", voice.texts)
			}
			final := w
			final.Kind = "result"
			final.Worker = "worker"
			final.Status = "completed"
			rt.consumeThreadWork(Event{Work: &final, Text: r.response.Text})
			if len(voice.texts) != 2 || strings.Contains(voice.texts[1].Content, sentence) {
				t.Fatal("final repeated verified sentence")
			}
			t.Logf("model=gpt-6.1-sol first_verified_context_ms=%d completion_ms=%d lead_ms=%d", first.Sub(started).Milliseconds(), r.at.Sub(started).Milliseconds(), r.at.Sub(first).Milliseconds())
			return
		}
	}
}
