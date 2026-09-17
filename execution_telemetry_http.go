package core

import (
	"context"
	"errors"
	"io"
	"net/http"
	"sync"
	"time"
)

type requestTraceKey struct{}

// Observe each physical submission, including adapter-local auth/options
// retries. Never record URLs, headers, credentials or request/response bodies.
func tracedProviderHTTP(req *http.Request) (*http.Response, error) {
	r, _ := req.Context().Value(requestTraceKey{}).(*requestTrace)
	if r == nil {
		return llmHTTPClient.Do(req)
	}
	started := time.Now()
	id := newTraceID("http")
	emit := func(phase string, data map[string]any) {
		data["http_attempt_id"], data["provider"], data["model"] = id, r.provider, r.model
		data["elapsed_ms"] = time.Since(started).Milliseconds()
		r.t.emitTrace("llm.http."+phase, data, r.ids)
	}
	emit("started", map[string]any{})
	resp, err := llmHTTPClient.Do(req)
	if err != nil {
		outcome := "failed"
		if errors.Is(err, context.Canceled) {
			outcome = "cancelled"
		}
		if errors.Is(err, context.DeadlineExceeded) {
			outcome = "timeout"
		}
		emit("finished", map[string]any{"outcome": outcome, "reason": "http_request_failed"})
		return resp, err
	}
	ids := extractProviderRequestIDs(resp.Header)
	emit("headers", map[string]any{"status_code": resp.StatusCode, "provider_request_ids": ids})
	resp.Body = &tracedProviderBody{ReadCloser: resp.Body, ctx: req.Context(), finish: func(outcome string) {
		emit("finished", map[string]any{"outcome": outcome, "status_code": resp.StatusCode, "provider_request_ids": ids})
	}, status: resp.StatusCode}
	return resp, nil
}

type tracedProviderBody struct {
	io.ReadCloser
	ctx    context.Context
	finish func(string)
	once   sync.Once
	status int
}

func (b *tracedProviderBody) Close() error {
	err := b.ReadCloser.Close()
	b.once.Do(func() {
		outcome := "closed"
		if b.status >= 400 || err != nil {
			outcome = "failed"
		}
		if b.ctx.Err() == context.Canceled {
			outcome = "cancelled"
		}
		if b.ctx.Err() == context.DeadlineExceeded {
			outcome = "timeout"
		}
		b.finish(outcome)
	})
	return err
}
