package core

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

// A provider whose credentials or account are rejected fails every request
// the same way. The pool remembers that, so requests stop paying for the
// failure: the provider is skipped, as primary or as fallback, until a
// cooldown has passed. Then one real request is let through to find out
// whether it works again. Provider keys reach Core only at start, so a new
// key also means a new process and a fresh record.
const providerUnhealthyCooldown = 10 * time.Minute

// ProviderHealthStatus describes a provider the pool is currently skipping.
type ProviderHealthStatus struct {
	Provider string    `json:"provider"`
	State    string    `json:"state"`
	Reason   string    `json:"reason"`
	Error    string    `json:"error,omitempty"`
	Since    time.Time `json:"since"`
	RetryAt  time.Time `json:"retry_at"`
	Failures int       `json:"failures"`
}

type providerHealth struct {
	mu       sync.Mutex
	entries  map[string]*ProviderHealthStatus
	cooldown time.Duration
	now      func() time.Time
}

func newProviderHealth() *providerHealth {
	return &providerHealth{entries: map[string]*ProviderHealthStatus{}, cooldown: providerUnhealthyCooldown, now: time.Now}
}

// Pools are often built as literals; the record is created on first use.
var providerHealthInitMu sync.Mutex

func (pp *ProviderPool) healthStore() *providerHealth {
	if pp == nil {
		return nil
	}
	providerHealthInitMu.Lock()
	defer providerHealthInitMu.Unlock()
	if pp.health == nil {
		pp.health = newProviderHealth()
	}
	return pp.health
}

// ProviderHealth lists the providers the pool is skipping, by name.
func (pp *ProviderPool) ProviderHealth() []ProviderHealthStatus {
	h := pp.healthStore()
	if h == nil {
		return nil
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	out := make([]ProviderHealthStatus, 0, len(h.entries))
	for _, entry := range h.entries {
		out = append(out, *entry)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Provider < out[j].Provider })
	return out
}

// admit reports whether a request may use the provider now. An unhealthy
// provider is refused until its cooldown has passed. Then one request is let
// through and the next try is pushed back, so concurrent requests don't all
// test a dead key at once.
func (h *providerHealth) admit(name string) (ProviderHealthStatus, bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	entry := h.entries[name]
	if entry == nil {
		return ProviderHealthStatus{}, true
	}
	now := h.now()
	if now.Before(entry.RetryAt) {
		return *entry, false
	}
	entry.RetryAt = now.Add(h.cooldown)
	return *entry, true
}

// markUnhealthy records a credential or account failure. It reports whether
// the provider was healthy before.
func (h *providerHealth) markUnhealthy(name, reason string, err error) (ProviderHealthStatus, bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	now := h.now()
	detail := ""
	if err != nil {
		detail = err.Error()
		if len(detail) > 300 {
			detail = detail[:300] + "..."
		}
	}
	if entry := h.entries[name]; entry != nil {
		entry.Failures++
		entry.Reason, entry.Error = reason, detail
		entry.RetryAt = now.Add(h.cooldown)
		return *entry, false
	}
	entry := &ProviderHealthStatus{
		Provider: name, State: "unhealthy", Reason: reason, Error: detail,
		Since: now, RetryAt: now.Add(h.cooldown), Failures: 1,
	}
	h.entries[name] = entry
	return *entry, true
}

// markHealthy clears the record after a successful call. It reports whether
// the provider had been unhealthy, and what was recorded.
func (h *providerHealth) markHealthy(name string) (ProviderHealthStatus, bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	entry := h.entries[name]
	if entry == nil {
		return ProviderHealthStatus{}, false
	}
	delete(h.entries, name)
	return *entry, true
}

// inheritHealth carries the record over from the pool this one replaces. A
// settings save that resends the same providers keeps one shared record,
// because child threads keep the old pool in that case. Otherwise only
// providers whose configuration is unchanged keep their state; a changed
// provider gets a fresh start.
func (pp *ProviderPool) inheritHealth(old *ProviderPool, oldConfigs, newConfigs []ProviderConfig, realtimeChanged bool) {
	if pp == nil || old == nil || pp == old {
		return
	}
	previous := old.healthStore()
	if providerConfigsEqual(oldConfigs, newConfigs) && !realtimeChanged {
		providerHealthInitMu.Lock()
		pp.health = previous
		providerHealthInitMu.Unlock()
		return
	}
	before := providerConfigFingerprints(oldConfigs)
	after := providerConfigFingerprints(newConfigs)
	next := pp.healthStore()
	previous.mu.Lock()
	defer previous.mu.Unlock()
	next.mu.Lock()
	defer next.mu.Unlock()
	for name, entry := range previous.entries {
		if fingerprint, ok := after[name]; ok && fingerprint == before[name] {
			copied := *entry
			next.entries[name] = &copied
		}
	}
}

// providerConfigFingerprints keys each provider's settings by name. Which
// provider is the default is not part of a provider's own health.
func providerConfigFingerprints(configs []ProviderConfig) map[string]string {
	out := make(map[string]string, len(configs))
	for _, config := range configs {
		config.Default = false
		raw, err := json.Marshal(config)
		if err != nil {
			continue
		}
		out[config.Name] = string(raw)
	}
	return out
}

// providerCredentialFailure reports errors that mean the provider cannot
// serve any request with its current credentials or account: a rejected key,
// a refused permission or exhausted billing. Errors about one request (400,
// 404, 422, context length) and transient ones (429, 5xx, timeouts) say
// nothing about the provider's health.
func providerCredentialFailure(err error) (string, bool) {
	if err == nil {
		return "", false
	}
	var skipped *providerUnhealthyError
	if errors.As(err, &skipped) {
		return "", false
	}
	msg := strings.ToLower(err.Error())
	for _, rule := range []struct{ needle, reason string }{
		{"api error 401:", "credentials rejected (HTTP 401)"},
		{"api error 402:", "payment required (HTTP 402)"},
		{"api error 403:", "access denied (HTTP 403)"},
		{"invalid_api_key", "invalid API key"},
		{"invalid api key", "invalid API key"},
	} {
		if strings.Contains(msg, rule.needle) {
			return rule.reason, true
		}
	}
	return "", false
}

// providerUnhealthyError stands in for a provider that was not called
// because its credentials or account were rejected recently.
type providerUnhealthyError struct{ Status ProviderHealthStatus }

func (e *providerUnhealthyError) Error() string {
	return fmt.Sprintf("not called: %s since %s, next try after %s",
		e.Status.Reason, e.Status.Since.UTC().Format(time.RFC3339), e.Status.RetryAt.UTC().Format(time.RFC3339))
}

// fallbackSkippedError records a provider the request was too large for.
type fallbackSkippedError struct{ Budget requestBudget }

func (e *fallbackSkippedError) Error() string {
	return fmt.Sprintf("not called: request too large (%d input tokens, budget %d for %s)",
		e.Budget.InputTokens, e.Budget.InputBudget, e.Budget.Model)
}

// recordProviderOutcome updates the pool's provider health after a call and
// reports changes. Cancelled calls say nothing about the provider.
func (t *Thinker) recordProviderOutcome(ctx context.Context, provider LLMProvider, resp ChatResponse, err error) {
	if t.pool == nil || provider == nil || managesInferenceLifecycle(provider) {
		return
	}
	health := t.pool.healthStore()
	name := provider.Name()
	if err == nil {
		previous, recovered := health.markHealthy(name)
		if recovered {
			logMsg("PROVIDER", fmt.Sprintf("[%s] %s works again after %s", t.threadID, name, time.Since(previous.Since).Round(time.Second)))
			if t.telemetry != nil {
				t.telemetry.Emit("llm.provider_recovered", t.threadID, map[string]any{
					"provider": name, "reason": previous.Reason, "failures": previous.Failures,
					"unhealthy_seconds": int(health.now().Sub(previous.Since).Seconds()),
				})
			}
		}
		return
	}
	if ctx.Err() != nil {
		return
	}
	reason, ok := providerCredentialFailure(err)
	if !ok {
		return
	}
	status, newly := health.markUnhealthy(name, reason, err)
	if !newly {
		return
	}
	logMsg("PROVIDER", fmt.Sprintf("[%s] %s marked unhealthy: %s; skipped until %s", t.threadID, name, reason, status.RetryAt.UTC().Format(time.RFC3339)))
	if t.telemetry != nil {
		model := resp.Model
		if model == "" {
			model = modelIDForProvider(provider, t.model)
		}
		t.telemetry.Emit("llm.provider_unhealthy", t.threadID, map[string]any{
			"provider": name, "model": model, "reason": reason, "error": status.Error,
			"retry_at": status.RetryAt, "cooldown_seconds": int(health.cooldown.Seconds()),
		})
	}
}

// unhealthyPrimary returns the reason not to call the primary at all: its
// credentials were rejected recently and other providers are configured.
// A sole provider is always called, since nothing could answer in its place.
func (t *Thinker) unhealthyPrimary(primary LLMProvider) error {
	if primary == nil || t.pool == nil || t.pool.Count() < 2 || managesInferenceLifecycle(primary) {
		return nil
	}
	if t.pool.Get(primary.Name()) == nil {
		return nil
	}
	if status, ok := t.pool.healthStore().admit(primary.Name()); !ok {
		return &providerUnhealthyError{Status: status}
	}
	return nil
}
