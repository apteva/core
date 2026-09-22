package core

import (
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

const (
	grokBuildDefaultBaseURL = "https://cli-chat-proxy.grok.com/v1"
	grokBuildClientVersion  = "1.0.38"
	grokBuildStateProvider  = "grok-build-responses"
)

// responsesSessionProviderProfile contains the behavior that differs between
// subscription-backed Responses providers. The shared transport stays unaware
// of vendor-specific environment names and headers.
type responsesSessionProviderProfile struct {
	responseStateProvider       string
	subscriptionBacked          bool
	advertiseBuiltinTools       bool
	allowConfiguredBuiltinTools bool
	sendPromptCacheHints        bool
	refreshBeforeRequest        bool
	retryAfterAuthError         bool
	replaceIdentityOnRefresh    bool
	validateRuntimeProvider     bool
	defaultReasoningEffort      string
	defaultReasoningSummary     string
	allowXHighReasoning         bool
	normalizeToolParameters     func(map[string]any) map[string]any
	applyRequestHeaders         func(http.Header, responsesSessionSnapshot, string)
}

type responsesSessionSnapshot struct {
	AccessToken       string
	ResponsesURL      string
	AccountID         string
	AccountEmail      string
	UserID            string
	PrincipalType     string
	PrincipalID       string
	LastTokenRefresh  time.Time
	RefreshGeneration uint64
}

type responsesRuntimeToken struct {
	AccessToken   string
	AccountID     string
	AccountEmail  string
	UserID        string
	PrincipalType string
	PrincipalID   string
	BaseURL       string
}

// responsesSessionState is deliberately shared by WithReasoning/WithBuiltins
// clones. A refresh performed by any clone immediately updates every request
// path and its identity headers.
type responsesSessionState struct {
	mu        sync.RWMutex
	refreshMu sync.Mutex
	value     responsesSessionSnapshot
}

func newResponsesSessionState(value responsesSessionSnapshot) *responsesSessionState {
	return &responsesSessionState{value: value}
}

func (s *responsesSessionState) snapshot() responsesSessionSnapshot {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.value
}

func (s *responsesSessionState) updateFromRuntimeToken(token responsesRuntimeToken, replaceIdentity bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.value.AccessToken = strings.TrimSpace(token.AccessToken)
	if replaceIdentity {
		s.value.AccountID = strings.TrimSpace(token.AccountID)
		s.value.AccountEmail = strings.TrimSpace(token.AccountEmail)
		s.value.UserID = strings.TrimSpace(token.UserID)
		s.value.PrincipalType = strings.TrimSpace(token.PrincipalType)
		s.value.PrincipalID = strings.TrimSpace(token.PrincipalID)
	} else if accountID := strings.TrimSpace(token.AccountID); accountID != "" {
		s.value.AccountID = accountID
	}
	if baseURL := strings.TrimSpace(token.BaseURL); baseURL != "" {
		s.value.ResponsesURL = responsesEndpoint(baseURL)
	}
	s.value.LastTokenRefresh = time.Now()
	s.value.RefreshGeneration++
}

var openAICodexSessionProviderProfile = responsesSessionProviderProfile{
	responseStateProvider:       openAIResponsesStateProvider,
	subscriptionBacked:          true,
	advertiseBuiltinTools:       false,
	allowConfiguredBuiltinTools: true,
	sendPromptCacheHints:        true,
	refreshBeforeRequest:        true,
	retryAfterAuthError:         true,
	applyRequestHeaders: func(headers http.Header, state responsesSessionSnapshot, _ string) {
		if state.AccountID != "" {
			headers.Set("ChatGPT-Account-ID", state.AccountID)
		}
	},
}

var grokBuildSessionProviderProfile = responsesSessionProviderProfile{
	responseStateProvider:       grokBuildStateProvider,
	subscriptionBacked:          true,
	advertiseBuiltinTools:       false,
	allowConfiguredBuiltinTools: false,
	sendPromptCacheHints:        false,
	refreshBeforeRequest:        true,
	retryAfterAuthError:         true,
	replaceIdentityOnRefresh:    true,
	validateRuntimeProvider:     true,
	defaultReasoningEffort:      "high",
	defaultReasoningSummary:     "concise",
	allowXHighReasoning:         true,
	normalizeToolParameters:     normalizeGrokBuildToolParameters,
	applyRequestHeaders: func(headers http.Header, state responsesSessionSnapshot, model string) {
		headers.Set("X-XAI-Token-Auth", "xai-grok-cli")
		headers.Set("x-grok-client-version", grokBuildClientVersion)
		headers.Set("x-grok-client-identifier", "apteva-core")
		headers.Set("x-grok-client-mode", "headless")
		headers.Set("x-authenticateresponse", "authenticate-response")
		headers.Set("x-userid", state.UserID)
		if state.AccountEmail != "" {
			headers.Set("x-email", state.AccountEmail)
		}
		if strings.TrimSpace(model) != "" {
			headers.Set("x-grok-model-override", strings.TrimSpace(model))
		}
	},
}

// normalizeGrokBuildToolParameters keeps the shared tool definition intact
// while adapting the request copy to Grok Build's stricter client-tool schema
// validator. Grok requires a plain object at the parameter root and rejects
// root-level composition keywords such as search_tools' query-or-queries
// anyOf, even when the schema also declares type=object. Nested property
// schemas remain untouched; Core's handlers retain their runtime validation.
func normalizeGrokBuildToolParameters(schema map[string]any) map[string]any {
	cloned, _ := cloneJSONValue(schema).(map[string]any)
	if cloned == nil {
		cloned = map[string]any{}
	}
	cloned["type"] = "object"
	if _, ok := cloned["properties"]; !ok {
		cloned["properties"] = map[string]any{}
	}
	delete(cloned, "allOf")
	delete(cloned, "anyOf")
	delete(cloned, "oneOf")
	return cloned
}

func NewGrokBuildProvider(accessToken string) LLMProvider {
	baseURL := strings.TrimSpace(os.Getenv("GROK_BUILD_BASE_URL"))
	if baseURL == "" {
		baseURL = grokBuildDefaultBaseURL
	}
	url := responsesEndpoint(baseURL)
	state := responsesSessionSnapshot{
		AccessToken:   strings.TrimSpace(accessToken),
		ResponsesURL:  url,
		AccountEmail:  strings.TrimSpace(os.Getenv("GROK_BUILD_ACCOUNT_EMAIL")),
		UserID:        strings.TrimSpace(os.Getenv("GROK_BUILD_USER_ID")),
		PrincipalType: strings.TrimSpace(os.Getenv("GROK_BUILD_PRINCIPAL_TYPE")),
		PrincipalID:   strings.TrimSpace(os.Getenv("GROK_BUILD_PRINCIPAL_ID")),
	}
	return &OpenAINativeProvider{
		name:            "grok-build",
		apiKey:          state.AccessToken,
		responsesURL:    url,
		forceStoreFalse: true,
		sessionProfile:  &grokBuildSessionProviderProfile,
		sessionState:    newResponsesSessionState(state),
		runtimeTokenURL: sessionRuntimeTokenURL("GROK_BUILD_PROVIDER_ID"),
		serverAPIKey:    os.Getenv("APTEVA_API_KEY"),
		models: map[ModelTier]string{
			ModelLarge:  "grok-4.6",
			ModelMedium: "grok-4.6",
			ModelSmall:  "grok-4.6",
		},
	}
}

func sessionRuntimeTokenURL(providerIDEnv string) string {
	serverURL := strings.TrimRight(strings.TrimSpace(os.Getenv("SERVER_URL")), "/")
	providerID := strings.TrimSpace(os.Getenv(providerIDEnv))
	if serverURL == "" || providerID == "" {
		return ""
	}
	return serverURL + "/api/providers/" + providerID + "/auth/runtime-token"
}

func responsesEndpoint(baseURL string) string {
	baseURL = strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if strings.HasSuffix(baseURL, "/responses") {
		return baseURL
	}
	return baseURL + "/responses"
}

func (p *OpenAINativeProvider) currentSessionSnapshot() responsesSessionSnapshot {
	if p.sessionState != nil {
		return p.sessionState.snapshot()
	}
	p.tokenMu.RLock()
	defer p.tokenMu.RUnlock()
	return responsesSessionSnapshot{
		AccessToken: p.apiKey, ResponsesURL: p.responsesURL,
		AccountID: p.accountID, LastTokenRefresh: p.lastTokenRefresh,
	}
}

func (p *OpenAINativeProvider) responsesEndpoint() string {
	if url := strings.TrimSpace(p.currentSessionSnapshot().ResponsesURL); url != "" {
		return url
	}
	return "https://api.openai.com/v1/responses"
}

func (p *OpenAINativeProvider) refreshesBeforeRequest() bool {
	return p.sessionProfile != nil && p.sessionProfile.refreshBeforeRequest || p.Name() == "openai-codex"
}

func (p *OpenAINativeProvider) retriesAfterAuthError() bool {
	return p.sessionProfile != nil && p.sessionProfile.retryAfterAuthError || p.Name() == "openai-codex"
}

func (p *OpenAINativeProvider) responseStateProvider() string {
	if p.sessionProfile != nil && p.sessionProfile.responseStateProvider != "" {
		return p.sessionProfile.responseStateProvider
	}
	return openAIResponsesStateProvider
}

var _ LLMProvider = (*OpenAINativeProvider)(nil)
