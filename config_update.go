package core

import (
	"encoding/json"
	"io"
	"net/http"
	"reflect"
	"strings"
)

type configUpdate struct {
	AutomaticToolLoading *AutomaticToolLoadingConfig `json:"automatic_tool_loading,omitempty"`
	MemoryPolicy         *MemoryPolicy               `json:"memory_policy,omitempty"`
	Directive            string                      `json:"directive,omitempty"`
	Provider             *ProviderConfig             `json:"provider,omitempty"`
	Providers            []ProviderConfig            `json:"providers,omitempty"`
	Computer             json.RawMessage             `json:"computer,omitempty"`
	MCPServers           []MCPServerConfig           `json:"mcp_servers,omitempty"`
	Execution            *ExecutionControlConfig     `json:"execution_control,omitempty"`
	RealtimeEnabled      *bool                       `json:"realtime_enabled,omitempty"`
	RealtimeVoice        *string                     `json:"realtime_voice,omitempty"`
	RealtimeVoiceMCP     *[]string                   `json:"realtime_voice_mcp,omitempty"`
	Reset                *struct {
		History bool `json:"history,omitempty"`
		Memory  bool `json:"memory,omitempty"`
		Threads bool `json:"threads,omitempty"`
	} `json:"reset,omitempty"`
}

type configUpdateKey struct{}

// Read one coherent configuration snapshot. An omitted field and an explicitly
// unchanged field both leave the current inference valid.
func configUpdateChanges(cfg *Config, body configUpdate) bool {
	if body.Reset != nil && (body.Reset.History || body.Reset.Memory || body.Reset.Threads) {
		return true
	}
	cfg.mu.RLock()
	defer cfg.mu.RUnlock()
	if body.Directive != "" && body.Directive != cfg.Directive {
		return true
	}
	if body.AutomaticToolLoading != nil {
		current := (&Config{AutomaticToolLoading: cfg.AutomaticToolLoading}).GetAutomaticToolLoading()
		next := (&Config{AutomaticToolLoading: body.AutomaticToolLoading}).GetAutomaticToolLoading()
		if !reflect.DeepEqual(current, next) {
			return true
		}
	}
	if body.MemoryPolicy != nil && !reflect.DeepEqual(*body.MemoryPolicy, cfg.MemoryPolicy) {
		return true
	}
	if body.Execution != nil && !reflect.DeepEqual(*body.Execution, cfg.Execution) {
		return true
	}
	if body.RealtimeEnabled != nil && *body.RealtimeEnabled != cfg.RealtimeEnabled {
		return true
	}
	if body.RealtimeVoice != nil && strings.TrimSpace(*body.RealtimeVoice) != cfg.RealtimeVoice {
		return true
	}
	if body.RealtimeVoiceMCP != nil {
		if strings.Join(compactStringList(*body.RealtimeVoiceMCP), "\x00") != strings.Join(compactStringList(cfg.RealtimeVoiceMCP), "\x00") {
			return true
		}
	}
	if body.MCPServers != nil && (len(body.MCPServers) != len(cfg.MCPServers) || len(body.MCPServers) > 0 && !reflect.DeepEqual(body.MCPServers, cfg.MCPServers)) {
		return true
	}
	providers := cfg.Providers
	if len(providers) == 0 && cfg.Provider != nil {
		providers = []ProviderConfig{*cfg.Provider}
	}
	if len(body.Providers) > 0 && !reflect.DeepEqual(body.Providers, providers) {
		return true
	}
	if body.Provider != nil && !reflect.DeepEqual(mergeProviderConfig(cloneProviderConfigs(providers), *body.Provider), providers) {
		return true
	}
	return false
}

func (a *APIServer) decodeConfigUpdate(w http.ResponseWriter, r *http.Request) (configUpdate, bool) {
	var body configUpdate

	decoder := json.NewDecoder(r.Body)
	if err := decoder.Decode(&body); err != nil {
		http.Error(w, "invalid JSON", http.StatusBadRequest)
		return body, false
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		http.Error(w, "invalid trailing JSON", http.StatusBadRequest)
		return body, false
	}

	if err := validateAutomaticToolLoading(body.AutomaticToolLoading); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return body, false
	}
	if err := validateMemoryPolicy(body.MemoryPolicy); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return body, false
	}
	if len(body.Computer) > 0 {
		http.Error(w, "core computer config has been removed; use the Computer app MCP tools instead", http.StatusGone)
		return body, false
	}
	for _, server := range body.MCPServers {
		if err := validateMCPToolLoading(server); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return body, false
		}
	}
	if body.RealtimeVoice != nil {
		candidate := &Config{}
		if err := candidate.SetRealtimeVoice(*body.RealtimeVoice); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return body, false
		}
	}
	if len(body.Providers) > 0 || body.Provider != nil || body.RealtimeEnabled != nil {
		providers := a.thinker.config.GetProviders()
		if len(body.Providers) > 0 {
			providers = cloneProviderConfigs(body.Providers)
		}
		if body.Provider != nil {
			providers = mergeProviderConfig(providers, *body.Provider)
		}
		enabled := a.thinker.config.RealtimeEnabledFlag()
		if body.RealtimeEnabled != nil {
			enabled = *body.RealtimeEnabled
		}
		if _, err := buildProviderPoolWithCapabilities(&Config{Providers: providers, RealtimeEnabled: enabled}, false); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return body, false
		}
	}
	return body, true
}
