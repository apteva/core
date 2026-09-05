package core

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

const configFile = "config.json"

// PersistentPaceState is runtime scheduling state, not directive content.
// Sleep records the duration used for the last timing decision; NextWakeAt is
// the single agent-owned pending timer. A zero NextWakeAt means event-only
// waiting. A future deadline remains capped by the runtime's 24-hour maximum
// when restored.
type PersistentPaceState struct {
	Sleep      string    `json:"sleep,omitempty"`
	NextWakeAt time.Time `json:"next_wake_at,omitempty"`
}

// PersistentThreadEvent is an idempotent API-created inbox event. Pending
// entries retain their payload so they can be replayed after a crash;
// consumed entries retain only ID+hash as the bounded deduplication ledger.
type PersistentThreadEvent struct {
	ExecutionIDs   []string      `json:"execution_ids,omitempty"`
	ID             string        `json:"id"`
	Hash           string        `json:"hash"`
	Text           string        `json:"text,omitempty"`
	Parts          []ContentPart `json:"parts,omitempty"`
	Consumed       bool          `json:"consumed,omitempty"`
	TrackLifecycle bool          `json:"track_lifecycle,omitempty"`
	ExecutionID    string        `json:"execution_id,omitempty"`
}

type PersistentThread struct {
	ID            string                       `json:"id"`
	Name          string                       `json:"name,omitempty"`      // human-readable label; empty = display as ID
	ParentID      string                       `json:"parent_id,omitempty"` // empty = child of main
	Depth         int                          `json:"depth,omitempty"`     // 0 = main's direct child
	System        bool                         `json:"system,omitempty"`    // system thread (can't be killed by LLM)
	Directive     string                       `json:"directive"`
	Tools         []string                     `json:"tools"`
	MCPNames      []string                     `json:"mcp_names,omitempty"`      // MCP servers to connect on respawn
	Model         string                       `json:"model,omitempty"`          // starting model tier: large, medium, small
	Reasoning     string                       `json:"reasoning,omitempty"`      // starting reasoning effort: auto, low, medium, high, ...
	Provider      string                       `json:"provider,omitempty"`       // provider selected for this thread
	Realtime      bool                         `json:"realtime,omitempty"`       // spawn as a realtime (voice/audio) thread
	AllowNoSpawn  bool                         `json:"allow_no_spawn,omitempty"` // authenticated system/API grant for explicitly attached no_spawn MCPs
	Voice         string                       `json:"voice,omitempty"`          // realtime voice id (e.g. "marin"); empty = provider default
	TurnDetection *RealtimeTurnDetectionConfig `json:"turn_detection,omitempty"` // realtime VAD/turn-taking profile and overrides
	Pace          *PersistentPaceState         `json:"pace,omitempty"`           // runtime cadence/deadline; never part of the directive
	Events        []PersistentThreadEvent      `json:"events,omitempty"`         // pending inbox payloads + bounded consumed-ID ledger
}

// RunMode controls the agent's safety behavior via system prompt guidance.
type RunMode string

const (
	ModeAutonomous RunMode = "autonomous" // agent operates freely, asks when it thinks it should
	ModeCautious   RunMode = "cautious"   // agent asks before destructive/external actions
	ModeLearn      RunMode = "learn"      // agent actively asks about new tool types, builds safety profile
)

// ProviderConfig persists a provider and its model selections.
type ModelReasoningCapability struct {
	Effort      string `json:"effort"`
	Description string `json:"description,omitempty"`
}

type ModelCapabilities struct {
	ContextWindow                 int                        `json:"context_window,omitempty"`
	MaxContextWindow              int                        `json:"max_context_window,omitempty"`
	EffectiveContextWindowPercent int                        `json:"effective_context_window_percent,omitempty"`
	DefaultReasoningLevel         string                     `json:"default_reasoning_level,omitempty"`
	SupportedReasoningLevels      []ModelReasoningCapability `json:"supported_reasoning_levels,omitempty"`
	InputModalities               []string                   `json:"input_modalities,omitempty"`
	SupportsParallelToolCalls     *bool                      `json:"supports_parallel_tool_calls,omitempty"`
	SupportsReasoningSummaries    *bool                      `json:"supports_reasoning_summaries,omitempty"`
	SupportsImageDetailOriginal   *bool                      `json:"supports_image_detail_original,omitempty"`
	SupportsSearchTool            *bool                      `json:"supports_search_tool,omitempty"`
}

type ProviderConfig struct {
	Name              string                       `json:"name"`                         // "google", "openai", "anthropic", "fireworks", "xai", "ollama", or a "*-realtime" companion
	Default           bool                         `json:"default,omitempty"`            // true = default provider (first match wins)
	Models            map[string]string            `json:"models,omitempty"`             // "large" → model ID, "medium" → ..., "small" → ...
	ModelCapabilities map[string]ModelCapabilities `json:"model_capabilities,omitempty"` // selected model metadata keyed by model ID
	BuiltinTools      []string                     `json:"builtin_tools,omitempty"`      // e.g. ["code_execution"]
	RealtimeVoice     string                       `json:"realtime_voice,omitempty"`     // default voice for realtime providers (e.g. "marin")
}

type Config struct {
	RuntimeSequence uint64 `json:"runtime_sequence,omitempty"`
	mu              sync.RWMutex
	saveMu          sync.Mutex
	path            string
	loadErr         error
	Directive       string  `json:"directive"`
	Mode            RunMode `json:"mode,omitempty"`
	Unconscious     bool    `json:"unconscious,omitempty"`      // enable background memory consolidation thread
	RealtimeEnabled bool    `json:"realtime_enabled,omitempty"` // master switch for realtime (voice/audio) threads; off = main never sees the capability and spawn rejects realtime=true
	RealtimeVoice   string  `json:"realtime_voice,omitempty"`   // dashboard/default realtime voice (provider validates at session open)
	// RealtimeVoiceMCP is the operator-selected subset of attached MCP
	// servers exposed to dashboard voice sessions. It is policy/config only;
	// realtime execution still uses the normal thread registry and gates.
	RealtimeVoiceMCP     []string                   `json:"realtime_voice_mcp,omitempty"`
	Providers            []ProviderConfig           `json:"providers,omitempty"` // multi-provider pool
	Provider             *ProviderConfig            `json:"provider,omitempty"`  // legacy single-provider (auto-migrated to Providers on load)
	Threads              []PersistentThread         `json:"threads,omitempty"`
	MainPace             *PersistentPaceState       `json:"main_pace,omitempty"`
	MainEvents           []PersistentThreadEvent    `json:"main_events,omitempty"`
	EventExecutions      []PersistentEventExecution `json:"event_executions,omitempty"`
	EventLifecycleOutbox []EventLifecycleTransition `json:"event_lifecycle_outbox,omitempty"`
	MCPServers           []MCPServerConfig          `json:"mcp_servers,omitempty"`
	Execution            ExecutionControlConfig     `json:"execution_control,omitempty"`
}

func NewConfig() *Config {
	c := &Config{
		path:      configFile,
		Directive: "Idle. Waiting for configuration via directive.",
	}
	c.loadErr = c.load()
	return c
}

func (c *Config) load() error {
	data, err := os.ReadFile(c.path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := json.Unmarshal(data, c); err != nil {
		return fmt.Errorf("decode config: %w", err)
	}

	// Migrate legacy single Provider → Providers array
	if c.Provider != nil && c.Provider.Name != "" && len(c.Providers) == 0 {
		c.Provider.Default = true
		c.Providers = []ProviderConfig{*c.Provider}
		c.Provider = nil
	}
	for _, server := range c.MCPServers {
		if err := validateMCPToolLoading(server); err != nil {
			return err
		}
	}
	return c.loadRuntimeJournalLocked()
}

func (c *Config) LoadError() error {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.loadErr
}

func (c *Config) Save() error {
	c.saveMu.Lock()
	defer c.saveMu.Unlock()
	c.mu.RLock()
	data, err := json.MarshalIndent(c, "", "  ")
	path := c.path
	c.mu.RUnlock()
	if err != nil {
		return err
	}
	if path == "" {
		return nil
	}
	return atomicWriteFile(path, data, 0600)
}

// update applies a configuration mutation and persists it as one serialized
// transaction. If the write fails, the previous in-memory configuration is
// restored so callers never observe a setting that was not durable.
func (c *Config) update(fn func()) error {
	c.saveMu.Lock()
	defer c.saveMu.Unlock()

	c.mu.Lock()
	before, err := json.Marshal(c)
	if err != nil {
		c.mu.Unlock()
		return err
	}
	fn()
	after, err := json.MarshalIndent(c, "", "  ")
	path := c.path
	c.mu.Unlock()
	if err != nil {
		c.restore(before)
		return err
	}
	if path == "" {
		return nil
	}
	if err := atomicWriteFile(path, after, 0600); err != nil {
		c.restore(before)
		return err
	}
	return nil
}

func (c *Config) restore(data []byte) {
	c.mu.Lock()
	defer c.mu.Unlock()
	// Decode into a fresh value before copying persisted fields back. Decoding
	// directly into c does not clear fields omitted by `omitempty`; for example,
	// rolling back the first SaveThread would leave the newly-added Threads
	// entry in memory because the pre-mutation JSON contained no `threads` key.
	// Runtime-only state and mutexes intentionally remain on c.
	var restored Config
	if err := json.Unmarshal(data, &restored); err != nil {
		return
	}
	c.RuntimeSequence = restored.RuntimeSequence
	c.Directive = restored.Directive
	c.Mode = restored.Mode
	c.Unconscious = restored.Unconscious
	c.RealtimeEnabled = restored.RealtimeEnabled
	c.RealtimeVoice = restored.RealtimeVoice
	c.RealtimeVoiceMCP = restored.RealtimeVoiceMCP
	c.Providers = restored.Providers
	c.Provider = restored.Provider
	c.Threads = restored.Threads
	c.MainPace = restored.MainPace
	c.MainEvents = restored.MainEvents
	c.EventExecutions = restored.EventExecutions
	c.EventLifecycleOutbox = restored.EventLifecycleOutbox
	c.MCPServers = restored.MCPServers
	c.Execution = restored.Execution
}

func (c *Config) GetDirective() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.Directive
}

// RealtimeEnabledFlag returns whether realtime (voice/audio) threads
// are enabled on this instance. Read under the config lock so it
// reflects the current value if toggled at runtime via the HTTP
// config endpoint. Naming avoids collision with the bare field.
func (c *Config) RealtimeEnabledFlag() bool {
	if c == nil {
		return false
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.RealtimeEnabled
}

func (c *Config) SetRealtimeEnabled(enabled bool) error {
	return c.update(func() { c.RealtimeEnabled = enabled })
}

func (c *Config) GetRealtimeVoice() string {
	if c == nil {
		return ""
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.RealtimeVoice
}

func (c *Config) SetRealtimeVoice(voice string) error {
	voice = strings.TrimSpace(voice)
	if len(voice) > 64 {
		return fmt.Errorf("realtime voice is too long")
	}
	return c.update(func() { c.RealtimeVoice = voice })
}

func (c *Config) GetRealtimeVoiceMCP() []string {
	if c == nil {
		return nil
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return append([]string(nil), c.RealtimeVoiceMCP...)
}

func (c *Config) SetRealtimeVoiceMCP(names []string) error {
	clean := make([]string, 0, len(names))
	seen := map[string]bool{}
	for _, name := range names {
		name = strings.TrimSpace(name)
		if name == "" || seen[name] {
			continue
		}
		seen[name] = true
		clean = append(clean, name)
	}
	sort.Strings(clean)
	return c.update(func() { c.RealtimeVoiceMCP = clean })
}

func (c *Config) SetDirective(d string) error {
	return c.update(func() { c.Directive = d })
}

func clonePersistentPaceState(state *PersistentPaceState) *PersistentPaceState {
	if state == nil {
		return nil
	}
	copy := *state
	return &copy
}

func cloneRealtimeTurnDetectionConfig(config *RealtimeTurnDetectionConfig) *RealtimeTurnDetectionConfig {
	if config == nil {
		return nil
	}
	copy := *config
	return &copy
}

func (c *Config) GetMainPace() *PersistentPaceState {
	if c == nil {
		return nil
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return clonePersistentPaceState(c.MainPace)
}

func (c *Config) SetMainPace(state PersistentPaceState) error {
	return c.update(func() { c.MainPace = clonePersistentPaceState(&state) })
}

func (c *Config) ClearThreads() error {
	return c.update(func() { c.Threads = nil })
}

func (c *Config) SaveThread(pt PersistentThread) error {
	return c.update(func() {
		found := false
		for i, t := range c.Threads {
			if t.ID == pt.ID {
				c.Threads[i] = pt
				found = true
				break
			}
		}
		if !found {
			c.Threads = append(c.Threads, pt)
		}
	})
}

func (c *Config) RemoveThread(id string) error {
	return c.update(func() {
		for i, t := range c.Threads {
			if t.ID == id {
				c.Threads = append(c.Threads[:i], c.Threads[i+1:]...)
				break
			}
		}
	})
}

func (c *Config) RenameThread(oldID string, pt PersistentThread) error {
	return c.update(func() {
		for i, t := range c.Threads {
			if t.ID == oldID {
				c.Threads[i] = pt
				return
			}
		}
		c.Threads = append(c.Threads, pt)
	})
}

func (c *Config) SaveMCPServer(cfg MCPServerConfig) error {
	return c.update(func() {
		found := false
		for i, s := range c.MCPServers {
			if s.Name == cfg.Name {
				c.MCPServers[i] = cfg
				found = true
				break
			}
		}
		if !found {
			c.MCPServers = append(c.MCPServers, cfg)
		}
	})
}

func (c *Config) RemoveMCPServer(name string) error {
	return c.update(func() {
		for i, s := range c.MCPServers {
			if s.Name == name {
				c.MCPServers = append(c.MCPServers[:i], c.MCPServers[i+1:]...)
				break
			}
		}
	})
}

func (c *Config) GetThreads() []PersistentThread {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]PersistentThread, len(c.Threads))
	copy(out, c.Threads)
	for i := range out {
		out[i].Pace = clonePersistentPaceState(out[i].Pace)
		out[i].TurnDetection = cloneRealtimeTurnDetectionConfig(out[i].TurnDetection)
		out[i].Events = clonePersistentThreadEvents(out[i].Events)
	}
	return out
}

func (c *Config) GetMCPServers() []MCPServerConfig {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]MCPServerConfig, len(c.MCPServers))
	copy(out, c.MCPServers)
	return out
}

func (c *Config) GetMode() RunMode {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.Mode == "" {
		return ModeAutonomous
	}
	return c.Mode
}

func (c *Config) SetMode(m RunMode) error {
	return c.update(func() { c.Mode = m })
}

func (c *Config) GetExecutionControl() ExecutionControlConfig {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := c.Execution
	if len(c.Execution.Breakpoints) > 0 {
		out.Breakpoints = append([]string(nil), c.Execution.Breakpoints...)
	}
	return out
}

func (c *Config) SetExecutionControl(ec ExecutionControlConfig) error {
	if len(ec.Breakpoints) > 0 {
		ec.Breakpoints = append([]string(nil), ec.Breakpoints...)
	}
	return c.update(func() { c.Execution = ec })
}

// GetProviders returns a copy of the providers list.
func (c *Config) GetProviders() []ProviderConfig {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]ProviderConfig, len(c.Providers))
	for i, p := range c.Providers {
		cp := ProviderConfig{Name: p.Name, Default: p.Default, BuiltinTools: p.BuiltinTools, RealtimeVoice: p.RealtimeVoice}
		if p.Models != nil {
			cp.Models = make(map[string]string)
			for k, v := range p.Models {
				cp.Models[k] = v
			}
		}
		cp.ModelCapabilities = cloneModelCapabilitiesMap(p.ModelCapabilities)
		out[i] = cp
	}
	return out
}

// GetDefaultProvider returns the default provider config, or nil.
func (c *Config) GetDefaultProvider() *ProviderConfig {
	c.mu.RLock()
	defer c.mu.RUnlock()
	for _, p := range c.Providers {
		if p.Default {
			cp := p
			return &cp
		}
	}
	if len(c.Providers) > 0 {
		cp := c.Providers[0]
		return &cp
	}
	return nil
}

// GetProvider returns the persisted default provider config, or nil.
// Backward-compatible wrapper around GetDefaultProvider.
func (c *Config) GetProvider() *ProviderConfig {
	return c.GetDefaultProvider()
}

// GetProviderByName returns a provider config by name, or nil.
func (c *Config) GetProviderByName(name string) *ProviderConfig {
	c.mu.RLock()
	defer c.mu.RUnlock()
	for _, p := range c.Providers {
		if p.Name == name {
			cp := p
			return &cp
		}
	}
	return nil
}

// SetProvider adds or updates a provider in the list. If it's the only one, marks it default.
func (c *Config) SetProvider(pc *ProviderConfig) error {
	return c.update(func() {
		found := false
		for i, p := range c.Providers {
			if p.Name == pc.Name {
				c.Providers[i] = *pc
				found = true
				break
			}
		}
		if !found {
			c.Providers = append(c.Providers, *pc)
		}
		if len(c.Providers) == 1 {
			c.Providers[0].Default = true
		}
		c.Provider = nil
	})
}

func (c *Config) ReplaceProviders(providers []ProviderConfig) error {
	providers = cloneProviderConfigs(providers)
	return c.update(func() {
		c.Providers = providers
		c.Provider = nil
	})
}

// SetProviderName adds or updates a provider by name with default flag.
func (c *Config) SetProviderName(name string) error {
	return c.update(func() {
		found := false
		for _, p := range c.Providers {
			if p.Name == name {
				found = true
				break
			}
		}
		if !found {
			pc := ProviderConfig{Name: name}
			if len(c.Providers) == 0 {
				pc.Default = true
			}
			c.Providers = append(c.Providers, pc)
		}
		c.Provider = nil
	})
}

// SetProviderModel updates a single model tier for a provider (default if not specified).
func (c *Config) SetProviderModel(tier string, modelID string) error {
	return c.update(func() {
		if len(c.Providers) == 0 {
			c.Providers = []ProviderConfig{{Name: "unknown", Default: true}}
		}
		for i, p := range c.Providers {
			if p.Default || i == 0 {
				if c.Providers[i].Models == nil {
					c.Providers[i].Models = make(map[string]string)
				}
				c.Providers[i].Models[tier] = modelID
				break
			}
		}
		c.Provider = nil
	})
}

// SetDefaultProvider marks a provider as default (clears default on others).
func (c *Config) SetDefaultProvider(name string) error {
	return c.update(func() {
		for i := range c.Providers {
			c.Providers[i].Default = c.Providers[i].Name == name
		}
	})
}

// RemoveProvider removes a provider by name.
func (c *Config) RemoveProvider(name string) error {
	return c.update(func() {
		for i, p := range c.Providers {
			if p.Name == name {
				c.Providers = append(c.Providers[:i], c.Providers[i+1:]...)
				break
			}
		}
	})
}

func cloneProviderConfigs(in []ProviderConfig) []ProviderConfig {
	out := make([]ProviderConfig, len(in))
	for i, p := range in {
		out[i] = p
		out[i].BuiltinTools = append([]string(nil), p.BuiltinTools...)
		if p.Models != nil {
			out[i].Models = make(map[string]string, len(p.Models))
			for tier, model := range p.Models {
				out[i].Models[tier] = model
			}
		}
		out[i].ModelCapabilities = cloneModelCapabilitiesMap(p.ModelCapabilities)
	}
	return out
}

func cloneModelCapabilitiesMap(in map[string]ModelCapabilities) map[string]ModelCapabilities {
	if in == nil {
		return nil
	}
	out := make(map[string]ModelCapabilities, len(in))
	for model, caps := range in {
		caps.SupportedReasoningLevels = append([]ModelReasoningCapability(nil), caps.SupportedReasoningLevels...)
		caps.InputModalities = append([]string(nil), caps.InputModalities...)
		out[model] = caps
	}
	return out
}

func mergeProviderConfig(providers []ProviderConfig, update ProviderConfig) []ProviderConfig {
	providers = cloneProviderConfigs(providers)
	index := -1
	if update.Name != "" {
		for i := range providers {
			if providers[i].Name == update.Name {
				index = i
				break
			}
		}
	} else {
		for i := range providers {
			if providers[i].Default {
				index = i
				break
			}
		}
		if index < 0 && len(providers) > 0 {
			index = 0
		}
	}
	if index < 0 {
		if update.Name != "" {
			providers = append(providers, update)
			if len(providers) == 1 {
				providers[0].Default = true
			}
		}
		return providers
	}
	if update.Name != "" {
		providers[index].Name = update.Name
	}
	if update.Models != nil {
		if providers[index].Models == nil {
			providers[index].Models = make(map[string]string)
		}
		for tier, model := range update.Models {
			providers[index].Models[tier] = model
		}
	}
	if update.BuiltinTools != nil {
		providers[index].BuiltinTools = append([]string(nil), update.BuiltinTools...)
	}
	if update.RealtimeVoice != "" {
		providers[index].RealtimeVoice = update.RealtimeVoice
	}
	if update.Default {
		for i := range providers {
			providers[i].Default = i == index
		}
	}
	return providers
}
