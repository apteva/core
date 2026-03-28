package main

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

var (
	titleStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("170")).
			Padding(0, 1)

	statsStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("241")).
			Padding(0, 1)

	thoughtHeaderStyle = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("39"))

	thoughtStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("252"))

	statusBarStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("235")).
			Background(lipgloss.Color("39")).
			Padding(0, 1).
			Bold(true)

	pausedStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("235")).
			Background(lipgloss.Color("208")).
			Padding(0, 1).
			Bold(true)

	helpStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("241")).
			Padding(0, 1)

	panelBorderStyle = lipgloss.NewStyle().
				Border(lipgloss.RoundedBorder()).
				BorderForeground(lipgloss.Color("62"))

	chatTitleStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("212")).
			Padding(0, 1)

	chatUserStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("170"))

	chatAgentStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("252"))

	chatAgentLabelStyle = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("39"))

	inputLabelStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("212")).
			Bold(true)

	memoryTitleStyle = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("141")).
				Padding(0, 1)

	memoryTextStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("252"))

	memoryAgeStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("241")).
			Italic(true)

	memorySelectedStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("235")).
				Background(lipgloss.Color("141")).
				Bold(true)

	threadTitleStyle = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("77")).
				Padding(0, 1)

	threadActiveStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("77"))

	threadSelectedStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("235")).
				Background(lipgloss.Color("77")).
				Bold(true)

	toolsTitleStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("49")).
			Padding(0, 1)

	toolCoreStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("49"))

	toolDiscoverStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("252"))

	toolMCPStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("212"))

	tabActiveStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("235")).
			Background(lipgloss.Color("39")).
			Padding(0, 1)

	tabInactiveStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("241")).
				Padding(0, 1)

	consoleTitleStyle = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("208")).
				Padding(0, 1)

	consoleLineStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("208"))

	telemetryTitleStyle = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("214")).
				Padding(0, 1)

	telemetryLLMStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("214"))

	telemetryThreadStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("77"))

	telemetryToolStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("49"))

	telemetryErrorStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("196"))

	telemetryDimStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("241"))

	directiveTitleStyle = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("220")).
				Padding(0, 1)

	directiveLineStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("252"))

	directiveCursorStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("235")).
				Background(lipgloss.Color("220"))
)

type tickMsg time.Time
type busEventMsg Event

const (
	priceInputPerToken  = 0.60 / 1_000_000
	priceCachedPerToken = 0.10 / 1_000_000
	priceOutputPerToken = 3.00 / 1_000_000
)

type chatMessage struct {
	isUser   bool
	text     string
	threadID string // which thread this belongs to
}

type panelMode int

const (
	panelChat      panelMode = iota
	panelConsole
	panelMemory
	panelThreads
	panelDirective
	panelTools
	panelBus
	panelTelemetry
)

type inputMode int

const (
	inputChat    inputMode = iota
	inputConsole
	inputDirective
)

// Sidebar menu items
type sidebarSection int

const (
	sidebarView sidebarSection = iota
	sidebarSettings
)

type sidebarItem struct {
	label   string
	panel   panelMode
	section sidebarSection
	action  string // non-panel actions: "provider", "model", "mode"
}

var sidebarItems = []sidebarItem{
	// View section
	{label: "Chat", panel: panelChat, section: sidebarView},
	{label: "Threads", panel: panelThreads, section: sidebarView},
	{label: "Memory", panel: panelMemory, section: sidebarView},
	{label: "Tools", panel: panelTools, section: sidebarView},
	{label: "Bus", panel: panelBus, section: sidebarView},
	{label: "Telemetry", panel: panelTelemetry, section: sidebarView},
	{label: "Console", panel: panelConsole, section: sidebarView},
	// Settings section
	{label: "Provider", section: sidebarSettings, action: "provider"},
	{label: "Model", section: sidebarSettings, action: "model"},
	{label: "Mode", section: sidebarSettings, action: "mode"},
	{label: "Directive", panel: panelDirective, section: sidebarSettings},
}

// Picker mode for selecting from a list
type pickerMode int

const (
	pickerNone     pickerMode = iota
	pickerProvider
	pickerModel
	pickerMode_
)

// Command palette
type paletteState int

const (
	paletteHidden paletteState = iota
	paletteOpen
)

type threadView struct {
	thoughts     []thought
	currentChunk *strings.Builder
	iteration    int
	rate         ThinkRate
	model        ModelTier
	contextMsgs  int
	contextChars int
	// Per-thread cost tracking
	cost         float64
	iterations   int
	started      time.Time
	lastThought  time.Time
	sleepDur     time.Duration // current sleep duration for this thread
}

type model struct {
	thinker      *Thinker
	busSub       *Subscription
	width        int
	height       int
	scrollOffset int
	paused       bool
	startTime    time.Time
	lastDuration time.Duration
	input        textinput.Model
	inputActive  bool
	inputMode    inputMode

	chat           []chatMessage
	consoleHistory []string
	busLog         []string // recent bus events for display
	telemetryLog   []string // recent telemetry events for display
	telemetryCursor int
	memoryCount    int
	threadCount    int

	// Supervised mode approval
	pendingApproval *ToolCallData

	panel        panelMode
	memoryCursor int
	threadCursor int

	// Directive editing
	directiveLines []string
	directiveCursor int

	// Tab system
	activeTab    string
	tabs         []string
	threadViews  map[string]*threadView

	userID string

	totalPromptTokens     int
	totalCachedTokens     int
	totalCompletionTokens int
	totalCost             float64

	// Sidebar navigation
	sidebarCursor int

	// Picker (provider/model/mode selection)
	picker       pickerMode
	pickerCursor int
	pickerItems  []string

	// Command palette
	palette      paletteState
	paletteInput textinput.Model
	paletteItems []sidebarItem // filtered items
}

type thought struct {
	iteration int
	content   string
	duration  time.Duration
}

func newModel(thinker *Thinker) model {
	ti := textinput.New()
	ti.Placeholder = "message..."
	ti.CharLimit = 500

	pi := textinput.New()
	pi.Placeholder = "type to filter..."
	pi.CharLimit = 100

	return model{
		thinker:     thinker,
		busSub:      thinker.bus.SubscribeAll("tui", 500),
		startTime:   time.Now(),
		input:       ti,
		memoryCount: thinker.memory.Count(),
		userID:      "user",
		activeTab:   "main",
		tabs:        []string{"main"},
		threadViews: map[string]*threadView{
			"main": {currentChunk: &strings.Builder{}},
		},
		directiveLines: strings.Split(thinker.config.GetDirective(), "\n"),
		paletteInput:   pi,
		paletteItems:   append([]sidebarItem{}, sidebarItems...),
	}
}

func (m *model) getOrCreateView(id string) *threadView {
	if v, ok := m.threadViews[id]; ok {
		return v
	}
	v := &threadView{currentChunk: &strings.Builder{}, started: time.Now()}
	m.threadViews[id] = v
	// Add tab if not present
	found := false
	for _, t := range m.tabs {
		if t == id {
			found = true
			break
		}
	}
	if !found {
		m.tabs = append(m.tabs, id)
	}
	return v
}

func (m *model) removeTab(id string) {
	delete(m.threadViews, id)
	for i, t := range m.tabs {
		if t == id {
			m.tabs = append(m.tabs[:i], m.tabs[i+1:]...)
			break
		}
	}
	if m.activeTab == id {
		m.activeTab = "main"
		m.scrollOffset = 0
	}
}

func (m model) Init() tea.Cmd {
	return tea.Batch(
		pollBusEvent(m.busSub),
		tickCmd(),
	)
}

func pollBusEvent(sub *Subscription) tea.Cmd {
	return func() tea.Msg {
		return busEventMsg(<-sub.C)
	}
}

func tickCmd() tea.Cmd {
	return tea.Tick(time.Second, func(t time.Time) tea.Msg {
		return tickMsg(t)
	})
}

// openPicker populates the picker with items and enters picker mode.
func (m *model) openPicker(mode pickerMode) {
	m.picker = mode
	m.pickerCursor = 0
	switch mode {
	case pickerProvider:
		var items []string
		for _, p := range availableProviders() {
			name := p.Name()
			if m.thinker.provider != nil && name == m.thinker.provider.Name() {
				name += " ●"
			}
			items = append(items, name)
		}
		m.pickerItems = items
	case pickerModel:
		if gp, ok := m.thinker.provider.(*GoogleProvider); ok {
			var items []string
			for _, mid := range gp.AvailableModels() {
				gm := geminiModels[mid]
				label := fmt.Sprintf("%-30s $%.2f / $%.2f", mid, gm.InputPer1M, gm.OutputPer1M)
				if mid == gp.ActiveModel() {
					label += " ●"
				}
				items = append(items, label)
			}
			m.pickerItems = items
		} else {
			models := m.thinker.provider.Models()
			var items []string
			for tier, id := range models {
				items = append(items, fmt.Sprintf("%s: %s", tier, id))
			}
			m.pickerItems = items
		}
	case pickerMode_:
		current := string(m.thinker.config.GetMode())
		m.pickerItems = []string{"autonomous", "supervised"}
		for i, item := range m.pickerItems {
			if item == current {
				m.pickerItems[i] = item + " ●"
				m.pickerCursor = i
			}
		}
	}
}

// openPalette opens the command palette.
func (m *model) openPalette() tea.Cmd {
	m.palette = paletteOpen
	m.paletteInput.Reset()
	m.paletteInput.Focus()
	m.paletteItems = append([]sidebarItem{}, sidebarItems...)
	return textinput.Blink
}

// filterPalette filters palette items by query.
func (m *model) filterPalette() {
	query := strings.ToLower(m.paletteInput.Value())
	if query == "" {
		m.paletteItems = append([]sidebarItem{}, sidebarItems...)
		return
	}
	var filtered []sidebarItem
	for _, item := range sidebarItems {
		if strings.Contains(strings.ToLower(item.label), query) {
			filtered = append(filtered, item)
		}
	}
	m.paletteItems = filtered
	m.pickerCursor = 0
}

// executeSidebarItem handles selecting a sidebar or palette item.
func (m *model) executeSidebarItem(item sidebarItem) tea.Cmd {
	switch item.action {
	case "provider":
		m.openPicker(pickerProvider)
		return nil
	case "model":
		m.openPicker(pickerModel)
		return nil
	case "mode":
		m.openPicker(pickerMode_)
		return nil
	default:
		if item.panel == panelDirective {
			m.panel = panelDirective
			m.directiveLines = strings.Split(m.thinker.config.GetDirective(), "\n")
			m.directiveCursor = 0
		} else {
			m.panel = item.panel
			if item.panel == panelMemory {
				m.memoryCursor = 0
			}
			if item.panel == panelThreads {
				m.threadCursor = 0
			}
		}
		return nil
	}
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		// ── Command palette mode ──
		if m.palette == paletteOpen {
			switch msg.String() {
			case "esc":
				m.palette = paletteHidden
				m.paletteInput.Blur()
				return m, nil
			case "enter":
				if len(m.paletteItems) > 0 {
					idx := m.pickerCursor
					if idx >= len(m.paletteItems) {
						idx = 0
					}
					cmd := m.executeSidebarItem(m.paletteItems[idx])
					m.palette = paletteHidden
					m.paletteInput.Blur()
					return m, cmd
				}
				return m, nil
			case "up", "ctrl+p":
				if m.pickerCursor > 0 {
					m.pickerCursor--
				}
				return m, nil
			case "down", "ctrl+n":
				if m.pickerCursor < len(m.paletteItems)-1 {
					m.pickerCursor++
				}
				return m, nil
			default:
				var cmd tea.Cmd
				m.paletteInput, cmd = m.paletteInput.Update(msg)
				m.filterPalette()
				return m, cmd
			}
		}

		// ── Picker mode (provider/model/mode selection) ──
		if m.picker != pickerNone {
			switch msg.String() {
			case "esc", "q":
				m.picker = pickerNone
				return m, nil
			case "j", "down":
				if m.pickerCursor < len(m.pickerItems)-1 {
					m.pickerCursor++
				}
				return m, nil
			case "k", "up":
				if m.pickerCursor > 0 {
					m.pickerCursor--
				}
				return m, nil
			case "enter":
				switch m.picker {
				case pickerProvider:
					providers := availableProviders()
					if m.pickerCursor < len(providers) {
						selected := providers[m.pickerCursor]
						m.thinker.provider = selected
						m.thinker.config.SetProviderName(selected.Name())
						m.consoleHistory = append(m.consoleHistory, fmt.Sprintf("provider → %s (saved)", selected.Name()))
					}
				case pickerModel:
					if gp, ok := m.thinker.provider.(*GoogleProvider); ok {
						models := gp.AvailableModels()
						if m.pickerCursor < len(models) {
							gp.SetModel(models[m.pickerCursor])
							m.thinker.config.SetProviderModel("large", models[m.pickerCursor])
							m.thinker.config.SetProviderModel("small", models[m.pickerCursor])
							m.consoleHistory = append(m.consoleHistory, fmt.Sprintf("model → %s (saved)", models[m.pickerCursor]))
						}
					}
				case pickerMode_:
					modes := []RunMode{ModeAutonomous, ModeSupervised}
					if m.pickerCursor < len(modes) {
						m.thinker.config.SetMode(modes[m.pickerCursor])
						if m.thinker.telemetry != nil {
							m.thinker.telemetry.Emit("mode.changed", "main", map[string]string{"mode": string(modes[m.pickerCursor])})
						}
						m.consoleHistory = append(m.consoleHistory, fmt.Sprintf("mode → %s", modes[m.pickerCursor]))
					}
				}
				m.picker = pickerNone
				return m, nil
			}
			return m, nil
		}

		// ── Directive editing mode ──
		if m.panel == panelDirective {
			switch msg.String() {
			case "esc":
				directive := strings.Join(m.directiveLines, "\n")
				m.thinker.config.SetDirective(directive)
				m.thinker.ReloadDirective()
				m.panel = panelChat
				return m, nil
			case "enter":
				m.directiveLines = append(m.directiveLines[:m.directiveCursor+1],
					append([]string{""}, m.directiveLines[m.directiveCursor+1:]...)...)
				m.directiveCursor++
				return m, nil
			case "backspace":
				if m.directiveCursor > 0 && len(m.directiveLines[m.directiveCursor]) == 0 {
					m.directiveLines = append(m.directiveLines[:m.directiveCursor], m.directiveLines[m.directiveCursor+1:]...)
					m.directiveCursor--
				} else if len(m.directiveLines[m.directiveCursor]) > 0 {
					line := m.directiveLines[m.directiveCursor]
					m.directiveLines[m.directiveCursor] = line[:len(line)-1]
				}
				return m, nil
			case "up":
				if m.directiveCursor > 0 {
					m.directiveCursor--
				}
				return m, nil
			case "down":
				if m.directiveCursor < len(m.directiveLines)-1 {
					m.directiveCursor++
				}
				return m, nil
			default:
				if len(msg.String()) == 1 || msg.String() == "space" {
					ch := msg.String()
					if ch == "space" {
						ch = " "
					}
					m.directiveLines[m.directiveCursor] += ch
				}
				return m, nil
			}
		}

		// ── Text input mode (chat/console) ──
		if m.inputActive {
			switch msg.String() {
			case "enter":
				val := strings.TrimSpace(m.input.Value())
				if val != "" {
					switch m.inputMode {
					case inputChat:
						// TODO: re-enable media auto-detection later
						// if parts := detectImageParts(val); len(parts) > 0 {
						// 	m.thinker.InjectWithParts(val, parts)
						// 	m.chat = append(m.chat, chatMessage{isUser: true, text: val + " " + mediaLabel(parts), threadID: m.userID})
						// } else {
						m.thinker.InjectUserMessage(m.userID, val)
						m.chat = append(m.chat, chatMessage{isUser: true, text: val, threadID: m.userID})
						// }
					case inputConsole:
						// TODO: re-enable media auto-detection later
						m.thinker.InjectConsole(val)
						m.consoleHistory = append(m.consoleHistory, val)
					}
				}
				m.input.Reset()
				m.input.Blur()
				m.inputActive = false
				return m, nil
			case "esc":
				m.input.Reset()
				m.input.Blur()
				m.inputActive = false
				return m, nil
			default:
				var cmd tea.Cmd
				m.input, cmd = m.input.Update(msg)
				return m, cmd
			}
		}

		// ── Supervised mode: handle approval keys ──
		if m.pendingApproval != nil {
			switch msg.String() {
			case "y":
				select {
				case m.thinker.approvalCh <- true:
				default:
				}
				m.pendingApproval = nil
				return m, nil
			case "n":
				select {
				case m.thinker.approvalCh <- false:
				default:
				}
				m.pendingApproval = nil
				return m, nil
			}
		}

		// ── Normal mode: sidebar + global keys ──
		switch msg.String() {
		case "q", "ctrl+c":
			m.thinker.Stop()
			return m, tea.Quit
		case "/":
			cmd := m.openPalette()
			return m, cmd
		case "i":
			m.panel = panelChat
			m.inputMode = inputChat
			m.input.Placeholder = "message..."
			m.inputActive = true
			m.input.Focus()
			return m, textinput.Blink
		case "c":
			m.panel = panelConsole
			m.inputMode = inputConsole
			m.input.Placeholder = "command..."
			m.inputActive = true
			m.input.Focus()
			return m, textinput.Blink
		case " ":
			m.paused = !m.paused
			m.thinker.TogglePause()
			return m, nil
		case "enter":
			// Select sidebar item
			if m.sidebarCursor < len(sidebarItems) {
				cmd := m.executeSidebarItem(sidebarItems[m.sidebarCursor])
				return m, cmd
			}
			return m, nil
		case "d":
			// Delete in memory/thread panels
			if m.panel == panelMemory {
				if m.memoryCount > 0 && m.memoryCursor < m.memoryCount {
					realIndex := m.thinker.memory.Count() - 1 - m.memoryCursor
					m.thinker.memory.Delete(realIndex)
					m.memoryCount = m.thinker.memory.Count()
					if m.memoryCursor >= m.memoryCount && m.memoryCursor > 0 {
						m.memoryCursor--
					}
				}
				return m, nil
			}
			if m.panel == panelThreads {
				threads := m.thinker.threads.List()
				if m.threadCursor < len(threads) {
					id := threads[m.threadCursor].ID
					m.thinker.threads.Kill(id)
					m.thinker.config.RemoveThread(id)
					m.removeTab(id)
					m.threadCount = m.thinker.threads.Count()
					if m.threadCursor >= m.threadCount && m.threadCursor > 0 {
						m.threadCursor--
					}
				}
				return m, nil
			}
		case "]", "tab":
			for i, tab := range m.tabs {
				if tab == m.activeTab {
					m.activeTab = m.tabs[(i+1)%len(m.tabs)]
					m.scrollOffset = 0
					break
				}
			}
			return m, nil
		case "[", "shift+tab":
			for i, tab := range m.tabs {
				if tab == m.activeTab {
					idx := (i - 1 + len(m.tabs)) % len(m.tabs)
					m.activeTab = m.tabs[idx]
					m.scrollOffset = 0
					break
				}
			}
			return m, nil
		case "j", "down":
			// Sidebar navigation takes priority when in chat panel (default)
			switch m.panel {
			case panelChat:
				if m.sidebarCursor < len(sidebarItems)-1 {
					m.sidebarCursor++
				}
			case panelMemory:
				if m.memoryCursor < m.memoryCount-1 {
					m.memoryCursor++
				}
			case panelThreads:
				threads := m.thinker.threads.List()
				if m.threadCursor < len(threads)-1 {
					m.threadCursor++
				}
			default:
				m.scrollOffset = min(m.scrollOffset+1, m.maxScroll())
			}
			return m, nil
		case "k", "up":
			switch m.panel {
			case panelChat:
				if m.sidebarCursor > 0 {
					m.sidebarCursor--
				}
			case panelMemory:
				if m.memoryCursor > 0 {
					m.memoryCursor--
				}
			case panelThreads:
				if m.threadCursor > 0 {
					m.threadCursor--
				}
			default:
				m.scrollOffset = max(m.scrollOffset-1, 0)
			}
			return m, nil
		case "l", "right":
			// Scroll thoughts down
			m.scrollOffset = min(m.scrollOffset+3, m.maxScroll())
			return m, nil
		case "h", "left":
			// Scroll thoughts up
			m.scrollOffset = max(m.scrollOffset-3, 0)
			return m, nil
		case "G":
			m.scrollOffset = m.maxScroll()
			return m, nil
		case "g":
			m.scrollOffset = 0
			return m, nil
		case "esc":
			// Return to chat from any sub-panel
			if m.panel != panelChat {
				m.panel = panelChat
				return m, nil
			}
		}

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		return m, nil

	case busEventMsg:
		ev := Event(msg)
		// Log meaningful events to the bus panel (skip chunks and broadcasts with no target)
		if ev.Type != EventChunk {
			ts := time.Now().Format("15:04:05")
			text := ev.Text
			if len(text) > 60 {
				text = text[:60] + "..."
			}
			var logLine string
			if ev.Type == EventThinkDone {
				evCount := len(ev.ConsumedEvents)
				logLine = fmt.Sprintf("%s %s %s events=%d", ts, ev.Type, ev.From, evCount)
			} else if ev.To != "" {
				logLine = fmt.Sprintf("%s %s %s→%s %s", ts, ev.Type, ev.From, ev.To, text)
			} else {
				logLine = fmt.Sprintf("%s %s %s %s", ts, ev.Type, ev.From, text)
			}
			m.busLog = append(m.busLog, logLine)
			if len(m.busLog) > 200 {
				m.busLog = m.busLog[len(m.busLog)-200:]
			}
		}

		switch ev.Type {
		case EventChunk:
			v := m.getOrCreateView(ev.From)
			v.currentChunk.WriteString(ev.Text)
			v.iteration = ev.Iteration
			if m.activeTab == ev.From {
				m.scrollOffset = m.maxScroll()
			}
		case EventThinkDone:
			v := m.getOrCreateView(ev.From)
			v.thoughts = append(v.thoughts, thought{
				iteration: ev.Iteration,
				content:   v.currentChunk.String(),
				duration:  ev.Duration,
			})
			v.currentChunk.Reset()
			v.rate = ev.Rate
			v.model = ev.Model
			v.contextMsgs = ev.ContextMsgs
			v.contextChars = ev.ContextChars
			v.iterations++
			v.lastThought = time.Now()
			m.lastDuration = ev.Duration
			u := ev.Usage
			m.totalPromptTokens += u.PromptTokens
			m.totalCachedTokens += u.CachedTokens
			m.totalCompletionTokens += u.CompletionTokens
			iterCost := 0.0
			if m.thinker.provider != nil {
				iterCost = calculateCostForProvider(m.thinker.provider, u)
				m.totalCost += iterCost
			}
			v.cost += iterCost
			v.sleepDur = ev.SleepDuration
			if ev.From == "main" {
				m.memoryCount = ev.MemoryCount
				m.threadCount = ev.ThreadCount
			}
			if m.activeTab == ev.From {
				m.scrollOffset = m.maxScroll()
			}
		case EventThinkError:
			v := m.getOrCreateView(ev.From)
			v.thoughts = append(v.thoughts, thought{
				iteration: ev.Iteration,
				content:   fmt.Sprintf("ERROR: %v", ev.Error),
			})
			if m.activeTab == ev.From {
				m.scrollOffset = m.maxScroll()
			}
		case EventThreadStart:
			m.getOrCreateView(ev.From)
			m.threadCount = m.thinker.threads.Count()
		case EventThreadDone:
			m.removeTab(ev.From)
			m.threadCount = m.thinker.threads.Count()
		case EventThreadReply:
			m.chat = append(m.chat, chatMessage{isUser: false, text: ev.Text, threadID: ev.From})
		}
		return m, pollBusEvent(m.busSub)

	case tickMsg:
		// Poll telemetry events for the TUI panel
		if m.thinker.telemetry != nil {
			events, newCursor := m.thinker.telemetry.Events(m.telemetryCursor)
			m.telemetryCursor = newCursor
			for _, ev := range events {
				ts := ev.Time.Format("15:04:05")
				var line string
				switch {
				case ev.Type == "llm.done":
					var d LLMDoneData
					if json.Unmarshal(ev.Data, &d) == nil {
						line = fmt.Sprintf("%s %s %s %d→%d tok %dms $%.4f #%d",
							ts, telemetryLLMStyle.Render("llm.done"), ev.ThreadID,
							d.TokensIn, d.TokensOut, d.DurationMs, d.CostUSD, d.Iteration)
					}
				case ev.Type == "llm.error":
					var d LLMErrorData
					if json.Unmarshal(ev.Data, &d) == nil {
						line = fmt.Sprintf("%s %s %s %s",
							ts, telemetryErrorStyle.Render("llm.err"), ev.ThreadID, d.Error)
					}
				case ev.Type == "thread.spawn":
					var d ThreadSpawnData
					if json.Unmarshal(ev.Data, &d) == nil {
						dir := d.Directive
						if len(dir) > 40 {
							dir = dir[:40] + "..."
						}
						line = fmt.Sprintf("%s %s %s parent=%s %s",
							ts, telemetryThreadStyle.Render("t.spawn"), ev.ThreadID, d.ParentID, dir)
					}
				case ev.Type == "thread.done":
					line = fmt.Sprintf("%s %s %s",
						ts, telemetryThreadStyle.Render("t.done"), ev.ThreadID)
				case ev.Type == "thread.message":
					var d ThreadMessageData
					if json.Unmarshal(ev.Data, &d) == nil {
						msg := d.Message
						if len(msg) > 50 {
							msg = msg[:50] + "..."
						}
						line = fmt.Sprintf("%s %s %s→%s %s",
							ts, telemetryThreadStyle.Render("t.msg"), d.From, d.To, msg)
					}
				case ev.Type == "tool.pending":
					var d ToolCallData
					if json.Unmarshal(ev.Data, &d) == nil {
						m.pendingApproval = &d
						line = fmt.Sprintf("%s %s %s %s(%s)",
							ts, telemetryToolStyle.Render("APPROVE?"), ev.ThreadID, d.Name, d.Args)
					}
				case ev.Type == "tool.approved":
					var d ToolCallData
					if json.Unmarshal(ev.Data, &d) == nil {
						m.pendingApproval = nil
						line = fmt.Sprintf("%s %s %s %s",
							ts, telemetryToolStyle.Render("approved"), ev.ThreadID, d.Name)
					}
				case ev.Type == "tool.rejected":
					var d ToolCallData
					if json.Unmarshal(ev.Data, &d) == nil {
						m.pendingApproval = nil
						line = fmt.Sprintf("%s %s %s %s",
							ts, telemetryToolStyle.Render("rejected"), ev.ThreadID, d.Name)
					}
				case ev.Type == "tool.call":
					var d ToolCallData
					if json.Unmarshal(ev.Data, &d) == nil {
						line = fmt.Sprintf("%s %s %s %s",
							ts, telemetryToolStyle.Render("tool"), ev.ThreadID, d.Name)
					}
				case ev.Type == "tool.result":
					var d ToolResultData
					if json.Unmarshal(ev.Data, &d) == nil {
						status := "ok"
						if !d.Success {
							status = "fail"
						}
						line = fmt.Sprintf("%s %s %s %s %s %dms",
							ts, telemetryToolStyle.Render("tool.r"), ev.ThreadID, d.Name, status, d.DurationMs)
					}
				default:
					line = fmt.Sprintf("%s %s %s", ts, telemetryDimStyle.Render(ev.Type), ev.ThreadID)
				}
				if line != "" {
					m.telemetryLog = append(m.telemetryLog, line)
				}
			}
			if len(m.telemetryLog) > 500 {
				m.telemetryLog = m.telemetryLog[len(m.telemetryLog)-500:]
			}
		}
		return m, tickCmd()
	}

	return m, nil
}

func (m model) maxScroll() int {
	thoughtsWidth := m.thoughtsPanelWidth()
	content := m.renderThoughts(thoughtsWidth)
	lines := strings.Count(content, "\n")
	viewHeight := m.height - 6 // header(2) + tabs(1) + footer(1) + padding
	if lines > viewHeight {
		return lines - viewHeight
	}
	return 0
}

func (m model) leftPanelWidth() int {
	if m.width < 80 {
		return 0
	}
	return m.width / 3
}

func (m model) thoughtsPanelWidth() int {
	cp := m.leftPanelWidth()
	if cp == 0 {
		return m.width
	}
	return m.width - cp - 1
}

func (m model) renderThoughts(maxWidth int) string {
	v, ok := m.threadViews[m.activeTab]
	if !ok {
		return ""
	}

	var sb strings.Builder
	contentWidth := maxWidth - 4
	if contentWidth < 10 {
		contentWidth = 10
	}

	for _, t := range v.thoughts {
		header := thoughtHeaderStyle.Render(fmt.Sprintf("━━━ Thought #%d", t.iteration))
		if t.duration > 0 {
			header += statsStyle.Render(fmt.Sprintf(" (%s)", t.duration.Round(time.Millisecond)))
		}
		sb.WriteString(header + "\n")
		sb.WriteString(thoughtStyle.Render(wrapText(t.content, contentWidth)) + "\n\n")
	}

	if v.currentChunk.Len() > 0 {
		header := thoughtHeaderStyle.Render(fmt.Sprintf("━━━ Thought #%d", v.iteration))
		sb.WriteString(header + " ▍\n")
		sb.WriteString(thoughtStyle.Render(wrapText(v.currentChunk.String(), contentWidth)))
	}

	return sb.String()
}

func (m model) renderChatPanel(width, height int) string {
	if width <= 0 {
		return ""
	}
	innerWidth := width - 4
	if innerWidth < 5 {
		innerWidth = 5
	}

	title := chatTitleStyle.Render("Chat") + statsStyle.Render(fmt.Sprintf(" [%d thr │ %d mem]", m.threadCount, m.memoryCount))

	var inputArea string
	if m.inputActive {
		inputArea = inputLabelStyle.Render("> ") + m.input.View()
	} else {
		inputArea = helpStyle.Render("i: chat  c: command  /: menu")
	}

	listHeight := height - 4
	if listHeight < 1 {
		listHeight = 1
	}

	var lines []string
	for _, msg := range m.chat {
		if msg.isUser {
			wrapped := wrapText(msg.text, innerWidth-6)
			for i, line := range strings.Split(wrapped, "\n") {
				if i == 0 {
					lines = append(lines, chatUserStyle.Render("you: ")+line)
				} else {
					lines = append(lines, "     "+line)
				}
			}
		} else {
			label := "  ↩ "
			if msg.threadID != "" {
				label = fmt.Sprintf(" [%s] ", msg.threadID)
			}
			wrapped := wrapText(msg.text, innerWidth-len(label)-2)
			for i, line := range strings.Split(wrapped, "\n") {
				if i == 0 {
					lines = append(lines, chatAgentLabelStyle.Render(label)+chatAgentStyle.Render(line))
				} else {
					lines = append(lines, strings.Repeat(" ", len(label))+chatAgentStyle.Render(line))
				}
			}
		}
		lines = append(lines, "")
	}

	if len(lines) == 0 {
		lines = append(lines, statsStyle.Render("no messages yet"))
	}

	if len(lines) > listHeight {
		lines = lines[len(lines)-listHeight:]
	}
	for len(lines) < listHeight {
		lines = append(lines, "")
	}

	body := title + "\n" + strings.Join(lines, "\n") + "\n" + inputArea
	return panelBorderStyle.Width(innerWidth).Height(height - 2).Render(body)
}

func (m model) renderMemoryPanel(width, height int) string {
	if width <= 0 {
		return ""
	}
	innerWidth := width - 4
	if innerWidth < 5 {
		innerWidth = 5
	}

	title := memoryTitleStyle.Render(fmt.Sprintf("Memory (%d)", m.memoryCount))
	footer := helpStyle.Render("j/k: nav │ d: del │ esc: back")

	listHeight := height - 4
	if listHeight < 1 {
		listHeight = 1
	}

	entries := m.thinker.memory.Recent(m.memoryCount)
	var lines []string
	if len(entries) == 0 {
		lines = append(lines, statsStyle.Render("no memories yet"))
	} else {
		for idx := len(entries) - 1; idx >= 0; idx-- {
			e := entries[idx]
			displayIdx := len(entries) - 1 - idx
			age := formatAge(time.Since(e.Time))
			text := truncate(e.Text, innerWidth-6)

			if displayIdx == m.memoryCursor {
				lines = append(lines, memorySelectedStyle.Render(fmt.Sprintf(" %s ", text)))
				lines = append(lines, memoryAgeStyle.Render(fmt.Sprintf("  %s ago │ %s", age, e.Session[:min(8, len(e.Session))])))
			} else {
				lines = append(lines, memoryTextStyle.Render("  "+text))
				lines = append(lines, memoryAgeStyle.Render(fmt.Sprintf("  %s ago", age)))
			}
			lines = append(lines, "")
		}
	}

	cursorLine := m.memoryCursor * 3
	scrollStart := 0
	if cursorLine >= listHeight {
		scrollStart = cursorLine - listHeight + 3
	}
	if scrollStart > len(lines) {
		scrollStart = len(lines)
	}
	endLine := scrollStart + listHeight
	if endLine > len(lines) {
		endLine = len(lines)
	}

	visible := lines[scrollStart:endLine]
	for len(visible) < listHeight {
		visible = append(visible, "")
	}

	body := title + "\n" + strings.Join(visible, "\n") + "\n" + footer
	return panelBorderStyle.Width(innerWidth).Height(height - 2).Render(body)
}

func (m model) renderThreadPanel(width, height int) string {
	if width <= 0 {
		return ""
	}
	innerWidth := width - 4
	if innerWidth < 5 {
		innerWidth = 5
	}

	title := threadTitleStyle.Render(fmt.Sprintf("Threads (%d)", m.threadCount))
	footer := helpStyle.Render("j/k: nav │ d: kill │ esc: back")

	listHeight := height - 4
	if listHeight < 1 {
		listHeight = 1
	}

	threads := m.thinker.threads.List()
	var lines []string
	if len(threads) == 0 {
		lines = append(lines, statsStyle.Render("no threads"))
	} else {
		for i, thr := range threads {
			age := formatAge(time.Since(thr.Started))
			info := fmt.Sprintf("%s/%s #%d", thr.Rate, thr.Model, thr.Iteration)
			ctx := fmt.Sprintf("ctx:%dm/%dk", thr.ContextMsgs, thr.ContextChars/1000)
			toolStr := strings.Join(thr.Tools, ",")

			if i == m.threadCursor {
				lines = append(lines, threadSelectedStyle.Render(fmt.Sprintf(" %s ", thr.ID)))
				lines = append(lines, statsStyle.Render(fmt.Sprintf("  %s │ %s │ %s", info, ctx, age)))
				lines = append(lines, statsStyle.Render(fmt.Sprintf("  tools: %s", toolStr)))
			} else {
				lines = append(lines, threadActiveStyle.Render("  "+thr.ID))
				lines = append(lines, statsStyle.Render(fmt.Sprintf("  %s │ %s │ %s", info, ctx, age)))
			}
			lines = append(lines, "")
		}
	}

	cursorLine := m.threadCursor * 3
	scrollStart := 0
	if cursorLine >= listHeight {
		scrollStart = cursorLine - listHeight + 3
	}
	if scrollStart > len(lines) {
		scrollStart = len(lines)
	}
	endLine := scrollStart + listHeight
	if endLine > len(lines) {
		endLine = len(lines)
	}

	visible := lines[scrollStart:endLine]
	for len(visible) < listHeight {
		visible = append(visible, "")
	}

	body := title + "\n" + strings.Join(visible, "\n") + "\n" + footer
	return panelBorderStyle.Width(innerWidth).Height(height - 2).Render(body)
}

func (m model) renderConsolePanel(width, height int) string {
	if width <= 0 {
		return ""
	}
	innerWidth := width - 4
	if innerWidth < 5 {
		innerWidth = 5
	}

	title := consoleTitleStyle.Render("Console")

	var inputArea string
	if m.inputActive && m.inputMode == inputConsole {
		inputArea = consoleTitleStyle.Render("> ") + m.input.View()
	} else {
		inputArea = helpStyle.Render("c: command │ esc: back")
	}

	listHeight := height - 4
	if listHeight < 1 {
		listHeight = 1
	}

	var lines []string
	for _, cmd := range m.consoleHistory {
		lines = append(lines, consoleLineStyle.Render("> "+truncate(cmd, innerWidth-4)))
	}
	if len(lines) == 0 {
		lines = append(lines, statsStyle.Render("no commands yet"))
	}
	if len(lines) > listHeight {
		lines = lines[len(lines)-listHeight:]
	}
	for len(lines) < listHeight {
		lines = append(lines, "")
	}

	body := title + "\n" + strings.Join(lines, "\n") + "\n" + inputArea
	return panelBorderStyle.Width(innerWidth).Height(height - 2).Render(body)
}

func (m model) renderDirectivePanel(width, height int) string {
	if width <= 0 {
		return ""
	}
	innerWidth := width - 4
	if innerWidth < 5 {
		innerWidth = 5
	}

	title := directiveTitleStyle.Render("Directive (esc: save)")
	footer := helpStyle.Render("↑↓: nav │ type to edit │ esc: save")

	listHeight := height - 4
	if listHeight < 1 {
		listHeight = 1
	}

	var lines []string
	editWidth := innerWidth - 4
	if editWidth < 10 {
		editWidth = 10
	}
	for i, line := range m.directiveLines {
		wrapped := wrapText(line, editWidth)
		wrappedLines := strings.Split(wrapped, "\n")
		for j, wl := range wrappedLines {
			if i == m.directiveCursor {
				suffix := ""
				if j == len(wrappedLines)-1 {
					suffix = "▍"
				}
				lines = append(lines, directiveCursorStyle.Render(" "+wl+" ")+suffix)
			} else {
				lines = append(lines, directiveLineStyle.Render("  "+wl))
			}
		}
	}

	if len(lines) > listHeight {
		// Keep cursor visible
		start := 0
		if m.directiveCursor >= listHeight {
			start = m.directiveCursor - listHeight + 1
		}
		end := start + listHeight
		if end > len(lines) {
			end = len(lines)
		}
		lines = lines[start:end]
	}
	for len(lines) < listHeight {
		lines = append(lines, "")
	}

	body := title + "\n" + strings.Join(lines, "\n") + "\n" + footer
	return panelBorderStyle.Width(innerWidth).Height(height - 2).Render(body)
}

func (m model) renderToolsPanel(width, height int) string {
	if width <= 0 {
		return ""
	}
	innerWidth := width - 4
	if innerWidth < 5 {
		innerWidth = 5
	}

	toolCount := 0
	if m.thinker.registry != nil {
		toolCount = m.thinker.registry.Count()
	}
	title := toolsTitleStyle.Render(fmt.Sprintf("Tools (%d)", toolCount))
	footer := helpStyle.Render("esc: back")

	listHeight := height - 4
	if listHeight < 1 {
		listHeight = 1
	}

	var lines []string
	if m.thinker.registry != nil {
		tools := m.thinker.registry.AllTools()
		for _, tool := range tools {
			var style lipgloss.Style
			prefix := ""
			if tool.Core {
				style = toolCoreStyle
				prefix = "[core] "
			} else if strings.HasPrefix(tool.Description, "[") {
				style = toolMCPStyle
				prefix = "[mcp]  "
			} else {
				style = toolDiscoverStyle
				prefix = "[rag]  "
			}
			lines = append(lines, style.Render(prefix+tool.Name))
			desc := truncate(tool.Description, innerWidth-4)
			lines = append(lines, statsStyle.Render("  "+desc))
			lines = append(lines, "")
		}
	}

	if len(lines) == 0 {
		lines = append(lines, statsStyle.Render("no tools registered"))
	}

	if len(lines) > listHeight {
		lines = lines[:listHeight]
	}
	for len(lines) < listHeight {
		lines = append(lines, "")
	}

	body := title + "\n" + strings.Join(lines, "\n") + "\n" + footer
	return panelBorderStyle.Width(innerWidth).Height(height - 2).Render(body)
}

func (m model) renderBusPanel(width, height int) string {
	if width <= 0 {
		return ""
	}
	innerWidth := width - 4
	if innerWidth < 5 {
		innerWidth = 5
	}

	title := consoleTitleStyle.Render(fmt.Sprintf("Event Bus (%d)", len(m.busLog)))
	footer := helpStyle.Render("esc: back")

	listHeight := height - 4
	if listHeight < 1 {
		listHeight = 1
	}

	var lines []string
	if len(m.busLog) == 0 {
		lines = append(lines, statsStyle.Render("no events yet"))
	} else {
		for _, entry := range m.busLog {
			// Wrap long lines
			if len(entry) > innerWidth-2 {
				entry = entry[:innerWidth-2]
			}
			lines = append(lines, statsStyle.Render(entry))
		}
	}

	// Show most recent at bottom
	if len(lines) > listHeight {
		lines = lines[len(lines)-listHeight:]
	}
	for len(lines) < listHeight {
		lines = append(lines, "")
	}

	body := title + "\n" + strings.Join(lines, "\n") + "\n" + footer
	return panelBorderStyle.Width(innerWidth).Height(height - 2).Render(body)
}

func (m model) renderTelemetryPanel(width, height int) string {
	if width <= 0 {
		return ""
	}
	innerWidth := width - 4
	if innerWidth < 5 {
		innerWidth = 5
	}

	title := telemetryTitleStyle.Render(fmt.Sprintf("Telemetry (%d)", len(m.telemetryLog)))
	footer := helpStyle.Render("esc: back")

	listHeight := height - 4
	if listHeight < 1 {
		listHeight = 1
	}

	var lines []string
	if len(m.telemetryLog) == 0 {
		lines = append(lines, statsStyle.Render("no telemetry events yet"))
	} else {
		for _, entry := range m.telemetryLog {
			if len(entry) > innerWidth-2 {
				entry = entry[:innerWidth-2]
			}
			lines = append(lines, entry)
		}
	}

	// Show most recent at bottom
	if len(lines) > listHeight {
		lines = lines[len(lines)-listHeight:]
	}
	for len(lines) < listHeight {
		lines = append(lines, "")
	}

	body := title + "\n" + strings.Join(lines, "\n") + "\n" + footer
	return panelBorderStyle.Width(innerWidth).Height(height - 2).Render(body)
}

func (m model) renderSidebar(width, height int) string {
	if width <= 0 {
		return ""
	}
	innerWidth := width - 4
	if innerWidth < 8 {
		innerWidth = 8
	}

	var lines []string
	lastSection := sidebarSection(-1)

	for i, item := range sidebarItems {
		// Section headers
		if item.section != lastSection {
			if lastSection != -1 {
				lines = append(lines, "")
			}
			switch item.section {
			case sidebarView:
				lines = append(lines, helpStyle.Render("── View ──"))
			case sidebarSettings:
				lines = append(lines, helpStyle.Render("── Settings ──"))
			}
			lastSection = item.section
		}

		label := item.label

		// Show current value for settings items (truncated to fit)
		maxVal := innerWidth - len(item.label) - 6 // "   " prefix + "  " gap + some padding
		if maxVal < 4 {
			maxVal = 4
		}
		switch item.action {
		case "provider":
			if m.thinker.provider != nil {
				val := m.thinker.provider.Name()
				if len(val) > maxVal {
					val = val[:maxVal]
				}
				label += helpStyle.Render(" " + val)
			}
		case "model":
			if gp, ok := m.thinker.provider.(*GoogleProvider); ok {
				val := strings.TrimPrefix(gp.ActiveModel(), "gemini-")
				if len(val) > maxVal {
					val = val[:maxVal]
				}
				label += helpStyle.Render(" " + val)
			}
		case "mode":
			val := string(m.thinker.config.GetMode())
			if len(val) > maxVal {
				val = val[:maxVal]
			}
			label += helpStyle.Render(" " + val)
		}

		// Mark active panel
		isActive := (item.action == "" && item.panel == m.panel)

		// Truncate label to fit sidebar width
		maxLabel := innerWidth - 5 // account for " ▸ " prefix + padding
		if maxLabel < 4 {
			maxLabel = 4
		}
		if len(label) > maxLabel {
			label = label[:maxLabel]
		}

		if i == m.sidebarCursor {
			rendered := fmt.Sprintf(" ▸ %s", label)
			// Pad to full width for consistent highlight
			for len(rendered) < innerWidth {
				rendered += " "
			}
			lines = append(lines, lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("235")).
				Background(lipgloss.Color("39")).
				Render(rendered))
		} else if isActive {
			lines = append(lines, lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("39")).
				Render("   "+label))
		} else {
			lines = append(lines, lipgloss.NewStyle().
				Foreground(lipgloss.Color("252")).
				Render("   "+label))
		}
	}

	// Pad remaining height
	for len(lines) < height-2 {
		lines = append(lines, "")
	}
	if len(lines) > height-2 {
		lines = lines[:height-2]
	}

	return panelBorderStyle.Width(innerWidth).Height(height - 2).Render(strings.Join(lines, "\n"))
}

func (m model) renderPicker(width, height int) string {
	if width <= 0 {
		return ""
	}
	innerWidth := width - 4
	if innerWidth < 10 {
		innerWidth = 10
	}

	var title string
	switch m.picker {
	case pickerProvider:
		title = "Select Provider"
	case pickerModel:
		title = "Select Model"
	case pickerMode_:
		title = "Select Mode"
	}

	titleLine := lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("39")).Padding(0, 1).Render(title)
	footer := helpStyle.Render("↑/↓: navigate │ enter: select │ esc: back")

	listHeight := height - 4
	if listHeight < 1 {
		listHeight = 1
	}

	var lines []string
	for i, item := range m.pickerItems {
		if i == m.pickerCursor {
			rendered := fmt.Sprintf(" ▸ %s ", item)
			if len(rendered) > innerWidth {
				rendered = rendered[:innerWidth]
			}
			lines = append(lines, lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("235")).
				Background(lipgloss.Color("39")).
				Render(rendered))
		} else {
			lines = append(lines, lipgloss.NewStyle().
				Foreground(lipgloss.Color("252")).
				Render("   "+item))
		}
		lines = append(lines, "")
	}

	if len(lines) > listHeight {
		lines = lines[:listHeight]
	}
	for len(lines) < listHeight {
		lines = append(lines, "")
	}

	body := titleLine + "\n" + strings.Join(lines, "\n") + "\n" + footer
	return panelBorderStyle.Width(innerWidth).Height(height - 2).Render(body)
}

func (m model) renderPalette() string {
	maxWidth := m.width / 2
	if maxWidth < 40 {
		maxWidth = 40
	}
	if maxWidth > 60 {
		maxWidth = 60
	}
	innerWidth := maxWidth - 4

	inputLine := lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("39")).Render("/ ") + m.paletteInput.View()

	var lines []string
	for i, item := range m.paletteItems {
		label := item.label
		if item.section == sidebarSettings {
			label += helpStyle.Render(" (setting)")
		}
		if i == m.pickerCursor {
			rendered := fmt.Sprintf(" ▸ %s", label)
			if len(rendered) > innerWidth {
				rendered = rendered[:innerWidth]
			}
			lines = append(lines, lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("235")).
				Background(lipgloss.Color("39")).
				Render(rendered+" "))
		} else {
			lines = append(lines, lipgloss.NewStyle().
				Foreground(lipgloss.Color("252")).
				Render("   "+label))
		}
	}

	maxItems := 12
	if len(lines) > maxItems {
		lines = lines[:maxItems]
	}

	body := inputLine + "\n" + strings.Join(lines, "\n")

	return lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("39")).
		Width(innerWidth).
		Padding(0, 1).
		Render(body)
}

func (m model) renderTabBar() string {
	var parts []string
	for _, tab := range m.tabs {
		label := tab
		if v, ok := m.threadViews[tab]; ok && tab != "main" {
			label = fmt.Sprintf("%s #%d", tab, v.iteration)
		}
		if tab == m.activeTab {
			parts = append(parts, tabActiveStyle.Render(label))
		} else {
			parts = append(parts, tabInactiveStyle.Render(label))
		}
	}
	return strings.Join(parts, " ")
}

func (m model) View() string {
	if m.width == 0 {
		return "Loading..."
	}

	elapsed := time.Since(m.startTime).Round(time.Second)
	title := titleStyle.Render("Apteva")

	// Show active tab's status
	var statusRender string
	var ctxInfo string
	providerName := ""
	if m.thinker.provider != nil {
		if gp, ok := m.thinker.provider.(*GoogleProvider); ok {
			// Shorten: "gemini-3.1-pro-preview" → "g-3.1-pro"
			name := gp.ActiveModel()
			name = strings.TrimPrefix(name, "gemini-")
			name = strings.TrimSuffix(name, "-preview")
			providerName = "g-" + name
		} else {
			providerName = m.thinker.provider.Name()
		}
	}
	if m.paused {
		statusRender = pausedStyle.Render("PAUSED")
	} else if v, ok := m.threadViews[m.activeTab]; ok {
		sleepStr := formatSleep(v.sleepDur)
		if v.sleepDur == 0 {
			sleepStr = v.rate.String()
		}
		statusRender = statusBarStyle.Render(fmt.Sprintf("THINKING (%s/%s/%s)", providerName, sleepStr, v.model))
		if v.contextChars > 0 {
			estTokens := v.contextChars / 4 // rough estimate: ~4 chars per token
			ctxInfo = fmt.Sprintf(" │ ctx: %dm/~%dtok", v.contextMsgs, estTokens)
		}
	} else {
		statusRender = statusBarStyle.Render("THINKING")
	}

	toolInfo := ""
	if m.thinker.registry != nil {
		core, rag, total := m.thinker.registry.Counts()
		toolInfo = fmt.Sprintf(" │ tools: %d(%dc+%dr)", total, core, rag)
	}

	totalTok := m.totalPromptTokens + m.totalCompletionTokens

	// Projected cost
	projectedPerHour := 0.0
	for _, v := range m.threadViews {
		if v.iterations == 0 || v.cost == 0 {
			continue
		}
		costPerIter := v.cost / float64(v.iterations)
		threadElapsed := time.Since(v.started)
		avgThinkDur := threadElapsed / time.Duration(v.iterations)
		if avgThinkDur > v.sleepDur {
			avgThinkDur = 2 * time.Second
		}
		cycleDur := avgThinkDur + v.sleepDur
		if cycleDur < time.Second {
			cycleDur = time.Second
		}
		itersPerHour := float64(time.Hour) / float64(cycleDur)
		projectedPerHour += costPerIter * itersPerHour
	}
	projectedPerDay := projectedPerHour * 24

	// Single header line: title + status + provider + cost
	header := fmt.Sprintf(
		"%s  %s  %s  tok:%d $%.4f $%.2f/d │ %s │ thr:%d mem:%d%s%s",
		title, statusRender, statsStyle.Render(providerName),
		totalTok, m.totalCost, projectedPerDay,
		elapsed, m.threadCount, m.memoryCount, ctxInfo, toolInfo,
	)
	// Truncate to terminal width to prevent wrapping
	headerPlain := lipgloss.Width(header)
	if headerPlain > m.width {
		// Simplified fallback
		header = fmt.Sprintf("%s  %s  %s  tok:%d $%.4f",
			title, statusRender, statsStyle.Render(providerName), totalTok, m.totalCost)
	}

	// Tab bar
	tabBar := m.renderTabBar() + statsStyle.Render("  tab/shift+tab: switch")

	var footer string
	if m.pendingApproval != nil {
		args := m.pendingApproval.Args
		if len(args) > 60 {
			args = args[:60] + "..."
		}
		footer = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("11")).Render(
			fmt.Sprintf("⚡ APPROVE? %s(%s)  [y] approve  [n] reject", m.pendingApproval.Name, args))
	} else {
		footer = helpStyle.Render("space: pause │ i: chat │ c: cmd │ /: menu │ tab: threads │ g/G: top/btm │ q: quit")
	}

	// header(1) + tab bar(1) + footer(1) = 3 chrome lines + 1 buffer
	viewHeight := m.height - 4
	if viewHeight < 1 {
		viewHeight = 1
	}

	// Layout: sidebar (narrow) │ left panel │ thoughts
	sidebarWidth := 0
	if m.width >= 100 {
		sidebarWidth = 22
	}

	leftWidth := m.leftPanelWidth()
	if sidebarWidth > 0 {
		// Reduce left panel to make room for sidebar
		leftWidth = (m.width - sidebarWidth - 2) / 3
	}

	thoughtsWidth := m.width - sidebarWidth - leftWidth - 2
	if thoughtsWidth < 20 {
		thoughtsWidth = 20
	}

	content := m.renderThoughts(thoughtsWidth)
	lines := strings.Split(content, "\n")

	start := m.scrollOffset
	if start > len(lines) {
		start = len(lines)
	}
	end := start + viewHeight
	if end > len(lines) {
		end = len(lines)
	}

	visible := strings.Join(lines[start:end], "\n")
	visibleLines := strings.Count(visible, "\n") + 1
	if visibleLines < viewHeight {
		visible += strings.Repeat("\n", viewHeight-visibleLines)
	}

	// Assemble panels left to right
	if leftWidth > 0 {
		var leftPanel string
		if m.picker != pickerNone {
			leftPanel = m.renderPicker(leftWidth, viewHeight)
		} else {
			switch m.panel {
			case panelConsole:
				leftPanel = m.renderConsolePanel(leftWidth, viewHeight)
			case panelMemory:
				leftPanel = m.renderMemoryPanel(leftWidth, viewHeight)
			case panelThreads:
				leftPanel = m.renderThreadPanel(leftWidth, viewHeight)
			case panelDirective:
				leftPanel = m.renderDirectivePanel(leftWidth, viewHeight)
			case panelTools:
				leftPanel = m.renderToolsPanel(leftWidth, viewHeight)
			case panelBus:
				leftPanel = m.renderBusPanel(leftWidth, viewHeight)
			case panelTelemetry:
				leftPanel = m.renderTelemetryPanel(leftWidth, viewHeight)
			default:
				leftPanel = m.renderChatPanel(leftWidth, viewHeight)
			}
		}
		visible = lipgloss.JoinHorizontal(lipgloss.Top, leftPanel, " ", visible)
	}

	if sidebarWidth > 0 {
		sidebar := m.renderSidebar(sidebarWidth, viewHeight)
		visible = lipgloss.JoinHorizontal(lipgloss.Top, sidebar, " ", visible)
	}

	// Ensure visible doesn't exceed viewHeight lines (panels/JoinHorizontal can add extra)
	visLines := strings.Split(visible, "\n")
	if len(visLines) > viewHeight {
		visLines = visLines[:viewHeight]
		visible = strings.Join(visLines, "\n")
	}

	result := header + "\n" + tabBar + "\n" + visible + "\n" + footer

	// Overlay command palette if open
	if m.palette == paletteOpen {
		palette := m.renderPalette()
		// Place palette roughly centered
		paletteLines := strings.Split(palette, "\n")
		resultLines := strings.Split(result, "\n")
		startY := 3 // below header
		startX := (m.width - 60) / 2
		if startX < 0 {
			startX = 0
		}
		for i, pl := range paletteLines {
			row := startY + i
			if row < len(resultLines) {
				line := resultLines[row]
				// Overlay palette onto the line
				if startX < len(line) {
					padded := pl
					for len(padded) < len(pl) {
						padded += " "
					}
					resultLines[row] = line[:startX] + padded
				} else {
					resultLines[row] = line + strings.Repeat(" ", startX-len(line)) + pl
				}
			}
		}
		result = strings.Join(resultLines, "\n")
	}

	return result
}

func wrapText(s string, width int) string {
	if width <= 0 {
		return s
	}
	var result strings.Builder
	for _, paragraph := range strings.Split(s, "\n") {
		if result.Len() > 0 {
			result.WriteByte('\n')
		}
		words := strings.Fields(paragraph)
		if len(words) == 0 {
			continue
		}
		lineLen := 0
		for i, w := range words {
			wl := len(w)
			if i > 0 && lineLen+1+wl > width {
				result.WriteByte('\n')
				lineLen = 0
			} else if i > 0 {
				result.WriteByte(' ')
				lineLen++
			}
			result.WriteString(w)
			lineLen += wl
		}
	}
	return result.String()
}

func truncate(s string, maxLen int) string {
	if maxLen <= 0 {
		return ""
	}
	if len(s) <= maxLen {
		return s
	}
	if maxLen <= 3 {
		return s[:maxLen]
	}
	return s[:maxLen-3] + "..."
}

// mediaLabel returns a display label like "[+2 img, +1 audio]" for media parts.
func mediaLabel(parts []ContentPart) string {
	imgs, auds := 0, 0
	for _, p := range parts {
		switch p.Type {
		case "image_url":
			imgs++
		case "audio_url", "input_audio":
			auds++
		}
	}
	if imgs > 0 && auds > 0 {
		return fmt.Sprintf("[+%d img, +%d audio]", imgs, auds)
	}
	if imgs > 0 {
		return fmt.Sprintf("[+%d img]", imgs)
	}
	if auds > 0 {
		return fmt.Sprintf("[+%d audio]", auds)
	}
	return ""
}

// mediaURLRe matches image and audio URLs in text.
var mediaURLRe = regexp.MustCompile(`https?://\S+\.(?:png|jpg|jpeg|gif|webp|mp3|wav|aac|ogg|flac|aiff|m4a)(?:\?\S*)?`)

// audioExts maps file extensions to their media type.
var audioExts = map[string]bool{
	"mp3": true, "wav": true, "aac": true, "ogg": true,
	"flac": true, "aiff": true, "m4a": true,
}

// detectImageParts scans text for image and audio URLs and returns ContentParts if found.
// Returns nil if no media URLs detected.
func detectImageParts(text string) []ContentPart {
	matches := mediaURLRe.FindAllString(text, -1)
	if len(matches) == 0 {
		return nil
	}
	// Build parts: text first, then media
	cleanText := text
	for _, url := range matches {
		cleanText = strings.Replace(cleanText, url, "", 1)
	}
	cleanText = strings.TrimSpace(cleanText)

	var parts []ContentPart
	if cleanText != "" {
		parts = append(parts, ContentPart{Type: "text", Text: cleanText})
	}
	for _, url := range matches {
		// Detect extension to classify as image or audio
		ext := strings.ToLower(url)
		if idx := strings.LastIndex(ext, "."); idx >= 0 {
			ext = ext[idx+1:]
		}
		// Strip query params from extension
		if idx := strings.Index(ext, "?"); idx >= 0 {
			ext = ext[:idx]
		}
		if audioExts[ext] {
			parts = append(parts, ContentPart{Type: "audio_url", AudioURL: &AudioURL{URL: url}})
		} else {
			parts = append(parts, ContentPart{Type: "image_url", ImageURL: &ImageURL{URL: url}})
		}
	}
	return parts
}
