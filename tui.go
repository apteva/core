package main

import (
	"fmt"
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
)

type inputMode int

const (
	inputChat    inputMode = iota
	inputConsole
	inputDirective
)

type threadView struct {
	thoughts     []thought
	currentChunk *strings.Builder
	iteration    int
	rate         ThinkRate
	model        ModelTier
	contextMsgs  int
	contextChars int
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
	memoryCount    int
	threadCount    int

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
	}
}

func (m *model) getOrCreateView(id string) *threadView {
	if v, ok := m.threadViews[id]; ok {
		return v
	}
	v := &threadView{currentChunk: &strings.Builder{}}
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

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		// Directive editing mode
		if m.panel == panelDirective {
			switch msg.String() {
			case "esc":
				// Save and exit
				directive := strings.Join(m.directiveLines, "\n")
				m.thinker.config.SetDirective(directive)
				m.thinker.ReloadDirective()
				m.panel = panelChat
				return m, nil
			case "enter":
				// Insert new line after cursor
				m.directiveLines = append(m.directiveLines[:m.directiveCursor+1],
					append([]string{""}, m.directiveLines[m.directiveCursor+1:]...)...)
				m.directiveCursor++
				return m, nil
			case "backspace":
				if m.directiveCursor > 0 && len(m.directiveLines[m.directiveCursor]) == 0 {
					// Delete empty line
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
				// Type characters into current line
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

		if m.inputActive {
			switch msg.String() {
			case "enter":
				val := strings.TrimSpace(m.input.Value())
				if val != "" {
					switch m.inputMode {
					case inputChat:
						m.thinker.InjectUserMessage(m.userID, val)
						m.chat = append(m.chat, chatMessage{isUser: true, text: val, threadID: m.userID})
					case inputConsole:
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

		switch msg.String() {
		case "q", "ctrl+c":
			m.thinker.Stop()
			return m, tea.Quit
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
		case "e":
			m.panel = panelDirective
			m.directiveLines = strings.Split(m.thinker.config.GetDirective(), "\n")
			m.directiveCursor = 0
			return m, nil
		case "b":
			if m.panel == panelBus {
				m.panel = panelChat
			} else {
				m.panel = panelBus
			}
			return m, nil
		case "o":
			if m.panel == panelTools {
				m.panel = panelChat
			} else {
				m.panel = panelTools
			}
			return m, nil
		case "m":
			if m.panel == panelMemory {
				m.panel = panelChat
			} else {
				m.panel = panelMemory
				m.memoryCursor = 0
			}
			return m, nil
		case "t":
			if m.panel == panelThreads {
				m.panel = panelChat
			} else {
				m.panel = panelThreads
				m.threadCursor = 0
			}
			return m, nil
		case "d":
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
			// Next tab
			for i, tab := range m.tabs {
				if tab == m.activeTab {
					m.activeTab = m.tabs[(i+1)%len(m.tabs)]
					m.scrollOffset = 0
					break
				}
			}
			return m, nil
		case "[", "shift+tab":
			// Prev tab
			for i, tab := range m.tabs {
				if tab == m.activeTab {
					idx := (i - 1 + len(m.tabs)) % len(m.tabs)
					m.activeTab = m.tabs[idx]
					m.scrollOffset = 0
					break
				}
			}
			return m, nil
		case " ":
			m.paused = !m.paused
			m.thinker.TogglePause()
			return m, nil
		case "j", "down":
			switch m.panel {
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
		case "G":
			if m.panel == panelChat {
				m.scrollOffset = m.maxScroll()
			}
			return m, nil
		case "g":
			if m.panel == panelChat {
				m.scrollOffset = 0
			}
			return m, nil
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
			m.lastDuration = ev.Duration
			u := ev.Usage
			m.totalPromptTokens += u.PromptTokens
			m.totalCachedTokens += u.CachedTokens
			m.totalCompletionTokens += u.CompletionTokens
			uncachedInput := u.PromptTokens - u.CachedTokens
			if uncachedInput < 0 {
				uncachedInput = 0
			}
			m.totalCost += float64(uncachedInput)*priceInputPerToken +
				float64(u.CachedTokens)*priceCachedPerToken +
				float64(u.CompletionTokens)*priceOutputPerToken
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
		inputArea = helpStyle.Render("i:chat c:cmd e:dir o:tools m:mem b:bus")
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
	footer := helpStyle.Render("j/k: nav │ d: del │ m: back")

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
	footer := helpStyle.Render("j/k: nav │ d: kill │ t: back")

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
		inputArea = helpStyle.Render("c: command │ i: chat │ e: dir")
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
	footer := helpStyle.Render("o: back")

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
	footer := helpStyle.Render("b: back")

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
	title := titleStyle.Render("Continuous Thinking Engine")

	// Show active tab's status
	var statusRender string
	var ctxInfo string
	if m.paused {
		statusRender = pausedStyle.Render("PAUSED")
	} else if v, ok := m.threadViews[m.activeTab]; ok {
		statusRender = statusBarStyle.Render(fmt.Sprintf("THINKING (%s/%s)", v.rate, v.model))
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

	stats := statsStyle.Render(fmt.Sprintf(
		"%s │ thr: %d │ mem: %d%s%s",
		elapsed, m.threadCount, m.memoryCount, ctxInfo, toolInfo,
	))

	header := lipgloss.JoinHorizontal(lipgloss.Center, title, "  ", statusRender, "  ", stats)

	totalTok := m.totalPromptTokens + m.totalCompletionTokens
	costPerHour := 0.0
	costPerDay := 0.0
	elapsedHours := elapsed.Hours()
	if elapsedHours > 0 {
		costPerHour = m.totalCost / elapsedHours
		costPerDay = costPerHour * 24
	}

	costLine := statsStyle.Render(fmt.Sprintf(
		"tok: %d (in:%d cached:%d out:%d) │ $%.4f │ $%.2f/hr │ $%.2f/day",
		totalTok, m.totalPromptTokens, m.totalCachedTokens, m.totalCompletionTokens,
		m.totalCost, costPerHour, costPerDay,
	))

	header = header + "\n" + costLine

	// Tab bar
	tabBar := m.renderTabBar() + statsStyle.Render("  [/]: switch tabs")

	footer := helpStyle.Render("space: pause │ j/k: scroll │ g/G: top/bottom │ q: quit")

	// 3 header lines + tab bar + footer = 5 non-content lines
	viewHeight := m.height - 6
	if viewHeight < 1 {
		viewHeight = 1
	}

	thoughtsWidth := m.thoughtsPanelWidth()
	leftWidth := m.leftPanelWidth()

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

	if leftWidth > 0 {
		var leftPanel string
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
		default:
			leftPanel = m.renderChatPanel(leftWidth, viewHeight)
		}
		visible = lipgloss.JoinHorizontal(lipgloss.Top, leftPanel, " ", visible)
	}

	return header + "\n" + tabBar + "\n" + visible + "\n" + footer
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
