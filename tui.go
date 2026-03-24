package main

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// Styles
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
)

type tickMsg time.Time
type thinkEventMsg ThinkEvent

const (
	priceInputPerToken  = 0.60 / 1_000_000
	priceCachedPerToken = 0.10 / 1_000_000
	priceOutputPerToken = 3.00 / 1_000_000
)

type chatMessage struct {
	isUser bool
	text   string
}

type panelMode int

const (
	panelChat   panelMode = iota
	panelMemory
)

type model struct {
	thinker      *Thinker
	thoughts     []thought
	currentChunk *strings.Builder
	width        int
	height       int
	scrollOffset int
	paused       bool
	iteration    int
	startTime    time.Time
	lastDuration time.Duration
	input        textinput.Model
	inputActive  bool

	chat        []chatMessage
	rate        ThinkRate
	memoryCount int

	// Left panel mode
	panel         panelMode
	memoryCursor  int

	// Token accounting
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
		thinker:      thinker,
		thoughts:     []thought{},
		currentChunk: &strings.Builder{},
		startTime:    time.Now(),
		input:        ti,
		rate:         RateSlow,
		memoryCount:  thinker.memory.Count(),
	}
}

func (m model) Init() tea.Cmd {
	return tea.Batch(
		pollEvents(m.thinker),
		tickCmd(),
	)
}

func pollEvents(t *Thinker) tea.Cmd {
	return func() tea.Msg {
		ev := <-t.events
		return thinkEventMsg(ev)
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
		if m.inputActive {
			switch msg.String() {
			case "enter":
				val := strings.TrimSpace(m.input.Value())
				if val != "" {
					m.thinker.InjectUserMessage(val)
					m.chat = append(m.chat, chatMessage{isUser: true, text: val})
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
		case "i", "/":
			if m.panel == panelChat {
				m.inputActive = true
				m.input.Focus()
				return m, textinput.Blink
			}
		case "m":
			if m.panel == panelChat {
				m.panel = panelMemory
				m.memoryCursor = 0
			} else {
				m.panel = panelChat
			}
			return m, nil
		case "d":
			if m.panel == panelMemory {
				entries := m.thinker.memory.Recent(m.thinker.memory.Count())
				if len(entries) > 0 && m.memoryCursor < len(entries) {
					// Delete from the actual store (index is reversed since Recent returns newest-last)
					realIndex := m.thinker.memory.Count() - 1 - m.memoryCursor
					m.thinker.memory.Delete(realIndex)
					m.memoryCount = m.thinker.memory.Count()
					if m.memoryCursor >= m.memoryCount && m.memoryCursor > 0 {
						m.memoryCursor--
					}
				}
				return m, nil
			}
		case " ":
			m.paused = !m.paused
			m.thinker.TogglePause()
			return m, nil
		case "j", "down":
			if m.panel == panelMemory {
				if m.memoryCursor < m.memoryCount-1 {
					m.memoryCursor++
				}
			} else {
				m.scrollOffset = min(m.scrollOffset+1, m.maxScroll())
			}
			return m, nil
		case "k", "up":
			if m.panel == panelMemory {
				if m.memoryCursor > 0 {
					m.memoryCursor--
				}
			} else {
				m.scrollOffset = max(m.scrollOffset-1, 0)
			}
			return m, nil
		case "G":
			if m.panel != panelMemory {
				m.scrollOffset = m.maxScroll()
			}
			return m, nil
		case "g":
			if m.panel != panelMemory {
				m.scrollOffset = 0
			}
			return m, nil
		}

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		return m, nil

	case thinkEventMsg:
		ev := ThinkEvent(msg)
		if ev.Error != nil {
			m.thoughts = append(m.thoughts, thought{
				iteration: ev.Iteration,
				content:   fmt.Sprintf("ERROR: %v", ev.Error),
			})
			m.scrollOffset = m.maxScroll()
			return m, pollEvents(m.thinker)
		}

		if ev.Chunk != "" {
			m.currentChunk.WriteString(ev.Chunk)
			m.iteration = ev.Iteration
			m.scrollOffset = m.maxScroll()
		}

		if ev.Done {
			content := m.currentChunk.String()
			m.thoughts = append(m.thoughts, thought{
				iteration: ev.Iteration,
				content:   content,
				duration:  ev.Duration,
			})
			m.currentChunk.Reset()
			m.lastDuration = ev.Duration
			m.rate = ev.Rate
			m.memoryCount = ev.MemoryCount

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

			for _, reply := range ev.Replies {
				m.chat = append(m.chat, chatMessage{isUser: false, text: reply})
			}

			m.scrollOffset = m.maxScroll()
		}

		return m, pollEvents(m.thinker)

	case tickMsg:
		return m, tickCmd()
	}

	return m, nil
}

func (m model) maxScroll() int {
	thoughtsWidth := m.thoughtsPanelWidth()
	content := m.renderThoughts(thoughtsWidth)
	lines := strings.Count(content, "\n")
	viewHeight := m.height - 5
	if lines > viewHeight {
		return lines - viewHeight
	}
	return 0
}

func (m model) leftPanelWidth() int {
	if m.width < 80 {
		return 0
	}
	w := m.width / 3
	if w > 44 {
		w = 44
	}
	if w < 30 {
		w = 30
	}
	return w
}

func (m model) thoughtsPanelWidth() int {
	cp := m.leftPanelWidth()
	if cp == 0 {
		return m.width
	}
	return m.width - cp - 1
}

func (m model) renderThoughts(maxWidth int) string {
	var sb strings.Builder

	contentWidth := maxWidth - 4
	if contentWidth < 10 {
		contentWidth = 10
	}

	for _, t := range m.thoughts {
		header := thoughtHeaderStyle.Render(fmt.Sprintf("━━━ Thought #%d", t.iteration))
		if t.duration > 0 {
			header += statsStyle.Render(fmt.Sprintf(" (%s)", t.duration.Round(time.Millisecond)))
		}
		sb.WriteString(header + "\n")
		sb.WriteString(thoughtStyle.Render(wrapText(t.content, contentWidth)) + "\n\n")
	}

	if m.currentChunk.Len() > 0 {
		header := thoughtHeaderStyle.Render(fmt.Sprintf("━━━ Thought #%d", m.iteration))
		sb.WriteString(header + " ▍\n")
		sb.WriteString(thoughtStyle.Render(wrapText(m.currentChunk.String(), contentWidth)))
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

	title := chatTitleStyle.Render("Chat") + statsStyle.Render(fmt.Sprintf(" [%d mem]", m.memoryCount))

	var inputArea string
	if m.inputActive {
		inputArea = inputLabelStyle.Render("> ") + m.input.View()
	} else {
		inputArea = helpStyle.Render("i: chat │ m: memory")
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
			wrapped := wrapText(msg.text, innerWidth-6)
			for i, line := range strings.Split(wrapped, "\n") {
				if i == 0 {
					lines = append(lines, chatAgentLabelStyle.Render("  ↩ ")+chatAgentStyle.Render(line))
				} else {
					lines = append(lines, "    "+chatAgentStyle.Render(line))
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

	return panelBorderStyle.
		Width(innerWidth).
		Height(height - 2).
		Render(body)
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

	footer := helpStyle.Render("j/k: nav │ d: delete │ m: chat")

	listHeight := height - 4
	if listHeight < 1 {
		listHeight = 1
	}

	// Get recent memories (newest last)
	entries := m.thinker.memory.Recent(m.memoryCount)

	var lines []string
	if len(entries) == 0 {
		lines = append(lines, statsStyle.Render("no memories yet"))
	} else {
		// Show newest first
		for idx := len(entries) - 1; idx >= 0; idx-- {
			e := entries[idx]
			displayIdx := len(entries) - 1 - idx
			age := formatAge(time.Since(e.Time))
			text := truncate(e.Text, innerWidth-6)

			if displayIdx == m.memoryCursor {
				lines = append(lines, memorySelectedStyle.Render(fmt.Sprintf(" %s ", text)))
				lines = append(lines, memoryAgeStyle.Render(fmt.Sprintf("  %s ago │ %s", age, e.Session[:8])))
			} else {
				lines = append(lines, memoryTextStyle.Render("  "+text))
				lines = append(lines, memoryAgeStyle.Render(fmt.Sprintf("  %s ago", age)))
			}
			lines = append(lines, "")
		}
	}

	// Scroll to keep cursor visible
	cursorLine := m.memoryCursor * 3 // 3 lines per entry (text + age + blank)
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

	return panelBorderStyle.
		Width(innerWidth).
		Height(height - 2).
		Render(body)
}

func (m model) View() string {
	if m.width == 0 {
		return "Loading..."
	}

	elapsed := time.Since(m.startTime).Round(time.Second)
	title := titleStyle.Render("Continuous Thinking Engine")

	var statusRender string
	if m.paused {
		statusRender = pausedStyle.Render("PAUSED")
	} else {
		statusRender = statusBarStyle.Render(fmt.Sprintf("THINKING (%s)", m.rate))
	}

	stats := statsStyle.Render(fmt.Sprintf(
		"#%d │ %s │ %s/thought │ next: %s",
		m.iteration, elapsed, m.lastDuration.Round(time.Millisecond), m.rate.Delay(),
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
		"tok: %d (in:%d cached:%d out:%d) │ $%.4f │ $%.2f/hr │ $%.2f/day │ mem: %d",
		totalTok, m.totalPromptTokens, m.totalCachedTokens, m.totalCompletionTokens,
		m.totalCost, costPerHour, costPerDay, m.memoryCount,
	))

	header = header + "\n" + costLine

	footer := helpStyle.Render("space: pause │ j/k: scroll │ g/G: top/bottom │ q: quit")

	viewHeight := m.height - 5
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
		case panelMemory:
			leftPanel = m.renderMemoryPanel(leftWidth, viewHeight)
		default:
			leftPanel = m.renderChatPanel(leftWidth, viewHeight)
		}
		visible = lipgloss.JoinHorizontal(lipgloss.Top, leftPanel, " ", visible)
	}

	return header + "\n" + visible + "\n" + footer
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
