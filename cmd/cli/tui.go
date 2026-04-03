package main

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// Messages
type respondMsg string
type askMsg string
type statusMsg statusUpdate
type connectedMsg struct{}
type tickMsg time.Time

type tuiModel struct {
	th          theme
	mcp         *mcpServer
	client      *coreClient
	input       textinput.Model
	lines       []styledLine // scrollback buffer
	scrollOff   int
	width       int
	height      int
	connected   bool
	waiting     bool // waiting for core to cli_respond
	asking      bool // core asked a question via cli_ask
	statusLine  string
	statusLevel string
	startTime   time.Time
}

type styledLine struct {
	text  string
	style string // "input", "output", "dim", "warn", "alert", "system"
}

func newTUI(th theme, mcp *mcpServer, client *coreClient) tuiModel {
	ti := textinput.New()
	ti.Placeholder = ""
	ti.CharLimit = 1000
	ti.Prompt = ""
	ti.Focus()
	ti.TextStyle = lipgloss.NewStyle().Foreground(th.Primary)
	ti.Cursor.Style = lipgloss.NewStyle().Foreground(th.Accent)

	return tuiModel{
		th:        th,
		mcp:       mcp,
		client:    client,
		input:     ti,
		startTime: time.Now(),
	}
}

func (m tuiModel) Init() tea.Cmd {
	return tea.Batch(
		textinput.Blink,
		listenRespond(m.mcp),
		listenAsk(m.mcp),
		listenStatus(m.mcp),
		tickEvery(),
	)
}

func listenRespond(mcp *mcpServer) tea.Cmd {
	return func() tea.Msg {
		return respondMsg(<-mcp.respond)
	}
}

func listenAsk(mcp *mcpServer) tea.Cmd {
	return func() tea.Msg {
		return askMsg(<-mcp.askCh)
	}
}

func listenStatus(mcp *mcpServer) tea.Cmd {
	return func() tea.Msg {
		return statusMsg(<-mcp.statusCh)
	}
}

func tickEvery() tea.Cmd {
	return tea.Tick(time.Second, func(t time.Time) tea.Msg {
		return tickMsg(t)
	})
}

func (m tuiModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c":
			return m, tea.Quit
		case "enter":
			return m.handleInput()
		case "pgup":
			if m.scrollOff < len(m.lines)-1 {
				m.scrollOff += 5
			}
			return m, nil
		case "pgdown":
			m.scrollOff -= 5
			if m.scrollOff < 0 {
				m.scrollOff = 0
			}
			return m, nil
		}

	case respondMsg:
		m.waiting = false
		m.addLine(string(msg), "output")
		m.scrollOff = 0
		cmds = append(cmds, listenRespond(m.mcp))

	case askMsg:
		m.asking = true
		m.waiting = false
		m.addLine(string(msg), "output")
		m.scrollOff = 0
		cmds = append(cmds, listenAsk(m.mcp))

	case statusMsg:
		m.statusLine = msg.Line
		m.statusLevel = msg.Level
		cmds = append(cmds, listenStatus(m.mcp))

	case connectedMsg:
		m.connected = true

	case tickMsg:
		cmds = append(cmds, tickEvery())

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.input.Width = m.chatWidth() - 6
	}

	var inputCmd tea.Cmd
	m.input, inputCmd = m.input.Update(msg)
	cmds = append(cmds, inputCmd)

	return m, tea.Batch(cmds...)
}

func (m *tuiModel) chatWidth() int {
	return m.width * 2 / 3
}

func (m *tuiModel) sideWidth() int {
	return m.width - m.chatWidth() - 1 // -1 for the vertical border
}

func (m *tuiModel) handleInput() (tuiModel, tea.Cmd) {
	text := strings.TrimSpace(m.input.Value())
	if text == "" {
		return *m, nil
	}
	m.input.SetValue("")

	// If answering a cli_ask question
	if m.asking {
		m.asking = false
		m.addLine("> "+text, "input")
		m.mcp.askReply <- text
		return *m, nil
	}

	// Local commands
	if strings.HasPrefix(text, "/") {
		return m.handleCommand(text)
	}

	// Send to core
	m.addLine("> "+text, "input")
	m.waiting = true
	m.scrollOff = 0
	go m.client.sendEvent("[cli] "+text, "main")

	return *m, nil
}

func (m *tuiModel) handleCommand(text string) (tuiModel, tea.Cmd) {
	parts := strings.Fields(text)
	cmd := parts[0]

	switch cmd {
	case "/quit", "/exit":
		return *m, tea.Quit

	case "/clear":
		m.lines = nil
		m.scrollOff = 0

	case "/status":
		m.addLine("REQUESTING STATUS...", "dim")
		go func() {
			st, err := m.client.status()
			if err != nil {
				m.mcp.respond <- fmt.Sprintf("ERROR: %v", err)
				return
			}
			uptime, _ := st["uptime_seconds"].(float64)
			iter, _ := st["iteration"].(float64)
			rate, _ := st["rate"].(string)
			model, _ := st["model"].(string)
			threads, _ := st["threads"].(float64)
			memories, _ := st["memories"].(float64)
			mode, _ := st["mode"].(string)
			paused, _ := st["paused"].(bool)

			status := "RUNNING"
			if paused {
				status = "PAUSED"
			}

			m.mcp.respond <- fmt.Sprintf(
				"CORE STATUS: %s\nUPTIME: %s | ITERATION: %.0f | RATE: %s\nMODEL: %s | MODE: %s\nTHREADS: %.0f | MEMORY: %.0f entries",
				status, formatDuration(time.Duration(uptime)*time.Second), iter, rate, model, mode, threads, memories,
			)
		}()

	case "/threads":
		m.addLine("REQUESTING THREADS...", "dim")
		go func() {
			threads, err := m.client.threads()
			if err != nil {
				m.mcp.respond <- fmt.Sprintf("ERROR: %v", err)
				return
			}
			var sb strings.Builder
			sb.WriteString(fmt.Sprintf("THREADS: %d active\n", len(threads)))
			for _, t := range threads {
				id, _ := t["id"].(string)
				rate, _ := t["rate"].(string)
				model, _ := t["model"].(string)
				age, _ := t["age"].(string)
				iter, _ := t["iteration"].(float64)
				sb.WriteString(fmt.Sprintf("  %-12s  iter=%.0f  rate=%s  model=%s  age=%s\n", id, iter, rate, model, age))
			}
			m.mcp.respond <- sb.String()
		}()

	case "/pause":
		go func() {
			paused, err := m.client.pause()
			if err != nil {
				m.mcp.respond <- fmt.Sprintf("ERROR: %v", err)
				return
			}
			if paused {
				m.mcp.respond <- "CORE PAUSED"
			} else {
				m.mcp.respond <- "CORE RESUMED"
			}
		}()

	case "/approve":
		go func() {
			if err := m.client.approve(true); err != nil {
				m.mcp.respond <- fmt.Sprintf("ERROR: %v", err)
			} else {
				m.mcp.respond <- "APPROVED"
			}
		}()

	case "/reject":
		go func() {
			if err := m.client.approve(false); err != nil {
				m.mcp.respond <- fmt.Sprintf("ERROR: %v", err)
			} else {
				m.mcp.respond <- "REJECTED"
			}
		}()

	case "/config":
		m.addLine("REQUESTING CONFIG...", "dim")
		go func() {
			cfg, err := m.client.getConfig()
			if err != nil {
				m.mcp.respond <- fmt.Sprintf("ERROR: %v", err)
				return
			}
			mode, _ := cfg["mode"].(string)
			directive, _ := cfg["directive"].(string)
			if len(directive) > 80 {
				directive = directive[:80] + "..."
			}
			var sb strings.Builder
			sb.WriteString(fmt.Sprintf("MODE: %s\n", mode))
			sb.WriteString(fmt.Sprintf("DIRECTIVE: %s\n", directive))
			if mcps, ok := cfg["mcp_servers"].([]any); ok && len(mcps) > 0 {
				sb.WriteString(fmt.Sprintf("MCP SERVERS: %d connected\n", len(mcps)))
				for _, raw := range mcps {
					if m, ok := raw.(map[string]any); ok {
						name, _ := m["name"].(string)
						sb.WriteString(fmt.Sprintf("  - %s\n", name))
					}
				}
			}
			m.mcp.respond <- sb.String()
		}()

	default:
		m.addLine(fmt.Sprintf("UNKNOWN COMMAND: %s", cmd), "warn")
		m.addLine("COMMANDS: /status /threads /pause /approve /reject /config /clear /quit", "dim")
	}

	return *m, nil
}

func (m *tuiModel) addLine(text string, style string) {
	for _, line := range strings.Split(text, "\n") {
		m.lines = append(m.lines, styledLine{text: line, style: style})
	}
}

// wrapText wraps a string to fit within maxWidth, breaking on spaces.
func wrapText(s string, maxWidth int) []string {
	if maxWidth <= 0 {
		return []string{s}
	}
	var result []string
	for _, line := range strings.Split(s, "\n") {
		if lipgloss.Width(line) <= maxWidth {
			result = append(result, line)
			continue
		}
		words := strings.Fields(line)
		if len(words) == 0 {
			result = append(result, "")
			continue
		}
		cur := words[0]
		for _, w := range words[1:] {
			test := cur + " " + w
			if lipgloss.Width(test) > maxWidth {
				result = append(result, cur)
				cur = w
			} else {
				cur = test
			}
		}
		result = append(result, cur)
	}
	return result
}

func (m tuiModel) View() string {
	if m.width == 0 || m.height == 0 {
		return ""
	}

	primary := lipgloss.NewStyle().Foreground(m.th.Primary)
	dim := lipgloss.NewStyle().Foreground(m.th.Dim)
	accent := lipgloss.NewStyle().Foreground(m.th.Accent)
	warn := lipgloss.NewStyle().Foreground(m.th.Warn)
	alert := lipgloss.NewStyle().Foreground(m.th.Alert)

	chatW := m.chatWidth()
	sideW := m.sideWidth()
	innerChat := chatW - 4 // 2 padding each side

	// Layout: header(1) + separator(1) + content + separator(1) + input(1) = 4 chrome lines
	contentHeight := m.height - 4
	if contentHeight < 1 {
		contentHeight = 1
	}

	// ── Header ──
	connStatus := dim.Render("◉ DISCONNECTED")
	if m.connected {
		connStatus = accent.Render("◉ CORE LIVE")
	}
	title := primary.Bold(true).Render("APTEVA")
	headerPad := m.width - lipgloss.Width(title) - lipgloss.Width(connStatus)
	if headerPad < 1 {
		headerPad = 1
	}
	header := title + strings.Repeat(" ", headerPad) + connStatus
	sep := dim.Render(strings.Repeat("─", m.width))

	// ── Chat panel (left) ──
	// Wrap and collect visible lines
	var wrappedLines []styledLine
	for _, sl := range m.lines {
		wrapped := wrapText(sl.text, innerChat)
		for _, w := range wrapped {
			wrappedLines = append(wrappedLines, styledLine{text: w, style: sl.style})
		}
	}

	// Visible region
	chatContentH := contentHeight - 2 // -2 for status line + input separator inside chat
	if chatContentH < 1 {
		chatContentH = 1
	}

	start := len(wrappedLines) - chatContentH - m.scrollOff
	if start < 0 {
		start = 0
	}
	end := start + chatContentH
	if end > len(wrappedLines) {
		end = len(wrappedLines)
	}

	var chatLines []string
	for i := start; i < end; i++ {
		line := wrappedLines[i]
		var styled string
		switch line.style {
		case "input":
			styled = primary.Bold(true).Render(line.text)
		case "output":
			styled = primary.Render(line.text)
		case "dim", "system":
			styled = dim.Render(line.text)
		case "warn":
			styled = warn.Render(line.text)
		case "alert":
			styled = alert.Render(line.text)
		default:
			styled = line.text
		}
		chatLines = append(chatLines, styled)
	}

	// Working indicator
	if m.waiting && len(chatLines) < chatContentH {
		chatLines = append(chatLines, accent.Render("WORKING..."))
	}

	// Pad to fill
	for len(chatLines) < chatContentH {
		chatLines = append(chatLines, "")
	}

	// Status line inside chat
	statusText := m.statusLine
	if statusText == "" {
		statusText = "READY"
	}
	var statusStyled string
	switch m.statusLevel {
	case "warn":
		statusStyled = warn.Render(statusText)
	case "alert":
		statusStyled = alert.Render(statusText)
	default:
		statusStyled = dim.Render(statusText)
	}

	// Input line
	prompt := primary.Bold(true).Render("> ")
	inputLine := prompt + m.input.View()

	// Build chat column
	chatLines = append(chatLines, dim.Render(strings.Repeat("─", innerChat)))
	chatLines = append(chatLines, inputLine)

	chatPanel := lipgloss.NewStyle().
		Width(chatW).
		Padding(0, 2).
		Render(strings.Join(chatLines, "\n"))

	// ── Side panel (right) ──
	sideLines := m.renderSidePanel(sideW-2, contentHeight, dim, primary, accent, warn)
	sidePanel := lipgloss.NewStyle().
		Width(sideW).
		Padding(0, 1).
		Render(strings.Join(sideLines, "\n"))

	// ── Vertical border ──
	var borderLines []string
	for i := 0; i < contentHeight; i++ {
		borderLines = append(borderLines, dim.Render("│"))
	}
	border := strings.Join(borderLines, "\n")

	// ── Compose ──
	body := lipgloss.JoinHorizontal(lipgloss.Top, chatPanel, border, sidePanel)

	// Bottom status bar across full width
	bottomBar := dim.Render(strings.Repeat("─", chatW)) +
		dim.Render("┴") +
		dim.Render(strings.Repeat("─", sideW))
	_ = statusStyled

	return header + "\n" + sep + "\n" + body + "\n" + bottomBar + " " + statusStyled
}

func (m tuiModel) renderSidePanel(w, h int, dim, primary, accent, warn lipgloss.Style) []string {
	var lines []string

	// Title
	lines = append(lines, accent.Bold(true).Render("SYSTEM"))
	lines = append(lines, dim.Render(strings.Repeat("─", w)))
	lines = append(lines, "")

	// Uptime
	uptime := formatDuration(time.Since(m.startTime))
	lines = append(lines, dim.Render("UPTIME")+primary.Render("  "+uptime))
	lines = append(lines, "")

	// Connection
	if m.connected {
		lines = append(lines, dim.Render("CORE")+accent.Render("    CONNECTED"))
	} else {
		lines = append(lines, dim.Render("CORE")+warn.Render("    OFFLINE"))
	}
	lines = append(lines, "")

	// Status
	lines = append(lines, dim.Render("STATUS"))
	if m.waiting {
		lines = append(lines, accent.Render("  PROCESSING..."))
	} else if m.asking {
		lines = append(lines, warn.Render("  AWAITING INPUT"))
	} else {
		lines = append(lines, primary.Render("  IDLE"))
	}
	lines = append(lines, "")

	// MCP
	lines = append(lines, dim.Render("MCP")+primary.Render("     cli"))
	lines = append(lines, "")

	// Separator
	lines = append(lines, dim.Render(strings.Repeat("─", w)))
	lines = append(lines, "")

	// Commands help
	lines = append(lines, accent.Bold(true).Render("COMMANDS"))
	lines = append(lines, dim.Render(strings.Repeat("─", w)))
	lines = append(lines, "")
	cmds := []struct{ cmd, desc string }{
		{"/status", "core status"},
		{"/threads", "list threads"},
		{"/pause", "toggle pause"},
		{"/approve", "approve tool"},
		{"/reject", "reject tool"},
		{"/config", "show config"},
		{"/clear", "clear screen"},
		{"/quit", "disconnect"},
	}
	for _, c := range cmds {
		label := primary.Render(fmt.Sprintf("%-10s", c.cmd))
		desc := dim.Render(c.desc)
		lines = append(lines, label+desc)
	}

	// Pad to fill height
	for len(lines) < h {
		lines = append(lines, "")
	}

	return lines[:h]
}

func formatDuration(d time.Duration) string {
	h := int(d.Hours())
	m := int(d.Minutes()) % 60
	s := int(d.Seconds()) % 60
	if h > 0 {
		return fmt.Sprintf("%dh%02dm", h, m)
	}
	if m > 0 {
		return fmt.Sprintf("%dm%02ds", m, s)
	}
	return fmt.Sprintf("%ds", s)
}
