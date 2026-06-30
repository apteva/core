package core

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

type directiveEdit struct {
	Mode      string `json:"mode,omitempty"`
	EditMode  string `json:"edit_mode,omitempty"`
	Directive string `json:"directive,omitempty"`
	Section   string `json:"section,omitempty"`
	Match     string `json:"match,omitempty"`
	Content   string `json:"content,omitempty"`
}

var markdownHeadingRe = regexp.MustCompile(`^(#{1,6})\s+(.+?)\s*#*\s*$`)

func hasDirectiveEditArgs(args map[string]string) bool {
	if args == nil {
		return false
	}
	for _, key := range []string{"directive", "edit_mode", "edits", "section", "content", "match"} {
		if strings.TrimSpace(args[key]) != "" {
			return true
		}
	}
	return false
}

func applyDirectiveEdit(current string, args map[string]string) (string, string, error) {
	if strings.TrimSpace(args["edits"]) != "" {
		var edits []directiveEdit
		if err := json.Unmarshal([]byte(args["edits"]), &edits); err != nil {
			return "", "", fmt.Errorf("invalid edits JSON: %w", err)
		}
		if len(edits) == 0 {
			return "", "", fmt.Errorf("edits must contain at least one edit")
		}
		updated := current
		for i, edit := range edits {
			next, err := applySingleDirectiveEdit(updated, edit)
			if err != nil {
				return "", "", fmt.Errorf("edit %d: %w", i+1, err)
			}
			updated = next
		}
		return updated, fmt.Sprintf("%d directive edits applied", len(edits)), nil
	}

	edit := directiveEdit{
		Mode:      args["edit_mode"],
		Directive: args["directive"],
		Section:   args["section"],
		Match:     args["match"],
		Content:   args["content"],
	}
	updated, err := applySingleDirectiveEdit(current, edit)
	if err != nil {
		return "", "", err
	}
	return updated, directiveEditSummary(edit), nil
}

func applySingleDirectiveEdit(current string, edit directiveEdit) (string, error) {
	mode := directiveEditMode(edit)
	switch mode {
	case "replace":
		replacement := edit.Directive
		if replacement == "" {
			replacement = edit.Content
		}
		if strings.TrimSpace(replacement) == "" {
			return "", fmt.Errorf("replace requires directive or content")
		}
		if directiveHasMarkdownSections(current) {
			return "", fmt.Errorf("full directive replacement is disabled for structured Markdown directives; use section edit modes")
		}
		return replacement, nil
	case "section_append":
		if strings.TrimSpace(edit.Section) == "" || strings.TrimSpace(edit.Content) == "" {
			return "", fmt.Errorf("section_append requires section and content")
		}
		return markdownSectionAppend(current, edit.Section, edit.Content), nil
	case "section_replace":
		if strings.TrimSpace(edit.Section) == "" {
			return "", fmt.Errorf("section_replace requires section")
		}
		return markdownSectionReplace(current, edit.Section, edit.Content), nil
	case "section_rename":
		if strings.TrimSpace(edit.Section) == "" || strings.TrimSpace(edit.Content) == "" {
			return "", fmt.Errorf("section_rename requires section and content")
		}
		return markdownSectionRename(current, edit.Section, edit.Content)
	case "section_replace_line":
		if strings.TrimSpace(edit.Section) == "" || strings.TrimSpace(edit.Match) == "" || strings.TrimSpace(edit.Content) == "" {
			return "", fmt.Errorf("section_replace_line requires section, match, and content")
		}
		return markdownSectionReplaceLine(current, edit.Section, edit.Match, edit.Content)
	case "section_remove_line":
		if strings.TrimSpace(edit.Section) == "" || strings.TrimSpace(edit.Match) == "" {
			return "", fmt.Errorf("section_remove_line requires section and match")
		}
		return markdownSectionRemoveLine(current, edit.Section, edit.Match)
	default:
		return "", fmt.Errorf("unknown edit_mode %q", firstNonEmptyDirectiveEdit(edit.EditMode, edit.Mode))
	}
}

func normalizeDirectiveEditMode(mode string) string {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "", "replace", "full_replace":
		return "replace"
	case "append", "section_append":
		return "section_append"
	case "replace_section", "section_replace":
		return "section_replace"
	case "rename_section", "section_rename":
		return "section_rename"
	case "replace_line", "section_replace_line":
		return "section_replace_line"
	case "remove_line", "section_remove_line":
		return "section_remove_line"
	default:
		return strings.ToLower(strings.TrimSpace(mode))
	}
}

func directiveEditMode(edit directiveEdit) string {
	raw := firstNonEmptyDirectiveEdit(edit.EditMode, edit.Mode)
	if strings.TrimSpace(raw) != "" {
		return normalizeDirectiveEditMode(raw)
	}
	if strings.TrimSpace(edit.Section) != "" {
		return "section_append"
	}
	return "replace"
}

func directiveEditSummary(edit directiveEdit) string {
	mode := directiveEditMode(edit)
	if mode == "replace" {
		return "directive replaced"
	}
	if strings.TrimSpace(edit.Section) == "" {
		return mode
	}
	return fmt.Sprintf("%s %q", mode, edit.Section)
}

func markdownSectionAppend(current, section, content string) string {
	lines := directiveLines(current)
	start, _, end, ok := findMarkdownSection(lines, section)
	contentLines := directiveLines(strings.Trim(content, "\n"))
	if !ok {
		return appendMarkdownSection(lines, section, contentLines)
	}
	insert := end
	add := append([]string{}, contentLines...)
	if insert > start+1 && lines[insert-1] != "" && len(add) > 0 && add[0] == "" {
		add = add[1:]
	}
	return strings.Join(insertLines(lines, insert, add), "\n")
}

func markdownSectionReplace(current, section, content string) string {
	lines := directiveLines(current)
	_, bodyStart, bodyEnd, ok := findMarkdownSection(lines, section)
	contentLines := directiveLines(strings.Trim(content, "\n"))
	if !ok {
		return appendMarkdownSection(lines, section, contentLines)
	}
	out := append([]string{}, lines[:bodyStart]...)
	out = append(out, contentLines...)
	out = append(out, lines[bodyEnd:]...)
	return strings.Join(out, "\n")
}

func markdownSectionRename(current, section, newSection string) (string, error) {
	lines := directiveLines(current)
	header, _, _, ok := findMarkdownSection(lines, section)
	if !ok {
		return "", fmt.Errorf("section %q not found", section)
	}
	level, ok := markdownHeadingLevel(lines[header])
	if !ok {
		return "", fmt.Errorf("section %q heading is invalid", section)
	}
	lines[header] = strings.Repeat("#", level) + " " + strings.TrimSpace(newSection)
	return strings.Join(lines, "\n"), nil
}

func markdownSectionReplaceLine(current, section, match, content string) (string, error) {
	lines := directiveLines(current)
	_, bodyStart, bodyEnd, ok := findMarkdownSection(lines, section)
	if !ok {
		return "", fmt.Errorf("section %q not found", section)
	}
	for i := bodyStart; i < bodyEnd; i++ {
		if strings.Contains(lines[i], match) {
			lines[i] = strings.Trim(content, "\n")
			return strings.Join(lines, "\n"), nil
		}
	}
	return "", fmt.Errorf("no line matching %q in section %q", match, section)
}

func markdownSectionRemoveLine(current, section, match string) (string, error) {
	lines := directiveLines(current)
	_, bodyStart, bodyEnd, ok := findMarkdownSection(lines, section)
	if !ok {
		return "", fmt.Errorf("section %q not found", section)
	}
	for i := bodyStart; i < bodyEnd; i++ {
		if strings.Contains(lines[i], match) {
			out := append([]string{}, lines[:i]...)
			out = append(out, lines[i+1:]...)
			return strings.Join(out, "\n"), nil
		}
	}
	return "", fmt.Errorf("no line matching %q in section %q", match, section)
}

func directiveLines(s string) []string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")
	s = strings.Trim(s, "\n")
	if s == "" {
		return nil
	}
	return strings.Split(s, "\n")
}

func findMarkdownSection(lines []string, section string) (header, bodyStart, bodyEnd int, ok bool) {
	want := normalizeSectionName(section)
	for i, line := range lines {
		name, isHeading := markdownHeadingName(line)
		if !isHeading || normalizeSectionName(name) != want {
			continue
		}
		level, _ := markdownHeadingLevel(line)
		end := len(lines)
		for j := i + 1; j < len(lines); j++ {
			nextLevel, isNextHeading := markdownHeadingLevel(lines[j])
			if isNextHeading && nextLevel <= level {
				end = j
				break
			}
		}
		return i, i + 1, end, true
	}
	return 0, 0, 0, false
}

func markdownHeadingName(line string) (string, bool) {
	m := markdownHeadingRe.FindStringSubmatch(strings.TrimSpace(line))
	if len(m) != 3 {
		return "", false
	}
	return strings.TrimSpace(m[2]), true
}

func markdownHeadingLevel(line string) (int, bool) {
	m := markdownHeadingRe.FindStringSubmatch(strings.TrimSpace(line))
	if len(m) != 3 {
		return 0, false
	}
	return len(m[1]), true
}

func directiveHasMarkdownSections(s string) bool {
	for _, line := range directiveLines(s) {
		if _, ok := markdownHeadingName(line); ok {
			return true
		}
	}
	return false
}

func normalizeSectionName(s string) string {
	return strings.ToLower(strings.TrimSpace(s))
}

func appendMarkdownSection(lines []string, section string, contentLines []string) string {
	out := append([]string{}, lines...)
	if len(out) > 0 && out[len(out)-1] != "" {
		out = append(out, "")
	}
	out = append(out, "# "+strings.TrimSpace(section))
	out = append(out, contentLines...)
	return strings.Join(out, "\n")
}

func insertLines(lines []string, idx int, add []string) []string {
	out := make([]string, 0, len(lines)+len(add))
	out = append(out, lines[:idx]...)
	out = append(out, add...)
	out = append(out, lines[idx:]...)
	return out
}

func firstNonEmptyDirectiveEdit(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}
