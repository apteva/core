package core

import (
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
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
var markdownRuleRe = regexp.MustCompile(`^(?:[-*+]\s+)?([A-Za-z][A-Za-z0-9 _-]{0,63}):\s*(.+)$`)

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
		var warnings []string
		for i, edit := range edits {
			next, editWarnings, err := applySingleDirectiveEdit(updated, edit)
			if err != nil {
				return "", "", fmt.Errorf("edit %d: %w", i+1, err)
			}
			updated = next
			warnings = append(warnings, editWarnings...)
		}
		warnings = append(warnings, introducedDirectiveWarnings(current, updated)...)
		return updated, directiveEditSummaryWithWarnings(fmt.Sprintf("%d directive edits applied", len(edits)), warnings), nil
	}

	edit := directiveEdit{
		Mode:      args["edit_mode"],
		Directive: args["directive"],
		Section:   args["section"],
		Match:     args["match"],
		Content:   args["content"],
	}
	updated, warnings, err := applySingleDirectiveEdit(current, edit)
	if err != nil {
		return "", "", err
	}
	warnings = append(warnings, introducedDirectiveWarnings(current, updated)...)
	return updated, directiveEditSummaryWithWarnings(directiveEditSummary(edit), warnings), nil
}

func applySingleDirectiveEdit(current string, edit directiveEdit) (string, []string, error) {
	mode := directiveEditMode(edit)
	switch mode {
	case "replace":
		replacement := edit.Directive
		if replacement == "" {
			replacement = edit.Content
		}
		if strings.TrimSpace(replacement) == "" {
			return "", nil, fmt.Errorf("replace requires directive or content")
		}
		if directiveHasMarkdownSections(current) {
			return "", nil, fmt.Errorf("full directive replacement is disabled for structured Markdown directives; use section_append to add a rule or missing section, section_replace to rewrite one section, or edits for multiple sections; do not pass the complete directive")
		}
		return replacement, nil, nil
	case "section_append":
		if strings.TrimSpace(edit.Section) == "" || strings.TrimSpace(edit.Content) == "" {
			return "", nil, fmt.Errorf("section_append requires section and content")
		}
		if err := validateMarkdownSectionContent(edit.Section, edit.Content); err != nil {
			return "", nil, err
		}
		content, stripped := normalizeMarkdownSectionContent(edit.Section, edit.Content)
		return markdownSectionAppend(current, edit.Section, content), redundantHeadingWarnings(edit.Section, stripped), nil
	case "section_replace":
		if strings.TrimSpace(edit.Section) == "" {
			return "", nil, fmt.Errorf("section_replace requires section")
		}
		if err := validateMarkdownSectionContent(edit.Section, edit.Content); err != nil {
			return "", nil, err
		}
		content, stripped := normalizeMarkdownSectionContent(edit.Section, edit.Content)
		return markdownSectionReplace(current, edit.Section, content), redundantHeadingWarnings(edit.Section, stripped), nil
	case "section_rename":
		if strings.TrimSpace(edit.Section) == "" || strings.TrimSpace(edit.Content) == "" {
			return "", nil, fmt.Errorf("section_rename requires section and content")
		}
		updated, err := markdownSectionRename(current, edit.Section, edit.Content)
		return updated, nil, err
	case "section_replace_line":
		if strings.TrimSpace(edit.Section) == "" || strings.TrimSpace(edit.Match) == "" || strings.TrimSpace(edit.Content) == "" {
			return "", nil, fmt.Errorf("section_replace_line requires section, match, and content")
		}
		updated, err := markdownSectionReplaceLine(current, edit.Section, edit.Match, edit.Content)
		return updated, nil, err
	case "section_remove_line":
		if strings.TrimSpace(edit.Section) == "" || strings.TrimSpace(edit.Match) == "" {
			return "", nil, fmt.Errorf("section_remove_line requires section and match")
		}
		updated, err := markdownSectionRemoveLine(current, edit.Section, edit.Match)
		return updated, nil, err
	default:
		return "", nil, fmt.Errorf("unknown edit_mode %q", firstNonEmptyDirectiveEdit(edit.EditMode, edit.Mode))
	}
}

func directiveEditCorrectionResult(err error) string {
	return fmt.Sprintf("error: %v. Correct the evolve arguments and retry once now with the smallest valid section edit; do not abandon the durable update or repeat the same invalid call", err)
}

func directiveEditFinalFailureResult(err error) string {
	return fmt.Sprintf("error: %v. The correction was also rejected; do not call evolve again for this instruction. Report the failure to the requester before pacing", err)
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

func directiveEditSummaryWithWarnings(summary string, warnings []string) string {
	warnings = uniqueDirectiveWarnings(warnings)
	if len(warnings) == 0 {
		return summary
	}
	return summary + "; warning: " + strings.Join(warnings, "; ")
}

func directiveEditToolResult(result, summary string) string {
	const marker = "; warning: "
	if i := strings.Index(summary, marker); i >= 0 {
		return result + summary[i:]
	}
	return result
}

func redundantHeadingWarnings(section string, count int) []string {
	if count == 0 {
		return nil
	}
	return []string{fmt.Sprintf("removed %d redundant %q heading(s) from section content", count, strings.TrimSpace(section))}
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
	sections := findMarkdownSections(lines, section)
	contentLines := directiveLines(strings.Trim(content, "\n"))
	if len(sections) == 0 {
		return appendMarkdownSection(lines, section, contentLines)
	}

	canonical := sections[0]
	out := append([]string{}, lines[:canonical.header+1]...)
	out = append(out, contentLines...)
	cursor := canonical.end
	for _, duplicate := range sections[1:] {
		// A nested same-name section inside the canonical section was
		// already removed with the canonical body.
		if duplicate.header < cursor {
			continue
		}
		out = append(out, lines[cursor:duplicate.header]...)
		cursor = duplicate.end
	}
	out = append(out, lines[cursor:]...)
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
	header, bodyStart, bodyEnd, ok := findMarkdownSection(lines, section)
	if !ok {
		return "", fmt.Errorf("section %q not found", section)
	}
	for i := bodyStart; i < bodyEnd; i++ {
		if strings.Contains(lines[i], match) {
			hasRemainingContent := false
			for j := bodyStart; j < bodyEnd; j++ {
				if j != i && strings.TrimSpace(lines[j]) != "" {
					hasRemainingContent = true
					break
				}
			}
			if !hasRemainingContent {
				out := append([]string{}, lines[:header]...)
				out = append(out, lines[bodyEnd:]...)
				return strings.Trim(strings.Join(out, "\n"), "\n"), nil
			}
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
	sections := findMarkdownSections(lines, section)
	if len(sections) == 0 {
		return 0, 0, 0, false
	}
	first := sections[0]
	return first.header, first.header + 1, first.end, true
}

type markdownSection struct {
	header int
	end    int
}

func findMarkdownSections(lines []string, section string) []markdownSection {
	want := normalizeSectionName(section)
	var sections []markdownSection
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
		sections = append(sections, markdownSection{header: i, end: end})
	}
	return sections
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

func validateMarkdownSectionContent(section, content string) error {
	want := normalizeSectionName(section)
	for _, line := range directiveLines(content) {
		name, isHeading := markdownHeadingName(line)
		if !isHeading || normalizeSectionName(name) == want {
			continue
		}
		return fmt.Errorf("section %q content must not contain the unrelated Markdown heading %q; pass only this section's body, or use edits for multiple sections", strings.TrimSpace(section), strings.TrimSpace(name))
	}
	return nil
}

func normalizeMarkdownSectionContent(section, content string) (string, int) {
	lines := directiveLines(strings.Trim(content, "\n"))
	want := normalizeSectionName(section)
	out := make([]string, 0, len(lines))
	stripped := 0
	for _, line := range lines {
		name, isHeading := markdownHeadingName(line)
		if isHeading && normalizeSectionName(name) == want {
			stripped++
			continue
		}
		out = append(out, line)
	}
	for len(out) > 0 && strings.TrimSpace(out[0]) == "" {
		out = out[1:]
	}
	for len(out) > 0 && strings.TrimSpace(out[len(out)-1]) == "" {
		out = out[:len(out)-1]
	}
	return strings.Join(out, "\n"), stripped
}

func introducedDirectiveWarnings(before, after string) []string {
	beforeIssues := directiveStructuralIssues(before)
	afterIssues := directiveStructuralIssues(after)
	warnings := make([]string, 0, len(afterIssues))
	for key, warning := range afterIssues {
		if _, existed := beforeIssues[key]; !existed {
			warnings = append(warnings, warning)
		}
	}
	sort.Strings(warnings)
	return warnings
}

func directiveStructuralIssues(directive string) map[string]string {
	issues := make(map[string]string)
	headingCounts := make(map[string]int)
	headingLabels := make(map[string]string)
	ruleValues := make(map[string]map[string]struct{})
	ruleLabels := make(map[string][2]string)
	section := ""

	for _, line := range directiveLines(directive) {
		if name, ok := markdownHeadingName(line); ok {
			section = normalizeSectionName(name)
			headingCounts[section]++
			if headingLabels[section] == "" {
				headingLabels[section] = strings.TrimSpace(name)
			}
			continue
		}
		if section == "" {
			continue
		}
		match := markdownRuleRe.FindStringSubmatch(strings.TrimSpace(line))
		if len(match) != 3 {
			continue
		}
		key := normalizeDirectiveRuleKey(match[1])
		value := strings.ToLower(strings.Join(strings.Fields(match[2]), " "))
		issueKey := "rule:" + section + ":" + key
		if ruleValues[issueKey] == nil {
			ruleValues[issueKey] = make(map[string]struct{})
			ruleLabels[issueKey] = [2]string{strings.TrimSpace(match[1]), headingLabels[section]}
		}
		ruleValues[issueKey][value] = struct{}{}
	}

	for name, count := range headingCounts {
		if count > 1 {
			key := "heading:" + name
			issues[key] = fmt.Sprintf("directive contains %d %q headings", count, headingLabels[name])
		}
	}
	for key, values := range ruleValues {
		if len(values) > 1 {
			labels := ruleLabels[key]
			issues[key] = fmt.Sprintf("conflicting %q rules in section %q", labels[0], labels[1])
		}
	}
	return issues
}

func normalizeDirectiveRuleKey(key string) string {
	return strings.ToLower(strings.Join(strings.Fields(key), " "))
}

func uniqueDirectiveWarnings(warnings []string) []string {
	seen := make(map[string]struct{}, len(warnings))
	result := make([]string, 0, len(warnings))
	for _, warning := range warnings {
		if warning == "" {
			continue
		}
		if _, ok := seen[warning]; ok {
			continue
		}
		seen[warning] = struct{}{}
		result = append(result, warning)
	}
	return result
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
