package core

import (
	"strings"
	"testing"
)

func TestApplyDirectiveEditFullReplace(t *testing.T) {
	got, summary, err := applyDirectiveEdit("old", map[string]string{"directive": "new"})
	if err != nil {
		t.Fatalf("applyDirectiveEdit error: %v", err)
	}
	if got != "new" {
		t.Fatalf("directive = %q, want new", got)
	}
	if summary != "directive replaced" {
		t.Fatalf("summary = %q, want directive replaced", summary)
	}
}

func TestApplyDirectiveEditRejectsFullReplaceForMarkdown(t *testing.T) {
	_, _, err := applyDirectiveEdit("# Role\nOld role\n# Goals\n- Ship", map[string]string{"directive": "# Role\nNew role"})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "full directive replacement is disabled for structured Markdown directives") {
		t.Fatalf("error = %q", err)
	}
}

func TestApplyDirectiveEditRejectsExplicitReplaceForMarkdown(t *testing.T) {
	_, _, err := applyDirectiveEdit("# Role\nOld role\n# Goals\n- Ship", map[string]string{
		"edit_mode": "replace",
		"content":   "# Role\nNew role",
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "full directive replacement is disabled for structured Markdown directives") {
		t.Fatalf("error = %q", err)
	}
}

func TestApplyDirectiveEditSectionAppendCreatesSection(t *testing.T) {
	got, _, err := applyDirectiveEdit("# Goals\n- Ship", map[string]string{
		"edit_mode": "section_append",
		"section":   "Schedule",
		"content":   "- daily_check: 07:30 Europe/Madrid",
	})
	if err != nil {
		t.Fatalf("applyDirectiveEdit error: %v", err)
	}
	want := "# Goals\n- Ship\n\n# Schedule\n- daily_check: 07:30 Europe/Madrid"
	if got != want {
		t.Fatalf("directive:\n%s\nwant:\n%s", got, want)
	}
}

func TestApplyDirectiveEditSectionPayloadDefaultsToAppend(t *testing.T) {
	got, summary, err := applyDirectiveEdit("", map[string]string{
		"section": "Goals",
		"content": "- Ship",
	})
	if err != nil {
		t.Fatalf("applyDirectiveEdit error: %v", err)
	}
	want := "# Goals\n- Ship"
	if got != want {
		t.Fatalf("directive:\n%s\nwant:\n%s", got, want)
	}
	if summary != `section_append "Goals"` {
		t.Fatalf("summary = %q", summary)
	}
}

func TestApplyDirectiveEditIdenticalSectionReplaceIsStable(t *testing.T) {
	current := "# Goals\n- Ship\n\n# Schedule\n- cadence: weekly"
	got, _, err := applyDirectiveEdit(current, map[string]string{
		"edit_mode": "section_replace",
		"section":   "Schedule",
		"content":   "- cadence: weekly",
	})
	if err != nil {
		t.Fatalf("applyDirectiveEdit: %v", err)
	}
	if got != current {
		t.Fatalf("identical edit changed directive:\ngot:\n%s\nwant:\n%s", got, current)
	}
}

func TestApplyDirectiveEditRemoveFinalLineRemovesEmptySection(t *testing.T) {
	current := "# Goals\n- Ship\n\n# Affiliate Reporting\n- cadence: weekly"
	got, _, err := applyDirectiveEdit(current, map[string]string{
		"edit_mode": "section_remove_line",
		"section":   "Affiliate Reporting",
		"match":     "cadence: weekly",
	})
	if err != nil {
		t.Fatalf("applyDirectiveEdit: %v", err)
	}
	if got != "# Goals\n- Ship" {
		t.Fatalf("empty section was not removed: %q", got)
	}
}

func TestApplyDirectiveEditBatchSectionPayloadDefaultsToAppend(t *testing.T) {
	got, summary, err := applyDirectiveEdit("", map[string]string{
		"edits": `[
			{"section":"Role","content":"You are a test agent."},
			{"section":"Goals","content":"- Ship"}
		]`,
	})
	if err != nil {
		t.Fatalf("applyDirectiveEdit error: %v", err)
	}
	want := "# Role\nYou are a test agent.\n\n# Goals\n- Ship"
	if got != want {
		t.Fatalf("directive:\n%s\nwant:\n%s", got, want)
	}
	if summary != "2 directive edits applied" {
		t.Fatalf("summary = %q, want 2 directive edits applied", summary)
	}
}

func TestApplyDirectiveEditSectionAppendExisting(t *testing.T) {
	got, _, err := applyDirectiveEdit("# Schedule\n- old\n# Goals\n- Ship", map[string]string{
		"edit_mode": "append",
		"section":   "Schedule",
		"content":   "- new",
	})
	if err != nil {
		t.Fatalf("applyDirectiveEdit error: %v", err)
	}
	want := "# Schedule\n- old\n- new\n# Goals\n- Ship"
	if got != want {
		t.Fatalf("directive:\n%s\nwant:\n%s", got, want)
	}
}

func TestApplyDirectiveEditSectionAppendStripsMatchingHeading(t *testing.T) {
	got, summary, err := applyDirectiveEdit("# Goals\n- Ship", map[string]string{
		"edit_mode": "section_append",
		"section":   "Goals",
		"content":   "# Goals\n\n- Keep tests green",
	})
	if err != nil {
		t.Fatalf("applyDirectiveEdit error: %v", err)
	}
	want := "# Goals\n- Ship\n- Keep tests green"
	if got != want {
		t.Fatalf("directive:\n%s\nwant:\n%s", got, want)
	}
	if !strings.Contains(summary, `warning: removed 1 redundant "Goals" heading(s)`) {
		t.Fatalf("summary = %q", summary)
	}
}

func TestApplyDirectiveEditSectionReplace(t *testing.T) {
	got, _, err := applyDirectiveEdit("# Schedule\n- old\n# Goals\n- Ship", map[string]string{
		"edit_mode": "section_replace",
		"section":   "Schedule",
		"content":   "- daily_check: 07:30 Europe/Madrid",
	})
	if err != nil {
		t.Fatalf("applyDirectiveEdit error: %v", err)
	}
	want := "# Schedule\n- daily_check: 07:30 Europe/Madrid\n# Goals\n- Ship"
	if got != want {
		t.Fatalf("directive:\n%s\nwant:\n%s", got, want)
	}
}

func TestApplyDirectiveEditSectionReplaceStripsMatchingHeading(t *testing.T) {
	got, summary, err := applyDirectiveEdit("# Schedule\n- old\n# Goals\n- Ship", map[string]string{
		"edit_mode": "section_replace",
		"section":   "Schedule",
		"content":   "## Schedule ##\n- daily_check: 07:30 Europe/Madrid",
	})
	if err != nil {
		t.Fatalf("applyDirectiveEdit error: %v", err)
	}
	want := "# Schedule\n- daily_check: 07:30 Europe/Madrid\n# Goals\n- Ship"
	if got != want {
		t.Fatalf("directive:\n%s\nwant:\n%s", got, want)
	}
	if !strings.Contains(summary, `warning: removed 1 redundant "Schedule" heading(s)`) {
		t.Fatalf("summary = %q", summary)
	}
}

func TestApplyDirectiveEditSectionReplaceConsolidatesDuplicateSections(t *testing.T) {
	before := "# Schedule\n- old first\n# Goals\n- Ship\n# Schedule\n- stale duplicate\n# Notes\n- Keep"
	got, summary, err := applyDirectiveEdit(before, map[string]string{
		"edit_mode": "section_replace",
		"section":   "Schedule",
		"content":   "- daily_check: 07:30 Europe/Madrid",
	})
	if err != nil {
		t.Fatalf("applyDirectiveEdit error: %v", err)
	}
	want := "# Schedule\n- daily_check: 07:30 Europe/Madrid\n# Goals\n- Ship\n# Notes\n- Keep"
	if got != want {
		t.Fatalf("directive:\n%s\nwant:\n%s", got, want)
	}
	if strings.Contains(summary, "warning:") {
		t.Fatalf("resolved duplicate should not produce a warning: %q", summary)
	}
}

func TestApplyDirectiveEditRejectsContentThatIntroducesUnrelatedHeading(t *testing.T) {
	_, _, err := applyDirectiveEdit("# Schedule\n- daily\n# Goals\n- Ship", map[string]string{
		"edit_mode": "section_append",
		"section":   "Schedule",
		"content":   "# Goals\n- Added in the wrong section",
	})
	if err == nil {
		t.Fatal("expected unrelated heading to be rejected")
	}
	if !strings.Contains(err.Error(), `must not contain the unrelated Markdown heading "Goals"`) {
		t.Fatalf("error = %q", err)
	}
}

func TestApplyDirectiveEditWarnsWhenAppendIntroducesConflictingRule(t *testing.T) {
	_, summary, err := applyDirectiveEdit("# Schedule\n- daily_check: 09:00 Europe/Madrid", map[string]string{
		"edit_mode": "section_append",
		"section":   "Schedule",
		"content":   "- daily_check: 07:30 Europe/Madrid",
	})
	if err != nil {
		t.Fatalf("applyDirectiveEdit error: %v", err)
	}
	if !strings.Contains(summary, `warning: conflicting "daily_check" rules in section "Schedule"`) {
		t.Fatalf("summary = %q", summary)
	}
}

func TestApplyDirectiveEditSectionRename(t *testing.T) {
	got, _, err := applyDirectiveEdit("## Daily Schedule\n- daily_check: 09:00 Europe/Madrid\n# Goals\n- Ship", map[string]string{
		"edit_mode": "section_rename",
		"section":   "Daily Schedule",
		"content":   "Schedule",
	})
	if err != nil {
		t.Fatalf("applyDirectiveEdit error: %v", err)
	}
	want := "## Schedule\n- daily_check: 09:00 Europe/Madrid\n# Goals\n- Ship"
	if got != want {
		t.Fatalf("directive:\n%s\nwant:\n%s", got, want)
	}
}

func TestApplyDirectiveEditSectionReplaceLine(t *testing.T) {
	got, _, err := applyDirectiveEdit("# Schedule\n- daily_check: 09:00 Europe/Madrid\n- weekly_check: Friday\n# Goals\n- Ship", map[string]string{
		"edit_mode": "section_replace_line",
		"section":   "Schedule",
		"match":     "daily_check:",
		"content":   "- daily_check: 07:30 Europe/Madrid",
	})
	if err != nil {
		t.Fatalf("applyDirectiveEdit error: %v", err)
	}
	want := "# Schedule\n- daily_check: 07:30 Europe/Madrid\n- weekly_check: Friday\n# Goals\n- Ship"
	if got != want {
		t.Fatalf("directive:\n%s\nwant:\n%s", got, want)
	}
}

func TestApplyDirectiveEditSectionReplaceLineSearchesNestedSubsections(t *testing.T) {
	before := "# Current Offer Focus\nDeskora Reception.\n\n## Deskora Reception Overview\nWebsite: http://deskorareception.com/\nOutreach email: contact@deskoraception.com\n\n# Responsibilities\n- Qualify leads"
	got, _, err := applyDirectiveEdit(before, map[string]string{
		"edit_mode": "section_replace_line",
		"section":   "Current Offer Focus",
		"match":     "Outreach email:",
		"content":   "Outreach email: contact@deskorareception.com",
	})
	if err != nil {
		t.Fatalf("applyDirectiveEdit error: %v", err)
	}
	want := "# Current Offer Focus\nDeskora Reception.\n\n## Deskora Reception Overview\nWebsite: http://deskorareception.com/\nOutreach email: contact@deskorareception.com\n\n# Responsibilities\n- Qualify leads"
	if got != want {
		t.Fatalf("directive:\n%s\nwant:\n%s", got, want)
	}
}

func TestApplyDirectiveEditSectionReplaceLineStopsAtSiblingSection(t *testing.T) {
	_, _, err := applyDirectiveEdit("# Current Offer Focus\nDeskora Reception.\n\n# Responsibilities\nOutreach email: contact@deskoraception.com", map[string]string{
		"edit_mode": "section_replace_line",
		"section":   "Current Offer Focus",
		"match":     "Outreach email:",
		"content":   "Outreach email: contact@deskorareception.com",
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), `no line matching "Outreach email:"`) {
		t.Fatalf("error = %q", err)
	}
}

func TestApplyDirectiveEditSectionRemoveLine(t *testing.T) {
	got, _, err := applyDirectiveEdit("# Schedule\n- daily_check: 09:00 Europe/Madrid\n- weekly_check: Friday", map[string]string{
		"edit_mode": "section_remove_line",
		"section":   "Schedule",
		"match":     "weekly_check:",
	})
	if err != nil {
		t.Fatalf("applyDirectiveEdit error: %v", err)
	}
	want := "# Schedule\n- daily_check: 09:00 Europe/Madrid"
	if got != want {
		t.Fatalf("directive:\n%s\nwant:\n%s", got, want)
	}
}

func TestApplyDirectiveEditBatch(t *testing.T) {
	got, summary, err := applyDirectiveEdit("# Daily Schedule\n- daily_check: 09:00 Europe/Madrid\n# Goals\n- Ship", map[string]string{
		"edits": `[
			{"mode":"section_rename","section":"Daily Schedule","content":"Schedule"},
			{"mode":"section_replace_line","section":"Schedule","match":"daily_check:","content":"- daily_check: 07:30 Europe/Madrid"},
			{"mode":"section_append","section":"Goals","content":"- Report release readiness"}
		]`,
	})
	if err != nil {
		t.Fatalf("applyDirectiveEdit error: %v", err)
	}
	want := "# Schedule\n- daily_check: 07:30 Europe/Madrid\n# Goals\n- Ship\n- Report release readiness"
	if got != want {
		t.Fatalf("directive:\n%s\nwant:\n%s", got, want)
	}
	if summary != "3 directive edits applied" {
		t.Fatalf("summary = %q, want 3 directive edits applied", summary)
	}
}

func TestApplyDirectiveEditSectionRenameMissingSection(t *testing.T) {
	_, _, err := applyDirectiveEdit("# Schedule\n- daily_check: 09:00 Europe/Madrid", map[string]string{
		"edit_mode": "section_rename",
		"section":   "Daily Schedule",
		"content":   "Schedule",
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), `section "Daily Schedule" not found`) {
		t.Fatalf("error = %q", err)
	}
}

func TestApplyDirectiveEditMissingLineMatch(t *testing.T) {
	_, _, err := applyDirectiveEdit("# Schedule\n- daily_check: 09:00 Europe/Madrid", map[string]string{
		"edit_mode": "section_replace_line",
		"section":   "Schedule",
		"match":     "weekly_check:",
		"content":   "- weekly_check: Friday",
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), `no line matching "weekly_check:"`) {
		t.Fatalf("error = %q", err)
	}
}
