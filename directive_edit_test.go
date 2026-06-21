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
