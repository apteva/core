package main

import (
	"path/filepath"
	"testing"
)

func TestAuditCodebasePathEscape(t *testing.T) {
	codebaseDir = filepath.Join(t.TempDir(), "project")
	got, err := safePath("../project-private/secret.txt")
	if err == nil {
		t.Fatalf("sibling outside root accepted: %s", got)
	}
}
