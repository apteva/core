package core

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func TestAtomicWriteFileIsPrivateAndReplacesContent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "state.json")
	if err := atomicWriteFile(path, []byte("first"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := atomicWriteFile(path, []byte("second"), 0600); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "second" {
		t.Fatalf("content = %q", data)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := info.Mode().Perm(); got != 0600 {
		t.Fatalf("mode = %o, want 600", got)
	}
}

func TestConfigLoadRejectsMalformedJSON(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte("{"), 0600); err != nil {
		t.Fatal(err)
	}
	cfg := &Config{path: path}
	if err := cfg.load(); err == nil {
		t.Fatal("expected malformed config error")
	}
}

func TestConfigMutationRollsBackWhenPersistenceFails(t *testing.T) {
	base := t.TempDir()
	blocker := filepath.Join(base, "not-a-directory")
	if err := os.WriteFile(blocker, []byte("x"), 0600); err != nil {
		t.Fatal(err)
	}
	cfg := &Config{path: filepath.Join(blocker, "config.json"), Directive: "before"}
	if err := cfg.SetDirective("after"); err == nil {
		t.Fatal("expected persistence failure")
	}
	if got := cfg.GetDirective(); got != "before" {
		t.Fatalf("directive changed after failed write: %q", got)
	}
}

func TestConfigFailedFirstThreadSaveRestoresOmittedEmptySlice(t *testing.T) {
	blocker := filepath.Join(t.TempDir(), "file")
	if err := os.WriteFile(blocker, []byte("x"), 0600); err != nil {
		t.Fatal(err)
	}
	cfg := &Config{Directive: "test", path: filepath.Join(blocker, "config.json")}
	if err := cfg.SaveThread(PersistentThread{ID: "must-rollback", Directive: "work"}); err == nil {
		t.Fatal("SaveThread unexpectedly succeeded")
	}
	if threads := cfg.GetThreads(); len(threads) != 0 {
		t.Fatalf("failed SaveThread left an in-memory record: %#v", threads)
	}
}

func TestConfigConcurrentMutationsAlwaysPersistValidJSON(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.json")
	cfg := &Config{path: path}
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			if err := cfg.SetDirective(fmt.Sprintf("directive-%d", i)); err != nil {
				t.Errorf("SetDirective: %v", err)
			}
		}(i)
	}
	wg.Wait()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var persisted Config
	if err := json.Unmarshal(data, &persisted); err != nil {
		t.Fatalf("invalid persisted JSON: %v", err)
	}
	if persisted.Directive == "" || persisted.Directive != cfg.GetDirective() {
		t.Fatalf("persisted directive %q != memory %q", persisted.Directive, cfg.GetDirective())
	}
}

func TestSessionReasoningAndLargeEntryRoundTrip(t *testing.T) {
	s := NewSession(t.TempDir(), "large")
	content := strings.Repeat("x", 2<<20)
	msg := Message{Role: "assistant", Content: content, Reasoning: "kept reasoning"}
	if err := s.AppendMessage(msg, 1, TokenUsage{}); err != nil {
		t.Fatal(err)
	}
	got, _ := s.LoadTail(1)
	if len(got) != 1 {
		t.Fatalf("loaded %d messages", len(got))
	}
	if got[0].Content != content || got[0].Reasoning != msg.Reasoning {
		t.Fatal("large content or reasoning did not round trip")
	}
}

func TestSessionAppendFailureDoesNotIncrementCount(t *testing.T) {
	s := &Session{path: t.TempDir()}
	if err := s.Append(SessionEntry{Role: "user", Content: "x"}); err == nil {
		t.Fatal("expected append failure")
	}
	if s.count != 0 {
		t.Fatalf("count = %d after failed append", s.count)
	}
}

func TestSessionCompactionRetainsConcurrentAppend(t *testing.T) {
	s := NewSession(t.TempDir(), "concurrent")
	for i := 0; i < 5; i++ {
		if err := s.Append(SessionEntry{Role: "user", Content: fmt.Sprintf("old-%d", i)}); err != nil {
			t.Fatal(err)
		}
	}
	started := make(chan struct{})
	release := make(chan struct{})
	done := make(chan struct{})
	go func() {
		s.ForceCompact(2, func(string) string {
			close(started)
			<-release
			return "semantic summary"
		})
		close(done)
	}()
	<-started
	if err := s.Append(SessionEntry{Role: "user", Content: "arrived-during-summary"}); err != nil {
		t.Fatal(err)
	}
	close(release)
	<-done

	s.mu.Lock()
	entries, err := s.readEntriesLocked()
	s.mu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, entry := range entries {
		if entry.Content == "arrived-during-summary" {
			found = true
		}
	}
	if !found {
		t.Fatal("concurrent append was lost during compaction")
	}
}
