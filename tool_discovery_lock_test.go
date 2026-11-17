package core

import (
	"context"
	"os"
	"os/exec"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Use a subprocess so a regression cannot strand the test suite with a reader
// and writer permanently waiting on each other. The child timeout dumps stacks.
func TestToolDiscoveryAuthorizationPendingWriter(t *testing.T) {
	const childEnv = "APTEVA_TEST_DISCOVERY_PENDING_WRITER"
	if os.Getenv(childEnv) != "1" {
		binary, err := os.Executable()
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, binary, "-test.run=^TestToolDiscoveryAuthorizationPendingWriter$", "-test.v", "-test.timeout=3s")
		cmd.Env = append(os.Environ(), childEnv+"=1")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("discovery authorization blocked with a pending catalog writer: %v\n%s", err, out)
		}
		return
	}
	for _, query := range []string{"billing_invoices_search", "search open invoices", "invoices_search", "invoice_lookup"} {
		t.Run(query, func(t *testing.T) { testDiscoveryAuthorizationPendingWriter(t, query) })
	}
}

func testDiscoveryAuthorizationPendingWriter(t *testing.T, query string) {
	index := NewToolIndex()
	index.Add("billing", []mcpToolDef{mkTool("invoices_search", "Search open billing invoices")}, false)
	index.Add("private", []mcpToolDef{mkTool("invoices_search", "Search private invoices")}, false)
	if err := index.RegisterAlias("invoice_lookup", "billing_invoices_search"); err != nil {
		t.Fatal(err)
	}
	thinker := &Thinker{toolIndex: index, toolAllowlist: map[string]bool{}, toolMCPScopes: map[string]bool{"billing": true}}
	predicate := thinker.discoveryAuthorizedPredicate()
	writerDone := make(chan struct{})
	started := false
	result := index.searchDetailedWithOptions(query, 5, false, func(entry IndexEntry) bool {
		if !started {
			started = true
			go func() {
				index.mu.Lock()
				index.mu.Unlock()
				close(writerDone)
			}()
			// Search already holds a read lock, so TryRLock can only fail once
			// the writer is pending. No sleeps or scheduling guess establish it.
			deadline := time.Now().Add(time.Second)
			for index.mu.TryRLock() {
				index.mu.RUnlock()
				if time.Now().After(deadline) {
					t.Fatal("catalog writer did not become pending")
				}
				runtime.Gosched()
			}
			t.Log("catalog writer is pending; evaluating worker authorization under the search read lock")
		}
		return predicate(entry)
	}, indexSearchOptions{})
	if !started {
		t.Fatal("search did not exercise authorization")
	}
	if query == "invoices_search" {
		if len(result.Hits) != 0 || !reflect.DeepEqual(result.Ambiguous[query], []string{"billing_invoices_search"}) {
			t.Fatalf("ambiguous local identity lost its permission boundary: %+v", result)
		}
	} else if len(result.Hits) != 1 || result.Hits[0].Name != "billing_invoices_search" {
		t.Fatalf("worker lost its server-scoped tool: %+v", result)
	}
	select {
	case <-writerDone:
	case <-time.After(time.Second):
		t.Fatal("catalog writer remained blocked after discovery")
	}
}

func TestToolDiscoveryAuthorizationUsesCatalogEntryAndGrantSnapshot(t *testing.T) {
	index := NewToolIndex()
	index.Add("billing", []mcpToolDef{mkTool("invoices_search", "Search invoices")}, false)
	entry, _ := index.Get("billing_invoices_search")
	thinker := &Thinker{toolIndex: index, toolAllowlist: map[string]bool{"billing_invoices_search": true}, toolMCPScopes: map[string]bool{"billing": false}}
	predicate := thinker.discoveryAuthorizedPredicate()
	thinker.toolAllowlist[entry.Name] = false
	thinker.toolMCPScopes[entry.Server] = true
	if !predicate(entry) {
		t.Fatal("later grants changed the captured exact-tool permission")
	}
	ungranted := entry
	ungranted.Name = "billing_customers_search"
	if predicate(ungranted) {
		t.Fatal("later MCP scope leaked into the captured authorization state")
	}
	// The predicate must use its supplied entry, even when the live catalog
	// now contains different policy metadata for that same name.
	index.UpdatePolicy("billing", true, nil)
	if !predicate(entry) {
		t.Fatal("authorization consulted live index metadata instead of the search snapshot")
	}
	entry.NoSpawn = true
	if predicate(entry) {
		t.Fatal("exact grant bypassed no_spawn in the supplied catalog snapshot")
	}
	thinker.allowNoSpawn = true
	if predicate(entry) {
		t.Fatal("later no_spawn override changed the captured policy")
	}
	if !thinker.discoveryAuthorizedPredicate()(entry) {
		t.Fatal("explicit no_spawn override did not authorize the new snapshot")
	}
	thinker.toolAllowlist = nil
	if thinker.discoveryAuthorizedPredicate() != nil {
		t.Fatal("unrestricted main should not need an authorization callback")
	}
}

func TestNativeToolPreparationConcurrentCatalogRefresh(t *testing.T) {
	const childEnv = "APTEVA_TEST_NATIVE_PREPARATION_REFRESH"
	if os.Getenv(childEnv) != "1" {
		binary, err := os.Executable()
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, binary, "-test.run=^TestNativeToolPreparationConcurrentCatalogRefresh$", "-test.v", "-test.timeout=15s")
		cmd.Env = append(os.Environ(), childEnv+"=1")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("native preparation failed during catalog refresh: %v\n%s", err, out)
		}
		return
	}
	t.Setenv("APTEVA_TOOL_SEARCH", "on")
	index, registry := NewToolIndex(), NewToolRegistry("")
	billing := []mcpToolDef{mkTool("invoices_search", "Search open billing invoices"), mkTool("customers_get", "Read billing customer details")}
	for server, tools := range map[string][]mcpToolDef{
		"billing": billing,
		"crm":     {mkTool("customers_search", "Search customer invoices")},
		"private": {mkTool("secrets_get", "Read private invoice secrets")},
	} {
		index.Add(server, tools, server == "private")
		registerTestMCPTools(registry, server, tools)
	}
	newThinker := func(id string) *Thinker {
		return &Thinker{threadID: id, registry: registry, toolIndex: index, activeTools: map[string]bool{},
			messages: []Message{{Role: "user", Content: "Search open billing invoices"}},
			config:   &Config{AutomaticToolLoading: &AutomaticToolLoadingConfig{Enabled: true, WorkflowTools: []string{"billing_invoices_search"}, MaxTools: 2, MaxSchemaTokens: 2000}}}
	}
	main, worker := newThinker("main"), newThinker("worker")
	worker.toolAllowlist, worker.toolMCPScopes = map[string]bool{}, map[string]bool{"billing": true}
	start, stop, writerDone := make(chan struct{}), make(chan struct{}), make(chan struct{})
	var refreshes atomic.Int64
	go func() {
		defer close(writerDone)
		<-start
		for {
			select {
			case <-stop:
				return
			default:
				index.Add("billing", billing, false)
				index.UpdatePolicy("billing", false, nil)
				refreshes.Add(1)
				runtime.Gosched()
			}
		}
	}()
	var readers sync.WaitGroup
	for _, thinker := range []*Thinker{main, worker} {
		readers.Go(func() {
			<-start
			for i := 0; i < 100; i++ {
				tools := thinker.prepareNativeTools("openai-codex")
				found := false
				for _, tool := range tools {
					found = found || tool.Name == "billing_invoices_search"
					if thinker == worker && !strings.HasPrefix(tool.Name, "billing_") {
						t.Errorf("worker schema leaked an ungranted tool: %s", tool.Name)
					}
				}
				if !found {
					t.Error("native preparation lost the authorized workflow tool")
					return
				}
			}
		})
	}
	close(start)
	readers.Wait()
	close(stop)
	<-writerDone
	if refreshes.Load() == 0 {
		t.Fatal("test did not exercise catalog refresh")
	}
	t.Logf("main and worker each completed 100 native preparations alongside %d catalog refreshes", refreshes.Load())
}
