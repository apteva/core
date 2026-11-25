package core

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func behaviorBlock(rule string) string {
	return "<!-- apteva:behavior:v1:start -->\n## Behavior\n" + rule + "\n<!-- apteva:behavior:end -->"
}

func TestWithManagedBlocksReplacesInPlaceAndIsStable(t *testing.T) {
	worker := "## Role\nSummarize reports.\n\n" + behaviorBlock("Proactivity 25.") + "\n\n## Output\nOne line."
	next := withManagedBlocks(worker, []managedBlock{{Name: "behavior", Text: behaviorBlock("Proactivity 60.")}})
	want := "## Role\nSummarize reports.\n\n" + behaviorBlock("Proactivity 60.") + "\n\n## Output\nOne line."
	if next != want {
		t.Fatalf("block moved or text changed:\n%s", next)
	}
	if again := withManagedBlocks(next, managedBlocks(next)); again != next {
		t.Fatalf("applying the same blocks changed the directive:\n%s", again)
	}
}

func TestWithManagedBlocksAppendsMissingAndDropsStaleAndDuplicates(t *testing.T) {
	worker := "## Role\nWork.\n\n" + behaviorBlock("old") + "\n\n<!-- apteva:retired:start -->\nx\n<!-- apteva:retired:end -->\n\n" + behaviorBlock("dup")
	blocks := []managedBlock{{Name: "behavior", Text: behaviorBlock("new")}, {Name: "extra", Text: "<!-- apteva:extra:start -->\nS\n<!-- apteva:extra:end -->"}}
	got := withManagedBlocks(worker, blocks)
	if strings.Count(got, "apteva:behavior:v1:start") != 1 || !strings.Contains(got, "\nnew\n") || strings.Contains(got, "dup") || strings.Contains(got, "old") {
		t.Fatalf("behavior block not reduced to one current copy:\n%s", got)
	}
	if strings.Contains(got, "retired") {
		t.Fatalf("block outside the set kept:\n%s", got)
	}
	if !strings.HasSuffix(got, "\n\n<!-- apteva:extra:start -->\nS\n<!-- apteva:extra:end -->") {
		t.Fatalf("missing block not appended after a blank line:\n%q", got)
	}
	if plain := withoutManagedBlocks(got); strings.Contains(plain, "apteva:") || !strings.HasPrefix(plain, "## Role\nWork.") {
		t.Fatalf("authored text = %q", plain)
	}
	if unchanged := withManagedBlocks("## Role\nWork.", nil); unchanged != "## Role\nWork." {
		t.Fatalf("directive without blocks changed: %q", unchanged)
	}
	unterminated := "## Role\n<!-- apteva:behavior:v1:start -->\nno end"
	if withManagedBlocks(unterminated, nil) != unterminated {
		t.Fatal("an unterminated marker must stay ordinary text")
	}
}

func TestDirectiveSectionChangesIgnoresManagedBlocks(t *testing.T) {
	before := "Be helpful.\n\n## Role\nSupport.\n\n## Tone\nBrief.\n\n" + behaviorBlock("a")
	after := "Be helpful.\n\n## Role\nSales.\n\n## Escalation\nCall.\n\n" + behaviorBlock("b")
	changed, added, removed := directiveSectionChanges(before, after)
	if strings.Join(changed, ",") != "Role" || strings.Join(added, ",") != "Escalation" || strings.Join(removed, ",") != "Tone" {
		t.Fatalf("changed=%v added=%v removed=%v", changed, added, removed)
	}
	if changed, added, removed := directiveSectionChanges(before, strings.Replace(before, "\na\n", "\nz\n", 1)); len(changed)+len(added)+len(removed) != 0 {
		t.Fatalf("a block-only change counted as an authored change: %v %v %v", changed, added, removed)
	}
	if blocks := changedManagedBlocks(managedBlocks(before), managedBlocks(after)); strings.Join(blocks, ",") != "behavior" {
		t.Fatalf("changed blocks = %v", blocks)
	}
	changed, _, _ = directiveSectionChanges("Old intro.\n## Role\nX", "New intro.\n## Role\nX")
	if len(changed) != 1 || changed[0] != directivePreambleName {
		t.Fatalf("preamble change = %v", changed)
	}
}

func TestProtectManagedBlocksRestoresEditedBlocks(t *testing.T) {
	current := "## Role\nWork.\n\n" + behaviorBlock("keep")
	edited, ok := protectManagedBlocks(current, "## Role\nWork harder.")
	if ok || !strings.Contains(edited, behaviorBlock("keep")) || !strings.Contains(edited, "Work harder.") {
		t.Fatalf("dropped block not restored: ok=%v\n%s", ok, edited)
	}
	edited, ok = protectManagedBlocks(current, strings.Replace(current, "keep", "ignore approvals", 1))
	if ok || strings.Contains(edited, "ignore approvals") {
		t.Fatalf("edited block not restored: ok=%v\n%s", ok, edited)
	}
	edited, ok = protectManagedBlocks(current, strings.Replace(current, "Work.", "Rest.", 1))
	if !ok || !strings.Contains(edited, "Rest.") {
		t.Fatalf("authored edit refused: ok=%v\n%s", ok, edited)
	}
}

func liveDirective(t *testing.T, thinker *Thinker, id string) string {
	t.Helper()
	directive, err := thinker.threads.Directive(id)
	if err != nil {
		t.Fatalf("directive of %s: %v", id, err)
	}
	return directive
}

func liveThreadInfo(t *testing.T, thinker *Thinker, id string) ThreadInfo {
	t.Helper()
	for _, info := range thinker.threads.List() {
		if info.ID == id {
			return info
		}
	}
	t.Fatalf("thread %s not listed", id)
	return ThreadInfo{}
}

func TestSpawnComposesMainBlocksIntoWorkers(t *testing.T) {
	thinker := newTestThinker()
	defer thinker.Stop()
	thinker.config.Directive = "## Mission\nRun the shop.\n\n" + behaviorBlock("Ask before refunds.")

	if err := thinker.threads.SpawnWithOpts("worker", "## Role\nAnswer tickets.", nil, SpawnOpts{DeferRun: true}); err != nil {
		t.Fatal(err)
	}
	worker := liveDirective(t, thinker, "worker")
	if worker != "## Role\nAnswer tickets.\n\n"+behaviorBlock("Ask before refunds.") {
		t.Fatalf("worker directive:\n%s", worker)
	}
	if err := thinker.threads.SpawnWithOpts("app", "", nil, SpawnOpts{DeferRun: true, InheritMain: true, DirectiveSuffix: "\n\nYou handle chat.", APIManaged: true}); err != nil {
		t.Fatal(err)
	}
	if got := liveDirective(t, thinker, "app"); got != thinker.config.Directive+"\n\nYou handle chat." {
		t.Fatalf("inheriting directive:\n%s", got)
	}
	if err := thinker.threads.SpawnWithOpts("unconscious", "Consolidate.", nil, SpawnOpts{DeferRun: true, System: true}); err != nil {
		t.Fatal(err)
	}
	if got := liveDirective(t, thinker, "unconscious"); got != "Consolidate." {
		t.Fatalf("system thread directive changed: %q", got)
	}
}

func TestSyncThreadDirectivesRefreshesInheritingQuietlyAndNotifiesWorkers(t *testing.T) {
	thinker := newTestThinker()
	defer thinker.Stop()
	thinker.config.path = filepath.Join(t.TempDir(), "config.json")
	thinker.config.Directive = "## Mission\nRun the shop.\n\n" + behaviorBlock("Proactivity 25.")

	if err := thinker.threads.SpawnWithOpts("reporter", "## Role\nReport.\n\n"+behaviorBlock("Proactivity 25.")+"\n\n## Output\nOne line.", nil, SpawnOpts{DeferRun: true}); err != nil {
		t.Fatal(err)
	}
	if err := thinker.threads.SpawnWithOpts("chat", "", nil, SpawnOpts{DeferRun: true, InheritMain: true, DirectiveSuffix: "\n\nReply with CODE: ALPHA.", APIManaged: true}); err != nil {
		t.Fatal(err)
	}
	thinker.drainEventTexts()
	threadThinker := func(id string) *Thinker {
		thinker.threads.mu.RLock()
		defer thinker.threads.mu.RUnlock()
		return thinker.threads.threads[id].Thinker
	}
	threadThinker("reporter").drainEventTexts()
	threadThinker("chat").drainEventTexts()

	before := thinker.config.Directive
	thinker.config.Directive = "## Mission\nRun the store.\n\n" + behaviorBlock("Proactivity 60.")
	sync := thinker.syncThreadDirectives()
	if strings.Join(sync.Inherited, ",") != "chat" || strings.Join(sync.Policy, ",") != "reporter" || len(sync.Failed) != 0 {
		t.Fatalf("sync = %+v", sync)
	}

	reporter := liveDirective(t, thinker, "reporter")
	want := "## Role\nReport.\n\n" + behaviorBlock("Proactivity 60.") + "\n\n## Output\nOne line."
	if reporter != want {
		t.Fatalf("worker block not replaced in place:\n%s", reporter)
	}
	if strings.Contains(reporter, "Proactivity 25") {
		t.Fatal("worker kept the old rule next to the new one")
	}
	events := drainEventTextsWait(t, threadThinker("reporter"), 1, time.Second)
	if len(events) != 1 || !strings.Contains(events[0], "[policy updated] behavior policy") {
		t.Fatalf("worker notice = %v", events)
	}
	if got := liveDirective(t, thinker, "chat"); got != thinker.config.Directive+"\n\nReply with CODE: ALPHA." {
		t.Fatalf("inheriting thread not recomputed:\n%s", got)
	}
	if got := drainEventTextsWait(t, threadThinker("chat"), 1, 100*time.Millisecond); len(got) != 0 {
		t.Fatalf("inheriting thread was woken: %v", got)
	}
	stored, ok := persistentThreadByID(thinker.config.GetThreads(), "chat")
	if !ok || !stored.InheritMain || !stored.APIManaged || stored.DirectiveSuffix != "\n\nReply with CODE: ALPHA." || stored.Directive != thinker.config.Directive+"\n\nReply with CODE: ALPHA." {
		t.Fatalf("stored inheriting thread = %+v", stored)
	}

	notice := thinker.directiveChangeNotice(before, thinker.config.Directive, sync)
	for _, part := range []string{"[directive updated]", "Changed sections: Mission.", "Workers you manage were set up under the previous directive: reporter.", "kept in line with it for you: chat. Do not update or kill them."} {
		if !strings.Contains(notice, part) {
			t.Fatalf("main notice missing %q:\n%s", part, notice)
		}
	}
	if again := thinker.syncThreadDirectives(); len(again.Inherited)+len(again.Policy) != 0 {
		t.Fatalf("second sync changed threads again: %+v", again)
	}
	if notice := thinker.directiveChangeNotice(thinker.config.Directive, thinker.config.Directive, directiveSyncResult{}); notice != "" {
		t.Fatalf("no-op change produced a notice: %q", notice)
	}
	blockOnly := strings.Replace(thinker.config.Directive, "Proactivity 60.", "Proactivity 70.", 1)
	if notice := thinker.directiveChangeNotice(thinker.config.Directive, blockOnly, directiveSyncResult{}); !strings.HasPrefix(notice, "[policy updated]") {
		t.Fatalf("block-only change notice = %q", notice)
	}
}

func TestAPIManagedThreadsRefuseAgentUpdateAndKillButAcceptSend(t *testing.T) {
	thinker := newTestThinker()
	defer thinker.Stop()
	if err := thinker.threads.SpawnWithOpts("chat", "", nil, SpawnOpts{DeferRun: true, InheritMain: true, APIManaged: true}); err != nil {
		t.Fatal(err)
	}
	if err := thinker.threads.SpawnWithOpts("worker", "## Role\nWork.", nil, SpawnOpts{DeferRun: true}); err != nil {
		t.Fatal(err)
	}
	err := thinker.threads.ValidateManagedTarget("chat")
	if err == nil || !strings.Contains(err.Error(), "created by the platform") || !strings.Contains(err.Error(), "send") {
		t.Fatalf("API thread check = %v", err)
	}
	if err := thinker.threads.ValidateManagedTarget("worker"); err != nil {
		t.Fatalf("agent's own worker refused: %v", err)
	}
	if err := thinker.threads.SendAgentWithParts("chat", "hello", nil); err != nil {
		t.Fatalf("agent send to API thread refused: %v", err)
	}
	roster := rosterEntry(liveThreadInfo(t, thinker, "chat"))
	if !strings.Contains(roster, "created by the platform") {
		t.Fatalf("roster does not mark the thread: %s", roster)
	}
}

func putJSONForTest(t *testing.T, handler func(http.ResponseWriter, *http.Request), path string, body any) *httptest.ResponseRecorder {
	t.Helper()
	payload, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPut, path, bytes.NewReader(payload))
	rec := httptest.NewRecorder()
	handler(rec, req)
	return rec
}

func mainInboxTexts(thinker *Thinker) []string {
	var out []string
	for {
		select {
		case event := <-thinker.sub.C:
			out = append(out, event.Text)
		default:
			return out
		}
	}
}

func TestAPIThreadsAreManagedAndFollowMain(t *testing.T) {
	api, thinker, _ := newPersistentThreadTestAPI(t)
	thinker.config.Directive = "## Mission\nRun the shop.\n\n" + behaviorBlock("Ask first.")

	if w := postThreadForTest(t, api, "app-chat", map[string]any{"directive_suffix": "\n\nApp role."}); w.Code != http.StatusOK {
		t.Fatalf("spawn app thread: %d %s", w.Code, w.Body.String())
	}
	if w := postThreadForTest(t, api, "explicit", map[string]any{"directive": "## Role\nOwn text."}); w.Code != http.StatusOK {
		t.Fatalf("spawn explicit thread: %d %s", w.Code, w.Body.String())
	}

	app := liveThreadInfo(t, thinker, "app-chat")
	if !app.APIManaged || !app.InheritMain || app.DirectiveSuffix != "\n\nApp role." {
		t.Fatalf("app thread = managed %v inherit %v suffix %q", app.APIManaged, app.InheritMain, app.DirectiveSuffix)
	}
	explicit := liveThreadInfo(t, thinker, "explicit")
	if !explicit.APIManaged || explicit.InheritMain || explicit.Directive != "## Role\nOwn text.\n\n"+behaviorBlock("Ask first.") {
		t.Fatalf("explicit thread = %+v", explicit)
	}
	if stored, ok := persistentThreadByID(thinker.config.GetThreads(), "app-chat"); !ok || !stored.APIManaged || !stored.InheritMain {
		t.Fatalf("stored app thread = %+v", stored)
	}

	// An explicit directive on PUT ends inheritance and gets main's blocks.
	if w := putJSONForTest(t, func(w http.ResponseWriter, r *http.Request) { api.updateThread(w, r, "app-chat") }, "/threads/app-chat", map[string]any{"directive": "## Role\nFixed."}); w.Code != http.StatusOK {
		t.Fatalf("explicit update: %d %s", w.Code, w.Body.String())
	}
	app = liveThreadInfo(t, thinker, "app-chat")
	if app.InheritMain || !app.APIManaged || app.Directive != "## Role\nFixed.\n\n"+behaviorBlock("Ask first.") {
		t.Fatalf("after explicit update = %+v", app)
	}
	// A suffix on PUT makes it follow main again.
	if w := putJSONForTest(t, func(w http.ResponseWriter, r *http.Request) { api.updateThread(w, r, "app-chat") }, "/threads/app-chat", map[string]any{"directive_suffix": "\n\nNew role."}); w.Code != http.StatusOK {
		t.Fatalf("suffix update: %d %s", w.Code, w.Body.String())
	}
	app = liveThreadInfo(t, thinker, "app-chat")
	if !app.InheritMain || app.DirectiveSuffix != "\n\nNew role." || app.Directive != thinker.config.Directive+"\n\nNew role." {
		t.Fatalf("after suffix update = %+v", app)
	}

	rec := httptest.NewRecorder()
	api.threads(rec, httptest.NewRequest(http.MethodGet, "/threads", nil))
	var listed []map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &listed); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, row := range listed {
		if row["id"] == "app-chat" {
			found = row["api_managed"] == true && row["inherit_main"] == true && row["directive_suffix"] == "\n\nNew role."
		}
	}
	if !found {
		t.Fatalf("GET /threads does not report the flags: %s", rec.Body.String())
	}
}

func TestAPIBackfillsFlagsOnRepost(t *testing.T) {
	api, thinker, _ := newPersistentThreadTestAPI(t)
	thinker.config.Directive = "## Mission\nRun."
	// A thread restored from before Core recorded these flags.
	if err := thinker.threads.SpawnWithOpts("legacy", thinker.config.Directive+" suffix", nil, SpawnOpts{DeferRun: true}); err != nil {
		t.Fatal(err)
	}
	if info := liveThreadInfo(t, thinker, "legacy"); info.APIManaged || info.InheritMain {
		t.Fatalf("fixture already flagged: %+v", info)
	}
	if w := postThreadForTest(t, api, "legacy", map[string]any{"directive_suffix": " suffix"}); w.Code != http.StatusOK {
		t.Fatalf("repost: %d %s", w.Code, w.Body.String())
	}
	info := liveThreadInfo(t, thinker, "legacy")
	if !info.APIManaged || !info.InheritMain || info.DirectiveSuffix != " suffix" {
		t.Fatalf("backfill = %+v", info)
	}
	if stored, ok := persistentThreadByID(thinker.config.GetThreads(), "legacy"); !ok || !stored.APIManaged || !stored.InheritMain {
		t.Fatalf("backfill not persisted: %+v", stored)
	}
	// A diverged directive is not claimed as inheriting.
	if err := thinker.threads.SpawnWithOpts("custom", "## Role\nCustom.", nil, SpawnOpts{DeferRun: true}); err != nil {
		t.Fatal(err)
	}
	postThreadForTest(t, api, "custom", map[string]any{"directive_suffix": " suffix"})
	if info := liveThreadInfo(t, thinker, "custom"); info.InheritMain || !info.APIManaged {
		t.Fatalf("diverged thread backfill = %+v", info)
	}
}

func TestAPIConfigDirectiveChangeSyncsThreadsAndNotifiesMain(t *testing.T) {
	api, thinker, _ := newPersistentThreadTestAPI(t)
	thinker.config.Directive = "## Mission\nRun the shop.\n\n## Tone\nBrief.\n\n" + behaviorBlock("Proactivity 25.")
	if w := postThreadForTest(t, api, "app-chat", map[string]any{"directive_suffix": "\n\nApp."}); w.Code != http.StatusOK {
		t.Fatalf("spawn: %d %s", w.Code, w.Body.String())
	}
	if err := thinker.threads.SpawnWithOpts("reporter", "## Role\nReport.", nil, SpawnOpts{DeferRun: true}); err != nil {
		t.Fatal(err)
	}
	mainInboxTexts(thinker)

	newDirective := "## Mission\nRun the store.\n\n## Tone\nBrief.\n\n" + behaviorBlock("Proactivity 60.")
	if w := putJSONForTest(t, api.config, "/config", map[string]any{"directive": newDirective}); w.Code != http.StatusOK {
		t.Fatalf("config put: %d %s", w.Code, w.Body.String())
	}
	if got := liveDirective(t, thinker, "app-chat"); got != newDirective+"\n\nApp." {
		t.Fatalf("inheriting thread:\n%s", got)
	}
	if got := liveDirective(t, thinker, "reporter"); !strings.Contains(got, "Proactivity 60.") || strings.Contains(got, "Proactivity 25.") {
		t.Fatalf("worker policy:\n%s", got)
	}
	inbox := strings.Join(mainInboxTexts(thinker), "\n")
	if !strings.Contains(inbox, "[console] [directive updated]") || !strings.Contains(inbox, "Changed sections: Mission.") || !strings.Contains(inbox, "reporter") {
		t.Fatalf("main notice:\n%s", inbox)
	}
}

func TestEvolveKeepsManagedBlocksAndInheritance(t *testing.T) {
	main := "## Mission\nRun.\n\n" + behaviorBlock("Ask first.")
	// A worker evolve that drops the block gets it back.
	edited, ok := protectManagedBlocks("## Role\nA.\n\n"+behaviorBlock("Ask first."), "## Role\nB.")
	if ok || !strings.Contains(edited, behaviorBlock("Ask first.")) {
		t.Fatalf("evolve protection: %v\n%s", ok, edited)
	}
	// An inheriting thread that only appends keeps following main.
	inherit, suffix := evolvedInheritance(main, true, " old", main+" old plus")
	if !inherit || suffix != " old plus" {
		t.Fatalf("append-only evolve: inherit=%v suffix=%q", inherit, suffix)
	}
	inherit, _ = evolvedInheritance(main, true, " old", "## Mission\nSomething else.")
	if inherit {
		t.Fatal("rewriting main's part must end inheritance")
	}
}

func TestRestoreKeepsFlagsAndRecomputesFromCurrentMain(t *testing.T) {
	t.Chdir(t.TempDir())
	cfg := &Config{path: configFile, Directive: "## Mission\nOld.\n\n" + behaviorBlock("Old rule.")}
	if err := cfg.SaveThread(PersistentThread{ID: "chat", Directive: cfg.Directive + " suffix", APIManaged: true, InheritMain: true, DirectiveSuffix: " suffix"}); err != nil {
		t.Fatal(err)
	}
	if err := cfg.SaveThread(PersistentThread{ID: "reporter", Directive: "## Role\nReport.\n\n" + behaviorBlock("Old rule.")}); err != nil {
		t.Fatal(err)
	}
	// The operator changed main while the agent was stopped.
	cfg.Directive = "## Mission\nNew.\n\n" + behaviorBlock("New rule.")
	if err := cfg.Save(); err != nil {
		t.Fatal(err)
	}
	reloaded := NewConfig()
	if err := reloaded.LoadError(); err != nil {
		t.Fatal(err)
	}
	restarted := NewThinker("", newRecordingThreadEventProvider(), reloaded)
	defer func() {
		restarted.threads.KillAll()
		restarted.Stop()
	}()
	chat := liveThreadInfo(t, restarted, "chat")
	if !chat.APIManaged || !chat.InheritMain || chat.Directive != reloaded.GetDirective()+" suffix" {
		t.Fatalf("restored inheriting thread = managed %v inherit %v directive %q", chat.APIManaged, chat.InheritMain, chat.Directive)
	}
	reporter := liveThreadInfo(t, restarted, "reporter")
	if !strings.Contains(reporter.Directive, "New rule.") || strings.Contains(reporter.Directive, "Old rule.") || !strings.HasPrefix(reporter.Directive, "## Role\nReport.") {
		t.Fatalf("restored worker policy = %q", reporter.Directive)
	}
}
