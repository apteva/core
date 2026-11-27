package core

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"
	"time"
)

// Directive propagation rules (main is the root of the policy):
//
//   - Threads that inherit main (created through the API with a directive
//     suffix) are recomputed as main's directive + suffix. They are not woken;
//     the new prompt applies on their next event.
//   - Every other non-system, non-realtime worker carries main's managed
//     blocks. When one changes, it is replaced where it stands and the worker
//     gets a short notice naming it.
//   - Workers' own text is never rewritten. Main is told which sections
//     changed and which of its workers to review (see directiveChangeNotice).

type directiveSyncResult struct {
	Inherited []string // recomputed from main without waking them
	Policy    []string // managed blocks updated, notice sent
	Skipped   []string // realtime threads, left as configured by their caller
	Failed    []string
}

type threadSyncTarget struct {
	id          string
	system      bool
	realtime    bool
	inheritMain bool
	suffix      string
	directive   string
}

func (tm *ThreadManager) collectSyncTargets() []threadSyncTarget {
	if tm == nil {
		return nil
	}
	tm.mu.RLock()
	targets := make([]threadSyncTarget, 0, len(tm.threads))
	children := make([]*ThreadManager, 0)
	for _, thread := range tm.threads {
		targets = append(targets, threadSyncTarget{
			id: thread.ID, system: thread.System, realtime: thread.IsRealtime,
			inheritMain: thread.InheritMain, suffix: thread.DirectiveSuffix, directive: thread.Directive,
		})
		if thread.Children != nil {
			children = append(children, thread.Children)
		}
	}
	tm.mu.RUnlock()
	sort.Slice(targets, func(i, j int) bool { return targets[i].id < targets[j].id })
	for _, child := range children {
		targets = append(targets, child.collectSyncTargets()...)
	}
	return targets
}

// syncThreadDirectives brings every thread in line with main's current
// directive. It runs on main's loop after the directive changes.
func (t *Thinker) syncThreadDirectives() directiveSyncResult {
	var result directiveSyncResult
	if t == nil || t.threads == nil || t.config == nil {
		return result
	}
	mainDirective := t.config.GetDirective()
	blocks := managedBlocks(mainDirective)
	syncedAt := time.Now().UnixNano()
	for _, target := range t.threads.collectSyncTargets() {
		if target.system {
			continue
		}
		if target.inheritMain {
			next := mainDirective + target.suffix
			if next == target.directive {
				continue
			}
			if _, err := t.threads.UpdateWithEvents(target.id, "", next, nil, ThreadUpdateOptions{Quiet: true}, nil); err != nil {
				logMsg("DIRECTIVE", fmt.Sprintf("refresh inherited directive of %s failed: %v", target.id, err))
				result.Failed = append(result.Failed, target.id)
				continue
			}
			result.Inherited = append(result.Inherited, target.id)
			continue
		}
		if target.realtime {
			if withManagedBlocks(target.directive, blocks) != target.directive {
				result.Skipped = append(result.Skipped, target.id)
			}
			continue
		}
		next := withManagedBlocks(target.directive, blocks)
		if next == target.directive {
			continue
		}
		changed := changedManagedBlocks(managedBlocks(target.directive), managedBlocks(next))
		notice := policyUpdateNotice(changed)
		// A directive can return to an earlier text (a mode changed, then
		// changed back), so the notice ID is unique per sync rather than per text.
		event := PersistentThreadEvent{ID: fmt.Sprintf("directive-policy-%d-%s", syncedAt, shortDirectiveHash(next)), Text: notice, Notice: true}
		event.Hash = threadEventHash(event.Text, nil)
		if _, err := t.threads.UpdateWithEvents(target.id, "", next, nil, ThreadUpdateOptions{}, []PersistentThreadEvent{event}); err != nil {
			logMsg("DIRECTIVE", fmt.Sprintf("apply directive policy to %s failed: %v", target.id, err))
			result.Failed = append(result.Failed, target.id)
			continue
		}
		result.Policy = append(result.Policy, target.id)
	}
	if t.telemetry != nil && (len(result.Inherited)+len(result.Policy)+len(result.Skipped)+len(result.Failed)) > 0 {
		t.telemetry.Emit("directive.synced", t.threadID, map[string]any{
			"inherited": result.Inherited, "policy": result.Policy,
			"skipped": result.Skipped, "failed": result.Failed,
		})
	}
	return result
}

func shortDirectiveHash(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:8])
}

func managedBlockLabel(name string) string {
	switch name {
	case "behavior":
		return "behavior policy"
	default:
		return name + " policy"
	}
}

func policyUpdateNotice(changed []string) string {
	labels := make([]string, 0, len(changed))
	for _, name := range changed {
		labels = append(labels, managedBlockLabel(name))
	}
	if len(labels) == 0 {
		labels = append(labels, "platform-maintained sections")
	}
	return "[policy updated] " + strings.Join(labels, ", ") +
		" in your directive changed. The new text is already applied; follow it from now on. These sections are maintained for you: do not evolve them."
}

// directiveChangeNotice tells main what an operator edit changed and which of
// its own workers to review. A change limited to platform blocks gets the short
// policy notice; no change at all gets "".
func (t *Thinker) directiveChangeNotice(before, after string, sync directiveSyncResult) string {
	changed, added, removed := directiveSectionChanges(before, after)
	blockChanges := changedManagedBlocks(managedBlocks(before), managedBlocks(after))
	if len(changed)+len(added)+len(removed) == 0 {
		if len(blockChanges) == 0 {
			return ""
		}
		return policyUpdateNotice(blockChanges)
	}
	var sb strings.Builder
	sb.WriteString("[directive updated] The operator changed your directive.")
	writeList := func(label string, names []string) {
		if len(names) > 0 {
			sb.WriteString(" " + label + ": " + strings.Join(names, ", ") + ".")
		}
	}
	writeList("Changed sections", changed)
	writeList("Added sections", added)
	writeList("Removed sections", removed)
	var managed, platform []string
	if t.threads != nil {
		for _, info := range t.threads.List() {
			if info.System {
				continue
			}
			if info.APIManaged || info.InheritMain {
				platform = append(platform, info.ID)
				continue
			}
			managed = append(managed, info.ID)
		}
	}
	sort.Strings(managed)
	sort.Strings(platform)
	if len(managed) > 0 {
		sb.WriteString("\nWorkers you manage were set up under the previous directive: " + strings.Join(managed, ", ") +
			". For each, decide whether the change affects its work; if it does, patch only the affected section with update. Leave unaffected workers alone.")
	}
	if len(platform) > 0 {
		sb.WriteString("\nThreads the platform created are kept in line with it for you: " + strings.Join(platform, ", ") + ". Do not update or kill them.")
	}
	return sb.String()
}

// leaderDirectiveNotice is appended to a "[directive updated]" message sent to
// a leader, so the cascade reaches its own workers one level at a time.
func leaderDirectiveNotice(children *ThreadManager) string {
	if children == nil {
		return ""
	}
	var ids []string
	for _, info := range children.List() {
		if !info.System && !info.APIManaged && !info.InheritMain {
			ids = append(ids, info.ID)
		}
	}
	if len(ids) == 0 {
		return ""
	}
	sort.Strings(ids)
	return "\n\nYour workers were set up under your previous directive: " + strings.Join(ids, ", ") +
		". If the change affects them, patch only the affected section with update."
}

// protectManagedBlocks keeps a model edit (evolve or update) from changing or
// dropping the managed blocks a directive must carry.
func protectManagedBlocks(current, edited string) (string, bool) {
	want := managedBlocks(current)
	if sameManagedBlocks(want, managedBlocks(edited)) {
		return edited, true
	}
	return withManagedBlocks(edited, want), false
}

// evolvedInheritance decides whether an inheriting thread still follows main
// after it edits its own directive. It does while main's directive is still an
// unchanged prefix; whatever follows becomes its suffix.
func evolvedInheritance(mainDirective string, inheritMain bool, suffix, next string) (bool, string) {
	if !inheritMain {
		return false, ""
	}
	if strings.HasPrefix(next, mainDirective) {
		return true, next[len(mainDirective):]
	}
	return false, ""
}
