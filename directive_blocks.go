package core

import (
	"regexp"
	"strings"
)

// Managed directive blocks are platform-owned regions of a directive:
//
//	<!-- apteva:NAME[:vN]:start --> ... <!-- apteva:NAME:end -->
//
// Core treats their contents as opaque. The blocks in main's directive are
// agent-wide policy (the server's behavior block, for example): every worker
// carries the same blocks, a block stays where it already sits in a worker's
// own text, and no model edit (evolve or update) may change or drop one.

var managedBlockStartRe = regexp.MustCompile(`<!-- apteva:([a-z][a-z0-9-]*)(?::v[0-9]+)?:start -->`)

type managedBlock struct {
	Name string
	Text string // the whole block, markers included
}

type managedBlockSpan struct {
	name       string
	start, end int // byte offsets, end exclusive
}

func findManagedBlockSpans(directive string) []managedBlockSpan {
	var spans []managedBlockSpan
	offset := 0
	for offset < len(directive) {
		loc := managedBlockStartRe.FindStringSubmatchIndex(directive[offset:])
		if loc == nil {
			break
		}
		name := directive[offset+loc[2] : offset+loc[3]]
		bodyStart := offset + loc[1]
		endMarker := "<!-- apteva:" + name + ":end -->"
		rel := strings.Index(directive[bodyStart:], endMarker)
		if rel < 0 {
			// An unterminated marker is ordinary text, not a block.
			offset = bodyStart
			continue
		}
		end := bodyStart + rel + len(endMarker)
		spans = append(spans, managedBlockSpan{name: name, start: offset + loc[0], end: end})
		offset = end
	}
	return spans
}

// managedBlocks returns the first block of each name, in directive order.
func managedBlocks(directive string) []managedBlock {
	seen := map[string]bool{}
	var out []managedBlock
	for _, span := range findManagedBlockSpans(directive) {
		if seen[span.name] {
			continue
		}
		seen[span.name] = true
		out = append(out, managedBlock{Name: span.name, Text: directive[span.start:span.end]})
	}
	return out
}

func sameManagedBlocks(a, b []managedBlock) bool {
	if len(a) != len(b) {
		return false
	}
	byName := make(map[string]string, len(a))
	for _, block := range a {
		byName[block.Name] = block.Text
	}
	for _, block := range b {
		if text, ok := byName[block.Name]; !ok || text != block.Text {
			return false
		}
	}
	return true
}

// withManagedBlocks makes directive carry exactly blocks. A block that is
// already present is replaced where it stands, duplicates and blocks outside
// the set are removed, and missing blocks are appended after a blank line.
// The result is stable: applying the same blocks again changes nothing.
func withManagedBlocks(directive string, blocks []managedBlock) string {
	want := make(map[string]string, len(blocks))
	order := make([]string, 0, len(blocks))
	for _, block := range blocks {
		if _, dup := want[block.Name]; dup {
			continue
		}
		want[block.Name] = block.Text
		order = append(order, block.Name)
	}
	var out strings.Builder
	placed := map[string]bool{}
	last := 0
	for _, span := range findManagedBlockSpans(directive) {
		prefix := directive[last:span.start]
		if text, keep := want[span.name]; keep && !placed[span.name] {
			out.WriteString(prefix)
			out.WriteString(text)
			placed[span.name] = true
		} else {
			out.WriteString(strings.TrimSuffix(prefix, "\n\n"))
		}
		last = span.end
	}
	out.WriteString(directive[last:])
	result := out.String()
	for _, name := range order {
		if placed[name] {
			continue
		}
		if strings.TrimSpace(result) == "" {
			result = want[name]
		} else {
			result = strings.TrimRight(result, "\n") + "\n\n" + want[name]
		}
	}
	return result
}

// withoutManagedBlocks returns the operator- or model-authored text.
func withoutManagedBlocks(directive string) string {
	return withManagedBlocks(directive, nil)
}

// changedManagedBlocks names the blocks whose presence or text differs.
func changedManagedBlocks(before, after []managedBlock) []string {
	beforeText := map[string]string{}
	for _, block := range before {
		beforeText[block.Name] = block.Text
	}
	var changed []string
	seen := map[string]bool{}
	for _, block := range after {
		seen[block.Name] = true
		if text, ok := beforeText[block.Name]; !ok || text != block.Text {
			changed = append(changed, block.Name)
		}
	}
	for _, block := range before {
		if !seen[block.Name] {
			changed = append(changed, block.Name)
		}
	}
	return changed
}

// directiveSectionChanges compares the authored text of two directives by
// Markdown section. Text before the first heading counts as its own section.
func directiveSectionChanges(before, after string) (changed, added, removed []string) {
	beforeSections := authoredSections(before)
	afterSections := authoredSections(after)
	beforeByKey := map[string]authoredSection{}
	for _, section := range beforeSections {
		beforeByKey[section.key] = section
	}
	afterKeys := map[string]bool{}
	for _, section := range afterSections {
		afterKeys[section.key] = true
		previous, ok := beforeByKey[section.key]
		switch {
		case !ok:
			added = append(added, section.name)
		case previous.body != section.body:
			changed = append(changed, section.name)
		}
	}
	for _, section := range beforeSections {
		if !afterKeys[section.key] {
			removed = append(removed, section.name)
		}
	}
	return changed, added, removed
}

type authoredSection struct {
	key, name, body string
}

const directivePreambleName = "(text before the first heading)"

func authoredSections(directive string) []authoredSection {
	lines := directiveLines(withoutManagedBlocks(directive))
	var sections []authoredSection
	var preamble []string
	i := 0
	for ; i < len(lines); i++ {
		if _, ok := markdownHeadingName(lines[i]); ok {
			break
		}
		preamble = append(preamble, lines[i])
	}
	if text := strings.TrimSpace(strings.Join(preamble, "\n")); text != "" {
		sections = append(sections, authoredSection{key: "\x00preamble", name: directivePreambleName, body: text})
	}
	seen := map[string]int{}
	for i < len(lines) {
		name, _ := markdownHeadingName(lines[i])
		j := i + 1
		for j < len(lines) {
			if _, ok := markdownHeadingName(lines[j]); ok {
				break
			}
			j++
		}
		key := normalizeSectionName(name)
		seen[key]++
		if seen[key] > 1 {
			key = key + "\x00" + strings.Repeat("+", seen[key])
		}
		sections = append(sections, authoredSection{key: key, name: name, body: strings.TrimSpace(strings.Join(lines[i+1:j], "\n"))})
		i = j
	}
	return sections
}
