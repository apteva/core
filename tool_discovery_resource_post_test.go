package core

import (
	"reflect"
	"testing"
)

func TestPostResourceReadClassificationRetainsMutationPrecedence(t *testing.T) {
	cases := []struct {
		name  string
		hints map[string]any
		want  ToolAccessClass
	}{
		{"post_list", nil, ToolAccessRead}, {"social_post_list", nil, ToolAccessRead},
		{"get_post", nil, ToolAccessRead}, {"post_history", nil, ToolAccessRead},
		{"post", nil, ToolAccessCreate}, {"post_create", nil, ToolAccessCreate},
		{"post_create_list", nil, ToolAccessCreate}, {"post_update_list", nil, ToolAccessUpdate},
		{"post_delete_list", nil, ToolAccessDelete}, {"post_send_list", nil, ToolAccessCreate},
		{"post_retry_list", nil, ToolAccessExecute}, {"post_schedule_list", nil, ToolAccessExecute},
		{"post_run_get", nil, ToolAccessExecute}, {"post_list", map[string]any{"readOnlyHint": false}, ToolAccessUnknown},
		{"post_list", map[string]any{"destructiveHint": true}, ToolAccessDelete},
		{"post_list", map[string]any{"readOnlyHint": true, "destructiveHint": true}, ToolAccessUnknown},
		{"opaque", map[string]any{"readOnlyHint": true, "destructiveHint": false}, ToolAccessRead},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tool := mkTool(tc.name, "Social operation")
			tool.Annotations = tc.hints
			if got := classifyToolAccess(tool); got != tc.want {
				t.Fatalf("got %s want %s", got, tc.want)
			}
		})
	}
}

func TestStrictReadOnlySocialDiscoveryLoadsPostListOnly(t *testing.T) {
	index := NewToolIndex()
	tools := []mcpToolDef{mkTool("post_list", "List stored Social posts with per-target status"), mkTool("post_create", "Create Social post"), mkTool("post_retry", "Retry Social post"), mkTool("post_update", "Update Social post"), mkTool("post_delete", "Delete Social post")}
	index.Add("social", tools, false)
	thinker := &Thinker{threadID: "main", toolIndex: index, activeTools: map[string]bool{}}
	result := thinker.discoverTools(DiscoveryRequest{Intents: []DiscoveryIntent{{Query: "post", Servers: []string{"social"}, Access: DiscoveryAccessReadOnly, Limit: 10, Priority: 90}}, AllowNoSpawn: true, MaxTools: 10, MaxSchemaTokens: 16000, Activate: true})
	if !reflect.DeepEqual(result.Loaded, []string{"social_post_list"}) {
		t.Fatalf("unsafe/missing reads: %+v", result)
	}
}
