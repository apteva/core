package core

import (
	"encoding/json"
	"net/http"
)

// memoryForgetSource logically removes attributable memories and blocks future
// consolidation/search of the source. It does not erase transcripts or backups.
func (a *APIServer) memoryForgetSource(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	if a.thinker.memory == nil {
		http.Error(w, "memory store not initialized", http.StatusServiceUnavailable)
		return
	}
	var body struct {
		ThreadID string `json:"thread_id"`
		Reason   string `json:"reason"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil || !validMemoryThreadID(body.ThreadID) {
		http.Error(w, "valid thread_id and reason required", http.StatusBadRequest)
		return
	}
	count, err := a.thinker.memory.ForgetSource(body.ThreadID, body.Reason)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	writeJSON(w, map[string]any{"forgotten": count, "thread_id": body.ThreadID,
		"coverage": "logical memory deletion and future ingestion/search blocked; transcripts, journal and backups are not erased"})
}
