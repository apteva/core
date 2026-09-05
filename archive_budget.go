package core

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"sync"
)

var archiveBudgets = struct {
	sync.Mutex
	used map[string]int64
}{used: map[string]int64{}}

// Bound disk growth without deleting audit history. Limits can be raised by
// the host; exhaustion is a persistence error, so further effects stop.
func reserveArchiveBytes(root string, n int64) (func(bool), error) {
	root, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	limit := int64(1 << 30)
	if configured, err := strconv.ParseInt(os.Getenv("APTEVA_TOOL_ARCHIVE_MAX_BYTES"), 10, 64); err == nil && configured > 0 {
		limit = configured
	}
	archiveBudgets.Lock()
	used, ok := archiveBudgets.used[root]
	if !ok {
		err = filepath.WalkDir(root, func(_ string, d fs.DirEntry, e error) error {
			if os.IsNotExist(e) {
				return nil
			}
			if e != nil {
				return e
			}
			if !d.IsDir() {
				i, e := d.Info()
				if e != nil {
					return e
				}
				used += i.Size()
			}
			return nil
		})
		if err != nil {
			archiveBudgets.Unlock()
			return nil, err
		}
	}
	if n > limit-used {
		archiveBudgets.Unlock()
		return nil, fmt.Errorf("tool archive capacity exceeded (%d bytes); increase APTEVA_TOOL_ARCHIVE_MAX_BYTES or archive history", limit)
	}
	archiveBudgets.used[root] = used + n
	archiveBudgets.Unlock()
	return func(committed bool) {
		if !committed {
			archiveBudgets.Lock()
			archiveBudgets.used[root] -= n
			archiveBudgets.Unlock()
		}
	}, nil
}
