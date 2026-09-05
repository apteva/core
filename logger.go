package core

import (
	"fmt"
	"os"
	"sync"
	"time"
)

var (
	logFile  *os.File
	logMu    sync.Mutex
	logReady bool
	logBytes int64
)

func initLogger() {
	f, err := os.OpenFile("apteva-core.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0600)
	if err != nil {
		fmt.Fprintf(os.Stderr, "warning: could not open log file: %v\n", err)
		return
	}
	_ = f.Chmod(0600)
	logFile = f
	logReady = true

	// Truncate if too large (>5MB)
	info, _ := f.Stat()
	if info != nil {
		logBytes = info.Size()
	}
	if info != nil && info.Size() > 5*1024*1024 {
		f.Truncate(0)
		logBytes = 0
		f.Seek(0, 0)
		logMsg("LOG", "truncated (was >5MB)")
	}
}

// Categories shown on stderr. Everything else is file-only.
var logStderrCategories = map[string]bool{
	"BOOT":   true,
	"API":    true,
	"CRASH":  true,
	"SIGNAL": true,
	"EXIT":   true,
}

func logMsg(category, msg string) {
	logMu.Lock()
	defer logMu.Unlock()
	ts := time.Now().Format("15:04:05.000")
	line := fmt.Sprintf("%s [%s] %s\n", ts, category, msg)
	if logReady {
		if logBytes+int64(len(line)) > 5<<20 {
			name := logFile.Name()
			_ = logFile.Close()
			_ = os.Rename(name, name+".1")
			var err error
			logFile, err = os.OpenFile(name, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0600)
			if err != nil {
				logReady = false
				return
			}
			logBytes = 0
		}
		n, _ := fmt.Fprint(logFile, line)
		logBytes += int64(n)
	}
	if logStderrCategories[category] {
		fmt.Fprint(os.Stderr, line)
	}
}
