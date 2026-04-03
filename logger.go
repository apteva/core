package main

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
)

func initLogger() {
	f, err := os.OpenFile("apteva-core.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		fmt.Fprintf(os.Stderr, "warning: could not open log file: %v\n", err)
		return
	}
	logFile = f
	logReady = true

	// Truncate if too large (>5MB)
	info, _ := f.Stat()
	if info != nil && info.Size() > 5*1024*1024 {
		f.Truncate(0)
		f.Seek(0, 0)
		logMsg("LOG", "truncated (was >5MB)")
	}
}

// Categories shown on stderr. Everything else is file-only.
var logStderrCategories = map[string]bool{
	"BOOT": true,
	"API":  true,
}

func logMsg(category, msg string) {
	logMu.Lock()
	defer logMu.Unlock()
	ts := time.Now().Format("15:04:05.000")
	line := fmt.Sprintf("%s [%s] %s\n", ts, category, msg)
	if logReady {
		fmt.Fprint(logFile, line)
	}
	if logStderrCategories[category] {
		fmt.Fprint(os.Stderr, line)
	}
}
