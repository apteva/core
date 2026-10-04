// apteva-core — agent runtime binary.
//
// This file is a thin shim. All runtime logic lives in github.com/apteva/core
// (the parent package). Keeping `package main` minimal here lets the rest
// of core be a plain library that scenarios, tests, and other binaries
// can import directly — `package main` cannot be imported.
//
// Build:
//
//	go build -o apteva-core ./cmd/apteva-core
//
// Build with versioning (matches the existing Dockerfile + scripts):
//
//	go build -ldflags "-X main.Version=$APTEVA_VERSION \
//	                   -X main.BuildTime=$BUILD_TIME \
//	                   -X main.CLIVersion=$CLI_VERSION \
//	                   -X main.DashboardVersion=$DASHBOARD_VERSION \
//	                   -X main.IntegrationsVersion=$INTEGRATIONS_VERSION \
//	                   -X main.CoreVersion=$CORE_VERSION" \
//	  -o apteva-core ./cmd/apteva-core
package main

import (
	"encoding/json"
	"fmt"
	"github.com/apteva/core"
	"io"
	"os"
)

// Version + BuildTime are injected by ldflags at build time. The remaining
// vars exist purely so the umbrella ldflags string the monorepo uses
// (shared with apteva-server) doesn't error on unknown symbols when
// applied to this binary. Only Version + BuildTime are forwarded into
// core; the rest are intentionally unused.
var (
	Version             = "dev"
	BuildTime           = "dev"
	CLIVersion          = "dev"
	DashboardVersion    = "dev"
	IntegrationsVersion = "dev"
	CoreVersion         = "dev"
)

func main() {
	if len(os.Args) > 1 && (os.Args[1] == "--version" || os.Args[1] == "version") {
		// Keep --version first: older binaries safely return their version when
		// Server probes this protocol instead of starting an agent.
		if len(os.Args) > 2 && os.Args[2] == "--builtin-capabilities" {
			_ = json.NewEncoder(os.Stdout).Encode(core.ProviderBuiltinCatalog())
			return
		}
		if len(os.Args) > 2 && os.Args[2] == "--validate-builtins" {
			var request struct {
				Provider string                            `json:"provider"`
				Builtins map[string]core.BuiltinToolConfig `json:"builtins"`
			}
			decoder := json.NewDecoder(io.LimitReader(os.Stdin, 1<<20))
			decoder.DisallowUnknownFields()
			err := decoder.Decode(&request)
			var result map[string]core.BuiltinToolConfig
			if err == nil {
				result, err = core.ValidateProviderBuiltins(request.Provider, request.Builtins)
			}
			if err != nil {
				_ = json.NewEncoder(os.Stdout).Encode(map[string]any{"error": err.Error()})
				return
			}
			_ = json.NewEncoder(os.Stdout).Encode(map[string]any{"builtins": result})
			return
		}
		fmt.Printf("apteva-core %s (%s)\n", Version, BuildTime)
		return
	}
	core.SetVersion(Version, BuildTime)
	core.Run()
}

// Reference the unused version vars so the linker keeps them targetable
// by `-X main.X=...` ldflags. Without this, an over-aggressive linker
// could elide them, making the ldflag write a silent no-op.
var _ = []string{CLIVersion, DashboardVersion, IntegrationsVersion, CoreVersion}
