package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"strings"
)

// versionCmd prints the binary version and Go runtime version.
// With --json it prints a machine-readable object per spec §5.4.
func versionCmd(args []string) int {
	fs := flag.NewFlagSet("version", flag.ContinueOnError)
	jsonOut := fs.Bool("json", false, "machine-readable JSON output")
	if err := fs.Parse(args); err != nil {
		fmt.Fprintf(os.Stderr, "gonn version: %v\n", err)
		return exitUserConfig
	}

	modVersion := moduleVersion()
	goVersion := strings.TrimPrefix(runtime.Version(), "go")

	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetEscapeHTML(false)
		_ = enc.Encode(struct {
			Version   string `json:"version"`
			GoVersion string `json:"go_version"`
		}{
			Version:   modVersion,
			GoVersion: goVersion,
		})
		return exitOK
	}

	fmt.Printf("gonn %s (go%s)\n", modVersion, goVersion)
	return exitOK
}

// moduleVersion returns the module version from build info, or "dev" when
// running outside a versioned module (e.g. go run . from the repo root).
func moduleVersion() string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return "dev"
	}
	v := info.Main.Version
	if v == "" || v == "(devel)" {
		return "dev"
	}
	return v
}
