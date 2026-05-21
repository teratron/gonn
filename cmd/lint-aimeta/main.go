// cmd/lint-aimeta checks AI-Meta doc-comment block compliance across Go packages.
//
// Usage:
//
//	lint-aimeta [flags] <pkg-path> [<pkg-path>...]
//
// Flags:
//
//	--json              newline-delimited JSON output instead of text
//	--include-tests     include _test.go files (not yet wired — reserved)
//	--no-internal       skip TIER checks for internal-tier symbols
//	--resolve           activate symbol resolver (reserved for Phase 16)
//
// Exit codes: 0 = clean, 1 = violations found, 2 = fatal error, 3 = usage error.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/teratron/gonn/pkg/aimeta"
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("lint-aimeta", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "emit newline-delimited JSON instead of text")
	noInternal := fs.Bool("no-internal", false, "skip TIER checks for internal-tier symbols")
	_ = fs.Bool("include-tests", false, "include _test.go files (reserved)")
	resolve := fs.Bool("resolve", false, "auto-fix LABEL/INDENT/LAST violations in-place")

	if err := fs.Parse(args); err != nil {
		_, _ = fmt.Fprintf(stderr, "lint-aimeta: %v\n", err)
		return exitUsage
	}

	paths := fs.Args()
	if len(paths) == 0 {
		_, _ = fmt.Fprintln(stderr, "lint-aimeta: at least one package path required")
		fs.Usage()
		return exitUsage
	}

	opts := aimeta.Options{
		IncludeInternal: !*noInternal,
		Resolve:         *resolve,
	}

	// --resolve: apply mechanical fixes first, report what was changed.
	if *resolve {
		return runResolve(paths, opts, stdout, stderr)
	}

	var all []aimeta.Violation
	for _, p := range paths {
		viols, err := aimeta.Check(p, opts)
		if err != nil {
			_, _ = fmt.Fprintf(stderr, "lint-aimeta: %v\n", err)
			return exitError
		}
		all = append(all, viols...)
	}

	if *jsonOut {
		if err := printJSON(stdout, all); err != nil {
			_, _ = fmt.Fprintf(stderr, "lint-aimeta: json encode: %v\n", err)
			return exitError
		}
	} else {
		printText(stdout, all)
	}

	return resolveExitCode(all, nil)
}
