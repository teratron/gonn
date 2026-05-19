package main

import (
	"fmt"
	"io"

	"github.com/teratron/gonn/pkg/aimeta"
)

// runResolve applies mechanical fixes (LABEL/INDENT/LAST) to each path and
// reports what was changed. After fixing, it re-lints each path and reports
// any remaining violations. Exit code mirrors resolveExitCode on the residual.
func runResolve(paths []string, opts aimeta.Options, stdout, stderr io.Writer) int {
	var fixedAll []aimeta.Violation
	for _, p := range paths {
		fixes, err := aimeta.Resolve(p, opts)
		if err != nil {
			fmt.Fprintf(stderr, "lint-aimeta --resolve: %v\n", err)
			return exitError
		}
		fixedAll = append(fixedAll, fixes...)
	}

	if len(fixedAll) > 0 {
		fmt.Fprintf(stdout, "lint-aimeta --resolve: applied %d fix(es)\n", len(fixedAll))
		printText(stdout, fixedAll)
	} else {
		fmt.Fprintln(stdout, "lint-aimeta --resolve: no fixable violations found")
	}

	// Re-lint to surface any remaining violations.
	checkOpts := opts
	checkOpts.Resolve = false
	var remaining []aimeta.Violation
	for _, p := range paths {
		viols, err := aimeta.Check(p, checkOpts)
		if err != nil {
			fmt.Fprintf(stderr, "lint-aimeta --resolve re-lint: %v\n", err)
			return exitError
		}
		remaining = append(remaining, viols...)
	}

	if len(remaining) > 0 {
		fmt.Fprintln(stdout, "\nRemaining violations after resolve:")
		printText(stdout, remaining)
	}

	return resolveExitCode(remaining, nil)
}
