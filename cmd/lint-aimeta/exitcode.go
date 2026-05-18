package main

import (
	"github.com/teratron/gonn/pkg/aimeta"
)

// exit codes per l2-aimeta-linter §5.5 / l2-cli-client §5.3.
const (
	exitOK       = 0 // no violations found
	exitViolated = 1 // one or more violations found
	exitError    = 2 // fatal error (parse failure, bad path)
	exitUsage    = 3 // bad flags / usage error
)

// resolveExitCode returns the appropriate exit code for the given result.
func resolveExitCode(violations []aimeta.Violation, err error) int {
	if err != nil {
		return exitError
	}
	if len(violations) > 0 {
		return exitViolated
	}
	return exitOK
}
