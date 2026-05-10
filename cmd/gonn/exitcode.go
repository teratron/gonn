package main

import (
	"errors"
	"fmt"
	"os"

	"github.com/teratron/gonn/pkg/utils"
)

// Exit-code contract per l2-cli-client §5.3.
const (
	exitOK              = 0
	exitGeneric         = 1
	exitUserConfig      = 2
	exitInputData       = 3
	exitTrainingFailure = 4
	exitIntegrity       = 5
	exitIO              = 6
	exitUnsupported     = 7
)

// exitCode maps an error to the canonical CLI exit code by walking the
// error chain via errors.Is against pkg/utils sentinels.
func exitCode(err error) int {
	switch {
	case err == nil:
		return exitOK
	case errors.Is(err, utils.ErrUserConfig):
		return exitUserConfig
	case errors.Is(err, utils.ErrInputData):
		return exitInputData
	case errors.Is(err, utils.ErrCompute):
		return exitTrainingFailure
	case errors.Is(err, utils.ErrIntegrity):
		return exitIntegrity
	case errors.Is(err, utils.ErrIO):
		return exitIO
	default:
		return exitGeneric
	}
}

// die prints err to stderr and calls os.Exit with the appropriate code.
// Callers that cannot propagate the error up the call stack should call
// die; all others should return the error for testability.
func die(err error) {
	fmt.Fprintf(os.Stderr, "gonn: %v\n", err)
	os.Exit(exitCode(err))
}
