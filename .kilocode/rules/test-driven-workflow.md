# Test-Driven Development Workflow

## Brief overview

This document outlines the test-driven development (TDD) workflow for this Go project. Adherence to these rules is mandatory to ensure code quality, maintainability, and robustness. Every change must be validated through this process.

## core Principles

- **Test-First Approach**: All code modifications (bug fixes, new features, refactoring) must be covered by Go unit tests. If tests are missing for the modified code, the first step is to write them using the standard `testing` package.
- **Isolate and Conquer**: When tests reveal failures in unrelated modules, create a new, dedicated task to debug and fix the separate problem. Do not proceed with the original task until the blocking issue is resolved.
- **Mandatory Validation**: A task is not complete until all quality checks pass. Never assume a change works without verification.
- **Role-Based Delegation**: Leverage specialized modes for specific sub-tasks. For instance, delegate comprehensive test creation to a `test-engineer` persona and complex debugging to a `debug` persona.

## Development & CI Workflow

The project uses a modern Go stack managed by Go modules. Adhere strictly to the following commands for all operations:

- **Dependency Management**: Use Go modules for all package installation and management (e.g., `go mod tidy`, `go get`, `go mod vendor`).
- **Code Quality & Verification (Run before every completion)**:
    1. Format Code: `go fmt ./...`
    2. Lint and Static Analysis: `golangci-lint run` (or `go vet ./...` for basic checks)
    3. Run Automated Tests: `go test ./...` (with coverage: `go test -coverprofile=coverage.out ./...`)
    4. Run Benchmarks: `go test -bench=./... -benchmem` for performance validation
    5. Treat Warnings as Errors: All warnings generated during testing must be treated as errors and resolved immediately.
- **Package Building**: When ready for distribution, build the package using `go build` or `go install`.
- **Module Management**: Ensure `go.mod` and `go.sum` are properly maintained and committed with all changes.
