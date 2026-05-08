# Contributing to GoNN

Thank you for your interest in contributing to **GoNN** (Go Neural Network) — a pure-Go library for building and training neural networks with zero external dependencies.

This document describes the development workflow, coding standards, and review process. By contributing, you agree to license your work under the [MIT License](LICENSE).

## Code of Conduct

Be respectful. Disagreements are resolved with technical arguments and reproducible examples, not personal commentary. Maintainers may close issues or PRs that violate this principle.

## Project Structure

```plaintext
gonn/
├── pkg/
│   ├── activation/   # 10 activation functions (sigmoid, relu, softmax, tanh, ...)
│   ├── layer/        # Layer hierarchy (Input, Dense, Output) via embedding
│   ├── loss/         # 18 loss functions (MSE, BCE, CCE, Huber, ...)
│   ├── network/      # Internal computational graph (Network[T], bundles)
│   ├── neuron/       # Neuron, Cell, Axon model
│   ├── nn/           # Public API facade — NN[T] builder
│   └── utils/        # Float constraint, logger
├── examples/         # Runnable examples (perceptron, ...)
├── .design/          # SDD specifications (architecture & invariants)
├── .agents/          # AI agent workflows and rules
└── .magic/           # SDD engine (Magic Spec — read-only)
```

The `.design/`, `.agents/`, and `.magic/` directories are part of the **Specification-Driven Development (SDD)** workflow. They coordinate the human + AI agent collaboration model used to evolve the library — they are **not** part of the published Go module.

## Prerequisites

- **Go**: `1.26.3` or later (see [`go.mod`](go.mod))
- **Tooling**:
  - `gofmt` (bundled with Go)
  - `goimports` — `go install golang.org/x/tools/cmd/goimports@latest`
  - `golangci-lint` — see [installation guide](https://golangci-lint.run/usage/install/)
- **Optional** (for contributors using SDD workflows):
  - Node.js ≥ 18 (used by the Magic Spec engine in `.magic/scripts/`)
  - An AI agent that supports slash commands (Claude Code, Cursor, Windsurf, etc.)

## Getting Started

```bash
git clone https://github.com/teratron/gonn.git
cd gonn
go mod download
go build ./...
go test ./...
```

To run a sample example:

```bash
go run ./examples/perceptron
```

## Coding Standards

### Language & Formatting

- All code, identifiers, comments, and docstrings are in **English**.
- Code MUST be formatted with `gofmt` and imports organized with `goimports` before submission.
- Code MUST pass `golangci-lint run` with the project's configuration (no new warnings).
- Follow [Effective Go](https://go.dev/doc/effective_go) and idiomatic patterns.

### Project Conventions

The library enforces five hard conventions (see [`.design/RULES.md §7`](.design/RULES.md) for the full constitution):

| ID | Convention | Summary |
| :--- | :--- | :--- |
| **C25** | Generic Float Constraint | All numeric computation uses `utils.Float` (`float32 \| float64`). No hardcoded float types in generic code. |
| **C26** | Compile-Time Interface Verification | Every concrete type implementing an interface MUST include `var _ InterfaceName[float32] = (*TypeName[float32])(nil)`. |
| **C27** | Dispatcher Pattern | Enum-based dispatchers follow: (a) `Type uint8` enum with `String()`, (b) one file per enum value, (c) switch-based dispatcher. Used in `pkg/activation/` and `pkg/loss/`. |
| **C28** | Composition via Embedding | Type hierarchies use Go struct embedding, not interface embedding or inheritance simulation. Pattern: base unexported → extended unexported → exported specialized. |
| **C29** | Zero External Dependencies | Only the Go standard library is permitted. Third-party dependencies require explicit justification and approval in a PR discussion. |

Adding a new activation or loss function requires all three steps of C27 (enum entry + implementation file + dispatcher case) plus a `String()` method update.

### Error Handling

- Handle every error immediately at the call site; return errors as the last value.
- Wrap errors with context using `fmt.Errorf("...: %w", err)`.
- `panic` is reserved for truly unrecoverable conditions (e.g., constant-time invariant violations).

### Concurrency

- Share memory by communicating (channels), not by communicating through shared memory.
- Manage goroutine lifecycles with `sync.WaitGroup` or `context.Context`.
- Run all concurrent code under `go test -race` before submission.

## Testing

| Requirement | Threshold |
| :--- | :--- |
| Code coverage for new packages | **≥ 80%** |
| Test style | Table-driven (`tests := []struct{...}{...}`) |
| Race detection | Required (`go test -race ./...`) |
| Benchmarks | Required for performance-critical paths (`testing.B`) |
| Fuzzing | Encouraged for input validation (`testing.F`) |

```bash
# Run all tests with race detector
go test -race -cover ./...

# Generate coverage report
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out
```

Pull requests that decrease coverage on existing packages will be flagged. Pull requests that introduce a new package without `_test.go` files will not be merged.

## Specification-Driven Development

This project uses SDD: every non-trivial change is rooted in a specification under [`.design/specifications/`](.design/specifications/). The eight current specs (8 × Stable, L1/L2) cover:

- L1: neural network architecture, math functions framework
- L2: NN facade, network graph, layer types, neuron model, activation functions, loss functions

### When a spec is required

| Change type | Spec required? |
| :--- | :--- |
| Bug fix (no behavior change) | No |
| Adding a test | No |
| New activation / loss function | Update existing L2 spec |
| New layer type | Update `l2-layer-types.md` |
| New public API (e.g., new builder method) | Update `l2-nn-facade.md` (RFC → Stable) |
| Architectural change (e.g., new training algorithm) | New L1 spec + dependent L2s |

### AI-assisted workflow (optional)

If you use a supported AI agent, you can drive the spec/code lifecycle with slash commands:

| Command | Purpose |
| :--- | :--- |
| `/magic-spec` | Create or amend a specification |
| `/magic-task` | Generate the implementation plan from approved specs |
| `/magic-run` | Execute tasks and write implementation code |
| `/magic-analyze` | Audit code/spec drift, coverage, and structural integrity |
| `/magic-rule` | Add or amend a project convention in `RULES.md §7` |

The engine lives in `.magic/` and is invoked through the cross-platform script runner: `node .magic/scripts/executor.js <script-name> [args]`. Direct manual edits to `.magic/` are discouraged — see convention `C1` in `RULES.md`.

Manual contributions without an AI agent are equally welcome — just edit the spec file directly and open a PR.

## Pull Request Process

1. **Branch from `master`** (or `develop` for ongoing feature work). Use a descriptive name: `feat/add-prelu-activation`, `fix/dense-layer-bias-init`, `docs/clarify-loss-readme`.
2. **Make your change**:
   - Update or create a specification if the change introduces new public API or architectural concepts.
   - Implement the change with accompanying tests.
   - Run `gofmt`, `goimports`, `golangci-lint`, and `go test -race -cover ./...` locally.
3. **Commit**:
   - Use [Conventional Commits](https://www.conventionalcommits.org/) style: `feat(activation): add PReLU function`, `fix(layer/dense): correct bias gradient`, `docs(loss): clarify Huber delta parameter`.
   - All commit messages and PR descriptions in English.
4. **Open a PR**:
   - Link the related spec(s) under `.design/specifications/`.
   - Note any new convention entries you propose in `RULES.md §7`.
   - Confirm test coverage threshold is met.
5. **Review**:
   - Maintainers will review for spec alignment, convention compliance, and test quality.
   - Address feedback by pushing additional commits to the same branch (no force-push during active review).
6. **Merge**: Maintainers merge using squash-merge to keep `master` history linear.

## Reporting Issues

When opening an issue, include:

- Go version (`go version`)
- OS and architecture
- Minimal reproducible example (Go playground link or short snippet)
- Expected vs actual behavior
- For panics: full stack trace

For security-sensitive issues, please email the maintainer privately rather than opening a public issue.

## License

By submitting code, documentation, or specifications, you agree that your contribution is licensed under the [MIT License](LICENSE) of this project.
