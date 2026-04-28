# Agent Rules

## 1. Language Policy

Consistency in communication and code is paramount.

### 1.1 Technical Content (English ONLY)

- **Codebase**: All identifiers (variables, functions, classes), comments, and docstrings.
- **Documentation**: Technical guides, READMEs, and implementation notes.
- **Process**: Commit messages, PR descriptions, and issue titles.
- **Environment**: Error messages, logs, and API definitions.

### 1.2 Communication (Russian)

- **Chat Interaction**: Discussions, explanations, and project planning.
- **Decision Making**: Strategic choices and high-level feature discussions.
- **Reviews**: Conversational feedback during pair programming.

## 2. Markdown Guidelines

- **Separators**: Avoid horizontal rules (`---`). Use them only in the footer if absolutely necessary.

## 3. Technology Stack

- **Language**: Go (latest stable version).
- **Dependencies**: Prefer standard library. After the local standard library, the next in priority is the remote standard library <https://cs.opensource.google/go>, <https://pkg.go.dev/golang.org/x>. Third-party packages require explicit justification.

## 4. Go Development Rules

### 4.1 Fundamentals & Best Practices

- **Idiomatic Go**: Follow *Effective Go* and community standards.
- **Simplicity**: Prioritize readability over cleverness.
- **Explicit Errors**: Handle all errors immediately; return them as the last value. Use `fmt.Errorf("%w", err)` for wrapping.
- **Composition**: Prefer composition over inheritance. Use interfaces for flexibility.
- **Project Layout**: Follow standard Go project structure (`/cmd`, `/pkg`, `/internal`).
- **Formatting**: Always use `gofmt` and `goimports` to maintain consistent code style.
- **Linting**: Ensure code passes `golangci-lint` with the project's configuration before submission.

### 4.2 Concurrency (Goroutines & Channels)

- **Communication**: Share memory by communicating (via channels); do not communicate by sharing memory.
- **Lifecycle**: Always manage goroutine lifecycles to avoid leaks. Use `sync.WaitGroup` or `context.Context` for synchronization/cancellation.
- **Safety**: Protect shared state with `sync.Mutex`/`sync.RWMutex` or atomic operations. Always run tests with `-race`.

### 4.3 Testing & Quality

- **Coverage**: Maintain a minimum of **80% code coverage** for all new Go files and packages. Focus on critical logic and edge cases.
- **Table-Driven Tests**: Use for covering multiple scenarios efficiently.
- **Benchmarks**: Write benchmarks for performance-critical paths using `testing.B`.
- **Fuzzing**: Use Go native fuzzing for input validation testing.

### 4.4 Performance Optimization

- **Profiling**: Use `pprof` to identify bottlenecks (CPU, Memory, Block).
- **Memory**: Minimize heap allocations; use `sync.Pool` for object reuse where applicable.
- **Preallocation**: Preallocate slices and maps if the size is known.

### 4.5 Integration & Architecture

- **gRPC/Protobuf**: Use for high-performance internal RPC.
- **Database**: Use connection pooling. Prevent SQL injection by using prepared statements or proper ORM/sqlx patterns.
- **Microservices**: Decouple by domain. Use lightweight communication and implement observability (tracing, metrics, logs).
- **Web**: Use `net/http` for simple services; use frameworks like Gin or Echo for complex routing/middleware while maintaining clean architecture.

## 5. Completion Protocol (Mandatory Checklist)

Before finishing any task, the agent MUST verify the following:

- [ ] **Technical Language**: All code, identifiers, comments, and technical docs are in English.
- [ ] **Communication Language**: All conversational responses and planning are in Russian.
- [ ] **Technology Stack**: Code is written in Go (latest stable), prioritizing the standard library.
- [ ] **Cognitive Discipline**: No steps skipped, no assumptions made without asking.
- [ ] **Visual Excellence**: Web/UI components (if any) follow premium design guidelines.
- [ ] **Code Quality**:
  - [ ] All new Go files have at least 80% test coverage.
  - [ ] Code is formatted with `gofmt` and follows standard linting rules.
- [ ] **Formatting**: No horizontal rules (---) used except in footers.
