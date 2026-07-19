package utils

import (
	"context"
	"io"
	"log/slog"
	"os"
)

// LevelTrace is a custom slog level below Debug (-8). Use it for highly
// verbose per-iteration log lines in the training hot path. Handlers with
// level > LevelTrace will discard these entries with negligible overhead.
//
// AI-Meta:
//   - Purpose: Sub-Debug slog level for per-iteration tracing in training loops.
//   - Usage: logger.Log(ctx, utils.LevelTrace, "iter", "batch", i, "loss", l).
//   - Related: [GoLogger.Trace], [NewGoLogger].
//   - Stability: Stable.
const LevelTrace = slog.Level(-8)

// Logger is the package-level structured logger shared by all GoNN internals.
// Replace before network construction to route library output to a custom sink.
//
// AI-Meta:
//   - Purpose: Shared slog.Logger instance for structured, levelled output across the library.
//   - Usage: utils.Logger = slog.New(slog.NewJSONHandler(w, nil)) before calling NewBuilder or New.
//   - Concurrency: Safe; slog.Logger is goroutine-safe after assignment.
var Logger *slog.Logger

func init() {
	Logger = slog.New(slog.NewTextHandler(os.Stdout, nil))
}

// GoLogger wraps a *slog.Logger and injects library metadata (lib_version,
// network_id) at construction time via slog.Logger.With. It adds a Trace
// method for sub-Debug verbosity and provides the same Debug/Info/Warn/Error
// methods as slog.Logger for drop-in use inside pkg/nn/.
//
// A nil inner logger passed to NewGoLogger is replaced with a no-op discard
// handler so callers never need to nil-check the returned *GoLogger.
//
// AI-Meta:
//   - Purpose: Attributed slog wrapper used by NN training loop for structured lifecycle events.
//   - Usage: log := utils.NewGoLogger(cfg.Logger, "0.8.0", ""); log.Info("training started").
//   - Concurrency: Safe; inherits slog.Logger goroutine-safety.
//   - Related: [LevelTrace], [Logger].
//   - Stability: Stable.
type GoLogger struct {
	inner *slog.Logger
}

// NewGoLogger constructs a GoLogger that decorates l with lib_version and
// network_id attributes. If l is nil a discard handler is used so Trace/Debug
// etc. are always safe to call.
//
// AI-Meta:
//   - Purpose: Construct an attributed GoLogger; nil-safe via discard fallback.
//   - Usage: log := utils.NewGoLogger(userLogger, "0.8.0", netID).
//   - Related: [GoLogger], [LevelTrace].
//   - Stability: Stable.
func NewGoLogger(l *slog.Logger, libVersion, networkID string) *GoLogger {
	if l == nil {
		l = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
	return &GoLogger{
		inner: l.With("lib_version", libVersion, "network_id", networkID),
	}
}

// Trace emits a message at LevelTrace. No-op unless the handler's minimum
// level is ≤ LevelTrace.
func (g *GoLogger) Trace(msg string, args ...any) {
	g.inner.Log(context.Background(), LevelTrace, msg, args...)
}

// Debug emits a message at slog.LevelDebug.
func (g *GoLogger) Debug(msg string, args ...any) { g.inner.Debug(msg, args...) }

// Info emits a message at slog.LevelInfo.
func (g *GoLogger) Info(msg string, args ...any) { g.inner.Info(msg, args...) }

// Warn emits a message at slog.LevelWarn.
func (g *GoLogger) Warn(msg string, args ...any) { g.inner.Warn(msg, args...) }

// Error emits a message at slog.LevelError.
func (g *GoLogger) Error(msg string, args ...any) { g.inner.Error(msg, args...) }
