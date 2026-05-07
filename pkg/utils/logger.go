package utils

import (
	"log/slog"
	"os"
)

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
