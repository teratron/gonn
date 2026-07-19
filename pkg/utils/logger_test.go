package utils

import (
	"bytes"
	"log/slog"
	"strings"
	"testing"
)

func TestLevelTraceConstant(t *testing.T) {
	if LevelTrace >= slog.LevelDebug {
		t.Errorf("LevelTrace (%d) must be less than LevelDebug (%d)", LevelTrace, slog.LevelDebug)
	}
	if int(LevelTrace) != -8 {
		t.Errorf("LevelTrace must equal -8, got %d", int(LevelTrace))
	}
}

func TestNewGoLogger_RequiredFields(t *testing.T) {
	var buf bytes.Buffer
	handler := slog.NewJSONHandler(&buf, &slog.HandlerOptions{Level: LevelTrace})
	base := slog.New(handler)
	log := NewGoLogger(base, "0.8.0", "test-net")

	log.Info("hello")
	out := buf.String()
	if !strings.Contains(out, "lib_version") {
		t.Errorf("expected lib_version in log output, got: %s", out)
	}
	if !strings.Contains(out, "network_id") {
		t.Errorf("expected network_id in log output, got: %s", out)
	}
	if !strings.Contains(out, "0.8.0") {
		t.Errorf("expected version value in log output, got: %s", out)
	}
}

func TestNewGoLogger_NilSafe(t *testing.T) {
	// Must not panic when nil is passed
	log := NewGoLogger(nil, "0.8.0", "")
	log.Trace("trace")
	log.Debug("debug")
	log.Info("info")
	log.Warn("warn")
	log.Error("error")
}

func TestGoLogger_TraceFiltered(t *testing.T) {
	var buf bytes.Buffer
	// Handler level = Debug; Trace messages should be suppressed
	handler := slog.NewJSONHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})
	base := slog.New(handler)
	log := NewGoLogger(base, "test", "")
	log.Trace("should be filtered")
	if buf.Len() > 0 {
		t.Errorf("expected Trace to be filtered at LevelDebug handler, got: %s", buf.String())
	}
}

func TestGoLogger_TraceEnabled(t *testing.T) {
	var buf bytes.Buffer
	handler := slog.NewJSONHandler(&buf, &slog.HandlerOptions{Level: LevelTrace})
	base := slog.New(handler)
	log := NewGoLogger(base, "test", "")
	log.Trace("trace-message")
	if !strings.Contains(buf.String(), "trace-message") {
		t.Errorf("expected trace-message in output, got: %s", buf.String())
	}
}
