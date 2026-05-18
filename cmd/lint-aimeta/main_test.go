package main

import (
	"bytes"
	"encoding/json"
	"io"
	"strings"
	"testing"
)

func TestRunCleanPackageExitsZero(t *testing.T) {
	t.Parallel()
	code := run([]string{"testdata/clean_pkg"}, io.Discard, io.Discard)
	if code != exitOK {
		t.Errorf("clean package: exit code = %d, want %d", code, exitOK)
	}
}

func TestRunViolatingPackageExitsOne(t *testing.T) {
	t.Parallel()
	code := run([]string{"testdata/violating_pkg"}, io.Discard, io.Discard)
	if code != exitViolated {
		t.Errorf("violating package: exit code = %d, want %d", code, exitViolated)
	}
}

func TestRunNoPathUsageError(t *testing.T) {
	t.Parallel()
	code := run([]string{}, io.Discard, io.Discard)
	if code != exitUsage {
		t.Errorf("no path: exit code = %d, want %d", code, exitUsage)
	}
}

func TestRunBadFlagUsageError(t *testing.T) {
	t.Parallel()
	code := run([]string{"--unknown-flag-xyz"}, io.Discard, io.Discard)
	if code != exitUsage {
		t.Errorf("bad flag: exit code = %d, want %d", code, exitUsage)
	}
}

func TestRunBadPathReturnsError(t *testing.T) {
	t.Parallel()
	code := run([]string{"testdata/does_not_exist"}, io.Discard, io.Discard)
	if code != exitError {
		t.Errorf("bad path: exit code = %d, want %d", code, exitError)
	}
}

func TestTextOutputContainsRule(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	code := run([]string{"testdata/violating_pkg"}, &buf, io.Discard)
	if code != exitViolated {
		t.Fatalf("expected exit 1, got %d", code)
	}
	if !strings.Contains(buf.String(), "[VOCAB]") {
		t.Errorf("text output missing [VOCAB]: %q", buf.String())
	}
}

func TestJSONOutputParseable(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	code := run([]string{"--json", "testdata/violating_pkg"}, &buf, io.Discard)
	if code != exitViolated {
		t.Fatalf("expected exit 1, got %d", code)
	}

	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(lines) == 0 {
		t.Fatal("expected at least one JSON line")
	}
	var rec violationJSON
	if err := json.Unmarshal([]byte(lines[0]), &rec); err != nil {
		t.Fatalf("first JSON line not valid JSON: %v", err)
	}
	if rec.Rule != "VOCAB" {
		t.Errorf("first record Rule = %q, want VOCAB", rec.Rule)
	}
}

func TestMultiplePaths(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	code := run([]string{"testdata/clean_pkg", "testdata/violating_pkg"}, &buf, io.Discard)
	if code != exitViolated {
		t.Errorf("mixed paths: exit code = %d, want %d", code, exitViolated)
	}
}
