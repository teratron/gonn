package utils

import (
	"errors"
	"fmt"
	"strings"
	"testing"
)

// allSentinels lists every public category sentinel exported by errors.go.
// New sentinels MUST be added here so the routing test catches them
// automatically — this is the single point of contact between the registry
// and the test suite.
var allSentinels = []struct {
	name     string
	sentinel error
}{
	{"ErrUserConfig", ErrUserConfig},
	{"ErrInputData", ErrInputData},
	{"ErrCompute", ErrCompute},
	{"ErrControl", ErrControl},
	{"ErrIntegrity", ErrIntegrity},
	{"ErrIO", ErrIO},
}

func TestSentinelsAreOrthogonal(t *testing.T) {
	t.Parallel()
	for i, a := range allSentinels {
		for j, b := range allSentinels {
			if i == j {
				continue
			}
			if errors.Is(a.sentinel, b.sentinel) {
				t.Errorf("%s must not match %s via errors.Is", a.name, b.name)
			}
		}
	}
}

func TestNewfRoutesThroughCategory(t *testing.T) {
	t.Parallel()
	for _, s := range allSentinels {
		t.Run(s.name, func(t *testing.T) {
			t.Parallel()
			err := Newf(s.sentinel, "field %q out of range, got %d", "size", 0)
			if !errors.Is(err, s.sentinel) {
				t.Fatalf("errors.Is(err, %s) = false; want true", s.name)
			}
			for _, other := range allSentinels {
				if other.sentinel == s.sentinel {
					continue
				}
				if errors.Is(err, other.sentinel) {
					t.Errorf("Newf must not transitively match %s", other.name)
				}
			}
		})
	}
}

func TestWrapPreservesCauseAndCategory(t *testing.T) {
	t.Parallel()
	cause := errors.New("disk full")
	err := Wrap(ErrIO, cause, "writing checkpoint %q", "weights.json")
	if !errors.Is(err, ErrIO) {
		t.Errorf("errors.Is(err, ErrIO) = false; want true")
	}
	if !errors.Is(err, cause) {
		t.Errorf("errors.Is(err, cause) = false; want true (cause must be reachable)")
	}
	if errors.Is(err, ErrIntegrity) {
		t.Errorf("errors.Is(err, ErrIntegrity) = true; want false (no cross-category leak)")
	}
}

func TestWrapNilCauseReturnsNil(t *testing.T) {
	t.Parallel()
	if got := Wrap(ErrIO, nil, "noop"); got != nil {
		t.Errorf("Wrap(ErrIO, nil, ...) = %v; want nil", got)
	}
}

func TestNewfPanicsOnNilCategory(t *testing.T) {
	t.Parallel()
	defer func() {
		if r := recover(); r == nil {
			t.Fatalf("Newf(nil, ...) must panic, got no panic")
		}
	}()
	_ = Newf(nil, "boom")
}

func TestWrapPanicsOnNilCategoryWithNonNilCause(t *testing.T) {
	t.Parallel()
	defer func() {
		if r := recover(); r == nil {
			t.Fatalf("Wrap(nil, cause, ...) must panic, got no panic")
		}
	}()
	_ = Wrap(nil, errors.New("x"), "boom")
}

func TestNewSizeError(t *testing.T) {
	t.Parallel()
	err := NewSizeError("Input", 0, "positive")
	if !errors.Is(err, ErrUserConfig) {
		t.Errorf("NewSizeError must wrap ErrUserConfig")
	}
	wantSubstr := []string{"Input", "positive", "0"}
	for _, s := range wantSubstr {
		if !strings.Contains(err.Error(), s) {
			t.Errorf("error %q missing required field %q", err.Error(), s)
		}
	}
}

func TestNewActivationError(t *testing.T) {
	t.Parallel()
	err := NewActivationError("XELU")
	if !errors.Is(err, ErrUserConfig) {
		t.Errorf("NewActivationError must wrap ErrUserConfig")
	}
	if !strings.Contains(err.Error(), `"XELU"`) {
		t.Errorf("error %q must mention the unknown symbol", err.Error())
	}
}

func TestNewIntegrityError(t *testing.T) {
	t.Parallel()
	err := NewIntegrityError("checkpoint.weights.len", "1024", "1023")
	if !errors.Is(err, ErrIntegrity) {
		t.Errorf("NewIntegrityError must wrap ErrIntegrity")
	}
	for _, s := range []string{"checkpoint.weights.len", "1024", "1023"} {
		if !strings.Contains(err.Error(), s) {
			t.Errorf("error %q missing required field %q", err.Error(), s)
		}
	}
}

// TestForbiddenPhrasesC32 enforces RULES.md §C32.1: error messages must not
// contain these vague phrases. The test sweeps every helper output so a
// future helper added without specificity blows up here first.
func TestForbiddenPhrasesC32(t *testing.T) {
	t.Parallel()
	forbidden := []string{
		"something went wrong",
		"internal error",
		"unknown error",
	}
	samples := []error{
		Newf(ErrUserConfig, "field %s constraint %s got %d", "size", "positive", -1),
		Wrap(ErrIO, errors.New("EOF"), "reading %s", "weights.json"),
		NewSizeError("Input", 0, "positive"),
		NewActivationError("XELU"),
		NewIntegrityError("ckpt.crc", "0xDEAD", "0xBEEF"),
	}
	for _, err := range samples {
		msg := strings.ToLower(err.Error())
		for _, phrase := range forbidden {
			if strings.Contains(msg, phrase) {
				t.Errorf("error %q contains forbidden phrase %q (C32 §1)", err.Error(), phrase)
			}
		}
		// "invalid input" without a specifier is forbidden — accept the
		// phrase only when followed by an identifier-like token.
		if _, after, found := strings.Cut(msg, "invalid input"); found {
			tail := strings.TrimSpace(after)
			if tail == "" {
				t.Errorf("error %q uses bare 'invalid input' (C32 §1)", err.Error())
			}
		}
	}
}

func TestLocationHintReturnsCallerSite(t *testing.T) {
	t.Parallel()
	hint := LocationHint(0)
	if hint == "" {
		t.Fatal("LocationHint(0) returned empty; runtime.Caller must succeed in tests")
	}
	if !strings.Contains(hint, "errors_test.go") {
		t.Errorf("LocationHint(0) = %q; want a path containing errors_test.go", hint)
	}
	if !strings.Contains(hint, ":") {
		t.Errorf("LocationHint(0) = %q; want a file:line format", hint)
	}
}

func TestLocationHintHandlesUnreachableSkip(t *testing.T) {
	t.Parallel()
	// A skip larger than the call stack must return the empty string,
	// not panic. Use a very large skip to outrun any reasonable depth.
	if got := LocationHint(1 << 20); got != "" {
		t.Errorf("LocationHint(huge) = %q; want empty", got)
	}
}

func TestTrimToPackagePathFallbacks(t *testing.T) {
	t.Parallel()
	cases := []struct {
		in, want string
	}{
		{"/home/user/proj/gonn/pkg/utils/errors.go", "pkg/utils/errors.go"},
		{`C:\Projects\src\github.com\teratron\gonn\pkg\utils\errors.go`, `pkg\utils\errors.go`},
		{"/tmp/external/file.go", "file.go"},
		{`D:\external\file.go`, "file.go"},
		{"file.go", "file.go"},
	}
	for _, c := range cases {
		got := trimToPackagePath(c.in)
		if got != c.want {
			t.Errorf("trimToPackagePath(%q) = %q; want %q", c.in, got, c.want)
		}
	}
}

func TestIndexLastEdgeCases(t *testing.T) {
	t.Parallel()
	if got := indexLast("", "x"); got != -1 {
		t.Errorf("indexLast(empty, x) = %d; want -1", got)
	}
	if got := indexLast("abc", ""); got != -1 {
		t.Errorf("indexLast(abc, empty) = %d; want -1", got)
	}
	if got := indexLast("abc", "abcd"); got != -1 {
		t.Errorf("indexLast(short, longer) = %d; want -1", got)
	}
	if got := indexLast("aXbXc", "X"); got != 3 {
		t.Errorf("indexLast(aXbXc, X) = %d; want 3", got)
	}
}

// TestExampleCallSite documents the canonical call shape so future
// reviewers can copy-paste without re-deriving it from the spec.
func TestExampleCallSite(t *testing.T) {
	t.Parallel()
	produce := func(size int) error {
		if size <= 0 {
			return NewSizeError("Input", size, "positive")
		}
		return nil
	}
	err := produce(0)
	if !errors.Is(err, ErrUserConfig) {
		t.Fatalf("produce(0) must yield ErrUserConfig, got %v", err)
	}
	// Ensure we also keep the message shape for human readers.
	if want := "size must be positive"; !strings.Contains(err.Error(), want) {
		t.Errorf("error message %q missing %q", err.Error(), want)
	}
	// fmt.Errorf wrapping invariant: %w returns the inner sentinel via Unwrap.
	var inner = errors.Unwrap(fmt.Errorf("ctx: %w", err))
	if !errors.Is(inner, ErrUserConfig) {
		t.Errorf("re-wrapped chain must still route to ErrUserConfig")
	}
}
