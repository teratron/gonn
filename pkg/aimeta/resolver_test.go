package aimeta_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/teratron/gonn/pkg/aimeta"
)

func TestResolver(t *testing.T) {
	t.Parallel()

	t.Run("fixLabel", func(t *testing.T) {
		t.Parallel()
		dir := t.TempDir()
		src := "package p\n\n// Foo does foo.\n//\n// AI-meta:\n//   - Purpose: test.\n//   - Stability: Stable.\nfunc Foo() {}\n"
		path := filepath.Join(dir, "foo.go")
		if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}

		fixes, err := aimeta.Resolve(dir, aimeta.Options{})
		if err != nil {
			t.Fatalf("Resolve: %v", err)
		}

		hasLabel := false
		for _, f := range fixes {
			if f.Rule == aimeta.RuleResolveLabel {
				hasLabel = true
			}
		}
		if !hasLabel {
			t.Errorf("expected RESOLVE-LABEL fix, got %v", fixes)
		}

		// After resolve the file must lint clean on LABEL.
		viols, err := aimeta.Check(dir, aimeta.Options{})
		if err != nil {
			t.Fatalf("Check after resolve: %v", err)
		}
		for _, v := range viols {
			if v.Rule == aimeta.RuleLabel {
				t.Errorf("LABEL violation persists after resolve: %v", v)
			}
		}
	})

	t.Run("fixIndent", func(t *testing.T) {
		t.Parallel()
		dir := t.TempDir()
		// Field line has "// - Purpose:" (one space instead of two-space indent).
		src := "package p\n\n// Foo does foo.\n//\n// AI-Meta:\n// - Purpose: test.\n//   - Stability: Stable.\nfunc Foo() {}\n"
		path := filepath.Join(dir, "foo.go")
		if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}

		fixes, err := aimeta.Resolve(dir, aimeta.Options{})
		if err != nil {
			t.Fatalf("Resolve: %v", err)
		}
		hasIndent := false
		for _, f := range fixes {
			if f.Rule == aimeta.RuleResolveIndent {
				hasIndent = true
			}
		}
		if !hasIndent {
			t.Errorf("expected RESOLVE-INDENT fix, got %v", fixes)
		}

		// Verify the written content is corrected.
		data, _ := os.ReadFile(path)
		if !strings.Contains(string(data), "//   - Purpose:") {
			t.Errorf("fixed file does not contain '//   - Purpose:'; got:\n%s", data)
		}
	})

	t.Run("fixLast", func(t *testing.T) {
		t.Parallel()
		dir := t.TempDir()
		// File missing terminal newline.
		src := "package p\n\nfunc Bar() {}"
		path := filepath.Join(dir, "bar.go")
		if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}

		fixes, err := aimeta.Resolve(dir, aimeta.Options{})
		if err != nil {
			t.Fatalf("Resolve: %v", err)
		}
		hasLast := false
		for _, f := range fixes {
			if f.Rule == aimeta.RuleResolveLast {
				hasLast = true
			}
		}
		if !hasLast {
			t.Errorf("expected RESOLVE-LAST fix, got %v", fixes)
		}
		data, _ := os.ReadFile(path)
		if len(data) == 0 || data[len(data)-1] != '\n' {
			t.Errorf("file does not end with newline after resolve")
		}
	})

	t.Run("noFix_alreadyClean", func(t *testing.T) {
		t.Parallel()
		dir := t.TempDir()
		src := "package p\n\n// Foo is good.\n//\n// AI-Meta:\n//   - Purpose: test.\n//   - Stability: Stable.\nfunc Foo() {}\n"
		path := filepath.Join(dir, "foo.go")
		if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
		fixes, err := aimeta.Resolve(dir, aimeta.Options{})
		if err != nil {
			t.Fatalf("Resolve: %v", err)
		}
		if len(fixes) != 0 {
			t.Errorf("expected no fixes on clean file, got %v", fixes)
		}
	})
}

// TestResolveRoundTrip verifies that resolve → re-lint produces 0 LABEL/INDENT violations.
func TestResolveRoundTrip(t *testing.T) {
	t.Parallel()
	dir := filepath.Join("testdata", "violating_pkg")

	_, err := aimeta.Resolve(dir, aimeta.Options{})
	if err != nil {
		t.Fatalf("Resolve violating_pkg: %v", err)
	}

	viols, err := aimeta.Check(dir, aimeta.Options{})
	if err != nil {
		t.Fatalf("Check after resolve: %v", err)
	}
	for _, v := range viols {
		if v.Rule == aimeta.RuleLabel || v.Rule == aimeta.RuleIndent {
			t.Errorf("fixable violation persists after resolve: [%s] %s", v.Rule, v.Message)
		}
	}
}
