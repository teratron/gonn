package aimeta_test

import (
	"testing"

	"github.com/teratron/gonn/pkg/aimeta"
)

func TestGoldenFiles(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		dir       string
		wantRules []string // expected rule codes, one per violation
	}{
		{
			name:      "good — no violations",
			dir:       "testdata/good",
			wantRules: nil,
		},
		{
			name:      "bad_vocab — unknown field Author",
			dir:       "testdata/bad_vocab",
			wantRules: []string{aimeta.RuleVocab},
		},
		{
			name:      "bad_cap — block exceeds 12 lines",
			dir:       "testdata/bad_cap",
			wantRules: []string{aimeta.RuleCap},
		},
		{
			name:      "bad_multi — nested bullet value",
			dir:       "testdata/bad_multi",
			wantRules: []string{aimeta.RuleMulti, aimeta.RuleMulti},
		},
		{
			name:      "bad_last — block not last in comment",
			dir:       "testdata/bad_last",
			wantRules: []string{aimeta.RuleLast},
		},
		{
			name:      "bad_artifact — INV-N and .design/ reference in one field",
			dir:       "testdata/bad_artifact",
			wantRules: []string{aimeta.RuleArtifact},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			violations, err := aimeta.Check(tc.dir, aimeta.Options{})
			if err != nil {
				t.Fatalf("Check(%q): unexpected error: %v", tc.dir, err)
			}

			if len(violations) != len(tc.wantRules) {
				t.Errorf("got %d violations, want %d", len(violations), len(tc.wantRules))
				for i, v := range violations {
					t.Logf("  [%d] %s: [%s] %s", i, v.Symbol, v.Rule, v.Message)
				}
				return
			}

			for i, want := range tc.wantRules {
				if violations[i].Rule != want {
					t.Errorf("violation[%d].Rule = %q, want %q (msg: %s)",
						i, violations[i].Rule, want, violations[i].Message)
				}
			}
		})
	}
}

func TestParseBlockClean(t *testing.T) {
	t.Parallel()
	// Stability without trailing period so the raw stored value is "Stable".
	raw := "AI-Meta:\n  - Purpose: Foo does bar.\n  - Stability: Stable"
	block, viols := aimeta.ParseBlock(raw)
	if len(viols) != 0 {
		t.Fatalf("ParseBlock clean: expected 0 violations, got %d: %v", len(viols), viols)
	}
	if block.Fields["Purpose"] != "Foo does bar." {
		t.Errorf("Purpose = %q, want %q", block.Fields["Purpose"], "Foo does bar.")
	}
	if block.Fields["Stability"] != "Stable" {
		t.Errorf("Stability = %q, want %q", block.Fields["Stability"], "Stable")
	}
}

func TestParseBlockBadLabel(t *testing.T) {
	t.Parallel()
	_, viols := aimeta.ParseBlock("ai-meta:\n  - Purpose: x.")
	if len(viols) != 1 || viols[0].Rule != aimeta.RuleLabel {
		t.Errorf("expected 1 LABEL violation, got %v", viols)
	}
}

func TestParseBlockBadEnum(t *testing.T) {
	t.Parallel()
	_, viols := aimeta.ParseBlock("AI-Meta:\n  - Stability: Unknown.")
	found := false
	for _, v := range viols {
		if v.Rule == aimeta.RuleEnum {
			found = true
		}
	}
	if !found {
		t.Errorf("expected ENUM violation, got %v", viols)
	}
}

func TestParseBlockConcurrencyWithClarifier(t *testing.T) {
	t.Parallel()
	// Concurrency value with "; clarifier" must still be valid.
	_, viols := aimeta.ParseBlock(
		"AI-Meta:\n  - Purpose: x.\n  - Concurrency: NotSafe; must not share rng across goroutines.")
	for _, v := range viols {
		if v.Rule == aimeta.RuleEnum {
			t.Errorf("unexpected ENUM violation for Concurrency with clarifier: %s", v.Message)
		}
	}
}

func TestExtractBlockAbsent(t *testing.T) {
	t.Parallel()
	// ExtractBlock on nil returns (Block{}, false).
	_, ok := aimeta.ExtractBlock(nil)
	if ok {
		t.Error("ExtractBlock(nil) should return ok=false")
	}
}
