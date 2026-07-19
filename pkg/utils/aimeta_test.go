package utils

import (
	"testing"

	"github.com/teratron/gonn/pkg/aimeta"
)

// TestAIMetaCompliance is the first per-package AI-Meta compliance hook per
// l2-aimeta-linter §5.4 and l2-ai-doc-metadata §8 (Phase 2 rollout).
// It becomes the template for rollout phases 3-5 (Phase 16+).
func TestAIMetaCompliance(t *testing.T) {
	t.Parallel()
	violations, err := aimeta.Check(".", aimeta.Options{IncludeInternal: true})
	if err != nil {
		t.Fatalf("aimeta.Check: %v", err)
	}
	for _, v := range violations {
		t.Errorf("%s:%d [%s] %s: %s",
			v.Pos.Filename, v.Pos.Line, v.Rule, v.Symbol, v.Message)
	}
}
