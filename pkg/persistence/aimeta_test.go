package persistence

import (
	"testing"

	"github.com/teratron/gonn/pkg/aimeta"
)

// TestAIMetaCompliance verifies that all AI-Meta blocks present in the package
// satisfy the grammar and vocabulary rules (l2-aimeta-linter §5.4, rollout phase 5).
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
