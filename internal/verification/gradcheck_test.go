package verification

import (
	"fmt"
	"testing"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
)

// gradTolerance is the worst acceptable relative error between the analytic
// gradient (AppendFlatGradients) and central differences, in float64.
const gradTolerance = 1e-4

// TestGradcheckDenseMatrix is the primary oracle for Phase 1. It sweeps
// topologies × activations for the MSE loss (element-wise, always defined) and
// asserts backprop matches central differences.
//
// Against UNFIXED code this FAILS for every multi-hidden or multi-output case:
// CalculateMisses drops the downstream activation derivative, so hidden-layer
// gradients are off by the missing σ′ factors (audit block A1). Single-hidden
// single-output MSE is the one benign case and stays green.
func TestGradcheckDenseMatrix(t *testing.T) {
	acts := []activation.Type{
		activation.SIGMOID, activation.TanH, activation.ReLU,
		activation.LeakyReLU, activation.ELU, activation.SELU,
		activation.SWISH, activation.ELISH,
	}
	topos := []struct {
		name    string
		hidden  []uint
		outputs uint
	}{
		{"1h-1out", []uint{4}, 1},
		{"1h-3out", []uint{4}, 3},
		{"2h-1out", []uint{4, 3}, 1},
		{"2h-3out", []uint{5, 4}, 3},
		{"3h-2out", []uint{4, 4, 3}, 2},
	}
	input := []float64{0.3, -0.6, 0.15}
	for _, topo := range topos {
		for _, act := range acts {
			name := fmt.Sprintf("%s/%s", topo.name, act.String())
			t.Run(name, func(t *testing.T) {
				spec := netSpec{
					name:    name,
					hidden:  topo.hidden,
					act:     act,
					outAct:  activation.SIGMOID,
					outputs: topo.outputs,
					lossT:   loss.MSE,
				}
				n := buildDense(t, spec, uint(len(input)))
				target := make([]float64, topo.outputs)
				for i := range target {
					target[i] = float64(i%2) * 0.8
				}
				worst := gradcheckDense(t, n, input, target, loss.MSE)
				if worst > gradTolerance {
					t.Errorf("worst relative gradient error %.4e exceeds %.0e "+
						"(backprop drops downstream σ′ — audit A1)", worst, gradTolerance)
				}
			})
		}
	}
}

// TestGradcheckLinearOutput checks a Linear output head (regression), where the
// output σ′ = 1 so the single-hidden case masks less of the bug.
func TestGradcheckLinearOutput(t *testing.T) {
	input := []float64{0.2, 0.5}
	spec := netSpec{
		name:    "regression",
		hidden:  []uint{5, 4},
		act:     activation.ReLU,
		outAct:  activation.Linear,
		outputs: 2,
		lossT:   loss.MSE,
	}
	n := buildDense(t, spec, uint(len(input)))
	worst := gradcheckDense(t, n, input, []float64{0.7, -0.3}, loss.MSE)
	if worst > gradTolerance {
		t.Errorf("linear-output gradcheck worst=%.4e > %.0e", worst, gradTolerance)
	}
}
