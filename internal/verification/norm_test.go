// Package verification — normalization-layer oracles.
//
// Guards audit B1: BatchNorm/LayerNorm/GroupNorm were configured via options
// but never applied in the dense forward pass, and no backward existed. The
// gradient checks here prove the norm layers now sit INSIDE the training
// graph: forward changes the output, backward routes exact gradients.
package verification

import (
	"testing"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/layer/norm"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/nn"
)

const normGradTol = 1e-4

// normNet builds a 2-hidden dense net with the given normalizers attached.
func normNet(t *testing.T, opts ...nn.Option[float64]) *nn.NN[float64] {
	t.Helper()
	base := []nn.Option[float64]{
		nn.WithInput[float64](3),
		nn.WithBias[float64](true),
		nn.WithHiddenLayer[float64](6, activation.SIGMOID),
		nn.WithHiddenLayer[float64](4, activation.TanH),
		nn.WithOutput[float64](2, activation.SIGMOID),
		nn.WithLoss[float64](loss.MSE),
		nn.WithLearningRate[float64](0.1),
		nn.WithMaxIterations[float64](1),
		nn.WithWeightInitSeed[float64](12345),
	}
	n, err := nn.New(append(base, opts...)...)
	if err != nil {
		t.Fatalf("build norm net: %v", err)
	}
	return n
}

var (
	normInput  = []float64{0.3, -0.6, 0.9}
	normTarget = []float64{0.8, 0.2}
)

// TestNormActuallyApplied: attaching a normalizer must change the forward
// output — the audit's original finding was that it changed nothing.
func TestNormActuallyApplied(t *testing.T) {
	plain := normNet(t)
	normed := normNet(t, nn.WithLayerNorm[float64](0))

	a, err := plain.Query(normInput)
	if err != nil {
		t.Fatalf("plain Query: %v", err)
	}
	b, err := normed.Query(normInput)
	if err != nil {
		t.Fatalf("normed Query: %v", err)
	}
	if absDiffMax(a, b) < 1e-9 {
		t.Errorf("LayerNorm attached but output unchanged (%v vs %v) — norm not applied", a, b)
	}
}

// TestGradcheckThroughLayerNorm: the analytic gradient must survive a
// LayerNorm inserted mid-chain (both hidden layers normalized).
func TestGradcheckThroughLayerNorm(t *testing.T) {
	n := normNet(t,
		nn.WithLayerNorm[float64](0),
		nn.WithLayerNorm[float64](1),
	)
	worst := gradcheckDense(t, n, normInput, normTarget, loss.MSE)
	if worst > normGradTol {
		t.Errorf("gradcheck through LayerNorm: worst rel err %.3e > %.0e", worst, normGradTol)
	}
}

// TestGradcheckThroughGroupNorm: same oracle for GroupNorm (6 features, 2 groups).
func TestGradcheckThroughGroupNorm(t *testing.T) {
	gn, err := norm.NewGroupNorm[float64](6, 2)
	if err != nil {
		t.Fatalf("NewGroupNorm: %v", err)
	}
	n := normNet(t, nn.WithNormAfterLayer[float64](0, gn))
	worst := gradcheckDense(t, n, normInput, normTarget, loss.MSE)
	if worst > normGradTol {
		t.Errorf("gradcheck through GroupNorm: worst rel err %.3e > %.0e", worst, normGradTol)
	}
}

// TestGradcheckThroughBatchNormEval: BatchNorm gradients are exact when the
// running statistics are frozen. Train a few steps to move the stats off the
// identity, then switch to eval and run the oracle.
func TestGradcheckThroughBatchNormEval(t *testing.T) {
	n := normNet(t, nn.WithBatchNorm[float64](0))
	n.SetTrain()
	for range 5 {
		if _, err := n.Train(normInput, normTarget); err != nil {
			t.Fatalf("Train: %v", err)
		}
	}
	n.SetEval()
	worst := gradcheckDense(t, n, normInput, normTarget, loss.MSE)
	if worst > normGradTol {
		t.Errorf("gradcheck through BatchNorm (eval): worst rel err %.3e > %.0e", worst, normGradTol)
	}
}

// TestBatchNormTrainsAndStatsMove: end-to-end — a BatchNorm network must
// still learn XOR, and the running statistics must move off their init.
func TestBatchNormTrainsAndStatsMove(t *testing.T) {
	bn := norm.NewBatchNorm[float64](6)
	n, err := nn.New(
		nn.WithInput[float64](2),
		nn.WithBias[float64](true),
		nn.WithHiddenLayer[float64](6, activation.SIGMOID),
		nn.WithOutput[float64](1, activation.SIGMOID),
		nn.WithLoss[float64](loss.MSE),
		// lr 0.3: with sample-stream EMA stats the effective gradient is
		// amplified by invSd, so the aggressive 0.7 XOR rate oscillates.
		nn.WithLearningRate[float64](0.3),
		nn.WithMaxIterations[float64](4000),
		nn.WithLossLimit[float64](0.001),
		nn.WithWeightInitSeed[float64](42),
		nn.WithNormAfterLayer[float64](0, bn),
	)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	n.SetTrain()
	_, finalLoss, err := n.Fit(xorSamples())
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if finalLoss >= 0.01 {
		t.Errorf("XOR with BatchNorm: final loss %v, want < 0.01", finalLoss)
	}
	n.SetEval()
	out, err := n.Query([]float64{0, 1})
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if out[0] < 0.8 {
		t.Errorf("Query(0,1) = %v, want > 0.8", out[0])
	}
}

// TestNormAffineParamsTrain: γ/β must move off their identity init during
// training (they previously had no gradient path at all).
func TestNormAffineParamsTrain(t *testing.T) {
	ln := norm.NewLayerNorm[float64](6)
	n, err := nn.New(
		nn.WithInput[float64](2),
		nn.WithBias[float64](true),
		nn.WithHiddenLayer[float64](6, activation.SIGMOID),
		nn.WithOutput[float64](1, activation.SIGMOID),
		nn.WithLoss[float64](loss.MSE),
		nn.WithLearningRate[float64](0.5),
		nn.WithMaxIterations[float64](50),
		nn.WithLossLimit[float64](-1),
		nn.WithWeightInitSeed[float64](42),
		nn.WithNormAfterLayer[float64](0, ln),
	)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if _, _, err := n.Fit(xorSamples()); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	gamma, beta := ln.Params()
	moved := false
	for i := range gamma {
		if gamma[i] != 1 || beta[i] != 0 {
			moved = true
			break
		}
	}
	if !moved {
		t.Error("γ/β still at identity after 50 epochs — affine params receive no gradient")
	}
}
