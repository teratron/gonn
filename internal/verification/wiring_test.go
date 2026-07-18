package verification

import (
	"math"
	"testing"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/nn"
	"github.com/teratron/gonn/pkg/optimizer"
	"github.com/teratron/gonn/pkg/regularizer"
)

// TestL2ShrinksWeights guards the audit finding that L1/L2 never reached the
// gradient: with a large λ and zero-target data, weight decay must pull the L2
// network's weights measurably smaller than an unregularized twin.
func TestL2ShrinksWeights(t *testing.T) {
	mk := func(reg regularizer.Regularizer[float64]) *nn.NN[float64] {
		opts := []nn.Option[float64]{
			nn.WithInput[float64](2),
			nn.WithHiddenLayer[float64](4, activation.SIGMOID),
			nn.WithOutput[float64](1, activation.SIGMOID),
			nn.WithLoss[float64](loss.MSE),
			nn.WithLearningRate[float64](0.1),
			nn.WithMaxIterations[float64](200),
			nn.WithLossLimit[float64](-1), // never early-stop
			nn.WithWeightInitSeed[float64](42),
		}
		if reg != nil {
			opts = append(opts, nn.WithRegularizer[float64](reg))
		}
		return nn.MustNew(opts...)
	}
	l2norm := func(w []float64) float64 {
		var s float64
		for _, v := range w {
			s += v * v
		}
		return math.Sqrt(s)
	}
	plain := mk(nil)
	reg := mk(regularizer.NewL2[float64](0.5))
	_, _, _ = plain.Fit(xorSamples())
	_, _, _ = reg.Fit(xorSamples())

	plainNorm := l2norm(plain.FlatWeights())
	regNorm := l2norm(reg.FlatWeights())
	if regNorm >= plainNorm {
		t.Errorf("L2 did not shrink weights: |w| with L2 = %.4f, without = %.4f", regNorm, plainNorm)
	}
}

// TestSchedulerDrivesOptimizer guards the audit finding that WithScheduler was
// decorative: after training, a bound StepLR must have decayed the optimizer's
// effective learning rate.
func TestSchedulerDrivesOptimizer(t *testing.T) {
	sgd := optimizer.NewSGD[float64](0.5)
	n := nn.MustNew(
		nn.WithInput[float64](2),
		nn.WithHiddenLayer[float64](4, activation.SIGMOID),
		nn.WithOutput[float64](1, activation.SIGMOID),
		nn.WithOptimizer[float64](sgd),
		nn.WithScheduler[float64](optimizer.NewStepLR[float64](0.5, 1, 0.1)),
		nn.WithMaxIterations[float64](3),
		nn.WithWeightInitSeed[float64](42),
	)
	if _, _, err := n.Fit(xorSamples()); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if got := sgd.LearningRate(); got >= 0.5 {
		t.Errorf("optimizer LR after 3 epochs = %.5f, want decayed below 0.5 (scheduler not wired)", got)
	}
}
