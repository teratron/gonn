package verification

import (
	"math"
	"testing"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/nn"
)

// TestTanhDerivativeExact pins the pre-activation tanh derivative (audit A5:
// the engine feeds preact z, so f′ must be 1 − tanh(z)², recomputed from z).
func TestTanhDerivativeExact(t *testing.T) {
	for _, z := range []float64{-2, -0.5, 0.5, 1, 2, 3} {
		got := activation.Derivative(z, activation.TanH)
		want := 1 - math.Tanh(z)*math.Tanh(z)
		if math.Abs(got-want) > 1e-12 {
			t.Errorf("tanh'(%.1f)=%.6f want %.6f", z, got, want)
		}
	}
}

// TestCCEActuallyTrains guards audit A3: CROSS_ENTROPY used to return a
// constant 0, so Fit "converged" at epoch 1 without learning. It must now
// drive a softmax head to the correct one-hot class.
func TestCCEActuallyTrains(t *testing.T) {
	samples := []nn.Sample[float64]{
		{Input: []float64{0, 0}, Target: []float64{1, 0}},
		{Input: []float64{0, 1}, Target: []float64{0, 1}},
		{Input: []float64{1, 0}, Target: []float64{0, 1}},
		{Input: []float64{1, 1}, Target: []float64{1, 0}},
	}
	mk := func() *nn.NN[float64] {
		return nn.MustNew(
			nn.WithInput[float64](2),
			nn.WithHiddenLayer[float64](8, activation.SIGMOID),
			nn.WithOutput[float64](2, activation.SOFTMAX),
			nn.WithLoss[float64](loss.CCE),
			nn.WithLearningRate[float64](0.5),
			nn.WithMaxIterations[float64](8000),
			nn.WithLossLimit[float64](1e-3),
			nn.WithWeightInitSeed[float64](7),
		)
	}
	initial := meanLoss(t, mk(), samples, loss.CCE)

	n := mk()
	epochs, finalLoss, err := n.Fit(samples)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if epochs <= 1 {
		t.Fatalf("CCE stopped at epoch %d — false convergence (loss==0 bug)", epochs)
	}
	if finalLoss > 0.3*initial {
		t.Errorf("CCE did not learn: final %.4f vs initial %.4f", finalLoss, initial)
	}
	// XOR-parity classification must be correct after training.
	for _, s := range samples {
		out, _ := n.Query(s.Input)
		pred := 0
		if out[1] > out[0] {
			pred = 1
		}
		want := 0
		if s.Target[1] > s.Target[0] {
			want = 1
		}
		if pred != want {
			t.Errorf("input %v: predicted class %d want %d (%v)", s.Input, pred, want, out)
		}
	}
}

// meanLoss returns the mean per-sample loss of an untrained (or trained)
// network — the pre-training baseline the learning tests improve against.
func meanLoss(t *testing.T, n *nn.NN[float64], samples []nn.Sample[float64], mode loss.Type) float64 {
	t.Helper()
	var total float64
	for _, s := range samples {
		l, err := n.Verify(s.Input, s.Target)
		if err != nil {
			t.Fatalf("Verify: %v", err)
		}
		total += l
	}
	return total / float64(len(samples))
}

// TestSoftmaxSumsToOne guards audit A6: SOFTMAX was a per-element sigmoid whose
// outputs did not form a distribution. A true softmax head must sum to 1.
func TestSoftmaxSumsToOne(t *testing.T) {
	n := nn.MustNew(
		nn.WithInput[float64](3),
		nn.WithHiddenLayer[float64](5, activation.ReLU),
		nn.WithOutput[float64](4, activation.SOFTMAX),
		nn.WithLoss[float64](loss.CCE),
		nn.WithLearningRate[float64](0.05),
		nn.WithMaxIterations[float64](1),
		nn.WithWeightInitSeed[float64](3),
	)
	out, err := n.Query([]float64{0.4, -0.2, 0.9})
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	sum := 0.0
	for _, v := range out {
		if v < 0 {
			t.Errorf("softmax produced negative probability %.4f", v)
		}
		sum += v
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Errorf("softmax outputs sum to %.6f, want 1", sum)
	}
}

// TestSoftmaxRejectsNonCCE guards the compile-time contract: softmax is only
// well-defined with cross-entropy, so SOFTMAX+MSE must be an error, not a
// silently-wrong network.
func TestSoftmaxRejectsNonCCE(t *testing.T) {
	_, err := nn.New(
		nn.WithInput[float64](2),
		nn.WithHiddenLayer[float64](3, activation.ReLU),
		nn.WithOutput[float64](2, activation.SOFTMAX),
		nn.WithLoss[float64](loss.MSE),
	)
	if err == nil {
		t.Fatal("SOFTMAX+MSE compiled without error — should be rejected")
	}
}

// TestBCEPositiveAndTrains guards audit A4/A2: BCE reporting was negative
// garbage and the loss never reached gradients. It must now report a positive
// loss and train a sigmoid head.
func TestBCEPositiveAndTrains(t *testing.T) {
	mk := func() *nn.NN[float64] {
		return nn.MustNew(
			nn.WithInput[float64](2),
			nn.WithHiddenLayer[float64](6, activation.SIGMOID),
			nn.WithOutput[float64](1, activation.SIGMOID),
			nn.WithLoss[float64](loss.BCE),
			nn.WithLearningRate[float64](1.0),
			nn.WithMaxIterations[float64](8000),
			nn.WithLossLimit[float64](1e-2),
			nn.WithWeightInitSeed[float64](42),
		)
	}
	initial := meanLoss(t, mk(), xorSamples(), loss.BCE)
	if initial < 0 {
		t.Fatalf("BCE baseline loss %.4f is negative — reporting bug", initial)
	}

	n := mk()
	_, finalLoss, err := n.Fit(xorSamples())
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if finalLoss < 0 {
		t.Errorf("BCE loss %.4f is negative", finalLoss)
	}
	if finalLoss > 0.3*initial {
		t.Errorf("BCE did not learn: final %.4f vs initial %.4f", finalLoss, initial)
	}
}

// TestConvergenceMatrix asserts several activation/loss heads actually reach a
// low loss on XOR — the end-to-end proof that the corrected gradients learn.
func TestConvergenceMatrix(t *testing.T) {
	cases := []struct {
		name   string
		hidden activation.Type
		out    activation.Type
		lossT  loss.Type
		thresh float64
	}{
		{"sigmoid-mse", activation.SIGMOID, activation.SIGMOID, loss.MSE, 0.02},
		{"tanh-mse", activation.TanH, activation.SIGMOID, loss.MSE, 0.05},
		{"relu-mse", activation.ReLU, activation.SIGMOID, loss.MSE, 0.05},
		{"sigmoid-bce", activation.SIGMOID, activation.SIGMOID, loss.BCE, 0.10},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			n := nn.MustNew(
				nn.WithInput[float64](2),
				nn.WithHiddenLayer[float64](5, c.hidden),
				nn.WithOutput[float64](1, c.out),
				nn.WithLoss[float64](c.lossT),
				nn.WithLearningRate[float64](0.5),
				nn.WithMaxIterations[float64](5000),
				nn.WithLossLimit[float64](c.thresh/2),
				nn.WithWeightInitSeed[float64](42),
			)
			_, finalLoss, err := n.Fit(xorSamples())
			if err != nil {
				t.Fatalf("Fit: %v", err)
			}
			if finalLoss > c.thresh {
				t.Errorf("%s did not converge: final loss %.4f > %.4f", c.name, finalLoss, c.thresh)
			}
		})
	}
}
