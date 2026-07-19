package verification

import (
	"encoding/json"
	"math"
	"net/http"
	"testing"
	"time"

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

// TestVisSnapshotIsReal guards the audit finding that /v1/snapshot served a
// hardcoded stub: after Fit, the endpoint must report the real epoch count,
// a non-zero loss, real layer sizes, and hidden activations.
func TestVisSnapshotIsReal(t *testing.T) {
	n := nn.MustNew(
		nn.WithInput[float64](2),
		nn.WithHiddenLayer[float64](4, activation.SIGMOID),
		nn.WithOutput[float64](1, activation.SIGMOID),
		nn.WithLoss[float64](loss.MSE),
		nn.WithMaxIterations[float64](10),
		nn.WithLossLimit[float64](-1),
		nn.WithWeightInitSeed[float64](42),
		nn.WithVisualizationEndpoint[float64]("127.0.0.1:0"),
	)
	t.Cleanup(func() { _ = n.Close() })

	epochs, fitLoss, err := n.Fit(xorSamples())
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	addr := n.VisAddr()
	if addr == "" {
		t.Fatal("VisAddr is empty — visualization server did not start")
	}
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get("http://" + addr + "/v1/snapshot")
	if err != nil {
		t.Fatalf("GET /v1/snapshot: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /v1/snapshot status = %d, want 200", resp.StatusCode)
	}
	// Every response is wrapped in {"data": ..., "protocol_version": ...}.
	var body struct {
		Data struct {
			Layers []struct {
				Type string `json:"type"`
				Size int    `json:"size"`
			} `json:"layers"`
			Control     string      `json:"control"`
			Activations [][]float64 `json:"activations"`
			Loss        float64     `json:"loss"`
			Epoch       uint64      `json:"epoch"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode snapshot: %v", err)
	}
	snap := body.Data
	if snap.Epoch != uint64(epochs) {
		t.Errorf("snapshot epoch = %d, want %d", snap.Epoch, epochs)
	}
	if snap.Loss <= 0 || math.Abs(snap.Loss-float64(fitLoss)) > 1e-9 {
		t.Errorf("snapshot loss = %v, want Fit's final loss %v", snap.Loss, fitLoss)
	}
	if snap.Control != "idle" {
		t.Errorf("snapshot control = %q, want \"idle\" after Fit returns", snap.Control)
	}
	wantSizes := []int{2, 4, 1}
	if len(snap.Layers) != len(wantSizes) {
		t.Fatalf("snapshot has %d layers, want %d", len(snap.Layers), len(wantSizes))
	}
	for i, w := range wantSizes {
		if snap.Layers[i].Size != w {
			t.Errorf("layer %d size = %d, want %d", i, snap.Layers[i].Size, w)
		}
	}
	if len(snap.Activations) != 1 || len(snap.Activations[0]) != 4 {
		t.Errorf("activations shape = %v, want 1 hidden layer of 4", snap.Activations)
	}
}

// TestNegativeLearningRateRejected guards the audit finding that
// WithLearningRate(-0.5) silently became the 0.3 default: a negative rate
// must now surface as a compile error, while a negative LossLimit stays the
// documented "never stop early" sentinel.
func TestNegativeLearningRateRejected(t *testing.T) {
	_, err := nn.New(
		nn.WithInput[float64](2),
		nn.WithHiddenLayer[float64](4, activation.SIGMOID),
		nn.WithOutput[float64](1, activation.SIGMOID),
		nn.WithLearningRate[float64](-0.5),
	)
	if err == nil {
		t.Error("compile accepted a negative learning rate")
	}

	n := nn.MustNew(
		nn.WithInput[float64](2),
		nn.WithHiddenLayer[float64](4, activation.SIGMOID),
		nn.WithOutput[float64](1, activation.SIGMOID),
		nn.WithMaxIterations[float64](30),
		nn.WithLossLimit[float64](-1), // disable early stopping
		nn.WithWeightInitSeed[float64](42),
	)
	epochs, _, err := n.Fit(xorSamples())
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if epochs != 30 {
		t.Errorf("negative LossLimit: ran %d epochs, want all 30 (early stop must be disabled)", epochs)
	}
}

// TestDropoutGatesForward guards the audit finding that the dropout mask ran
// AFTER the forward pass and never influenced the output. With the honest
// in-graph mask, the training-time loss is stochastic (different masks per
// step), while inference (Query) stays deterministic and unmasked.
func TestDropoutGatesForward(t *testing.T) {
	n := nn.MustNew(
		nn.WithInput[float64](2),
		nn.WithHiddenLayer[float64](16, activation.SIGMOID),
		nn.WithOutput[float64](1, activation.SIGMOID),
		nn.WithLoss[float64](loss.MSE),
		// Near-zero LR: weight drift is O(1e-12) per step, so any material
		// loss variation across steps can only come from the dropout mask
		// inside the forward pass. (Exactly 0 would be replaced by the 0.3
		// default in applyDefaults.)
		nn.WithLearningRate[float64](1e-12),
		nn.WithMaxIterations[float64](1),
		nn.WithWeightInitSeed[float64](42),
		nn.WithRegularizer[float64](regularizer.NewDropoutSeeded[float64](0.5, 7)),
	)
	in, tgt := []float64{1, 0}, []float64{1}

	lo, hi := math.Inf(1), math.Inf(-1)
	for range 8 {
		l, err := n.Train(in, tgt)
		if err != nil {
			t.Fatalf("Train: %v", err)
		}
		lo = math.Min(lo, float64(l))
		hi = math.Max(hi, float64(l))
	}
	if hi-lo < 1e-3 {
		t.Errorf("training loss spread %.3e across 8 masked steps — dropout does not gate the forward pass", hi-lo)
	}

	q1, err := n.Query(in)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	q2, _ := n.Query(in)
	if q1[0] != q2[0] {
		t.Errorf("inference is stochastic (%v vs %v) — dropout leaked into Query", q1[0], q2[0])
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
