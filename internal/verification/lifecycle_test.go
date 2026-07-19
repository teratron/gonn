package verification

import (
	"math"
	"sync"
	"testing"
	"time"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/network"
	"github.com/teratron/gonn/pkg/nn"
	"github.com/teratron/gonn/pkg/optimizer"
)

func trainedXOR(t *testing.T, extra ...nn.Option[float64]) *nn.NN[float64] {
	t.Helper()
	opts := append([]nn.Option[float64]{
		nn.WithInput[float64](2),
		nn.WithHiddenLayer[float64](4, activation.SIGMOID),
		nn.WithOutput[float64](1, activation.SIGMOID),
		nn.WithLoss[float64](loss.MSE),
		nn.WithLearningRate[float64](0.5),
		nn.WithMaxIterations[float64](2000),
		nn.WithLossLimit[float64](1e-3),
		nn.WithWeightInitSeed[float64](42),
	}, extra...)
	n := nn.MustNew(opts...)
	if _, _, err := n.Fit(xorSamples()); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	return n
}

// TestConcurrentQueryRaceFree guards audit D1: Query on a pure dense network is
// now a stateless read and must return identical answers under a concurrent
// storm (run with -race to also catch the data race).
func TestConcurrentQueryRaceFree(t *testing.T) {
	n := trainedXOR(t)
	inputs := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	refs := make([][]float64, len(inputs))
	for i, in := range inputs {
		out, err := n.Query(in)
		if err != nil {
			t.Fatalf("Query: %v", err)
		}
		refs[i] = out
	}

	var wg sync.WaitGroup
	var wrong int64
	var mu sync.Mutex
	for range 16 {
		wg.Go(func() {
			for iter := range 400 {
				i := iter % len(inputs)
				out, err := n.Query(inputs[i])
				if err != nil || math.Abs(out[0]-refs[i][0]) > 1e-12 {
					mu.Lock()
					wrong++
					mu.Unlock()
				}
			}
		})
	}
	wg.Wait()
	if wrong != 0 {
		t.Errorf("%d concurrent Query answers were wrong — not read-safe", wrong)
	}
}

// TestStopThenFitResumes guards audit D5: after Stop() the network must be
// reusable, not permanently sterile.
func TestStopThenFitResumes(t *testing.T) {
	n := nn.MustNew(
		nn.WithInput[float64](2),
		nn.WithHiddenLayer[float64](4, activation.SIGMOID),
		nn.WithOutput[float64](1, activation.SIGMOID),
		nn.WithLoss[float64](loss.MSE),
		nn.WithLearningRate[float64](0.5),
		nn.WithMaxIterations[float64](100000),
		nn.WithLossLimit[float64](1e-12),
		nn.WithWeightInitSeed[float64](42),
	)
	done := make(chan struct{})
	go func() {
		_, _, _ = n.Fit(xorSamples())
		close(done)
	}()
	time.Sleep(30 * time.Millisecond)
	_ = n.Stop()
	<-done

	epochs, _, err := n.Fit(xorSamples())
	if err != nil {
		t.Fatalf("Fit after Stop: %v", err)
	}
	if epochs == 0 {
		t.Fatal("Fit after Stop ran 0 epochs — network sterilized by Stop")
	}
}

// TestAndTrainLearningRateApplies guards audit D7: WithLearningRate(0) passed to
// AndTrain must actually freeze the weights.
func TestAndTrainLearningRateApplies(t *testing.T) {
	n := trainedXOR(t, nn.WithMaxIterations[float64](5))
	before := n.FlatWeights()
	if _, _, err := n.AndTrain(xorSamples(), nn.WithLearningRate[float64](0)); err != nil {
		t.Fatalf("AndTrain: %v", err)
	}
	after := n.FlatWeights()
	if d := absDiffMax(before, after); d != 0 {
		t.Errorf("AndTrain(WithLearningRate(0)) changed weights by %.6g — LR override ignored", d)
	}
}

// TestTopologyPreservesWeights guards audit D4: growing a hidden layer must keep
// the learned weights of the surviving neurons.
func TestTopologyPreservesWeights(t *testing.T) {
	n := trainedXOR(t, nn.WithTopologyMode[float64](network.Dynamic), nn.WithMaxIterations[float64](500))
	before := n.Network.Hiddens[0].Cells()[0].Axons[0].Weight
	if err := n.AddNeuron(0, 1); err != nil {
		t.Fatalf("AddNeuron: %v", err)
	}
	after := n.Network.Hiddens[0].Cells()[0].Axons[0].Weight
	if before != after {
		t.Errorf("surviving weight changed on grow: %.6f -> %.6f", before, after)
	}
}

// TestAdamTopologyNoPanic guards audit D3: Fit after AddNeuron with Adam must not
// panic on stale moment buffers.
func TestAdamTopologyNoPanic(t *testing.T) {
	n := nn.MustNew(
		nn.WithInput[float64](2),
		nn.WithHiddenLayer[float64](4, activation.SIGMOID),
		nn.WithOutput[float64](1, activation.SIGMOID),
		nn.WithOptimizer[float64](optimizer.NewAdam[float64](0.01)),
		nn.WithTopologyMode[float64](network.Dynamic),
		nn.WithMaxIterations[float64](3),
		nn.WithWeightInitSeed[float64](42),
	)
	if _, _, err := n.Fit(xorSamples()); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if err := n.AddNeuron(0, 2); err != nil {
		t.Fatalf("AddNeuron: %v", err)
	}
	if _, _, err := n.Fit(xorSamples()); err != nil {
		t.Fatalf("Fit after AddNeuron: %v", err)
	}
}

// TestNaNInputRejected guards audit D6: a non-finite input must be rejected, not
// silently poison the weights.
func TestNaNInputRejected(t *testing.T) {
	n := trainedXOR(t)
	if _, err := n.Train([]float64{math.NaN(), 1}, []float64{1}); err == nil {
		t.Error("Train with NaN input returned nil error — should reject")
	}
	// Weights must remain finite and usable.
	out, err := n.Query([]float64{0, 1})
	if err != nil {
		t.Fatalf("Query after rejected NaN: %v", err)
	}
	if math.IsNaN(out[0]) {
		t.Error("network output is NaN — weights were poisoned despite rejection")
	}
}
