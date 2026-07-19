package network

import (
	"math"
	"testing"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/layer"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/utils"
)

// linearChain builds a Linear-activation chain (no bias) so that forward
// values, derivatives, and weight gradients are bit-stable for the
// hand-computed golden vectors used by Phase 5 Track A multi-hidden
// regression tests. Linear σ(x) = x, σ'(x) = 1.
func linearChain[T utils.Float](inSize int, hiddenSizes []int, outSize int) (*Network[T], error) {
	in := layer.NewInput[T](inSize)
	hiddens := make([]*layer.Dense[T], len(hiddenSizes))
	for i, sz := range hiddenSizes {
		hiddens[i] = layer.NewDense[T](sz, activation.Linear, false)
	}
	out := layer.NewOutput[T](outSize, activation.Linear, loss.MSE, false)
	n := New[T]()
	if err := n.SetLayers(in, hiddens, out); err != nil {
		return nil, err
	}
	if err := n.Build(); err != nil {
		return nil, err
	}
	return &n, nil
}

// setHiddenWeights overwrites every incoming axon weight in Hiddens[layer]
// with the supplied row-major matrix; weights[cellIdx][srcIdx] feeds
// axon srcIdx of cell cellIdx. Used by golden-math tests to neutralise
// random init.
func setHiddenWeights[T utils.Float](n *Network[T], layerIdx int, weights [][]T) {
	for cellIdx, h := range n.Hiddens[layerIdx].Cells() {
		for srcIdx, w := range weights[cellIdx] {
			h.Axons[srcIdx].SetW(w)
		}
	}
}

// setOutputWeights mirrors setHiddenWeights for the Output layer.
func setOutputWeights[T utils.Float](n *Network[T], weights [][]T) {
	for cellIdx, o := range n.Output.Cells() {
		for srcIdx, w := range weights[cellIdx] {
			o.Axons[srcIdx].SetW(w)
		}
	}
}

// approxEqual reports whether |a-b| ≤ tol; centralised to keep test
// readability up while comparing float64 chains.
func approxEqual(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}

// TestBuildWiresMultiHiddenChain checks T-5A03's chain wiring rule on a
// three-hidden topology: every Hiddens[i] cell receives one axon per
// prev-layer cell (plus a bias when the layer requested one), and the
// Output draws from Hiddens[len-1]. The numbers in the table are
// purely structural — no math runs here.
func TestBuildWiresMultiHiddenChain(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name        string
		inSize      int
		hiddenSizes []int
		outSize     int
		bias        bool
	}{
		{"two_hidden_no_bias", 3, []int{4, 2}, 1, false},
		{"two_hidden_with_bias", 3, []int{4, 2}, 1, true},
		{"three_hidden_no_bias", 2, []int{3, 3, 3}, 2, false},
		{"three_hidden_with_bias", 2, []int{3, 3, 3}, 2, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			n, err := newTrainableChain[float64](tc.inSize, tc.hiddenSizes, tc.outSize, tc.bias)
			if err != nil {
				t.Fatalf("setup: %v", err)
			}
			biasInc := 0
			if tc.bias {
				biasInc = 1
			}

			for i, hb := range n.Hiddens {
				wantSrc := tc.inSize
				if i > 0 {
					wantSrc = tc.hiddenSizes[i-1]
				}
				wantAxons := wantSrc + biasInc
				if got := hb.Len(); got != tc.hiddenSizes[i] {
					t.Errorf("Hiddens[%d].Len = %d; want %d", i, got, tc.hiddenSizes[i])
				}
				for cellIdx, h := range hb.Cells() {
					if got := len(h.Axons); got != wantAxons {
						t.Errorf("Hiddens[%d].cell[%d] axons = %d; want %d",
							i, cellIdx, got, wantAxons)
					}
				}
			}
			lastHidden := tc.hiddenSizes[len(tc.hiddenSizes)-1]
			wantOutAxons := lastHidden + biasInc
			for cellIdx, o := range n.Output.Cells() {
				if got := len(o.Axons); got != wantOutAxons {
					t.Errorf("Output.cell[%d] axons = %d; want %d",
						cellIdx, got, wantOutAxons)
				}
			}
		})
	}
}

// TestForwardTwoHiddenGolden runs a Linear-activation 2 → 2 → 2 → 1
// chain with hand-set weights and compares the output against the
// closed-form forward computation. Linear avoids σ'/numerical drift, so
// any deviation here means the chain wiring or the per-layer
// pre-activation capture is broken.
func TestForwardTwoHiddenGolden(t *testing.T) {
	t.Parallel()
	n, err := linearChain[float64](2, []int{2, 2}, 1)
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	// W1 (Hiddens[0]): cells 0,1 each receive 2 input axons.
	setHiddenWeights(n, 0, [][]float64{{0.1, 0.2}, {0.3, 0.4}})
	// W2 (Hiddens[1]): cells 0,1 each receive 2 axons from Hiddens[0].
	setHiddenWeights(n, 1, [][]float64{{0.5, 0.6}, {0.7, 0.8}})
	// Output: 1 cell receives 2 axons from Hiddens[1].
	setOutputWeights(n, [][]float64{{0.9, 1.0}})

	if err := n.SetInputs([]float64{1, 2}); err != nil {
		t.Fatalf("SetInputs: %v", err)
	}
	n.CalculateValues()

	wantH0 := []float64{
		0.1*1 + 0.2*2,
		0.3*1 + 0.4*2,
	}
	wantH1 := []float64{
		0.5*wantH0[0] + 0.6*wantH0[1],
		0.7*wantH0[0] + 0.8*wantH0[1],
	}
	wantOut := 0.9*wantH1[0] + 1.0*wantH1[1]

	for i, w := range wantH0 {
		got := *n.Hiddens[0].Cells()[i].GetValue()
		if !approxEqual(got, w, 1e-12) {
			t.Errorf("Hiddens[0][%d] = %v; want %v", i, got, w)
		}
		if pre := n.preactHiddens[0][i]; !approxEqual(pre, w, 1e-12) {
			t.Errorf("preactHiddens[0][%d] = %v; want %v (Linear)", i, pre, w)
		}
	}
	for i, w := range wantH1 {
		got := *n.Hiddens[1].Cells()[i].GetValue()
		if !approxEqual(got, w, 1e-12) {
			t.Errorf("Hiddens[1][%d] = %v; want %v", i, got, w)
		}
	}
	if got := *n.Output.Cells()[0].GetValue(); !approxEqual(got, wantOut, 1e-12) {
		t.Errorf("Output[0] = %v; want %v", got, wantOut)
	}
}

// TestBackwardWeightUpdateTwoHiddenGolden verifies one full
// forward + backward + weight-update step against analytical δ and
// gradient calculations. Linear activations make σ'(z) = 1 everywhere,
// so the weight delta reduces to rate × δ_layer × source_value. Any
// off-by-one in the right-to-left δ aggregation surfaces as a wrong
// W1 update — the deepest layer is the most sensitive to chain bugs.
func TestBackwardWeightUpdateTwoHiddenGolden(t *testing.T) {
	t.Parallel()
	n, err := linearChain[float64](2, []int{2, 2}, 1)
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	w1 := [][]float64{{0.1, 0.2}, {0.3, 0.4}}
	w2 := [][]float64{{0.5, 0.6}, {0.7, 0.8}}
	wo := [][]float64{{0.9, 1.0}}
	setHiddenWeights(n, 0, w1)
	setHiddenWeights(n, 1, w2)
	setOutputWeights(n, wo)

	const rate = 0.05
	n.LearningRate = rate

	input := []float64{1, 2}
	target := []float64{2.0}
	if _, err := n.Train(input, target); err != nil {
		t.Fatalf("Train: %v", err)
	}

	// Re-derive the analytical gradients (Linear → derivative = 1).
	h0 := []float64{w1[0][0]*input[0] + w1[0][1]*input[1], w1[1][0]*input[0] + w1[1][1]*input[1]}
	h1 := []float64{w2[0][0]*h0[0] + w2[0][1]*h0[1], w2[1][0]*h0[0] + w2[1][1]*h0[1]}
	out := wo[0][0]*h1[0] + wo[0][1]*h1[1]
	deltaO := target[0] - out
	deltaH1 := []float64{deltaO * wo[0][0], deltaO * wo[0][1]}
	deltaH0 := []float64{
		deltaH1[0]*w2[0][0] + deltaH1[1]*w2[1][0],
		deltaH1[0]*w2[0][1] + deltaH1[1]*w2[1][1],
	}

	// Expected post-step weights.
	wantW1 := [][]float64{
		{w1[0][0] + rate*deltaH0[0]*input[0], w1[0][1] + rate*deltaH0[0]*input[1]},
		{w1[1][0] + rate*deltaH0[1]*input[0], w1[1][1] + rate*deltaH0[1]*input[1]},
	}
	wantW2 := [][]float64{
		{w2[0][0] + rate*deltaH1[0]*h0[0], w2[0][1] + rate*deltaH1[0]*h0[1]},
		{w2[1][0] + rate*deltaH1[1]*h0[0], w2[1][1] + rate*deltaH1[1]*h0[1]},
	}
	wantWo := [][]float64{
		{wo[0][0] + rate*deltaO*h1[0], wo[0][1] + rate*deltaO*h1[1]},
	}

	const tol = 1e-12
	for cellIdx, h := range n.Hiddens[0].Cells() {
		for srcIdx, a := range h.Axons {
			want := wantW1[cellIdx][srcIdx]
			if !approxEqual(a.W(), want, tol) {
				t.Errorf("W1[%d][%d] = %v; want %v", cellIdx, srcIdx, a.W(), want)
			}
		}
	}
	for cellIdx, h := range n.Hiddens[1].Cells() {
		for srcIdx, a := range h.Axons {
			want := wantW2[cellIdx][srcIdx]
			if !approxEqual(a.W(), want, tol) {
				t.Errorf("W2[%d][%d] = %v; want %v", cellIdx, srcIdx, a.W(), want)
			}
		}
	}
	for cellIdx, o := range n.Output.Cells() {
		for srcIdx, a := range o.Axons {
			want := wantWo[cellIdx][srcIdx]
			if !approxEqual(a.W(), want, tol) {
				t.Errorf("Wo[%d][%d] = %v; want %v", cellIdx, srcIdx, a.W(), want)
			}
		}
	}
}

// TestBackwardThreeHiddenGolden extends the analytical check to a three-
// hidden chain. The right-to-left δ aggregation must thread through two
// intermediate layers without dropping factors — a regression that the
// two-hidden test cannot catch (only one chain hop). Linear activation
// keeps every derivative at 1.
func TestBackwardThreeHiddenGolden(t *testing.T) {
	t.Parallel()
	n, err := linearChain[float64](2, []int{2, 2, 2}, 1)
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	w1 := [][]float64{{0.1, 0.2}, {0.3, 0.4}}
	w2 := [][]float64{{0.5, 0.6}, {0.7, 0.8}}
	w3 := [][]float64{{1.1, 1.2}, {1.3, 1.4}}
	wo := [][]float64{{0.9, 1.0}}
	setHiddenWeights(n, 0, w1)
	setHiddenWeights(n, 1, w2)
	setHiddenWeights(n, 2, w3)
	setOutputWeights(n, wo)

	const rate = 0.01
	n.LearningRate = rate

	input := []float64{1, 2}
	target := []float64{1.5}
	if _, err := n.Train(input, target); err != nil {
		t.Fatalf("Train: %v", err)
	}

	h0 := []float64{w1[0][0]*input[0] + w1[0][1]*input[1], w1[1][0]*input[0] + w1[1][1]*input[1]}
	h1 := []float64{w2[0][0]*h0[0] + w2[0][1]*h0[1], w2[1][0]*h0[0] + w2[1][1]*h0[1]}
	h2 := []float64{w3[0][0]*h1[0] + w3[0][1]*h1[1], w3[1][0]*h1[0] + w3[1][1]*h1[1]}
	out := wo[0][0]*h2[0] + wo[0][1]*h2[1]

	deltaO := target[0] - out
	deltaH2 := []float64{deltaO * wo[0][0], deltaO * wo[0][1]}
	deltaH1 := []float64{
		deltaH2[0]*w3[0][0] + deltaH2[1]*w3[1][0],
		deltaH2[0]*w3[0][1] + deltaH2[1]*w3[1][1],
	}
	deltaH0 := []float64{
		deltaH1[0]*w2[0][0] + deltaH1[1]*w2[1][0],
		deltaH1[0]*w2[0][1] + deltaH1[1]*w2[1][1],
	}

	wantW1 := [][]float64{
		{w1[0][0] + rate*deltaH0[0]*input[0], w1[0][1] + rate*deltaH0[0]*input[1]},
		{w1[1][0] + rate*deltaH0[1]*input[0], w1[1][1] + rate*deltaH0[1]*input[1]},
	}

	const tol = 1e-12
	// W1 is the chain's deepest layer — sensitive to any factor lost in
	// right-to-left δ aggregation across W3 and W2.
	for cellIdx, h := range n.Hiddens[0].Cells() {
		for srcIdx, a := range h.Axons {
			want := wantW1[cellIdx][srcIdx]
			if !approxEqual(a.W(), want, tol) {
				t.Errorf("W1[%d][%d] = %v; want %v", cellIdx, srcIdx, a.W(), want)
			}
		}
	}
}

// TestXORTwoHiddenConvergence exercises the full Sigmoid pipeline on a
// 2-2-2-1 multi-hidden chain so that derivative folding (CalculateMisses)
// and value propagation (CalculateValues) are validated end-to-end.
// XOR is the canonical non-linear smoke task; convergence below 0.05
// within the budget proves the chain trains, not just wires.
func TestXORTwoHiddenConvergence(t *testing.T) {
	t.Parallel()
	// Deterministic Xavier init: the default time-seeded U[-0.5,0.5] start
	// occasionally lands in a local minimum that misses the 0.05 target
	// within the epoch budget (rare pre-existing flake).
	rng, _ := utils.NewRNG(12345)
	sampler := func(fanIn, fanOut int) float64 { return utils.XavierUniform[float64](rng, fanIn, fanOut) }
	n, err := newTrainableChainSampled(2, []int{4, 4}, 1, true, sampler)
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	n.LearningRate = 0.5

	dataset := [][2][]float64{
		{{0, 0}, {0}},
		{{0, 1}, {1}},
		{{1, 0}, {1}},
		{{1, 1}, {0}},
	}
	const (
		maxEpochs  = 30000
		targetLoss = 0.05
	)
	var lastLoss float64
	for epoch := range maxEpochs {
		var total float64
		for _, sample := range dataset {
			l, err := n.Train(sample[0], sample[1])
			if err != nil {
				t.Fatalf("Train epoch=%d: %v", epoch, err)
			}
			total += float64(l)
		}
		lastLoss = total / float64(len(dataset))
		if lastLoss < targetLoss {
			t.Logf("XOR (2-hidden) converged in %d epochs, mean loss = %v", epoch+1, lastLoss)
			break
		}
	}
	if lastLoss >= targetLoss {
		t.Errorf("XOR (2-hidden) did not converge: mean loss after %d epochs = %v (want < %v)",
			maxEpochs, lastLoss, targetLoss)
	}
}
