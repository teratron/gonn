// Example E04 — Binary classification on a Gaussian-blob dataset.
//
// See [.design/specifications/l2-usage-examples.md] §5.2 / E04. Phase 5
// promotes this example from the v0.2-deferred bucket once
// [l2-multihidden-impl] lifts the single-hidden gate.
//
// Topology: 2 → ReLU(8) → ReLU(8) → Sigmoid(1), bias on every layer.
// Dataset: 200 points sampled from two well-separated 2-D Gaussian
// blobs centred at (-1.5, -1.5) and (+1.5, +1.5) with σ = 0.6. Labels
// are {0, 1}. The dataset is generated deterministically so the
// example's smoke test is reproducible without storing fixtures on
// disk.
//
// Hyper-parameters: rate = 0.01, loss = BCE (binary cross-entropy),
// init = He, max iterations = 2000. A 20 % hold-out split feeds the
// final accuracy report.
package main

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/nn"
)

func main() {
	trainAcc, testAcc, finalLoss, err := runE04(1234)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("final loss = %.6f, train acc = %.3f, test acc = %.3f\n",
		finalLoss, trainAcc, testAcc)
}

// labelledSample bundles one (input, label) pair so the smoke test
// can run accuracy assertions without re-deriving labels from the
// raw nn.Sample.
type labelledSample struct {
	x     []float32
	label float32
}

// generateBlobs synthesises 2*n labelled points — n per class — using
// the supplied PCG seed. Centres are well separated so a 2-hidden ReLU
// network can reach > 90 % accuracy without dataset tricks.
func generateBlobs(seed uint64, n int) []labelledSample {
	rng := rand.New(rand.NewPCG(seed, seed^0x9E3779B97F4A7C15))
	out := make([]labelledSample, 0, 2*n)
	const sigma = 0.6
	centres := [2][2]float32{{-1.5, -1.5}, {1.5, 1.5}}
	for class := range 2 {
		c := centres[class]
		for range n {
			x := []float32{
				c[0] + float32(rng.NormFloat64())*sigma,
				c[1] + float32(rng.NormFloat64())*sigma,
			}
			out = append(out, labelledSample{x: x, label: float32(class)})
		}
	}
	rng.Shuffle(len(out), func(i, j int) { out[i], out[j] = out[j], out[i] })
	return out
}

// runE04 builds the network, runs Fit, and reports train / test
// accuracy plus the final mean-epoch loss. seed is forwarded to the
// data generator so smoke tests can pin a deterministic split.
func runE04(seed uint64) (trainAcc, testAcc, finalLoss float32, err error) {
	const totalPerClass = 100
	all := generateBlobs(seed, totalPerClass)

	// 80 / 20 split — first 160 train, remaining 40 test.
	const trainCut = 160
	trainSet := all[:trainCut]
	testSet := all[trainCut:]

	samples := make([]nn.Sample[float32], len(trainSet))
	for i, s := range trainSet {
		samples[i] = nn.Sample[float32]{Input: s.x, Target: []float32{s.label}}
	}

	n, err := nn.New[float32](
		nn.WithInput[float32](2),
		nn.WithBias[float32](true),
		nn.WithHiddenLayer[float32](8, activation.ReLU),
		nn.WithHiddenLayer[float32](8, activation.ReLU),
		nn.WithOutput[float32](1, activation.SIGMOID),
		nn.WithLearningRate[float32](0.01),
		nn.WithLoss[float32](loss.BCE),
		nn.WithWeightInit[float32](nn.WeightInitHe),
		nn.WithMaxIterations[float32](2000),
		nn.WithLossLimit[float32](-1),
	)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("Compile: %w", err)
	}
	_, fl, ferr := n.Fit(samples)
	if ferr != nil {
		return 0, 0, 0, fmt.Errorf("Fit: %w", ferr)
	}

	trainAcc = float32(accuracy(n, trainSet))
	testAcc = float32(accuracy(n, testSet))
	finalLoss = fl
	return
}

// accuracy returns the fraction of correctly classified samples using a
// 0.5 threshold against the Sigmoid output. Exposed as a free helper so
// tests can assert per-split accuracy independently of training.
func accuracy(n *nn.NN[float32], set []labelledSample) float64 {
	if len(set) == 0 {
		return 0
	}
	correct := 0
	for _, s := range set {
		out, err := n.Query(s.x)
		if err != nil {
			return math.NaN()
		}
		pred := float32(0)
		if out[0] >= 0.5 {
			pred = 1
		}
		if pred == s.label {
			correct++
		}
	}
	return float64(correct) / float64(len(set))
}
