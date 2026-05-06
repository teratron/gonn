// Example E07 — Sine-wave regression.
//
// See [.design/specifications/l2-usage-examples.md] §5.2 / E07.
//
// Topology: 1 → TanH(16) → TanH(16) → Linear(1), bias on every layer.
// Dataset: 200 evenly-spaced points on x ∈ [0, 2π] with y = sin(x).
// Loss: MSE, rate = 0.005, init = Xavier, max iterations = 10000.
// A 20 % hold-out split feeds the final RMSE report.
// Target: test RMSE ≤ 0.10.
package main

import (
	"fmt"
	"math"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/nn"
)

func main() {
	trainRMSE, testRMSE, finalLoss, err := runE07(10000)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("final loss = %.6f, train RMSE = %.4f, test RMSE = %.4f\n",
		finalLoss, trainRMSE, testRMSE)
}

type sinPoint struct {
	x, y float32
}

func generateSin(n int) []sinPoint {
	pts := make([]sinPoint, n)
	for i := range n {
		x := float64(i) / float64(n-1) * 2 * math.Pi
		pts[i] = sinPoint{x: float32(x), y: float32(math.Sin(x))}
	}
	return pts
}

// runE07 builds the network, trains on the first 80 % of the dataset
// (in-order, no shuffle — the x-grid is already uniformly spaced), and
// reports train / test RMSE plus the final mean-epoch loss.
func runE07(maxIter uint) (trainRMSE, testRMSE, finalLoss float32, err error) {
	const n = 200
	all := generateSin(n)

	splitAt := n * 8 / 10
	trainSet := all[:splitAt]
	testSet := all[splitAt:]

	samples := make([]nn.Sample[float32], len(trainSet))
	for i, p := range trainSet {
		samples[i] = nn.Sample[float32]{
			Input:  []float32{p.x},
			Target: []float32{p.y},
		}
	}

	net, err := nn.New[float32](
		nn.WithInput[float32](1),
		nn.WithBias[float32](true),
		nn.WithHiddenLayer[float32](16, activation.TanH),
		nn.WithHiddenLayer[float32](16, activation.TanH),
		nn.WithOutput[float32](1, activation.Linear),
		nn.WithLearningRate[float32](0.005),
		nn.WithLoss[float32](loss.MSE),
		nn.WithWeightInit[float32](nn.WeightInitXavier),
		nn.WithMaxIterations[float32](maxIter),
	)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("Compile: %w", err)
	}

	_, fl, ferr := net.Fit(samples)
	if ferr != nil {
		return 0, 0, 0, fmt.Errorf("Fit: %w", ferr)
	}

	trainRMSE = float32(rmse(net, trainSet))
	testRMSE = float32(rmse(net, testSet))
	finalLoss = fl
	return
}

// rmse computes the root mean squared error between the network output
// and the true sin(x) target for every point in set.
func rmse(net *nn.NN[float32], set []sinPoint) float64 {
	if len(set) == 0 {
		return 0
	}
	var sum float64
	for _, p := range set {
		out, err := net.Query([]float32{p.x})
		if err != nil {
			return math.NaN()
		}
		d := float64(out[0]) - float64(p.y)
		sum += d * d
	}
	return math.Sqrt(sum / float64(len(set)))
}
