// Multi-output regression on synthetic 5-dimensional data.
//
// Topology: 5 → ReLU(16) → ReLU(8) → Linear(3), bias on every layer.
// Dataset: 500 samples, inputs drawn from U[-1, 1]^5, targets are three
// deterministic functions of the inputs so ground truth is exact.
// Loss: MSE, rate = 0.005, init = He, max iterations = 5000.
// A 20 % hold-out split feeds the final per-dim RMSE report.
// Target: per-dimension test RMSE ≤ 0.20.
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
	trainRMSE, testRMSE, finalLoss, err := run(42, 5000)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("final loss = %.6f\n", finalLoss)
	fmt.Printf("train RMSE per dim = [%.4f  %.4f  %.4f]\n",
		trainRMSE[0], trainRMSE[1], trainRMSE[2])
	fmt.Printf("test  RMSE per dim = [%.4f  %.4f  %.4f]\n",
		testRMSE[0], testRMSE[1], testRMSE[2])
}

type multiSample struct {
	x []float32 // 5 inputs
	y []float32 // 3 targets
}

// targets maps a 5-dim input to three scalar targets using lightweight
// analytic functions that the network can learn without huge depth.
//   - y0 = x0 + x1           (linear)
//   - y1 = x2 * x3           (bilinear)
//   - y2 = 0.5*(x4^2 - x0)   (quadratic)
func targets(x []float32) []float32 {
	return []float32{
		x[0] + x[1],
		x[2] * x[3],
		0.5 * (x[4]*x[4] - x[0]),
	}
}

func generateMulti(seed uint64, n int) []multiSample {
	rng := rand.New(rand.NewPCG(seed, seed^0x123456789ABCDEF0))
	out := make([]multiSample, n)
	for i := range n {
		x := make([]float32, 5)
		for j := range 5 {
			x[j] = float32(rng.Float64()*2 - 1)
		}
		out[i] = multiSample{x: x, y: targets(x)}
	}
	return out
}

// run builds the network, trains on the first 80 % of the dataset, and
// reports per-dimension RMSE plus the final mean-epoch loss.
func run(seed uint64, maxIter uint) (trainRMSE, testRMSE [3]float64, finalLoss float32, err error) {
	const n = 500
	all := generateMulti(seed, n)

	splitAt := n * 8 / 10
	trainSet := all[:splitAt]
	testSet := all[splitAt:]

	samples := make([]nn.Sample[float32], len(trainSet))
	for i, s := range trainSet {
		samples[i] = nn.Sample[float32]{Input: s.x, Target: s.y}
	}

	net, err := nn.New(
		nn.WithInput[float32](5),
		nn.WithBias[float32](true),
		nn.WithHiddenLayer[float32](16, activation.ReLU),
		nn.WithHiddenLayer[float32](8, activation.ReLU),
		nn.WithOutput[float32](3, activation.Linear),
		nn.WithLearningRate[float32](0.005),
		nn.WithLoss[float32](loss.MSE),
		nn.WithWeightInit[float32](nn.WeightInitHe),
		nn.WithMaxIterations[float32](maxIter),
	)
	if err != nil {
		return trainRMSE, testRMSE, 0, fmt.Errorf("Compile: %w", err)
	}

	_, fl, ferr := net.Fit(samples)
	if ferr != nil {
		return trainRMSE, testRMSE, 0, fmt.Errorf("Fit: %w", ferr)
	}

	trainRMSE = perDimRMSE(net, trainSet)
	testRMSE = perDimRMSE(net, testSet)
	finalLoss = fl
	return
}

// perDimRMSE computes root mean squared error independently for each of
// the three output dimensions.
func perDimRMSE(net *nn.NN[float32], set []multiSample) [3]float64 {
	if len(set) == 0 {
		return [3]float64{}
	}
	var sumSq [3]float64
	for _, s := range set {
		out, err := net.Query(s.x)
		if err != nil {
			return [3]float64{math.NaN(), math.NaN(), math.NaN()}
		}
		for d := range 3 {
			diff := float64(out[d]) - float64(s.y[d])
			sumSq[d] += diff * diff
		}
	}
	var result [3]float64
	for d := range 3 {
		result[d] = math.Sqrt(sumSq[d] / float64(len(set)))
	}
	return result
}
