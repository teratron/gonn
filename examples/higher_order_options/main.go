// Example E13 — Higher-order options: Sequential and DeepNetwork.
//
// See [.design/specifications/l2-usage-examples.md] §5.2 / E13.
//
// Demonstrates that Sequential and DeepNetwork are ordinary Option[T]
// values and compose freely with the rest of the Options API.
//
//   - Sequential(2, 16, SIGMOID) — two identical Sigmoid(16) hidden layers
//   - DeepNetwork(32, 3, SIGMOID) — three halving layers: 32 → 16 → 8
//
// Both topologies are trained on the Fisher iris dataset (4 inputs,
// 3 classes, 150 samples, 80 / 20 split) using MSE + Xavier + 3000
// epochs. Target: test accuracy ≥ 85 % for each variant.
package main

import (
	_ "embed"
	"encoding/csv"
	"fmt"
	"math/rand/v2"
	"strconv"
	"strings"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/nn"
)

//go:embed iris.csv
var irisCSV string

func main() {
	seqAcc, err := runSequential(2024, 3000)
	if err != nil {
		fmt.Printf("Sequential error: %v\n", err)
		return
	}
	deepAcc, err := runDeepNetwork(2024, 3000)
	if err != nil {
		fmt.Printf("DeepNetwork error: %v\n", err)
		return
	}
	fmt.Printf("Sequential  test acc = %.3f\n", seqAcc)
	fmt.Printf("DeepNetwork test acc = %.3f\n", deepAcc)
}

type irisRecord struct {
	features []float32
	class    int
}

func loadIris() ([]irisRecord, error) {
	r := csv.NewReader(strings.NewReader(irisCSV))
	rows, err := r.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("read csv: %w", err)
	}
	out := make([]irisRecord, 0, len(rows))
	for _, row := range rows {
		if len(row) != 5 {
			continue
		}
		cls, convErr := strconv.Atoi(strings.TrimSpace(row[4]))
		if convErr != nil {
			continue
		}
		feats := make([]float32, 4)
		for i := range 4 {
			v, ferr := strconv.ParseFloat(strings.TrimSpace(row[i]), 32)
			if ferr != nil {
				return nil, fmt.Errorf("parse feature %d: %w", i, ferr)
			}
			feats[i] = float32(v)
		}
		out = append(out, irisRecord{features: feats, class: cls})
	}
	return out, nil
}

func oneHot(class int) []float32 {
	t := make([]float32, 3)
	t[class] = 1
	return t
}

func argmax(v []float32) int {
	best := 0
	for i := 1; i < len(v); i++ {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}

// irisMinMax holds the per-feature normalization bounds computed from the
// full Fisher iris dataset. Scaling inputs to [0,1] prevents large feature
// magnitudes from destabilising ReLU-network training.
var irisMinMax = [4][2]float32{
	{4.3, 7.9}, // sepal_length
	{2.0, 4.4}, // sepal_width
	{1.0, 6.9}, // petal_length
	{0.1, 2.5}, // petal_width
}

func normalizeFeatures(f []float32) []float32 {
	n := make([]float32, len(f))
	for i, v := range f {
		lo, hi := irisMinMax[i][0], irisMinMax[i][1]
		n[i] = (v - lo) / (hi - lo)
	}
	return n
}

func splitIris(seed uint64) (trainSet, testSet []irisRecord, err error) {
	all, err := loadIris()
	if err != nil {
		return nil, nil, err
	}
	for i := range all {
		all[i].features = normalizeFeatures(all[i].features)
	}
	rng := rand.New(rand.NewPCG(seed, seed^0xDEADBEEFCAFEBABE))
	rng.Shuffle(len(all), func(i, j int) { all[i], all[j] = all[j], all[i] })
	splitAt := len(all) * 8 / 10
	return all[:splitAt], all[splitAt:], nil
}

func irisAccuracy(net *nn.NN[float32], set []irisRecord) float32 {
	if len(set) == 0 {
		return 0
	}
	correct := 0
	for _, rec := range set {
		out, err := net.Query(rec.features)
		if err != nil {
			return 0
		}
		if argmax(out) == rec.class {
			correct++
		}
	}
	return float32(correct) / float32(len(set))
}

func trainIris(net *nn.NN[float32], trainSet []irisRecord) error {
	samples := make([]nn.Sample[float32], len(trainSet))
	for i, rec := range trainSet {
		samples[i] = nn.Sample[float32]{Input: rec.features, Target: oneHot(rec.class)}
	}
	_, _, err := net.Fit(samples)
	return err
}

// runSequential builds a topology via Sequential(2, 16, ReLU) and
// trains it on iris. Returns test accuracy.
func runSequential(seed uint64, maxIter uint) (float32, error) {
	trainSet, testSet, err := splitIris(seed)
	if err != nil {
		return 0, err
	}

	net, err := nn.New(
		nn.WithInput[float32](4),
		nn.WithBias[float32](true),
		nn.Sequential[float32](2, 16, activation.SIGMOID),
		nn.WithOutput[float32](3, activation.SOFTMAX),
		nn.WithLearningRate[float32](0.01),
		nn.WithLoss[float32](loss.MSE),
		nn.WithWeightInit[float32](nn.WeightInitXavier),
		nn.WithMaxIterations[float32](maxIter),
	)
	if err != nil {
		return 0, fmt.Errorf("Sequential Compile: %w", err)
	}
	if err = trainIris(net, trainSet); err != nil {
		return 0, fmt.Errorf("Sequential Fit: %w", err)
	}
	return irisAccuracy(net, testSet), nil
}

// runDeepNetwork builds a topology via DeepNetwork(32, 3, ReLU) — three
// layers of sizes 32 → 16 → 8 — and trains it on iris.
func runDeepNetwork(seed uint64, maxIter uint) (float32, error) {
	trainSet, testSet, err := splitIris(seed)
	if err != nil {
		return 0, err
	}

	net, err := nn.New(
		nn.WithInput[float32](4),
		nn.WithBias[float32](true),
		nn.DeepNetwork[float32](32, 3, activation.SIGMOID),
		nn.WithOutput[float32](3, activation.SOFTMAX),
		nn.WithLearningRate[float32](0.01),
		nn.WithLoss[float32](loss.MSE),
		nn.WithWeightInit[float32](nn.WeightInitXavier),
		nn.WithMaxIterations[float32](maxIter),
	)
	if err != nil {
		return 0, fmt.Errorf("DeepNetwork Compile: %w", err)
	}
	if err = trainIris(net, trainSet); err != nil {
		return 0, fmt.Errorf("DeepNetwork Fit: %w", err)
	}
	return irisAccuracy(net, testSet), nil
}
