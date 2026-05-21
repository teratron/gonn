// Example E05 — Iris multi-class classification.
//
// See [.design/specifications/l2-usage-examples.md] §5.2 / E05.
//
// Topology: 4 → ReLU(16) → ReLU(8) → SoftMax(3), bias on every layer.
// Dataset: Fisher iris (150 samples, 4 features, 3 classes).
// Loss: MSE, rate = 0.01, init = Xavier, max iterations = 5000.
// An EpochCallback is registered to observe per-epoch loss.
// A 20 % hold-out split feeds the final accuracy report.
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
	trainAcc, testAcc, finalLoss, err := runE05(2024)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("final loss = %.6f, train acc = %.3f, test acc = %.3f\n",
		finalLoss, trainAcc, testAcc)
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
			continue // skip header or malformed rows
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

// minMaxNorm normalises all records in-place to [0, 1] per feature using
// the global min/max over the full slice. Must be called before splitting
// so the same scale is applied to both train and test partitions.
func minMaxNorm(recs []irisRecord) {
	if len(recs) == 0 {
		return
	}
	var mins, maxs [4]float32
	for i := range 4 {
		mins[i] = recs[0].features[i]
		maxs[i] = recs[0].features[i]
	}
	for _, r := range recs[1:] {
		for i := range 4 {
			if r.features[i] < mins[i] {
				mins[i] = r.features[i]
			}
			if r.features[i] > maxs[i] {
				maxs[i] = r.features[i]
			}
		}
	}
	for ri := range recs {
		for i := range 4 {
			if span := maxs[i] - mins[i]; span > 0 {
				recs[ri].features[i] = (recs[ri].features[i] - mins[i]) / span
			}
		}
	}
}

// runE05 builds the network, trains on an 80 % split, and reports
// train / test accuracy plus the final mean-epoch loss. seed is
// forwarded to the shuffle so smoke tests can pin a deterministic split.
func runE05(seed uint64) (trainAcc, testAcc, finalLoss float32, err error) {
	all, err := loadIris()
	if err != nil {
		return 0, 0, 0, err
	}

	// Scale features to [0, 1] before shuffling so both partitions share
	// the same normalisation statistics (derived from all 150 samples).
	minMaxNorm(all)

	rng := rand.New(rand.NewPCG(seed, seed^0xDEADBEEFCAFEBABE))
	rng.Shuffle(len(all), func(i, j int) { all[i], all[j] = all[j], all[i] })

	splitAt := len(all) * 8 / 10
	trainSet := all[:splitAt]
	testSet := all[splitAt:]

	samples := make([]nn.Sample[float32], len(trainSet))
	for i, rec := range trainSet {
		samples[i] = nn.Sample[float32]{Input: rec.features, Target: oneHot(rec.class)}
	}

	var lastLoss float32
	net, err := nn.New(
		nn.WithInput[float32](4),
		nn.WithBias[float32](true),
		nn.WithHiddenLayer[float32](16, activation.ReLU),
		nn.WithHiddenLayer[float32](8, activation.ReLU),
		nn.WithOutput[float32](3, activation.SOFTMAX),
		nn.WithLearningRate[float32](0.01),
		nn.WithLoss[float32](loss.MSE),
		nn.WithWeightInit[float32](nn.WeightInitXavier),
		nn.WithMaxIterations[float32](5000),
		nn.WithEpochCallback[float32](func(_ uint, lossValue float32) {
			lastLoss = lossValue
		}),
	)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("Compile: %w", err)
	}

	_, fl, ferr := net.Fit(samples)
	if ferr != nil {
		return 0, 0, 0, fmt.Errorf("Fit: %w", ferr)
	}
	_ = lastLoss

	trainAcc = float32(irisAccuracy(net, trainSet))
	testAcc = float32(irisAccuracy(net, testSet))
	finalLoss = fl
	return
}

// irisAccuracy returns the fraction of correctly classified records
// using argmax of the three SoftMax outputs.
func irisAccuracy(net *nn.NN[float32], set []irisRecord) float64 {
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
	return float64(correct) / float64(len(set))
}
