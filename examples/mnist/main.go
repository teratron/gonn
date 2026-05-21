// Example: MNIST digit recognition.
//
// Demonstrates loading the MNIST dataset via IDXReader / MNISTLoader,
// building a two-hidden-layer network with BatchNorm, and using AndTrain
// for a second training epoch without losing the weights from the first.
//
// Architecture: 784 → 128 (ReLU + BatchNorm) → 64 (ReLU) → 10 (Sigmoid)
//
// Usage:
//
//	go run ./examples/mnist/ \
//	    -images train-images-idx3-ubyte \
//	    -labels train-labels-idx1-ubyte \
//	    -n 500
//
// The IDX files are not bundled in this repository. Download them from
// the MNIST mirror listed in README.md and place them in the working
// directory (or pass their paths via -images / -labels).
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/dataset"
	"github.com/teratron/gonn/pkg/nn"
)

const numClasses = 10

// oneHot converts a raw class label (0–9) to a unit vector of length numClasses.
// MNISTLoader.Next returns targets as []float32{label}, so the caller is
// responsible for encoding the supervision signal for multi-class output.
func oneHot(label float32) []float32 {
	v := make([]float32, numClasses)
	idx := int(label)
	if idx >= 0 && idx < numClasses {
		v[idx] = 1
	}
	return v
}

// loadSamples drains loader into a flat Sample slice (at most maxSamples records).
// Targets are one-hot encoded inline so that the 10-output network receives
// the correct supervision signal.
func loadSamples(loader *dataset.MNISTLoader[float32], maxSamples int) ([]nn.Sample[float32], error) {
	ctx := context.Background()
	samples := make([]nn.Sample[float32], 0, maxSamples)
	for len(samples) < maxSamples {
		batch, err := loader.Next(ctx)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		for i := range batch.Inputs {
			if len(samples) >= maxSamples {
				break
			}
			samples = append(samples, nn.Sample[float32]{
				Input:  batch.Inputs[i],
				Target: oneHot(batch.Targets[i][0]),
			})
		}
	}
	return samples, nil
}

// argmax returns the index of the largest element in v.
// Used to convert the 10-dimensional network output into a digit prediction.
func argmax(v []float32) int {
	best := 0
	for i := 1; i < len(v); i++ {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}

func main() {
	imgPath := flag.String("images", "train-images-idx3-ubyte", "path to MNIST training images IDX3 file")
	lblPath := flag.String("labels", "train-labels-idx1-ubyte", "path to MNIST training labels IDX1 file")
	maxSamples := flag.Int("n", 500, "number of training samples to load")
	flag.Parse()

	loader, err := dataset.NewMNISTLoaderFiles[float32](*imgPath, *lblPath, 64, 255.0)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open MNIST: %v\n", err)
		fmt.Fprintln(os.Stderr, "Download train-images-idx3-ubyte and train-labels-idx1-ubyte (see README.md).")
		os.Exit(1)
	}
	defer loader.Close()

	samples, err := loadSamples(loader, *maxSamples)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load samples: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Loaded %d MNIST samples\n", len(samples))

	// 784 → 128 (ReLU) → BatchNorm → 64 (ReLU) → 10 (Sigmoid).
	// WithBatchNorm(0) inserts a BatchNorm layer after the first hidden layer (index 0).
	net, err := nn.New(
		nn.WithInput[float32](784),
		nn.WithHiddenLayer[float32](128, activation.ReLU),
		nn.WithBatchNorm[float32](0),
		nn.WithHiddenLayer[float32](64, activation.ReLU),
		nn.WithOutput[float32](numClasses, activation.SIGMOID),
		nn.WithLearningRate[float32](0.01),
		nn.WithMaxIterations[float32](1),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "build network: %v\n", err)
		os.Exit(1)
	}

	// Epoch 1 — initial Fit pass through all loaded samples.
	epochs, loss, err := net.Fit(samples)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fit: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Epoch 1 (Fit):     %d iteration(s), loss=%.4f\n", epochs, loss)

	// Epoch 2 — AndTrain at a reduced learning rate.
	// The weight slab is preserved from epoch 1 (FMT-6); only the convergence
	// counters reset, so the second pass can fine-tune rather than relearn from scratch.
	epochs2, loss2, err := net.AndTrain(samples, nn.WithLearningRate[float32](0.001))
	if err != nil {
		fmt.Fprintf(os.Stderr, "AndTrain: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Epoch 2 (AndTrain): %d iteration(s), loss=%.4f\n", epochs2, loss2)

	// Query the first sample and report the predicted vs actual digit.
	first := samples[0]
	logits, err := net.Query(first.Input)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Query: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Sample 0: predicted=%d, actual=%d\n", argmax(logits), argmax(first.Target))
}
