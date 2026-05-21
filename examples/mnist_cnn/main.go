// Example: MNIST digit recognition with a 2-D convolutional prefix (E16).
//
// Demonstrates building a CNN feature extractor (Conv2D → MaxPool2D →
// Flatten2D) followed by a dense classification head, trained on MNIST.
// This is the canonical E16 example for the GoNN Conv2D layer set.
//
// Architecture:
//
//	Input  1×28×28 (784 flat)
//	Conv2D   8 filters, 3×3, stride 1×1, PadSame   → 8×28×28 = 6272
//	MaxPool2D 2×2                                   → 8×14×14 = 1568
//	Flatten2D                                       → 1568
//	Dense    128 (ReLU)
//	Dense     64 (ReLU)
//	Output    10 (Sigmoid)
//
// Usage:
//
//	go run ./examples/mnist_cnn/ \
//	    -images train-images-idx3-ubyte \
//	    -labels train-labels-idx1-ubyte \
//	    -n 200
//
// The IDX files are not bundled in this repository. Download them from the
// MNIST mirror listed in the README and place them in the working directory
// (or pass their paths via -images / -labels).
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/dataset"
	convPkg "github.com/teratron/gonn/pkg/layer/conv"
	"github.com/teratron/gonn/pkg/nn"
)

const numClasses = 10

// oneHot converts a raw class label (0–9) to a unit vector of length numClasses.
func oneHot(label float32) []float32 {
	v := make([]float32, numClasses)
	idx := int(label)
	if idx >= 0 && idx < numClasses {
		v[idx] = 1
	}
	return v
}

// loadSamples drains loader into a Sample slice (at most maxSamples records).
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
	maxSamples := flag.Int("n", 200, "number of training samples to load")
	flag.Parse()

	loader, err := dataset.NewMNISTLoaderFiles[float32](*imgPath, *lblPath, 64, 255.0)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open MNIST: %v\n", err)
		fmt.Fprintln(os.Stderr, "Download train-images-idx3-ubyte and train-labels-idx1-ubyte (see README.md).")
		os.Exit(1)
	}
	defer loader.Close()

	// Annotate the loader with the explicit 2-D shape so compile() can detect
	// it via the ImageShaper interface when no WithInputShape is supplied.
	loader.WithImageShape(1, 28, 28)

	samples, err := loadSamples(loader, *maxSamples)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load samples: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Loaded %d MNIST samples\n", len(samples))

	// Build the CNN:
	//   Conv2D  (8 filters, 3×3, PadSame) followed by MaxPool2D (2×2)
	//   and Flatten2D, then two dense hidden layers and a 10-class output.
	//
	// WithInputShape(1, 28, 28) declares the CHW layout to compile(); the flat
	// input width 784 = 1×28×28 is still passed to WithInput.
	net, err := nn.New(
		nn.WithInput[float32](784),
		nn.WithInputShape[float32](1, 28, 28),
		nn.WithConv2D[float32](8, 1, 3, 3, 1, 1, convPkg.PadSame, true),
		nn.WithMaxPool2D[float32](2, 2),
		nn.WithFlatten2D[float32](),
		nn.WithHiddenLayer[float32](128, activation.ReLU),
		nn.WithHiddenLayer[float32](64, activation.ReLU),
		nn.WithOutput[float32](numClasses, activation.SIGMOID),
		nn.WithLearningRate[float32](0.01),
		nn.WithMaxIterations[float32](1),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "build network: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("Network compiled successfully")

	// Single training epoch over the loaded samples.
	epochs, loss, err := net.Fit(samples)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fit: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Epoch 1 (Fit): %d iteration(s), loss=%.4f\n", epochs, loss)

	// Second pass at a lower learning rate (fine-tuning).
	epochs2, loss2, err := net.AndTrain(samples, nn.WithLearningRate[float32](0.001))
	if err != nil {
		fmt.Fprintf(os.Stderr, "AndTrain: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Epoch 2 (AndTrain): %d iteration(s), loss=%.4f\n", epochs2, loss2)

	// Query the first sample.
	first := samples[0]
	logits, err := net.Query(first.Input)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Query: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Sample 0: predicted=%d, actual=%d\n", argmax(logits), argmax(first.Target))
}
