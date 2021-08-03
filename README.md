<div style="text-align: center">
  <a href="https://pkg.go.dev/github.com/zigenzoog/gonn"><img src="https://pkg.go.dev/badge/github.com/zigenzoog/gonn.svg" alt="Go Reference"></a>
  <a href="https://github.com/zigenzoog/gonn/releases/tag/v0.3.3" title="0.3.3" rel="nofollow"><img src="https://img.shields.io/badge/version-0.3.3-blue.svg?style=flat" alt="0.3.3"></a>
  <a href="https://goreportcard.com/report/github.com/zigenzoog/gonn"><img src="https://goreportcard.com/badge/github.com/zigenzoog/gonn" alt="Code Status" /></a>
</div>

# About
gonn - Neural Network for Golang

# Install

    $ go get -u github.com/zigenzoog/gonn

# Getting Started

```go
package main

import "github.com/zigenzoog/gonn/pkg/nn"

func main() {
	// New returns a new neural network
	// instance with the default parameters,
	// same n := nn.New("perceptron").
	n := nn.New()

	// The neuron bias, false or true.
	n.SetNeuronBias(true)    

	// Array of the number of neurons in each hidden layer.
	n.SetHiddenLayer(3)           

	// Activation function mode.
	n.SetActivationMode(nn.ModeSIGMOID)

	// The mode of calculation of the total error.
	n.SetLossMode(nn.ModeMSE)

	// Minimum (sufficient) limit of the average of the error during training.
	n.SetLossLimit(.001)

	// Learning coefficient (greater than 0 and less than or equal to 1).
	n.SetLearningRate(nn.DefaultRate)

	// Dataset.
	input  := []float64{1, 1}
	target := []float64{0}

	// Training dataset.
	_, _ = n.Train(input, target)

	// Writing the neural network configuration to a file.
	_ = n.WriteConfig("perceptron.json")
}
```

# Documentation
More documentation is available at the [gonn website](https://zigenzoog.github.io/gonn/) or on [pkg.go.dev](https://pkg.go.dev/github.com/zigenzoog/gonn).

# Examples
You can find examples of neural networks in the [examples directory](https://github.com/zigenzoog/gonn/tree/master/examples/).
