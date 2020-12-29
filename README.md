<div style="text-align: center">
  <!--
  <a href="https://pkg.go.dev/zigenzoog/gonn?tab=doc" title="Go API Reference" rel="nofollow"><img src="https://img.shields.io/badge/go-documentation-blue.svg?style=flat" alt="Go API Reference"></a>
  <a href="https://github.com/zigenzoog/gonn/releases/tag/v0.0.1" title="0.0.1 Release" rel="nofollow"><img src="https://img.shields.io/badge/version-0.0.1-blue.svg?style=flat" alt="0.0.1 release"></a>
  <a href="https://goreportcard.com/report/zigenzoog/gonn"><img src="https://goreportcard.com/badge/zigenzoog/gonn" alt="Code Status" /></a>
  -->
  <!--a href="https://travis-ci.org/zigenzoog/gonn"><img src="https://travis-ci.org/zigenzoog/gonn.svg" alt="Build Status" /></a-->
  <!--a href='https://coveralls.io/github/zigenzoog/gonn?branch=develop'><img src='https://coveralls.io/repos/github/zigenzoog/gonn/badge.svg?branch=develop' alt='Coverage Status' /></a-->
  <!--a href='https://sourcegraph.com/github.com/zigenzoog/gonn?badge'><img src='https://sourcegraph.com/github.com/zigenzoog/gonn/-/badge.svg' alt='Used By' /></a-->
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
	// same n := nn.New(nn.Perceptron())
	n := nn.New()

	// The neuron bias, false or true
	n.SetNeuronBias(true)

	// Array of the number of neurons in each hidden layer
	n.SetHiddenLayer(3, 2)

	// Activation function mode
	n.SetActivationMode(nn.ModeSIGMOID)

	// The mode of calculation of the total error
	n.SetLossMode(nn.ModeMSE)

	// Minimum (sufficient) limit of the average of the error during training
	n.SetLossLimit(.1)

	// Learning coefficient, from 0 to 1
	n.SetLearningRate(nn.DefaultRate)

	// Training dataset
	input  := []float64{1, 0}
	target := []float64{0, 1}

	// Training
	_, _ = n.Train(input, target)

	// Writing the neural network configuration to a file
	n.Write(nn.JSON("perceptron.json"))
}
```

# Documentation
More documentation is available at the [gonn website](https://zigenzoog.github.io/gonn/) or on [pkg.go.dev](https://pkg.go.dev/zigenzoog/gonn?tab=doc).

# Examples
You can find examples of neural networks in the [examples directory](https://github.com/zigenzoog/gonn/tree/master/examples/).
