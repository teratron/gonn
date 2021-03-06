<div style="text-align: center">
  <!--a href="https://pkg.go.dev/github.com/zigenzoog/gonn?tab=doc" title="Go API Reference" rel="nofollow"><img src="https://img.shields.io/badge/go-documentation-blue.svg?style=flat" alt="Go API Reference"></a-->
  <a href="https://pkg.go.dev/github.com/zigenzoog/gonn"><img src="https://pkg.go.dev/badge/github.com/zigenzoog/gonn.svg" alt="Go Reference"></a>
  <a href="https://github.com/zigenzoog/gonn/releases/tag/v0.2.3" title="0.2.3" rel="nofollow"><img src="https://img.shields.io/badge/version-0.2.3-blue.svg?style=flat" alt="0.2.2"></a>
  <a href="https://goreportcard.com/report/github.com/zigenzoog/gonn"><img src="https://goreportcard.com/badge/github.com/zigenzoog/gonn" alt="Code Status" /></a>

  <!--a href="https://travis-ci.org/zigenzoog/gonn"><img src="https://travis-ci.org/zigenzoog/gonn.svg" alt="Build Status" /></a-->
  <!--a href='https://coveralls.io/github/zigenzoog/gonn?branch=develop'><img src='https://coveralls.io/repos/github/zigenzoog/gonn/badge.svg?branch=develop' alt='Coverage Status' /></a-->
  <!--a href='https://sourcegraph.com/github.com/zigenzoog/gonn?badge'><img src='https://sourcegraph.com/github.com/zigenzoog/gonn/-/badge.svg' alt='Used By' /></a-->
</div>

# About
gonn - Neural Network for Golang

# Install

    $ go get github.com/zigenzoog/gonn

# Getting Started

```go
package main

import (
	"path/filepath"

	"github.com/teratron/gonn/pkg/nn"
)

func main() {
	// New returns a new neural network
	// instance with the default parameters,
	// same n := nn.New(nn.Perceptron())
	n := nn.New()

	n.SetNeuronBias(true)
	n.SetHiddenLayer(3)
	n.SetActivationMode(nn.ModeSIGMOID)
	n.SetLossMode(nn.ModeMSE)
	n.SetLossLimit(.0001)
	n.SetLearningRate(nn.DefaultRate)

	input  := []float64{1, 1}
	target := []float64{0}

	_, _ = n.Train(input, target)

	_ = n.Write(nn.JSON(filepath.Join("perceptron.json")))
}
```

# Documentation
More documentation is available at the [gonn website](https://zigenzoog.github.io/gonn/) or on [pkg.go.dev](https://pkg.go.dev/github.com/zigenzoog/gonn).

# Examples
You can find examples of neural networks in the [examples directory](https://github.com/zigenzoog/gonn/tree/master/examples/).
