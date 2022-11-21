# GONN

<div>
  <a href="https://pkg.go.dev/github.com/zigenzoog/gonn?tab=doc" title="Go API Reference" rel="nofollow"><img src="https://pkg.go.dev/badge/github.com/zigenzoog/gonn.svg" alt="Go API Reference"></a>
  <a href="https://github.com/zigenzoog/gonn/releases/tag/v0.4.0" title="0.4.0" rel="nofollow"><img src="https://img.shields.io/badge/version-0.4.0-blue.svg?style=flat" alt="0.4.0"></a>
  <a href="https://goreportcard.com/report/github.com/zigenzoog/gonn"><img src="https://goreportcard.com/badge/github.com/zigenzoog/gonn" alt="Code Status" /></a>
</div>

## About

**gonn** - Neural Network Library

## Install

```shell
$ go get -u github.com/zigenzoog/gonn
```

## Getting Started

```go
package main

import "github.com/zigenzoog/gonn/pkg/nn"

func main() {
	// New returns a new neural network
	// instance with the default parameters.
	n := nn.New()

	// Dataset.
	input  := []float64{.27, .31}
	target := []float64{.7}

	// Training dataset.
	_, _ = n.Train(input, target)
}
```

## Documentation

More documentation is available at the [gonn website](https://zigenzoog.github.io/gonn) or
on [pkg.go.dev](https://pkg.go.dev/github.com/zigenzoog/gonn).

## Examples

You can find examples of neural networks in
the [example's directory](https://github.com/zigenzoog/gonn/tree/master/examples).
