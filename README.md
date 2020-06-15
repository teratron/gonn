<p style="text-align: center">
  <a href="https://pkg.go.dev/zigenzoog/gonn?tab=doc" title="Go API Reference" rel="nofollow"><img src="https://img.shields.io/badge/go-documentation-blue.svg?style=flat" alt="Go API Reference"></a>
  <a href="https://github.com/zigenzoog/gonn/releases/tag/v0.0.1" title="0.0.1 Release" rel="nofollow"><img src="https://img.shields.io/badge/version-0.0.1-blue.svg?style=flat" alt="0.0.1 release"></a>
  <br />
  <a href="https://goreportcard.com/report/zigenzoog/gonn"><img src="https://goreportcard.com/badge/zigenzoog/gonn" alt="Code Status" /></a>
  <a href="https://travis-ci.org/zigenzoog/gonn"><img src="https://travis-ci.org/zigenzoog/gonn.svg" alt="Build Status" /></a>
  <a href='https://coveralls.io/github/zigenzoog/gonn?branch=develop'><img src='https://coveralls.io/repos/github/zigenzoog/gonn/badge.svg?branch=develop' alt='Coverage Status' /></a>
  <a href='https://sourcegraph.com/github.com/zigenzoog/gonn?badge'><img src='https://sourcegraph.com/github.com/zigenzoog/gonn/-/badge.svg' alt='Used By' /></a>
</p>

# About
gonn - Neural Network for Golang

# Install

    $ go get -u github.com/zigenzoog/gonn

# Initializing

```go
package main

import (
    "github.com/zigenzoog/gonn/nn"
)

func main() {
    // Creat new Neural Network
    n := nn.New()   // Default Feed Forward Neural Network, same f := nn.New().FeedForward()
    // or
    n = nn.New().Perceptron()
    // or
    n = nn.New().RadialBasis()
    
    // Set
    n.SetBias(1)
    // or
    n.Set(nn.Bias(0))

    // Get
    b := n.Bias()
    // or
    b = n.GetBias()
}
```

# Getting Started
And you can run that simply as:

    $ go run main.go

# Documentation


# Examples
You can find examples of neural networks in the [examples repository](https://github.com/zigenzoog/gonn-examples/).
