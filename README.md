# GoNN

[![Go API Reference](https://pkg.go.dev/badge/github.com/teratron/gonn.svg)](https://pkg.go.dev/github.com/teratron/gonn?tab=doc)
[![0.5.0](https://img.shields.io/badge/version-0.5.0-blue.svg?style=flat)](https://github.com/teratron/gonn/releases/tag/v0.5.0)
[![Go Code Status](https://goreportcard.com/badge/github.com/teratron/gonn)](https://goreportcard.com/report/github.com/teratron/gonn)

---

## Description

Neural network library for go.

## Visuals

## Installation

```shell
go get -u github.com/teratron/gonn
```

## Usage

```go
package main

import "github.com/teratron/gonn/pkg/nn"

func main() {
	// New returns a perceptron neural network
	// instance with the default parameters.
	n := nn.New()

	// Dataset.
	input := []float64{.27, .31}
	target := []float64{.7}

	// Training dataset.
	_, _ = n.Train(input, target)
}
```

## Documentation

### Properties of Perceptron Neural Network

#### _Name_

Neural network architecture name (required field for a config).

#### _Bias_

The neuron bias, false or true (required field for a config).

#### _HiddenLayer_

Array of the number of neurons in each hidden layer.

#### _ActivationMode_

ActivationMode function mode (required field for a config).

| Code | Activation | Description                              |
|------|------------|------------------------------------------|
| 0    | LINEAR     | Linear/identity                          |
| 1    | RELU       | ReLu (rectified linear unit)             |
| 2    | LEAKYRELU  | Leaky ReLu (leaky rectified linear unit) |
| 3    | SIGMOID    | Logistic, a.k.a. sigmoid or soft step    |
| 4    | TANH       | TanH (hyperbolic tangent)                |

#### _LossMode_

The mode of calculation of the total error.

| Code | Loss   | Description             |
|------|--------|-------------------------|
| 0    | MSE    | Mean Squared Error      |
| 1    | RMSE   | Root Mean Squared Error |
| 2    | ARCTAN | Arctan                  |
| 3    | AVG    | Average                 |

#### _LossLimit_

Minimum (sufficient) limit of the average of the error during training.

#### _Rate_

Learning coefficient (greater than 0.0 and less than or equal to 1.0).

More documentation is available at the [gonn website](https://teratron.github.io/gonn) or
on [pkg.go.dev](https://pkg.go.dev/github.com/teratron/gonn).

## Examples

You can find examples of neural networks in the [example's directory](examples).

- [perceptron](examples/perceptron)
- [linear](examples/linear)
- [query](examples/query)
- [and_train](examples/and_train)
- [json](examples/json)

## Support

## Roadmap

## Contributing

## Authors and acknowledgment

## License

[MIT License](LICENSE).

## Project status

_Project at the initial stage._

See the latest [commits](https://github.com/teratron/gonn/commits/master).

---