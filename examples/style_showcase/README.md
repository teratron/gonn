# Example: Style Showcase

This example provides a side-by-side comparison of the three different ways to construct an identical network in `GoNN`.

## Network Topology

All three styles construct the same canonical XOR network:

- **Input**: 2 neurons
- **Hidden Layer**: 4 neurons with `Sigmoid` activation
- **Output**: 1 neuron with `Sigmoid` activation

## Features

- **Side-by-Side Comparison**: Demonstrates the three construction styles:
  1. **Builder API**: Fluent, chainable interface.
  2. **Functional Options API**: Declarative options passed to `nn.New`.
  3. **Preset Bundle**: Using pre-configured named networks (e.g., `PresetXOR`).
- **Style Equivalence**: Proves that all styles converge to the same network behavior under identical hyperparameters.
- **Composition**: Shows how presets can be customized with additional functional options.

## Running the example

```bash
go run ./examples/style_showcase
```

## Running the tests

```bash
go test ./examples/style_showcase
```
