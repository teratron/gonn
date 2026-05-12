# Example: Progress Callbacks

This example demonstrates how to use progress callbacks in `gonn`. It trains a canonical XOR network and utilizes both epoch and batch callbacks to monitor the training progress.

## Network Topology

This example uses the `PresetXOR` helper, which configures a standard XOR network:

- **Input**: 2 neurons
- **Hidden Layer**: 4 neurons with `Sigmoid` activation
- **Output**: 1 neuron with `Sigmoid` activation

## Features

- **Epoch Callback**: Demonstrates `WithEpochCallback` to monitor loss at the end of each epoch (e.g., logging every 100 iterations).
- **Batch Callback**: Demonstrates `WithBatchCallback` for granular monitoring of training progress within an epoch.
- **Preset Usage**: Shows how to combine presets like `PresetXOR` with additional custom options.

## Running the example

```bash
go run ./examples/callbacks
```

## Running the tests

```bash
go test ./examples/callbacks
```
