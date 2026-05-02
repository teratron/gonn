# Example: Progress Callbacks

This example demonstrates how to use progress callbacks in `gonn`. It trains a canonical XOR network and utilizes both epoch and batch callbacks to monitor the training progress.

## Features

- **Epoch Callback**: Prints the loss every 100 epochs.
- **Batch Callback**: Throttled to log only at the start of each new epoch stripe to keep output readable.

## Running the example

```bash
go run main.go
```
