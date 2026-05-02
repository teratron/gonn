# Example: Network Persistence

This example demonstrates the round-trip integrity of saving and reloading a neural network. It trains an XOR network, exports its configuration and weights to JSON files, and then reloads them into a fresh network to verify the outputs match.

## Features

- Usage of `pkg/persistence` for on-disk storage.
- Manual weight extraction and installation (internal hooks coming in future versions).
- Verification of post-reload Query output within tolerance.

## Running the example

```bash
go run main.go
```
