# E06 — MNIST Digit Recognition

Demonstrates loading the MNIST dataset via `IDXReader`/`MNISTLoader`,
building a multi-hidden-layer network with BatchNorm, and using `AndTrain`
to extend training across two epochs while preserving the learned weights.

## Architecture

```
Input: 784  (28×28 flattened pixels, normalised to [0, 1])
  └─ Hidden: 128 (ReLU)
      └─ BatchNorm
          └─ Hidden: 64 (ReLU)
              └─ Output: 10 (Sigmoid, one node per digit class 0–9)
```

## Prerequisites: downloading MNIST files

The IDX files are **not bundled** in this repository. Download them from one
of the mirrors below and decompress them (`.gz`) before running.

| File | Description |
|------|-------------|
| `train-images-idx3-ubyte` | 60 000 training images (28×28 px, uint8) |
| `train-labels-idx1-ubyte` | 60 000 training labels (0–9, uint8) |

Mirror: <https://ossci-datasets.s3.amazonaws.com/mnist/>

```bash
# Example using curl (macOS / Linux):
curl -O https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
curl -O https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
gzip -d train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz
```

## Running the example

```bash
go run ./examples/mnist/ \
    -images train-images-idx3-ubyte \
    -labels train-labels-idx1-ubyte \
    -n 500
```

Flag `-n` controls how many training samples to load (default 500). Increase
to 60 000 for the full training set (slower but higher accuracy).

Expected output (values will vary with random weight initialisation):

```
Loaded 500 MNIST samples
Epoch 1 (Fit):      1 iteration(s), loss=0.0892
Epoch 2 (AndTrain): 1 iteration(s), loss=0.0784
Sample 0: predicted=5, actual=5
```

## Concepts illustrated

### IDX binary format

MNIST files use the IDX format: a 4-byte magic number, followed by dimension
sizes (4 bytes each, big-endian), followed by raw row-major data. The
`IDXReader` in `pkg/dataset/mnist.go` validates the magic bytes and exposes
the data as a streaming `Next()` interface.

### One-hot encoding

`MNISTLoader.Next()` returns raw class labels as `[]float32{label}` (a single
value 0–9). A 10-output network needs a *one-hot* target vector where only
the index matching the true class is 1:

```
label = 5  →  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
```

The `oneHot` helper in `main.go` performs this conversion. `argmax` then
reverses it on the output side to recover the predicted digit.

### BatchNorm and why it helps

`WithBatchNorm(0)` inserts a BatchNorm layer after hidden layer 0 (the 128-
neuron layer). During training BatchNorm normalises each activation to zero
mean and unit variance across the current mini-batch, then applies learnable
affine parameters (scale γ, shift β). Benefits:

- **Faster convergence**: the gradient signal is less sensitive to weight
  scale, so larger learning rates can be used safely.
- **Regularisation effect**: normalisation adds noise that acts similarly to
  dropout, reducing overfitting.
- **Training/eval distinction**: `BatchNorm` tracks exponential moving
  averages of mean and variance during training; at inference it uses these
  frozen statistics (switched via `SetTrain(false)`).

### AndTrain for epoch continuation

`Fit` runs one full epoch (MaxIterations=1 means one pass over all samples).
`AndTrain` then runs a second epoch at a reduced learning rate (0.001 vs
0.01) *without resetting the learned weights* — it only resets the
convergence counters. This mirrors common practice in machine learning:
train aggressively first, then fine-tune at a lower rate once the network
has settled into a reasonable region.
