# E16 — MNIST CNN

Demonstrates 2-D convolutional layers (Conv2D → MaxPool2D → Flatten2D) followed
by a fully-connected classifier trained on the MNIST handwritten digit dataset.

## Architecture

```
Input  1×28×28  (784 flat)
Conv2D   8 filters, 3×3, PadSame, bias   →  8×28×28 = 6272
MaxPool2D  2×2                           →  8×14×14 = 1568
Flatten2D                                →  1568
Dense  128  ReLU
Dense   64  ReLU
Output  10  Sigmoid
```

## Quick start

```sh
# Step 1 — download IDX files (≈ 10 MB, one-time):
go run ./examples/mnist_cnn/ -download

# Step 2 — train on 200 samples:
go run ./examples/mnist_cnn/ -n 200
```

`-download` fetches `train-images-idx3-ubyte` and `train-labels-idx1-ubyte`
from the Google MNIST mirror, decompresses them in place, and exits.
Re-running `-download` skips files that already exist.

## Flags

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-download` | `false` | Fetch and decompress IDX files, then exit |
| `-images` | `train-images-idx3-ubyte` | Path to the images IDX3 file |
| `-labels` | `train-labels-idx1-ubyte` | Path to the labels IDX1 file |
| `-n` | `200` | Number of training samples to load |

Increase `-n` to 10 000–60 000 for meaningful accuracy.

## Dataset

The MNIST IDX files are not bundled in this repository.
`-download` uses the Google mirror automatically. Alternative sources:

- <http://yann.lecun.com/exdb/mnist/> (canonical; sometimes unavailable)
- <https://storage.googleapis.com/cvdf-datasets/mnist/> (Google mirror; used by `-download`)

To place files manually, download the `.gz` archives, decompress them, and
pass the resulting paths via `-images` / `-labels`.

## Key API

```go
nn.WithInputShape[float32](1, 28, 28)          // declare CHW layout
nn.WithConv2D[float32](8, 1, 3, 3, 1, 1, conv.PadSame, true)
nn.WithMaxPool2D[float32](2, 2)
nn.WithFlatten2D[float32]()
```

`WithInputShape` tells `compile()` the (channels, height, width) layout of the
flat 784-element input vector so the Conv2D prefix can resolve output shapes
before the first training step.
