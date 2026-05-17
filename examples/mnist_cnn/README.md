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

## Running

```sh
go run ./examples/mnist_cnn/ \
    -images train-images-idx3-ubyte \
    -labels train-labels-idx1-ubyte \
    -n 200
```

The `-n` flag limits the number of training samples loaded from the IDX file
(default 200). Increase to 10 000–60 000 for meaningful accuracy.

## Dataset

The MNIST IDX files are not bundled in this repository. Download them from:

- <http://yann.lecun.com/exdb/mnist/> (canonical mirror)
- <https://storage.googleapis.com/cvdf-datasets/mnist/> (Google mirror)

Required files (place in the working directory or pass via flags):

| Flag | File |
| :--- | :--- |
| `-images` | `train-images-idx3-ubyte` |
| `-labels` | `train-labels-idx1-ubyte` |

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
