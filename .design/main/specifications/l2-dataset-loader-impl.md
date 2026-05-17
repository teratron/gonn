# Dataset Loader — Go Implementation

**Version:** 0.1.1
**Status:** Stable
**Layer:** implementation
**Implements:** l1-dataset-formats.md

## Overview

Go realization of the dataset formats contract from `l1-dataset-formats.md`.
Adds `IDXReader` and `MNISTLoader[T]` to `pkg/dataset/` as a new file `mnist.go`.
Adds `AndTrain[T]` as a method on `*NN[T]` in `pkg/nn/andtrain.go`. Both integrate
with existing interfaces so callers need no new imports beyond `pkg/dataset` and `pkg/nn`.

## Related Specifications

- [l1-dataset-formats.md](l1-dataset-formats.md) — L1 parent (Draft v0.1.0)
- [l2-streaming-impl.md](l2-streaming-impl.md) — `DataSet[T]` interface; MNISTLoader must implement it
- [l2-nn-facade.md](l2-nn-facade.md) — `AndTrain` is a new method on `*NN[T]`
- [l2-training-loop.md](l2-training-loop.md) — `AndTrain` re-enters the same convergence loop
- [l2-control-impl.md](l2-control-impl.md) — state machine reset between `Train` and `AndTrain`
- [l2-callbacks-impl.md](l2-callbacks-impl.md) — callbacks remain active across `AndTrain` calls
- [l2-errors-impl.md](l2-errors-impl.md) — new `ErrIDXMagic`, `ErrMNISTRecordMismatch` sentinels

## 4. Invariant Compliance

| L1 Invariant | Implementation |
| :--- | :--- |
| FMT-1 IDX magic number validation | `readIDXHeader` checks bytes 0-1 == 0x00, dtype byte in known set; returns `ErrIDXMagic` otherwise |
| FMT-2 Dimension header (big-endian int32) | `binary.BigEndian.Uint32` for each dimension; product computed and checked |
| FMT-3 Row-major element order | Elements read sequentially; sample = `io.ReadFull` of `product(dims[1:])` elements |
| FMT-4 DataSet[T] streaming interface | `MNISTLoader[T]` implements `dataset.DataSet[T]`; bounded memory via per-record reads |
| FMT-5 Image+label count parity | Constructor reads both headers, returns `ErrMNISTRecordMismatch` if counts differ |
| FMT-6 AndTrain fresh convergence | `AndTrain` resets `minLoss`, `minIter` counters; does NOT reset weights |
| FMT-7 Callback continuity | `nn.cfg.Callbacks` is unchanged between `Train` and `AndTrain` |
| FMT-8 State machine reset | `AndTrain` checks `ctrl.State() == Idle`; returns `ErrNetworkRunning` otherwise |

## 5. Detailed Design

### 5.1 Package Structure

```plaintext
pkg/dataset/
└── mnist.go      # IDXReader, MNISTLoader[T], readIDXHeader

pkg/nn/
└── andtrain.go   # (*NN[T]).AndTrain(ds dataset.DataSet[T], opts ...Option[T]) (Result[T], error)
```

### 5.2 IDX Reader

```
// [REFERENCE]
type idxHeader struct {
    dtype  byte
    ndim   byte
    dims   []int32   // len = ndim
    count  int       // dims[0] = number of records
    recLen int       // product(dims[1:]) = flat vector length per record
}

func readIDXHeader(r io.Reader) (idxHeader, error)
// Reads 4-byte magic + ndim*4 dimension bytes.
// Returns ErrIDXMagic on invalid magic.

type IDXReader struct {
    r      io.Reader
    hdr    idxHeader
    buf    []byte    // reused read buffer of len hdr.recLen
}
// Next() ([]byte, error) — reads next record; returns io.EOF at end
```

### 5.3 MNISTLoader

```
// [REFERENCE]
type MNISTLoader[T utils.Float] struct {
    images    *IDXReader
    labels    *IDXReader
    norm      T          // pixel normalization divisor (255.0 for uint8 images)
    imgShape  [3]int     // [C, H, W]; zero means "infer from IDX dims"
}

// implements dataset.DataSet[T]:
// Next() (input []T, target []T, err error)
// input  = image pixels normalised to [0,1], length = C*H*W
// target = one-element slice with raw label cast to T (caller encodes one-hot)
```

### 5.3a WithImageShape

`WithImageShape` is a functional option for `MNISTLoader[T]` that overrides the
image shape used during the training run. When provided, `nn.compile()` reads
`Channels`, `Height`, `Width` from the loader and sets `Config.InputC / InputH /
InputW` automatically — the user does not need to call `WithInputShape` separately.

```
// [REFERENCE]
func WithImageShape[T utils.Float](channels, height, width int) MNISTOption[T]
// Validates: channels >= 1, height >= 1, width >= 1, channels*height*width == images.RecordLen().
// Stored as loader.imgShape = [3]int{channels, height, width}.
```

The `MNISTLoader[T]` exposes its resolved shape via:

```
// ImageShape returns (channels, height, width) as set by WithImageShape, or
// inferred from IDX header dims when ndim == 3 and WithImageShape was not used.
// Returns (1, S, S) for flat ndim==1 files with perfect-square record length.
func (m *MNISTLoader[T]) ImageShape() (channels, height, width int)
```

`compile()` integration point: when `cfg.InputC == 0` and the first element of
`cfg.ConvPrefix` is a `*conv.Conv2D[T]`, `setupConv2DShapes` calls `ImageShape()`
on the dataset (if it satisfies the optional `ImageShaper` interface below):

```
// ImageShaper is an optional interface satisfied by MNISTLoader and any
// dataset adapter that carries a known (C, H, W) image shape.
type ImageShaper interface {
    ImageShape() (channels, height, width int)
}
```

### 5.4 AndTrain

`AndTrain` re-enters the training loop with a fresh data source. It accepts the same
variadic `Option[T]` as `Train` to allow overriding convergence parameters (e.g., a
lower learning rate for fine-tuning) without changing persistent config.

```
// [REFERENCE]
func (nn *NN[T]) AndTrain(ds dataset.DataSet[T], opts ...Option[T]) (Result[T], error)
// 1. Check ctrl.State() == Idle  → ErrNetworkRunning if not
// 2. Apply opts to a shallow copy of cfg (non-destructive to base config)
// 3. Re-enter convergence loop with ds as data source
// 4. Callbacks remain; OnTrainEnd fires at completion
// 5. ctrl.State() returns to Idle on exit
```

### 5.5 Error Handling

- `ErrIDXMagic` — bytes 0-1 not 0x00 or dtype byte unrecognised.
- `ErrMNISTRecordMismatch` — image file count ≠ label file count.
- `ErrNetworkRunning` — `AndTrain` called while `ctrl.State() != Idle`.

## 6. Implementation Notes

1. `pkg/dataset/mnist.go` — `IDXReader` first (no generics needed), then `MNISTLoader[T]`.
2. Pixel normalisation divisor is a constructor parameter (default 255.0); keeps loader general.
3. `pkg/nn/andtrain.go` — thin wrapper; share the inner `train()` helper with `Train`.
4. `pkg/utils/errors.go` — add `ErrIDXMagic`, `ErrMNISTRecordMismatch`, `ErrNetworkRunning`.
5. Examples E06 and E10 are the acceptance tests; wire them after unit tests pass.

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-05-12 | Initial Draft — IDXReader, MNISTLoader[T], AndTrain method. |
| 0.1.0 | 2026-05-14 | Promoted Draft → Stable via magic.task Trust Mode (parent `l1-dataset-formats` Stable; MVC satisfied). |
| 0.1.1 | 2026-05-17 | Added §5.3a WithImageShape — MNISTLoader 2-D shape option, ImageShape() accessor, ImageShaper interface, compile() integration point (Phase 14 / CONV2D-5). |
