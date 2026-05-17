# 2-D Convolutional Layers

**Version:** 0.2.0
**Status:** Draft
**Layer:** concept

## Overview

Defines the contract for 2-D convolutional layers that process image-shaped tensors and
spatial feature maps. Covers four composable primitives: `Conv2D` (parameterised
sliding-window feature extraction across height and width), `MaxPool2D` / `AvgPool2D`
(non-parametric spatial downsampling), and `Flatten2D` (shape collapse to a 1-D vector
compatible with existing Dense layers). Companion specification to the 1-D contract in
[l1-conv-layers.md](l1-conv-layers.md) — same invariant philosophy generalised to two
spatial axes. Scope is intentionally limited to 2-D; 3-D volumetric and grouped/depthwise
convolutions are deferred to future amendments.

## Related Specifications

- [l1-conv-layers.md](l1-conv-layers.md) — sibling 1-D contract; invariant naming style (CONV-N) is reused as `CONV2D-N`
- [l1-neural-network-architecture.md](l1-neural-network-architecture.md) — topology model; conv-2d layers extend the layer category hierarchy
- [l2-layer-types.md](l2-layer-types.md) — L2 layer hierarchy; Conv2D joins Input/Dense/Conv1D/Output
- [l1-normalization-layers.md](l1-normalization-layers.md) — peer composable primitive; Conv2D+BatchNorm is the canonical CV pattern
- [l1-training-semantics.md](l1-training-semantics.md) — conv-2d layers participate in the forward/backward convergence loop
- [l1-weight-initialization.md](l1-weight-initialization.md) — kernel weights require initialization (Xavier/He/Random)
- [l1-network-persistence.md](l1-network-persistence.md) — kernel weights + biases must survive JSON round-trip
- [l1-dynamic-topology.md](l1-dynamic-topology.md) — conv-2d layers can be inserted/removed at topology safe-points
- [l1-dataset-formats.md](l1-dataset-formats.md) — image-shaped tensors enter the network from this layer; layout contract anchors here
- [l1-performance-contract.md](l1-performance-contract.md) — Conv2D hotspot governed by PERF-1/PERF-4 (zero-alloc backward, pooled scratch buffers)

## 1. Motivation

GoNN currently supports Dense and 1-D convolutional layers. The 1-D primitive enables
sequence and time-series workloads, but leaves an explicit gap for 2-D image data
flagged in `l1-conv-layers.md` §7. The existing MNIST loader from
`l1-dataset-formats.md` already emits 28×28 byte tensors that today flow into the
network only as flattened 784-element vectors, discarding spatial structure.

Adding `Conv2D` / `MaxPool2D` / `AvgPool2D` / `Flatten2D` as first-class layer types:

1. Closes the deferred 2-D scope from `l1-conv-layers.md` §7 with its own L1 contract.
2. Enables canonical computer-vision examples (MNIST CNN, CIFAR-class architectures)
   without requiring users to pre-flatten or hand-roll convolutions outside the library.
3. Reuses the same `Layer[T]` interface and `compile()` prepend strategy already proven
   by the 1-D conv stack — topology model unchanged, only the layer catalog grows.
4. Provides the foundation primitive for future ResNet-style skip connections and 2-D
   batch normalisation extensions.

## 2. Constraints & Assumptions

- **CONV2D-C1 (2-D spatial only)**: Input is a tensor with explicit shape
  `(channels, height, width)` (or its row-major flattening). Output preserves the same
  three-axis structure. No 3-D / volumetric / time-major sequences.
- **CONV2D-C2 (Dense integration)**: Conv2D output feeds Dense layers through
  `Flatten2D`. No separate CNN-only execution mode — the standard `compile()` pipeline
  remains canonical.
- **CONV2D-C3 (Padding modes)**: Two modes for v1.0: `PadValid` (no padding, output
  smaller than input) and `PadSame` (zero-padded, output size ≈ input divided by
  stride). Asymmetric and reflect padding are deferred.
- **CONV2D-C4 (Per-axis stride ≥ 1)**: `(stride_h, stride_w)` integer pair. Dilated
  convolutions are deferred to a future amendment.
- **CONV2D-C5 (Pool is non-parametric)**: `MaxPool2D` and `AvgPool2D` carry no trainable
  weights. Non-overlapping windows by default; overlapping pooling is a v1.x extension.
- **CONV2D-C6 (Flatten2D is row-major)**: Output is a single 1-D vector concatenating
  channels in declared layout order. Output size = `outChannels × outH × outW`.
- **CONV2D-C7 (Bias optional)**: Per-filter scalar bias is opt-in; default is no bias.
  Bias adds one value per output channel (broadcast across height and width).
- **CONV2D-C8 (Channels in ≥ 1)**: Multi-channel input is supported from day one
  (1-channel MNIST and 3-channel RGB use the same code path). The 1-D conv contract
  exposed only single-channel filters; the 2-D contract makes channels explicit.
- **CONV2D-C9 (Tensor layout)**: Canonical layout is `CHW` (channels-first,
  row-major). For an input `(C_in, H, W)`, element `(c, y, x)` resides at flat
  index `c * H * W + y * W + x`. Filters are stored as flat `[]T` of length
  `F * C_in * K_h * K_w` in filter-major order — filter `f`, input channel `c`,
  row `i`, column `j` resides at `f * (C_in * K_h * K_w) + c * (K_h * K_w) + i * K_w + j`.
  This layout matches PyTorch/cuDNN convention, preserves per-channel cache
  locality for compositions with `l1-normalization-layers.md` primitives, and
  generalises the Conv1D Variant A filter-major storage pattern by adding the
  inner-channel axis. Dataset loaders MAY accept HWC byte input but MUST
  internally transpose to CHW before handing the tensor to the network.

## 3. Core Invariants

- **CONV2D-1 (Output shape)**: Given input shape `(C_in, H, W)`, kernel `(K_h, K_w)`,
  stride `(S_h, S_w)`, padding mode `P`, and `F` filters:
  - `PadValid`: `H_out = floor((H - K_h) / S_h) + 1`, `W_out = floor((W - K_w) / S_w) + 1`.
    Requires `H ≥ K_h` and `W ≥ K_w`.
  - `PadSame`: `H_out = ceil(H / S_h)`, `W_out = ceil(W / S_w)`.
  - Output tensor shape is `(F, H_out, W_out)`.
- **CONV2D-2 (Filter weights)**: Each of the `F` filters is an independent weight
  tensor of shape `(C_in, K_h, K_w)`. Filters and optional per-filter scalar biases
  are the only trainable parameters.
- **CONV2D-3 (Forward determinism)**: Given identical weights, biases, and input,
  the forward pass is deterministic; no stochasticity in inference or training-forward.
- **CONV2D-4 (Gradient correctness)**: Gradient of the loss w.r.t. each filter weight
  equals the cross-correlation of the input tensor (per input channel) with the
  upstream gradient (per output channel), summed over output spatial positions. This
  invariant is necessary for convergence and matches the 1-D contract generalised to
  two spatial axes.
- **CONV2D-5 (Pool semantics)**: `MaxPool2D` selects the maximum value in each
  non-overlapping `(P_h, P_w)` window, per channel independently. `AvgPool2D` computes
  the arithmetic mean. Neither has trainable parameters. Backward pass routes the
  upstream gradient only to the argmax position (max) or evenly across the window
  (avg).
- **CONV2D-6 (Flatten2D)**: Output is a single 1-D vector. Element order follows the
  declared tensor layout (CONV2D-C9); the order MUST be deterministic and documented
  so persistence and gradient routing remain stable across versions.
- **CONV2D-7 (Weight persistence)**: Conv2D kernel weights, biases, and shape metadata
  (`numFilters`, `inChannels`, `kernelH`, `kernelW`, `strideH`, `strideW`, `padding`,
  `layout`) MUST survive a JSON round-trip identically (`l1-network-persistence.md`
  contract). Pool and Flatten2D layers persist only their configuration scalars.
- **CONV2D-8 (Weight initialization)**: Kernel weights MUST use one of the strategies
  from `l1-weight-initialization.md` (Xavier, He, or Random). Fan-in for He/Xavier is
  `C_in × K_h × K_w` (the input volume per output activation).
- **CONV2D-9 (Layer interface compatibility)**: `Conv2D`, `MaxPool2D`, `AvgPool2D`,
  and `Flatten2D` MUST satisfy the same layer interface as `Dense`, `Conv1D`,
  `Input`, and `Output`, so the network graph can iterate them uniformly without
  type switches or specialised dispatch paths.

> L2 spec cannot reach RFC status until all nine invariants here are addressed in its
> `Invariant Compliance` section.

## 4. Detailed Design

### 4.1 Layer Composition

```mermaid
graph LR
    In[Input Layer] --> C2[Conv2D]
    C2 --> N[BatchNorm-2D optional]
    N --> P2[MaxPool2D]
    P2 --> C2b[Conv2D 2nd block]
    C2b --> P2b[MaxPool2D]
    P2b --> F[Flatten2D]
    F --> D[Dense]
    D --> O[Output]
```

Typical stack: `Input → Conv2D(F1, KxK) → MaxPool2D(PxP) → Conv2D(F2, KxK) → MaxPool2D(PxP) → Flatten2D → Dense → Output`.

### 4.2 Shape Propagation Example

Given input shape `(1, 28, 28)` (single-channel MNIST), Conv2D with
`F = 8`, `K = 3`, `S = 1`, `PadValid`:

- `H_out = floor((28 - 3)/1) + 1 = 26`, `W_out = 26`
- Conv2D output shape: `(8, 26, 26)` → 5408 elements
- MaxPool2D(2×2): `(8, 13, 13)` → 1352 elements
- Conv2D(`F = 16`, `K = 3`, `S = 1`, `PadValid`): `(16, 11, 11)` → 1936 elements
- MaxPool2D(2×2, `PadSame`): `(16, 6, 6)` → 576 elements
- Flatten2D → 576-element vector suitable as Dense input

### 4.3 Gradient Flow

Backpropagation through the conv-2d stack requires three pieces, each generalising
the 1-D analogue from `l1-conv-layers.md` §4.3:

1. **Flatten2D → Pool gradient**: rearrange the upstream gradient back into the
   feature-map shape declared by CONV2D-C9. Element order MUST mirror the forward
   flattening order — any deviation silently breaks gradient routing.
2. **Pool → Conv2D input gradient (∂L/∂X)**: for MaxPool2D, route the upstream
   gradient only to the argmax position per window; for AvgPool2D, distribute it
   evenly. Then full 2-D correlation of the resulting feature-map gradient with the
   filters, summed over output channels.
3. **Weight gradient (∂L/∂W)**: per filter and per input channel, cross-correlation
   of the input tensor (sliced by input channel) with the upstream gradient (sliced
   by output channel). Bias gradient (if `UseBias`) is the sum of upstream gradient
   across spatial axes per output channel.

### 4.4 Layer Composition with Existing Specs

- **BatchNorm-2D**: out of scope for v1.0. `l1-normalization-layers.md` v1.0 covers
  1-D and per-sample norms; a future amendment introduces 2-D variants that consume
  Conv2D output. Current spec assumes BatchNorm is applied either before flatten on
  per-channel 1-D statistics, or omitted.
- **AndTrain continuation** (`l1-dataset-formats.md`): Conv2D layers participate in
  the resumed training loop exactly like Dense layers; weights and shape metadata are
  the only persistence delta.
- **Dynamic topology** (`l1-dynamic-topology.md`): Conv2D layers MAY be mutated at
  safe-points, but adding or removing a Conv2D block changes the downstream Dense
  input size — the transaction protocol MUST recompute `outputLen()` for the entire
  conv prefix before committing.

## 5. Performance Considerations

The Conv2D hotspot dominates CV workloads. The L1 contract surfaces three performance
boundaries that the L2 implementation MUST honour without prescribing the algorithm:

1. **Backward pass allocation** — per `l1-performance-contract.md` PERF-4, the
   backward path MUST allocate zero per iteration once steady-state buffers are
   primed. Gradient scratch buffers (`gradW`, `gradX`, `gradB`) reuse the same
   shape across iterations and MAY participate in `sync.Pool`.
2. **Cache-friendly inner loop** — the implementation MAY choose between direct
   loop-based correlation and im2col-then-matmul. Both are stdlib-compatible (C29);
   the choice is an L2 concern unless `pprof` shows the L1 invariants forcing a
   pathological access pattern.
3. **Determinism over throughput** — when forced to choose, CONV2D-3 wins. Any
   parallel reduction MUST use a fixed accumulation order or compensated summation;
   non-deterministic atomic adds are forbidden in the forward path.

## 6. Implementation Notes

1. Layout is fixed to CHW (CONV2D-C9). Any L2 implementation MUST honour the flat
   index formulae stated there verbatim — independent re-derivation is forbidden.
2. Generalise `outputLen()` from `pkg/layer/conv/conv1d.go` into `outputShape()`
   returning `(C_out, H_out, W_out)` so `compile()` can size downstream layers.
3. The MNIST loader from `l1-dataset-formats.md` currently exposes flat 784-byte
   vectors. The L2 spec for Conv2D MUST coordinate with `l2-dataset-loader-impl.md` to
   add a `WithImageShape(channels, height, width)` adapter (or equivalent) so that
   existing loaders can re-tensorise their output without breaking the IDX contract.
4. Backward-pass tests MUST cover both padding modes, both pool variants, and at
   least one multi-channel case (`C_in ≥ 2`) — that is the regression risk most
   likely to expose CONV2D-4 violations.

## 7. Drawbacks & Alternatives

- **Alternative: extend `l1-conv-layers.md` v1.0 → v2.0 with a 2-D section** —
  rejected. v1.0 explicitly defers 2-D as a *separate* future spec (§7); reopening it
  for a major bump destabilises an already-Stable contract whose 1-D consumers are in
  production. Sibling spec preserves both stability and historical accuracy.
- **Alternative: depthwise / grouped / dilated convolutions in v1.0** — rejected.
  Each adds at least one more invariant axis (group count, dilation rate) and
  complicates the gradient definition. Deferred to a v1.x amendment after baseline
  Conv2D is Stable.
- **Alternative: explicit `Tensor[T]` type with shape metadata in the L1 contract** —
  rejected for v1.0 to preserve `Layer[T]` interface uniformity. Shape is carried as
  layer fields and propagated through `compile()`, matching the Conv1D precedent.
- **Alternative: per-axis padding amounts (top/bottom/left/right)** — rejected for
  v1.0. The two-mode `PadValid`/`PadSame` API covers all examples-catalog needs;
  custom asymmetric padding is a niche v1.x extension.

## Canonical References

<!-- Filled when promoting to Stable. Authoritative source files that downstream
     agents (L2 spec, examples, tests) MUST read before implementing this contract. -->

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[CONV1D-PARENT]` | `.design/main/specifications/l1-conv-layers.md` | Sibling 1-D contract whose invariant philosophy and `Layer[T]` integration this spec generalises |
| `[PERF]` | `.design/main/specifications/l1-performance-contract.md` | Zero-alloc backward and benchmark obligations that govern any Conv2D L2 implementation |
| `[DATASET]` | `.design/main/specifications/l1-dataset-formats.md` | IDX/MNIST loader source whose output MUST become 2-D-tensor compatible for end-to-end CV examples |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-05-17 | Initial Draft — Conv2D / MaxPool2D / AvgPool2D / Flatten2D contract with 9 invariants. |
| 0.2.0 | 2026-05-17 | CONV2D-C9 frozen: tensor layout is CHW row-major; filter storage filter-major flat `[]T` of length `F * C_in * K_h * K_w`. Rationale: cache locality with existing `pkg/layer/norm/` primitives, PyTorch/cuDNN parity, natural generalisation of Conv1D Variant A storage. |
