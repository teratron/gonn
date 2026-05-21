// Package quantization implements post-training int8 quantization for GoNN networks.
//
// Three top-level operations: [Quantize] transforms a trained [nn.Network] into a
// [QuantizedNetwork] with int8 weights; [Load] deserialises a .qnn.json artifact;
// [Evaluate] runs a side-by-side float vs quantized comparison per QUANT-8.
//
// Delivery phases:
//   - Phase α: weight-only Dense/Conv1D/Conv2D (no activation calibration required)
//   - Phase β: [CalibrationRunner] + full-int8 int32-accumulator GEMM
//   - Phase γ: attention projection matrices Wq/Wk/Wv/Wo
//
// Quantized layers own int8 weight arrays alongside float64 (scale, zero_point)
// metadata and are NOT wirable into pkg/nn ConvPrefix — they are inference-only
// (GC-5). The source network remains trainable after [Quantize] (GC-3).
//
// AI-Meta:
//   - Purpose: Post-training int8 quantization (PTQ) — weight-only + full-int8 + attention projections.
//   - Tier: L2-impl.
//   - Stability: Experimental.
package quantization
