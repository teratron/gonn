package quantization

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/utils"
)

// QuantizedLayerJSON is the per-layer wire record inside .qnn.json.
//
// AI-Meta:
//   - Purpose: JSON wire representation of one quantized layer inside .qnn.json.
type QuantizedLayerJSON struct {
	WoWeights   string    `json:"wo_weights,omitempty"`
	WoBias      string    `json:"wo_bias,omitempty"`
	WkBias      string    `json:"wk_bias,omitempty"`
	WvWeights   string    `json:"wv_weights,omitempty"`
	WkWeights   string    `json:"wk_weights,omitempty"`
	Weights     string    `json:"weights"`
	Bias        string    `json:"bias,omitempty"`
	WvBias      string    `json:"wv_bias,omitempty"`
	Type        string    `json:"type"`
	ZeroPoint   []int32   `json:"zero_point"`
	WoScale     []float64 `json:"wo_scale,omitempty"`
	WvZeroPoint []int32   `json:"wv_zero_point,omitempty"`
	WvScale     []float64 `json:"wv_scale,omitempty"`
	Scale       []float64 `json:"scale"`
	WoZeroPoint []int32   `json:"wo_zero_point,omitempty"`
	WkZeroPoint []int32   `json:"wk_zero_point,omitempty"`
	WkScale     []float64 `json:"wk_scale,omitempty"`
	KernelH     int       `json:"kernel_h,omitempty"`
	StrideH     int       `json:"stride_h,omitempty"`
	NumHeads    int       `json:"num_heads,omitempty"`
	SeqLen      int       `json:"seq_len,omitempty"`
	Cout        int       `json:"cout,omitempty"`
	Granularity int       `json:"granularity"`
	Cin         int       `json:"cin,omitempty"`
	Padding     int       `json:"padding,omitempty"`
	StrideW     int       `json:"stride_w,omitempty"`
	Strategy    int       `json:"strategy"`
	Stride      int       `json:"stride,omitempty"`
	InLen       int       `json:"in_len,omitempty"`
	InChannels  int       `json:"in_channels,omitempty"`
	KernelW     int       `json:"kernel_w,omitempty"`
	KernelSize  int       `json:"kernel_size,omitempty"`
	UseBias     bool      `json:"use_bias,omitempty"`
	Causal      bool      `json:"causal,omitempty"`
	// Act / ApplyAct carry the dense-head activation (QuantizedDense only).
	Act      uint8 `json:"act,omitempty"`
	ApplyAct bool  `json:"apply_act,omitempty"`
}

// QuantizedNetworkJSON is the top-level .qnn.json wire format.
//
// AI-Meta:
//   - Purpose: Top-level JSON envelope for .qnn.json; "Type":"quantized" discriminator (QUANT-6).
type QuantizedNetworkJSON struct {
	Type            string                `json:"type"`
	BaselineHash    string                `json:"baseline_hash"`
	Layers          []QuantizedLayerJSON  `json:"layers"`
	CalibProvenance CalibrationProvenance `json:"calib_provenance"`
}

// MarshalQNN serialises a QuantizedNetwork to .qnn.json bytes.
//
// AI-Meta:
//   - Purpose: Serialise a QuantizedNetwork to the .qnn.json wire format (QUANT-6).
//   - Stability: Experimental.
func MarshalQNN[T utils.Float](qnet *QuantizedNetwork[T]) ([]byte, error) {
	env := QuantizedNetworkJSON{
		Type:            "quantized",
		BaselineHash:    qnet.BaselineHash,
		CalibProvenance: qnet.CalibProvenance,
		Layers:          make([]QuantizedLayerJSON, 0, len(qnet.Layers)),
	}
	for _, l := range qnet.Layers {
		rec, err := marshalLayer(l)
		if err != nil {
			return nil, err
		}
		env.Layers = append(env.Layers, rec)
	}
	return json.MarshalIndent(env, "", "  ")
}

// UnmarshalQNN deserialises .qnn.json bytes into a QuantizedNetwork.
// Returns an error if the top-level "Type" field is not "quantized" (QUANT-6).
//
// AI-Meta:
//   - Purpose: Deserialise .qnn.json bytes into a QuantizedNetwork (QUANT-6 round-trip).
//   - Stability: Experimental.
func UnmarshalQNN[T utils.Float](data []byte) (*QuantizedNetwork[T], error) {
	var env QuantizedNetworkJSON
	if err := json.Unmarshal(data, &env); err != nil {
		return nil, err
	}
	if env.Type != "quantized" {
		return nil, fmt.Errorf("unmarshal qnn: wrong Type %q, expected \"quantized\"", env.Type)
	}
	layers := make([]quantizedLayer[T], 0, len(env.Layers))
	for _, rec := range env.Layers {
		l, err := unmarshalLayer[T](rec)
		if err != nil {
			return nil, err
		}
		layers = append(layers, l)
	}
	return &QuantizedNetwork[T]{
		BaselineHash:    env.BaselineHash,
		CalibProvenance: env.CalibProvenance,
		Layers:          layers,
	}, nil
}

// UnmarshalQNNFile reads a .qnn.json file from disk and deserialises it.
func UnmarshalQNNFile[T utils.Float](path string) (*QuantizedNetwork[T], error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("load qnn: %w", err)
	}
	return UnmarshalQNN[T](data)
}

// Save writes the .qnn.json artifact to the given path.
func (q *QuantizedNetwork[T]) Save(path string) error {
	data, err := MarshalQNN(q)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func marshalLayer[T utils.Float](l quantizedLayer[T]) (QuantizedLayerJSON, error) {
	switch v := l.(type) {
	case *QuantizedDense[T]:
		return QuantizedLayerJSON{
			Type:        "Dense",
			Cout:        v.Cout,
			Cin:         v.Cin,
			Scale:       v.WeightParams.Scale,
			ZeroPoint:   v.WeightParams.ZeroPoint,
			Weights:     encodeInt8(v.Weights),
			Bias:        encodeFloat64(v.Bias),
			Granularity: int(v.WeightParams.Granularity),
			Strategy:    int(v.WeightParams.Strategy),
			Act:         uint8(v.Act),
			ApplyAct:    v.ApplyAct,
		}, nil
	case *QuantizedConv1D[T]:
		return QuantizedLayerJSON{
			Type:        "Conv1D",
			Cout:        v.NumFilters,
			Cin:         v.KernelSize,
			Scale:       v.WeightParams.Scale,
			ZeroPoint:   v.WeightParams.ZeroPoint,
			Weights:     encodeInt8(v.Weights),
			Bias:        encodeFloat64(v.Bias),
			KernelSize:  v.KernelSize,
			InLen:       v.InLen,
			Stride:      v.Stride,
			Padding:     int(v.Padding),
			UseBias:     v.UseBias,
			Granularity: int(v.WeightParams.Granularity),
			Strategy:    int(v.WeightParams.Strategy),
		}, nil
	case *QuantizedConv2D[T]:
		return QuantizedLayerJSON{
			Type:        "Conv2D",
			Cout:        v.NumFilters,
			Cin:         v.InChannels * v.KernelH * v.KernelW,
			Scale:       v.WeightParams.Scale,
			ZeroPoint:   v.WeightParams.ZeroPoint,
			Weights:     encodeInt8(v.Weights),
			Bias:        encodeFloat64(v.Bias),
			KernelH:     v.KernelH,
			KernelW:     v.KernelW,
			InChannels:  v.InChannels,
			InLen:       v.InLen,
			StrideH:     v.StrideH,
			StrideW:     v.StrideW,
			Padding:     int(v.Padding),
			UseBias:     v.UseBias,
			Granularity: int(v.WeightParams.Granularity),
			Strategy:    int(v.WeightParams.Strategy),
		}, nil
	case *QuantizedAttentionProjections[T]:
		return QuantizedLayerJSON{
			Type:     "AttentionProj",
			Cout:     v.Dmodel,
			Cin:      v.Dmodel,
			NumHeads: v.NumHeads,
			SeqLen:   v.SeqLen,
			Causal:   v.Causal,
			// Wq reuses the top-level Scale/ZeroPoint/Weights/Bias fields
			Scale:       v.WqParams.Scale,
			ZeroPoint:   v.WqParams.ZeroPoint,
			Weights:     encodeInt8(v.Wq),
			Bias:        encodeFloat64(v.Bq),
			Granularity: int(v.WqParams.Granularity),
			Strategy:    int(v.WqParams.Strategy),
			// Wk
			WkWeights:   encodeInt8(v.Wk),
			WkScale:     v.WkParams.Scale,
			WkZeroPoint: v.WkParams.ZeroPoint,
			WkBias:      encodeFloat64(v.Bk),
			// Wv
			WvWeights:   encodeInt8(v.Wv),
			WvScale:     v.WvParams.Scale,
			WvZeroPoint: v.WvParams.ZeroPoint,
			WvBias:      encodeFloat64(v.Bv),
			// Wo
			WoWeights:   encodeInt8(v.Wo),
			WoScale:     v.WoParams.Scale,
			WoZeroPoint: v.WoParams.ZeroPoint,
			WoBias:      encodeFloat64(v.Bo),
		}, nil
	case *floatPassthrough[T]:
		return QuantizedLayerJSON{Type: "FloatPass"}, nil
	default:
		return QuantizedLayerJSON{Type: "Unknown"}, nil
	}
}

func unmarshalLayer[T utils.Float](rec QuantizedLayerJSON) (quantizedLayer[T], error) {
	wp := QuantizationParams{
		Scale:       rec.Scale,
		ZeroPoint:   rec.ZeroPoint,
		Granularity: Granularity(rec.Granularity),
		Strategy:    CalibStrategy(rec.Strategy),
	}
	weights, err := decodeInt8(rec.Weights)
	if err != nil {
		return nil, fmt.Errorf("unmarshal layer %s weights: %w", rec.Type, err)
	}
	bias, err := decodeFloat64(rec.Bias)
	if err != nil {
		return nil, fmt.Errorf("unmarshal layer %s bias: %w", rec.Type, err)
	}

	switch rec.Type {
	case "Dense":
		return &QuantizedDense[T]{
			WeightParams: wp,
			Weights:      weights,
			Bias:         bias,
			Cout:         rec.Cout,
			Cin:          rec.Cin,
			Act:          activation.Type(rec.Act),
			ApplyAct:     rec.ApplyAct,
		}, nil
	case "Conv1D":
		return &QuantizedConv1D[T]{
			WeightParams: wp,
			Weights:      weights,
			Bias:         bias,
			NumFilters:   rec.Cout,
			KernelSize:   rec.KernelSize,
			InLen:        rec.InLen,
			Stride:       rec.Stride,
			Padding:      convPad(rec.Padding),
			UseBias:      rec.UseBias,
		}, nil
	case "Conv2D":
		return &QuantizedConv2D[T]{
			WeightParams: wp,
			Weights:      weights,
			Bias:         bias,
			NumFilters:   rec.Cout,
			InChannels:   rec.InChannels,
			KernelH:      rec.KernelH,
			KernelW:      rec.KernelW,
			InLen:        rec.InLen,
			StrideH:      rec.StrideH,
			StrideW:      rec.StrideW,
			Padding:      convPad(rec.Padding),
			UseBias:      rec.UseBias,
		}, nil
	case "AttentionProj":
		wkWeights, err2 := decodeInt8(rec.WkWeights)
		if err2 != nil {
			return nil, fmt.Errorf("unmarshal layer AttentionProj wk weights: %w", err2)
		}
		wvWeights, err3 := decodeInt8(rec.WvWeights)
		if err3 != nil {
			return nil, fmt.Errorf("unmarshal layer AttentionProj wv weights: %w", err3)
		}
		woWeights, err4 := decodeInt8(rec.WoWeights)
		if err4 != nil {
			return nil, fmt.Errorf("unmarshal layer AttentionProj wo weights: %w", err4)
		}
		bk, err5 := decodeFloat64(rec.WkBias)
		if err5 != nil {
			return nil, fmt.Errorf("unmarshal layer AttentionProj wk bias: %w", err5)
		}
		bv, err6 := decodeFloat64(rec.WvBias)
		if err6 != nil {
			return nil, fmt.Errorf("unmarshal layer AttentionProj wv bias: %w", err6)
		}
		bo, err7 := decodeFloat64(rec.WoBias)
		if err7 != nil {
			return nil, fmt.Errorf("unmarshal layer AttentionProj wo bias: %w", err7)
		}
		return &QuantizedAttentionProjections[T]{
			WqParams: wp,
			WkParams: QuantizationParams{Scale: rec.WkScale, ZeroPoint: rec.WkZeroPoint, Granularity: Granularity(rec.Granularity), Strategy: CalibStrategy(rec.Strategy)},
			WvParams: QuantizationParams{Scale: rec.WvScale, ZeroPoint: rec.WvZeroPoint, Granularity: Granularity(rec.Granularity), Strategy: CalibStrategy(rec.Strategy)},
			WoParams: QuantizationParams{Scale: rec.WoScale, ZeroPoint: rec.WoZeroPoint, Granularity: Granularity(rec.Granularity), Strategy: CalibStrategy(rec.Strategy)},
			Wq:       weights,
			Wk:       wkWeights,
			Wv:       wvWeights,
			Wo:       woWeights,
			Bq:       bias,
			Bk:       bk,
			Bv:       bv,
			Bo:       bo,
			Dmodel:   rec.Cout,
			NumHeads: rec.NumHeads,
			SeqLen:   rec.SeqLen,
			Causal:   rec.Causal,
		}, nil
	case "FloatPass", "Unknown":
		return &opaquePassthrough[T]{}, nil
	default:
		return nil, fmt.Errorf("unmarshal layer: unknown type %q", rec.Type)
	}
}

// opaquePassthrough is a no-op layer used when a FloatPass or unknown layer is
// deserialised without access to the original conv.Layer[T].
type opaquePassthrough[T utils.Float] struct{}

func (o *opaquePassthrough[T]) Forward(x []T) []T { return x }

func encodeInt8(w []int8) string {
	b := make([]byte, len(w))
	for i, v := range w {
		b[i] = byte(v)
	}
	return base64.StdEncoding.EncodeToString(b)
}

func decodeInt8(s string) ([]int8, error) {
	if s == "" {
		return nil, nil
	}
	b, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		return nil, err
	}
	w := make([]int8, len(b))
	for i, v := range b {
		w[i] = int8(v)
	}
	return w, nil
}

// encodeFloat64 encodes a float64 slice as IEEE-754 little-endian base64.
func encodeFloat64(f []float64) string {
	if len(f) == 0 {
		return ""
	}
	b := make([]byte, len(f)*8)
	for i, v := range f {
		binary.LittleEndian.PutUint64(b[i*8:], math.Float64bits(v))
	}
	return base64.StdEncoding.EncodeToString(b)
}

// decodeFloat64 decodes a base64 IEEE-754 little-endian float64 slice.
func decodeFloat64(s string) ([]float64, error) {
	if s == "" {
		return nil, nil
	}
	b, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		return nil, err
	}
	if len(b)%8 != 0 {
		return nil, fmt.Errorf("float64 decode: byte length %d not multiple of 8", len(b))
	}
	f := make([]float64, len(b)/8)
	for i := range f {
		f[i] = math.Float64frombits(binary.LittleEndian.Uint64(b[i*8:]))
	}
	return f, nil
}

func convPad(v int) convPadMode {
	return convPadMode(v)
}
