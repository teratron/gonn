package quantization

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"

	attentionpkg "github.com/teratron/gonn/pkg/layer/attention"
	convpkg "github.com/teratron/gonn/pkg/layer/conv"
	"github.com/teratron/gonn/pkg/nn"
	"github.com/teratron/gonn/pkg/utils"
)

// quantizedLayer is the package-private forward interface satisfied by all
// quantized layer types. GC-5: these types do NOT implement conv.Layer[T].
type quantizedLayer[T utils.Float] interface {
	Forward(x []T) []T
}

// QuantizedNetwork is an inference-only quantized representation of a trained
// network. Layers mirrors net.Config().ConvPrefix in order; each element is
// either a typed quantized layer or a float-identity pass-through for
// unrecognised layer types (QUANT-10).
//
// AI-Meta:
//   - Purpose: Inference-only holder of quantized layers produced by Quantize[T].
//   - Stability: Experimental.
type QuantizedNetwork[T utils.Float] struct {
	BaselineHash    string
	Layers          []quantizedLayer[T]
	Config          QuantizationConfig[T]
	CalibProvenance CalibrationProvenance
}

// Forward chains the input through all quantized layers in order.
func (q *QuantizedNetwork[T]) Forward(x []T) []T {
	cur := x
	for _, l := range q.Layers {
		cur = l.Forward(cur)
	}
	return cur
}

// Quantize transforms the ConvPrefix layers AND the dense MLP head of net
// into a QuantizedNetwork whose Forward reproduces the full float inference
// path (activations included). The source network is NOT mutated (GC-3).
// calibSamples may be nil for WeightOnly mode — activation calibration is
// skipped in that case.
//
// Networks with normalization layers are rejected: the quantized chain has
// no norm stage, so quantizing would silently change the model function.
//
// AI-Meta:
//   - Purpose: Top-level PTQ entry point; produces a QuantizedNetwork from a trained NN[T].
//   - Errors: ErrUserConfig for unsupported configurations (norm layers).
//   - Stability: Experimental.
func Quantize[T utils.Float](net *nn.NN[T], calibSamples [][]T, cfg QuantizationConfig[T]) (*QuantizedNetwork[T], error) {
	netCfg := net.Config()
	if len(netCfg.NormLayers) > 0 {
		return nil, utils.Newf(utils.ErrUserConfig,
			"quantize: networks with normalization layers are not supported — the quantized chain would drop the norm stage")
	}
	prefix := netCfg.ConvPrefix
	layers := make([]quantizedLayer[T], 0, len(prefix)+len(netCfg.HiddenLayers)+1)

	for _, cl := range prefix {
		var ql quantizedLayer[T]
		switch l := cl.(type) {
		case *convpkg.Conv1D[T]:
			ql = newQuantizedConv1D(l, cfg)
		case *convpkg.Conv2D[T]:
			ql = newQuantizedConv2D(l, cfg)
		case *attentionpkg.MultiHeadAttention[T]:
			ql = newQuantizedAttentionProjections(l, cfg)
		default:
			// Unknown types pass through as float identity (QUANT-10).
			ql = &floatPassthrough[T]{inner: cl}
		}
		layers = append(layers, ql)
	}

	// Dense head: one QuantizedDense per hidden layer plus the output layer,
	// each carrying its activation so the chain reproduces the float forward
	// (audit B9: QuantizedDense existed but MLPs were never quantized).
	layers = append(layers, quantizeDenseHead(net, cfg)...)

	hash, err := baselineHash(net)
	if err != nil {
		return nil, fmt.Errorf("quantize: baseline hash: %w", err)
	}

	prov := CalibrationProvenance{
		Strategy: cfg.Strategy,
		Seed:     cfg.Seed,
	}

	return &QuantizedNetwork[T]{
		Config:          cfg,
		Layers:          layers,
		BaselineHash:    hash,
		CalibProvenance: prov,
	}, nil
}

// quantizeDenseHead extracts every dense layer's weight matrix (bias axons
// split into the float Bias slice) and returns the quantized layer chain
// with per-layer activations attached.
func quantizeDenseHead[T utils.Float](net *nn.NN[T], cfg QuantizationConfig[T]) []quantizedLayer[T] {
	netCfg := net.Config()
	out := make([]quantizedLayer[T], 0, len(netCfg.HiddenLayers)+1)
	prev := net.Network.Input.Len()

	for i := range net.Network.Hiddens {
		cells := net.Network.Hiddens[i].Cells()
		hasBias := i < len(netCfg.HiddenLayers) && netCfg.HiddenLayers[i].Bias
		w, b := extractDenseMatrix(len(cells), prev, hasBias, func(ci, ai int) float64 {
			return float64(cells[ci].Axons[ai].W())
		})
		qd := newQuantizedDense[T](w, b, len(cells), prev, cfg)
		if i < len(netCfg.HiddenLayers) {
			qd.Act = netCfg.HiddenLayers[i].Activation
			qd.ApplyAct = true
		}
		out = append(out, qd)
		prev = len(cells)
	}

	outCells := net.Network.Output.Cells()
	w, b := extractDenseMatrix(len(outCells), prev, netCfg.OutputBias, func(ci, ai int) float64 {
		return float64(outCells[ci].Axons[ai].W())
	})
	qd := newQuantizedDense[T](w, b, len(outCells), prev, cfg)
	qd.Act = netCfg.OutputActivation
	qd.ApplyAct = true
	out = append(out, qd)
	return out
}

// extractDenseMatrix reads a layer's axon weights into a row-major Cout×Cin
// matrix. The bias axon is the LAST axon of each cell when hasBias is true
// (Build's wiring order) and lands in the returned bias slice.
func extractDenseMatrix(cout, cin int, hasBias bool, weightAt func(cellIdx, axonIdx int) float64) ([]float64, []float64) {
	w := make([]float64, cout*cin)
	var b []float64
	if hasBias {
		b = make([]float64, cout)
	}
	for c := range cout {
		for k := range cin {
			w[c*cin+k] = weightAt(c, k)
		}
		if hasBias {
			b[c] = weightAt(c, cin)
		}
	}
	return w, b
}

// Load deserialises a .qnn.json file from disk.
//
// AI-Meta:
//   - Purpose: Load a .qnn.json artifact produced by QuantizedNetwork[T].Save.
//   - Errors: ErrUserConfig for corrupt or incompatible artifacts.
//   - Stability: Experimental.
func Load[T utils.Float](path string) (*QuantizedNetwork[T], error) {
	return UnmarshalQNNFile[T](path)
}

// quantizeWeightsTensor quantizes a flat weight tensor of shape Cout×Cin
// (row-major, one row per output channel) into int8 with the requested
// granularity. Returns the quantized bytes and the corresponding params.
//
// PerChannel: one (scale, zp) per output channel — len(params.Scale)==Cout.
// PerTensor:  one (scale, zp) for the whole tensor — len(params.Scale)==1.
func quantizeWeightsTensor(w []float64, cout, cin int, gran Granularity) ([]int8, QuantizationParams) {
	qw := make([]int8, len(w))

	if gran == PerChannel {
		scales := make([]float64, cout)
		zps := make([]int32, cout)

		for c := range cout {
			rMin, rMax := rowMinMax(w, c, cin)
			p := computeSymmetric(rMin, rMax)
			scales[c] = p.Scale[0]
			zps[c] = p.ZeroPoint[0]
			for k := range cin {
				row := QuantizationParams{
					Scale:       []float64{scales[c]},
					ZeroPoint:   []int32{zps[c]},
					Granularity: PerTensor,
				}
				qw[c*cin+k] = quantize(w[c*cin+k], row, 0)
			}
		}
		return qw, QuantizationParams{
			Scale:       scales,
			ZeroPoint:   zps,
			Granularity: PerChannel,
		}
	}

	// PerTensor
	rMin, rMax := sliceMinMax(w)
	p := computeSymmetric(rMin, rMax)
	for i, v := range w {
		qw[i] = quantize(v, p, 0)
	}
	return qw, p
}

// floatPassthrough wraps an unrecognised conv.Layer[T] as a quantizedLayer[T]
// that forwards float values unchanged (QUANT-10 identity semantics).
type floatPassthrough[T utils.Float] struct {
	inner convpkg.Layer[T]
}

func (f *floatPassthrough[T]) Forward(x []T) []T { return f.inner.Forward(x) }

// baselineHash digests the network's actual trained state: topology sizes
// plus the flat weight vector. The previous implementation hashed
// json.Marshal(net), which serialised the cell bundles as "{}" — two
// networks with identical shape but different weights collided (audit B9).
func baselineHash[T utils.Float](net *nn.NN[T]) (string, error) {
	cfg := net.Config()
	hidden := make([]uint, len(cfg.HiddenLayers))
	for i, h := range cfg.HiddenLayers {
		hidden[i] = h.Size
	}
	payload := struct {
		Hidden     []uint    `json:"hidden"`
		Weights    []float64 `json:"weights"`
		InputSize  uint      `json:"input_size"`
		OutputSize uint      `json:"output_size"`
	}{
		Hidden:     hidden,
		Weights:    toFloat64Slice(net.FlatWeights()),
		InputSize:  cfg.InputSize,
		OutputSize: cfg.OutputSize,
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

// rowMinMax returns min/max of row c in a Cout×Cin row-major flat slice.
func rowMinMax(w []float64, c, cin int) (float64, float64) {
	base := c * cin
	mn, mx := w[base], w[base]
	for k := 1; k < cin; k++ {
		v := w[base+k]
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}
	return mn, mx
}

func sliceMinMax(w []float64) (float64, float64) {
	if len(w) == 0 {
		return 0, 0
	}
	mn, mx := w[0], w[0]
	for _, v := range w[1:] {
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}
	return mn, mx
}

func toFloat64Slice[T utils.Float](src []T) []float64 {
	dst := make([]float64, len(src))
	for i, v := range src {
		dst[i] = float64(v)
	}
	return dst
}

func clampF64(v, lo, hi float64) float64 {
	return math.Max(lo, math.Min(hi, v))
}
