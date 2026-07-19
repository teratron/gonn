// Multi-hidden persistence regression — Phase 5 / Track C.
//
// Two scenarios:
//
//  1. v0.1 fixture forward-compat: a 1.0.0 wire artefact must still load
//     under the 1.1.0 schema (PERS-1 minor mismatch tolerance).
//  2. v0.2 multi-hidden round-trip: ConfigDoc with len(HiddenLayers) > 1
//     and a matching WeightsDoc (N+1 layers — hidden_0, hidden_1, …,
//     output) round-trip with bit-identical float64 values per PERS-4.
package persistence

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

// v01ConfigFixture mimics what examples/persistence/ produced under
// SchemaVersion = "1.0.0" (single hidden layer, no array suffix). The
// JSON literal is hand-crafted so this regression doesn't depend on the
// 1.0.0 source code being kept around.
const v01ConfigFixture = `{
  "schema_version": "1.0.0",
  "lib_version": "0.1.0",
  "precision": "float32",
  "input_size": 2,
  "hidden_layers": [
    {
      "size": 4,
      "activation": "Sigmoid",
      "bias": true
    }
  ],
  "output": {
    "size": 1,
    "activation": "Sigmoid",
    "bias": true
  },
  "training": {
    "learning_rate": 0.5,
    "loss": "MSE",
    "loss_limit": 0.0001,
    "max_iterations": 5000,
    "weight_init": "xavier"
  }
}
`

// TestReadV01ConfigFixtureForwardCompat verifies that a v0.1 (1.0.0)
// config artefact loads cleanly under the v0.2 (1.1.0) reader. PERS-1
// minor-version drift is silently tolerated; the major-version match
// keeps schema validation green.
func TestReadV01ConfigFixtureForwardCompat(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	if err := os.WriteFile(path, []byte(v01ConfigFixture), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	got, err := ReadConfig[float32](path)
	if err != nil {
		t.Fatalf("ReadConfig on v0.1 fixture must succeed under v0.2 reader, got %v", err)
	}
	if got.SchemaVersion != "1.0.0" {
		t.Errorf("SchemaVersion preserved on read: got %q, want %q", got.SchemaVersion, "1.0.0")
	}
	if len(got.HiddenLayers) != 1 || got.HiddenLayers[0].Size != 4 {
		t.Errorf("HiddenLayers misread: got %+v", got.HiddenLayers)
	}
	if got.Training.LearningRate != float32(0.5) {
		t.Errorf("LearningRate = %v; want 0.5", got.Training.LearningRate)
	}
}

// multiHiddenConfigDoc returns a 2-hidden ConfigDoc[float64] used by
// the multi-hidden round-trip test. Float64 lets us assert bit-equality
// per PERS-4; ULP-1 tolerance for float32 is already covered by the
// pre-existing TestConfigRoundTripF32.
func multiHiddenConfigDoc() ConfigDoc[float64] {
	return ConfigDoc[float64]{
		LibVersion: "0.2.0",
		InputSize:  3,
		HiddenLayers: []HiddenLayerDoc{
			{Size: 5, Activation: "ReLU", Bias: true},
			{Size: 4, Activation: "Sigmoid", Bias: false},
		},
		Output: OutputDoc{Size: 2, Activation: "Sigmoid", Bias: true},
		Training: TrainingDoc[float64]{
			LearningRate:  math.Pi,
			Loss:          "MSE",
			LossLimit:     1e-6,
			MaxIterations: 1000,
			WeightInit:    "he",
			RNGSeed:       1234,
		},
	}
}

// multiHiddenWeightsDoc constructs the matching WeightsDoc with N+1
// LayerWeights entries — hidden_0 (5 cells × 3 inputs), hidden_1 (4 ×
// 5), output (2 × 4). Values are deterministic for diff-friendly
// failure messages.
func multiHiddenWeightsDoc() WeightsDoc[float64] {
	mk := func(rows, cols int, base float64) [][]float64 {
		w := make([][]float64, rows)
		for i := range w {
			row := make([]float64, cols)
			for j := range row {
				row[j] = base + float64(i)*0.01 + float64(j)*0.001
			}
			w[i] = row
		}
		return w
	}
	mkBias := func(n int, base float64) []float64 {
		b := make([]float64, n)
		for i := range b {
			b[i] = base + float64(i)*0.1
		}
		return b
	}
	return WeightsDoc[float64]{
		Layers: []LayerWeights[float64]{
			{Name: "hidden_0", Weights: mk(5, 3, 0.10), Biases: mkBias(5, 1.0)},
			// hidden_1 has Bias == false in the config, so Biases stays empty.
			{Name: "hidden_1", Weights: mk(4, 5, 0.20)},
			{Name: "output", Weights: mk(2, 4, 0.30), Biases: mkBias(2, 3.0)},
		},
	}
}

// TestMultiHiddenRoundTripF64 writes a 2-hidden config + weights pair
// and reads them back, verifying bit-equality of every weight scalar
// and integrity of the SHA-256 anchor (PERS-3) across schema version
// 1.1.0. This is the v0.2 wire-format smoke that future schema bumps
// must keep green.
func TestMultiHiddenRoundTripF64(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.json")
	weightsPath := filepath.Join(dir, "weights.json")

	cfg := multiHiddenConfigDoc()
	w := multiHiddenWeightsDoc()

	if err := WriteConfig(cfgPath, cfg); err != nil {
		t.Fatalf("WriteConfig: %v", err)
	}
	if err := WriteWeights(weightsPath, cfg, w); err != nil {
		t.Fatalf("WriteWeights: %v", err)
	}

	gotCfg, gotW, err := ReadWeights[float64](cfgPath, weightsPath)
	if err != nil {
		t.Fatalf("ReadWeights: %v", err)
	}

	if gotCfg.SchemaVersion != SchemaVersion {
		t.Errorf("Config SchemaVersion = %q; want %q", gotCfg.SchemaVersion, SchemaVersion)
	}
	if gotW.SchemaVersion != SchemaVersion {
		t.Errorf("Weights SchemaVersion = %q; want %q", gotW.SchemaVersion, SchemaVersion)
	}
	if len(gotW.Layers) != 3 {
		t.Fatalf("Layers count = %d; want 3 (hidden_0, hidden_1, output)", len(gotW.Layers))
	}
	wantNames := []string{"hidden_0", "hidden_1", "output"}
	for i, want := range wantNames {
		if gotW.Layers[i].Name != want {
			t.Errorf("Layers[%d].Name = %q; want %q", i, gotW.Layers[i].Name, want)
		}
	}

	for li, layer := range w.Layers {
		gotLayer := gotW.Layers[li]
		if len(gotLayer.Weights) != len(layer.Weights) {
			t.Fatalf("Layer %s: rows = %d; want %d", layer.Name, len(gotLayer.Weights), len(layer.Weights))
		}
		for r, row := range layer.Weights {
			gotRow := gotLayer.Weights[r]
			for c, v := range row {
				if math.Float64bits(gotRow[c]) != math.Float64bits(v) {
					t.Errorf("%s[%d][%d] not bit-identical: got %x want %x",
						layer.Name, r, c, math.Float64bits(gotRow[c]), math.Float64bits(v))
				}
			}
		}
		if len(gotLayer.Biases) != len(layer.Biases) {
			t.Errorf("Layer %s: biases len = %d; want %d", layer.Name, len(gotLayer.Biases), len(layer.Biases))
		}
		for bi, b := range layer.Biases {
			if math.Float64bits(gotLayer.Biases[bi]) != math.Float64bits(b) {
				t.Errorf("%s.bias[%d] not bit-identical: got %x want %x",
					layer.Name, bi, math.Float64bits(gotLayer.Biases[bi]), math.Float64bits(b))
			}
		}
	}
}
