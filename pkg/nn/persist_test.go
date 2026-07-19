// Package nn — Save / Load persistence API tests.
//
// The resolver and installLayerDoc error-path tests were promoted here from
// cmd/gonn together with the logic they cover.
package nn

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/layer/conv"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/persistence"
)

// ── Save / Load round-trip ────────────────────────────────────────────────────

func TestSaveLoadRoundTrip(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.json")
	wPath := filepath.Join(dir, "weights.json")

	orig := MustNew(
		WithInput[float64](2),
		WithHiddenLayer[float64](4, activation.SIGMOID),
		WithOutput[float64](1, activation.SIGMOID),
		WithLoss[float64](loss.MSE),
		WithMaxIterations[float64](200),
		WithLossLimit[float64](-1),
		WithWeightInitSeed[float64](42),
	)
	samples := []Sample[float64]{
		{Input: []float64{0, 0}, Target: []float64{0}},
		{Input: []float64{0, 1}, Target: []float64{1}},
		{Input: []float64{1, 0}, Target: []float64{1}},
		{Input: []float64{1, 1}, Target: []float64{0}},
	}
	if _, _, err := orig.Fit(samples); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if err := orig.Save(cfgPath, wPath); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded, err := Load[float64](cfgPath, wPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	for _, s := range samples {
		want, err := orig.Query(s.Input)
		if err != nil {
			t.Fatalf("orig.Query: %v", err)
		}
		got, err := loaded.Query(s.Input)
		if err != nil {
			t.Fatalf("loaded.Query: %v", err)
		}
		if math.Abs(want[0]-got[0]) > 1e-12 {
			t.Errorf("Query(%v): loaded %v != original %v", s.Input, got[0], want[0])
		}
	}

	// Re-saving the loaded network must keep the hash chain intact: the
	// weights written against the cached doc must load against the config.
	wPath2 := filepath.Join(dir, "weights2.json")
	if err := loaded.Save("", wPath2); err != nil {
		t.Fatalf("re-Save weights: %v", err)
	}
	if _, err := Load[float64](cfgPath, wPath2); err != nil {
		t.Errorf("Load with re-saved weights: %v (config hash chain broken)", err)
	}
}

func TestSaveRejectsConvPrefix(t *testing.T) {
	n := MustNew(
		WithInput[float64](8),
		WithConv1D[float64](2, 3, 1, conv.PadValid, false),
		WithHiddenLayer[float64](4, activation.SIGMOID),
		WithOutput[float64](1, activation.SIGMOID),
	)
	if err := n.Save(filepath.Join(t.TempDir(), "c.json"), ""); err == nil {
		t.Error("Save on a conv-prefix network must error (schema is dense-only)")
	}
}

func TestSaveBothPathsEmpty(t *testing.T) {
	n := MustNew(
		WithInput[float64](2),
		WithHiddenLayer[float64](2, activation.SIGMOID),
		WithOutput[float64](1, activation.SIGMOID),
	)
	if err := n.Save("", ""); err == nil {
		t.Error("Save with both paths empty must error")
	}
}

// ── resolvers ────────────────────────────────────────────────────────────────

func TestResolveActivationName(t *testing.T) {
	cases := []struct {
		name string
		ok   bool
	}{
		{"SIGMOID", true},
		{"ReLU", true},
		{"TanH", true},
		{"Linear", true},
		{"NOTEXIST", false},
		{"", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := resolveActivationName(tc.name)
			if (err == nil) != tc.ok {
				t.Errorf("resolveActivationName(%q): err=%v, want ok=%v", tc.name, err, tc.ok)
			}
		})
	}
}

func TestResolveLossName(t *testing.T) {
	cases := []struct {
		name string
		ok   bool
	}{
		{"MSE", true},
		{"MAE", true},
		{"ARCTAN", true},
		{"NOTEXIST", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := resolveLossName(tc.name)
			if (err == nil) != tc.ok {
				t.Errorf("resolveLossName(%q): err=%v, want ok=%v", tc.name, err, tc.ok)
			}
		})
	}
}

func TestResolveWeightInitName(t *testing.T) {
	if _, err := resolveWeightInitName("xavier"); err != nil {
		t.Errorf("xavier: %v", err)
	}
	if _, err := resolveWeightInitName("he"); err != nil {
		t.Errorf("he: %v", err)
	}
	if _, err := resolveWeightInitName(""); err != nil {
		t.Errorf("empty → default: %v", err)
	}
	if _, err := resolveWeightInitName("unknown"); err == nil {
		t.Error("unknown should return error")
	}
}

// ── buildFromConfigDoc error paths ───────────────────────────────────────────

func TestBuildFromConfigDocBadActivation(t *testing.T) {
	doc := persistence.ConfigDoc[float32]{
		InputSize:    2,
		HiddenLayers: []persistence.HiddenLayerDoc{{Size: 4, Activation: "NOPE", Bias: true}},
		Output:       persistence.OutputDoc{Size: 1, Activation: "SIGMOID", Bias: true},
		Training:     persistence.TrainingDoc[float32]{LearningRate: 0.1, Loss: "MSE", MaxIterations: 10},
	}
	if _, err := buildFromConfigDoc(doc); err == nil {
		t.Error("expected error for unknown activation")
	}
}

func TestBuildFromConfigDocBadLoss(t *testing.T) {
	doc := persistence.ConfigDoc[float32]{
		InputSize:    2,
		HiddenLayers: []persistence.HiddenLayerDoc{{Size: 4, Activation: "SIGMOID", Bias: true}},
		Output:       persistence.OutputDoc{Size: 1, Activation: "SIGMOID", Bias: true},
		Training:     persistence.TrainingDoc[float32]{LearningRate: 0.1, Loss: "NOSUCHLOSS", MaxIterations: 10},
	}
	if _, err := buildFromConfigDoc(doc); err == nil {
		t.Error("expected error for unknown loss")
	}
}

// ── installLayerDoc error paths ──────────────────────────────────────────────

func TestInstallLayerDocCellCountMismatch(t *testing.T) {
	layer := persistence.LayerWeights[float32]{
		Name:    "test",
		Weights: [][]float32{{0.1}, {0.2}}, // 2 cells
	}
	err := installLayerDoc("test", false, 3, layer, // net has 3 cells
		func(ci, ai int, w float32) {},
		func(ci int) int { return 1 },
	)
	if err == nil {
		t.Error("expected error for cell count mismatch")
	}
}

func TestInstallLayerDocBiasCountMismatch(t *testing.T) {
	layer := persistence.LayerWeights[float32]{
		Name:    "test",
		Weights: [][]float32{{0.1}, {0.2}},
		Biases:  []float32{0.5}, // only 1 bias but 2 cells
	}
	err := installLayerDoc("test", true, 2, layer,
		func(ci, ai int, w float32) {},
		func(ci int) int { return 2 },
	)
	if err == nil {
		t.Error("expected error for bias count mismatch")
	}
}

func TestInstallLayerDocRowWidthMismatch(t *testing.T) {
	layer := persistence.LayerWeights[float32]{
		Name:    "test",
		Weights: [][]float32{{0.1, 0.2}}, // 2 weights in row
	}
	err := installLayerDoc("test", false, 1, layer,
		func(ci, ai int, w float32) {},
		func(ci int) int { return 3 }, // net cell has 3 axons
	)
	if err == nil {
		t.Error("expected error for row width mismatch")
	}
}
