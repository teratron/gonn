// Package persistence — round-trip and integrity tests.
//
// Covers PERS-1 (schema_version major guard), PERS-2 (deterministic
// canonical output), PERS-3 (config_hash anchor between config and
// weights), PERS-4 (ULP-1 float32 round-trip / bit-identical float64).
package persistence

import (
	"bytes"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/teratron/gonn/pkg/utils"
)

// sampleConfig produces a small but non-trivial ConfigDoc usable across
// the round-trip tests. Keeping a single source avoids drift between the
// f32 and f64 paths.
func sampleConfig[T utils.Float]() ConfigDoc[T] {
	return ConfigDoc[T]{
		LibVersion: "0.1.0",
		InputSize:  2,
		HiddenLayers: []HiddenLayerDoc{
			{Size: 4, Activation: "ReLU", Bias: true},
			{Size: 3, Activation: "Sigmoid", Bias: false},
		},
		Output: OutputDoc{Size: 1, Activation: "Sigmoid", Bias: true},
		Training: TrainingDoc[T]{
			LearningRate:  T(0.3),
			Loss:          "MSE",
			LossLimit:     T(1e-4),
			MaxIterations: 10_000,
			WeightInit:    "xavier",
			RNGSeed:       42,
		},
	}
}

func TestConfigRoundTripF32(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")

	want := sampleConfig[float32]()
	if err := WriteConfig(path, want); err != nil {
		t.Fatalf("WriteConfig: %v", err)
	}

	got, err := ReadConfig[float32](path)
	if err != nil {
		t.Fatalf("ReadConfig: %v", err)
	}

	if got.SchemaVersion != SchemaVersion {
		t.Errorf("SchemaVersion = %q, want %q", got.SchemaVersion, SchemaVersion)
	}
	if got.Precision != "float32" {
		t.Errorf("Precision = %q, want float32", got.Precision)
	}
	if got.Training.LearningRate != want.Training.LearningRate {
		t.Errorf("LearningRate = %v, want %v",
			got.Training.LearningRate, want.Training.LearningRate)
	}
	if len(got.HiddenLayers) != len(want.HiddenLayers) {
		t.Fatalf("HiddenLayers len = %d, want %d",
			len(got.HiddenLayers), len(want.HiddenLayers))
	}
}

func TestConfigRoundTripF64BitIdentical(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")

	want := sampleConfig[float64]()
	want.Training.LearningRate = math.Pi
	want.Training.LossLimit = math.SmallestNonzeroFloat64

	if err := WriteConfig(path, want); err != nil {
		t.Fatalf("WriteConfig: %v", err)
	}
	got, err := ReadConfig[float64](path)
	if err != nil {
		t.Fatalf("ReadConfig: %v", err)
	}

	if math.Float64bits(got.Training.LearningRate) != math.Float64bits(want.Training.LearningRate) {
		t.Errorf("LearningRate not bit-identical: got %x want %x",
			math.Float64bits(got.Training.LearningRate),
			math.Float64bits(want.Training.LearningRate))
	}
	if math.Float64bits(got.Training.LossLimit) != math.Float64bits(want.Training.LossLimit) {
		t.Errorf("LossLimit not bit-identical: got %x want %x",
			math.Float64bits(got.Training.LossLimit),
			math.Float64bits(want.Training.LossLimit))
	}
}

func TestConfigDeterminism(t *testing.T) {
	dir := t.TempDir()
	a := filepath.Join(dir, "a.json")
	b := filepath.Join(dir, "b.json")

	cfg := sampleConfig[float32]()
	if err := WriteConfig(a, cfg); err != nil {
		t.Fatalf("WriteConfig a: %v", err)
	}
	if err := WriteConfig(b, cfg); err != nil {
		t.Fatalf("WriteConfig b: %v", err)
	}

	bytesA, _ := os.ReadFile(a)
	bytesB, _ := os.ReadFile(b)
	if !bytes.Equal(bytesA, bytesB) {
		t.Errorf("PERS-2 violated: two writes produced different bytes\nA=%s\nB=%s",
			bytesA, bytesB)
	}
}

func TestSchemaVersionMajorMismatch(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")

	// Write a config then forcibly downgrade the schema_version on disk.
	cfg := sampleConfig[float32]()
	if err := WriteConfig(path, cfg); err != nil {
		t.Fatalf("WriteConfig: %v", err)
	}
	raw, _ := os.ReadFile(path)
	// SchemaVersion uses the live constant so this test keeps working
	// across minor bumps (1.0.0 → 1.1.0 in Phase 5 / Track C).
	corrupted := strings.Replace(string(raw),
		`"schema_version": "`+SchemaVersion+`"`,
		`"schema_version": "2.7.0"`, 1)
	if err := os.WriteFile(path, []byte(corrupted), 0o644); err != nil {
		t.Fatalf("rewrite: %v", err)
	}
	if !strings.Contains(string(corrupted), `"2.7.0"`) {
		t.Fatalf("test setup: SchemaVersion replace failed — corruption did not take effect")
	}

	_, err := ReadConfig[float32](path)
	if err == nil {
		t.Fatal("expected error on major mismatch, got nil")
	}
	if !errors.Is(err, utils.ErrUserConfig) {
		t.Errorf("expected ErrUserConfig, got %v", err)
	}
}

func TestSchemaVersionMissing(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	// Manually write a JSON without schema_version.
	body := `{"precision":"float32","input_size":2,"hidden_layers":[],"output":{"size":1,"activation":"Sigmoid","bias":true},"training":{"learning_rate":0.3,"loss":"MSE","loss_limit":0.0001,"max_iterations":1000,"weight_init":"xavier"}}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	_, err := ReadConfig[float32](path)
	if err == nil || !errors.Is(err, utils.ErrUserConfig) {
		t.Errorf("expected ErrUserConfig on missing schema_version, got %v", err)
	}
}

func TestWeightsRoundTrip(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.json")
	wPath := filepath.Join(dir, "weights.json")

	cfg := sampleConfig[float32]()
	if err := WriteConfig(cfgPath, cfg); err != nil {
		t.Fatalf("WriteConfig: %v", err)
	}

	weights := WeightsDoc[float32]{
		Layers: []LayerWeights[float32]{
			{
				Name:    "hidden_0",
				Weights: [][]float32{{0.1, 0.2}, {-0.3, 0.4}},
				Biases:  []float32{0.01, -0.02},
			},
		},
	}
	if err := WriteWeights(wPath, cfg, weights); err != nil {
		t.Fatalf("WriteWeights: %v", err)
	}

	gotCfg, gotW, err := ReadWeights[float32](cfgPath, wPath)
	if err != nil {
		t.Fatalf("ReadWeights: %v", err)
	}
	if gotCfg.InputSize != cfg.InputSize {
		t.Errorf("InputSize = %d, want %d", gotCfg.InputSize, cfg.InputSize)
	}
	if len(gotW.Layers) != 1 || gotW.Layers[0].Name != "hidden_0" {
		t.Fatalf("layers mismatch: %+v", gotW.Layers)
	}
	if gotW.Layers[0].Weights[1][0] != float32(-0.3) {
		t.Errorf("weight[1][0] = %v, want -0.3", gotW.Layers[0].Weights[1][0])
	}
}

func TestConfigHashMismatchYieldsErrIntegrity(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.json")
	wPath := filepath.Join(dir, "weights.json")

	cfg := sampleConfig[float32]()
	if err := WriteConfig(cfgPath, cfg); err != nil {
		t.Fatalf("WriteConfig: %v", err)
	}
	weights := WeightsDoc[float32]{
		Layers: []LayerWeights[float32]{{Name: "h0", Weights: [][]float32{{1}}}},
	}
	if err := WriteWeights(wPath, cfg, weights); err != nil {
		t.Fatalf("WriteWeights: %v", err)
	}

	// Mutate the on-disk config so the recomputed hash diverges.
	cfg.Training.LearningRate = 0.999
	if err := WriteConfig(cfgPath, cfg); err != nil {
		t.Fatalf("WriteConfig (mutated): %v", err)
	}

	_, _, err := ReadWeights[float32](cfgPath, wPath)
	if err == nil {
		t.Fatal("expected integrity error, got nil")
	}
	if !errors.Is(err, utils.ErrIntegrity) {
		t.Errorf("expected ErrIntegrity, got %v", err)
	}
}

func TestWeightsDeterminism(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.json")
	a := filepath.Join(dir, "a.json")
	b := filepath.Join(dir, "b.json")

	cfg := sampleConfig[float64]()
	if err := WriteConfig(cfgPath, cfg); err != nil {
		t.Fatalf("WriteConfig: %v", err)
	}
	weights := WeightsDoc[float64]{
		Layers: []LayerWeights[float64]{
			{Name: "h0", Weights: [][]float64{{0.5, -0.5}}, Biases: []float64{0.1}},
		},
	}
	if err := WriteWeights(a, cfg, weights); err != nil {
		t.Fatalf("WriteWeights a: %v", err)
	}
	if err := WriteWeights(b, cfg, weights); err != nil {
		t.Fatalf("WriteWeights b: %v", err)
	}
	bytesA, _ := os.ReadFile(a)
	bytesB, _ := os.ReadFile(b)
	if !bytes.Equal(bytesA, bytesB) {
		t.Errorf("weights determinism violated")
	}
}

func TestUnknownConfigHashAlgorithm(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.json")
	wPath := filepath.Join(dir, "weights.json")

	cfg := sampleConfig[float32]()
	if err := WriteConfig(cfgPath, cfg); err != nil {
		t.Fatalf("WriteConfig: %v", err)
	}
	body := `{"schema_version":"1.0.0","config_hash":"md5:deadbeef","layers":[]}`
	if err := os.WriteFile(wPath, []byte(body), 0o644); err != nil {
		t.Fatalf("write weights: %v", err)
	}
	_, _, err := ReadWeights[float32](cfgPath, wPath)
	if err == nil || !errors.Is(err, utils.ErrIntegrity) {
		t.Errorf("expected ErrIntegrity on unknown hash algo, got %v", err)
	}
}

func TestResolveActivationAndLoss(t *testing.T) {
	if _, err := resolveActivation("ReLU"); err != nil {
		t.Errorf("resolveActivation(ReLU): %v", err)
	}
	if _, err := resolveActivation("BogusActivation"); err == nil {
		t.Error("expected error for unknown activation")
	}
	if _, err := resolveLoss("MSE"); err != nil {
		t.Errorf("resolveLoss(MSE): %v", err)
	}
	if _, err := resolveLoss("BogusLoss"); err == nil {
		t.Error("expected error for unknown loss")
	}
}

func TestPrecisionDispatch(t *testing.T) {
	if got := precisionFor[float32](); got != "float32" {
		t.Errorf("precisionFor[float32] = %q", got)
	}
	if got := precisionFor[float64](); got != "float64" {
		t.Errorf("precisionFor[float64] = %q", got)
	}
}

func TestReadMissingFile(t *testing.T) {
	_, err := ReadConfig[float32](filepath.Join(t.TempDir(), "nope.json"))
	if err == nil || !errors.Is(err, utils.ErrIO) {
		t.Errorf("expected ErrIO for missing file, got %v", err)
	}
}

func TestReadConfigMalformedJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(path, []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := ReadConfig[float32](path)
	if err == nil || !errors.Is(err, utils.ErrIO) {
		t.Errorf("expected ErrIO on malformed JSON, got %v", err)
	}
}

func TestReadWeightsMalformedJSON(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.json")
	wPath := filepath.Join(dir, "weights.json")
	if err := WriteConfig(cfgPath, sampleConfig[float32]()); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(wPath, []byte("{broken"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, _, err := ReadWeights[float32](cfgPath, wPath)
	if err == nil || !errors.Is(err, utils.ErrIO) {
		t.Errorf("expected ErrIO, got %v", err)
	}
}

func TestReadWeightsMissingConfig(t *testing.T) {
	dir := t.TempDir()
	_, _, err := ReadWeights[float32](
		filepath.Join(dir, "missing-config.json"),
		filepath.Join(dir, "missing-weights.json"))
	if err == nil || !errors.Is(err, utils.ErrIO) {
		t.Errorf("expected ErrIO, got %v", err)
	}
}

func TestReadWeightsMissingWeights(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.json")
	if err := WriteConfig(cfgPath, sampleConfig[float32]()); err != nil {
		t.Fatal(err)
	}
	_, _, err := ReadWeights[float32](cfgPath, filepath.Join(dir, "nope.json"))
	if err == nil || !errors.Is(err, utils.ErrIO) {
		t.Errorf("expected ErrIO, got %v", err)
	}
}

func TestReadWeightsMissingSchemaVersion(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.json")
	wPath := filepath.Join(dir, "weights.json")
	cfg := sampleConfig[float32]()
	if err := WriteConfig(cfgPath, cfg); err != nil {
		t.Fatal(err)
	}
	hash, _ := configHashHex(cfg)
	body := `{"config_hash":"sha256:` + hash + `","layers":[]}`
	if err := os.WriteFile(wPath, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	_, _, err := ReadWeights[float32](cfgPath, wPath)
	if err == nil || !errors.Is(err, utils.ErrUserConfig) {
		t.Errorf("expected ErrUserConfig, got %v", err)
	}
}

func TestAtomicWriteCreatesParentDirs(t *testing.T) {
	dir := t.TempDir()
	nested := filepath.Join(dir, "sub", "deep", "config.json")
	if err := WriteConfig(nested, sampleConfig[float32]()); err != nil {
		t.Fatalf("WriteConfig nested: %v", err)
	}
	if _, err := os.Stat(nested); err != nil {
		t.Errorf("nested file missing: %v", err)
	}
}

func TestAtomicWriteOverwritesExisting(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	cfg := sampleConfig[float32]()
	if err := WriteConfig(path, cfg); err != nil {
		t.Fatal(err)
	}
	cfg.Training.LearningRate = 0.999
	if err := WriteConfig(path, cfg); err != nil {
		t.Fatalf("second WriteConfig: %v", err)
	}
	got, err := ReadConfig[float32](path)
	if err != nil {
		t.Fatal(err)
	}
	if got.Training.LearningRate != float32(0.999) {
		t.Errorf("LearningRate = %v, want 0.999", got.Training.LearningRate)
	}
}

func TestMajorOfNoDot(t *testing.T) {
	if got := majorOf("42"); got != "42" {
		t.Errorf("majorOf(42) = %q, want 42", got)
	}
	if got := majorOf("1.2.3"); got != "1" {
		t.Errorf("majorOf(1.2.3) = %q, want 1", got)
	}
}
