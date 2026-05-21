package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/teratron/gonn/pkg/persistence"
	"github.com/teratron/gonn/pkg/utils"
)

// ── helpers ──────────────────────────────────────────────────────────────────

// xorConfig returns a config.json for the canonical XOR problem.
func xorConfig(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	cfg := persistence.ConfigDoc[float32]{
		InputSize: 2,
		HiddenLayers: []persistence.HiddenLayerDoc{
			{Size: 4, Activation: "SIGMOID", Bias: true},
		},
		Output: persistence.OutputDoc{Size: 1, Activation: "SIGMOID", Bias: true},
		Training: persistence.TrainingDoc[float32]{
			LearningRate:  0.3,
			Loss:          "MSE",
			LossLimit:     1e-4,
			MaxIterations: 15_000,
			WeightInit:    "xavier",
		},
	}
	cfgPath := filepath.Join(dir, "config.json")
	if err := persistence.WriteConfig(cfgPath, cfg); err != nil {
		t.Fatalf("xorConfig: write: %v", err)
	}
	return cfgPath
}

// xorCSV writes the 4-row XOR dataset to a temp file and returns its path.
func xorCSV(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "xor.csv")
	content := "0,0,0\n0,1,1\n1,0,1\n1,1,0\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("xorCSV: write: %v", err)
	}
	return path
}

// ── dispatch / routing ───────────────────────────────────────────────────────

func TestDispatchUnknownSubcommand(t *testing.T) {
	code := dispatch("nosuchcmd", nil)
	if code != exitUserConfig {
		t.Errorf("unknown subcommand: want exit %d, got %d", exitUserConfig, code)
	}
}

func TestDispatchNoArgs(t *testing.T) {
	// Calling dispatch("version", nil) exercises the happy path — ensures
	// the router reaches versionCmd without panicking.
	code := dispatch("version", nil)
	if code != exitOK {
		t.Errorf("version with no args: want exit %d, got %d", exitOK, code)
	}
}

// ── version ──────────────────────────────────────────────────────────────────

func TestVersionCmd(t *testing.T) {
	code := versionCmd(nil)
	if code != exitOK {
		t.Errorf("versionCmd: want exit %d, got %d", exitOK, code)
	}
}

func TestVersionCmdJSON(t *testing.T) {
	// Capture stdout.
	orig := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	code := versionCmd([]string{"--json"})

	w.Close()
	os.Stdout = orig

	buf := make([]byte, 4096)
	n, _ := r.Read(buf)

	if code != exitOK {
		t.Fatalf("versionCmd --json: exit %d", code)
	}
	var out struct {
		Version   string `json:"version"`
		GoVersion string `json:"go_version"`
	}
	if err := json.Unmarshal(buf[:n], &out); err != nil {
		t.Fatalf("parse JSON: %v", err)
	}
	if out.GoVersion == "" {
		t.Error("go_version is empty")
	}
}

// ── exitCode mapping ─────────────────────────────────────────────────────────

func TestExitCodeMapping(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want int
	}{
		{"nil", nil, exitOK},
		{"ErrUserConfig", fmt.Errorf("bad arg: %w", errUserCfg), exitUserConfig},
		{"ErrInputData", fmt.Errorf("bad data: %w", errInputData), exitInputData},
		{"ErrCompute", fmt.Errorf("nan: %w", errCompute), exitTrainingFailure},
		{"ErrIntegrity", fmt.Errorf("hash mismatch: %w", errIntegrity), exitIntegrity},
		{"ErrIO", fmt.Errorf("disk full: %w", errIO), exitIO},
		{"generic", fmt.Errorf("something else"), exitGeneric},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := exitCode(tc.err); got != tc.want {
				t.Errorf("exitCode(%v): got %d, want %d", tc.err, got, tc.want)
			}
		})
	}
}

// ── resolvers ────────────────────────────────────────────────────────────────

func TestResolveActivation(t *testing.T) {
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
			_, err := resolveActivation(tc.name)
			if (err == nil) != tc.ok {
				t.Errorf("resolveActivation(%q): err=%v, want ok=%v", tc.name, err, tc.ok)
			}
		})
	}
}

func TestResolveLoss(t *testing.T) {
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
			_, err := resolveLoss(tc.name)
			if (err == nil) != tc.ok {
				t.Errorf("resolveLoss(%q): err=%v, want ok=%v", tc.name, err, tc.ok)
			}
		})
	}
}

func TestResolveWeightInit(t *testing.T) {
	if _, err := resolveWeightInit("xavier"); err != nil {
		t.Errorf("xavier: %v", err)
	}
	if _, err := resolveWeightInit("he"); err != nil {
		t.Errorf("he: %v", err)
	}
	if _, err := resolveWeightInit(""); err != nil {
		t.Errorf("empty → default: %v", err)
	}
	if _, err := resolveWeightInit("unknown"); err == nil {
		t.Error("unknown should return error")
	}
}

// ── parseFloats ───────────────────────────────────────────────────────────────

func TestParseFloats(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		got, err := parseFloats[float32]("1.0,0.0,0.5")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		want := []float32{1.0, 0.0, 0.5}
		for i := range want {
			if math.Abs(float64(got[i]-want[i])) > 1e-6 {
				t.Errorf("[%d]: got %v, want %v", i, got[i], want[i])
			}
		}
	})
	t.Run("bad float", func(t *testing.T) {
		_, err := parseFloats[float32]("1.0,abc,0.5")
		if err == nil {
			t.Error("expected error for non-numeric value")
		}
	})
}

// ── loadSamples ───────────────────────────────────────────────────────────────

func TestLoadSamplesFull(t *testing.T) {
	csvPath := xorCSV(t)
	samples, err := loadSamplesFull[float32](csvPath, 2, 1)
	if err != nil {
		t.Fatalf("loadSamplesFull: %v", err)
	}
	if len(samples) != 4 {
		t.Errorf("got %d samples, want 4", len(samples))
	}
	if samples[1].Target[0] != 1.0 {
		t.Errorf("row 1 target: got %v, want 1.0", samples[1].Target[0])
	}
}

func TestLoadSamplesThresholdTriggersStreaming(t *testing.T) {
	csvPath := xorCSV(t)

	// Force streaming by setting threshold to 0.
	old := csvStreamThreshold
	csvStreamThreshold = 0
	defer func() { csvStreamThreshold = old }()

	samples, err := loadSamples[float32](csvPath, 2, 1)
	if err != nil {
		t.Fatalf("streaming load: %v", err)
	}
	if len(samples) != 4 {
		t.Errorf("got %d samples, want 4", len(samples))
	}
}

func TestLoadSamplesColumnMismatch(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.csv")
	_ = os.WriteFile(path, []byte("1,0\n"), 0o644) // only 2 cols, expect 3
	_, err := loadSamplesFull[float32](path, 2, 1)
	if err == nil {
		t.Error("expected error for column count mismatch")
	}
}

// ── buildFromConfig ───────────────────────────────────────────────────────────

func TestBuildFromConfig(t *testing.T) {
	cfgPath := xorConfig(t)
	doc, err := persistence.ReadConfig[float32](cfgPath)
	if err != nil {
		t.Fatalf("ReadConfig: %v", err)
	}
	n, err := buildFromConfig(doc)
	if err != nil {
		t.Fatalf("buildFromConfig: %v", err)
	}
	// Basic smoke: query should not error on a valid input.
	if _, err := n.Query([]float32{0, 1}); err != nil {
		t.Errorf("Query: %v", err)
	}
}

// ── query subcommand ──────────────────────────────────────────────────────────

func TestQueryCmdMissingFlags(t *testing.T) {
	code := queryCmd(nil)
	if code != exitUserConfig {
		t.Errorf("missing flags: want exit %d, got %d", exitUserConfig, code)
	}
}

func TestQueryCmdBadPrecision(t *testing.T) {
	code := queryCmd([]string{"--config", "x", "--weights", "y", "--input", "1", "--precision", "float16"})
	if code != exitUnsupported {
		t.Errorf("bad precision: want exit %d, got %d", exitUnsupported, code)
	}
}

// ── train subcommand ──────────────────────────────────────────────────────────

func TestTrainCmdMissingFlags(t *testing.T) {
	code := trainCmd(nil)
	if code != exitUserConfig {
		t.Errorf("missing flags: want exit %d, got %d", exitUserConfig, code)
	}
}

func TestTrainCmdBadPrecision(t *testing.T) {
	code := trainCmd([]string{"--config", "x", "--data", "y", "--precision", "float16"})
	if code != exitUnsupported {
		t.Errorf("bad precision: want exit %d, got %d", exitUnsupported, code)
	}
}

// ── verify subcommand ─────────────────────────────────────────────────────────

func TestVerifyCmdMissingFlags(t *testing.T) {
	code := verifyCmd(nil)
	if code != exitUserConfig {
		t.Errorf("missing flags: want exit %d, got %d", exitUserConfig, code)
	}
}

// ── end-to-end XOR smoke test ─────────────────────────────────────────────────

// TestXOREndToEnd trains XOR, saves weights, then queries and verifies.
// This exercises the full train → query → verify pipeline.
func TestXOREndToEnd(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping end-to-end XOR smoke test in short mode")
	}

	dir := t.TempDir()
	cfgPath := xorConfig(t)
	csvPath := xorCSV(t)
	weightsPath := filepath.Join(dir, "weights.json")

	// Train.
	code := trainCmd([]string{
		"--config", cfgPath,
		"--data", csvPath,
		"--out", weightsPath,
	})
	if code != exitOK {
		t.Fatalf("train: exit code %d", code)
	}
	if _, err := os.Stat(weightsPath); err != nil {
		t.Fatalf("weights file not created: %v", err)
	}

	// Query.
	orig := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	code = queryCmd([]string{
		"--config", cfgPath,
		"--weights", weightsPath,
		"--input", "0,1",
		"--json",
	})
	w.Close()
	os.Stdout = orig

	buf := make([]byte, 4096)
	n, _ := r.Read(buf)

	if code != exitOK {
		t.Fatalf("query: exit code %d", code)
	}
	var qout struct {
		Output []float64 `json:"output"`
	}
	if err := json.Unmarshal(buf[:n], &qout); err != nil {
		t.Fatalf("query JSON: %v", err)
	}
	if len(qout.Output) != 1 {
		t.Fatalf("query output len %d, want 1", len(qout.Output))
	}
	// XOR(0,1) = 1 — trained network should be close to 1.
	if math.Abs(qout.Output[0]-1.0) > 0.1 {
		t.Errorf("query(0,1) = %v, want ≈1.0", qout.Output[0])
	}

	// Verify.
	origV := os.Stdout
	rv, wv, _ := os.Pipe()
	os.Stdout = wv

	code = verifyCmd([]string{
		"--config", cfgPath,
		"--weights", weightsPath,
		"--data", csvPath,
		"--json",
	})
	wv.Close()
	os.Stdout = origV

	bufV := make([]byte, 4096)
	nv, _ := rv.Read(bufV)

	if code != exitOK {
		t.Fatalf("verify: exit code %d", code)
	}
	var vout struct {
		Loss float64 `json:"loss"`
	}
	if err := json.Unmarshal(bufV[:nv], &vout); err != nil {
		t.Fatalf("verify JSON: %v", err)
	}
	if vout.Loss > 1e-3 {
		t.Errorf("verify loss %v > 1e-3", vout.Loss)
	}
}

// ── JSON --json flag ──────────────────────────────────────────────────────────

func TestTrainCmdJSONOutput(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}

	dir := t.TempDir()
	cfgPath := xorConfig(t)
	csvPath := xorCSV(t)
	weightsPath := filepath.Join(dir, "weights.json")

	orig := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	code := trainCmd([]string{
		"--config", cfgPath,
		"--data", csvPath,
		"--out", weightsPath,
		"--json",
	})
	w.Close()
	os.Stdout = orig

	buf := make([]byte, 4096)
	n, _ := r.Read(buf)

	if code != exitOK {
		t.Fatalf("train --json: exit %d", code)
	}
	var out struct {
		Epochs    uint    `json:"epochs"`
		FinalLoss float64 `json:"final_loss"`
	}
	if err := json.Unmarshal(buf[:n], &out); err != nil {
		t.Fatalf("parse JSON: %v (%s)", err, strings.TrimSpace(string(buf[:n])))
	}
	if out.Epochs == 0 {
		t.Error("epochs is 0")
	}
}

// ── dispatch coverage ─────────────────────────────────────────────────────────

func TestDispatchAllSubcommands(t *testing.T) {
	// Calling each subcommand with no args returns exitUserConfig (missing
	// required flags) — but exercises the dispatch router branches.
	cases := []string{"train", "query", "verify"}
	for _, sub := range cases {
		t.Run(sub, func(t *testing.T) {
			code := dispatch(sub, nil)
			if code != exitUserConfig {
				t.Errorf("dispatch(%q, nil): want %d, got %d", sub, exitUserConfig, code)
			}
		})
	}
}

// ── printErrorJSON ────────────────────────────────────────────────────────────

func TestPrintErrorJSON(t *testing.T) {
	orig := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	printErrorJSON(fmt.Errorf("test error"))

	w.Close()
	os.Stdout = orig
	buf := make([]byte, 1024)
	n, _ := r.Read(buf)

	var out struct {
		Error string `json:"error"`
	}
	if err := json.Unmarshal(buf[:n], &out); err != nil {
		t.Fatalf("parse JSON: %v", err)
	}
	if out.Error != "test error" {
		t.Errorf("error field: got %q, want %q", out.Error, "test error")
	}
}

// ── runTrain error paths ──────────────────────────────────────────────────────

func TestRunTrainBadConfigPath(t *testing.T) {
	code := runTrain[float32]("/no/such/config.json", "/no/data.csv", "weights.json", "", false)
	if code != exitIO {
		t.Errorf("bad config: want exit %d, got %d", exitIO, code)
	}
}

func TestRunTrainBadDataPath(t *testing.T) {
	cfgPath := xorConfig(t)
	dir := t.TempDir()
	code := runTrain[float32](cfgPath, "/no/such/data.csv", filepath.Join(dir, "w.json"), "", false)
	if code != exitIO {
		t.Errorf("bad data: want exit %d, got %d", exitIO, code)
	}
}

func TestRunTrainJSONOnError(t *testing.T) {
	// Exercises the --json error branch in runTrain.
	code := runTrain[float32]("/no/config.json", "/no/data.csv", "w.json", "", true)
	if code == exitOK {
		t.Error("expected non-zero exit code")
	}
}

// ── runQuery error paths ──────────────────────────────────────────────────────

func TestRunQueryBadConfigPath(t *testing.T) {
	code := runQuery[float32]("/no/config.json", "/no/weights.json", "0,1", false)
	if code != exitIO {
		t.Errorf("bad config: want exit %d, got %d", exitIO, code)
	}
}

func TestRunQueryJSONOnError(t *testing.T) {
	code := runQuery[float32]("/no/config.json", "/no/weights.json", "0,1", true)
	if code == exitOK {
		t.Error("expected non-zero exit code")
	}
}

func TestRunQueryBadInput(t *testing.T) {
	cfgPath := xorConfig(t)

	// Build and save a trained network first.
	dir := t.TempDir()
	weightsPath := filepath.Join(dir, "w.json")
	csvPath := xorCSV(t)
	_ = trainCmd([]string{"--config", cfgPath, "--data", csvPath, "--out", weightsPath})

	code := runQuery[float32](cfgPath, weightsPath, "not,a,number", false)
	if code != exitInputData {
		t.Errorf("bad input: want exit %d, got %d", exitInputData, code)
	}
}

// ── runVerify error paths ─────────────────────────────────────────────────────

func TestRunVerifyBadConfigPath(t *testing.T) {
	code := runVerify[float32]("/no/config.json", "/no/weights.json", "/no/data.csv", false)
	if code != exitIO {
		t.Errorf("bad config: want exit %d, got %d", exitIO, code)
	}
}

func TestRunVerifyJSONOnError(t *testing.T) {
	code := runVerify[float32]("/no/config.json", "/no/weights.json", "/no/data.csv", true)
	if code == exitOK {
		t.Error("expected non-zero exit code")
	}
}

func TestRunVerifyBadDataPath(t *testing.T) {
	cfgPath := xorConfig(t)
	dir := t.TempDir()
	weightsPath := filepath.Join(dir, "w.json")
	csvPath := xorCSV(t)
	_ = trainCmd([]string{"--config", cfgPath, "--data", csvPath, "--out", weightsPath})

	code := runVerify[float32](cfgPath, weightsPath, "/no/such/data.csv", false)
	if code != exitIO {
		t.Errorf("bad data: want exit %d, got %d", exitIO, code)
	}
}

// ── loadNetwork ───────────────────────────────────────────────────────────────

func TestLoadNetworkBadConfigPath(t *testing.T) {
	_, _, err := loadNetwork[float32]("/no/such/config.json", "")
	if err == nil {
		t.Fatal("expected error for missing config")
	}
}

func TestLoadNetworkBadWeightsPath(t *testing.T) {
	cfgPath := xorConfig(t)
	_, _, err := loadNetwork[float32](cfgPath, "/no/such/weights.json")
	if err == nil {
		t.Fatal("expected error for missing weights")
	}
}

// ── buildFromConfig error paths ───────────────────────────────────────────────

func TestBuildFromConfigBadActivation(t *testing.T) {
	doc := persistence.ConfigDoc[float32]{
		InputSize:    2,
		HiddenLayers: []persistence.HiddenLayerDoc{{Size: 4, Activation: "NOPE", Bias: true}},
		Output:       persistence.OutputDoc{Size: 1, Activation: "SIGMOID", Bias: true},
		Training:     persistence.TrainingDoc[float32]{LearningRate: 0.1, Loss: "MSE", MaxIterations: 10},
	}
	_, err := buildFromConfig(doc)
	if err == nil {
		t.Error("expected error for unknown activation")
	}
}

func TestBuildFromConfigBadLoss(t *testing.T) {
	doc := persistence.ConfigDoc[float32]{
		InputSize:    2,
		HiddenLayers: []persistence.HiddenLayerDoc{{Size: 4, Activation: "SIGMOID", Bias: true}},
		Output:       persistence.OutputDoc{Size: 1, Activation: "SIGMOID", Bias: true},
		Training:     persistence.TrainingDoc[float32]{LearningRate: 0.1, Loss: "NOSUCHLOSS", MaxIterations: 10},
	}
	_, err := buildFromConfig(doc)
	if err == nil {
		t.Error("expected error for unknown loss")
	}
}

// ── installLayerG error paths ─────────────────────────────────────────────────

func TestInstallLayerGCellCountMismatch(t *testing.T) {
	layer := persistence.LayerWeights[float32]{
		Name:    "test",
		Weights: [][]float32{{0.1}, {0.2}}, // 2 cells
	}
	err := installLayerG("test", false, 3, layer, // net has 3 cells
		func(ci, ai int, w float32) {},
		func(ci int) int { return 1 },
	)
	if err == nil {
		t.Error("expected error for cell count mismatch")
	}
}

func TestInstallLayerGBiasCountMismatch(t *testing.T) {
	layer := persistence.LayerWeights[float32]{
		Name:    "test",
		Weights: [][]float32{{0.1}, {0.2}},
		Biases:  []float32{0.5}, // only 1 bias but 2 cells
	}
	err := installLayerG("test", true, 2, layer,
		func(ci, ai int, w float32) {},
		func(ci int) int { return 2 },
	)
	if err == nil {
		t.Error("expected error for bias count mismatch")
	}
}

func TestInstallLayerGRowWidthMismatch(t *testing.T) {
	layer := persistence.LayerWeights[float32]{
		Name:    "test",
		Weights: [][]float32{{0.1, 0.2}}, // 2 weights in row
	}
	err := installLayerG("test", false, 1, layer,
		func(ci, ai int, w float32) {},
		func(ci int) int { return 3 }, // net cell has 3 axons
	)
	if err == nil {
		t.Error("expected error for row width mismatch")
	}
}

// ── moduleVersion ─────────────────────────────────────────────────────────────

func TestModuleVersion(t *testing.T) {
	v := moduleVersion()
	if v == "" {
		t.Error("moduleVersion returned empty string")
	}
}

// sentinel vars for the exitCode test table.
var (
	errUserCfg   = utils.ErrUserConfig
	errInputData = utils.ErrInputData
	errCompute   = utils.ErrCompute
	errIntegrity = utils.ErrIntegrity
	errIO        = utils.ErrIO
)
