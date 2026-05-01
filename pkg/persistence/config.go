// Package persistence — on-disk representation of GoNN networks.
//
// Implements [l2-persistence-impl] §5.1 (public surface), §5.2 (atomic
// write), §5.3 (canonicalisation) on top of the [l1-network-persistence]
// invariants PERS-1..PERS-4. The package owns two artefact types:
// ConfigDoc[T] (architecture + hyperparameters, human-editable) and
// WeightsDoc[T] (trained values, machine-only, in weights.go).
//
// Stdlib only per C29: encoding/json, crypto/sha256, os, io, strconv.
package persistence

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/utils"
)

// SchemaVersion is the wire-format version embedded in every ConfigDoc /
// WeightsDoc. Bump rules per [l1-network-persistence] PERS-1: major change
// is breaking; minor / patch are forward-compatible (warning, not fatal).
const SchemaVersion = "1.0.0"

// HiddenLayerDoc mirrors one [pkg/nn].HiddenLayerSpec entry on disk.
// Activation is serialised as the canonical String() form (e.g. "ReLU")
// so JSON files remain self-describing across language bindings.
type HiddenLayerDoc struct {
	Size       uint   `json:"size"`
	Activation string `json:"activation"`
	Bias       bool   `json:"bias"`
}

// OutputDoc captures the output-layer shape on disk. Split from
// HiddenLayerDoc to mirror the JSON schema in
// [l1-network-persistence] §5.2.
type OutputDoc struct {
	Size       uint   `json:"size"`
	Activation string `json:"activation"`
	Bias       bool   `json:"bias"`
}

// TrainingDoc holds the hyperparameters that feed Compile(). LearningRate
// and LossLimit are stored as JSON numbers; their Go type T is recovered
// on read via generic specialisation.
type TrainingDoc[T utils.Float] struct {
	LearningRate  T      `json:"learning_rate"`
	Loss          string `json:"loss"`
	LossLimit     T      `json:"loss_limit"`
	MaxIterations uint   `json:"max_iterations"`
	WeightInit    string `json:"weight_init"`
	RNGSeed       int64  `json:"rng_seed,omitempty"`
}

// ConfigDoc is the on-disk projection of the in-memory [pkg/nn].Config[T].
// It is intentionally a separate type: the persistence package depends on
// pkg/utils only — pulling pkg/nn would create an import cycle. Conversion
// happens at the pkg/nn boundary.
type ConfigDoc[T utils.Float] struct {
	SchemaVersion string           `json:"schema_version"`
	LibVersion    string           `json:"lib_version,omitempty"`
	Precision     string           `json:"precision"`
	InputSize     uint             `json:"input_size"`
	HiddenLayers  []HiddenLayerDoc `json:"hidden_layers"`
	Output        OutputDoc        `json:"output"`
	Training      TrainingDoc[T]   `json:"training"`
}

// WriteConfig serialises cfg to path atomically (§5.2). The format is
// deterministic per PERS-2: re-serialising a freshly read file yields a
// byte-identical artefact, which makes configs diffable in git.
//
// The on-disk file is written via tmp + Sync + Rename so a crash mid-write
// leaves either the prior file untouched or no file — never a half-written
// document. Errors wrap project sentinels: ErrIO for filesystem failures.
func WriteConfig[T utils.Float](path string, cfg ConfigDoc[T]) error {
	cfg = withDefaults(cfg)
	data, err := canonicalConfigBytes(cfg)
	if err != nil {
		return err
	}
	return atomicWrite(path, data)
}

// ReadConfig parses a config document from path and validates schema_version
// per PERS-1. Major-version mismatch returns an ErrUserConfig-wrapped error;
// minor / patch drift is silently tolerated to preserve forward compat as
// stated in [l1-network-persistence] §2.
func ReadConfig[T utils.Float](path string) (ConfigDoc[T], error) {
	var doc ConfigDoc[T]
	raw, err := os.ReadFile(path)
	if err != nil {
		return doc, utils.Wrap(utils.ErrIO, err, "ReadConfig: open %q", path)
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		return doc, utils.Wrap(utils.ErrIO, err, "ReadConfig: decode %q", path)
	}
	if err := checkSchemaVersion(doc.SchemaVersion); err != nil {
		return doc, err
	}
	return doc, nil
}

// configHashHex returns the SHA-256 of the canonical config bytes encoded
// as lowercase hex. Callers add the "sha256:" prefix when embedding the
// digest into WeightsDoc.ConfigHash.
func configHashHex[T utils.Float](cfg ConfigDoc[T]) (string, error) {
	data, err := canonicalConfigBytes(withDefaults(cfg))
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

// canonicalConfigBytes produces the stable byte representation per §5.3.
// stdlib encoding/json sorts map keys alphabetically and emits struct
// fields in declaration order, so PERS-2 holds without a custom encoder.
// PERS-4 round-trip integrity is satisfied because Go's json package
// uses strconv.FormatFloat(_, 'g', -1, …) internally — float32 weights
// retain ULP-1 precision and float64 weights are bit-identical.
func canonicalConfigBytes[T utils.Float](cfg ConfigDoc[T]) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(cfg); err != nil {
		return nil, utils.Wrap(utils.ErrIO, err, "canonicalConfigBytes: encode")
	}
	return buf.Bytes(), nil
}

// atomicWrite materialises data at path using the tmp + Sync + Rename
// dance. The temp file lives in the same directory so Rename is a true
// atomic operation on POSIX and Windows (same volume).
func atomicWrite(path string, data []byte) error {
	dir := filepath.Dir(path)
	if dir == "" {
		dir = "."
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return utils.Wrap(utils.ErrIO, err, "atomicWrite: mkdir %q", dir)
	}
	tmp, err := os.CreateTemp(dir, filepath.Base(path)+".tmp-*")
	if err != nil {
		return utils.Wrap(utils.ErrIO, err, "atomicWrite: create tmp in %q", dir)
	}
	tmpName := tmp.Name()
	cleanup := func() {
		_ = tmp.Close()
		_ = os.Remove(tmpName)
	}
	if _, err := tmp.Write(data); err != nil {
		cleanup()
		return utils.Wrap(utils.ErrIO, err, "atomicWrite: write %q", tmpName)
	}
	if err := tmp.Sync(); err != nil {
		cleanup()
		return utils.Wrap(utils.ErrIO, err, "atomicWrite: sync %q", tmpName)
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpName)
		return utils.Wrap(utils.ErrIO, err, "atomicWrite: close %q", tmpName)
	}
	if err := os.Rename(tmpName, path); err != nil {
		_ = os.Remove(tmpName)
		return utils.Wrap(utils.ErrIO, err, "atomicWrite: rename %q -> %q", tmpName, path)
	}
	return nil
}

// withDefaults populates fields whose zero value is ambiguous on the wire.
// SchemaVersion and Precision are required by PERS-1 / PERS-4 so we fill
// them automatically when callers leave them blank.
func withDefaults[T utils.Float](cfg ConfigDoc[T]) ConfigDoc[T] {
	if cfg.SchemaVersion == "" {
		cfg.SchemaVersion = SchemaVersion
	}
	if cfg.Precision == "" {
		cfg.Precision = precisionFor[T]()
	}
	return cfg
}

// precisionFor returns the JSON precision tag matching the generic T.
// Resolved via a stdlib type switch on a zero value of T — no reflect,
// no unsafe, no allocations after the first call.
func precisionFor[T utils.Float]() string {
	var z T
	switch any(z).(type) {
	case float32:
		return "float32"
	case float64:
		return "float64"
	default:
		return "float64"
	}
}

// checkSchemaVersion enforces PERS-1: same MAJOR is mandatory; minor /
// patch drift is silently accepted. Returns ErrUserConfig-wrapped error
// on major mismatch or empty schema_version.
func checkSchemaVersion(got string) error {
	if got == "" {
		return utils.Newf(utils.ErrUserConfig, "schema_version missing — file is not a GoNN artefact")
	}
	if majorOf(got) != majorOf(SchemaVersion) {
		return utils.Newf(utils.ErrUserConfig,
			"schema_version major mismatch: file %q, library %q", got, SchemaVersion)
	}
	return nil
}

// majorOf returns the MAJOR component of a semver string ("1.2.3" → "1").
// Inputs without a dot are treated as the whole value (e.g. "1" → "1").
func majorOf(v string) string {
	if major, _, ok := strings.Cut(v, "."); ok {
		return major
	}
	return v
}

// resolveActivation maps the canonical String() name back to an
// activation.Type. Unknown values yield ErrUserConfig so callers can
// route via errors.Is.
func resolveActivation(name string) (activation.Type, error) {
	for t := activation.Type(0); t <= activation.TanH; t++ {
		if t.String() == name {
			return t, nil
		}
	}
	return 0, utils.NewActivationError(name)
}

// resolveLoss maps the canonical loss String() name back to loss.Type.
// CROSS_ENTROPY and CCE share the same string, so both round-trip to the
// canonical CCE constant.
func resolveLoss(name string) (loss.Type, error) {
	for t := loss.Type(0); t <= loss.ARCTAN; t++ {
		if t.String() == name {
			return t, nil
		}
	}
	return 0, utils.Newf(utils.ErrUserConfig, "loss %q is not registered", name)
}

