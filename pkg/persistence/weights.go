// Package persistence — weights document.
//
// Implements [l2-persistence-impl] §5.1 read/write surface for weights.json
// per [l1-network-persistence] §5.3. The weights document references its
// companion config via SHA-256 (PERS-3) and round-trips trained float
// values with ULP-1 precision for float32, bit-identical for float64
// (PERS-4).
package persistence

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"

	"github.com/teratron/gonn/pkg/utils"
)

// configHashPrefix marks the algorithm of WeightsDoc.ConfigHash. Keeping
// the prefix explicit lets future schema versions add more algorithms
// without breaking existing readers (they reject unknown prefixes).
const configHashPrefix = "sha256:"

// LayerWeights is the on-disk image of one layer's trainable parameters.
// Weights is the dense matrix shaped [outputs][inputs]; Biases is the
// per-output bias vector (empty when the layer has Bias == false).
type LayerWeights[T utils.Float] struct {
	Name    string `json:"name"`
	Weights [][]T  `json:"weights"`
	Biases  []T    `json:"biases,omitempty"`
}

// WeightsDoc is the on-disk projection of a trained network's weights.
// The ConfigHash field anchors the document to a specific config.json so
// loading mismatched pairs surfaces an ErrIntegrity (PERS-3) instead of
// silently producing a network with the wrong topology.
type WeightsDoc[T utils.Float] struct {
	SchemaVersion string             `json:"schema_version"`
	ConfigHash    string             `json:"config_hash"`
	Layers        []LayerWeights[T]  `json:"layers"`
}

// WriteWeights serialises w to path atomically, embedding a SHA-256 of
// cfg as ConfigHash so subsequent ReadWeights calls can verify the
// pairing. The same canonical-config hashing rules from PERS-2 / PERS-3
// apply — re-writing an unchanged (cfg, w) pair yields a byte-identical
// file.
func WriteWeights[T utils.Float](path string, cfg ConfigDoc[T], w WeightsDoc[T]) error {
	hash, err := configHashHex(cfg)
	if err != nil {
		return err
	}
	w.SchemaVersion = SchemaVersion
	w.ConfigHash = configHashPrefix + hash
	data, err := canonicalWeightsBytes(w)
	if err != nil {
		return err
	}
	return atomicWrite(path, data)
}

// ReadWeights loads both config and weights documents, verifying the
// SHA-256 anchor between them per PERS-3. A mismatch is reported as an
// ErrIntegrity-wrapped error so callers can route via errors.Is. Schema
// drift is enforced separately by ReadConfig (PERS-1).
func ReadWeights[T utils.Float](configPath, weightsPath string) (ConfigDoc[T], WeightsDoc[T], error) {
	cfg, err := ReadConfig[T](configPath)
	if err != nil {
		return cfg, WeightsDoc[T]{}, err
	}

	raw, err := os.ReadFile(weightsPath)
	if err != nil {
		return cfg, WeightsDoc[T]{}, utils.Wrap(utils.ErrIO, err, "ReadWeights: open %q", weightsPath)
	}

	var w WeightsDoc[T]
	if err := json.Unmarshal(raw, &w); err != nil {
		return cfg, w, utils.Wrap(utils.ErrIO, err, "ReadWeights: decode %q", weightsPath)
	}

	if err := checkSchemaVersion(w.SchemaVersion); err != nil {
		return cfg, w, err
	}

	wantHash, err := configHashHex(cfg)
	if err != nil {
		return cfg, w, err
	}
	gotHash, ok := strings.CutPrefix(w.ConfigHash, configHashPrefix)
	if !ok {
		return cfg, w, utils.Newf(utils.ErrIntegrity,
			"weights config_hash uses unknown algorithm: %q", w.ConfigHash)
	}
	if gotHash != wantHash {
		return cfg, w, utils.NewIntegrityError("config_hash",
			configHashPrefix+wantHash, w.ConfigHash)
	}
	return cfg, w, nil
}

// canonicalWeightsBytes mirrors canonicalConfigBytes for the weights
// document. encoding/json's deterministic struct ordering plus the
// strconv-driven float formatting give us PERS-4 round-trip integrity
// without a bespoke encoder.
func canonicalWeightsBytes[T utils.Float](w WeightsDoc[T]) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(w); err != nil {
		return nil, utils.Wrap(utils.ErrIO, err, "canonicalWeightsBytes: encode")
	}
	return buf.Bytes(), nil
}
