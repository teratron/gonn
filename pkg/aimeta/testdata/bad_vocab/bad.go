// Package bad_vocab has a symbol with an unknown AI-Meta field (§6.2 vocab creep).
package bad_vocab

// BadVocab uses an unknown Author field.
//
// AI-Meta:
//   - Purpose: Tests VOCAB violation detection.
//   - Author: alice@example.com
//   - Stability: Stable.
type BadVocab struct{}
