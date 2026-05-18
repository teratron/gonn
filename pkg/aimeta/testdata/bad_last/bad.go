// Package bad_last has a symbol where the AI-Meta block is not last (§6.4).
package bad_last

// BadLast has prose after the AI-Meta block.
//
// AI-Meta:
//   - Purpose: Tests LAST violation detection.
//   - Stability: Stable.
//
// This additional prose appears after the AI-Meta block and triggers LAST.
type BadLast struct{}
