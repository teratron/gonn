// Package bad_artifact has a symbol with an SDD artifact reference (§6.1).
package bad_artifact

// BadArtifact references INV-2 in its Constraints field.
//
// AI-Meta:
//   - Purpose: Tests ARTIFACT violation detection.
//   - Constraints: INV-2 (immutable topology), see .design/main/specifications/l1-neural-network-architecture.md.
//   - Stability: Stable.
type BadArtifact struct{}
