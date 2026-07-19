// Package bad_cap has a symbol whose AI-Meta block exceeds 12 lines (§6.5).
package bad_cap

// BadCap has a block with 13 lines (label + 12 fields).
//
// AI-Meta:
//   - Purpose: Tests CAP violation detection.
//   - Usage: var _ = BadCap{}.
//   - Lifecycle: none.
//   - Concurrency: Safe.
//   - Errors: none.
//   - Related: none.
//   - Constraints: none.
//   - Implementations: none.
//   - Stability: Stable.
//   - Purpose: extra line one.
//   - Purpose: extra line two.
//   - Purpose: extra line three.
type BadCap struct{}
