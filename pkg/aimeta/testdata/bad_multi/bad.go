// Package bad_multi has a symbol with a multi-line value (§6.3).
package bad_multi

// BadMulti has a nested-bullet Errors value.
//
// AI-Meta:
//   - Purpose: Tests MULTI violation detection.
//   - Errors:
//     - ErrUserConfig
//     - ErrIntegrity
//   - Stability: Stable.
type BadMulti struct{}
