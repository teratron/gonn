// Package good is a clean fixture with no AI-Meta violations.
package good

// GoodType is a well-formed type.
//
// AI-Meta:
//   - Purpose: A type that serves as a clean fixture for AI-Meta compliance tests.
//   - Usage: var _ = GoodType{}.
//   - Concurrency: NotSafe.
//   - Related: [GoodFunc].
//   - Stability: Stable.
type GoodType struct{}

// GoodFunc is a well-formed function.
//
// AI-Meta:
//   - Purpose: A function that serves as a clean fixture for AI-Meta compliance tests.
//   - Related: [GoodType].
//   - Stability: Stable.
func GoodFunc() {}
