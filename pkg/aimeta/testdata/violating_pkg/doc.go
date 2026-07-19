// Package violating_pkg is a synthetic package for resolver golden tests.
// It intentionally contains fixable AI-Meta violations (LABEL and INDENT).
package violating_pkg

// Foo is a sample exported function with LABEL and INDENT violations.
//
// AI-Meta:
//   - Purpose: Sample function with wrong label casing and indent.
//   - Stability: Stable.
func Foo() {}
