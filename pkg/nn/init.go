// Package nn side-effect imports — ensure the CPU backend is always registered
// in the compute registry when the public nn API is used (COMP-3 fallback chain).
// This blank import is the only wiring needed; no call site changes are required.
package nn

import _ "github.com/teratron/gonn/pkg/compute/cpu"
