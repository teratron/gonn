package utils

// Float is the numeric type constraint used throughout the GoNN library.
// All generic numeric parameters must be bound with this constraint,
// which allows the caller to choose between float32 (speed) and float64
// (precision) at instantiation time.
//
// AI-Meta:
//   - Purpose: Type constraint restricting generic type parameters to float32 or float64.
//   - Usage: Declare generic functions as func Foo[T Float](x T); pass float32 or float64 at call site.
//   - Implementations: float32, float64.
type Float interface {
	float32 | float64
}
