// Level loss
package nn

type Loss Float

// The minimum value of the error limit at which training is forcibly terminated
const MinLevelLoss Loss = 10e-33

// Setter

// Getter

// Initializing

// Return

// Checking
func (l Loss) Check() Checker {
	switch {
	case l < 0:
		return MinLevelLoss
	default:
		return l
	}
}