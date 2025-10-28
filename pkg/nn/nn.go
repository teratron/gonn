package nn

import (
	"github.com/teratron/gonn/pkg/utils"
)

// NeuralNetwork interface.
/*type NeuralNetwork interface {
	pkg.NeuralNetwork
}*/

// Floater interface.
/*type Floater interface {
	Floater
}*/

// Activation function mode.
/*const (
	LINEAR    = activation.LINEAR    // LINEAR - Linear/identity.
	RELU      = activation.RELU      // RELU - ReLu (rectified linear unit).
	LEAKYRELU = activation.LEAKYRELU // LEAKYRELU - Leaky ReLu (leaky rectified linear unit).
	SIGMOID   = activation.SIGMOID   // SIGMOID - Logistic, a.k.a. sigmoid or soft step.
	TANH      = activation.TANH      // TANH - TanH (hyperbolic tangent).
)

// The mode of calculation of the total error.
const (
	MSE    = loss.MSE    // MSE - Mean Squared Error.
	RMSE   = loss.RMSE   // RMSE - Root Mean Squared Error.
	ARCTAN = loss.ARCTAN // ARCTAN - Arctan Error.
	AVG    = loss.AVG    // AVG - Average Error.
)*/

// New returns a new neural network instance.
func New[T float32 | float64](reader ...string) *NN[T] {
	if len(reader) > 0 {
		// return Get(reader[0])
		var err error
		r := utils.ReadFile(reader[0])
		switch v := r.GetValue("name").(type) {
		case error:
			err = v
		case string:
			if n := Get(v); n != nil {
				if err = r.Decode(n); err == nil {
					r.ClearData()
					n.Init(r)
					return n
				}
			}
		}

	}

	return perceptron[T]()
}
