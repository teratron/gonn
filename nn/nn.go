package nn

import (
	"math/rand"
	"time"
)

const (
	DEFRATE float32 = .3      // Default rate
	MINLOSS float32 = .001    // The minimum value of the sum of the average square error at which the training is forcibly terminated
	MAXITER int     = 1000000 // The maximum number of iterations after which training is forcibly terminated
)

//+-------------------------------------------------------------+
//|	Synapse														|
//+-------------------------------------------------------------+
//
func (s *Synapse) Set(Setter) {
}

// Weights update function
func (s *Synapse) Get() Getter {
	return s
}

//+-------------------------------------------------------------+
//|	Weights														|
//+-------------------------------------------------------------+
// The function fills all weights with random numbers from -0.5 to 0.5
func (w *Weight) Set(m *Matrix) {
	rand.Seed(time.Now().UTC().UnixNano())
	for i := 0; i < m.Index; i++ {
		n := m.Synapse[i].Size[0] - 1
		for j := 0; j < m.Synapse[i].Size[0]; j++ {
			for k := 0; k < m.Synapse[i].Size[1]; k++ {
				if j == n && m.Bias == 0 {
					m.Synapse[i].Weight[j][k] = 0
				} else {
					m.Synapse[i].Weight[j][k] = rand.Float32() - .5
				}
			}
		}
	}
}

// Weights update function
func (w *Weight) Get() Getter {
	return w
}

//+-------------------------------------------------------------+
//|	Layers														|
//+-------------------------------------------------------------+
//
/*func (l *Layer) Set(Setter)  {
}

//
func (l *Layer) Get() Getter {
	return l
}*/

//+-------------------------------------------------------------+
//|	Neurons														|
//+-------------------------------------------------------------+
//
func (n *Neuron) Set(Setter) {
}

// Function for calculating the values of neurons in a layer
func (n *Neuron) Get() Getter {
	return n
}

//+-------------------------------------------------------------+
//|	Errors														|
//+-------------------------------------------------------------+
//
func (e *Error) Set(Setter) {
}

//
func (e *Error) Get() Getter {
	return e
}
