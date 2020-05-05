package nn

import (
	_ "math/rand"
	_ "time"
)

const (
	DEFRATE float32 = .3      // Default rate
	MINLOSS float32 = .001    // The minimum value of the sum of the average square error at which the training is forcibly terminated
	MAXITER int     = 1000000 // The maximum number of iterations after which training is forcibly terminated
)

// Collection of neural network matrix parameters
type Matrix struct {
	Init    bool      // Matrix initialization flag

	Mode    uint8     // Activation function mode
	Bias    float32   // The neuron bias, 0 or 1
	Rate    float32   // Learning coefficient, from 0 to 1
	Limit   float32   // Minimum (sufficient) level of the average quadratic sum of the error during training

	Size    int       // Number of layers in the neural network (Input + Hidden + Output)
	Index   int       // Index of the output (last) layer of the neural network
	Layer   []Layer	  // Layer of the neural network
	Synapse	[]Synapse // The layer weights of connections

	Neuron  [][]Neuron  //
	Error   [][]Error	//
}

// Collection of neural layer parameters
type Layer struct {
	X		int      // Индекс слоя нейронов в матрице
	Size	int      // Number of neurons in the layer
}

// Collection of weight parameters
type Synapse struct {
	X		int		// Index of the weight layer in the matrix
	Size	[]int	// Number of neurons in the layer
}

type Neuron struct {
	X, Y	int					// X-index of the layer in the matrix, Y-index of the neuron in the layer
	Value	float32				// Neuron value
	N		*PrevNeuronLayer	//
	W		*PrevWeightLayer	//
}

type Error struct {
	X, Y	int					//
	Value	float32				// Error value
	E		*NextErrorLayer		//
	W		*NextWeightLayer	//
}

type Weight struct {

}

type (
	PrevNeuronLayer []float32
	PrevWeightLayer []float32
	NextErrorLayer  []float32
	NextWeightLayer []float32
)

func (s *Neuron) Get() float32 {
	return s.Value * 2
}

//+-------------------------------------------------------------+
//| Synapse                                                     |
//+-------------------------------------------------------------+
// The function fills all weights with random numbers from -0.5 to 0.5
/*func (s *Synapse) Set() {
	rand.Seed(time.Now().UTC().UnixNano())
	for i := 0; i < s.Index; i++ {
		n := s.Synapse[i].Size[0] - 1
		for j := 0; j < s.Synapse[i].Size[0]; j++ {
			for k := 0; k < s.Synapse[i].Size[1]; k++ {
				if j == n && s.Bias == 0 {
					s.Synapse[i].Weight[j][k] = 0
				} else {
					s.Synapse[i].Weight[j][k] = rand.Float32() - .5
				}
			}
		}
	}
}*/

// Weights update function
func (s *Synapse) Get() float32 {
	return 0
}

//+-------------------------------------------------------------+
//| Weight                                                      |
//+-------------------------------------------------------------+
//
/*func (w *Weight) Set() {
}

// Weights update function
func (w *Weight) Get() float32 {
	return 0
}*/

//+-------------------------------------------------------------+
//| Layer                                                       |
//+-------------------------------------------------------------+
//
/*func (l *Layer) Get() float32 {
	for i := 1; i < m.Size; i++ {
		n := i - 1
		for j := 0; j < m.Layer[i].Size; j++ {
			var sum float32 = 0
			for k, v := range m.Layer[n].Neuron {
				sum += v * m.Synapse[n].Weight[k][j]
			}
			m.Layer[i].Neuron[j] = GetActivation(sum, m.Mode)
		}
	}
	return 0
}*/

//
/*func (l *Layer) Set(Setter)  {
}*/

//+-------------------------------------------------------------+
//| Neuron                                                      |
//+-------------------------------------------------------------+
// Function for calculating the values of neurons in a layer
/*func (n *Neuron) Get() float32 {
	var sum float32 = 0
	x := n.X - 1
	for k, v := range m.Layer[x].Neuron {
		sum += v * m.Synapse[x].Weight[k][n.Y]
	}
	n.Value = n.Activation.Get(sum)
	return n.Value
}*/

//
/*func (n *Neuron) Set() {
}*/

//+-------------------------------------------------------------+
//| Error                                                       |
//+-------------------------------------------------------------+
//
/*func (e *Error) Set() {
}*/

//
func (e *Error) Get() float32 {
	return 0
}
