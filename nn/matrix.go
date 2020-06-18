//
package nn

import (
	"math/rand"
	"time"
)

//
func (n *nn) Train() {
}

//
func (n *nn) Query() {
}

//
func (n *nn) Verify() {
}

//
func forwardPropagation() {
}

//
func backwardPropagation() {
}

// The function fills all weights with random numbers from -0.5 to 0.5
func (n *nn) setWeight() {
	rand.Seed(time.Now().UTC().UnixNano())
	randWeight := func() Float {
		r := 0.
		for r == 0 {
			r = rand.Float64() - .5
		}
		return Float(r)
	}
	for _, a := range n.axon {
		if b, ok := a.synapse["bias"]; !ok || (ok && b.(Bias) == true) {
			a.weight = randWeight()
		}
	}
}

//
/*func (m *NN) getNeuron() {
	for _, n := range m.Neuron {
		for _, a := range n.Axon {
			n.Value += a.Weight * a.Synapse["input"].
		}

	}
}*/

// Collection of neural network matrix parameters
/*type Matrix struct {
	isInit	bool      // Matrix initialization flag

	Mode	uint8     // Activation function mode
	Bias    float64   // The neuron bias, 0 or 1
	Rate    float64   // Learning coefficient, from 0 to 1
	Limit   float64   // Minimum (sufficient) level of the average quadratic sum of the error during training

	//Size    int       // Number of layers in the neural network (Input + Hidden + Output)
	//Index   int       // Index of the output (last) layer of the neural network
	//Layer   []Layer	  // Layer of the neural network
	//Layer
	Input
	Output
	Hidden
	Synapse	[]Synapse // The layer weights of relationships

	//Neuron  [][]Neuron  //
	//Error   [][]Error	//
}

type Setting struct {
	Mode	uint8     // Activation function mode
	Bias    float64   // The neuron bias, 0 or 1
	Rate    float64   // Learning coefficient, from 0 to 1
	Limit   float64   // Minimum (sufficient) level of the average quadratic sum of the error during training
}

// Collection of neural layer parameters
type Input struct {
	Size	int
	Neuron	[]float64
}

type Output struct {
	Size	int
	Data	[]float64
	Neuron	[]float64
	Error	[]float64
}

type Hidden struct {
	Size	int
	Layer
}

type Layer []struct {
	X		int      // Индекс скрытого слоя
	Size	int      // Number of neurons in the layer
	Neuron	[]Neuron
	Error	[]Error
}

type Neuron struct {
	X, Y	int					// X-index of the layer in the matrix, Y-index of the neuron in the layer
	Value	float64				// Neuron value
	N		*PrevNeuronLayer	//
	W		*PrevWeightLayer	//
}

type Error struct {
	X, Y	int					//
	Value	float64				// Error value
	E		*NextErrorLayer		//
	W		*NextWeightLayer	//
}

// Collection of weight parameters
type Synapse struct {
	X		int		// Index of the weight layer in the matrix
	Size	[]int	// Number of neurons in the layer
}

type Weight struct {

}

type (
	PrevNeuronLayer []float64
	PrevWeightLayer []float64
	NextErrorLayer  []float64
	NextWeightLayer []float64
)*/

/*func (n *Neuron) Get() float64 {
	return n.Value * 2
}*/

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
/*func (s *Synapse) Get() float64 {
	return 0
}*/

//+-------------------------------------------------------------+
//| Weight                                                      |
//+-------------------------------------------------------------+
//
/*func (w *Weight) Set() {
}

// Weights update function
func (w *Weight) Get() float64 {
	return 0
}*/

//+-------------------------------------------------------------+
//| Layer                                                       |
//+-------------------------------------------------------------+
//
/*func (l *Layer) Get() float64 {
	for i := 1; i < m.Size; i++ {
		n := i - 1
		for j := 0; j < m.Layer[i].Size; j++ {
			var sum float64 = 0
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
/*func (n *Neuron) Get() float64 {
	var sum float64 = 0
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
/*func (e *Error) Get() float64 {
	return 0
}*/
