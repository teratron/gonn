//
package nn

import (
	_ "math/rand"
	_ "time"
)

const (
	DefaultRate  rate      = .3     // Default rate
	MinLossLimit lossLimit = 10e-33 // The minimum value of the error limit at which training is forcibly terminated
	MaxIteration uint32    = 10e+05 // The maximum number of iterations after which training is forcibly terminated
	ModeMSE      uint8     = 0      // Mean Squared Error
	ModeRMSE     uint8     = 1      // Root Mean Squared Error
	ModeARCTAN   uint8     = 2      // Arctan
)

type (
	rate      float32
	bias      float32
	floatType float32
	lossLimit floatType
	inputSet  []floatType
)

//
type neuralNetwork struct {
	NeuralNetwork // Architecture/type of neural network (configuration)
	isInit    bool
	rate
	lossMode uint8
	lossLimit
	lossFunc func() lossLimit

	//
	upperRange floatType // Range, Bound, Limit, Scope
	lowerRange floatType

	//
	language string

	//
	neuron []neuron
	axon   []axon

	//Neuroner
}

//
type neuron struct {
	index          uint32
	activationMode uint8
	value          floatType
	error          floatType
	axon           []axon
	//Neuroner
}

//
type axon struct {
	index  uint32
	weight floatType
	//Synapse map[string]Neuroner // map["bias"]Neuroner, map["input"]Neuroner, map["output"]Neuroner
}

//+-------------------------------------------------------------+
//| Neural network                                              |
//+-------------------------------------------------------------+
func (n *neuralNetwork) Set(setter Setter) {
	setter.Set(n)
}

func (n *neuralNetwork) Get() Getter {
	return n
}

//+-------------------------------------------------------------+
//| Neuron bias                                                 |
//+-------------------------------------------------------------+
// Initializing bias
func (n *neuralNetwork) SetBias(bias bias) {
	bias.Set(n)
}

func (b bias) Set(setter Setter) {
	if n, ok := setter.(*neuralNetwork); ok {
		if bias, ok := b.Check().(bias); ok {
			n.NeuralNetwork.Set(bias)
		}
	}
}

// Getting bias
/*func (n *neuralNetwork) GetBias() bias {
	return n.Bias()
}*/

/*func (n *neuralNetwork) Bias() bias {
	fmt.Printf("+++ %T %v\n", n.NeuralNetwork.Get(), n.NeuralNetwork.Get())
	//return n.NeuralNetwork.Get().Bias()
	return n.NeuralNetwork.(*feedForward).bias
}*/

func Bias() {
}

/*func (b Bias) Get() Getter {
	return b
}*/

// Checking bias
func (b bias) Check() Checker {
	switch {
	case b < 0:
		return bias(0)
	case b > 1:
		return bias(1)
	default:
		return b
	}
}

//+-------------------------------------------------------------+
//| Learning rate                                               |
//+-------------------------------------------------------------+
// Checking learning rate
func (r rate) Check() Checker {
	switch {
	case r < 0 || r > 1:
		return DefaultRate
	default:
		return r
	}
}

//
func (l lossLimit) Set(setter Setter) {}

//func (n *neuron) Set() {}

//
func (n *neuralNetwork) Init() {
}

//
func (n *neuralNetwork) Train() {
}

//
func (n *neuralNetwork) Query() {
}

//
func (n *neuralNetwork) Verify() {
}

// The function fills all weights with random numbers from -0.5 to 0.5
/*func (n *neuralNetwork) setWeight() {
	rand.Seed(time.Now().UTC().UnixNano())
	randWeight := func() float64 {
		r := 0.
		for r == 0 {
			r = rand.Float64() - .5
		}
		return r
	}
	for _, a := range n.Axon {
		if b, ok := a.Synapse["bias"]; !ok || (ok && *b.(*Bias) > 0) {
			a.Weight = randWeight()
		}
	}
}*/

//
/*func (m *Matrix) getNeuron() {
	for _, n := range m.Neuron {
		for _, a := range n.Axon {
			n.Value += a.Weight * a.Synapse["input"].
		}

	}
}*/

/*func (m *Matrix) Get() {
	for i, v := range m.Neuron {

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
