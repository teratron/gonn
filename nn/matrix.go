//
package nn

import (
	"fmt"
	"log"
	_ "math/rand"
	_ "time"
)

//+-------------------------------------------------------------+
//| Neural network                                              |
//+-------------------------------------------------------------+
func (n *NN) Set(args ...Setter) {
	if len(args) == 0 {
		log.Printf("Empty set %T\n", args)
	} else {
		for _, v := range args {
			if s, ok := v.(Setter); ok {
				//fmt.Printf("--- %T %v\n", s, s)
				s.Set(n)
			}
		}
	}
}

func (n *NN) Get(args ...Getter) Getter {
	if len(args) == 0 {
		log.Printf("%T %v\n", args, args)
		return n
	} else {
		fmt.Printf("--- %T %v\n", args[0], args[0])
		/*for _, v := range args {
			if s, ok := v.(Setter); ok {
				//fmt.Printf("--- %T %v\n", s, s)
				//s.Set(n)
			}
		}*/
	}
	//panic("implement me")
	return nil
}

//+-------------------------------------------------------------+
//| Neuron bias                                                 |
//+-------------------------------------------------------------+
// Initializing bias
func (b Bias) Set(args ...Setter) {
	if n, ok := args[0].(*NN); ok {
		if bias, ok := b.Check().(Bias); ok {
			n.architecture.Set(bias)
		}
	}
}

func (n *NN) SetBias(bias Bias) {
	bias.Set(n)
}

// Getting bias
func (n *NN) Bias() (bias Bias) {
	if v, ok := n.architecture.(NeuralNetwork); ok {
		bias = v.Bias()
	}
	return
}

func (n *NN) GetBias() Bias {
	return n.Bias()
}

func (b Bias) Get(args ...Getter) Getter {
	//panic("implement me")
	fmt.Printf("--- %T %v\n", args, args)
	return args[0]
}

// Checking bias
func (b Bias) Check() Checker {
	switch {
	case b < 0:
		return Bias(0)
	case b > 1:
		return Bias(1)
	default:
		return b
	}
}

//+-------------------------------------------------------------+
//| Learning rate                                               |
//+-------------------------------------------------------------+
// Initializing learning rate
func (r Rate) Set(args ...Setter) {
	if n, ok := args[0].(*NN); ok {
		if rate, ok := r.Check().(Rate); ok {
			n.rate = rate
		}
	}
}

func (n *NN) SetRate(rate Rate) {
	rate.Set(n)
}

// Getting learning rate
func (n *NN) Rate() Rate {
	return n.rate
}

func (n *NN) GetRate() Rate {
	return n.Rate()
}

// Checking learning rate
func (r Rate) Check() Checker {
	switch {
	case r < 0 || r > 1:
		return DefaultRate
	default:
		return r
	}
}

//
//func (l Loss) Set(setter Setter) {}

//func (n *neuron) Set() {}

//
func (n *NN) Init() {
}

//
func (n *NN) Train() {
}

//
func (n *NN) Query() {
}

//
func (n *NN) Verify() {
}

//
func forwardPropagation() {
}

//
func backwardPropagation() {
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
