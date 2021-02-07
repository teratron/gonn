package nn

const hopfieldName = "hopfield"

// Declare conformity with NeuralNetwork interface
var _ NeuralNetwork = (*hopfield)(nil)

// hopfield
type hopfield struct {
	NeuralNetwork `json:"-" xml:"-"`
	//Parameter     `json:"-" xml:"-"`

	// Neural network architecture name
	Name string `json:"name" xml:"name"`

	// Energy
	Energy float64 `json:"energy" xml:"energy"`

	// Weights values
	Weights float2Type `json:"weights" xml:"weights"`

	// Neuron
	neuron []*hopfieldNeuron

	// Settings
	lenInput int
	isInit   bool
	jsonName string
}

// hopfieldNeuron
type hopfieldNeuron struct {
	value float64
}

// Hopfield return
func Hopfield() *hopfield {
	return &hopfield{
		Name: hopfieldName,
	}
}
