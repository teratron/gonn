// Hopfield Neural Network - under construction
package nn

import (
	"log"

	"github.com/zigenzoog/gonn/pkg"
)

var _ NeuralNetwork = (*hopfield)(nil)

/*type HopfieldNN interface {
	Energy() float32
}*/

type hopfield struct {
	Architecture						`json:"-" xml:"-"`
	Parameter							`json:"-" xml:"-"`
	Constructor							`json:"-" xml:"-"`

	Configuration struct{
		Energy			floatType
		Weights			[][]floatType	`json:"weights" xml:"weights"`
	}									`json:"hopfield" xml:"hopfield"`

	// Matrix
	neuron []*Neuron
	axon   [][]*Axon
}

func Hopfield() *hopfield {
	return &hopfield{}
}

// Returns a new Hopfield neural network instance with the default parameters
func (n *NN) hopfield() NeuralNetwork {
	n.Architecture = &hopfield{
		Architecture: n,
	}
	return n
}

func (h *hopfield) Energy() float32 {
	return float32(h.Configuration.Energy)
}

// Setter
func (h *hopfield) Set(args ...pkg.Setter) {
	if len(args) > 0 {
		switch v := args[0].(type) {
		default:
			pkg.Log("This type of variable is missing for Hopfield Neural Network", true)
			log.Printf("\tset: %T %v\n", v, v) // !!!
		}
	} else {
		pkg.Log("Empty Set()", true) // !!!
	}
}

// Getter
func (h *hopfield) Get(args ...pkg.Getter) pkg.GetterSetter {
	if len(args) > 0 {
		switch args[0].(type) {
		default:
			pkg.Log("This type of variable is missing for Hopfield Neural Network", true)
			log.Printf("\tget: %T %v\n", args[0], args[0]) // !!!
			return nil
		}
	} else {
		return h
	}
}

// Initialization
func (h *hopfield) init(length int, args ...interface{}) bool {
	return true
}

// Train
/*func (h *hopfield) Train(input, target []float64) (loss float64, count int) {
	return
}

// Query
func (h *hopfield) Query(input []float64) []float64 {
	panic("implement me")
}*/

// Read
func (h *hopfield) Read(reader pkg.Reader) {
	/*switch r := reader.(type) {
	case jsonType:
		h.readJSON(string(r))
	case xml:
		h.readXML(v)
	case xml:
		h.readCSV(v)
	case db:
		h.readDB(v)
	default:
		pkg.Log("This type is missing for write", true) // !!!
		log.Printf("\tWrite: %T %v\n", r, r) // !!!
	}*/
}

// Write
func (h *hopfield) Write(writer ...pkg.Writer) {
	/*for _, w := range writer {
		switch v := w.(type) {
		case *report:
			h.writeReport(v)
		case jsonType:
			h.writeJSON(string(v))
		case xml:
			h.writeXML(v)
		case xml:
			h.writeCSV(v)
		case db:
			h.writeDB(v)
		default:
			pkg.Log("This type is missing for write", true) // !!!
			log.Printf("\tWrite: %T %v\n", w, w) // !!!
		}
	}*/
}

// readJSON
func (h *hopfield) readJSON(value interface{}) {
	panic("implement me")
}

// writeJSON
func (h *hopfield) writeJSON(filename string) {
	panic("implement me")
}
