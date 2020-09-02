package nn

import (
	"log"

	"github.com/zigenzoog/gonn/pkg"
)

// Synapser
type Synapser interface {
	pkg.GetSetter
}

// axon
type axon struct {
	weight  floatType
	synapse map[string]Synapser
}

// getSynapseInput
func (a *axon) getSynapseInput() (input floatType) {
	switch s := a.synapse["input"].(type) {
	case floatType:
		input = s
	case biasType:
		if s {
			input = 1
		}
	case *neuron:
		input = s.value
	default:
		panic("error!!!") // !!!
	}
	return
}

// weight
type weight struct {
	isInitWeight bool
	array        Floater
}

// Weight
func Weight(weights ...Floater) pkg.Controller {
	if len(weights) > 0 {
		switch weights[0].(type) {
		case Floater:
			return &weight{array: weights[0]}
		default:
			if w, ok := weights[0].(pkg.Controller); ok {
				return w
			}
		}
	} else {
		return &weight{}
	}
	return nil
}

func (n *nn) Weight() Floater {
	return n.Architecture.(Parameter).Weight()
}

// Set
func (w *weight) Set(args ...pkg.Setter) {
	if len(args) > 0 {
		if n, ok := args[0].(NeuralNetwork); ok {
			n.Get().Set(w)
		}
	} else {
		pkg.Log("Empty set", true) // !!!
	}
}

// Get
func (w *weight) Get(args ...pkg.Getter) pkg.GetSetter {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			return a.Get().Get(w)
		}
	} else {
		return w
	}
	return nil
}

// Copy
func (w *weight) Copy(copier pkg.Copier) {
	if n, ok := copier.(*nn); ok {
		if a, ok := n.Architecture.(NeuralNetwork); ok {
			a.Copy(w)
		}
	}
}

// Paste
func (w *weight) Paste(paster pkg.Paster) (err error) {
	if n, ok := paster.(*nn); ok {
		if a, ok := n.Architecture.(NeuralNetwork); ok {
			err = a.Paste(w)
		}
	}
	return
}

// Read
func (w *weight) Read(reader pkg.Reader) {
	reader.Read(w)
}

// Write
func (w *weight) Write(writer ...pkg.Writer) {
	if len(writer) > 0 {
		for _, v := range writer {
			v.Write(w)
		}
	} else {
		log.Println("Empty write") // !!!
	}
}
