package nn

import (
	"fmt"

	"github.com/teratron/gonn/pkg"
)

// Synapser
type Synapser interface {
	pkg.GetSetter
}

// axon
type axon struct {
	weight  FloatType
	synapse map[string]Synapser
}

// getSynapseInput
func (a *axon) getSynapseInput() (input FloatType) {
	switch s := a.synapse["input"].(type) {
	case FloatType:
		input = s
	case biasBool:
		if s {
			input = 1
		}
	case *neuron:
		input = s.value
	default:
		pkg.LogError(fmt.Errorf("%w for method getSynapseInput: %v", pkg.ErrMissingType, s))
	}
	return
}

// weight
type weight struct {
	isInitWeight bool
	buffer       Floater
}

// Weight
func Weight(args ...Floater) pkg.Controller {
	if len(args) > 0 {
		switch v := args[0].(type) {
		case *Float3Type, *Float2Type, *Float1Type:
			return &weight{
				isInitWeight: true,
				buffer:       v,
			}
		default:
			if w, ok := args[0].(pkg.Controller); ok {
				return w
			}
		}
	} else {
		return &weight{}
	}
	return nil
}

// Weight
/*func (n *nn) Weight() Floater {
	return n.Architecture.(Parameter).Weight()
}*/

// Set
func (w *weight) Set(args ...pkg.Setter) {
	if len(args) > 0 {
		if n, ok := args[0].(NeuralNetwork); ok {
			n.Get().Set(w)
		}
	} else {
		pkg.LogError(fmt.Errorf("%w set for weight", pkg.ErrEmpty))
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
	/*if n, ok := copier.(*nn); ok {
		if a, ok := n.Architecture.(NeuralNetwork); ok {
			a.Copy(w)
		}
	}*/
}

// Paste
func (w *weight) Paste(paster pkg.Paster) {
	/*if n, ok := paster.(*nn); ok {
		if a, ok := n.Architecture.(NeuralNetwork); ok {
			a.Paste(w)
		}
	}*/
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
		pkg.LogError(fmt.Errorf("%w write for weight", pkg.ErrEmpty))
	}
}
