package nn

import "github.com/zigenzoog/gonn/pkg"

type weightType floatType

//
type axon struct {
	weight  floatType             //
	synapse map[string]pkg.Getter //
}

func Weight() pkg.GetterSetter {
	return weightType(0)
}

// Set
func (a axon) Set(...pkg.Setter) {
	panic("implement me")
}

func (w weightType) Set(...pkg.Setter) {
	panic("implement me")
}

// Get
func (a axon) Get(...pkg.Getter) pkg.GetterSetter {
	panic("implement me")
}

func (w weightType) Get(...pkg.Getter) pkg.GetterSetter {
	panic("implement me")
}

func getSynapseInput(axon *axon) (input floatType) {
	switch s := axon.synapse["input"].(type) {
	case floatType:
		input = s
	case biasType:
		if s { input = 1 }
	case *neuron:
		input = s.value
	default:
		panic("error!!!") // !!!
	}
	return
}

/*func Axon() GetterSetter {
	return &axon{}
}

// Set
func (a *axon) Set(args ...Setter) {
	if len(args) > 0 {
		if n, ok := args[0].(NeuralNetwork); ok {
			n.Get().Set(a)
		}
	} else {
		Log("Empty Set()", true) // !!!
	}
}

// Get
func (a *axon) Get(args ...Getter) GetterSetter {
	if len(args) > 0 {
		if n, ok := args[0].(NeuralNetwork); ok {
			return n.Get().Get(a)
		}
	} else {
		return a
	}
	return nil
}*/