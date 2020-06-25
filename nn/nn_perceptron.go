// Perceptron Neural Network
package nn

import (
	"fmt"
	"log"
)

type Perceptron interface {
	Perceptron() NeuralNetwork
}

type perceptron struct {
	bias			biasType			//
	rate			rateType			//
	modeActivation	modeActivationType	//

	modeLoss  modeLossType  //
	levelLoss levelLossType // Minimum (sufficient) level of the average of the error during training

	hiddenLayer		HiddenType			// Array of the number of neurons in each hidden layer

	upperRange		floatType			// Range, Bound, Limit, Scope
	lowerRange		floatType

	lastNeuron		uint32				// Index of the last neuron of the neural network
	lastAxon		uint32				// Index of the last axon of the neural network

	neuron struct {
		error		floatType
	}

	Architecture // чтобы не создавать методы для всех типов нн
	Processor
}

// Initializing Perceptron Neural Network
func (n *nn) Perceptron() NeuralNetwork {
	n.architecture = &perceptron{
		bias:			false,
		rate:			DefaultRate,
		modeActivation:	ModeSIGMOID,
		modeLoss:		ModeMSE,
		levelLoss:		.0001,
		hiddenLayer:	HiddenType{},
		upperRange:		1,
		lowerRange:		0,
		lastNeuron:		0,
		lastAxon:		0,
	}
	//n.neuron[0].architecture =
	return n
}

// Preset
func (p *perceptron) Preset(name string) {
	switch name {
	default:
		fallthrough
	case "default":
		p.Set(
			Bias(false),
			Rate(DefaultRate),
			Activation(ModeSIGMOID),
			Loss(ModeMSE),
			LevelLoss(.0001),
			HiddenLayer())
	}
}

// Setter
func (p *perceptron) Set(set ...Setter) {
	switch v := set[0].(type) {
	case biasType:
		p.bias = v
	case rateType:
		p.rate = v
	case modeActivationType:
		p.modeActivation = v
	case modeLossType:
		p.modeLoss = v
	case levelLossType:
		p.levelLoss = v
	case HiddenType:
		p.hiddenLayer = v
		//fmt.Printf("%T %v\n", set[1].(*nn).neuron, set[1].(*nn).neuron)
		if len(p.hiddenLayer) > 0 {
			var n hiddenType = 0
			for _, h := range p.hiddenLayer {
				n += h /** hiddenType(i + 1)*/

			}
			set[1].(*nn).neuron = make([]neuron, n)
			fmt.Println(len(set[1].(*nn).neuron), cap(set[1].(*nn).neuron))
			set[1].(*nn).neuron = append(set[1].(*nn).neuron, neuron{})
			fmt.Println(len(set[1].(*nn).neuron), cap(set[1].(*nn).neuron))
		}
		//neurons := set[1].(*nn).neuron
		//set[1].(cH) <- true
		//fmt.Printf("3 go %T\n", set[1])
		//fmt.Println(*p)
		//p.initHidden()
		/*fmt.Println("***", func(d uint32) (h hiddenType) {
			h = 1
			for _, value := range p.hiddenLayer {
				h *= value
			}
			return h + hiddenType(d)
		}(4))*/
	default:
		Log("This type of variable is missing for Perceptron Neural Network", false)
		log.Printf("\tset: %T %v\n", v, v)
	}
}

// Getter
func (p *perceptron) Get(set ...Setter) Getter {
	switch set[0].(type) {
	case biasType:
		return p.bias
	case rateType:
		return p.rate
	case modeActivationType:
		return p.modeActivation
	case modeLossType:
		return p.modeLoss
	case levelLossType:
		return p.levelLoss
	case HiddenType:
		return p.hiddenLayer
	default:
		Log("This type of variable is missing for Perceptron Neural Network", false)
		log.Printf("\tget: %T %v\n", set[0], set[0])
		return nil
	}
}

// Train
/*func (p *perceptron) Train(input, target []float64) (loss float64, count int) {
	return
}

// Query
func (p *perceptron) Query(input []float64) []float64 {
	panic("implement me")
}*/

/*func (p *perceptron) initHidden() {
}*/