package nn

import (
	"fmt"
	"log"

	"github.com/teratron/gonn/pkg/activation"

	"github.com/teratron/gonn/pkg"
	"github.com/teratron/gonn/pkg/utils"
)

// Init initialize.
func (nn *NN[T]) Init(data ...interface{}) {
	var err error
	if len(data) > 0 {
		switch value := data[0].(type) {
		case utils.Filer:
			if _, ok := value.(utils.FileError); !ok {
				if len(nn.Weights) > 0 {
					nn.initFromWeight()
				}
				nn.config = value
			}
		case int:
			if len(data) == 2 {
				if v, ok := data[1].(int); ok {
					nn.initFromNew(value, v)
				}
			}
		default:
			err = fmt.Errorf("%T %w: %v", value, pkg.ErrMissingType, value)
		}
		if err == nil {
			nn.initCompletion()
		}
	} else {
		err = pkg.ErrNoArgs
	}

	if err != nil {
		log.Printf("perceptron.NN.Init: %v\n", err)
	}
}

// initFromNew initialize.
func (nn *NN[T]) initFromNew(lenInput, lenTarget int) {
	nn.lenInput = lenInput
	nn.lenOutput = lenTarget

	if nn.HiddenLayers[0] > 0 {
		nn.lastLayerIndex = len(nn.HiddenLayers)
	} else {
		nn.lastLayerIndex = 0
	}

	var layer []uint
	if nn.lastLayerIndex > 0 {
		layer = append(nn.HiddenLayers, uint(nn.lenOutput))
	} else {
		layer = []uint{uint(nn.lenOutput)}
	}
	lenLayer := len(layer)

	bias := 0
	if nn.Bias {
		bias = 1
	}
	biasInput := nn.lenInput + bias
	var biasLayer int

	nn.Weights = make([][][]T, lenLayer)
	nn.weights = make([][][]T, lenLayer)
	nn.neurons = make([][]*neuron[T], lenLayer)
	for i, v := range layer {
		nn.Weights[i] = make([][]T, v)
		nn.weights[i] = make([][]T, v)
		nn.neurons[i] = make([]*neuron[T], v)
		if i > 0 {
			biasLayer = int(layer[i-1]) + bias
		}

		for j := 0; j < int(v); j++ {
			if i > 0 {
				nn.Weights[i][j] = make([]T, biasLayer)
				nn.weights[i][j] = make([]T, biasLayer)
			} else {
				nn.Weights[i][j] = make([]T, biasInput)
				nn.weights[i][j] = make([]T, biasInput)
			}
			for k := range nn.weights[i][j] {
				if nn.ActivationMode == activation.LINEAR {
					nn.Weights[i][j][k] = .5
				} else {
					nn.Weights[i][j][k] = utils.GetRandFloat[T]()
				}
			}
			nn.neurons[i][j] = &neuron[T]{}
		}
	}
}

// initFromWeight.
func (nn *NN[T]) initFromWeight() {
	length := len(nn.Weights)
	nn.lastLayerIndex = length - 1
	nn.lenOutput = len(nn.Weights[nn.lastLayerIndex])
	nn.lenInput = len(nn.Weights[0][0])

	if length > 1 && len(nn.Weights[0])+1 == len(nn.Weights[1][0]) {
		nn.Bias = true
		nn.lenInput -= 1
	}

	if nn.lastLayerIndex > 0 {
		nn.HiddenLayers = make([]uint, nn.lastLayerIndex)
		for i := range nn.HiddenLayers {
			nn.HiddenLayers[i] = uint(len(nn.Weights[i]))
		}
	} else {
		nn.HiddenLayers = []uint{0}
	}

	nn.weights = make(nn.Float3Type, length)
	nn.neurons = make([][]*neuron, length)
	for i, v := range nn.Weights {
		length = len(v)
		nn.weights[i] = make(nn.Float2Type, length)
		nn.neurons[i] = make([]*neuron, length)
		for j, w := range v {
			nn.weights[i][j] = make(nn.Float1Type, len(w))
			nn.neurons[i][j] = &neuron[T]{}
		}
	}
}

// initCompletion.
func (nn *NN[T]) initCompletion() {
	nn.prevLayerIndex = nn.lastLayerIndex - 1
	nn.input = make([]T, nn.lenInput)
	nn.target = make([]T, nn.lenOutput)
	nn.output = make([]T, nn.lenOutput)
	nn.isInit = true
}
