package perceptron

import (
	"fmt"
	"log"

	"github.com/teratron/gonn/pkg"
	"github.com/teratron/gonn/pkg/params"
	"github.com/teratron/gonn/pkg/utils"
)

// Init.
func (nn *NN) Init(data ...interface{}) {
	var err error
	if len(data) > 0 {
		switch value := data[0].(type) {
		case utils.Filer:
			if len(nn.Weight) > 0 {
				nn.initFromWeight()
			}
			nn.config = value
		case int:
			if len(data) == 2 {
				if v, ok := data[1].(int); ok {
					nn.initFromNew(value, v)
				}
			}
		default:
			err = fmt.Errorf("%T %w: %v", value, pkg.ErrMissingType, value)
		}
	} else {
		err = pkg.ErrNoArgs
	}

	if err != nil {
		log.Printf("init: %v\n", err)
	}
}

// initFromNew initialize.
func (nn *NN) initFromNew(lenInput, lenTarget int) {
	nn.lenInput = lenInput
	nn.lenOutput = lenTarget
	nn.lastLayerIndex = len(nn.HiddenLayer)
	if nn.lastLayerIndex > 0 && nn.HiddenLayer[0] == 0 {
		nn.lastLayerIndex = 0
	}

	var layer []int
	if nn.lastLayerIndex > 0 {
		layer = append(nn.HiddenLayer, nn.lenOutput)
	} else {
		layer = []int{nn.lenOutput}
	}
	lenLayer := len(layer)

	bias := 0
	if nn.Bias {
		bias = 1
	}
	biasInput := nn.lenInput + bias
	var biasLayer int

	nn.Weight = make(pkg.Float3Type, lenLayer)
	nn.weight = make(pkg.Float3Type, lenLayer)
	nn.neuron = make([][]*neuron, lenLayer)
	for i, v := range layer {
		nn.Weight[i] = make(pkg.Float2Type, v)
		nn.weight[i] = make(pkg.Float2Type, v)
		nn.neuron[i] = make([]*neuron, v)
		if i > 0 {
			biasLayer = layer[i-1] + bias
		}

		for j := 0; j < v; j++ {
			if i > 0 {
				nn.Weight[i][j] = make(pkg.Float1Type, biasLayer)
				nn.weight[i][j] = make(pkg.Float1Type, biasLayer)
			} else {
				nn.Weight[i][j] = make(pkg.Float1Type, biasInput)
				nn.weight[i][j] = make(pkg.Float1Type, biasInput)
			}
			for k := range nn.weight[i][j] {
				if nn.ActivationMode == params.LINEAR {
					nn.Weight[i][j][k] = .5
					//nn.weight[i][j][k] = .5
				} else {
					nn.Weight[i][j][k] = params.GetRandFloat() //TODO:
					//nn.weight[i][j][k] = params.GetRandFloat()
				}
			}
			nn.neuron[i][j] = &neuron{}
		}
	}

	nn.initCompletion()
}

// initFromWeight.
func (nn *NN) initFromWeight() {
	length := len(nn.Weight)

	if !nn.Bias && length > 1 && len(nn.Weight[0])+1 == len(nn.Weight[1][0]) {
		nn.Bias = true
	}

	nn.lastLayerIndex = length - 1
	nn.lenOutput = len(nn.Weight[nn.lastLayerIndex])
	nn.lenInput = len(nn.Weight[0][0])
	if nn.Bias {
		nn.lenInput -= 1
	}

	if nn.lastLayerIndex > 0 {
		nn.HiddenLayer = make([]int, nn.lastLayerIndex)
		for i := range nn.HiddenLayer {
			nn.HiddenLayer[i] = len(nn.Weight[i])
		}
	} else {
		nn.HiddenLayer = []int{0}
	}

	nn.weight = make(pkg.Float3Type, length)
	nn.neuron = make([][]*neuron, length)
	for i, v := range nn.Weight {
		length = len(v)
		nn.weight[i] = make(pkg.Float2Type, length)
		nn.neuron[i] = make([]*neuron, length)
		for j, w := range v {
			nn.weight[i][j] = make(pkg.Float1Type, len(w))
			nn.neuron[i][j] = &neuron{}
		}
	}
	//_ = copy(nn.weight, nn.Weight)

	nn.initCompletion()
}

// initCompletion.
func (nn *NN) initCompletion() {
	nn.input = make([]float64, nn.lenInput)
	nn.output = make([]float64, nn.lenOutput)
	nn.isInit = true
}
