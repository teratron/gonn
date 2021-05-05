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
			if len(nn.Weights) > 0 {
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
	nn.lastLayerIndex = len(nn.Hidden)
	if nn.lastLayerIndex > 0 && nn.Hidden[0] == 0 {
		nn.lastLayerIndex = 0
	}

	var layer []int
	if nn.lastLayerIndex > 0 {
		layer = append(nn.Hidden, nn.lenOutput)
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

	nn.Weights = make(pkg.Float3Type, lenLayer)
	nn.neuron = make([][]*neuron, lenLayer)
	for i, v := range layer {
		nn.Weights[i] = make(pkg.Float2Type, v)
		nn.neuron[i] = make([]*neuron, v)
		if i > 0 {
			biasLayer = layer[i-1] + bias
		}

		for j := 0; j < v; j++ {
			if i > 0 {
				nn.Weights[i][j] = make(pkg.Float1Type, biasLayer)
			} else {
				nn.Weights[i][j] = make(pkg.Float1Type, biasInput)
			}
			for k := range nn.Weights[i][j] {
				if nn.Activation == params.LINEAR {
					nn.Weights[i][j][k] = .5
				} else {
					nn.Weights[i][j][k] = params.GetRandFloat()
				}
			}
			nn.neuron[i][j] = &neuron{}
		}
	}
	//nn.input = make([]float64, nn.lenInput)
	//nn.output = make([]float64, nn.lenOutput)
	nn.isInit = true
}

// initFromWeight.
func (nn *NN) initFromWeight() {
	length := len(nn.Weights)

	if !nn.Bias && length > 1 && len(nn.Weights[0])+1 == len(nn.Weights[1][0]) {
		nn.Bias = true
	}

	nn.lastLayerIndex = length - 1
	nn.lenOutput = len(nn.Weights[nn.lastLayerIndex])
	nn.lenInput = len(nn.Weights[0][0])
	if nn.Bias {
		nn.lenInput -= 1
	}

	if nn.lastLayerIndex > 0 {
		nn.Hidden = make([]int, nn.lastLayerIndex)
		for i := range nn.Hidden {
			nn.Hidden[i] = len(nn.Weights[i])
		}
	} else {
		nn.Hidden = []int{0}
	}

	nn.neuron = make([][]*neuron, length)
	for i, v := range nn.Weights {
		nn.neuron[i] = make([]*neuron, len(v))
		for j := range v {
			nn.neuron[i][j] = &neuron{}
		}
	}
	//nn.input = make([]float64, nn.lenInput)
	//nn.output = make([]float64, nn.lenOutput)
	nn.isInit = true
}
