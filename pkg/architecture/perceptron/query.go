package perceptron

import (
	"fmt"
	"log"

	"github.com/teratron/gonn/pkg"
)

// Query querying dataset.
func (nn *NN) Query(input []float64) []float64 {
	var err error
	if len(input) > 0 {
		nn.mutex.Lock()
		defer nn.mutex.Unlock()

		if !nn.isInit {
			err = pkg.ErrInit
			goto ERROR
		} else if nn.lenInput != len(input) {
			err = fmt.Errorf("invalid number of elements in the input data")
			goto ERROR
		}
		//fmt.Println(&nn.weights[0][0][0], &nn.Weights[0][0][0])
		if nn.Weights[0][0][0] != 0 {
			//nn.weights = nn.Weights
			//_ = copy(nn.weights, nn.Weights)
			nn.weights.Copy(nn.Weights)
		}
		//_ = copy(nn.weights[0][0], nn.Weights[0][0])
		//nn.weights[0][0] = nn.Weights[0][0]
		//nn.weights = append(nn.weights, nn.Weights...)
		//nn.weights.Copy(nn.Weights)
		//fmt.Println(nn.weights[0][0][0], nn.Weights[0][0][0])
		//nn.Weights[0][0][0] = 42
		//fmt.Println(&nn.weights[0][1][0], &nn.Weights[0][1][0])

		nn.input = pkg.ToFloat1Type(input)
		nn.calcNeurons()
		return nn.output
	} else {
		err = pkg.ErrNoInput
	}

ERROR:
	log.Printf("perceptron.NN.Query: %v\n", err)
	return nil
}
