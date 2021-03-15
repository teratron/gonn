package nn

import (
	"fmt"
	"log"

	"github.com/teratron/gonn"
	arch "github.com/teratron/gonn/architecture"
)

// NeuralNetwork
type NeuralNetwork interface {
	gonn.NeuralNetwork
	arch.Parameter
}

// Reader
type Reader interface {
	gonn.Reader
}

// Writer
type Writer interface {
	gonn.Writer
}

// Floater
type Floater interface {
	gonn.Floater
}

/*type NN struct {
	gonn.NeuralNetwork
	arch.Parameter
}*/

// New returns a new neural network instance.
func New(reader ...Reader) NeuralNetwork {
	if len(reader) > 0 {
		var err error
		switch r := reader[0].(type) {
		case NeuralNetwork:
			return r
		case gonn.Filer:
			switch v := r.GetValue("name").(type) {
			case string:
				n := architecture(v)
				//n := &NN{NeuralNetwork: architecture(v)}
				if err = n.Read(r); err == nil {
					return n
				}
			case error:
				err = v
			}
		default:
			err = fmt.Errorf("%T %w", r, gonn.ErrMissingType)
		}

		if err != nil {
			err = fmt.Errorf("new: %w", err)
			log.Println(err)
		}
		return nil
	}
	return Perceptron()
}
