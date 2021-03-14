package nn

import (
	"fmt"
	"log"

	"github.com/teratron/gonn"
)

// New returns a new neural network instance.
func New(reader ...gonn.Reader) gonn.NeuralNetwork {
	if len(reader) > 0 {
		var err error
		switch r := reader[0].(type) {
		case gonn.NeuralNetwork:
			return r
		case gonn.Filer:
			switch v := r.GetValue("name").(type) {
			case string:
				n := architecture(v)
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
