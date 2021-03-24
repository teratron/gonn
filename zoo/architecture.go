package zoo

import (
	"fmt"
	"log"
	"strings"

	"github.com/teratron/gonn"
	"github.com/teratron/gonn/utils"
	"github.com/teratron/gonn/zoo/hopfield"
	"github.com/teratron/gonn/zoo/perceptron"
)

const (
	Perceptron = perceptron.Name
	Hopfield   = hopfield.Name
)

// Get
func Get(name string) gonn.NeuralNetwork {
	var err error
	d := utils.GetFileType(name)
	if _, ok := d.(error); !ok {
		switch v := d.GetValue("name").(type) {
		case string:
			if n := Get(v); n != nil {
				if err = d.Decode(n); err == nil {
					if err = n.Init(d); err == nil {
						return n
					}
				}
			}
		case error:
			err = v
		}
	} else {
		switch strings.ToLower(name) {
		case Perceptron:
			return perceptron.Perceptron()
		case Hopfield:
			return hopfield.Hopfield()
		default:
			err = fmt.Errorf("neural network is %w", gonn.ErrNotRecognized)
		}
	}

	if err != nil {
		log.Println(fmt.Errorf("get architecture: %w", err))
	}
	return nil
}
