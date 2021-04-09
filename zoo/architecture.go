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
func Get(title string) gonn.NeuralNetwork {
	var err error
	d := utils.GetFileType(title)
	if _, ok := d.(error); !ok {
		switch v := d.GetValue("name").(type) {
		case error:
			err = v
		case string:
			if n := Get(v); n != nil {
				if err = d.Decode(n); err == nil {
					n.Init(d)
					return n
				}
			}
		}
	} else {
		switch strings.ToLower(title) {
		case Perceptron:
			return perceptron.New()
		case Hopfield:
			return hopfield.New()
		default:
			err = fmt.Errorf("neural network is %w", gonn.ErrNotRecognized)
		}
	}

	if err != nil {
		log.Println(fmt.Errorf("get architecture: %w", err))
	}
	return nil
}
