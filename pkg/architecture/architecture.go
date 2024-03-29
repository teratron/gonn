package architecture

import (
	"fmt"
	"log"
	"strings"

	"github.com/teratron/gonn/pkg"
	"github.com/teratron/gonn/pkg/architecture/perceptron"
	"github.com/teratron/gonn/pkg/utils"
)

const Perceptron = perceptron.Name

// Get.
func Get(reader string) pkg.NeuralNetwork {
	var err error
	r := utils.ReadFile(reader)
	if _, ok := r.(*utils.FileError); ok {
		switch strings.ToLower(reader) {
		case Perceptron:
			return perceptron.New()
		default:
			err = fmt.Errorf("neural network is %w", pkg.ErrNotRecognized)
		}
	} else {
		switch v := r.GetValue("name").(type) {
		case error:
			err = v
		case string:
			if n := Get(v); n != nil {
				if err = r.Decode(n); err == nil {
					r.ClearData()
					n.Init(r)
					return n
				}
			}
		}
	}

	if err != nil {
		log.Printf("architecture.Get: %v", err)
	}
	return nil
}
