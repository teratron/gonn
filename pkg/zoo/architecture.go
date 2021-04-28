package zoo

import (
	"fmt"
	"log"
	"strings"

	"github.com/zigenzoog/gonn/pkg"
	"github.com/zigenzoog/gonn/pkg/utils"
	"github.com/zigenzoog/gonn/pkg/zoo/hopfield"
	"github.com/zigenzoog/gonn/pkg/zoo/perceptron"
)

const (
	Perceptron = perceptron.Name
	Hopfield   = hopfield.Name
)

// Get.
func Get(title string) pkg.NeuralNetwork {
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
			err = fmt.Errorf("neural network is %w", pkg.ErrNotRecognized)
		}
	}

	if err != nil {
		log.Println("get architecture: %w", err)
	}
	return nil
}
