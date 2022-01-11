package arch

import (
	"fmt"
	"log"
	"strings"

	"github.com/teratron/gonn/pkg"
	"github.com/teratron/gonn/pkg/arch/hopfield"
	"github.com/teratron/gonn/pkg/arch/perceptron"
	"github.com/teratron/gonn/pkg/utils"
)

const (
	PERCEPTRON = perceptron.NAME
	HOPFIELD   = hopfield.NAME
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
					//log.Println("+++++++++++++++++ ", d)
					n.Init(d)
					return n
				}
			}
		}
	} else {
		switch strings.ToLower(title) {
		case PERCEPTRON:
			return perceptron.New()
		case HOPFIELD:
			return hopfield.New()
		default:
			err = fmt.Errorf("neural network is %w", pkg.ErrNotRecognized)
		}
	}

	if err != nil {
		log.Println("get architecture:", err)
	}
	return nil
}
