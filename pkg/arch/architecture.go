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
	Perceptron = perceptron.Name
	Hopfield   = hopfield.Name
)

// Get.
func Get(title string) pkg.NeuralNetwork {
	var err error
	d := utils.GetFileType(title)
	if u, ok := d.(utils.Filer); ok {
		fmt.Printf("%T, %v\n", d, d)
		fmt.Printf("%T, %v\n", u, u)
		switch w := d.(type) {
		case *utils.FileError:
			//fmt.Printf("%T, %v\n", w, w)
			err = w
		default:
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

	/*if _, ok := d.(error); ok {
		switch strings.ToLower(title) {
		case Perceptron:
			return perceptron.New()
		case Hopfield:
			return hopfield.New()
		default:
			err = fmt.Errorf("neural network is %w", pkg.ErrNotRecognized)
		}
	} else {
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
	}*/

	if err != nil {
		log.Println("get architecture:", err)
	}
	return nil
}
