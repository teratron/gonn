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
func Get(reader string) pkg.NeuralNetwork {
	//fmt.Printf("%v\n", utils.GetFileEncoding([]byte(reader)).(*utils.FileJSON).Data)
	var err error
	f := utils.GetFileEncoding([]byte(reader))
	if _, ok := f.(*utils.FileError); ok {
		f = utils.GetFileType(reader)
		if _, ok = f.(*utils.FileError); ok {
			switch strings.ToLower(reader) {
			case Perceptron:
				return perceptron.New()
			case Hopfield:
				return hopfield.New()
			default:
				err = fmt.Errorf("neural network is %w", pkg.ErrNotRecognized)
			}
		} else {
			switch v := f.GetValue("name").(type) {
			case error:
				err = v
			case string:
				if n := Get(v); n != nil {
					if err = f.Decode(n); err == nil {
						n.Init(f)
						return n
					}
				}
			}
		}
	} else {
		switch v := f.GetValue("name").(type) {
		case error:
			err = v
		case string:
			if n := Get(v); n != nil {
				if err = f.Decode(n); err == nil {
					n.Init(f)
					return n
				}
			}
		}
	}

	/*if _, ok := d.(error); ok {
		switch strings.ToLower(reader) {
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
