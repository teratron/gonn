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
				goto ERROR
			}
		}
	}

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

ERROR:
	if err != nil {
		log.Printf("arch.Get: %v", err)
	}
	return nil
}
