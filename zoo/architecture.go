package zoo

import (
	"fmt"
	"log"
	"path/filepath"
	"strings"

	"github.com/teratron/gonn"
	"github.com/teratron/gonn/nn/hopfield"
	"github.com/teratron/gonn/nn/perceptron"
	"github.com/teratron/gonn/util"
)

const (
	Perceptron = perceptron.Name
	Hopfield   = hopfield.Name
)

// Get
func Get(name string) gonn.NeuralNetwork {
	var err error
	if filepath.Ext(name) != "" {
		d := util.GetFileType(name)
		err, ok := d.(error)
		if !ok {
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
		}
		err = fmt.Errorf("file config: %w", err)
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
