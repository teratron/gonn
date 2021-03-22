package nn

import (
	"fmt"
	"log"
	"path/filepath"

	"github.com/teratron/gonn"
	"github.com/teratron/gonn/nn/architecture"
	"github.com/teratron/gonn/util"
)

// NeuralNetwork
type NeuralNetwork interface {
	gonn.NeuralNetwork
}

// Reader
/*type Reader interface {
	gonn.Reader
}

// Writer
type Writer interface {
	gonn.Writer
}*/

// Floater
type Floater interface {
	gonn.Floater
}

// New returns a new neural network instance.
func New(reader ...string) NeuralNetwork {
	if len(reader) > 0 {
		if filepath.Ext(reader[0]) != "" {
			d := util.GetFileType(reader[0])
			err, ok := d.(error)
			if !ok {
				switch v := d.GetValue("name").(type) {
				case string:
					n := architecture.Get(v)
					if err = d.Decode(n); err == nil {
						n.SetConfig(d)
						return n
					}
				case error:
					err = v
				}
			}

			if err != nil {
				err = fmt.Errorf("new: %w", err)
				log.Println(err)
			}
			return nil
		}
		return architecture.Get(reader[0])
	}
	return architecture.Get(architecture.Perceptron)
}
