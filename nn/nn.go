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
/*func New(reader ...Reader) NeuralNetwork {
	if len(reader) > 0 {
		var err error
		switch r := reader[0].(type) {
		case NeuralNetwork:
			return r
		case gonn.Filer:
			switch v := r.GetValue("name").(type) {
			case string:
				n := architecture.Get(v)
				if err = n.Read(r); err == nil {
					return n
				}
			case error:
				err = v
			}
		default:
			err = fmt.Errorf("%T %w", r, gonn.ErrMissingType)
		}

		if err != nil {
			err = fmt.Errorf("new: %w", err)
			log.Println(err)
		}
		return nil
	}
	return architecture.Get(architecture.Perceptron)
}*/

func New(reader ...string) NeuralNetwork {
	if len(reader) > 0 {
		ext := filepath.Ext(reader[0])
		if ext != "" /*strings.Index(reader[0], ".") > -1 strings.Contains(reader[0], ".")*/ {
			d := util.GetFileType(ext)
			err, ok := d.(error)
			if !ok {
				switch v := d.GetValue("name").(type) {
				case string:
					n := architecture.Get(v)
					if err = d.Decode(n); err == nil {
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
