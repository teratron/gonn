package perceptron

import (
	"fmt"

	"github.com/teratron/gonn"
	"github.com/teratron/gonn/util"
)

// Read
func (p *perceptron) Read(reader gonn.Reader) (err error) {
	switch r := reader.(type) {
	case gonn.Filer:
		err = r.Read(p)
		if len(p.Weights) > 0 {
			p.initFromWeight()
		}
		switch s := r.(type) {
		case util.JsonString:
			p.jsonName = string(s)
		}
	default:
		err = fmt.Errorf("%T %w: %v", r, gonn.ErrMissingType, r)
	}
	if err != nil {
		err = fmt.Errorf("perceptron read: %w", err)
	}
	return
}

// Write
func (p *perceptron) Write(writer ...gonn.Writer) (err error) {
	if len(writer) > 0 {
		for _, w := range writer {
			switch v := w.(type) {
			case gonn.Filer:
				err = v.Write(p)
			default:
				err = fmt.Errorf("%T %w: %v", v, gonn.ErrMissingType, w)
			}
		}
	} else {
		err = fmt.Errorf("%w args", gonn.ErrEmpty)
	}
	if err != nil {
		err = fmt.Errorf("perceptron write: %w", err)
	}
	return
}
