package perceptron

import (
	"fmt"

	"github.com/teratron/gonn/utils"
)

// WriteConfig writes the configuration and weights to the Filer interface object.
func (p *perceptron) WriteConfig(name ...string) (err error) {
	if len(name) > 0 {
		switch d := utils.GetFileType(name[0]).(type) {
		case utils.Filer:
			err = d.Encode(p)
		case error:
			err = d
		}
	} else if p.config != nil {
		err = p.config.Encode(p)
	}

	if err != nil {
		err = fmt.Errorf("write config: %w", err)
	}
	return
}

// WriteConfig writes weights to the Filer interface object.
func (p *perceptron) WriteWeight(name string) (err error) {
	switch d := utils.GetFileType(name).(type) {
	case utils.Filer:
		err = d.Encode(p.Weights)
	case error:
		err = d
	}

	if err != nil {
		err = fmt.Errorf("write weights: %w", err)
	}
	return
}
