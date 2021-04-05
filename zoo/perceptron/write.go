package perceptron

import (
	"fmt"

	"github.com/teratron/gonn/utils"
)

// WriteConfig writes the configuration and weights to the Filer interface object.
func (nn *NN) WriteConfig(name ...string) (err error) {
	if len(name) > 0 {
		switch d := utils.GetFileType(name[0]).(type) {
		case utils.Filer:
			err = d.Encode(nn)
		case error:
			err = d
		}
	} else if nn.config != nil {
		err = nn.config.Encode(nn)
	}

	if err != nil {
		err = fmt.Errorf("write config: %w", err)
	}
	return
}

// WriteConfig writes weights to the Filer interface object.
func (nn *NN) WriteWeight(name string) (err error) {
	switch d := utils.GetFileType(name).(type) {
	case utils.Filer:
		err = d.Encode(nn.Weights)
	case error:
		err = d
	}

	if err != nil {
		err = fmt.Errorf("write weights: %w", err)
	}
	return
}
