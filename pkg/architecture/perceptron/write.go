package perceptron

import (
	"fmt"
	"log"

	"github.com/teratron/gonn/pkg"
	"github.com/teratron/gonn/pkg/utils"
)

// WriteConfig writes the configuration and weights to the Filer interface object.
func (nn *NN) WriteConfig(name ...string) (err error) {
	if len(name) > 0 {
		switch d := utils.GetFileType(name[0]).(type) {
		case error:
			err = d
		case utils.Filer:
			err = d.Encode(nn)
		}
	} else if nn.config != nil {
		err = nn.config.Encode(nn)
	} else {
		err = pkg.ErrNoArgs
	}

	if err != nil {
		err = fmt.Errorf("perceptron.NN.WriteConfig: %w", err)
		log.Print(err)
	}
	return
}

// WriteWeight writes weights to the Filer interface object.
func (nn *NN) WriteWeight(name string) (err error) {
	switch d := utils.GetFileType(name).(type) {
	case error:
		err = d
	case utils.Filer:
		err = d.Encode(nn.Weight)
	}

	if err != nil {
		err = fmt.Errorf("perceptron.NN.WriteWeight: %w", err)
		log.Print(err)
	}
	return
}
