package perceptron

import (
	"fmt"
	"log"

	"github.com/teratron/gonn/pkg"
	"github.com/teratron/gonn/pkg/utils"
)

// WriteConfig writes the configuration and weights to the Filer interface object.
func (nn *NN) WriteConfig(filename ...string) (err error) {
	if len(filename) > 0 {
		switch d := utils.GetFileType(filename[0]).(type) {
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

// WriteWeights writes weights to the Filer interface object.
func (nn *NN) WriteWeights(filename string) (err error) {
	switch d := utils.GetFileType(filename).(type) {
	case error:
		err = d
	case utils.Filer:
		err = d.Encode(nn.Weights)
	}

	if err != nil {
		err = fmt.Errorf("perceptron.NN.WriteWeights: %w", err)
		log.Print(err)
	}
	return
}
