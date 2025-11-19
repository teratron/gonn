package layer

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/utils"
)

type Dense[T utils.Float] struct {
	*core
	Bias           bool            `json:"bias" xml:"bias"`
	ActivationMode activation.Type `json:"activation" xml:"activation"`
}
