package layer

import (
	"github.com/teratron/gonn/pkg/utils"
)

type Input[T utils.Float] core

func NewInput[T utils.Float]() *Input[T] {
	return &Input[T]{}
}
