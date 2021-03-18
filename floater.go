package gonn

import (
	"fmt"
	"log"
)

// Floater
type Floater interface {
	Length(...uint) int
}

type (
	Float1Type []float64
	Float2Type [][]float64
	Float3Type [][][]float64
)

func (f Float1Type) Length(...uint) int {
	return len(f)
}

func (f Float2Type) Length(ind ...uint) int {
	if len(ind) > 0 {
		if len(f) > int(ind[0]) {
			return len(f[ind[0]])
		}
		log.Println(fmt.Errorf("error Float2Type length: index exceeds array size"))
		return 0
	}
	return len(f)
}

func (f Float3Type) Length(ind ...uint) int {
	switch len(ind) {
	case 0:
		return len(f)
	case 1:
		if len(f) > int(ind[0]) {
			return len(f[ind[0]])
		}
	default:
		fallthrough
	case 2:
		if len(f) > int(ind[0]) && len(f[ind[0]]) > int(ind[1]) {
			return len(f[ind[0]][ind[1]])
		}
	}
	log.Println(fmt.Errorf("error Float3Type length: index exceeds arrays size"))
	return 0
}