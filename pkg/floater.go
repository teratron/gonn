package pkg

import "log"

// Floater.
type Floater interface {
	Length(...uint) int
}

type (
	FloatType  float32
	Float1Type []FloatType
	Float2Type [][]FloatType
	Float3Type [][][]FloatType
)

// Length.
func (f Float1Type) Length(...uint) int {
	return len(f)
}

// Length.
func (f Float2Type) Length(index ...uint) int {
	if len(index) > 0 {
		if len(f) > int(index[0]) {
			return len(f[index[0]])
		}
		log.Println("error Float2Type length: index exceeds array size")
		return 0
	}
	return len(f)
}

// Length.
func (f Float3Type) Length(index ...uint) int {
	switch len(index) {
	case 0:
		return len(f)
	case 1:
		if len(f) > int(index[0]) {
			return len(f[index[0]])
		}
	default:
		fallthrough
	case 2:
		if len(f) > int(index[0]) && len(f[index[0]]) > int(index[1]) {
			return len(f[index[0]][index[1]])
		}
	}
	log.Println("error Float3Type length: index exceeds arrays size")
	return 0
}
