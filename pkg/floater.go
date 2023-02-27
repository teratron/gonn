package pkg

import "log"

// Floater.
type Floater interface {
	Length(...uint) int
	Copy(Floater)
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

// Copy.
func (f Float1Type) Copy(src Floater) {
	if s, ok := src.(Float1Type); ok {
		_ = copy(f, s)
	}

}

// ToFloat1Type.
func ToFloat1Type(src []float64) Float1Type {
	dst := make(Float1Type, len(src))
	for i, v := range src {
		dst[i] = FloatType(v)
	}
	return dst
}

// Length.
func (f Float2Type) Length(index ...uint) int {
	if len(index) > 0 {
		if len(f) > int(index[0]) {
			return len(f[index[0]])
		}
		log.Println("pkg.Float2Type.Length error: index exceeds array size")
		return 0
	}
	return len(f)
}

// Copy.
func (f Float2Type) Copy(src Floater) {
	if s, ok := src.(Float2Type); ok {
		for i := range s {
			_ = copy(f[i], s[i])
		}
	}
}

// ToFloat2Type.
func ToFloat2Type(src [][]float64) Float2Type {
	dst := make(Float2Type, len(src))
	for i, v := range src {
		dst[i] = make(Float1Type, len(v))
		for j, w := range v {
			dst[i][j] = FloatType(w)
		}
	}
	return dst
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
	log.Println("pkg.Float3Type.Length error: index exceeds arrays size")
	return 0
}

// Copy.
func (f Float3Type) Copy(src Floater) {
	if s, ok := src.(Float3Type); ok {
		for i, v := range s {
			for j := range v {
				_ = copy(f[i][j], s[i][j])
			}
		}
	}
}

// ToFloat3Type.
func ToFloat3Type(src [][][]float64) Float3Type {
	dst := make(Float3Type, len(src))
	for i, u := range src {
		dst[i] = make(Float2Type, len(u))
		for j, v := range u {
			dst[i][j] = make(Float1Type, len(v))
			for k, w := range v {
				dst[i][j][k] = FloatType(w)
			}
		}
	}
	return dst
}
