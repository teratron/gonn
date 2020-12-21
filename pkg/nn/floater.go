package nn

// Floater
type Floater interface {
	Length() int
}

type (
	FloatType  float32
	Float1Type []FloatType
	Float2Type [][]FloatType
	Float3Type [][][]FloatType
)

func (f Float1Type) Length() int {
	return len(f)
}

func (f Float2Type) Length() int {
	return len(f)
}

func (f Float3Type) Length() int {
	return len(f)
}
