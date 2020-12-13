package nn

// Floater
type Floater interface {
	Dimension() int
}

type (
	FloatType  float32
	Float1Type []FloatType
	Float2Type [][]FloatType
	Float3Type [][][]FloatType
)

func (f FloatType) Dimension() int {
	return 0
}

func (f Float1Type) Dimension() int {
	return 1
}

func (f Float2Type) Dimension() int {
	return 2
}

func (f Float3Type) Dimension() int {
	return 3
}
