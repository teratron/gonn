package nn

// Floater
type Floater interface {
	length() int
}

type (
	floatType  float32
	Float1Type []floatType
	Float2Type [][]floatType
	Float3Type [][][]floatType
)

func (f Float1Type) length() int {
	return len(f)
}

func (f Float2Type) length() int {
	return len(f)
}

func (f Float3Type) length() int {
	return len(f)
}
