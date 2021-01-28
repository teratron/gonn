package nn

// Floater
type Floater interface {
	length() int
}

type (
	Float1Type []float64
	Float2Type [][]float64
	Float3Type [][][]float64
)

func (f Float1Type) length() int { return len(f) }
func (f Float2Type) length() int { return len(f) }
func (f Float3Type) length() int { return len(f) }
