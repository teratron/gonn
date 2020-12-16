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
	/*if len(row) > 0 {
		switch row[0] {
		case 0:
		default:
			return -1
		}
		for i, u := range row {
			for j, v := range u {
				for k := range v {

				}
			}
		}
	}*/
	return len(f)
}
