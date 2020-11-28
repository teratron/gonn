package nn

import "github.com/teratron/gonn/pkg"

// Floater
type Floater interface {
	pkg.GetSetter
}

type (
	FloatType  float32
	Float1Type []FloatType
	Float2Type [][]FloatType
	Float3Type [][][]FloatType
)

func (f FloatType) Set(...pkg.Setter) {}

func (f FloatType) Get(...pkg.Getter) pkg.GetSetter {
	return f
}

func (f *Float1Type) Set(...pkg.Setter) {}

func (f *Float1Type) Get(...pkg.Getter) pkg.GetSetter {
	return f
}

func (f *Float2Type) Set(...pkg.Setter) {}

func (f *Float2Type) Get(...pkg.Getter) pkg.GetSetter {
	return f
}

func (f *Float3Type) Set(...pkg.Setter) {}

func (f *Float3Type) Get(...pkg.Getter) pkg.GetSetter {
	return f
}
