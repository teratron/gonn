package nn

// Floater
type Floater interface {
	GetSetter
}

type (
	FloatType  float32
	Float1Type []FloatType
	Float2Type [][]FloatType
	Float3Type [][][]FloatType
)

func (f FloatType) Set(...Setter) {}

func (f FloatType) Get(...Getter) GetSetter {
	return f
}

func (f *Float1Type) Set(...Setter) {}

func (f *Float1Type) Get(...Getter) GetSetter {
	return f
}

func (f *Float2Type) Set(...Setter) {}

func (f *Float2Type) Get(...Getter) GetSetter {
	return f
}

func (f *Float3Type) Set(...Setter) {}

func (f *Float3Type) Get(...Getter) GetSetter {
	return f
}
