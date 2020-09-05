package nn

import (
	"github.com/zigenzoog/gonn/pkg"
)

// Floater
type Floater interface {
	//float()
	pkg.GetSetter
	//pkg.Controller
}

type (
	floatType  float32
	float1Type []floatType
	float2Type [][]floatType
	float3Type [][][]floatType
)

func (f floatType) Set(...pkg.Setter) {}

func (f floatType) Get(...pkg.Getter) pkg.GetSetter {
	return f
}

func (f *float1Type) Set(...pkg.Setter) {}

func (f *float1Type) Get(...pkg.Getter) pkg.GetSetter {
	return f
}

/*func (f float1Type) Copy(pkg.Copier) {}

func (f float1Type) Paste(pkg.Paster) {}

func (f float1Type) Read(pkg.Reader) {}

func (f float1Type) Write(...pkg.Writer) {}*/

func (f *float2Type) Set(...pkg.Setter) {}

func (f *float2Type) Get(...pkg.Getter) pkg.GetSetter {
	return f
}

/*func (f float2Type) Copy(pkg.Copier) {}

func (f float2Type) Paste(pkg.Paster) {}

func (f float2Type) Read(pkg.Reader) {}

func (f float2Type) Write(...pkg.Writer) {}*/

func (f *float3Type) Set(...pkg.Setter) {}

func (f *float3Type) Get(...pkg.Getter) pkg.GetSetter {
	return f
}

/*func (f float3Type) Copy(pkg.Copier) {}

func (f float3Type) Paste(pkg.Paster) {}

func (f float3Type) Read(pkg.Reader) {}

func (f float3Type) Write(...pkg.Writer) {}*/

/*func (f floatType) float()  {}
func (f float1Type) float() {}
func (f float2Type) float() {}
func (f float3Type) float() {}*/
