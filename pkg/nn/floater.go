package nn

import (
	"github.com/zigenzoog/gonn/pkg"
)

// Floater
type Floater interface {
	float()
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

func (f floatType) float()  {}
func (f float1Type) float() {}
func (f float2Type) float() {}
func (f float3Type) float() {}
