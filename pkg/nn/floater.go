package nn

import (
	"fmt"
	"github.com/zigenzoog/gonn/pkg"
)

type Floater interface {
	getFloat(...uint) floatType
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

func (f float1Type) getFloat(index ...uint) floatType {
	if len(index) >= 1 {
		return f[index[0]]
	}
	return 0
}

func (f float2Type) getFloat(index ...uint) floatType {
	fmt.Println(len(f))
	if len(index) >= 2 {
		return f[index[0]][index[1]]
	}
	return 0
}

func (f float3Type) getFloat(index ...uint) floatType {
	if len(index) >= 3 {
		return f[index[0]][index[1]][index[2]]
	}
	return 0
}