// Hidden layers
package nn

import "fmt"

type (
	hiddenType uint16
	HiddenType []hiddenType
)

func HiddenLayer(nums ...hiddenType) HiddenType {
	return nums
}

// Setter
func (h HiddenType) Set(set ...Setter) {
	fmt.Printf("%T %v\n", set[0], set[0])
	if v, ok := getArchitecture(set[0]); ok {
		//fmt.Printf("%T %v\n", v, v)
		v.Set(h)
	}
}

// Getter
func (h HiddenType) Get(set ...Setter) Getter {
	if v, ok := getArchitecture(set[0]); ok {
		return v.Get(h)
	}
	return nil
}

