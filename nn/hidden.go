// Hidden layers
package nn

type (
	hiddenType uint16
	HiddenType []hiddenType
)

func HiddenLayer(nums ...hiddenType) HiddenType {
	return nums
}

// Setter
func (h HiddenType) Set(set ...Setter) {
	if v, ok := getArchitecture(set[0]); ok {
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