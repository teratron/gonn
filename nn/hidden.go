// Hidden layers
package nn

type (
	hiddenType uint16
	HiddenType []hiddenType
	
	//cH chan bool
)

/*func (c cH) Set(...Setter) {
	panic("implement me")
}*/

func HiddenLayer(nums ...hiddenType) HiddenType {
	return nums
}

// Setter
func (h HiddenType) Set(args ...Setter) {
	//ch := make(cH)
	//fmt.Printf("%T %v\n", set[0], set[0])
	/*if v, ok := getArchitecture(set[0]); ok {
		//fmt.Printf("%T %v\n", v, v)
		fmt.Println("1 go", ch)
		go v.Set(h, ch)
		fmt.Println("2 go", <-ch)
	}*/

	if n, ok := args[0].(*nn); ok {
		if v, ok := n.architecture.(NeuralNetwork); ok {
			v.Set(h, n)
		}
	}
}

// Getter
func (h HiddenType) Get(args ...Setter) Getter {
	if a, ok := args[0].(Architecture); ok {
		if v, ok := getArchitecture(a); ok {
			return v.Get(h)
		}
	}
	return nil
}