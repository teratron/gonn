package main

import "fmt"

type Test struct {
	array []Array
}

type Array struct {
	value float32
}

func (t *Test) Testing() {
	for i, a := range t.array {
		if i == 0 {
			//t.array[i].value = 1.2
			fmt.Printf("%T %v\n", a, a.value)
			fmt.Printf("%T %v\n", t.array[i], t.array[i].value)
		}
	}
}

func main() {
	t := Test{}
	t.array = make([]Array, 5)

	t = Test{
		array: []Array{{value: 2.1}},
	}

	/*t.array[0].value = 1.2
	t.array[1].value = 3.2
	t.array[2].value = 1.6*/

	t.Testing()
}