package main

import "fmt"

type Test struct {
	array []*Array
}

type Array struct {
	value float32
}

func (t *Test) Testing() {
	for i, a := range t.array {
		if i == 0 {
			fmt.Printf("%T %v\n", a, &a.value)
			fmt.Printf("%T %v\n", t.array[i], &t.array[i].value)
		}
	}
}

/*func main() {
	t := Test{}
	t.array = make([]*Array, 5)
	for i := 0; i < 5; i++ {
		t.array[i] = &Array{}
	}
	t.Testing()
}*/