package main

import "fmt"

func main() {
	x := []int{1, 2}
	y := []int{3, 4}
	ref := x
	x = y
	fmt.Println(x, y, ref)

	fmt.Println(min(float32(23), 42))
	fmt.Println(max(23., 42.))
}

// With Generics
type number interface {
	float32 | float64 | int | int16 | int32 | int64 | uint | uint16 | uint32 | uint64
}

func min[T number](x, y T) T {
	if x > y {
		return y
	}
	return x
}

func max[T number](x, y T) T {
	if x > y {
		return x
	}
	return y
}

// Without Generics
//type Number interface {
//	min(int, int) int
//	//max(int, int) int
//}
//
//type intT int
//type floatT float64
//
//func Min(x, y intT) intT {
//	if x > y {
//		return y
//	}
//	return x
//}
