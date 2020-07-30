package main

import (
	"fmt"
	"reflect"
)

type Foo struct {
	FirstName string `tag_name:"tag 1"`
	LastName  string `tag_name:"tag 2"`
	Age       int    `tag_name:"tag 3"`
}

func (f *Foo) reflect() {
	val := reflect.ValueOf(f).Elem()

	for i := 0; i < val.NumField(); i++ {
		valueField := val.Field(i)
		typeField  := val.Type().Field(i)
		tag        := typeField.Tag

		fmt.Printf("Field Name: %s,\t Field Value: %v,\t Tag Value: %s\n", typeField.Name, valueField.Interface(), tag.Get("tag_name"))
	}
}

func main() {
	f := &Foo{
		FirstName: "Drew",
		LastName:  "Olson",
		Age:       30,
	}

	f.reflect()
}





// Golang program to illustrate
// reflect.NewAt() Function

/*package main

import (
	"fmt"
	"reflect"
	"unsafe"
)

func main() {
	var s = struct{ foo int }{100}
	var i int

	rs := reflect.ValueOf(&s).Elem()
	rf := rs.Field(0)
	ri := reflect.ValueOf(&i).Elem()

	// use of NewAt() method
	rf = reflect.NewAt(rf.Type(), unsafe.Pointer(rf.UnsafeAddr())).Elem()
	fmt.Println(rf)
	ri.Set(rf)
	rf.Set(ri)

	fmt.Println(rf)
}*/


/*package main

import (
	"fmt"
	"reflect"
)

func main() {

	type Product struct {
		Name  string
		Price string
	}

	var product Product
	productType := reflect.TypeOf(product)       // this type of this variable is reflect.Type
	fmt.Println(productType)

	productPointer := reflect.New(productType)   // this type of this variable is reflect.Value.
	fmt.Println(productPointer)

	productValue := productPointer.Elem()        // this type of this variable is reflect.Value.
	fmt.Println(productValue)

	productInterface := productValue.Interface() // this type of this variable is interface{}
	fmt.Println(productInterface)

	product2 := productInterface.(Product)       // this type of this variable is product
	fmt.Println(product2)

	product2.Name = "Toothbrush"
	product2.Price = "2.50"

	fmt.Println(product2.Name)
	fmt.Println(product2.Price)

}*/


/*package main

import (
	"fmt"
	"reflect"
)

type Config struct {
	Name string
	Meta struct {
		Desc string
		Properties map[string]string
		Users []string
	}
}

func initializeStruct(t reflect.Type, v reflect.Value) {
	for i := 0; i < v.NumField(); i++ {
		f := v.Field(i)
		ft := t.Field(i)
		switch ft.Type.Kind() {
		case reflect.Map:
			f.Set(reflect.MakeMap(ft.Type))
		case reflect.Slice:
			f.Set(reflect.MakeSlice(ft.Type, 0, 0))
		case reflect.Chan:
			f.Set(reflect.MakeChan(ft.Type, 0))
		case reflect.Struct:
			initializeStruct(ft.Type, f)
		case reflect.Ptr:
			fv := reflect.New(ft.Type.Elem())
			initializeStruct(ft.Type.Elem(), fv.Elem())
			f.Set(fv)
		default:
		}
	}
}

func main() {
	t := reflect.TypeOf(Config{})
	v := reflect.New(t)
	initializeStruct(t, v.Elem())
	c := v.Interface().(*Config)
	c.Meta.Properties["color"] = "red" // map was already made!
	c.Meta.Users = append(c.Meta.Users, "srid") // so was the slice.
	fmt.Println(v.Interface())
}*/


/*package main

import (
	"fmt"
	"reflect"
)

func main() {
	// one way is to have a value of the type you want already
	a := 1
	// reflect.New works kind of like the built-in function new
	// We'll get a reflected pointer to a new int value
	intPtr := reflect.New(reflect.TypeOf(a))
	// Just to prove it
	b := intPtr.Elem().Interface().(int)
	// Prints 0
	fmt.Println(intPtr, b)

	// We can also use reflect.New without having a value of the type
	var nilInt *int
	intType := reflect.TypeOf(nilInt).Elem()
	intPtr2 := reflect.New(intType)
	// Same as above
	c := intPtr2.Elem().Interface().(int)
	// Prints 0 again
	fmt.Println(intType, intPtr2, c)
}*/