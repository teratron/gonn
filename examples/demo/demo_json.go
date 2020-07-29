package main

import (
	"fmt"
	"github.com/zigenzoog/gonn/pkg/nn"
	"reflect"
)

type Message struct {
	Name string
	Body string
	Time int64
}

type NN struct {
	Architecture	Message `json:"architecture"`	// Architecture of neural network
	IsInit  bool `json:"isInit"`  // Neural network initializing flag
	IsTrain bool `json:"isTrain"` // Neural network training flag
	JSON	string
}

func (n NN) M() {}

func main() {
	// Neural Network
	n := nn.New(nn.JSON("perceptron.json"))

	fmt.Println("nn.New(JSON(\"file\")):", n)

	/*m := Message{
		"Alice",
		"Hello",
		1294706395881547000}*/

	m := NN{
		Message{},
		true,
		false,
		"json",
	}

	fmt.Printf("ValueOf-----%T -----%v\n", reflect.ValueOf(m), reflect.ValueOf(m))
	fmt.Printf("Kind-----%T -----%v\n", reflect.ValueOf(m).Kind(), reflect.ValueOf(m).Kind())
	fmt.Printf("Type-----%T -----%v\n", reflect.ValueOf(m).Type(), reflect.ValueOf(m).Type())
	fmt.Printf("NumField-----%T -----%v\n", reflect.ValueOf(m).NumField(), reflect.ValueOf(m).NumField())
	fmt.Printf("Field-----%T -----%v\n", reflect.ValueOf(m).Field(0), reflect.ValueOf(m).Field(0).Type().Name)
	fmt.Printf("Field-----%T -----%v\n", reflect.ValueOf(m).Field(3), reflect.ValueOf(m).Field(3))
	fmt.Printf("Field-----%T -----%v\n", reflect.ValueOf(m).Field(3), reflect.ValueOf(m).Type().Field(3).Name)
	fmt.Printf("NumMethod-----%T -----%v\n", reflect.ValueOf(m).NumMethod(), reflect.ValueOf(m).NumMethod())
	fmt.Printf("Method-----%T -----%v\n", reflect.ValueOf(m).Method(0), reflect.ValueOf(m).Method(0))
	fmt.Printf("Method-----%T -----%v\n", reflect.ValueOf(m).Method(0), reflect.ValueOf(m).Type().Method(0))
	fmt.Printf("Method-----%T -----%v\n", reflect.ValueOf(m).Method(0), reflect.ValueOf(m).Type().Method(0).Name)

	//b, err := json.Marshal(m)
	/*rawDataOut, err := json.MarshalIndent(m, "", "\t")
	if err != nil { panic("!!!") }
	fmt.Println("++++ \n", string(rawDataOut))*/
}