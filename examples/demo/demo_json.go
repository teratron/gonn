package main

import (
	"encoding/json"
	"fmt"
	"github.com/zigenzoog/gonn/pkg/nn"
)

type Message struct {
	Name string
	Body string
	Time int64
}



func main() {
	// Neural Network
	n := nn.New(nn.JSON("perceptron"))

	fmt.Println("nn.New(JSON(\"file\")):", n)




	m := Message{
		"Alice",
		"Hello",
		1294706395881547000}


	b, err := json.Marshal(m)
	if err != nil { panic("!!!") }
	fmt.Println(string(b))
}