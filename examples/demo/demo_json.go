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

type NN struct {
	Architecture	Message `json:"architecture"`	// Architecture of neural network

	IsInit  bool `json:"isInit"`  // Neural network initializing flag
	IsTrain bool `json:"isTrain"` // Neural network training flag

	JSON	string
	/*xml		string
	csv		string*/
}

func main() {
	// Neural Network
	n := nn.New(nn.JSON("perceptron.json"))

	fmt.Println("nn.New(JSON(\"file\")):", n)

	/*m := Message{
		"Alice",
		"Hello",
		1294706395881547000}*/

	m := &NN{
		Message{},
		true,
		false,
		"json",
	}

	//b, err := json.Marshal(m)
	rawDataOut, err := json.MarshalIndent(m, "", "\t")
	if err != nil { panic("!!!") }
	fmt.Println(string(rawDataOut))
}