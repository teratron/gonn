package main

import (
	"fmt"

	"github.com/zigenzoog/gonn/pkg/nn"
)

func main() {
	// Neural Network
	n := nn.New(nn.JSON("perceptron.json"))

	fmt.Println("nn.New(JSON(\"file\")):", n)

	n.Write(
		nn.JSON(),
		nn.XML())

	//b, err := json.Marshal(m)
	/*rawDataOutjson, err := json.MarshalIndent(m, "", "\t")
	if err != nil { panic("!!!") }
	fmt.Println("++++ \n", string(rawDataOutjson))

	rawDataOut, err := xml.MarshalIndent(m, "", "\t")
	if err != nil { panic("!!!") }
	fmt.Println("++++ \n", string(rawDataOut))*/
}