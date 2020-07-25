package main

import (
	"fmt"
	"os"

	"github.com/zigenzoog/gonn/pkg/nn"
)

func main() {
	//fmt.Printf("%T %v\n", c, c)

	// Application
	a := nn.App()
	fmt.Println("nn.App():", a)

	// Common
	/*a.Set(nn.Language("ru"),
		  nn.Logging(1))*/

	// Language
	//n.Set(nn.Language("ru"))
	//fmt.Println("n.Get(nn.Language()):", n.Get(nn.Language()))

	// Logging
	//n.Set(nn.Logging(0)) //set
	//fmt.Println("n.Get(nn.Logging()):", n.Get(nn.Logging()))

	// Neural Network
	n := nn.New()	// same n := nn.New().Perceptron()
	//n := nn.New().Perceptron()
	//n := nn.New().RadialBasis()
	//n := nn.New().Hopfield()
	//fmt.Println("nn.New():", n)

	// Common
	n.Set(
		nn.Bias(false),
		nn.Rate(nn.DefaultRate),
		nn.ModeActivation(nn.ModeTANH),
		nn.ModeLoss(nn.ModeARCTAN),
		nn.LevelLoss(.0005),
		nn.HiddenLayer(1, 5, 9))
	fmt.Printf("n.Get(): %T %v\n", n.Get(), n.Get())

	// Bias
	n.Set(nn.Bias(true))
	fmt.Println("n.Get(nn.Bias()):", n.Get(nn.Bias()))

	// Rate
	n.Set(nn.Rate(.1))
	fmt.Println("n.Get(nn.Rate()):", n.Get(nn.Rate()))

	// Activation
	n.Set(nn.ModeActivation(nn.ModeSIGMOID))
	fmt.Println("n.Get(nn.ModeActivation()):", n.Get(nn.ModeActivation()))

	// Loss
	n.Set(nn.ModeLoss(nn.ModeMSE))
	fmt.Println("n.Get(nn.ModeLoss()):", n.Get(nn.ModeLoss()))

	// Level loss
	n.Set(nn.LevelLoss(.04))
	fmt.Println("n.Get(nn.LevelLoss()):", n.Get(nn.LevelLoss()))

	// Hidden layers
	n.Set(nn.HiddenLayer(3, 2))
	fmt.Println("n.Get(nn.HiddenLayer()):", n.Get(nn.HiddenLayer()))

	//
	input  := []float64{2.3, 3.1}
	target := []float64{3.6}

	//
	loss, count := n.Train(input, target)

	n.Print(os.Stdout, input, loss, count)

	file, err := os.Create("print.txt")
	if err != nil {
		os.Exit(1)
	}
	defer func() { _ = file.Close() }()
	n.Print(file, input, loss, count)

	//
	//fmt.Println(n.Query(input))

	//
	//fmt.Println(n.Verify(input, target))


	n.Write(nn.JSON("perceptron.json"))

	//fmt.Println(n.Get(nn.Neuron()))
	//fmt.Printf("++++ Act: %.4f\n", 100*calcActivation(1, ModeSIGMOID))



}
