package main

import (
	"fmt"
	"github.com/zigenzoog/gonn/pkg/nn"
	"os"

	"github.com/zigenzoog/gonn/pkg/app"
)

func main() {
	//fmt.Printf("%T %v\n", c, c)

	// Application
	a := app.App()
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
	//n := nn.New(nn.Hopfield())
	//fmt.Println("nn.New():", n)

	// Common
	n.Set(
		nn.HiddenLayer(1, 5, 9),
		nn.Bias(false),
		nn.ActivationMode(nn.ModeTANH),
		nn.LossMode(nn.ModeARCTAN),
		nn.LossLevel(.0005),
		nn.Rate(nn.DefaultRate))
	fmt.Printf("n.Get(): %T %v\n", n.Get(), n.Get())

	fmt.Println(n.HiddenLayer())
	fmt.Println(n.Bias())
	fmt.Println(n.ActivationMode())
	fmt.Println(n.LossMode())
	fmt.Println(n.LossLevel())
	fmt.Println(n.Rate())

	// Hidden layers
	n.Set(nn.HiddenLayer(3, 2))
	fmt.Println("n.Get(nn.HiddenLayer()):", n.Get(nn.HiddenLayer()))

	// Bias
	n.Set(nn.Bias(true))
	fmt.Println("n.Get(nn.Bias()):", n.Get(nn.Bias()))

	// Activation
	n.Set(nn.ActivationMode(nn.ModeSIGMOID))
	fmt.Println("n.Get(nn.ModeActivation()):", n.Get(nn.ActivationMode()))

	// Loss
	n.Set(nn.LossMode(nn.ModeMSE))
	fmt.Println("n.Get(nn.ModeLoss()):", n.Get(nn.LossMode()))

	// Level loss
	n.Set(nn.LossLevel(.04))
	fmt.Println("n.Get(nn.LevelLoss()):", n.Get(nn.LossLevel()))

	// Rate
	n.Set(nn.Rate(.1))
	fmt.Println("n.Get(nn.Rate()):", n.Get(nn.Rate()))

	//
	input  := []float64{2.3, 3.1}
	target := []float64{3.6}

	//
	loss, count := n.Train(input, target)

	file, err := os.Create("report.txt")
	if err != nil {
		os.Exit(1)
	}
	defer func() { _ = file.Close() }()
	//n.Print(file, input, loss, count)
	//n.Print(os.Stdout, input, loss, count)

	//
	//fmt.Println(n.Query(input))

	//
	//fmt.Println(n.Verify(input, target))


	n.Write(nn.JSON("perceptron.json"))
	n.Set(nn.Bias(true))
	//fmt.Println(n.Get(nn.Neuron()))
	//fmt.Printf("++++ Act: %.4f\n", 100*calcActivation(1, ModeSIGMOID))

	n.Write(nn.JSON("perceptron.json"),
		nn.Report(file, input, loss, count),
		nn.Report(os.Stdout, input, loss, count))

}
