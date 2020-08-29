package main

import (
	"fmt"

	"github.com/zigenzoog/gonn/pkg/nn"
)

func main() {
	//fmt.Printf("%T %v\n", c, c)

	// Neural Network
	n := nn.New(nn.Perceptron())
	fmt.Println("nn.New():", n)

	// Common
	n.Set(
		nn.HiddenLayer(1, 5, 9),
		nn.Bias(true),
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

	// Loss mode
	n.Set(nn.LossMode(nn.ModeMSE))
	fmt.Println("n.Get(nn.ModeLoss()):", n.Get(nn.LossMode()))

	// Level loss
	n.Set(nn.LossLevel(.04))
	fmt.Println("n.Get(nn.LevelLoss()):", n.Get(nn.LossLevel()))

	// Rate
	n.Set(nn.Rate(.1))
	fmt.Println("n.Get(nn.Rate()):", n.Get(nn.Rate()))

	// Weight

	fmt.Println("n.Get(nn.Weight()):", n.Get(nn.Weight()))
	n.Set(nn.Weight())
}
