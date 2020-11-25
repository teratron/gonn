package main

import (
	"fmt"

	"github.com/zigenzoog/gonn/pkg/nn"
)

func main() {
	// Neural Network
	n := nn.New(nn.Perceptron())
	fmt.Println("nn.New():", n)

	// Common
	n.Set(
		nn.HiddenLayer(1, 5, 9),
		nn.NeuronBias(true),
		nn.ActivationMode(nn.ModeTANH),
		nn.LossMode(nn.ModeARCTAN),
		nn.LossLimit(.0005),
		nn.LearningRate(nn.DefaultRate))

	fmt.Printf("n.Get(): %T %v\n", n.Get(), n.Get())
	fmt.Println(n.HiddenLayer())
	fmt.Println(n.NeuronBias())
	fmt.Println(n.ActivationMode())
	fmt.Println(n.LossMode())
	fmt.Println(n.LossLimit())
	fmt.Println(n.LearningRate())

	// Hidden layers
	n.Set(nn.HiddenLayer(3, 2))
	fmt.Println("n.Get(nn.HiddenLayer()):", n.Get(nn.HiddenLayer()))

	// Bias
	n.Set(nn.NeuronBias(true))
	fmt.Println("n.Get(nn.NeuronBias()):", n.Get(nn.NeuronBias()))

	// Activation
	n.Set(nn.ActivationMode(nn.ModeSIGMOID))
	fmt.Println("n.Get(nn.ModeActivation()):", n.Get(nn.ActivationMode()))

	// Loss mode
	n.Set(nn.LossMode(nn.ModeMSE))
	fmt.Println("n.Get(nn.ModeLoss()):", n.Get(nn.LossMode()))

	// Limit loss
	n.Set(nn.LossLimit(.04))
	fmt.Println("n.Get(nn.LossLimit()):", n.Get(nn.LossLimit()))

	// Rate
	n.Set(nn.LearningRate(.1))
	fmt.Println("n.Get(nn.LearningRate()):", n.Get(nn.LearningRate()))

	// Weight
	fmt.Println("n.Get(nn.Weight()):", n.Get(nn.Weight()))
}
