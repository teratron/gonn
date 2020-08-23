package main

import (
	"github.com/zigenzoog/gonn/pkg/nn"
	"log"
)

func main() {
	// New returns a new neural network instance with the default parameters
	// same n := nn.New(Perceptron())
	n := nn.New(nn.Perceptron())

	// Set parameters
	n.Set(
		nn.HiddenLayer(3, 5),
		nn.Bias(true),
		nn.ActivationMode(nn.ModeSIGMOID),
		nn.LossMode(nn.ModeMSE),
		nn.LossLevel(.0001),
		nn.Rate(nn.DefaultRate))

	input  := []float64{2.3, 3.1}
	target := []float64{3.6}

	//
	//loss, count := n.Train(input, target)
	_, _ = n.Train(input, target)

	//
	//fmt.Println(n.Query(input))

	//
	//fmt.Println(n.Verify(input, target))
	n.Copy(nn.Weight())
	err := n.Paste(nn.Weight())
	if err != nil {
		log.Println(err)
	}

	//
	n.Write(
		nn.JSON("config/perceptron.json"),
		nn.XML("config/perceptron.xml"),
		/*nn.Report(nn.File("report.txt"), input, loss, count),
		nn.Report(os.Stdout, input, loss, count)*/)

	n.Read(nn.JSON("config/perceptron.json"))
	n.Write(nn.JSON("config/perceptron2.json"))
	//nn.Debug(n)

	//n.Read(nn.XML("config/perceptron.xml"))
	//n.Write(nn.XML("config/perceptron2.xml"))
}