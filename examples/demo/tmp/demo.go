package main

import (
	"github.com/teratron/gonn/pkg/nn"
)

func main() {
	n := nn.New( /*nn.Perceptron() nn.JSON("tmp.json")*/ )
	//fmt.Println(len(n.HiddenLayer()))
	for i := 0; i < 5; i++ {
		nn.Debug()
	}

	//n.Read(nn.JSON("tmp.json"))
	//fmt.Println(n, len([]float64{}))
	//fmt.Println(n.Train([]float64{1, 0}, []float64{0, 1}))
	//fmt.Println(n.Verify([]float64{1, 0}, []float64{0, 1}), n.Weight())

	//mt.Println(n.Error())

	/*n.SetHiddenLayer(0)
	n.SetNeuronBias(true)
	n.SetActivationMode(nn.ModeSIGMOID)
	n.SetLossMode(nn.ModeMSE)
	n.SetLossLimit(.1)
	n.SetLearningRate(nn.DefaultRate)*/
	//fmt.Println("2")

	//fmt.Println(n.Train([]float64{1, 0}, []float64{0, 1}))
	//fmt.Println("3")

	/*d := n.Weight()
	fmt.Println(d)
	n.SetWeight(d)
	fmt.Println(n.Weight())
	fmt.Println(d.Dimension())*/

	n.Write(
		nn.JSON("tmp5.json"),
		/*nn.Report( nn.File("report.txt"), []float64{1, 0}, 0, 0)*/)
	//n.Read(nn.JSON("tmp.json"))
	//n.Write(nn.JSON("tmp2.json"))
}
