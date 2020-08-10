package main

import (
	"github.com/zigenzoog/gonn/pkg/nn"
)

func main() {
	// New returns a new neural network
	// instance with the default parameters
	// same n := nn.New(Perceptron())
	n := nn.New(nn.Perceptron())

	//fmt.Println(n)

	// Set parameters
	n.Set(
		nn.HiddenLayer(3, 2),
		nn.Bias(true),
		nn.ActivationMode(nn.ModeSIGMOID),
		nn.LossMode(nn.ModeMSE),
		nn.LossLevel(.0001),
		nn.Rate(nn.DefaultRate))

	//
	//numInputData  := 8	// 5
	//numOutputData := 1
	//dataScale     := 1000.  // Коэфициент масштабирования данных, приводящих к промежутку от -1 до 1
	input  := []float64{2.3, 3.1}
	target := []float64{3.6}

	//
	loss, count := n.Train(input, target)
	//_, _ = n.Train(input, target)

	//
	//fmt.Println(n.Query(input))

	//
	//fmt.Println(n.Verify(input, target))

	// Обучение
	/*maxEpoch := 100000
	minError := 1.
	for epoch := 1; epoch <= maxEpoch; epoch++ {
		for i := numInputData; i <= len(dataSet) - numOutputData; i++ {
			//input  = getInputArray(dataset[i - numInputData:i])
			//target = getTargetArray(dataset[i:i + numOutputData])
			loss, count = n.Train(input, target)
		}

		// Verifying
		sum := 0.
		num := 0
		for i := numInputData; i <= len(dataSet) - numOutputData; i++ {
			loss = n.Verify(input, target)
			sum += loss
			num++
		}

		// Средняя ошибка за всю эпоху
		sum /= float64(num)

		//
		//if loss > mx.Limit {
			//if epoch == 1 || epoch == 10 || epoch % 1000 == 0 || epoch == maxEpoch {
				//fmt.Printf("+++++++++ Epoch: %v\tError: %.8f\n", epoch, sum)
			//}
			//continue
		//}

		// Минимальная средняя ошибка
		if sum < minError && epoch >= 1000 {
			minError = sum
			fmt.Println("--------- Epoch:", epoch, "\tmin avg error:", minError)
			//if epoch >= 10000 {
				//mx.CopyWeight(weight)
			//}
		}

		//
		if sum <= float64(n.Get(nn.LossLevel()).(nn.GetterSetter)) {
			break
		}
	}*/

	//
	/*file, err := os.Create("report.txt")
	if err != nil {
		os.Exit(1)
	}
	defer func() { _ = file.Close() }()*/

	//
	n.Write(
		nn.JSON("perceptron.json"),
		nn.Report(nn.File("report.txt"), input, loss, count),
		/*nn.Report(os.Stdout, input, loss, count)*/)

	n.Read(nn.JSON("perceptron.json"))
}