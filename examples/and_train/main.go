package main

import (
	"fmt"
	"path/filepath"

	"github.com/teratron/gonn/pkg/nn"
)

const jsonStream = `
{
	"name": "perceptron",
	"weight": [
		[
			[
				3.7148979,
				2.9444046,
				-5.72786,
				2.2840204,
				-1.6592604,
				0.33781952
			],
			[
				1.8408697,
				2.070344,
				-6.0672054,
				3.9654624,
				-2.7668004,
				2.3363395
			],
			[
				1.8098677,
				2.2063692,
				0.08325871,
				-4.959725,
				5.3901534,
				1.0965135
			]
		],
		[
			[
				2.1007898,
				6.552546,
				-5.262143,
				-1.1054513
			],
			[
				-6.4693666,
				-4.019415,
				-3.8858104,
				6.2537074
			]
		]
	]
}`

func main() {
	// New returns a new neural network from config.
	n := nn.New(filepath.Join("config", "perceptron.json"))
	//strings.NewReader(jsonStream)
	// Input dataset.
	input := []float64{.27, .31, .52}

	// Getting the results of the trained network.
	output := n.Query(input)
	fmt.Println(output)

	// Target dataset.
	target := []float64{.7, .1}

	//
	count, loss := n.AndTrain(output, target)
	fmt.Println(count, loss)
}
