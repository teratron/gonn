package main

import (
	"fmt"
	"path/filepath"
	"sync"

	"github.com/teratron/gonn/pkg/nn"
)

func main() {
	// New returns a new neural network from config.
	n := nn.New(filepath.Join("config", "perceptron.json"))

	// Input dataset.
	input := []float64{.27, .31, .52}

	// Getting the results of the trained network.
	//output := n.Query(input)
	//fmt.Println(output)
	var wg sync.WaitGroup
	wg.Add(5)
	query := func(i int) {
		fmt.Println(i, n.Query(input))
		wg.Done()
	}
	for i := 1; i <= 5; i++ {
		go query(i)
	}
	wg.Wait()
}
