package main

import (
	"fmt"
	"github.com/zigenzoog/gonn/nn"
	_ "image/color"
)

/*type Parent struct {
	child Child
}
type Child struct {
	num int
}

func (p Parent) print() {
	fmt.Println(p.child.num)
}
func (c Child) print() {
	fmt.Println(c.num + 1)
}*/

//type Layer []Layer

func main() {
	/*parent := Parent{Child{2}}
	defer parent.print()*/

	/*var i interface{}
	fmt.Printf("(%v, %T)\n", i, i)
	i = 0.42
	fmt.Printf("(%v, %T)\n", i, i)
	b := i.(float64)
	fmt.Printf("(%v, %T)\n", b, b)*/

	/*var mx nn.Matrix
	mx.Bias = 0.1
	fmt.Println(mx.Bias)
	a := nn.Activation{Mode: nn.TANH}
	fmt.Println(a, a.Get(.2))

	neu := nn.Neuron{
		X:	1,
		Y:	0,
		Value:	0.1,
		N:	&nn.PrevNeuronLayer{1, .2, .3},
		W:	&nn.PrevWeightLayer{.3, 52},
	}
	fmt.Println(neu, *neu.N, *neu.W)*/

	/*mx.Neuron  = make([][]nn.Neuron, 1)
	mx.Neuron[0] = make([]nn.Neuron, 1)
	mx.Neuron[0][0].Value = 0.02
	fmt.Println(mx.Neuron[0][0].Get())*/

	/*rgba := color.RGBA{0,0,0,255}
	fmt.Println(rgba)

	clr := color.Color(rgba)
	fmt.Println(clr)*/

	/*var mx nn.Matrix
	mx.Axon = make([]nn.Axon, 1)
	mx.Axon[0].Synapse = make(map[string]nn.Neuroner)

	a := nn.Bias(4)
	fmt.Printf("%T %v\n", a, a)
	mx.Axon[0].Synapse["bias"] = &a

	if b, ok := mx.Axon[0].Synapse["bias"]; ok && *b.(*nn.Bias) == 4 {
		fmt.Printf("%T, %v\n", b, *b.(*nn.Bias))
	}

	n := nn.Neuron{Value: .5}
	fmt.Printf("%T %v\n", n, n)

	mx.Axon[0].Synapse["input"] = n
	fmt.Printf("%T %v\n", mx.Axon[0].Synapse["input"], mx.Axon[0].Synapse["input"])

	c := nn.Axon{
		Synapse: map[string]nn.Neuroner{
			"input": n,
			"bias": &a,
		},
	}
	fmt.Printf("%T %v\n", c, c)

	in := c.Synapse["input"]
	fmt.Printf("%T %v\n", in, in.(*nn.Neuron).Value) // method
	//fmt.Printf("%T %v\n", in, in.(nn.Neuron).Value) // struct {Neuroner}*/

	/*var mx nn.Matrix
	fmt.Println(mx.IsInit)*/

	// Neural Network
	//n := nn.New()
	n := nn.New().Perceptron()
	fmt.Println(n)
	//n := nn.New().RadialBasis()

	// Common initializing
	n.Set() //empty set
	n.Set(nn.Bias(true),
		  nn.Rate(.33),
		  nn.HiddenLayer(1, 5, 9)) //set

	// Bias
	fmt.Println(n.Bias()) //get
	fmt.Println(n.GetBias()) //get
	fmt.Println(n.Rate()) //get

	n.SetBias(false) //set
	fmt.Println(n.Bias())

	// Rate

	fmt.Println(n.Get()) //get
	fmt.Println("+",n.Get(n.Bias())) //get

	// Hidden layers
	n.Set(nn.HiddenLayer(3, 5, 7))
	fmt.Println(n.GetHidden()) //get
	fmt.Println("n.Get(nn.HiddenLayer()):", n.Get(nn.HiddenLayer())) //get
	n.SetHidden(7, 5, 3)
	fmt.Println(n.GetHidden()) //get
}
