package main

import "testing"

// TestLogicGatesConverge runs each truth table and asserts the trained
// network reaches a recognisable fit. AND / OR / NAND are linearly
// separable — a 2-neuron hidden layer should hit very low loss. Threshold
// stays loose to absorb random-init drift since the public API does not
// yet expose an RNG seed.
func TestLogicGatesConverge(t *testing.T) {
	cases := []gate{
		{Name: "AND", Targets: []float32{0, 0, 0, 1}},
		{Name: "OR", Targets: []float32{0, 1, 1, 1}},
		{Name: "NAND", Targets: []float32{1, 1, 1, 0}},
	}
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			loss := trainGate(c)
			if loss > 0.15 {
				t.Errorf("%s loss = %v, want < 0.15", c.Name, loss)
			}
		})
	}
}
