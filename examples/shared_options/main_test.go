package main

import (
	"testing"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/nn"
)

// TestSharedOptsTrainBothTopologies asserts the same []Option[T] slice
// drives convergence in two distinct topologies. Loose threshold matches
// the broader Phase 4 v0.1 stance: random init is the dominant variable
// because the public API does not yet expose an RNG seed.
func TestSharedOptsTrainBothTopologies(t *testing.T) {
	common := commonOpts()

	resA := train("topology-A", append(append([]nn.Option[float32]{}, common...),
		nn.WithHiddenLayer[float32](8, activation.ReLU),
		nn.WithOutput[float32](1, activation.SIGMOID),
	))
	if resA.loss > 0.15 {
		t.Errorf("topology-A loss = %v, want < 0.15", resA.loss)
	}

	resB := train("topology-B", append(append([]nn.Option[float32]{}, common...),
		nn.WithHiddenLayer[float32](16, activation.ReLU),
		nn.WithOutput[float32](1, activation.SIGMOID),
	))
	if resB.loss > 0.15 {
		t.Errorf("topology-B loss = %v, want < 0.15", resB.loss)
	}
}
