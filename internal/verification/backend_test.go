package verification

import (
	"testing"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/compute"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/nn"
	"github.com/teratron/gonn/pkg/utils"
)

// backendNet builds the deterministic net used by every backend test.
func backendNet(t *testing.T, opts ...nn.Option[float64]) *nn.NN[float64] {
	t.Helper()
	base := []nn.Option[float64]{
		nn.WithInput[float64](2),
		nn.WithBias[float64](true),
		nn.WithHiddenLayer[float64](5, activation.SIGMOID),
		nn.WithHiddenLayer[float64](4, activation.TanH),
		nn.WithOutput[float64](1, activation.SIGMOID),
		nn.WithLoss[float64](loss.MSE),
		nn.WithLearningRate[float64](0.4),
		nn.WithMaxIterations[float64](120),
		nn.WithLossLimit[float64](-1), // never stop early: both runs must do equal work
		nn.WithWeightInitSeed[float64](7),
	}
	return nn.MustNew(append(base, opts...)...)
}

// TestBackendKernelsLiveByDefault guards the audit finding that the resolved
// compute backend was stored and never consulted: after a default compile the
// CPU backend's dense kernels must actually be driving the passes.
func TestBackendKernelsLiveByDefault(t *testing.T) {
	n := backendNet(t)
	if !n.Network.KernelsActive() {
		t.Error("compute kernels inactive after default compile — WithBackend is decorative again")
	}
}

// TestBackendParityWithReferenceLoops: delegating the matrix primitives to the
// backend must not change a single weight relative to the engine's internal
// reference loops. Same seed, same samples, same epoch count → same net.
func TestBackendParityWithReferenceLoops(t *testing.T) {
	withKernels := backendNet(t)
	if !withKernels.Network.KernelsActive() {
		t.Fatal("precondition: kernels must be active for the accelerated run")
	}

	reference := backendNet(t)
	reference.Network.SetBackend(nil) // force the internal reference path
	if reference.Network.KernelsActive() {
		t.Fatal("SetBackend(nil) must disable the accelerated path")
	}

	samples := xorSamples()
	if _, _, err := withKernels.Fit(samples); err != nil {
		t.Fatalf("Fit (kernels): %v", err)
	}
	if _, _, err := reference.Fit(samples); err != nil {
		t.Fatalf("Fit (reference): %v", err)
	}

	a := withKernels.Network.FlatWeights()
	b := reference.Network.FlatWeights()
	if len(a) != len(b) {
		t.Fatalf("weight count differs: kernels %d, reference %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("weight %d diverged: kernels %.17g, reference %.17g "+
				"(the delegated kernels are not the same math)", i, a[i], b[i])
		}
	}
}

// noKernelBackend implements compute.Backend but deliberately NOT
// compute.DenseKernels — the "backend that cannot accelerate" case.
type noKernelBackend[T utils.Float] struct{}

func (noKernelBackend[T]) Name() string { return "nokernel" }
func (noKernelBackend[T]) Forward(compute.LayerHandle[T], []T) ([]T, error) {
	return nil, nil
}

func (noKernelBackend[T]) Backward(compute.LayerHandle[T], []T) ([]T, error) {
	return nil, nil
}
func (noKernelBackend[T]) UpdateWeights(compute.LayerHandle[T], []T, []T, T) error { return nil }
func (noKernelBackend[T]) Allocate(size int) (compute.Buffer[T], error) {
	return compute.Buffer[T]{Data: make([]T, size)}, nil
}
func (noKernelBackend[T]) Free(compute.Buffer[T]) error { return nil }

// TestBackendWithoutKernelsStillTrains: a backend that does not implement the
// optional kernel interface must degrade to the reference loops rather than
// silently skipping the math or failing the compile.
func TestBackendWithoutKernelsStillTrains(t *testing.T) {
	n := backendNet(t, nn.WithBackend[float64](noKernelBackend[float64]{}))
	if n.Network.KernelsActive() {
		t.Error("a backend without DenseKernels must not report an active accelerated path")
	}
	_, finalLoss, err := n.Fit(xorSamples())
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if finalLoss >= 0.25 {
		t.Errorf("final loss %.4f — network did not train on the fallback path", finalLoss)
	}
}

// brokenKernelBackend implements DenseKernels but every kernel fails. It exists
// to prove the engine notices and falls back instead of training on garbage.
type brokenKernelBackend[T utils.Float] struct{ noKernelBackend[T] }

func (brokenKernelBackend[T]) MatVec(compute.DenseMatrix[T], []T, []T) error {
	return utils.Newf(utils.ErrCompute, "brokenKernelBackend: MatVec always fails")
}

func (brokenKernelBackend[T]) MatVecT(compute.DenseMatrix[T], []T, []T) error {
	return utils.Newf(utils.ErrCompute, "brokenKernelBackend: MatVecT always fails")
}

func (brokenKernelBackend[T]) GradOuter(compute.DenseMatrix[T], []T, []T, []T, T) error {
	return utils.Newf(utils.ErrCompute, "brokenKernelBackend: GradOuter always fails")
}

// TestBrokenKernelFallsBackAndStillTrains: when a backend kernel errors the
// engine must disable the accelerated path and finish the run on correct math —
// a failing accelerator must never become silently wrong training.
func TestBrokenKernelFallsBackAndStillTrains(t *testing.T) {
	n := backendNet(t, nn.WithBackend[float64](brokenKernelBackend[float64]{}))
	if !n.Network.KernelsActive() {
		t.Fatal("precondition: the broken backend does implement DenseKernels")
	}
	_, finalLoss, err := n.Fit(xorSamples())
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if n.Network.KernelsActive() {
		t.Error("engine kept using kernels after they reported errors")
	}
	if finalLoss >= 0.25 {
		t.Errorf("final loss %.4f — fallback path did not train correctly", finalLoss)
	}

	// And the fallback result must equal a run that never had kernels at all.
	reference := backendNet(t)
	reference.Network.SetBackend(nil)
	if _, _, err := reference.Fit(xorSamples()); err != nil {
		t.Fatalf("Fit (reference): %v", err)
	}
	a, b := n.Network.FlatWeights(), reference.Network.FlatWeights()
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("weight %d: fallback %.17g != reference %.17g", i, a[i], b[i])
		}
	}
}
