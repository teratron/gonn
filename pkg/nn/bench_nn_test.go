// Package nn — benchmark suite (PERF-1).
//
// Naming follows [l2-perf-impl] §5: Benchmark<Op>_<Topology>_<DType>.
// Run via:
//
//	go test -bench=. -benchmem -count=5 ./pkg/nn/
package nn

import (
	"github.com/teratron/gonn/pkg/utils"
	"testing"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
)

// xorDataset returns the canonical 4-sample XOR training set used by
// the public-facade benchmarks below.
func xorDataset[T utils.Float]() []Sample[T] {
	return []Sample[T]{
		{Input: []T{0, 0}, Target: []T{0}},
		{Input: []T{0, 1}, Target: []T{1}},
		{Input: []T{1, 0}, Target: []T{1}},
		{Input: []T{1, 1}, Target: []T{0}},
	}
}

// newXORNetwork constructs the canonical XOR topology: 2 → 4 (Sigmoid)
// → 1 (Sigmoid). MustNew is fine because compile errors here would be
// programming bugs, not runtime conditions.
func newXORNetwork[T utils.Float]() *NN[T] {
	return MustNew[T](
		WithInput[T](2),
		WithHiddenLayer[T](4, activation.SIGMOID),
		WithOutput[T](1, activation.SIGMOID),
		WithLearningRate[T](0.3),
		WithLoss[T](loss.MSE),
	)
}

// BenchmarkForward_XOR_f32 measures Query() latency on a compiled XOR
// network. Allocations should be small and stable per PERF-2.
func BenchmarkForward_XOR_f32(b *testing.B) {
	nn := newXORNetwork[float32]()
	input := []float32{1, 0}
	b.ResetTimer()
	b.ReportAllocs()
	for range b.N {
		_, _ = nn.Query(input)
	}
}

// BenchmarkBackward_XOR_f32 measures one Train() pass — forward +
// backward + weight update — on the same XOR network.
func BenchmarkBackward_XOR_f32(b *testing.B) {
	nn := newXORNetwork[float32]()
	in, tgt := []float32{1, 1}, []float32{0}
	b.ResetTimer()
	b.ReportAllocs()
	for range b.N {
		_, _ = nn.Train(in, tgt)
	}
}

// BenchmarkCompile_DeepNetwork_f32 measures compile-time on the
// 2 → 4 → 1 XOR topology. Multi-hidden topologies are deferred to v0.6;
// when they land this benchmark grows accordingly.
func BenchmarkCompile_DeepNetwork_f32(b *testing.B) {
	b.ReportAllocs()
	for range b.N {
		_ = newXORNetwork[float32]()
	}
}

// BenchmarkFit_XOR_f32 runs a tiny three-epoch Fit on the XOR set so a
// single benchmark exercises the whole training loop including the
// rollback machinery.
func BenchmarkFit_XOR_f32(b *testing.B) {
	data := xorDataset[float32]()
	b.ResetTimer()
	b.ReportAllocs()
	for range b.N {
		nn := MustNew[float32](
			WithInput[float32](2),
			WithHiddenLayer[float32](4, activation.SIGMOID),
			WithOutput[float32](1, activation.SIGMOID),
			WithLearningRate[float32](0.3),
			WithLoss[float32](loss.MSE),
			WithMaxIterations[float32](3),
		)
		_, _, _ = nn.Fit(data)
	}
}
