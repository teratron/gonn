// Package nn — meta-learning hooks.
//
// This file implements [l2-meta-learning-impl] §5.2: the ParamAccessor[T]
// interface, the two concrete wrappers (ScalarParam[T] and SliceParam[T]),
// the FeatureFunc[T] callback type, and the MetaLearner[T] orchestrator.
//
// The step method (T-12A02) is in this file too — it is the only integration
// point called from train.go after opt.Step and before fireEvent(OnIterationEnd).
package nn

import (
	"fmt"

	"github.com/teratron/gonn/pkg/utils"
)

// ParamAccessor is the uniform read/write interface for a single tunable
// hyperparameter registered with a MetaLearner. Implementations must be
// safe to call from the training loop goroutine only (no extra locking).
//
// AI-Meta:
//   - Purpose: Uniform interface for reading and writing a registered tunable parameter.
//   - Usage: Pass &ScalarParam[T]{...} or &SliceParam[T]{...} to MetaLearner.params.
//   - Related: [ScalarParam], [SliceParam], [MetaLearner].
//   - Stability: Stable.
type ParamAccessor[T utils.Float] interface {
	Get() []T
	Set([]T) error
	Name() string
}

// ScalarParam wraps a single *T so it satisfies ParamAccessor[T].
// Get returns a one-element copy; Set validates len==1 then writes the value.
//
// AI-Meta:
//   - Purpose: ParamAccessor adapter for scalar (*T) hyperparameters such as LearningRate.
//   - Usage: &ScalarParam[float32]{ptr: &cfg.LearningRate, name: "lr"}.
//   - Related: [ParamAccessor], [SliceParam], [MetaLearner].
//   - Stability: Stable.
type ScalarParam[T utils.Float] struct {
	ptr  *T
	name string
}

// Get returns the current value as a one-element slice.
func (p *ScalarParam[T]) Get() []T { return []T{*p.ptr} }

// Set validates that v has exactly one element, then writes it.
func (p *ScalarParam[T]) Set(v []T) error {
	if len(v) != 1 {
		return fmt.Errorf("ScalarParam %q: expected 1 element, got %d: %w",
			p.name, len(v), utils.ErrMetaLearnerShape)
	}
	*p.ptr = v[0]
	return nil
}

// Name returns the stable identifier used in log messages.
func (p *ScalarParam[T]) Name() string { return p.name }

// SliceParam wraps a *[]T so it satisfies ParamAccessor[T].
// Get returns a defensive copy; Set validates that len matches then copies in.
//
// AI-Meta:
//   - Purpose: ParamAccessor adapter for slice (*[]T) hyperparameters such as per-layer scales.
//   - Usage: &SliceParam[float32]{ptr: &someSlice, name: "scales"}.
//   - Related: [ParamAccessor], [ScalarParam], [MetaLearner].
//   - Stability: Stable.
type SliceParam[T utils.Float] struct {
	ptr  *[]T
	name string
}

// Get returns a copy of the current slice.
func (p *SliceParam[T]) Get() []T {
	out := make([]T, len(*p.ptr))
	copy(out, *p.ptr)
	return out
}

// Set validates that v matches the current slice length, then copies element-wise.
func (p *SliceParam[T]) Set(v []T) error {
	if len(v) != len(*p.ptr) {
		return fmt.Errorf("SliceParam %q: expected %d elements, got %d: %w",
			p.name, len(*p.ptr), len(v), utils.ErrMetaLearnerShape)
	}
	copy(*p.ptr, v)
	return nil
}

// Name returns the stable identifier used in log messages.
func (p *SliceParam[T]) Name() string { return p.name }

// FeatureFunc is the signature of a function that converts current training
// state into a feature vector consumed by the inner network. The default is
// DefaultFeatureFunc[T].
//
// AI-Meta:
//   - Purpose: Callback type for constructing the inner-network feature vector from training state.
//   - Usage: Set MetaLearner.Features to override the default two-element vector.
//   - Related: [DefaultFeatureFunc], [MetaLearner].
//   - Stability: Stable.
type FeatureFunc[T utils.Float] func(loss T, iter int, maxIter int) []T

// DefaultFeatureFunc returns the two-element feature vector [loss, iter/maxIter]
// used by MetaLearner when Features is nil. The normalised progress term
// T(iter)/T(maxIter) gives the inner network a horizon signal.
//
// AI-Meta:
//   - Purpose: Default feature builder: normalized training progress plus current loss.
//   - Usage: Assigned automatically when MetaLearner.Features is nil at step() time.
//   - Related: [FeatureFunc], [MetaLearner].
//   - Stability: Stable.
func DefaultFeatureFunc[T utils.Float](loss T, iter int, maxIter int) []T {
	return []T{loss, T(iter) / T(maxIter)}
}

// MetaLearner wires an inner *NN[T] into the outer training loop so that
// hyperparameters registered via params are updated each iteration by the
// inner network's inference output rather than held as static constants.
//
// Typical setup:
//
//	inner, _ := nn.New[float32](nn.WithInputSize(2), nn.WithOutputSize(1), ...)
//	ml := &nn.MetaLearner[float32]{inner: inner, params: []nn.ParamAccessor[float32]{
//	    &nn.ScalarParam[float32]{ptr: &outerCfg.LearningRate, name: "lr"},
//	}}
//	outer, _ := nn.New[float32](..., nn.WithMetaLearner(ml))
//
// AI-Meta:
//   - Purpose: Recursive self-optimization — an inner NN tunes the outer NN's hyperparameters.
//   - Usage: Construct and attach via WithMetaLearner; inner and params must be set before Compile.
//   - Concurrency: NotSafe; step() is called from the single-goroutine training loop only.
//   - Related: [ParamAccessor], [ScalarParam], [SliceParam], [WithMetaLearner].
//   - Stability: Stable.
type MetaLearner[T utils.Float] struct {
	inner    *NN[T]
	Features FeatureFunc[T]
	params   []ParamAccessor[T]
}

// step executes one meta-learning iteration. It is called from train.go after
// opt.Step and before fireEvent(OnIterationEnd). Errors from Set are advisory:
// all params receive their new value before the first error is returned
// (continue-on-error semantics per l2-meta-learning-impl §5.3).
func (m *MetaLearner[T]) step(loss T, iter, maxIter int) error {
	fn := m.Features
	if fn == nil {
		fn = DefaultFeatureFunc[T]
	}

	features := fn(loss, iter, maxIter)

	output, err := m.inner.Query(features)
	if err != nil {
		return fmt.Errorf("meta-learner: inner query failed: %w", err)
	}

	if len(output) != len(m.params) {
		return fmt.Errorf("meta-learner: inner output %d != %d params: %w",
			len(output), len(m.params), utils.ErrMetaLearnerShape)
	}

	var firstErr error
	for i, acc := range m.params {
		if setErr := acc.Set([]T{output[i]}); setErr != nil && firstErr == nil {
			firstErr = setErr
		}
	}
	return firstErr
}
