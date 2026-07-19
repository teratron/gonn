// Package axon — weighted directed connection between neural cells.
//
// An Axon stores the synaptic weight and references to both endpoints:
// the incoming Nucleus (whose value feeds the forward pass) and the
// outgoing Neuron (whose miss feeds the backward pass). Bundles of axons
// are owned by Dense / Hidden / Output cells.
package axon

import (
	"sync"

	"math/rand/v2"

	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/utils"
)

// Bundle is the standard collection used by every neuron-side cell to
// hold its incoming axons. Slice rather than map — order is meaningful
// for forward-pass determinism and matches the layer's cell order.
//
// AI-Meta:
//   - Purpose: Ordered slice of incoming axons owned by a Dense or Output cell.
//   - Usage: Iterate for forward pass: for _, a := range cell.Axons { sum += a.CalculateValue() }.
//   - Related: [Axon].
type Bundle[T utils.Float] []*Axon[T]

// Store is contiguous weight storage shared by all axons of one dense layer.
// The network allocates one flat array per layer (row-major [outCells][fanIn])
// and binds every axon to a slot in it, which makes the layer's weights a
// single cache-friendly run that compute backends can consume directly
// (structure-of-arrays layout).
//
// The indirection is deliberately a pointer to the STORE, not a pointer to a
// weight: a topology change reallocates W in place, so every bound axon keeps
// resolving correctly instead of silently reading freed memory.
//
// AI-Meta:
//   - Purpose: Contiguous per-layer weight storage backing a bundle of axons.
//   - Concurrency: NotSafe; the owning Network serialises access.
//   - Related: [Axon.Bind], [Axon.W], [Bundle].
//   - Stability: Stable.
type Store[T utils.Float] struct {
	W []T
}

// Len reports the number of weight slots; nil-safe.
//
// AI-Meta:
//   - Purpose: Nil-safe slot count for a weight store.
//   - Related: [Store].
//   - Stability: Stable.
func (s *Store[T]) Len() int {
	if s == nil {
		return 0
	}
	return len(s.W)
}

// Axon is the directed weighted connection between an incoming Nucleus and an outgoing Neuron.
// The weight participates in both passes:
//
//   - Forward: contributes W() * Cell.Value to the outgoing cell sum.
//   - Backward: receives a gradient and updates the weight in place.
//
// The weight itself lives in the layer's [Store] once the axon is bound by
// Network.Build — the axon is then a VIEW (store + index) rather than the
// owner. Axons constructed outside a network keep a private weight, so the
// type stays usable standalone.
//
// OutgoingCell is required for backprop because the per-axon miss flows
// from the cell on the receiving side of the connection. Storing both
// endpoints costs one pointer per axon — the alternative (re-deriving
// the outgoing cell from a reverse adjacency map) is hot-path expensive.
//
// AI-Meta:
//   - Purpose: Weighted synapse connecting an incoming cell (forward) and an outgoing cell (backward).
//   - Concurrency: NotSafe; CalculateWeight mutates the weight in-place.
//   - Related: [Bundle], [New], [NewWithWeight], [Store], [Axon.Bind].
type Axon[T utils.Float] struct {
	Cell         neuron.Nucleus[T] `json:"-" xml:"-"`
	OutgoingCell neuron.Neuron[T]  `json:"-" xml:"-"`
	// store is non-nil once Bind attaches this axon to layer storage; idx is
	// the slot within store.W. When store is nil the weight lives in `weight`.
	store  *Store[T]
	idx    int
	weight T
}

// W returns the synaptic weight, reading through the layer store when bound.
//
// AI-Meta:
//   - Purpose: Read the synaptic weight regardless of whether the axon is store-backed.
//   - Concurrency: ReadSafe when no writer is active.
//   - Related: [Axon.SetW], [Axon.Bind].
//   - Stability: Stable.
func (a *Axon[T]) W() T {
	if a.store != nil {
		return a.store.W[a.idx]
	}
	return a.weight
}

// SetW writes the synaptic weight through to the layer store when bound.
//
// AI-Meta:
//   - Purpose: Write the synaptic weight regardless of whether the axon is store-backed.
//   - Concurrency: NotSafe.
//   - Related: [Axon.W], [Axon.Bind].
//   - Stability: Stable.
func (a *Axon[T]) SetW(value T) {
	if a.store != nil {
		a.store.W[a.idx] = value
		return
	}
	a.weight = value
}

// Bind attaches the axon to slot idx of s, copying the axon's CURRENT weight
// into that slot. Re-binding an already-bound axon (topology rebuild) carries
// the live value across, so learned weights survive storage reallocation.
//
// AI-Meta:
//   - Purpose: Attach an axon to contiguous layer storage, preserving its current weight.
//   - Concurrency: NotSafe; called during Build / topology rebuild only.
//   - Related: [Store], [Axon.W], [Axon.Unbind].
//   - Stability: Stable.
func (a *Axon[T]) Bind(s *Store[T], idx int) {
	s.W[idx] = a.W()
	a.store = s
	a.idx = idx
}

// Unbind detaches the axon from layer storage, copying the current weight back
// into the axon's private slot.
//
// AI-Meta:
//   - Purpose: Detach an axon from layer storage while preserving its weight.
//   - Related: [Axon.Bind].
//   - Stability: Stable.
func (a *Axon[T]) Unbind() {
	a.weight = a.W()
	a.store = nil
	a.idx = 0
}

// rng is the package-shared deterministic source used by New when the
// caller does not supply a pre-computed weight. Seeded from wall-clock
// on first use via utils.NewRNG(0). The mutex guards concurrent New
// invocations — the network builder is single-goroutine in practice but
// callers from tests and tooling may call New concurrently.
var (
	rngOnce sync.Once
	rng     *rand.Rand
	rngMu   sync.Mutex
)

func ensureRNG() {
	rngOnce.Do(func() {
		rng, _ = utils.NewRNG(0)
	})
}

// sampleDefaultWeight draws one weight from U[-0.5, 0.5] using the package
// PCG. Exposed only via New — callers that need deterministic seeding
// should use NewWithWeight instead and own their RNG explicitly.
func sampleDefaultWeight[T utils.Float]() T {
	ensureRNG()
	rngMu.Lock()
	defer rngMu.Unlock()
	return T(rng.Float64() - 0.5)
}

// New constructs an Axon connecting incoming → outgoing with a default
// weight drawn from U[-0.5, 0.5]. Callers that need reproducible weights
// (Xavier/He sampling) should call NewWithWeight with a pre-computed value.
//
// AI-Meta:
//   - Purpose: Construct an axon with a random default weight; use for quick wiring.
//   - Usage: a := axon.New[float32](inCell, outCell).
//   - Related: [NewWithWeight], [Axon].
func New[T utils.Float](incoming neuron.Nucleus[T], outgoing neuron.Neuron[T]) *Axon[T] {
	return &Axon[T]{
		weight:       sampleDefaultWeight[T](),
		Cell:         incoming,
		OutgoingCell: outgoing,
	}
}

// NewWithWeight constructs an Axon with a caller-supplied weight — used when
// a layer constructor has sampled from Xavier/He/Uniform with its own RNG.
//
// AI-Meta:
//   - Purpose: Construct an axon with a pre-computed deterministic weight.
//   - Usage: a := axon.NewWithWeight[float32](w, inCell, outCell).
//   - Related: [New], [Axon].
func NewWithWeight[T utils.Float](weight T, incoming neuron.Nucleus[T], outgoing neuron.Neuron[T]) *Axon[T] {
	return &Axon[T]{
		weight:       weight,
		Cell:         incoming,
		OutgoingCell: outgoing,
	}
}

// CalculateValue (FORWARD) returns W() * Cell.Value — the axon's contribution to the forward sum.
//
// AI-Meta:
//   - Purpose: Compute this axon's weighted contribution during the forward pass.
//   - Concurrency: Safe for concurrent reads; does not mutate any field.
//   - Related: [CalculateMiss], [CalculateWeight].
func (a *Axon[T]) CalculateValue() T {
	return *a.Cell.GetValue() * a.W()
}

// CalculateMiss (BACKWARD) returns OutgoingCell.Miss * W() — the error signal propagated back.
//
// AI-Meta:
//   - Purpose: Compute this axon's error contribution during the backward pass.
//   - Concurrency: Safe for concurrent reads; does not mutate any field.
//   - Related: [CalculateValue], [CalculateWeight].
func (a *Axon[T]) CalculateMiss() T {
	return *a.OutgoingCell.GetMiss() * a.W()
}

// CalculateWeight (BACKWARD) updates the weight in-place: w += *gradient * Cell.Value.
// The gradient (rate * miss) is computed and supplied by the owning cell.
//
// AI-Meta:
//   - Purpose: Apply one gradient-descent weight update for this axon.
//   - Concurrency: NotSafe; mutates the weight in-place.
//   - Related: [CalculateValue], [CalculateMiss].
func (a *Axon[T]) CalculateWeight(gradient *T) {
	a.SetW(a.W() + *gradient**a.Cell.GetValue())
}
