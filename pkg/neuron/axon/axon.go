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

// Axon is the directed weighted connection between an incoming Nucleus and an outgoing Neuron.
// Weight participates in both passes:
//
//   - Forward: contributes Weight * Cell.Value to the outgoing cell sum.
//   - Backward: receives a gradient and updates Weight in place.
//
// OutgoingCell is required for backprop because the per-axon miss flows
// from the cell on the receiving side of the connection. Storing both
// endpoints costs one pointer per axon — the alternative (re-deriving
// the outgoing cell from a reverse adjacency map) is hot-path expensive.
//
// AI-Meta:
//   - Purpose: Weighted synapse connecting an incoming cell (forward) and an outgoing cell (backward).
//   - Concurrency: NotSafe; CalculateWeight mutates Weight in-place.
//   - Related: [Bundle], [New], [NewWithWeight].
type Axon[T utils.Float] struct {
	Weight       T                 `json:"weight" xml:"weight"`
	Cell         neuron.Nucleus[T] `json:"-" xml:"-"`
	OutgoingCell neuron.Neuron[T]  `json:"-" xml:"-"`
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
		Weight:       sampleDefaultWeight[T](),
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
		Weight:       weight,
		Cell:         incoming,
		OutgoingCell: outgoing,
	}
}

// CalculateValue (FORWARD) returns Weight * Cell.Value — the axon's contribution to the forward sum.
//
// AI-Meta:
//   - Purpose: Compute this axon's weighted contribution during the forward pass.
//   - Concurrency: Safe for concurrent reads; does not mutate any field.
//   - Related: [CalculateMiss], [CalculateWeight].
func (a *Axon[T]) CalculateValue() T {
	return *a.Cell.GetValue() * a.Weight
}

// CalculateMiss (BACKWARD) returns OutgoingCell.Miss * Weight — the error signal propagated back.
//
// AI-Meta:
//   - Purpose: Compute this axon's error contribution during the backward pass.
//   - Concurrency: Safe for concurrent reads; does not mutate any field.
//   - Related: [CalculateValue], [CalculateWeight].
func (a *Axon[T]) CalculateMiss() T {
	return *a.OutgoingCell.GetMiss() * a.Weight
}

// CalculateWeight (BACKWARD) updates Weight in-place: w += *gradient * Cell.Value.
// The gradient (rate * miss) is computed and supplied by the owning cell.
//
// AI-Meta:
//   - Purpose: Apply one gradient-descent weight update for this axon.
//   - Concurrency: NotSafe; mutates Weight in-place.
//   - Related: [CalculateValue], [CalculateMiss].
func (a *Axon[T]) CalculateWeight(gradient *T) {
	a.Weight += *gradient * *a.Cell.GetValue()
}
