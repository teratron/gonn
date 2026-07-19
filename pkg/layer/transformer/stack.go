package transformer

import (
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"slices"

	"github.com/teratron/gonn/pkg/layer"
	"github.com/teratron/gonn/pkg/layer/attention"
	"github.com/teratron/gonn/pkg/utils"
)

// Stack composes N homogeneous transformer blocks sequentially.
// All blocks share the same [TransformerConfig] but are independently
// initialised — there is no weight tying between layers (TRANS-C8).
//
// Stack holds no parameters of its own; it is a thin sequencer that fans
// [Forward], [Backward], [SetPaddingMask], and [ApplyGradSGD] across all N
// child blocks. [Mode] selects whether children are [EncoderBlock]s (Mode ==
// EncoderMode) or [DecoderBlock]s (Mode == DecoderMode).
//
// AI-Meta:
//   - Purpose: Sequential N-layer transformer stack; composes EncoderBlock or DecoderBlock children.
//   - Usage: s := transformer.NewStack[float32](cfg, transformer.EncoderMode, 6); s.Init(rng).
//   - Concurrency: NotSafe; Forward/Backward mutate child caches in-place.
//   - Related: [EncoderBlock], [DecoderBlock], [TransformerConfig], [Mode].
//   - Stability: Experimental.
type Stack[T utils.Float] struct {
	Blocks []Block[T]
	Cfg    TransformerConfig[T]
	Mode   Mode
}

// NewStack allocates a Stack of n blocks with the given cfg and mode.
// Applies the same config defaults as [NewEncoderBlock] / [NewDecoderBlock]:
// Activation → ReLU when zero (TRANS-C4); Dff → 4·Dmodel when zero (TRANS-C5).
// Call [Stack.Init] before use to Xavier-initialise all weight matrices.
//
// AI-Meta:
//   - Purpose: Allocate a Stack; weights zeroed until Init.
//   - Related: [Stack.Init], [TransformerConfig], [NewEncoderBlock], [NewDecoderBlock].
//   - Stability: Experimental.
func NewStack[T utils.Float](cfg TransformerConfig[T], mode Mode, n int) *Stack[T] {
	if cfg.Activation == 0 {
		cfg.Activation = defaultActivation
	}
	if cfg.Dff == 0 {
		cfg.Dff = 4 * cfg.Dmodel
	}
	s := &Stack[T]{
		Cfg:    cfg,
		Mode:   mode,
		Blocks: make([]Block[T], n),
	}
	for i := range s.Blocks {
		if mode == DecoderMode {
			s.Blocks[i] = NewDecoderBlock(cfg)
		} else {
			s.Blocks[i] = NewEncoderBlock(cfg)
		}
	}
	return s
}

// Init Xavier-initialises all N child blocks using independent RNG streams
// forked from rng (TRANS-C8 seed isolation — no weight tying across layers).
//
// AI-Meta:
//   - Purpose: Initialise child weights with independent seeds; must be called before Forward.
//   - Concurrency: NotSafe.
//   - Related: [Stack], [NewStack].
//   - Stability: Experimental.
func (s *Stack[T]) Init(rng *rand.Rand) {
	// Both EncoderBlock and DecoderBlock implement Init(*rand.Rand); use a
	// private interface so the Block[T] contract stays minimal.
	type initer interface{ Init(*rand.Rand) }
	for _, blk := range s.Blocks {
		s0, s1 := rng.Uint64(), rng.Uint64()
		if it, ok := blk.(initer); ok {
			it.Init(rand.New(rand.NewPCG(s0, s1)))
		}
	}
}

// SetTraining propagates the training-mode flag to every child block.
// Only blocks that implement an optional training toggle are affected;
// blocks without it silently ignore the call.
func (s *Stack[T]) SetTraining(training bool) {
	type trainingToggle interface{ SetTraining(bool) }
	for _, blk := range s.Blocks {
		if tt, ok := blk.(trainingToggle); ok {
			tt.SetTraining(training)
		}
	}
}

// Forward chains all N blocks: out₀ = x; outᵢ = blocks[i].Forward(outᵢ₋₁).
// Shape is preserved: input and output are both [SeqLen*Dmodel] (TRANS-1).
//
// AI-Meta:
//   - Purpose: Stack forward pass; chains N blocks sequentially.
//   - Concurrency: NotSafe.
//   - Related: [Stack.Backward], [Stack.Init].
//   - Stability: Experimental.
func (s *Stack[T]) Forward(x []T) []T {
	out := x
	for _, blk := range s.Blocks {
		out = blk.Forward(out)
	}
	return out
}

// Backward chains all N blocks in reverse, accumulating parameter gradients
// in each child's GradSlots buffers. Returns ∂L/∂x.
//
// AI-Meta:
//   - Purpose: Stack backward pass; reverses the forward chain.
//   - Concurrency: NotSafe.
//   - Related: [Stack.Forward], [Stack.ApplyGradSGD].
//   - Stability: Experimental.
func (s *Stack[T]) Backward(upstream []T) []T {
	grad := upstream
	for _, v := range slices.Backward(s.Blocks) {
		grad = v.Backward(grad)
	}
	return grad
}

// GradSlots satisfies [layer.Layer]. Stack holds no parameters of its own;
// parameter gradients live inside each child block.
func (s *Stack[T]) GradSlots() (gradW, gradB []T) { return nil, nil }

// InputSize returns SeqLen * Dmodel (TRANS-1).
func (s *Stack[T]) InputSize() int { return s.Cfg.SeqLen * s.Cfg.Dmodel }

// OutputSize returns SeqLen * Dmodel (TRANS-1).
func (s *Stack[T]) OutputSize() int { return s.Cfg.SeqLen * s.Cfg.Dmodel }

// ApplyGradSGD fans the inline SGD step to every child block.
//
// AI-Meta:
//   - Purpose: Drive weight updates across all child parameter groups.
//   - Related: [Stack.Backward].
//   - Stability: Experimental.
func (s *Stack[T]) ApplyGradSGD(lr T) {
	for _, blk := range s.Blocks {
		blk.ApplyGradSGD(lr)
	}
}

// SetPaddingMask fans the padding mask to every child block (TRANS-10).
//
// AI-Meta:
//   - Purpose: Forward the padding mask to each block's inner MultiHeadAttention.
//   - Related: [attention.MaskedLayer], [EncoderBlock.SetPaddingMask].
//   - Stability: Experimental.
func (s *Stack[T]) SetPaddingMask(mask []bool) {
	for _, blk := range s.Blocks {
		blk.SetPaddingMask(mask)
	}
}

// MarshalJSON serialises the Stack to a JSON envelope per TRANS-9.
// Each child block is marshalled to a [json.RawMessage] so schema evolution
// inside individual blocks stays decoupled from the Stack envelope.
//
// AI-Meta:
//   - Purpose: JSON persistence for Stack checkpoint and round-trip.
//   - Related: [Stack.UnmarshalJSON], [TransformerConfig].
//   - Stability: Experimental.
func (s *Stack[T]) MarshalJSON() ([]byte, error) {
	rawBlocks := make([]json.RawMessage, len(s.Blocks))
	for i, blk := range s.Blocks {
		data, err := json.Marshal(blk)
		if err != nil {
			return nil, fmt.Errorf("Stack.MarshalJSON block[%d]: %w", i, err)
		}
		rawBlocks[i] = data
	}
	return json.Marshal(&struct {
		Type   string               `json:"Type"`
		Blocks []json.RawMessage    `json:"Blocks"`
		Config TransformerConfig[T] `json:"Config"`
		Mode   Mode                 `json:"Mode"`
	}{
		Type:   "transformer.Stack",
		Config: s.Cfg,
		Mode:   s.Mode,
		Blocks: rawBlocks,
	})
}

// UnmarshalJSON restores a Stack from a JSON envelope. Validates the Type tag,
// reconstructs the Cfg and Mode, then dispatches each block to
// [EncoderBlock.UnmarshalJSON] or [DecoderBlock.UnmarshalJSON] based on Mode.
//
// AI-Meta:
//   - Purpose: JSON restoration for Stack checkpoint round-trip.
//   - Related: [Stack.MarshalJSON], [NewStack].
//   - Stability: Experimental.
func (s *Stack[T]) UnmarshalJSON(data []byte) error {
	aux := &struct {
		Type   string               `json:"Type"`
		Blocks []json.RawMessage    `json:"Blocks"`
		Config TransformerConfig[T] `json:"Config"`
		Mode   Mode                 `json:"Mode"`
	}{}
	if err := json.Unmarshal(data, aux); err != nil {
		return err
	}
	if aux.Type != "transformer.Stack" {
		return fmt.Errorf("Stack.UnmarshalJSON: unexpected Type %q", aux.Type)
	}
	s.Cfg = aux.Config
	s.Mode = aux.Mode
	s.Blocks = make([]Block[T], len(aux.Blocks))
	for i, raw := range aux.Blocks {
		if s.Mode == DecoderMode {
			blk := &DecoderBlock[T]{}
			if err := json.Unmarshal(raw, blk); err != nil {
				return fmt.Errorf("Stack.UnmarshalJSON block[%d]: %w", i, err)
			}
			s.Blocks[i] = blk
		} else {
			blk := &EncoderBlock[T]{}
			if err := json.Unmarshal(raw, blk); err != nil {
				return fmt.Errorf("Stack.UnmarshalJSON block[%d]: %w", i, err)
			}
			s.Blocks[i] = blk
		}
	}
	return nil
}

// Compile-time interface assertions.
var (
	_ layer.Layer[float32]           = (*Stack[float32])(nil)
	_ layer.Layer[float64]           = (*Stack[float64])(nil)
	_ attention.MaskedLayer[float32] = (*Stack[float32])(nil)
	_ attention.MaskedLayer[float64] = (*Stack[float64])(nil)
	_ Block[float32]                 = (*Stack[float32])(nil)
	_ Block[float64]                 = (*Stack[float64])(nil)
)
