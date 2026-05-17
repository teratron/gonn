// Package utils — error sentinels and helper constructors.
//
// This file implements [l2-errors-impl] §5.1 (sentinels) and §5.2 (helpers)
// on top of the L1 error taxonomy. Every package in the GoNN library MUST
// route its errors through one of the six orthogonal categories defined here
// so that callers can dispatch via [errors.Is] (per C32 §4).
//
// Sentinel choice rule (one category per error site):
//
//   - ErrUserConfig — API misuse before runtime: bad size, unknown method symbol.
//   - ErrInputData  — runtime validation of caller-supplied data: bad batch shape, NaN sample.
//   - ErrCompute    — numeric / hardware faults inside the engine: NaN gradient, dim mismatch.
//   - ErrControl    — training-lifecycle state-machine violations.
//   - ErrIntegrity  — persisted-artifact integrity: bad checksum, schema-version mismatch.
//   - ErrIO         — filesystem / network: permission, EOF, disk full.
package utils

import (
	"errors"
	"fmt"
	"runtime"
)

// ErrUserConfig signals a configuration mistake made by the caller before
// runtime — typically a bad argument to a Builder method or an unknown enum
// symbol. Recovery is the caller's responsibility (fix the call).
//
// AI-Meta:
//   - Purpose: Sentinel for caller-side configuration faults (bad Builder args, unknown enum).
//   - Usage: Wrap: fmt.Errorf("...: %w", ErrUserConfig); detect: errors.Is(err, ErrUserConfig).
//   - Related: [ErrInputData], [ErrCompute], [ErrControl], [ErrIntegrity], [ErrIO], [Newf].
var ErrUserConfig = errors.New("user-config")

// ErrInputData signals invalid data supplied to a runtime entry point —
// a batch with the wrong shape, a sample containing NaN, an empty stream.
// Distinct from ErrUserConfig in that the data is valid Go but violates
// the runtime contract.
//
// AI-Meta:
//   - Purpose: Sentinel for runtime data-contract violations (wrong batch shape, NaN sample).
//   - Usage: Wrap: fmt.Errorf("...: %w", ErrInputData); detect: errors.Is(err, ErrInputData).
var ErrInputData = errors.New("input-data")

// ErrCompute signals a numeric or hardware-level fault inside the engine —
// NaN gradient, infinite loss, dimension mismatch between layers, missing
// CPU feature required by a backend. Indicates the engine cannot proceed
// without operator intervention (lower learning rate, switch backend).
//
// AI-Meta:
//   - Purpose: Sentinel for numeric or hardware faults inside the engine (NaN gradient, dim mismatch).
//   - Usage: Wrap: fmt.Errorf("...: %w", ErrCompute); detect: errors.Is(err, ErrCompute).
var ErrCompute = errors.New("compute")

// ErrControl signals a training-lifecycle state-machine violation —
// Pause issued on a Stopped network, double-Resume, mutating a frozen
// snapshot. Indicates a programming error in the orchestration layer.
//
// AI-Meta:
//   - Purpose: Sentinel for training lifecycle state-machine violations (Pause on stopped network).
//   - Usage: Wrap: fmt.Errorf("...: %w", ErrControl); detect: errors.Is(err, ErrControl).
var ErrControl = errors.New("control")

// ErrIntegrity signals a persisted-artifact integrity failure — checksum
// mismatch on a checkpoint, schema-version drift, weight-count mismatch
// after deserialization. Recovery requires re-creating or migrating the
// artifact.
//
// AI-Meta:
//   - Purpose: Sentinel for persisted-artifact integrity failures (checksum mismatch, schema drift).
//   - Usage: Wrap: fmt.Errorf("...: %w", ErrIntegrity); detect: errors.Is(err, ErrIntegrity).
var ErrIntegrity = errors.New("integrity")

// ErrIO signals a filesystem or network failure — permission denied, EOF
// before expected boundary, disk full, broken pipe. Distinct from
// ErrIntegrity in that the storage medium itself failed, not the contents.
//
// AI-Meta:
//   - Purpose: Sentinel for filesystem or network failures (permission denied, disk full, EOF).
//   - Usage: Wrap: fmt.Errorf("...: %w", ErrIO); detect: errors.Is(err, ErrIO).
var ErrIO = errors.New("io")

// Dynamic-topology sentinels (DYN-1..DYN-6, per l2-dynamic-topology-impl).
// These are returned exclusively by mutation methods on Network[T] and the
// wrapping methods on NN[T].

// ErrImmutableMode is returned when a topology mutation is attempted on a
// Network whose TopologyMode is Immutable (DYN-1).
//
// AI-Meta:
//   - Purpose: Sentinel for topology mutation on an immutable network (DYN-1).
//   - Usage: errors.Is(err, utils.ErrImmutableMode).
var ErrImmutableMode = errors.New("immutable-mode")

// ErrInvalidPosition is returned when a layer position or neuron index is
// out of the valid range for the current topology.
//
// AI-Meta:
//   - Purpose: Sentinel for out-of-range topology mutation positions.
//   - Usage: errors.Is(err, utils.ErrInvalidPosition).
var ErrInvalidPosition = errors.New("invalid-position")

// ErrImmutableLayer is returned when a mutation targets a layer that may not
// be mutated (e.g., Input or Output layers).
//
// AI-Meta:
//   - Purpose: Sentinel for mutations targeting non-mutable layers (Input/Output).
//   - Usage: errors.Is(err, utils.ErrImmutableLayer).
var ErrImmutableLayer = errors.New("immutable-layer")

// ErrMinimumTopology is returned when a removal operation would reduce the
// topology below the one-hidden-layer minimum contract (DYN-3).
//
// AI-Meta:
//   - Purpose: Sentinel for removal that would violate the minimum topology contract.
//   - Usage: errors.Is(err, utils.ErrMinimumTopology).
var ErrMinimumTopology = errors.New("minimum-topology")

// ErrEmptyLayer is returned when a size-zero layer is provided to a topology
// mutation (count = 0 or resulting layer size = 0 after removal).
//
// AI-Meta:
//   - Purpose: Sentinel for zero-size layer in topology mutations.
//   - Usage: errors.Is(err, utils.ErrEmptyLayer).
var ErrEmptyLayer = errors.New("empty-layer")

// ErrMutationFailed is returned when a topology mutation fails during the
// transactional commit phase and the rollback has been applied.
//
// AI-Meta:
//   - Purpose: Sentinel for unrecoverable topology mutation failure after rollback.
//   - Usage: errors.Is(err, utils.ErrMutationFailed).
var ErrMutationFailed = errors.New("mutation-failed")

// Convolutional-layer sentinels (CONV-1, CONV-5, per l2-conv-layers-impl).

// ErrConvShapeMismatch signals that the input feature map is shorter than the
// kernel size under PadValid (CONV-1 violation: outputLen would be ≤ 0).
//
// AI-Meta:
//   - Purpose: Sentinel for convolutional-layer input shorter than kernel size in PadValid mode.
//   - Usage: errors.Is(err, utils.ErrConvShapeMismatch).
//   - Related: [ErrConvPoolSizeMismatch], [ErrCompute].
//   - Stability: Stable.
var ErrConvShapeMismatch = errors.New("conv-shape-mismatch")

// ErrConvPoolSizeMismatch signals that the pooling window is larger than the
// feature map it operates on (CONV-5 violation: poolSize > len(input)).
//
// AI-Meta:
//   - Purpose: Sentinel for pooling window larger than the feature map length.
//   - Usage: errors.Is(err, utils.ErrConvPoolSizeMismatch).
//   - Related: [ErrConvShapeMismatch], [ErrCompute].
//   - Stability: Stable.
var ErrConvPoolSizeMismatch = errors.New("conv-pool-size-mismatch")

// ErrConv2DShapeMismatch signals that a 2-D convolutional input spatial
// dimension is smaller than the kernel dimension under PadValid (CONV2D-1
// violation: outputShape would be (0, 0) on at least one axis).
//
// AI-Meta:
//   - Purpose: Sentinel for Conv2D input H or W shorter than the corresponding kernel dimension in PadValid mode.
//   - Usage: errors.Is(err, utils.ErrConv2DShapeMismatch).
//   - Related: [ErrConvShapeMismatch], [ErrConvPoolSizeMismatch], [ErrCompute].
//   - Stability: Stable.
var ErrConv2DShapeMismatch = errors.New("conv2d-shape-mismatch")

// Dataset / continuation sentinels (FMT-1, FMT-5, FMT-8, per l2-dataset-loader-impl).

// ErrIDXMagic signals that the IDX header is malformed — either the leading
// two zero bytes are missing or the dtype byte is not in the recognised set
// (FMT-1 violation).
//
// AI-Meta:
//   - Purpose: Sentinel for malformed IDX dataset header (bad magic or unknown dtype).
//   - Usage: errors.Is(err, utils.ErrIDXMagic).
//   - Related: [ErrMNISTRecordMismatch], [ErrIntegrity].
//   - Stability: Stable.
var ErrIDXMagic = errors.New("idx-magic")

// ErrMNISTRecordMismatch signals that the image IDX file and the label IDX
// file report different record counts (FMT-5 violation).
//
// AI-Meta:
//   - Purpose: Sentinel for MNIST image/label record count mismatch.
//   - Usage: errors.Is(err, utils.ErrMNISTRecordMismatch).
//   - Related: [ErrIDXMagic], [ErrIntegrity].
//   - Stability: Stable.
var ErrMNISTRecordMismatch = errors.New("mnist-record-mismatch")

// ErrNetworkRunning signals that an operation requiring the Idle training-
// lifecycle state was attempted while the network was Training, Paused,
// Stopping, or Stopped (FMT-8 / continuation-API guard).
//
// AI-Meta:
//   - Purpose: Sentinel for AndTrain / lifecycle calls issued while the network is not Idle.
//   - Usage: errors.Is(err, utils.ErrNetworkRunning).
//   - Related: [ErrControl].
//   - Stability: Stable.
var ErrNetworkRunning = errors.New("network-running")

// ErrMetaLearnerShape signals that the inner network's output length does not
// match the number of registered ParamAccessors, or that a Set call received
// a slice whose length violates the parameter's shape contract.
//
// AI-Meta:
//   - Purpose: Sentinel for inner-NN output / param-count mismatch in meta-learning hooks.
//   - Usage: errors.Is(err, utils.ErrMetaLearnerShape).
//   - Related: [ErrMetaLearnerRunning], [ErrInputData].
//   - Stability: Stable.
var ErrMetaLearnerShape = errors.New("meta-learner-shape")

// ErrMetaLearnerRunning signals that WithMetaLearner was applied while the
// outer network was in a non-Idle training state (Training, Paused, Stopping,
// or Stopped). The option is rejected; the network state is unchanged.
//
// AI-Meta:
//   - Purpose: Sentinel for WithMetaLearner called while the outer network is not Idle.
//   - Usage: errors.Is(err, utils.ErrMetaLearnerRunning).
//   - Related: [ErrMetaLearnerShape], [ErrControl].
//   - Stability: Stable.
var ErrMetaLearnerRunning = errors.New("meta-learner-running")

// ErrCallbackPanic signals that a user-supplied training callback panicked.
// The panic is recovered internally; training continues. The error wraps
// ErrControl because a panicking callback is a programming fault in the
// orchestration layer, not a data or compute fault.
//
// AI-Meta:
//   - Purpose: Sentinel for a recovered panic inside a training callback; training continues.
//   - Usage: errors.Is(err, utils.ErrCallbackPanic) to detect callback panics in logs.
//   - Related: [ErrControl], [Newf].
//   - Stability: Stable.
var ErrCallbackPanic = errors.New("callback-panic")

// Newf builds a new error that wraps the given category sentinel and
// carries the caller-supplied identifying fields. The format string MUST
// be specific per C32 §1: it MUST identify the offending value (or its
// name) and the constraint violated.
//
// Example:
//
//	return utils.Newf(utils.ErrUserConfig, "Input(): size must be positive, got %d", size)
//
// Newf panics if cat is nil — a nil category is always a programming bug.
//
// AI-Meta:
//   - Purpose: Build an error wrapping a category sentinel with a caller-formatted message.
//   - Usage: utils.Newf(utils.ErrUserConfig, "Input(): size must be positive, got %d", n).
//   - Concurrency: Safe.
//   - Related: [Wrap], [NewSizeError], [NewActivationError], [NewIntegrityError].
func Newf(cat error, format string, args ...any) error {
	if cat == nil {
		panic("utils.Newf: nil category sentinel")
	}
	return fmt.Errorf(format+": %w", append(args, cat)...)
}

// Wrap attaches an additional message to an existing error while preserving
// the original chain. The returned error satisfies errors.Is for both the
// supplied category and any sentinel already present in cause. Use Wrap
// when re-routing an stdlib or third-party error into the project taxonomy.
//
// If cause is nil, Wrap returns nil — convenient for one-line propagation.
//
// AI-Meta:
//   - Purpose: Re-route a foreign error into the project taxonomy while preserving the original chain.
//   - Usage: utils.Wrap(utils.ErrIO, err, "checkpoint.Write(): failed to flush").
//   - Concurrency: Safe.
//   - Related: [Newf], [ErrIO], [ErrUserConfig].
func Wrap(cat error, cause error, format string, args ...any) error {
	if cause == nil {
		return nil
	}
	if cat == nil {
		panic("utils.Wrap: nil category sentinel")
	}
	msg := fmt.Sprintf(format, args...)
	return fmt.Errorf("%s: %w (%w)", msg, cause, cat)
}

// NewSizeError builds an ErrUserConfig describing a positional-size
// constraint violation in API arguments. field names the parameter,
// got is the offending value, constraint is a short phrase (e.g.
// "positive", ">= 1", "in [1, 1024]").
//
// AI-Meta:
//   - Purpose: Convenience constructor for ErrUserConfig about an API size constraint violation.
//   - Usage: utils.NewSizeError("Input()", size, "positive").
//   - Related: [Newf], [ErrUserConfig].
func NewSizeError(field string, got int, constraint string) error {
	return fmt.Errorf("%s: size must be %s, got %d: %w", field, constraint, got, ErrUserConfig)
}

// NewActivationError builds an ErrUserConfig describing a request for an
// unregistered activation symbol. symbol is the unknown identifier.
//
// AI-Meta:
//   - Purpose: Convenience constructor for ErrUserConfig about an unregistered activation symbol.
//   - Usage: utils.NewActivationError("SWISH").
//   - Related: [Newf], [ErrUserConfig].
func NewActivationError(symbol string) error {
	return fmt.Errorf("activation %q is not registered: %w", symbol, ErrUserConfig)
}

// NewIntegrityError builds an ErrIntegrity describing a mismatch between
// expected and observed values in a persisted artifact. what names the
// artifact slot ("checkpoint.weights[0].len"), expected and got are the
// rendered values being compared.
//
// AI-Meta:
//   - Purpose: Convenience constructor for ErrIntegrity about a persisted-artifact value mismatch.
//   - Usage: utils.NewIntegrityError("checkpoint.weights[0].len", "512", "256").
//   - Related: [Newf], [ErrIntegrity].
func NewIntegrityError(what string, expected, got string) error {
	return fmt.Errorf("integrity violation in %s: expected %s, got %s: %w", what, expected, got, ErrIntegrity)
}

// LocationHint returns a "file:line" string for the caller skip frames
// above the LocationHint call site. skip=0 reports the caller of
// LocationHint itself. Returns the empty string when runtime info is
// unavailable. Intended for caller hints in Error-level logs.
//
// AI-Meta:
//   - Purpose: Return a file:line string to include in Error-level log messages for diagnostics.
//   - Usage: utils.Logger.Error("failed", "at", utils.LocationHint(0)).
//   - Concurrency: Safe.
func LocationHint(skip int) string {
	_, file, line, ok := runtime.Caller(skip + 1)
	if !ok {
		return ""
	}
	return fmt.Sprintf("%s:%d", trimToPackagePath(file), line)
}

// trimToPackagePath shortens an absolute file path to the segment beginning
// at the project root marker so log lines remain readable across machines.
// Falls back to the basename when no marker is found.
func trimToPackagePath(file string) string {
	const marker = "/gonn/"
	if i := indexLast(file, marker); i >= 0 {
		return file[i+len(marker):]
	}
	if i := indexLast(file, "\\gonn\\"); i >= 0 {
		return file[i+len("\\gonn\\"):]
	}
	if i := indexLast(file, "/"); i >= 0 {
		return file[i+1:]
	}
	if i := indexLast(file, "\\"); i >= 0 {
		return file[i+1:]
	}
	return file
}

// indexLast returns the byte index of the last occurrence of sep in s, or
// -1 when sep is absent. Avoids the strings package to keep this file's
// import set minimal (errors, fmt, runtime).
func indexLast(s, sep string) int {
	if len(sep) == 0 || len(s) < len(sep) {
		return -1
	}
	for i := len(s) - len(sep); i >= 0; i-- {
		if s[i:i+len(sep)] == sep {
			return i
		}
	}
	return -1
}
