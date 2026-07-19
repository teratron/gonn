// Package nn — FitDataset bridge tests.
package nn

import (
	"context"
	"errors"
	"io"
	"testing"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/dataset"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/utils"
)

func xorSliceDataset(t *testing.T, batch int) dataset.Dataset[float64] {
	t.Helper()
	ds, err := dataset.NewSliceDataset(
		[][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}},
		[][]float64{{0}, {1}, {1}, {0}},
		batch,
	)
	if err != nil {
		t.Fatalf("NewSliceDataset: %v", err)
	}
	return ds
}

func fitDatasetNet() *NN[float64] {
	return MustNew(
		WithInput[float64](2),
		WithHiddenLayer[float64](4, activation.SIGMOID),
		WithOutput[float64](1, activation.SIGMOID),
		WithLoss[float64](loss.MSE),
		WithLearningRate[float64](0.7),
		WithMaxIterations[float64](5000),
		WithLossLimit[float64](0.001),
		WithWeightInitSeed[float64](42),
	)
}

// TestFitDatasetConvergesOnXOR guards the audit finding that pkg/dataset
// had no consumer: FitDataset must reach the same convergence Fit does.
func TestFitDatasetConvergesOnXOR(t *testing.T) {
	n := fitDatasetNet()
	epochs, lossVal, err := n.FitDataset(context.Background(), xorSliceDataset(t, 2))
	if err != nil {
		t.Fatalf("FitDataset: %v", err)
	}
	if epochs == 0 {
		t.Fatal("FitDataset completed zero epochs")
	}
	if lossVal >= 0.01 {
		t.Errorf("final loss = %v, want < 0.01 (epochs=%d)", lossVal, epochs)
	}
	out, err := n.Query([]float64{0, 1})
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if out[0] < 0.8 {
		t.Errorf("Query(0,1) = %v, want > 0.8 after XOR training", out[0])
	}
}

// TestFitDatasetContextCancel: a cancelled context stops training and
// surfaces the context error.
func TestFitDatasetContextCancel(t *testing.T) {
	n := fitDatasetNet()
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel before the first epoch
	_, _, err := n.FitDataset(ctx, xorSliceDataset(t, 2))
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
}

// noResetDataset simulates a true stream: one pass, then ErrUnsupported.
type noResetDataset struct {
	inner dataset.Dataset[float64]
}

func (d *noResetDataset) Next(ctx context.Context) (dataset.Batch[float64], error) {
	return d.inner.Next(ctx)
}
func (d *noResetDataset) Reset(context.Context) error { return dataset.ErrUnsupported }
func (d *noResetDataset) Len() (int, bool)            { return d.inner.Len() }

// TestFitDatasetSingleEpochStream: sources that cannot rewind train for
// exactly one epoch and return cleanly.
func TestFitDatasetSingleEpochStream(t *testing.T) {
	n := fitDatasetNet()
	epochs, _, err := n.FitDataset(context.Background(), &noResetDataset{inner: xorSliceDataset(t, 2)})
	if err != nil {
		t.Fatalf("FitDataset: %v", err)
	}
	if epochs != 1 {
		t.Errorf("epochs = %d, want exactly 1 for a non-rewindable stream", epochs)
	}
}

// TestFitDatasetEmptySourceErrors: an empty epoch is a data error, not a
// silent zero-loss success.
func TestFitDatasetEmptySourceErrors(t *testing.T) {
	n := fitDatasetNet()
	empty := &noResetDataset{inner: eofDataset{}}
	_, _, err := n.FitDataset(context.Background(), empty)
	if !errors.Is(err, utils.ErrInputData) {
		t.Errorf("err = %v, want ErrInputData for an empty dataset", err)
	}
}

type eofDataset struct{}

func (eofDataset) Next(context.Context) (dataset.Batch[float64], error) {
	return dataset.Batch[float64]{}, io.EOF
}
func (eofDataset) Reset(context.Context) error { return nil }
func (eofDataset) Len() (int, bool)            { return 0, true }

// TestPrefetchClose: Close must stop the producer goroutine and be
// idempotent (audit B10: goroutine/fd leak).
func TestPrefetchClose(t *testing.T) {
	ds, err := dataset.Prefetch(xorSliceDataset(t, 2), 2)
	if err != nil {
		t.Fatalf("Prefetch: %v", err)
	}
	closer, ok := ds.(io.Closer)
	if !ok {
		t.Fatal("prefetch dataset does not implement io.Closer")
	}
	if err := closer.Close(); err != nil {
		t.Errorf("Close: %v", err)
	}
	if err := closer.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
}
