// Package compute — registry behaviour tests.
//
// Direct tests live in this package (not pkg/compute/cpu) so coverage
// counters bind to compute/registry.go. The CPU backend's tests cover
// happy-path registration; here we exercise the unknown-name branch,
// the sort helper, and re-registration semantics for stub injection.
package compute

import (
	"errors"
	"slices"
	"testing"

	"github.com/teratron/gonn/pkg/utils"
)

type stubBackend[T utils.Float] struct{ name string }

func (s stubBackend[T]) Name() string                                    { return s.name }
func (s stubBackend[T]) Forward(LayerHandle[T], []T) ([]T, error)        { return nil, nil }
func (s stubBackend[T]) Backward(LayerHandle[T], []T) ([]T, error)       { return nil, nil }
func (s stubBackend[T]) UpdateWeights(LayerHandle[T], []T, []T, T) error { return nil }
func (s stubBackend[T]) Allocate(int) (Buffer[T], error)                 { return Buffer[T]{}, nil }
func (s stubBackend[T]) Free(Buffer[T]) error                            { return nil }

func TestRegisterAndGetF32(t *testing.T) {
	Register("stub32", func() Backend[float32] {
		return stubBackend[float32]{name: "stub32"}
	})
	got, err := Get[float32]("stub32")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got.Name() != "stub32" {
		t.Errorf("Name = %q", got.Name())
	}
}

func TestRegisterAndGetF64(t *testing.T) {
	Register("stub64", func() Backend[float64] {
		return stubBackend[float64]{name: "stub64"}
	})
	got, err := Get[float64]("stub64")
	if err != nil {
		t.Fatalf("Get f64: %v", err)
	}
	if got.Name() != "stub64" {
		t.Errorf("Name = %q", got.Name())
	}
}

func TestGetUnknownName(t *testing.T) {
	_, err := Get[float32]("does-not-exist")
	if err == nil || !errors.Is(err, utils.ErrUserConfig) {
		t.Errorf("expected ErrUserConfig, got %v", err)
	}
	_, err = Get[float64]("does-not-exist")
	if err == nil || !errors.Is(err, utils.ErrUserConfig) {
		t.Errorf("expected ErrUserConfig f64, got %v", err)
	}
}

func TestNamesSorted(t *testing.T) {
	Register("zzz", func() Backend[float32] { return stubBackend[float32]{name: "zzz"} })
	Register("aaa", func() Backend[float32] { return stubBackend[float32]{name: "aaa"} })
	names := Names[float32]()
	if !slices.IsSorted(names) {
		t.Errorf("Names not sorted: %v", names)
	}
	names64 := Names[float64]()
	if !slices.IsSorted(names64) {
		t.Errorf("Names f64 not sorted: %v", names64)
	}
}

func TestReRegisterOverwrites(t *testing.T) {
	Register("dup", func() Backend[float32] { return stubBackend[float32]{name: "first"} })
	Register("dup", func() Backend[float32] { return stubBackend[float32]{name: "second"} })
	got, err := Get[float32]("dup")
	if err != nil {
		t.Fatal(err)
	}
	if got.Name() != "second" {
		t.Errorf("expected second factory, got %q", got.Name())
	}
}

func TestSortStringsTinySort(t *testing.T) {
	xs := []string{"c", "a", "b"}
	sortStrings(xs)
	if xs[0] != "a" || xs[1] != "b" || xs[2] != "c" {
		t.Errorf("sortStrings produced %v", xs)
	}
}
