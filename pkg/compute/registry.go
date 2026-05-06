// Package compute — backend registry.
//
// Implements [l2-backend-cpu] §5.3 registration mechanism: backends
// register a name → factory mapping at package init time so callers can
// resolve a Backend[T] by string at Compile() time (COMP-2).
package compute

import (
	"sync"

	"github.com/teratron/gonn/pkg/utils"
)

// factoryF32 / factoryF64 are stored per generic specialisation. Go
// generics cannot dispatch through a single map[string]any without
// runtime type assertions, so we keep one registry per supported width.
// Both maps share registryMu.
var (
	registryMu sync.RWMutex
	factoryF32 = make(map[string]func() any)
	factoryF64 = make(map[string]func() any)
)

// Register installs factory under name for both float32 and float64
// specialisations. Backends that legitimately support only one width
// can use RegisterFloat32 / RegisterFloat64 directly. Re-registering
// the same name overwrites the prior factory — this lets tests inject
// stubs without poisoning later runs.
func Register[T utils.Float](name string, factory func() Backend[T]) {
	registryMu.Lock()
	defer registryMu.Unlock()
	var z T
	switch any(z).(type) {
	case float32:
		factoryF32[name] = func() any { return factory() }
	case float64:
		factoryF64[name] = func() any { return factory() }
	}
}

// Get resolves name into a fresh Backend[T] instance. Unknown names
// yield ErrUserConfig so callers can route via errors.Is. Each Get call
// returns a new backend value; backends are cheap to construct (no
// goroutines, no preallocated state) so caching is the caller's choice.
func Get[T utils.Float](name string) (Backend[T], error) {
	registryMu.RLock()
	defer registryMu.RUnlock()
	var z T
	var raw func() any
	var ok bool
	switch any(z).(type) {
	case float32:
		raw, ok = factoryF32[name]
	case float64:
		raw, ok = factoryF64[name]
	}
	if !ok {
		return nil, utils.Newf(utils.ErrUserConfig,
			"compute: backend %q is not registered", name)
	}
	b, castOK := raw().(Backend[T])
	if !castOK {
		return nil, utils.Newf(utils.ErrCompute,
			"compute: backend %q registered for incompatible type", name)
	}
	return b, nil
}

// Names returns the sorted list of registered backend names for the
// generic specialisation T. Useful for logging the available targets at
// startup or when reporting an unknown-backend error to the user.
func Names[T utils.Float]() []string {
	registryMu.RLock()
	defer registryMu.RUnlock()
	var z T
	var src map[string]func() any
	switch any(z).(type) {
	case float32:
		src = factoryF32
	case float64:
		src = factoryF64
	}
	names := make([]string, 0, len(src))
	for name := range src {
		names = append(names, name)
	}
	sortStrings(names)
	return names
}

// sortStrings is a tiny insertion sort kept local so the registry has
// no dependency on the sort package — keeping the hot init path lean.
func sortStrings(xs []string) {
	for i := 1; i < len(xs); i++ {
		for j := i; j > 0 && xs[j-1] > xs[j]; j-- {
			xs[j-1], xs[j] = xs[j], xs[j-1]
		}
	}
}
