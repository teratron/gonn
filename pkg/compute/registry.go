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

// Register installs factory under name for the T specialisation. Re-registering
// the same name overwrites the prior factory, letting tests inject stubs.
//
// AI-Meta:
//   - Purpose: Add a named Backend factory to the global registry; called from init() of backend packages.
//   - Usage: compute.Register[float32]("cpu", func() compute.Backend[float32] { return &cpuBackend{} }).
//   - Concurrency: Safe; guards registry with a write lock.
//   - Related: [Get], [Names], [Backend].
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

// Get resolves name into a fresh Backend[T] instance. Unknown names yield
// ErrUserConfig. Each call returns a new value; caching is the caller's choice.
//
// AI-Meta:
//   - Purpose: Retrieve a registered backend by name at compile/init time.
//   - Errors: ErrUserConfig (unknown name), ErrCompute (incompatible type registration).
//   - Concurrency: Safe; guards registry with a read lock.
//   - Related: [Register], [Names], [Backend].
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

// Names returns the sorted list of registered backend names for the T
// specialisation. Useful for startup logging and error messages.
//
// AI-Meta:
//   - Purpose: Enumerate available backends for diagnostics or user-facing selection UI.
//   - Concurrency: Safe; guards registry with a read lock.
//   - Related: [Register], [Get].
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
