// Package checkpoint — retention sweep.
//
// Implements [l2-checkpointing-impl] §5.4 (CHK-3): keeps the latest
// hotN snapshots uncompressed, gzips the next coldM, and deletes the
// rest. The sweep can be invoked imperatively (Sweep) for tests or via
// a goroutine driven by time.Ticker (StartSweeper).
package checkpoint

import (
	"context"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/teratron/gonn/pkg/utils"
)

// SweepConfig captures the retention policy. HotN snapshots remain
// uncompressed for fast resume; the next ColdM are gzipped; older
// snapshots are deleted. Zero or negative values disable the
// corresponding tier.
type SweepConfig struct {
	HotN  int
	ColdM int
}

// DefaultSweepConfig matches [l2-checkpointing-impl] §5.4 defaults
// (hotN=3, coldM=10) and is used when callers pass a zero-value
// SweepConfig to Sweep / StartSweeper.
var DefaultSweepConfig = SweepConfig{HotN: 3, ColdM: 10}

// Sweep applies the retention policy to dir once, returning the
// numbers of files kept hot, gzipped, and deleted. Errors from
// individual file operations are joined into err but do not abort the
// sweep — disk-full or permission failures should not poison the
// queue.
func Sweep(dir string, cfg SweepConfig) (hot, cold, deleted int, err error) {
	if cfg.HotN <= 0 && cfg.ColdM <= 0 {
		cfg = DefaultSweepConfig
	}
	if cfg.HotN < 0 {
		cfg.HotN = 0
	}
	if cfg.ColdM < 0 {
		cfg.ColdM = 0
	}

	entries, readErr := os.ReadDir(dir)
	if readErr != nil {
		return 0, 0, 0, utils.Wrap(utils.ErrIO, readErr, "Sweep: readdir %q", dir)
	}
	files := make([]string, 0, len(entries))
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if !strings.HasPrefix(name, "snap-") {
			continue
		}
		if !strings.HasSuffix(name, ".json") && !strings.HasSuffix(name, ".json.gz") {
			continue
		}
		files = append(files, filepath.Join(dir, name))
	}
	// Newest first by lexical order — snapshotName pads Iter so this is
	// equivalent to numeric ordering on Iter,Timestamp.
	sort.Sort(sort.Reverse(sort.StringSlice(files)))

	var errs []error
	var idx int

	for ; idx < len(files) && hot < cfg.HotN; idx++ {
		hot++
	}
	for ; idx < len(files) && cold < cfg.ColdM; idx++ {
		path := files[idx]
		if strings.HasSuffix(path, ".json.gz") {
			cold++
			continue
		}
		if _, gzErr := gzipFile(path); gzErr != nil {
			errs = append(errs, gzErr)
			continue
		}
		cold++
	}
	for ; idx < len(files); idx++ {
		if rmErr := os.Remove(files[idx]); rmErr != nil {
			errs = append(errs, utils.Wrap(utils.ErrIO, rmErr,
				"Sweep: remove %q", files[idx]))
			continue
		}
		deleted++
	}
	return hot, cold, deleted, joinErrors(errs)
}

// StartSweeper launches a goroutine that calls Sweep on dir every
// interval until ctx is cancelled. The returned Sweeper exposes Stop()
// for synchronous shutdown — Stop blocks until the running tick (if
// any) finishes, ensuring no half-applied sweep races test teardown.
func StartSweeper(ctx context.Context, dir string, cfg SweepConfig, interval time.Duration) *Sweeper {
	if interval <= 0 {
		interval = time.Minute
	}
	s := &Sweeper{stop: make(chan struct{})}
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-s.stop:
				return
			case <-ticker.C:
				_, _, _, _ = Sweep(dir, cfg)
			}
		}
	}()
	return s
}

// Sweeper is the handle returned by StartSweeper. Multiple Stop calls
// are safe; the goroutine is joined on the first call and subsequent
// calls return immediately.
type Sweeper struct {
	once sync.Once
	stop chan struct{}
	wg   sync.WaitGroup
}

// Stop signals the goroutine to exit and blocks until it has returned.
// Idempotent — second and later calls are no-ops.
func (s *Sweeper) Stop() {
	s.once.Do(func() {
		close(s.stop)
	})
	s.wg.Wait()
}

// joinErrors collapses a slice of errors into a single error or returns
// nil when the slice is empty. errors.Join would suffice but pulling it
// in here keeps the import set unchanged for callers; we re-route
// through utils.Wrap so the result still satisfies errors.Is.
func joinErrors(errs []error) error {
	if len(errs) == 0 {
		return nil
	}
	if len(errs) == 1 {
		return errs[0]
	}
	combined := errs[0].Error()
	for _, e := range errs[1:] {
		combined += "; " + e.Error()
	}
	return utils.Newf(utils.ErrIO, "Sweep: %d errors (%s)", len(errs), combined)
}
