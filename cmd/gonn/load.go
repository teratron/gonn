package main

import (
	"github.com/teratron/gonn/pkg/nn"
	"github.com/teratron/gonn/pkg/persistence"
	"github.com/teratron/gonn/pkg/utils"
)

// loadNetwork reads a config.json (and optionally a weights.json) from disk
// and returns a compiled, weight-installed *nn.NN[T] ready for inference or
// further training. When weightsPath is empty the returned network has
// randomly-initialised weights (suitable for training from scratch).
//
// Thin wrapper over nn.Load — the CLI additionally surfaces the ConfigDoc
// so subcommands can size CSV parsing without re-deriving the topology.
func loadNetwork[T utils.Float](cfgPath, weightsPath string) (*nn.NN[T], persistence.ConfigDoc[T], error) {
	doc, err := persistence.ReadConfig[T](cfgPath)
	if err != nil {
		return nil, doc, err
	}
	n, err := nn.Load[T](cfgPath, weightsPath)
	if err != nil {
		return nil, doc, err
	}
	return n, doc, nil
}

// saveWeights extracts trained weights from n and writes them to path. The
// cfg parameter is kept for call-site compatibility; the network's cached
// config document (captured by nn.Load) drives the hash chain.
func saveWeights[T utils.Float](path string, _ persistence.ConfigDoc[T], n *nn.NN[T]) error {
	return n.Save("", path)
}
