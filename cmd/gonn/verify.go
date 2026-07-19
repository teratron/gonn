package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/teratron/gonn/pkg/utils"
)

// verifyCmd implements the "verify" subcommand. It loads a compiled network,
// runs a forward pass + loss computation over the dataset (no weight update),
// and reports the mean loss.
func verifyCmd(args []string) int {
	fs := flag.NewFlagSet("verify", flag.ContinueOnError)
	cfgPath := fs.String("config", "", "path to config.json (required)")
	weightsPath := fs.String("weights", "", "path to weights.json (required)")
	dataPath := fs.String("data", "", "path to evaluation CSV file (required)")
	precision := fs.String("precision", "float32", "numeric precision: float32 or float64")
	jsonOut := fs.Bool("json", false, "machine-readable JSON output")

	if err := fs.Parse(args); err != nil {
		fmt.Fprintf(os.Stderr, "gonn verify: %v\n", err)
		return exitUserConfig
	}
	if *cfgPath == "" || *weightsPath == "" || *dataPath == "" {
		fmt.Fprintln(os.Stderr, "gonn verify: --config, --weights, and --data are required")
		fs.Usage()
		return exitUserConfig
	}

	switch *precision {
	case "float64":
		return runVerify[float64](*cfgPath, *weightsPath, *dataPath, *jsonOut)
	case "float32":
		return runVerify[float32](*cfgPath, *weightsPath, *dataPath, *jsonOut)
	default:
		fmt.Fprintf(os.Stderr, "gonn verify: unsupported --precision %q (float32|float64)\n", *precision)
		return exitUnsupported
	}
}

func runVerify[T utils.Float](cfgPath, weightsPath, dataPath string, jsonOut bool) int {
	n, doc, err := loadNetwork[T](cfgPath, weightsPath)
	if err != nil {
		if jsonOut {
			printErrorJSON(err)
		} else {
			fmt.Fprintf(os.Stderr, "gonn verify: load: %v\n", err)
		}
		return exitCode(err)
	}

	samples, err := loadSamples[T](dataPath, int(doc.InputSize), int(doc.Output.Size))
	if err != nil {
		if jsonOut {
			printErrorJSON(err)
		} else {
			fmt.Fprintf(os.Stderr, "gonn verify: data: %v\n", err)
		}
		return exitCode(err)
	}

	var totalLoss T
	for _, s := range samples {
		l, err := n.Verify(s.Input, s.Target)
		if err != nil {
			if jsonOut {
				printErrorJSON(err)
			} else {
				fmt.Fprintf(os.Stderr, "gonn verify: %v\n", err)
			}
			return exitCode(err)
		}
		totalLoss += l
	}

	var meanLoss T
	if len(samples) > 0 {
		meanLoss = totalLoss / T(len(samples))
	}

	if jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetEscapeHTML(false)
		_ = enc.Encode(struct {
			Loss float64 `json:"loss"`
		}{Loss: float64(meanLoss)})
	} else {
		fmt.Printf("loss: %g\n", meanLoss)
	}

	return exitOK
}
