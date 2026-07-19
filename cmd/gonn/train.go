package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/teratron/gonn/pkg/utils"
)

// trainCmd implements the "train" subcommand. It compiles a network from the
// supplied config, optionally restores a prior checkpoint, trains on the
// given dataset, and writes the output weights to disk.
func trainCmd(args []string) int {
	fs := flag.NewFlagSet("train", flag.ContinueOnError)
	cfgPath := fs.String("config", "", "path to config.json (required)")
	dataPath := fs.String("data", "", "path to training CSV file (required)")
	outPath := fs.String("out", "weights.json", "output weights path")
	resume := fs.String("resume", "", "existing weights.json to resume from")
	precision := fs.String("precision", "float32", "numeric precision: float32 or float64")
	jsonOut := fs.Bool("json", false, "machine-readable JSON output")

	if err := fs.Parse(args); err != nil {
		fmt.Fprintf(os.Stderr, "gonn train: %v\n", err)
		return exitUserConfig
	}
	if *cfgPath == "" || *dataPath == "" {
		fmt.Fprintln(os.Stderr, "gonn train: --config and --data are required")
		fs.Usage()
		return exitUserConfig
	}

	switch *precision {
	case "float64":
		return runTrain[float64](*cfgPath, *dataPath, *outPath, *resume, *jsonOut)
	case "float32":
		return runTrain[float32](*cfgPath, *dataPath, *outPath, *resume, *jsonOut)
	default:
		fmt.Fprintf(os.Stderr, "gonn train: unsupported --precision %q (float32|float64)\n", *precision)
		return exitUnsupported
	}
}

func runTrain[T utils.Float](cfgPath, dataPath, outPath, resumePath string, jsonOut bool) int {
	n, doc, err := loadNetwork[T](cfgPath, resumePath)
	if err != nil {
		if jsonOut {
			printErrorJSON(err)
		} else {
			fmt.Fprintf(os.Stderr, "gonn train: load: %v\n", err)
		}
		return exitCode(err)
	}

	samples, err := loadSamples[T](dataPath, int(doc.InputSize), int(doc.Output.Size))
	if err != nil {
		if jsonOut {
			printErrorJSON(err)
		} else {
			fmt.Fprintf(os.Stderr, "gonn train: data: %v\n", err)
		}
		return exitCode(err)
	}

	epochs, finalLoss, err := n.Fit(samples)
	if err != nil {
		if jsonOut {
			printErrorJSON(err)
		} else {
			fmt.Fprintf(os.Stderr, "gonn train: fit: %v\n", err)
		}
		return exitCode(err)
	}

	if err := saveWeights(outPath, doc, n); err != nil {
		if jsonOut {
			printErrorJSON(err)
		} else {
			fmt.Fprintf(os.Stderr, "gonn train: save: %v\n", err)
		}
		return exitCode(err)
	}

	if jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetEscapeHTML(false)
		_ = enc.Encode(struct {
			Epochs    uint    `json:"epochs"`
			FinalLoss float64 `json:"final_loss"`
		}{
			Epochs:    epochs,
			FinalLoss: float64(finalLoss),
		})
	} else {
		fmt.Printf("epochs: %d, loss: %g\n", epochs, finalLoss)
	}

	return exitOK
}

// printErrorJSON writes a JSON error object to stdout for machine consumers.
func printErrorJSON(err error) {
	enc := json.NewEncoder(os.Stdout)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(struct {
		Error string `json:"error"`
	}{Error: err.Error()})
}
