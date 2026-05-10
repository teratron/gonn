package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/teratron/gonn/pkg/utils"
)

// queryCmd implements the "query" subcommand. It loads a compiled network
// from disk, runs a single forward pass on the input vector, and prints the
// output.
func queryCmd(args []string) int {
	fs := flag.NewFlagSet("query", flag.ContinueOnError)
	cfgPath := fs.String("config", "", "path to config.json (required)")
	weightsPath := fs.String("weights", "", "path to weights.json (required)")
	inputStr := fs.String("input", "", "comma-separated input values (required)")
	precision := fs.String("precision", "float32", "numeric precision: float32 or float64")
	jsonOut := fs.Bool("json", false, "machine-readable JSON output")

	if err := fs.Parse(args); err != nil {
		fmt.Fprintf(os.Stderr, "gonn query: %v\n", err)
		return exitUserConfig
	}
	if *cfgPath == "" || *weightsPath == "" || *inputStr == "" {
		fmt.Fprintln(os.Stderr, "gonn query: --config, --weights, and --input are required")
		fs.Usage()
		return exitUserConfig
	}

	switch *precision {
	case "float64":
		return runQuery[float64](*cfgPath, *weightsPath, *inputStr, *jsonOut)
	case "float32":
		return runQuery[float32](*cfgPath, *weightsPath, *inputStr, *jsonOut)
	default:
		fmt.Fprintf(os.Stderr, "gonn query: unsupported --precision %q (float32|float64)\n", *precision)
		return exitUnsupported
	}
}

func runQuery[T utils.Float](cfgPath, weightsPath, inputStr string, jsonOut bool) int {
	n, _, err := loadNetwork[T](cfgPath, weightsPath)
	if err != nil {
		if jsonOut {
			printErrorJSON(err)
		} else {
			fmt.Fprintf(os.Stderr, "gonn query: load: %v\n", err)
		}
		return exitCode(err)
	}

	input, err := parseFloats[T](inputStr)
	if err != nil {
		if jsonOut {
			printErrorJSON(err)
		} else {
			fmt.Fprintf(os.Stderr, "gonn query: --input: %v\n", err)
		}
		return exitInputData
	}

	output, err := n.Query(input)
	if err != nil {
		if jsonOut {
			printErrorJSON(err)
		} else {
			fmt.Fprintf(os.Stderr, "gonn query: %v\n", err)
		}
		return exitCode(err)
	}

	if jsonOut {
		floats := make([]float64, len(output))
		for i, v := range output {
			floats[i] = float64(v)
		}
		enc := json.NewEncoder(os.Stdout)
		enc.SetEscapeHTML(false)
		_ = enc.Encode(struct {
			Output []float64 `json:"output"`
		}{Output: floats})
	} else {
		parts := make([]string, len(output))
		for i, v := range output {
			parts[i] = fmt.Sprintf("%g", v)
		}
		fmt.Println(strings.Join(parts, ", "))
	}

	return exitOK
}

// parseFloats parses a comma-separated string into a []T.
func parseFloats[T utils.Float](s string) ([]T, error) {
	parts := strings.Split(s, ",")
	out := make([]T, len(parts))
	for i, p := range parts {
		v, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
		if err != nil {
			return nil, utils.Newf(utils.ErrInputData, "field %d: %v", i+1, err)
		}
		out[i] = T(v)
	}
	return out, nil
}
