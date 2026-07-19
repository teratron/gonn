// Binary gonn — command-line interface for GoNN neural networks.
//
// Subcommands:
//
//	train   — train a network from config + CSV dataset
//	query   — run a single forward pass on an input vector
//	verify  — compute mean loss over an evaluation dataset (no weight update)
//	version — print the binary version
//
// Use "gonn <subcommand> --help" for per-subcommand flags.
package main

import (
	"fmt"
	"os"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(exitUserConfig)
	}

	code := dispatch(os.Args[1], os.Args[2:])
	os.Exit(code)
}

// dispatch routes os.Args[1] to the matching subcommand handler.
// Returns the exit code; callers pass it to os.Exit.
func dispatch(sub string, args []string) int {
	switch sub {
	case "train":
		return trainCmd(args)
	case "query":
		return queryCmd(args)
	case "verify":
		return verifyCmd(args)
	case "version":
		return versionCmd(args)
	default:
		fmt.Fprintf(os.Stderr, "gonn: unknown subcommand %q\n\n", sub)
		usage()
		return exitUserConfig
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "Usage: gonn <subcommand> [flags]")
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "Subcommands:")
	fmt.Fprintln(os.Stderr, "  train    Train a network from config + CSV dataset")
	fmt.Fprintln(os.Stderr, "  query    Run a forward pass on a comma-separated input vector")
	fmt.Fprintln(os.Stderr, "  verify   Compute mean loss over an evaluation CSV (no weight update)")
	fmt.Fprintln(os.Stderr, "  version  Print version information")
}
