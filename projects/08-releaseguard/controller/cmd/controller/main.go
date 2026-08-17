// Command controller is ReleaseGuard's CLI entrypoint (spec repo layout:
// controller/cmd/). Phase 4/5 give it two subcommands, both offline (no
// live Prometheus or Kubernetes needed): `validate` checks a policy file
// against the typed schema (internal/policy), and `simulate` runs a
// recorded-telemetry scenario (internal/scenario) through the real
// evaluator + rollout state machine and prints the resulting audit trail.
// This is the "local reproducibility command" Part 4 of the spec requires
// every task to have. Phase 6 will add the commands that talk to a live
// cluster/Prometheus; nothing here does yet, on purpose.
package main

import (
	"fmt"
	"os"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}

	var err error
	switch os.Args[1] {
	case "validate":
		err = runValidate(os.Args[2:])
	case "simulate":
		err = runSimulate(os.Args[2:])
	case "-h", "--help", "help":
		usage()
		return
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n\n", os.Args[1])
		usage()
		os.Exit(2)
	}

	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `releaseguard controller

Usage:
  controller validate -policy <path>
      Parse and validate a policy file (YAML or JSON) against the typed
      schema. Exit 0 and a summary on success; exit 1 and every problem
      found on failure.

  controller simulate -policy <path> -scenario <path> [-audit-out <path>]
      Run a recorded-telemetry scenario file through the real evaluator and
      rollout state machine, end to end, and print the resulting audit
      trail as JSON Lines. With -audit-out, also (or instead, with
      -audit-out -) persist the trail via the same JSONLLogger production
      code uses (ADR 0005).

Examples:
  go run ./cmd/controller validate -policy policy/examples/demo-service-default.yaml
  go run ./cmd/controller simulate \
      -policy policy/examples/demo-service-default.yaml \
      -scenario scenarios/a-healthy-promotion.json
`)
}
