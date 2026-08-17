package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"releaseguard/controller/internal/audit"
	"releaseguard/controller/internal/policy"
	"releaseguard/controller/internal/rollout"
	"releaseguard/controller/internal/scenario"
)

func runSimulate(args []string) error {
	fs := flag.NewFlagSet("simulate", flag.ContinueOnError)
	policyPath := fs.String("policy", "", "path to a policy file (YAML or JSON)")
	scenarioPath := fs.String("scenario", "", "path to a scenario file (JSON, see controller/scenarios/*.json)")
	auditOut := fs.String("audit-out", "", "optional path to also persist the audit trail as JSONL (ADR 0005)")
	rolloutID := fs.String("id", "", "rollout ID to use (default: derived from the scenario file name)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *policyPath == "" || *scenarioPath == "" {
		fs.Usage()
		return fmt.Errorf("-policy and -scenario are both required")
	}

	p, err := policy.Load(*policyPath)
	if err != nil {
		return fmt.Errorf("load policy: %w", err)
	}
	scen, err := scenario.Load(*scenarioPath)
	if err != nil {
		return fmt.Errorf("load scenario: %w", err)
	}

	id := *rolloutID
	if id == "" {
		id = fmt.Sprintf("sim-%d", time.Now().UnixNano())
	}

	start := time.Now().UTC()
	action := &rollout.RecordingAction{}
	res, err := scenario.Run(context.Background(), id, p, action, nil, scen, start)
	if err != nil {
		return fmt.Errorf("run scenario: %w", err)
	}

	if scen.Description != "" {
		fmt.Printf("# %s\n", scen.Description)
	}
	fmt.Printf("# rollout=%s policy=%s final_state=%s final_decision=%s windows=%d promote_calls=%d rollback_calls=%d\n",
		id, p.Name, res.Rollout.State(), res.Rollout.FinalDecision(), res.Rollout.WindowCount(),
		action.PromoteCalls, action.RollbackCalls)

	enc := json.NewEncoder(os.Stdout)
	for _, e := range res.Events {
		if err := enc.Encode(e); err != nil {
			return fmt.Errorf("encode audit event: %w", err)
		}
	}

	if *auditOut != "" {
		logger, err := audit.NewJSONLLogger(*auditOut)
		if err != nil {
			return fmt.Errorf("open -audit-out: %w", err)
		}
		defer logger.Close()
		for _, e := range res.Events {
			if err := logger.Record(e); err != nil {
				return fmt.Errorf("persist audit event: %w", err)
			}
		}
		fmt.Fprintf(os.Stderr, "audit trail written to %s (%d events)\n", *auditOut, len(res.Events))
	}

	return nil
}
