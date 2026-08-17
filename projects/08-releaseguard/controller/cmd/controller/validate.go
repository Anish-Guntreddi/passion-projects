package main

import (
	"flag"
	"fmt"
	"os"

	"releaseguard/controller/internal/policy"
)

func runValidate(args []string) error {
	fs := flag.NewFlagSet("validate", flag.ContinueOnError)
	policyPath := fs.String("policy", "", "path to a policy file (YAML or JSON)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *policyPath == "" {
		fs.Usage()
		return fmt.Errorf("-policy is required")
	}

	p, err := policy.Load(*policyPath)
	if err != nil {
		return err
	}

	fmt.Printf("OK: %s (schema version %d)\n", p.Name, p.Version)
	fmt.Printf("  window=%s interval=%s max_rollout_duration=%s\n",
		p.Evaluation.Window.Duration, p.Evaluation.Interval.Duration, p.Evaluation.MaxRolloutDuration.Duration)
	fmt.Printf("  consecutive_healthy_windows=%d consecutive_unhealthy_windows=%d\n",
		p.Evaluation.ConsecutiveHealthyWindows, p.Evaluation.ConsecutiveUnhealthyWindows)
	fmt.Printf("  on_missing_telemetry=%s baseline_track=%s\n",
		p.Evaluation.OnMissingTelemetry, p.Evaluation.BaselineTrack)
	if p.Evaluation.MinSampleCount > 0 {
		fmt.Printf("  min_sample_count=%d\n", p.Evaluation.MinSampleCount)
	}
	fmt.Printf("  signals (%d):\n", len(p.Signals))
	for _, s := range p.Signals {
		optional := ""
		if s.Optional {
			optional = " [optional]"
		}
		bound := ""
		if s.Threshold.Max != nil {
			bound += fmt.Sprintf(" max=%v", *s.Threshold.Max)
		}
		if s.Threshold.Min != nil {
			bound += fmt.Sprintf(" min=%v", *s.Threshold.Min)
		}
		relative := ""
		if s.Relative != nil {
			if s.Relative.MaxRatio != nil {
				relative += fmt.Sprintf(" relative_max_ratio=%v", *s.Relative.MaxRatio)
			}
			if s.Relative.MinRatio != nil {
				relative += fmt.Sprintf(" relative_min_ratio=%v", *s.Relative.MinRatio)
			}
		}
		fmt.Fprintf(os.Stdout, "    - %s:%s%s%s\n", s.Name, bound, relative, optional)
	}
	return nil
}
