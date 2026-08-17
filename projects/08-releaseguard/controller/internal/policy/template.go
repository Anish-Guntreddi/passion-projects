package policy

import (
	"bytes"
	"fmt"
	"text/template"
)

// TemplateContext is the data a signal's (or the sample-count gate's)
// query template is rendered against. Track is substituted for "canary"
// or the configured BaselineTrack (default "stable"); Window is the
// policy's evaluation window rendered as a Go duration string (e.g.
// "30s") for direct use inside a PromQL range vector like
// `rate(x[{{.Window}}])`.
type TemplateContext struct {
	Track  string
	Window string
}

// RenderQuery renders a query template against ctx. Used both by
// Validate (to catch a malformed template at policy-load time, before any
// rollout ever depends on it) and by the evaluator (internal/evaluator) to
// build the concrete PromQL string sent to a metrics.Querier.
func RenderQuery(tmplText string, ctx TemplateContext) (string, error) {
	tmpl, err := template.New("query").Option("missingkey=error").Parse(tmplText)
	if err != nil {
		return "", fmt.Errorf("policy: parse query template: %w", err)
	}
	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, ctx); err != nil {
		return "", fmt.Errorf("policy: render query template: %w", err)
	}
	return buf.String(), nil
}
