package policy_test

import (
	"strings"
	"testing"

	"releaseguard/controller/internal/policy"
)

func TestRenderQuery(t *testing.T) {
	got, err := policy.RenderQuery(
		`sum(rate(http_requests_total{track="{{.Track}}"}[{{.Window}}]))`,
		policy.TemplateContext{Track: "canary", Window: "30s"},
	)
	if err != nil {
		t.Fatalf("RenderQuery returned error: %v", err)
	}
	want := `sum(rate(http_requests_total{track="canary"}[30s]))`
	if got != want {
		t.Errorf("RenderQuery = %q, want %q", got, want)
	}
}

func TestRenderQuery_TrackAndBaselineDiffer(t *testing.T) {
	tmpl := `sum(rate(http_requests_total{track="{{.Track}}"}[{{.Window}}]))`
	canary, err := policy.RenderQuery(tmpl, policy.TemplateContext{Track: "canary", Window: "30s"})
	if err != nil {
		t.Fatalf("RenderQuery(canary) returned error: %v", err)
	}
	stable, err := policy.RenderQuery(tmpl, policy.TemplateContext{Track: "stable", Window: "30s"})
	if err != nil {
		t.Fatalf("RenderQuery(stable) returned error: %v", err)
	}
	if canary == stable {
		t.Fatalf("canary and stable renders were identical (%q) -- a track-scoped template must differ", canary)
	}
	if !strings.Contains(canary, `"canary"`) || !strings.Contains(stable, `"stable"`) {
		t.Errorf("renders did not contain expected track literals: canary=%q stable=%q", canary, stable)
	}
}

func TestRenderQuery_MalformedTemplate(t *testing.T) {
	if _, err := policy.RenderQuery(`sum({{.Track}`, policy.TemplateContext{Track: "canary", Window: "30s"}); err == nil {
		t.Fatal("RenderQuery(malformed template) returned nil error")
	}
}

func TestRenderQuery_UnknownField(t *testing.T) {
	// {{.Bogus}} isn't a field on TemplateContext; with missingkey=error
	// this must fail loudly rather than rendering "<no value>" into a
	// PromQL string that Prometheus would then reject (or worse, silently
	// misinterpret) far from where the typo actually is.
	if _, err := policy.RenderQuery(`sum({{.Bogus}})`, policy.TemplateContext{Track: "canary", Window: "30s"}); err == nil {
		t.Fatal("RenderQuery(unknown field) returned nil error")
	}
}
