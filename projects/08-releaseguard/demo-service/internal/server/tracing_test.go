package server

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"releaseguard/demo-service/internal/config"
)

// newRecordedTracerProvider returns an SDK TracerProvider that samples
// every span and captures them in-memory via sr, instead of exporting
// anywhere -- exactly the "real production code path, fake destination"
// pattern config_test.go's fakeEnv uses for environment variables.
func newRecordedTracerProvider() (*sdktrace.TracerProvider, *tracetest.SpanRecorder) {
	sr := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithSpanProcessor(sr),
		sdktrace.WithSampler(sdktrace.AlwaysSample()),
	)
	return tp, sr
}

func findSpan(spans []sdktrace.ReadOnlySpan, name string) sdktrace.ReadOnlySpan {
	for _, s := range spans {
		if s.Name() == name {
			return s
		}
	}
	return nil
}

func attrValue(attrs []attribute.KeyValue, key attribute.Key) (attribute.Value, bool) {
	for _, a := range attrs {
		if a.Key == key {
			return a.Value, true
		}
	}
	return attribute.Value{}, false
}

func TestInstrument_CreatesRequestSpanWithTrackAndVersion(t *testing.T) {
	tp, sr := newRecordedTracerProvider()
	srv := newTestServer(t, func(c *config.Config) {
		c.ReleaseTrack = config.TrackCanary
		c.ReleaseVersion = "v2.0.0-canary"
	}, WithTracerProvider(tp))

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	srv.Handler().ServeHTTP(httptest.NewRecorder(), req)

	spans := sr.Ended()
	span := findSpan(spans, "/health")
	if span == nil {
		t.Fatalf("no span named /health recorded; got spans: %v", spanNames(spans))
	}

	track, ok := attrValue(span.Attributes(), attribute.Key("release.track"))
	if !ok || track.AsString() != "canary" {
		t.Errorf("span release.track = %v (ok=%v), want canary", track, ok)
	}
	version, ok := attrValue(span.Attributes(), attribute.Key("release.version"))
	if !ok || version.AsString() != "v2.0.0-canary" {
		t.Errorf("span release.version = %v (ok=%v), want v2.0.0-canary", version, ok)
	}
	status, ok := attrValue(span.Attributes(), attribute.Key("http.status_code"))
	if !ok || status.AsInt64() != http.StatusOK {
		t.Errorf("span http.status_code = %v (ok=%v), want 200", status, ok)
	}
	if span.Status().Code == codes.Error {
		t.Errorf("healthy request span has Error status: %s", span.Status().Description)
	}
}

func TestInstrument_MarksSpanErrorOn5xx(t *testing.T) {
	tp, sr := newRecordedTracerProvider()
	srv := newTestServer(t, func(c *config.Config) { c.ErrorRate = 1.0 }, WithTracerProvider(tp))

	req := httptest.NewRequest(http.MethodGet, "/work", nil)
	srv.Handler().ServeHTTP(httptest.NewRecorder(), req)

	span := findSpan(sr.Ended(), "/work")
	if span == nil {
		t.Fatalf("no span named /work recorded")
	}
	if span.Status().Code != codes.Error {
		t.Errorf("span Status().Code = %v, want Error for a 503 response", span.Status().Code)
	}
}

func TestHandleWork_CreatesDependencyChildSpan(t *testing.T) {
	tp, sr := newRecordedTracerProvider()
	srv := newTestServer(t, func(c *config.Config) { c.DependencyDown = true }, WithTracerProvider(tp))

	req := httptest.NewRequest(http.MethodGet, "/work", nil)
	srv.Handler().ServeHTTP(httptest.NewRecorder(), req)

	ended := sr.Ended()
	root := findSpan(ended, "/work")
	dep := findSpan(ended, "dependency.call")
	if root == nil || dep == nil {
		t.Fatalf("expected both /work and dependency.call spans; got: %v", spanNames(ended))
	}

	if dep.Parent().SpanID() != root.SpanContext().SpanID() {
		t.Errorf("dependency.call span's parent SpanID = %s, want the /work span's SpanID %s",
			dep.Parent().SpanID(), root.SpanContext().SpanID())
	}
	if dep.Status().Code != codes.Error {
		t.Errorf("dependency.call span Status().Code = %v, want Error (DependencyDown=true)", dep.Status().Code)
	}
	reason, ok := attrValue(dep.Attributes(), attribute.Key("dependency.failure_reason"))
	if !ok || reason.AsString() != "down" {
		t.Errorf("dependency.call span dependency.failure_reason = %v (ok=%v), want down", reason, ok)
	}
}

func TestHandleWork_DependencySpanHealthyHasNoErrorStatus(t *testing.T) {
	tp, sr := newRecordedTracerProvider()
	srv := newTestServer(t, nil, WithTracerProvider(tp))

	req := httptest.NewRequest(http.MethodGet, "/work", nil)
	srv.Handler().ServeHTTP(httptest.NewRecorder(), req)

	dep := findSpan(sr.Ended(), "dependency.call")
	if dep == nil {
		t.Fatalf("no dependency.call span recorded")
	}
	if dep.Status().Code == codes.Error {
		t.Errorf("healthy dependency.call span has Error status: %s", dep.Status().Description)
	}
}

func spanNames(spans []sdktrace.ReadOnlySpan) []string {
	names := make([]string, len(spans))
	for i, s := range spans {
		names[i] = s.Name()
	}
	return names
}
