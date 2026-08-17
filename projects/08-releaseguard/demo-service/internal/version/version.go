// Package version exposes build-time metadata for the demo-service binary.
//
// The three variables below are the zero-value defaults used by `go run`
// and `go test`. Release builds overwrite them via linker flags, e.g.:
//
//	go build -ldflags "-X .../internal/version.Version=v1.2.3 \
//	    -X .../internal/version.GitCommit=$(git rev-parse HEAD) \
//	    -X .../internal/version.BuildTime=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
//
// The Dockerfile and CI workflow both set these so that every image is
// traceable back to the exact commit and build time that produced it,
// which is what makes the resulting tag+digest an "immutable" reference
// rather than just a mutable label.
package version

var (
	// Version is a human-assigned release identifier (e.g. a git short SHA
	// or semver tag). Defaults to "dev" for local, unreleased builds.
	Version = "dev"

	// GitCommit is the full commit SHA the binary was built from.
	GitCommit = "unknown"

	// BuildTime is an RFC3339 UTC timestamp of when the binary was built.
	BuildTime = "unknown"
)

// Info is the JSON-serializable snapshot returned by the /version endpoint.
type Info struct {
	Version   string `json:"version"`
	GitCommit string `json:"git_commit"`
	BuildTime string `json:"build_time"`
}

// Current returns the build metadata captured in this package.
func Current() Info {
	return Info{
		Version:   Version,
		GitCommit: GitCommit,
		BuildTime: BuildTime,
	}
}
