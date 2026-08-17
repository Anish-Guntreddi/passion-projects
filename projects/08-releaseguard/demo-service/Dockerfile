# syntax=docker/dockerfile:1
#
# Multi-stage build for the ReleaseGuard demo-service.
#
# Build-time metadata (VERSION/GIT_COMMIT/BUILD_TIME) is injected via
# --build-arg and baked into the binary with -ldflags, and mirrored onto
# OCI labels on the final image. This is what lets the /version endpoint
# and `docker inspect` agree on exactly which commit produced a given
# image -- the "immutable tagged/digested image" Phase 0's exit criterion
# asks for is the combination of this build-arg provenance plus the
# registry-assigned content digest captured by CI.

FROM golang:1.22-alpine AS build
WORKDIR /src

# Cache module downloads independently of source changes.
COPY go.mod go.sum ./
RUN go mod download

COPY . .

ARG VERSION=dev
ARG GIT_COMMIT=unknown
ARG BUILD_TIME=unknown

RUN CGO_ENABLED=0 GOOS=linux go build \
    -ldflags "-s -w \
      -X releaseguard/demo-service/internal/version.Version=${VERSION} \
      -X releaseguard/demo-service/internal/version.GitCommit=${GIT_COMMIT} \
      -X releaseguard/demo-service/internal/version.BuildTime=${BUILD_TIME}" \
    -o /out/demo-service \
    ./cmd/demo-service

FROM gcr.io/distroless/static-debian12:nonroot AS runtime

ARG VERSION=dev
ARG GIT_COMMIT=unknown

LABEL org.opencontainers.image.title="releaseguard-demo-service" \
      org.opencontainers.image.description="Instrumented HTTP canary target for ReleaseGuard" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${GIT_COMMIT}"

COPY --from=build /out/demo-service /demo-service

USER nonroot:nonroot
EXPOSE 8080
ENTRYPOINT ["/demo-service"]
