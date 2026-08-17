# ReleaseGuard -- convenience targets for Phases 0-1.
#
# Every target here is a thin wrapper around a command documented in
# README.md; nothing here is "the" reproducibility mechanism, it just
# saves typing. Run `make help` for a summary.

.PHONY: help build test vet run docker-build cluster-up cluster-down smoke-test smoke-test-canary canary-healthy canary-unhealthy tidy fmt clean

DEMO_DIR := demo-service
IMAGE    := releaseguard/demo-service:local
BIN      := $(DEMO_DIR)/bin/demo-service

help:
	@echo "make build         - compile demo-service to $(BIN)"
	@echo "make test          - go vet + go test (race, cover) for demo-service"
	@echo "make run           - go run the demo-service locally (PORT=8080 by default)"
	@echo "make docker-build  - build the demo-service container image ($(IMAGE))"
	@echo "make cluster-up       - create/reuse local kind cluster, deploy stable+canary, run smoke tests"
	@echo "make cluster-down     - delete the local kind cluster"
	@echo "make smoke-test       - run tests/e2e/smoke_test.sh (stable) against whatever cluster is current"
	@echo "make smoke-test-canary - run tests/e2e/canary_smoke_test.sh (canary + split routing)"
	@echo "make canary-healthy   - flip demo-service-canary to a healthy config (see canary-scenario.sh)"
	@echo "make canary-unhealthy - flip demo-service-canary to an unhealthy config (see canary-scenario.sh)"
	@echo "make tidy             - go mod tidy for demo-service"
	@echo "make fmt              - gofmt demo-service"

build:
	cd $(DEMO_DIR) && go build -o bin/demo-service ./cmd/demo-service

vet:
	cd $(DEMO_DIR) && go vet ./...

test: vet
	cd $(DEMO_DIR) && go test ./... -race -cover

run:
	cd $(DEMO_DIR) && go run ./cmd/demo-service

docker-build:
	docker build -t $(IMAGE) $(DEMO_DIR)

cluster-up:
	./deploy/local-cluster/setup-cluster.sh

cluster-down:
	./deploy/local-cluster/teardown-cluster.sh

smoke-test:
	./tests/e2e/smoke_test.sh

smoke-test-canary:
	./tests/e2e/canary_smoke_test.sh

canary-healthy:
	./deploy/local-cluster/canary-scenario.sh healthy

canary-unhealthy:
	./deploy/local-cluster/canary-scenario.sh unhealthy

tidy:
	cd $(DEMO_DIR) && go mod tidy

fmt:
	cd $(DEMO_DIR) && gofmt -l -w .

clean:
	rm -rf $(DEMO_DIR)/bin
