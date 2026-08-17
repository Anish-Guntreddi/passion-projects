# ReleaseGuard -- convenience targets for Phases 0-5.
#
# Every target here is a thin wrapper around a command documented in
# README.md; nothing here is "the" reproducibility mechanism, it just
# saves typing. Run `make help` for a summary.

.PHONY: help build test vet run docker-build cluster-up cluster-down smoke-test smoke-test-canary canary-healthy canary-unhealthy tidy fmt clean controller-test controller-vet controller-build controller-validate-policy controller-simulate controller-tidy controller-fmt

DEMO_DIR       := demo-service
CONTROLLER_DIR := controller
IMAGE          := releaseguard/demo-service:local
BIN            := $(DEMO_DIR)/bin/demo-service
POLICY         := policy/examples/demo-service-default.yaml
SCENARIO       := scenarios/a-healthy-promotion.json

help:
	@echo "make build         - compile demo-service to $(BIN)"
	@echo "make test          - go vet + go test (race, cover) for demo-service AND controller"
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
	@echo "make controller-test           - go vet + go test (race, cover) for controller (Phase 4-5)"
	@echo "make controller-build          - compile the controller CLI"
	@echo "make controller-validate-policy - validate policy/examples/demo-service-default.yaml"
	@echo "make controller-simulate       - run scenarios/a-healthy-promotion.json through the state machine"
	@echo "make controller-tidy / controller-fmt - go mod tidy / gofmt for controller"

build:
	cd $(DEMO_DIR) && go build -o bin/demo-service ./cmd/demo-service

vet:
	cd $(DEMO_DIR) && go vet ./...

test: vet controller-vet
	cd $(DEMO_DIR) && go test ./... -race -cover
	cd $(CONTROLLER_DIR) && go test ./... -race -cover

run:
	cd $(DEMO_DIR) && go run ./cmd/demo-service

docker-build:
	docker build -t $(IMAGE) $(DEMO_DIR)

controller-vet:
	cd $(CONTROLLER_DIR) && go vet ./...

controller-test: controller-vet
	cd $(CONTROLLER_DIR) && go test ./... -race -cover

controller-build:
	cd $(CONTROLLER_DIR) && go build -o bin/controller ./cmd/controller

controller-validate-policy:
	cd $(CONTROLLER_DIR) && go run ./cmd/controller validate -policy $(POLICY)

controller-simulate:
	cd $(CONTROLLER_DIR) && go run ./cmd/controller simulate -policy $(POLICY) -scenario $(SCENARIO)

controller-tidy:
	cd $(CONTROLLER_DIR) && go mod tidy

controller-fmt:
	cd $(CONTROLLER_DIR) && gofmt -l -w .

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
	rm -rf $(DEMO_DIR)/bin $(CONTROLLER_DIR)/bin
