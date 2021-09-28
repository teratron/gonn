#

test: ## coverage test
	go test -v -cover ./pkg/... ./internal/...

build-example-perceptron: ## build
	go build -v -o ./examples/perceptron ./examples/perceptron/main.go

build-example-hopfield: ## build
	go build -v -o ./examples/hopfield ./examples/hopfield/main.go

build-example-linear: ## build
	go build -v -o ./examples/linear ./examples/linear/main.go

build-example-query: ## build
	go build -v -o ./examples/query ./examples/query/main.go

build-example-json: ## build
	go build -v -o ./examples/json ./examples/json/main.go

build-example-yaml: ## build
	go build -v -mod vendor -o ./examples/yaml ./examples/yaml/main.go

setup: ## setup
	go mod init
	go mod tidy -v
	go mod vendor -v

deps: ## setup deps
	go mod verify
	go mod tidy -v
	go mod vendor -v

clean: ## clean
	go clean -modcache

#VERSION := $(shell cat ./VERSION)

release: ## release
	git tag -a $(VERSION) -m "Release" || true
	git push origin $(VERSION)
	goreleaser --rm-dist

.PHONY: help
help:
	@awk '                                             \
		BEGIN {FS = ":.*?## "}                         \
		/^[a-zA-Z_-]+:.*?## /                          \
		{printf "\033[36m%-24s\033[0m %s\n", $$1, $$2} \
	'                                                  \
	$(MAKEFILE_LIST)

.DEFAULT_GOAL := help
