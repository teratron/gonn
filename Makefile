# Makefile

test: ## coverage test
	go test -v -cover ./pkg/... ./internal/...

build-example-perceptron: ## build example perceptron
	go build -v -o ./examples/perceptron ./examples/perceptron/main.go

build-example-hopfield: ## build example hopfield
	go build -v -o ./examples/hopfield ./examples/hopfield/main.go

build-example-linear: ## build example linear
	go build -v -o ./examples/linear ./examples/linear/main.go

build-example-and-train: ## build example and_train
	go build -v -o ./examples/and_train ./examples/and_train/main.go

build-example-query: ## build example query
	go build -v -o ./examples/query ./examples/query/main.go

build-example-json: ## build example json
	go build -v -o ./examples/json ./examples/json/main.go

setup: ## setup
	go mod init
	go mod tidy -v
	go mod vendor -v

deps: ## setup deps
	go mod verify
	go mod tidy -v
	go mod vendor -v

clean: ## clean cache
	go clean -modcache

#@find . -path '/examples/**/*.exe' -delete
#@find ./examples/perceptron/ -name "*.exe" -delete
delete: ## delete garbage
	@find . -empty -delete

#VERSION := $(shell cat ./VERSION)
VERSION := 0.4.0
release: ## release
	git tag -a $(VERSION) -m "Release" || true
	git push origin $(VERSION)
	goreleaser --rm-dist

set_url: ## git remote set-url origin git@github.com:login/repo.git
	git remote set-url origin git@github.com:teratron/gonn.git

message = Update.
branch = master
add-commit-push: ## add commit push
	git add -u
	git commit -m "$(message)"
	git push origin $(branch)

.PHONY: help
help:
	@awk '                                             \
		BEGIN {FS = ":.*?## "}                         \
		/^[a-zA-Z_-]+:.*?## /                          \
		{printf "\033[36m%-24s\033[0m %s\n", $$1, $$2} \
	'                                                  \
	$(MAKEFILE_LIST)

.DEFAULT_GOAL := help
