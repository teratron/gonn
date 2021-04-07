
test:
	go test -v -cover ./nn/... ./params/... ./utils/... ./zoo/...

testing:
	go test -v -cover $(go list ./... | grep -v /examples)

run-example:
	go run -v ./examples/main.go

run-example-hopfield:
	go run -v ./examples/hopfield/main.go

run-example-query:
	go run -v ./examples/query/main.go

run-example-json:
	go run -v ./examples/json/main.go

run-example-yaml:
	go build -v -mod vendor ./examples/yaml/main.go

deps:
	go get
	go mod verify
	go mod tidy -v
	go mod vendor -v

clean:
	go clean -modcache

clean-project:
	for file in *.exe *.log *.synctex.gz *.aux *.out *.toc; do \
		if [ -e "$file" ]; then rm "$$file" || exit 1; \
        else printf 'No such file: %q\n' "$file" \
        fi \
	done
