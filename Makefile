.PHONY: install lock test docker-build

install:
	uv pip install

lock:
	uv lock

test:
	uv run pytest tests

docker-build:
	docker build --progress=plain -t ml_nlp_service:latest -f Dockerfile .
