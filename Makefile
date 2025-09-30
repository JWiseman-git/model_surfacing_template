.PHONY: install install-main lock test notebooks pull_spacy_corpora docker-build

install:
	uv pip install

install-main:
	uv pip install

lock:
	uv lock

test:
	uv run pytest tests

pull_spacy_corpora:
	uv run python -m spacy download en_core_web_sm

notebooks:
	uv run jupyter notebook

docker-build:
	docker build --progress=plain -t ml_nlp_service:latest -f docker/Dockerfile .
