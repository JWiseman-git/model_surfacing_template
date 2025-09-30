.PHONY: install install-main lock test notebooks pull_spacy_corpora

install:
	poetry install

install-main:
	poetry install --only main

lock:
	poetry lock --no-update

test:
	poetry run pytest tests

pull-spacy-corpora:
	poetry run python -m spacy download en_core_web_sm

notebooks:
	poetry run jupyter notebook

docker-build:
	docker build --progress=plain -t ml_nlp_service:latest -f docker/Dockerfile .
