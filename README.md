# ML Service

![Coveo Logo](docs/graphics/coveo-logo.png "Coveo Logo")

> PROPRIETARY and CONFIDENTIAL, Coveo Solutions Inc.
> v2.0.0 - Last Edited 2024-10-01.

> Are you looking for the description of the challenge? Refer to [INSTRUCTIONS.md](INSTRUCTIONS.md).

This is a barebones ML project that serves an ML model via an HTTP API. This serves a dummy ML model, which is a basic text chunker.

In the [notebooks](notebooks) directory, a notebook for training a Multiple Choice Question Answering (MCQA) can be found.

## Quick start

Documentation on getting started.

### Prerequisites

- Python 3.10
- Poetry
- Makefile (GNU Make)

### Commands

Commands are provided in this project's Makefile:

- `make install` Install project dependencies (via Poetry).
- `make lock` Update the Poetry lock file.
- `make test` Run the test suite.
- `make notebooks` Run a Jupyter environment.
- `make docker-build`: Build a docker image. TODO: This is currently not a complete dockerfile.

### Structure

```text
artifacts/           Skeleton structure for artifacts
    datasets/
    models/
docker/              Dockerfile
docs/                Documentation and graphics
ml_service/          Python source code
notebooks/           Jupyter notebooks
tests/               Python test suite
Makefile             Runnable commands
pyproject.toml       Configuration for project tools
README.md            This read me
```

### Run the service

Set up:

```bash
make install
make pull-spacy-corpora
```

Start the service:

```bash
export DEVELOPMENT_MODE=true
poetry run python ml_service/main.py
```

Use the service:

- Visit the index (e.g., via a browser) at: `http://127.0.0.1:8080`
- Visit the docs at: `http://127.0.0.1:8080/docs`
- Send a request: `http://127.0.0.1:8080/chunkify?text=Hello%20word.%20Nice%20to%20meet%20again.`
