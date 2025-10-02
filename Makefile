PYTHON := python
VENV_DIR := venv
UV := uv

# Default Python environment setup
setup:
	$(PYTHON) -m venv $(VENV_DIR)
	# Install uv if missing
	. $(VENV_DIR)/Scripts/activate && pip install --upgrade pip uv
	# Install project dependencies from pyproject.toml
	. $(VENV_DIR)/bin/activate && uv install -e .

# Run the FastAPI server locally
run:
	. $(VENV_DIR)/bin/activate && uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Build Docker image
docker-build:
	docker build -t mcqa-api .

# Run Docker container
docker-run:
	docker run -p 8000:8000 mcqa-api
