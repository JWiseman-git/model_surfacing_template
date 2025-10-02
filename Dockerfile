FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:$PATH"
ENV MLFLOW_TRACKING_URI=file:/mlruns

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock* ./

RUN uv sync --locked --no-dev

COPY docker .

# Create local directory for MLflow runs
RUN mkdir -p /mlruns

EXPOSE 8080
EXPOSE 5000

CMD ["uv", "run", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
