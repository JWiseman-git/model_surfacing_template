# Operational characteristics - MCQA Service

## Summary
Service exposes a single prediction endpoint `/predict` and a health endpoint `/healthz`. The service loads a HuggingFace PyTorch `AutoModelForMultipleChoice` and its tokenizer from a local model directory.

## Resource requirements
- **Memory (RAM)**: Model size ~20MB on disk suggests modest RAM — however PyTorch model loaded into memory will expand. Expect **~200–800MB** RAM for small models; larger transformer variants may consume multiple GB.
- **CPU**: Single-request inference on CPU for small models ~50–200 ms per request depending on CPU. Heavy CPUs (4+ vCPUs) recommended for moderate throughput.
- **GPU (optional)**: If low-latency (<20ms) and high throughput required, use a GPU (T4 or equivalent). With CUDA-enabled PyTorch, latency reduces substantially and batched throughput increases.
- **Disk**: Model files ~20MB + tokeniser files. Keep at least 1GB free for OS and logs.

## Performance expectations
- **Latency (single request, CPU)**: Local run returned ~50–500ms - would be dependent on CPU.
- **Throughput**: On CPU, expect tens of requests/sec with concurrency and worker processes; on GPU, hundreds/sec depending on batch strategy.
- **Scalability**:
  - Scale horizontally with multiple container replicas behind a load balancer.
  - For GPU-only inference, scale by multiple GPU instances 
  - The project is well set up to handle deployment into an AWS native architecture - tasks handled by Fargate/ ECS would be a good construction. 

## Bottlenecks & mitigation
- Model loading time: ~1–5s on start — use warm-up or keep a pool of ready containers.
- Tokenizer CPU cost: minimal but non-negligible under high QPS — pre-tokenise/ integrate a cached service like Redis if identical patterns used frequently.
- Concurrency: FastAPI is capable of handling asynchronous calls. Expand the application to use async worker or multiple uvicorn workers. Will need some consideration of available memory/compute.

## Deployment recommendations
- Start with 1 replica (CPU) for dev. For production:
  - If latency <=100ms required → deploy GPU-backed instances.
  - Use horizontal autoscaling based on CPU/GPU utilisation and request latency.
  - Add monitoring (latency, error rate, GPU memory) and alerting.

## Observability
- Track: request latency (currently implemented), failure rate, model load time, GPU util, memory usage.
- Export metrics via Prometheus-compatible client and logs via structured JSON.

