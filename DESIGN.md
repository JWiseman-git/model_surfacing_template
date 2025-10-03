# Report on design

This document outlines the: design choices, known limitations and future improvements for the MCQA serving application.
The goal of this service is to expose a multiple-choice question answering model via a REST API, using a pretrained SafeTransformers DistilBERT model.

## Design Choices

1. **Modular Structure**:
- **Motivation:** Clear separation of responsibilities makes the code easier to maintain, test, and extend.
- main.py handles FastAPI app instantiation and server orchestration. 
- Endpoints/mcqa.py contains all routes related to MCQA predictions, keeping the API logic clean and extensible.
- model_mcqa.py encapsulates the ML model, including loading, inference, and optional post-processing.
- utils.py contains reusable helper functions.

2.  **Pretrained Transformer Model** (preknown design choice)
- **Motivation:** Transformers like DistilBERT understand contextual relationships in text. This is crucial for MCQA, where the missing word depends on context.
- The application uses a SafeTransformers DistilBERT model with a multiple-choice head.  
- SafeTransformers provides efficient inference, robust model loading, and compatibility with production pipelines, making it easier to deploy reliably.

3. **Model I/O Handling**  
- **Motivation**: The MCQA task requires evaluating each candidate word in context, so input formatting is critical.  
- Inputs are tokenized with the matching tokenizer and arranged into a batch with shape [batch_size, num_choices, seq_len] to match the transformer model’s expected input.  
- This design ensures correct tensor shapes, avoids runtime errors, and allows scaling to batch inference in the future.
- Logic is encapsulated in mcqa_model.py allowing endpoints to remain simple and decoupled from the model input and output processing logic.

4. **Endpoint Design**  
- **Motivation**: API endpoints should be intuitive, reusable, and decoupled from ML logic.  
- The use of Pydantic request and response models ensures input validation and structured output, reducing the risk of errors in production.  
- Future endpoints can easily extend functionality (e.g., top-N predictions, analytics) without changing core model logic.

5. **MLOps-Readiness**  
- **Motivation**: The design anticipates production deployment, observability, and model versioning.  
- Model files are stored in a dedicated models/ folder, enabling easy integration with a model registry like MLflow in the future.  
- The architecture supports containerization, with FastAPI decoupled from model logic, allowing clean deployment in Docker.  
- The model has extensive logging capability - a first step to integrating telemetry like Prometheus/Grafana.
- A cicd (guthub actions) is provisioned for the automated released of new features - whilst ensuring formatting best practices and tests are auto run. 
- Automated release capability allows for easy extension of terraform provisioned env specific infra (test, dev and prod)
- Code version control via github.

---

## Known limitations and improvements

1. **Single-instance inference**  
   - Currently, the API runs inference  with a single model instance. 
   - High concurrency could lead to performance bottlenecks. 
   - **Improvement**: introduce async model calls or multiple worker processes in production.
   - The design would allow for easily allow for this extension. 

2. **Model storage**  
   - Models are loaded from the local filesystem; no versioning is implemented yet.
   - The current spin up and spin down nature of this docker strategy tears down the registry with - meaning model versioning is not currently persistent. 
   - For the moment this inclusion is to demonstrate the capability and easy switch over (a util for model loading is already shown)
   - **Improvement**: integrate with MLflow Model Registry with S3 for managed versioning and rollback.

3. **Input constraints**  
   - The API requires exactly 4 candidate choices; more flexible handling is not yet implemented.  
   - Improvement: allow variable-length choice lists and proper validation.

4. **Observability**  
   - Logging and metrics are restricted to local logs and mlflow run logs.  
   - Improvement: add **structured logging**, latency tracking, and request-level metrics using tools like **Prometheus**.

---

## Future design

1. **Model Registry Integration**  
   - Store models in a persistent aws hosted MLflow instance. Enabling model version control, reproducibility, and easy rollout of new versions.  
   - With this implemented, it would allow for dynamic model loading based on version requests.
   - It would enable safe rollouts and rollback strategies. 

2. **Batch and Async Inference**  
   - Support batch requests to improve throughput.  
   - Move inference to async or GPU-accelerated pipelines.
   - Alongside this, queing logic could be enabled/streaming of requests to ensure a smooth balancing of request loads. 

3. **Extended Endpoints**  
   - Some possible extensions
   - `/mcqa/predict_top_n`: return top-N predictions with probabilities.  
   - `/mcqa/analytics`: return inference statistics, confidence distributions, or aggregated metrics.

4. **CI/CD Pipeline (Including inclusion of training)**  
   - Extend the deployment to a full end-to-end pipeline where additions to a model triggers a distinct training step.
   - The training script could submit the newest version of the model to sagemaker for training. The model can be defaulted to use this latest model but also specific models tagged with a commit sha.
   - Following this, the standard steps of: testing, Docker image builds, and deployment via terraform.  
   - This would allow this project to become a development AND deployment/serving template. 

5. **Monitoring & Alerts**  
   - Set alerts for unusual API behaviour or degraded model performance.
   - Configure services like Prometheus or Grafana for observability of monitoring. 
   - Track model accuracy over time such as model drift. 

# BONUS - Serving improvement 

Some ideas for serving improvements:

- Cache previously used predictions for repeated passages and choices (LRU cache/redis). This could extended to passage tokenization caching as well. 
- Model could be loaded onto a dedicated inference server - efficiently serve multiple API reqs. Would allow for horizontal scaling as well. 
- Model level optimizations could include a distillation technique where a smaller a model is used for faster inference. 
- Async/Parallel serving - discussed above, but would mean having multiple workers handling multiple requests in parallel/concurrently.