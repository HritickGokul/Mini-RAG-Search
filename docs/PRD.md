# Mini Inference Platform — Product Requirements Document

**Status:** V1 design, pre-implementation
**Owner:** platform author
**Last updated:** see git history for this file

---

## 1. Problem Statement

Serving a language model in production is not the same problem as calling one.

A working single-user demo looks like this:

```python
output = model.generate(prompt)
```

That line is correct, and it is also the entire reason naive inference services fall over. The line
hides every property that matters once more than one client exists:

**Inference is expensive and non-uniform.** A generation request occupies a model for a duration
that is roughly proportional to the number of tokens it produces, which is not known when the
request arrives. A 20-token completion and a 500-token completion arrive looking identical. This
makes inference work fundamentally different from a typical web request: the service cannot assume
requests are short, cheap, or predictable.

**A model has a hard concurrency limit.** A single model instance on a single device can only do
so much work at once. Admitting 200 simultaneous requests to one model does not make them run in
parallel; it makes all 200 run slowly, thrash memory, and possibly exhaust the device. Concurrency
that exceeds the capacity of the executor is not throughput, it is queueing with extra steps —
except the queueing is happening implicitly inside the runtime, where it cannot be measured,
bounded, or prioritised.

**Unbounded queueing converts an overload into an outage.** If a service accepts every request and
buffers it, the buffer grows without limit under sustained overload. Every queued request ages.
Clients time out and retry, adding load. The service continues to do work whose results nobody is
waiting for any more. The queue grows faster. This is the classic congestion-collapse failure, and
it is entirely avoidable by bounding the queue and rejecting work the service cannot complete in
time. Rejecting a request quickly is a feature.

**Tail latency is the real latency.** Mean latency is close to useless for an inference service.
A system where the median request takes 400 ms and the 99th percentile takes 40 s is a system that
is failing for a meaningful fraction of its users, and the mean will not say so. Queue wait time,
not inference time, is usually what dominates the tail under load — which means the tail cannot be
diagnosed at all unless queue wait is measured separately from execution.

**Clients retry, and retries are dangerous.** A client that times out will usually retry. If the
original request is still executing, the service is now doing the same expensive work twice, at
exactly the moment it is least able to afford it. Retries amplify overload unless the service
either deduplicates them or makes them cheap to reject.

**Cancellation is not free.** When a client disconnects or a deadline expires, the tokens are
still being generated somewhere. Unless the execution path cooperates with cancellation, a
"timeout" only frees the client, not the server — so the server keeps paying for abandoned work
while new work waits behind it.

**Failures are normal.** Models raise. Out-of-memory happens. A worker dies mid-request. A backend
process becomes unreachable. The question is never whether these occur, but whether the platform
degrades predictably when they do, tells an operator what happened, and states clearly whether the
failed operation is safe to retry.

**None of this is visible without instrumentation.** An operator looking at an inference service
under load needs to answer: is the model slow, or is the queue deep? Are we rejecting traffic, and
why? How many workers are actually busy? Without those signals, every incident is guesswork.

The Mini Inference Platform exists to model this gap explicitly: the engineering that sits *around*
`model.generate()` and turns it into a service.

---

## 2. Product Goals

The platform must demonstrate the following capabilities, each observable and each tested:

| Goal | What it means concretely |
|---|---|
| HTTP inference API | A documented, validated JSON API for submitting generation requests |
| Asynchronous request handling | The API event loop is never blocked by model execution |
| Controlled concurrency | The number of requests reaching the model at once is bounded and configurable |
| Request queueing | Admitted work waits in an explicit, bounded, measurable queue |
| Scheduling | The policy choosing the next job is a named, replaceable component |
| Worker execution | A configurable pool of workers consumes the queue and executes inference |
| Model backend abstraction | The platform depends on an interface, not on Hugging Face |
| Rate limiting | Per-client request admission limits, enforced before work is queued |
| Backpressure | Overload produces fast, explicit rejection rather than unbounded latency |
| Timeout handling | Separate, enforced deadlines for queue wait and for execution |
| Safe retry behaviour | Every error response states whether the operation may be retried |
| Idempotency | Duplicate submissions are deduplicated, not re-executed |
| Graceful shutdown | In-flight and queued work is resolved deliberately, not dropped |
| Metrics | Queue depth, wait time, inference time, latency, errors, rejections |
| Benchmarking | Reproducible load generation producing machine-readable results |
| Failure testing | Deterministic failure injection with tests asserting system behaviour |

---

## 3. Non-Goals

V1 explicitly does **not** attempt to:

- **Replace or compete with vLLM, TGI, TensorRT-LLM, or SGLang.** Those are inference *engines*.
  This is a thin platform *around* an engine. The interesting content here is scheduling,
  admission and reliability, not kernel efficiency.
- **Implement CUDA kernels or any custom compute.**
- **Implement continuous batching or paged attention.** These are engine-level concerns. The
  platform is designed so a batching-capable backend could be slotted in later (§11).
- **Provide distributed inference** across machines or devices.
- **Provide multi-region or high-availability deployment.**
- **Provide durable queue guarantees.** V1 is single-process and in-memory. Work in flight is lost
  if the process dies. This is stated in the README, not hidden.
- **Implement model training or fine-tuning.**
- **Be a chatbot product.** There is no chat UI, no conversation memory, no prompt templating
  library, no agent framework.
- **Support arbitrary Hugging Face models.** V1 targets small causal language models that fit in
  host memory. Anything larger is a configuration problem for the operator.
- **Provide autoscaling or any Kubernetes integration.**
- **Provide authentication or multi-tenancy** beyond a client identifier used for rate limiting.

---

## 4. Target Users

**Primary: the engineer evaluating this repository.** The repository is a portfolio artifact. Its
job is to show that the author can design, build, measure and debug inference infrastructure. Every
decision below is made with the assumption that a reviewer will read the code and the measurements,
not just the README.

**Secondary: a developer who wants a local inference API with guardrails.** Someone running a small
model locally who wants a queue, a timeout and a metrics endpoint rather than a bare
`model.generate()` loop.

**Tertiary: an engineer learning inference-serving concepts.** The platform is small enough to read
end to end in an afternoon, and each mechanism (queue, scheduler, admission, deadline) is isolated
in its own module with tests that demonstrate the behaviour.

---

## 5. User Stories

**Client**

1. As a client, I can submit a prompt and generation parameters and receive generated text.
2. As a client, I receive a machine-readable error with a stable error code when my request is
   rejected, and that error tells me whether retrying is safe.
3. As a client, I receive a `Retry-After` hint when I am rejected for load or rate reasons.
4. As a client, I can supply an idempotency key so that a retry after a network failure does not
   cause the same expensive generation to run twice.
5. As a client, I can see how long my request spent waiting versus executing, so I can tell whether
   the service is slow or merely busy.

**Operator**

6. As an operator, I can see current queue depth, active workers and rejection counts.
7. As an operator, I can distinguish queue wait latency from inference latency in metrics.
8. As an operator, I can configure worker count, queue capacity, rate limits and timeouts without
   code changes.
9. As an operator, I can shut the service down without abandoning admitted work.
10. As an operator, I can read structured logs correlated by request ID across state transitions.
11. As an operator, I can distinguish "the process is alive" from "the process can serve traffic".

**Engineer**

12. As an engineer, I can run a load generator at configurable concurrency and get percentile
    latency, throughput, queue wait and error rates as machine-readable output.
13. As an engineer, I can inject deterministic failures (exceptions, slow responses, hangs) and
    assert how the platform responds.
14. As an engineer, I can replace the model backend without modifying the API, queue, scheduler or
    worker code.
15. As an engineer, I can reproduce every performance number in the documentation by running a
    script.

---

## 6. Functional Requirements

Identifiers are referenced by the technical design and by tests.

### 6.1 API

- **FR-API-1** `POST /generate` accepts a JSON body containing `prompt` (required), `max_tokens`,
  `temperature`, and optional `metadata`, and returns generated text plus timing information.
- **FR-API-2** All input is validated before any resource is consumed. Invalid input returns `422`
  with field-level detail and never reaches the queue.
- **FR-API-3** Prompt length and `max_tokens` are bounded by configuration. Requests exceeding the
  bounds are rejected at validation time.
- **FR-API-4** Every request is assigned a server-generated `request_id` (UUID4), returned in the
  response body and in the `x-request-id` response header, and present in every log line for that
  request.
- **FR-API-5** Every error response uses a single envelope shape containing an error `code`, a
  human-readable `message`, a `retryable` boolean and the `request_id`.
- **FR-API-6** Successful responses include `queue_wait_ms`, `inference_ms` and `total_ms`.
- **FR-API-7** The API accepts an optional `Idempotency-Key` header.
- **FR-API-8** The API accepts an optional `X-Client-Id` header used as the rate-limiting key,
  falling back to the peer address.

### 6.2 Queue

- **FR-Q-1** Admitted requests are placed on a bounded FIFO queue whose capacity is configurable.
- **FR-Q-2** Enqueue is non-blocking. If the queue is full the request is rejected immediately;
  the API never waits for queue space.
- **FR-Q-3** Queue depth is exported as a gauge and updated on every enqueue and dequeue.
- **FR-Q-4** Time spent in the queue is measured per request and exported as a histogram.
- **FR-Q-5** A job whose queue-wait deadline has expired is discarded at dequeue time rather than
  executed, and this is counted.

### 6.3 Scheduler

- **FR-S-1** The component that chooses the next job to execute is separate from the component that
  stores jobs, and is selected by configuration.
- **FR-S-2** V1 provides a FIFO policy: jobs execute in admission order.
- **FR-S-3** The scheduler interface is sufficient to express at least one non-FIFO policy without
  changing the worker pool or the API.

### 6.4 Workers

- **FR-W-1** A configurable number of workers consume from the scheduler concurrently.
- **FR-W-2** A worker executes exactly one job at a time. The count of workers is therefore the
  upper bound on concurrent inference.
- **FR-W-3** A worker that encounters an exception fails only the job it was executing, records the
  failure, and continues consuming subsequent jobs. A single bad request cannot kill the pool.
- **FR-W-4** Workers export a busy/idle state aggregated into an `active_workers` gauge.
- **FR-W-5** Workers stop cleanly on shutdown (§6.10).

### 6.5 Inference backend

- **FR-B-1** The platform defines an `InferenceBackend` interface with `startup`, `generate`,
  `health` and `shutdown`.
- **FR-B-2** No module outside `inference/` imports a model library.
- **FR-B-3** V1 ships an `echo` backend (deterministic, dependency-free, with failure injection)
  and a `huggingface` backend (local `transformers` causal LM).
- **FR-B-4** Backend execution that blocks must not block the event loop.
- **FR-B-5** A backend reports its own health, and the platform's readiness depends on it.

### 6.6 Rate limiter

- **FR-RL-1** Per-client rate limiting uses a token bucket with configurable sustained rate and
  burst size.
- **FR-RL-2** Rate limiting is evaluated before queue admission, so a limited client cannot consume
  queue capacity.
- **FR-RL-3** A rejected request receives `429` with a `Retry-After` header derived from the bucket
  state, and is marked retryable.
- **FR-RL-4** Rate-limiter state is bounded in size; an unbounded number of distinct client IDs
  must not exhaust memory.

### 6.7 Timeouts

- **FR-T-1** Three deadlines are distinguished and separately configurable: queue wait, inference
  execution, and total request deadline.
- **FR-T-2** A request that exceeds its queue-wait deadline is never executed.
- **FR-T-3** A request that exceeds its inference deadline is cancelled cooperatively where the
  backend supports it, and the client receives `504`.
- **FR-T-4** The limits of cancellation are documented per backend rather than assumed.

### 6.8 Idempotency

- **FR-I-1** Two requests with the same idempotency key and the same payload return the same result.
- **FR-I-2** A duplicate arriving while the original is still in flight attaches to the original
  result instead of enqueueing new work.
- **FR-I-3** A duplicate key with a *different* payload returns `409 Conflict`.
- **FR-I-4** Idempotency records expire after a configurable TTL and the store is bounded.
- **FR-I-5** Replayed responses are marked as replays in a response header.

### 6.9 Health and metrics

- **FR-H-1** `GET /health` reports process liveness and does not depend on the backend.
- **FR-H-2** `GET /ready` reports serving readiness, including backend health and shutdown state.
- **FR-H-3** `GET /metrics` exposes metrics in Prometheus text exposition format.
- **FR-H-4** `GET /metrics.json` exposes the same data as JSON for humans and benchmark tooling.
- **FR-H-5** `GET /stats` exposes a small operational snapshot (queue depth, workers, config).

### 6.10 Graceful shutdown

- **FR-GS-1** On shutdown the service stops admitting new requests and answers them with `503`.
- **FR-GS-2** Queued work is drained, up to a configurable grace period.
- **FR-GS-3** In-flight work is given the grace period to complete before cancellation.
- **FR-GS-4** Work cancelled by shutdown returns a retryable error, not a silent drop.
- **FR-GS-5** The backend's `shutdown` hook runs last.

---

## 7. Non-Functional Requirements

**Correctness.** Concurrency behaviour is asserted by tests, not by inspection. Specifically: FIFO
ordering, capacity enforcement, deadline enforcement, and single-flight idempotency are each covered
by a test that fails if the property is broken.

**Reliability.** No single request can terminate a worker, the pool, or the process. Every failure
path produces a defined HTTP status, a defined error code, and a metric.

**Observability.** Any question an operator would ask during an incident — is the queue deep, are we
rejecting, are workers busy, is the model slow — is answerable from `/metrics` alone. Logs are
structured and correlated by request ID.

**Testability.** The full platform must be testable without downloading a model. The echo backend
exists for this reason, and CI runs against it.

**Maintainability.** Modules are small and single-purpose. Abstractions exist only where a second
implementation is real or imminent (backends, schedulers). No plugin system, no service locator.

**Latency.** Platform overhead — the time a request spends in the platform excluding model
execution — should be small relative to inference time. The actual figure is measured in
§8 rather than asserted here.

**Throughput.** Throughput is bounded by the backend and the worker count. The platform must not
introduce a serialisation point that reduces throughput below what the backend can sustain. This is
verified by benchmark, not assumed.

**Resource limits.** Every unbounded collection is a bug. Queue, rate-limiter state and idempotency
store all have explicit caps and eviction.

---

## 8. Success Metrics

These are the quantities the platform measures and reports. **No target values are stated here.**
Targets would be fabrications until the system has been built and benchmarked; measured values live
in `docs/BENCHMARKS.md` and are generated from raw result files, never typed by hand.

**Throughput**
- Completed requests per second, at a stated concurrency and backend
- Generated tokens per second

**Latency** (end-to-end, client-observed)
- p50, p95, p99, max

**Latency decomposition** (server-measured, per request)
- Queue wait seconds — p50/p95/p99
- Inference seconds — p50/p95/p99
- Platform overhead = total − queue wait − inference

**Saturation**
- Queue depth over time (gauge, sampled)
- Active workers (gauge)

**Errors and load shedding**
- Error rate by error code
- Rejections by reason: rate limit, queue full, queue-wait timeout, shutdown
- Timeout count, separated into queue-wait timeouts and inference timeouts

**Correctness under duplication**
- Idempotency cache hits and in-flight coalesces

Each metric must be derivable from `/metrics` or from benchmark output. A metric that cannot be
produced by the running system does not belong in this list.

---

## 9. Failure Scenarios

Expected behaviour for each. The technical design (§17) restates this as an implementation-level
table with error codes and metric names.

| Scenario | Expected platform behaviour | Client sees | Retry safe? |
|---|---|---|---|
| Model raises during generation | Worker catches, fails that job only, increments backend error counter, continues | `500`, error code `backend_error` | No by default — work may have partially executed; safe with an idempotency key |
| Worker task raises outside generation | Worker loop catches, logs with worker ID, restarts its loop; pool stays at full size | Affected job gets `500` | No |
| Inference exceeds deadline | Cooperative cancellation where supported; job marked timed out; metric incremented | `504`, code `inference_timeout` | No by default; safe with idempotency key |
| Queue-wait deadline exceeded before execution | Job discarded at dequeue, never executed | `503`, code `queue_timeout` | **Yes** — the request provably never ran |
| Queue full at admission | Immediate rejection, no waiting | `503`, code `queue_full`, `Retry-After` | **Yes** — never admitted |
| Rate limit exceeded | Immediate rejection before queue admission | `429`, code `rate_limited`, `Retry-After` | **Yes** — never admitted |
| Client retries a request already in flight (same idempotency key) | Second request attaches to the first; no duplicate execution | Same result as the original, `x-idempotent-replay: coalesced` | n/a |
| Client retries a completed request (same idempotency key) | Cached result returned within TTL | Original result, `x-idempotent-replay: cached` | n/a |
| Same idempotency key, different payload | Rejected without execution | `409`, code `idempotency_conflict` | No — fix the request |
| Client disconnects mid-request | Job continues to completion if already executing; result discarded. Documented as a known inefficiency, not silently ignored | n/a | n/a |
| Shutdown while requests queued | Queue drained within grace period; remainder cancelled | `503`, code `shutting_down` | **Yes** |
| Shutdown while requests executing | Grace period to finish, then cancellation | `200` if finished, else `503` | **Yes** if cancelled |
| Backend unhealthy at startup | Startup fails loudly rather than serving broken traffic | Process exits non-zero | n/a |
| Backend becomes unhealthy while running | `/ready` fails; `/health` still succeeds | `503` on `/ready` | n/a |

---

## 10. V1 Scope

Built, tested and documented in V1:

1. FastAPI application with `/generate`, `/health`, `/ready`, `/metrics`, `/metrics.json`, `/stats`.
2. Pydantic request/response schemas with bounds enforced by configuration.
3. `InferenceBackend` interface with two implementations: `echo` (deterministic, failure-injecting)
   and `huggingface` (local `transformers` causal LM, executed off the event loop).
4. Bounded FIFO request queue with depth and wait-time instrumentation.
5. Scheduler abstraction with a FIFO implementation.
6. Configurable asyncio worker pool with per-worker state and failure isolation.
7. Token-bucket rate limiter with bounded state.
8. Admission control producing explicit backpressure.
9. Three-level timeout model with cooperative cancellation where the backend allows it.
10. In-memory idempotency store with TTL, single-flight coalescing and conflict detection.
11. Structured JSON logging correlated by request ID.
12. Hand-rolled metrics registry with Prometheus text and JSON exposition.
13. Graceful shutdown with draining.
14. Deterministic failure injection via the echo backend.
15. Unit, integration, concurrency and failure test suites.
16. Load generator supporting closed-loop and open-loop (arrival-rate) modes, emitting JSON.
17. A summariser that regenerates `docs/BENCHMARKS.md` from raw results.
18. One documented bottleneck investigation with before/after measurements.
19. Dockerfile, compose file, CI running lint and tests.
20. PRD, technical design, architecture notes, benchmark report, README.

Deliberately excluded from V1 despite being easy to add, because neither is justified by a measured
need: any external datastore (Redis, Postgres), any message broker, any container orchestration, any
metrics backend (Prometheus server, Grafana), and any frontend.

---

## 11. Future Scope

Candidate follow-on work, with the condition that would justify it. None of these are in V1.

| Future item | Justifying condition |
|---|---|
| vLLM / TGI backend | The single-model backend becomes the measured throughput ceiling and continuous batching is the fix |
| Dynamic / continuous batching | Benchmarks show worker-level parallelism is bounded by per-request overhead that batching would amortise |
| Streaming token responses (SSE) | Time-to-first-token matters to a real client, which it does not for a benchmark harness |
| Priority or fairness scheduling | A measured scenario where FIFO produces unacceptable tail latency for short requests behind long ones |
| Shortest-job-first approximation | `max_tokens` proves to be a usable proxy for job cost in measurement |
| Redis-backed queue | Durability across restarts, or more than one API process, becomes a requirement |
| Distributed worker pool | A single host stops being enough capacity |
| Kubernetes deployment + autoscaling | There is more than one instance and a real deployment target |
| Prometheus + Grafana | Someone actually needs dashboards and historical retention |
| GPU-aware scheduling / multi-device | More than one accelerator exists to schedule across |
| Persistent idempotency store | Deduplication must survive a process restart |
| Admission control by predicted cost | Measurement shows token-count-blind admission is the source of tail latency |

The discipline this table encodes: each item names the evidence that would justify building it.
Nothing here is built to look impressive.
