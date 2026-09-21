# Mini Inference Platform — Technical Design

Companion to [`docs/PRD.md`](PRD.md). The PRD says *what* and *why*; this document says *how*, in
enough detail that another engineer could implement it without further clarification.

---

## 1. Architecture Overview

```
                              Client
                                |
                                | HTTP/JSON
                                v
                    +-----------------------+
                    |        FastAPI        |   src/api/
                    |  validation, routing   |
                    +-----------+-----------+
                                |
                                v
                    +-----------------------+
                    |   Admission Control    |   src/reliability/
                    |  idempotency lookup     |
                    |  rate limiter (token    |
                    |  bucket)                |
                    +-----------+-----------+
                                | accepted
                                v
                    +-----------------------+
                    |    Request Queue       |   src/queue/
                    |  bounded asyncio.Queue |
                    |  FIFO, capacity-limited|
                    +-----------+-----------+
                                |
                                v
                    +-----------------------+
                    |      Scheduler         |   src/scheduler/
                    |  FIFO policy (v1)      |
                    +-----------+-----------+
                                |
              +-----------------+------------------+
              |                 |                   |
              v                 v                   v
        +-----------+     +-----------+       +-----------+
        | Worker 1  |     | Worker 2  |  ...  | Worker N  |   src/workers/
        +-----+-----+     +-----+-----+       +-----+-----+
              |                 |                   |
              +-----------------+-------------------+
                                |
                                v
                    +-----------------------+
                    |   InferenceBackend      |   src/inference/
                    |  echo | huggingface     |
                    +-----------------------+

   Cross-cutting, touching every layer above:
   +----------------------------------------------------------+
   |  Observability (src/observability/)                       |
   |   - structured logs, correlated by request_id             |
   |   - metrics registry -> /metrics (Prometheus text), /metrics.json |
   +----------------------------------------------------------+
   |  Configuration (src/config.py) — env-driven, one object    |
   +----------------------------------------------------------+
```

Every arrow above is an `await`. There is one OS process, one asyncio event loop, and no network
hop until the (optional) HTTP call inside `HuggingFaceBackend` — which there isn't; the HF backend
runs the model in-process, off the loop, via `run_in_executor`. The only real network boundary is
the client's HTTP connection to FastAPI.

---

## 2. Request Lifecycle

A single `POST /generate` request moves through these states. Each transition is logged with
`request_id` and emits or updates a metric.

```
1.  ARRIVED        HTTP request received by FastAPI.
2.  VALIDATED       Pydantic schema validation. Failure -> 422, terminal, no request_id cost beyond logging.
3.  IDENTIFIED       request_id generated (uuid4). Client-supplied Idempotency-Key (if any) read.
4.  IDEMPOTENCY_CHECK  Look up key in IdempotencyStore.
       -> HIT (cached, complete)   -> return cached response immediately. terminal.
       -> HIT (in-flight)          -> await the existing future. terminal (shares result).
       -> HIT (conflict, diff body) -> 409. terminal.
       -> MISS                     -> continue, reserve the key.
5.  RATE_LIMIT_CHECK   Token bucket for the client key.
       -> DENIED -> 429 + Retry-After. terminal.
       -> OK     -> continue.
6.  ADMISSION (queue offer)  Non-blocking put onto the bounded queue.
       -> QUEUE FULL -> 503 (queue_full). terminal.
       -> ENQUEUED   -> continue. queue_depth += 1. enqueued_at = now().
7.  QUEUED            Waiting for a worker. Bounded by queue_wait_timeout.
       -> WAIT TIMEOUT (popped past deadline) -> 503 (queue_timeout). terminal. job never runs.
       -> DEQUEUED by scheduler -> continue. queue_wait_seconds recorded. queue_depth -= 1.
8.  SCHEDULED         Scheduler hands the job to a free worker.
9.  EXECUTING         Worker calls backend.generate(), bounded by inference_timeout.
       -> BACKEND EXCEPTION -> 500 (backend_error). terminal.
       -> TIMEOUT           -> cancel attempted, 504 (inference_timeout). terminal.
       -> SUCCESS           -> continue. inference_seconds recorded.
10. COMPLETED         Result stored in IdempotencyStore (if a key was supplied). Response built.
11. RESPONDED         HTTP response sent. total_seconds recorded. metrics flushed.
```

Steps 4-6 happen inside the API coroutine, before the request ever touches the queue — this is
deliberate: rejecting a request that will never be admitted must not cost a queue slot or a worker.

---

## 3. Core Components

### API Layer (`src/api/`)
- `routes.py` — route handlers; no business logic beyond orchestrating the calls below.
- `schemas.py` — Pydantic models: `GenerateRequest`, `GenerateResponse`, `ErrorResponse`.
- `dependencies.py` — FastAPI dependency providers that hand routes the shared `AppState`
  (queue, scheduler, rate limiter, idempotency store, metrics, config) constructed once at
  startup via the lifespan context.

Responsibility: HTTP concerns only — parsing, status codes, headers. It calls into
`reliability/` and `queue/`; it does not know how a job is executed.

### Request Queue (`src/queue/request_queue.py`)
Wraps `asyncio.Queue[Job]` with a fixed `maxsize`. Responsibility: hold admitted work and expose
depth. It does not decide *which* job runs next beyond FIFO storage order — that decision belongs
to the scheduler, which in v1 happens to also be FIFO, kept as a separate class so the boundary is
real (see §6).

### Admission Controller (`src/reliability/`, orchestrated from `routes.py`)
Not a single class — a sequence of two checks (idempotency, rate limit) applied before the queue
offer. Kept as discrete, independently testable units rather than one "AdmissionController" god
object, because each has a different failure mode and a different owner conceptually (idempotency
is about correctness, rate limiting is about fairness).

### Rate Limiter (`src/reliability/rate_limiter.py`)
`TokenBucketLimiter`. One bucket per client key, created lazily, stored in a bounded LRU dict
(`RATE_LIMIT_MAX_CLIENTS`, default 10_000; oldest-idle evicted first). Thread-unsafe by design —
single event loop, no locks needed.

### Scheduler (`src/scheduler/`)
`base.py` defines `Scheduler` (an ABC: `submit(job)`, `next_job() -> Job`). `fifo.py` implements it
directly on top of `RequestQueue`. Workers depend on `Scheduler`, never on `RequestQueue` directly
— this is the seam a smarter policy would occupy later (§6).

### Worker Pool (`src/workers/`)
`worker.py` — a single worker's loop: pull a job from the scheduler, check its deadline, execute it
against the backend, publish the result, repeat. `pool.py` — starts/stops `N` worker coroutines as
asyncio tasks, tracks per-worker busy/idle state for the `active_workers` gauge.

### Inference Backend (`src/inference/`)
`base.py` — the `InferenceBackend` ABC. `echo.py` — dependency-free deterministic backend with
configurable latency and failure injection, used by every test that isn't specifically testing the
HF backend. `huggingface.py` — loads a small causal LM via `transformers` at `startup()`, runs
`generate` in a thread executor.

### Metrics (`src/observability/metrics.py`)
A small hand-rolled registry (counters, gauges, histograms) with no external dependency, exported
in Prometheus text format and as JSON. See §15 for why this is hand-rolled rather than
`prometheus_client`.

### Idempotency Store (`src/reliability/idempotency.py`)
`IdempotencyStore` — `dict[key -> Record]`, `Record` is `{payload_hash, status, result_or_future,
expires_at}`. In-flight requests are represented by an `asyncio.Future` stored under the key so
concurrent duplicates `await` the same future instead of re-running the job.

### Configuration (`src/config.py`)
One `Settings` (Pydantic `BaseSettings`) object, populated from environment variables, constructed
once at process startup, passed down explicitly. No global mutable config object imported ad hoc
around the codebase.

---

## 4. Concurrency Model

Everything lives on one asyncio event loop in one process. Three distinct notions of "concurrency"
exist and are deliberately not conflated:

1. **API concurrency** — how many HTTP requests FastAPI is handling at once. Effectively unbounded
   (limited only by the ASGI server's connection handling), because validation, idempotency lookup
   and rate-limit checks are all cheap, non-blocking, in-memory operations. This is fine: none of
   this work touches the model.

2. **Queued concurrency** — how many admitted jobs are waiting, bounded by `QUEUE_CAPACITY`. This
   is where the system absorbs a burst: work above worker capacity waits here rather than piling
   onto the model or the event loop.

3. **Inference concurrency** — how many jobs are actually executing against the backend at once.
   Hard-bounded by `WORKER_COUNT`, because each worker processes exactly one job at a time
   (FR-W-2). This is the number that actually protects the model.

**What prevents unlimited requests from reaching the model:** the worker pool is the only code path
that calls `backend.generate()`, there are exactly `WORKER_COUNT` worker coroutines, and each
worker's loop is strictly sequential (`await job; await execute(job); repeat`). There is no
`asyncio.gather` or `create_task` anywhere that would let a worker start a second job before
finishing its first. Concurrency into the backend is therefore bounded by construction, not by
convention.

The `HuggingFaceBackend.generate()` call is CPU-bound (PyTorch on CPU releases the GIL during the
C++ compute, but the call itself is synchronous Python). It runs via
`loop.run_in_executor(thread_pool, ...)` so it does not block the event loop while it runs — other
API requests (validation, health checks, rejections) stay responsive even while all workers are
executing inference. The executor's thread pool is sized to `WORKER_COUNT`, matching the intended
inference concurrency exactly; it is not a second, independent concurrency knob.

---

## 5. Queue Design

- **Type:** `asyncio.Queue`, wrapped rather than subclassed, in `RequestQueue`. Justification in
  §23 — no compelling reason to start with anything heavier for a single-process system.
- **Capacity:** fixed at construction from `QUEUE_CAPACITY` (default 64). `asyncio.Queue(maxsize=N)`.
- **Enqueue:** `put_nowait`. Never `await put(...)`, because an awaiting put would silently turn
  "queue full" into "client request hangs" — exactly the failure mode §11 of the PRD forbids.
  `QueueFull` is caught and converted to a `503`.
- **Dequeue:** workers `await queue.get()`. On dequeue, the job's enqueued timestamp is compared
  against `now() - QUEUE_WAIT_TIMEOUT`; if already expired, the job is discarded (not executed) and
  counted under `request_rejections_total{reason="queue_timeout"}`, and the worker immediately pulls
  the next job. This is a pull-side check because `asyncio.Queue` has no way to expire entries in
  place without a second data structure and a background sweep — checking on dequeue is simpler and
  sufficient, since a job cannot be executed before it is dequeued anyway.
- **Queue-full behaviour:** immediate `503`, `Retry-After` computed from a short fixed backoff
  (not derived from queue depth in v1 — see limitation in §17).
- **Metrics:** `queue_depth` (gauge, updated on every put/get), `queue_wait_seconds` (histogram,
  recorded on dequeue), `request_rejections_total{reason="queue_full"}`.

---

## 6. Scheduler Design

V1 policy: **FIFO**, implemented directly as a thin pass-through over `RequestQueue.get()`.

The separation from the queue exists so a future scheduler can change *selection* without changing
*storage*:

```python
class Scheduler(ABC):
    def submit(self, job: Job) -> None: ...
    async def next_job(self) -> Job: ...
```

`FIFOScheduler` implements this with one `RequestQueue` instance. A future `PriorityScheduler`
could hold several internal queues (by priority tier) and implement `next_job` as "check high
priority first, else low priority" without the worker pool or API changing at all — workers only
ever call `scheduler.next_job()`. A shortest-job-first approximation could similarly reorder based
on `max_tokens` as a proxy for cost. Neither is built in v1 (see PRD §11); the point of this design
is that they *could* be, in isolation.

---

## 7. Worker Design

**Lifecycle:** created as an `asyncio.Task` running `Worker.run()` when the app starts;
cancelled and awaited when the app shuts down.

**Job acquisition:** `job = await scheduler.next_job()`. This suspends the worker until a job is
available — an idle worker consumes no CPU.

**Failure propagation:** `Worker.run()` wraps each job's execution in a `try/except Exception`
that is scoped to *that job only*. An exception increments
`backend_errors_total`, completes the job's result future with the exception (so any idempotency
waiter also sees the failure), logs it with the job's `request_id` and the worker's ID, and the loop
continues to the next job. The outer `run()` loop itself only exits on `asyncio.CancelledError`
(shutdown) — a bug in job handling cannot silently shrink the pool.

**Cancellation:** shutdown cancels each worker's task. If a worker is mid-`await
backend.generate(...)`, cancellation propagates into that await; the `HuggingFaceBackend` executor
call is wrapped so cancellation at minimum stops *waiting* on the thread (the underlying model call
in a thread pool cannot be forcibly killed — documented honestly in §12, not glossed over).

**Health state:** each `Worker` exposes `.state` (`idle` / `busy`) read by the pool to compute the
`active_workers` gauge; no locking needed since only the worker itself writes its own state and the
gauge read is eventually-consistent by design (it's a monitoring signal, not a scheduling input).

**Graceful shutdown:** see §18 for the pool-level sequencing.

---

## 8. Model Backend Interface

```python
class InferenceBackend(ABC):
    async def startup(self) -> None: ...
    async def generate(self, request: InferenceJob) -> InferenceResult: ...
    async def health(self) -> bool: ...
    async def shutdown(self) -> None: ...
```

No module outside `src/inference/` imports `transformers`, `torch`, or any model-specific library
(enforced informally by code review / grep in CI, not by a hard runtime boundary — acceptable for a
project this size).

**`EchoBackend`** — deterministic: returns a fixed transformation of the prompt (e.g. word count +
truncated echo), with a configurable artificial delay (to simulate inference time under load) and a
configurable failure-injection hook (raise / hang / slow-respond, keyed by a `metadata.fail_mode`
field on the request). This is what every test that isn't specifically about the HF backend runs
against — it makes tests fast, deterministic and free of a model download.

**`HuggingFaceBackend`** — loads a small causal LM (`AutoModelForCausalLM`,
`AutoTokenizer`) at `startup()`, keeps it resident, and executes `model.generate(...)` inside a
`ThreadPoolExecutor` via `run_in_executor` on each call. `health()` returns `True` once the model
and tokenizer are loaded. Runs on CPU in v1 (see §23 tradeoff note); nothing in the interface
prevents a CUDA-backed instance later, it is a config flag away.

---

## 9. Request Model

```python
class GenerateRequest(BaseModel):
    prompt: str                    # required, 1..MAX_PROMPT_CHARS
    max_tokens: int = 128          # 1..MAX_TOKENS_LIMIT
    temperature: float = 0.7       # 0.0..2.0
    metadata: dict | None = None   # small, opaque, passed to backend (used by echo for fail injection)
```

`request_id`, `queue_wait_ms`, `inference_ms`, `total_ms` are server-assigned/measured, not
client-supplied. No `model` field in v1 — one backend instance per running process; selecting a
model is a deployment-time config choice, not a per-request one (keeps the request model small,
per PRD guidance).

---

## 10. Rate Limiting

**Algorithm: token bucket**, one per client key (`X-Client-Id` header, else peer IP).

- Bucket has capacity `RATE_LIMIT_BURST` and refills at `RATE_LIMIT_RPS` tokens/second,
  lazily computed on each check (`tokens = min(capacity, tokens + elapsed * rps)`) rather than via a
  background timer — no extra task, no drift accumulation, correct under any request pattern.
- A request consumes 1 token. If none available, reject with `429` and
  `Retry-After: ceil((1 - tokens) / rps)`.
- Chosen over a fixed window because token bucket tolerates short bursts (useful for a benchmark
  client ramping concurrency) while still bounding sustained rate — a fixed window either forbids
  all bursting or allows a 2x burst at window boundaries, both worse fits here.

---

## 11. Backpressure

**Rate limiting** answers "is this *client* sending too fast" — a per-key, fairness concern,
independent of overall system load. **Backpressure** answers "is the *system* currently able to
accept more work at all" — a global capacity concern. A single well-behaved client can trigger
backpressure (e.g. queue is full because of other clients); a misbehaving client triggers rate
limiting even when the system is idle. They are checked separately and rejected with different
codes so the client (and an operator reading logs) can tell which one happened.

- **Rate limit exceeded → `429 Too Many Requests`.** Semantically about the client's request rate.
  A `Retry-After` tied to the bucket's own refill rate.
- **Queue full / overloaded → `503 Service Unavailable`.** Semantically "the server cannot handle
  this right now," independent of which client is asking. `Retry-After` is a short fixed value in
  v1 (not derived from measured drain rate — see §17 limitation).

This mapping follows the HTTP semantics most inference-adjacent services (and RFC 6585 for 429)
already use, rather than inventing a bespoke pair of codes.

---

## 12. Timeout Model

Three deadlines, deliberately not collapsed into one:

1. **Queue-wait timeout** (`QUEUE_WAIT_TIMEOUT`, default 10s) — how long a job may sit admitted but
   unexecuted. Enforced by the pull-side check in §5. If exceeded, the job is *never executed* —
   the cleanest of the three cases, because no partial work exists.

2. **Inference timeout** (`INFERENCE_TIMEOUT`, default 30s) — how long a single `generate()` call
   may run once a worker has picked it up. Enforced with `asyncio.wait_for` around the backend call.

3. **Total request deadline** (`REQUEST_TIMEOUT`, default = queue-wait + inference timeout,
   overridable) — the client-facing ceiling from HTTP arrival to response, mostly useful as a
   documentation figure for capacity planning rather than a separately enforced mechanism (it falls
   out of the two above by construction).

**On cancellation being non-trivial:** when the inference timeout fires against the
`EchoBackend`, cancellation is real and immediate — it's a Python coroutine with an `await
asyncio.sleep(...)` that responds correctly to `CancelledError`. Against `HuggingFaceBackend`,
`model.generate()` runs inside a worker thread via the executor; `asyncio.wait_for` cancels the
*awaiting* coroutine and returns control to the worker immediately (so the worker is freed to take
the next job and the client gets its `504` on time), but the underlying thread keeps running the
model call to completion in the background and its result is discarded — CPython cannot forcibly
interrupt a thread mid-`torch` call. This is a real, documented limitation, not a simulated one: it
means a stream of inference timeouts can still exhaust CPU even though the platform "cancelled"
them. It's the same limitation real GPU inference servers have for kernel-level cancellation, and
it is exactly why the PRD does not claim GPU/model work can be cancelled trivially.

---

## 13. Retry Policy

The platform never retries on the client's behalf inside a single request — retries are a client
decision, informed by the `retryable` flag in every error response (FR-API-5). The platform's own
job is to make that flag correct:

- `retryable: true` — the request provably never began execution (`queue_full`, `rate_limited`,
  `queue_timeout`, `shutting_down`). Retrying costs nothing extra and cannot double-execute.
- `retryable: false` — the request may have partially or fully executed (`backend_error`,
  `inference_timeout`). Blind retry risks duplicate expensive work. The response instead
  recommends attaching an `Idempotency-Key` up front so a retry becomes safe by construction (§14)
  rather than the platform guessing.

No automatic internal retry (e.g. worker silently re-attempting a failed generate call) exists in
v1: a backend failure is far more likely to be systematic (bad input, OOM) than transient, and
retrying automatically would silently double the cost of exactly the requests that are already
failing under load.

---

## 14. Idempotency

Optional `Idempotency-Key` header. Behaviour, keyed by `(key)` with the request body hash stored
alongside for conflict detection:

- **No key supplied:** no deduplication; every call executes. Default behaviour, zero overhead.
- **New key:** reserved immediately (before queueing) with status `in_flight` and an
  `asyncio.Future`; the request proceeds through the normal pipeline; on completion the future is
  resolved and the record transitions to `complete` with the result cached until TTL
  (`IDEMPOTENCY_TTL`, default 10 minutes).
- **Key exists, status `in_flight`, same body hash:** the duplicate `await`s the existing future
  instead of entering the queue. Response header `x-idempotent-replay: coalesced`.
- **Key exists, status `complete`, same body hash, within TTL:** cached result returned directly,
  no queue involvement. Response header `x-idempotent-replay: cached`.
- **Key exists, different body hash:** `409 Conflict` — the client is reusing a key for a different
  request, which is a client bug, not a retry.

**Explicitly not durable.** The store is an in-memory `dict`. A process restart loses all records —
documented in the README and here, not hidden. This is acceptable for v1 because the platform is
already single-process with an in-memory queue; durable idempotency without a durable queue would
be a false guarantee.

---

## 15. Observability

**Metrics** (hand-rolled registry, no `prometheus_client` dependency — see §23 for why):

| Metric | Type | Labels |
|---|---|---|
| `requests_total` | counter | `status` (2xx/4xx/5xx bucket) |
| `request_errors_total` | counter | `code` (error code) |
| `request_rejections_total` | counter | `reason` (rate_limited, queue_full, queue_timeout, shutting_down) |
| `queue_depth` | gauge | — |
| `queue_wait_seconds` | histogram | — |
| `inference_seconds` | histogram | — |
| `request_latency_seconds` | histogram | — |
| `active_workers` | gauge | — |
| `completed_requests_total` | counter | — |
| `idempotency_replays_total` | counter | `kind` (cached, coalesced) |

Exposed at `/metrics` (Prometheus text exposition format, hand-formatted) and `/metrics.json`
(same data, JSON, consumed by the benchmark summariser and easier to eyeball in a browser).

**Structured logs** — one JSON line per state transition worth knowing about (admitted, rejected,
dequeued, completed, failed), each containing at minimum: `request_id`, `event`, `worker_id`
(where applicable), timestamps/durations relevant to that event, and `error` (where applicable).
Configured via `src/observability/logging.py` using the stdlib `logging` module with a JSON
formatter — no external logging service.

---

## 16. Health Endpoints

- **`GET /health`** — liveness only: "is the process able to respond at all." Always `200` unless
  the process is unable to serve HTTP, in which case it wouldn't respond anyway. Does not touch the
  backend or the queue. Used by an orchestrator to decide "restart this process," so it must not
  fail for reasons a restart wouldn't fix.
- **`GET /ready`** — readiness: `200` only if the backend's `health()` returns true *and* the
  service is not draining for shutdown. Used to decide "route traffic here."
- **`GET /metrics`**, **`GET /metrics.json`** — see §15.
- **`GET /stats`** — a convenience snapshot (queue depth, active workers, uptime, config summary)
  for a human looking at the service, not a monitoring integration.

---

## 17. Failure Handling

| Failure | Location | Behaviour | Client response | Retryable? | Signal |
|---|---|---|---|---|---|
| Invalid request body | API validation | Rejected before any resource used | `422` | No — fix request | `requests_total{status=4xx}` |
| Rate limit exceeded | Admission | Rejected before queue | `429` + `Retry-After` | Yes | `request_rejections_total{reason=rate_limited}` |
| Queue full | Admission | `put_nowait` raises `QueueFull`, caught | `503` + `Retry-After` | Yes | `request_rejections_total{reason=queue_full}` |
| Queue-wait timeout | Worker, pre-execution | Job discarded, not run | `503` | Yes | `request_rejections_total{reason=queue_timeout}` |
| Duplicate key, in flight | Admission | Attach to existing future | same as original | n/a | `idempotency_replays_total{kind=coalesced}` |
| Duplicate key, completed | Admission | Cached result returned | same as original | n/a | `idempotency_replays_total{kind=cached}` |
| Duplicate key, conflicting body | Admission | Rejected | `409` | No | `request_errors_total{code=idempotency_conflict}` |
| Backend raises | Worker, execution | Job failed, worker continues | `500` | No (safe with idempotency key) | `request_errors_total{code=backend_error}` |
| Inference timeout | Worker, execution | Await cancelled; underlying thread may continue (§12) | `504` | No (safe with idempotency key) | `request_errors_total{code=inference_timeout}` |
| Worker task itself raises (bug) | Worker loop | Caught at loop level, job failed, loop continues | `500` | No | logged with `worker_id`, `request_errors_total{code=backend_error}` |
| Backend unhealthy at startup | App startup | Startup aborts, process exits non-zero | n/a (never serves) | n/a | startup log |
| Backend unhealthy while running | Background / on `/ready` check | `/ready` returns `503` | depends on caller | n/a | readiness probe failure |
| Shutdown in progress | Admission | New requests rejected immediately | `503` | Yes | `request_rejections_total{reason=shutting_down}` |

---

## 18. Graceful Shutdown

Triggered by SIGTERM/SIGINT via FastAPI's lifespan shutdown, or `POST /admin/shutdown` in tests.

1. **Stop admitting.** A `draining` flag flips first. Every subsequent request to `/generate`
   short-circuits to `503 shutting_down` before touching the rate limiter or queue — cheap and
   immediate.
2. **Drain the queue.** Already-queued jobs continue to be dequeued and executed by workers for up
   to `SHUTDOWN_GRACE_SECONDS` (default 10s).
3. **Let in-flight work finish.** Jobs already executing are not interrupted by the drain step; they
   share the same grace period budget.
4. **Hard stop.** When the grace period elapses, remaining worker tasks are cancelled; any job still
   queued or mid-execution completes its response future with a `shutting_down` error (retryable).
5. **Backend shutdown.** `backend.shutdown()` is awaited last, after all workers have stopped, so the
   model is never torn down while a worker might still call it.

`/ready` starts returning `503` the instant draining begins (step 1), before the queue is even
touched — so a load balancer stops sending new traffic immediately while existing work is still
honoured.

---

## 19. Configuration

Single `Settings` object (`src/config.py`, Pydantic `BaseSettings`), environment-variable driven,
constructed once at process startup:

| Variable | Default | Meaning |
|---|---|---|
| `WORKER_COUNT` | 4 | Concurrent inference slots |
| `QUEUE_CAPACITY` | 64 | Max admitted-but-unexecuted jobs |
| `QUEUE_WAIT_TIMEOUT` | 10.0s | Max time a job waits before being discarded unexecuted |
| `INFERENCE_TIMEOUT` | 30.0s | Max time a single generate() call may run |
| `RATE_LIMIT_RPS` | 5.0 | Token bucket refill rate per client |
| `RATE_LIMIT_BURST` | 10 | Token bucket capacity per client |
| `RATE_LIMIT_MAX_CLIENTS` | 10000 | Bounded bucket-map size |
| `IDEMPOTENCY_TTL` | 600s | How long a completed result is replayed |
| `MAX_PROMPT_CHARS` | 4000 | Input validation bound |
| `MAX_TOKENS_LIMIT` | 512 | Upper bound on requested `max_tokens` |
| `SHUTDOWN_GRACE_SECONDS` | 10.0s | Drain window on shutdown |
| `BACKEND` | `echo` | `echo` \| `huggingface` |
| `MODEL_NAME` | `sshleifer/tiny-gpt2` | HF backend model id (only used if `BACKEND=huggingface`) |
| `LOG_LEVEL` | `INFO` | Python logging level |

Deliberately not configurable in v1: scheduler policy (only FIFO exists), metrics backend (only the
built-in registry exists). Adding a config knob for a choice that doesn't exist yet is sprawl.

---

## 20. Testing Strategy

- **Unit** (`tests/unit/`) — queue capacity and FIFO order in isolation; token bucket math; the
  idempotency store's state machine; scheduler `submit`/`next_job`; config parsing; metrics
  registry increments.
- **Integration** (`tests/integration/`) — full app via `httpx.AsyncClient` against the FastAPI app
  with the `EchoBackend`: `/generate` happy path, validation errors, health/ready semantics.
- **Concurrency** (`tests/integration/test_concurrency.py`) — N concurrent requests with
  `WORKER_COUNT=1` and an echo backend with artificial delay, asserting they complete in submission
  order and that `active_workers` never exceeds the configured count.
- **Failure** (`tests/integration/test_failures.py`) — echo backend's fail-injection modes:
  exception, artificial timeout, verifying the correct status code, `retryable` flag and metric
  increment for each.
- **Load** (`tests/load/`) — not pytest-based; the benchmark tool itself (§21), run manually /
  from a benchmark CI job, not part of the standard `pytest` gate (too slow/flaky for that).

Explicitly covered, per PRD §11 testing rules: queue full, concurrent requests, worker failure
isolation, both timeout types, idempotent duplicate (coalesced and cached), rate limiting,
graceful shutdown draining behaviour. No test depends on network access or a paid API — the HF
backend's own tests are marked and skipped if the model cache is unavailable.

---

## 21. Benchmark Strategy

`benchmarks/benchmark.py` is an independent async HTTP load generator (not using the app's internals
— it talks to a running server over real HTTP, so results reflect the whole stack including ASGI).

- **Modes:** closed-loop (N concurrent workers each issuing request-after-response) is the v1
  default; documented as closed-loop specifically because that distinction matters for interpreting
  results (closed-loop concurrency caps *offered* load at N in-flight, which is different from an
  open-loop fixed arrival rate).
- **Parameters:** `--concurrency`, `--requests` (total, or `--duration`), `--prompt-tokens`
  (approximate, via a repeated filler token), `--max-tokens`, `--base-url`.
- **Per-request capture:** timestamp, request_id, HTTP status, queue_wait_ms and inference_ms (read
  from the response body, which the API always includes), total latency measured client-side, and
  whether it errored.
- **Output:** one JSON-lines file per run under `benchmarks/results/`, named by timestamp and
  concurrency — raw data, never hand-edited.
- **Summary generation:** `benchmarks/summarize.py` reads the JSONL files and computes RPS, p50/p95/
  p99, mean queue wait, mean inference time, tokens/sec, error rate — printed and used to regenerate
  `docs/BENCHMARKS.md`. The doc is generated, not authored, specifically so numbers in it can never
  drift from a real run.
- **Concurrency sweep:** the standard run sweeps `{1, 2, 4, 8, 16, 32}` against a fixed backend and
  worker count, producing one file per level.

No numbers are pre-committed to documentation before the corresponding run exists in
`benchmarks/results/`.

---

## 22. Security Considerations

- **Input validation:** prompt length and `max_tokens` are bounded (§19); Pydantic rejects wrong
  types before any processing. There is no path from request content to shell execution, file
  paths, or code evaluation.
- **Request size:** FastAPI/uvicorn's default body-size handling applies; `MAX_PROMPT_CHARS` gives
  an additional application-level bound well below anything that would pressure memory.
- **Resource exhaustion:** every unbounded-growth risk is explicitly capped — queue (`QUEUE_CAPACITY`),
  rate-limiter bucket map (`RATE_LIMIT_MAX_CLIENTS`), idempotency store (TTL eviction). This is
  treated as a correctness requirement, not just a performance nicety (PRD §7).
- **Unsafe model inputs:** the echo backend cannot execute anything from the prompt by construction.
  The HF backend passes the prompt to a tokenizer and `generate()` only — no templating engine, no
  code execution, no tool-calling surface in v1, so prompt injection has no privileged action to
  escalate into.
- **Logging sensitive prompts:** prompts are user data and may be sensitive. Structured logs record
  prompt *length*, not prompt *content*, by default; full-prompt debug logging is opt-in via
  `LOG_LEVEL=DEBUG` and documented as such, so operators make that call deliberately.
- **No authentication in v1** — `X-Client-Id` is self-asserted and only used for fairness (rate
  limiting), not as an identity/authorization boundary. Documented as a limitation, not glossed
  over: this platform is designed for local/trusted-network use.

---

## 23. Tradeoffs

**`asyncio.Queue` vs Redis.** A Redis-backed queue would survive a process restart and support
multiple API processes sharing one queue. Neither is a v1 requirement — this is explicitly a
single-process system (PRD non-goals). Redis would add an external dependency, a new failure mode
(Redis down), and network latency on every enqueue/dequeue, to solve a durability problem the rest
of the system (in-memory idempotency store, in-process model weights) doesn't have either. Revisit
if the platform ever needs more than one process.

**asyncio vs Celery/RQ.** Celery is built for distributing work across processes/machines via a
broker. This system's bottleneck is a single model instance in a single process — there is nothing
to distribute yet. asyncio gives full control over exactly the mechanisms this project is meant to
demonstrate (queue, scheduler, worker pool, cancellation); Celery would hide all of them behind its
own abstractions, defeating the point of the project.

**Single process vs distributed.** A single process is sufficient to explore and measure every
reliability/scheduling concept in scope (queueing, backpressure, timeouts, idempotency,
observability). Distribution introduces partial failure, consensus and network reliability
concerns that are a different (and much larger) project. Documented as a hard v1 boundary, not an
oversight.

**FIFO vs a smarter scheduler.** FIFO is the correct default: it's fair, it's trivial to reason
about, and it's what a reviewer expects unless there's a measured reason to deviate. The scheduler
interface (§6) is deliberately factored out so a smarter policy is an isolated, testable addition
later — but building priority/SJF scheduling without a workload that needs it would be optimizing
without measurement, which PRD §2 explicitly rules out.

**In-memory idempotency vs a persistent store.** Matches the queue's own durability story — a
persistent idempotency store bolted onto an in-memory queue would promise a guarantee ("your
duplicate won't double-execute across a restart") the rest of the system can't back up, since the
original request's queued state is lost on restart regardless.

**Hand-rolled metrics vs `prometheus_client`.** `prometheus_client` is a fine library, but a small
hand-rolled counter/gauge/histogram registry (~100 lines) makes the metric definitions and the
Prometheus text-format encoding fully visible in this codebase rather than delegated to a
dependency — appropriate given that demonstrating observability instrumentation is itself part of
what this project is for. JSON export alongside it is what the benchmark summarizer actually
consumes, since parsing Prometheus text format for that purpose would be pure overhead.

**CPU vs GPU for the HuggingFace backend.** v1 runs the reference model on CPU. This keeps the
"real model" path runnable in CI and on any contributor's machine without a CUDA toolchain, and
— usefully for this project's actual purpose — CPU inference is slow enough relative to
platform overhead that queueing/scheduling behaviour is easy to observe and benchmark. GPU support
is a config change away in the backend, not an architectural one.

---

## 24. Future Architecture

None of the following are built in v1 (see PRD §11 for the evidence that would justify each); this
section is about *how the current design leaves room for them*, not a commitment to build them.

- **vLLM backend:** implement `InferenceBackend` against vLLM's engine. Nothing above
  `src/inference/` would need to change — the worker pool already treats `generate()` as an opaque
  async call.
- **Redis-backed queue:** implement `Scheduler`/`RequestQueue`'s storage against Redis (e.g.
  `RPUSH`/`BLPOP`) behind the same `submit`/`next_job` interface. Workers and the API are unaffected.
- **Distributed schedulers / worker pools:** would require moving from "workers are asyncio tasks
  in this process" to "workers are separate processes/machines pulling from a shared, durable
  queue" — this is exactly why the queue and scheduler are already separate abstractions from the
  worker pool, even though v1's implementations are both in-process.
- **GPU worker pools:** `WORKER_COUNT` today maps 1:1 to "concurrent generate() calls"; a
  GPU-aware version would map workers to physical devices and the pool would need device-assignment
  logic, but the *interface* workers call (`InferenceBackend.generate`) does not change.
- **Kubernetes:** the app is already a stateless-except-for-in-memory-queue process behind
  `/health`/`/ready`; the honest blocker to a multi-replica deployment today is precisely the
  in-memory queue/idempotency store, which is why those are called out as the first things to
  replace, not autoscaling itself.
- **Dynamic batching:** would live entirely inside a new `InferenceBackend` implementation (collect
  N pending jobs, run one batched `generate()` call, split results) — the worker pool would need a
  batching-aware variant of `next_job()` (pull up to N, not always 1), which is the concrete reason
  the scheduler interface returns one job at a time today but is not typed as strictly 1:1 with
  workers.
