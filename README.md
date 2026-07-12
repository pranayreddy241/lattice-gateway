# Lattice — a fault-tolerant LLM inference gateway

Lattice sits in front of a fleet of model servers (vLLM, TGI, any
OpenAI-compatible endpoint) and makes them behave like one reliable
service. It answers a concrete question: **what happens to your LLM app
when one GPU box gets slow, one starts failing, and half your prompts
are near-duplicates?**

```
                          ┌──────────────────────────────┐
 clients ──► rate limit ──►  semantic cache (hit? done)  │
                          └──────────────┬───────────────┘
                                         │ miss
                          ┌──────────────▼───────────────┐
                          │  router: P2C over EWMA        │
                          │  latency × outstanding,       │
                          │  circuit breaker per backend  │
                          └──────┬───────┬───────┬───────┘
                                 │       │       │
                          micro-batch  micro-batch  micro-batch
                                 │       │       │
                              gpu-a    gpu-b    gpu-c
```

## What's inside

| Component | Design choice | Why |
|---|---|---|
| Routing | Power-of-two-choices over `EWMA latency × (outstanding + 1)` | Near-optimal load balance without a global scan; avoids herd behavior on stale stats |
| Fault tolerance | Per-backend circuit breaker (closed → open → half-open single probe) + retry on a *different* backend | Failing box stops receiving traffic within `failure_threshold` requests; recovers via one probe, not a thundering herd |
| Throughput | Adaptive micro-batching (flush on size or deadline) with bounded overlapping in-flight batches | Batched inference amortizes cost; overlap removes head-of-line blocking (see measurements) |
| Semantic cache | Hashed char-trigram embeddings, cosine threshold, LRU + TTL, exact-match fast path + inverted-index candidate pruning | Near-duplicate prompts skip inference entirely; only `temperature=0` responses are cacheable |
| Rate limiting | Token bucket per API key | Burst-friendly fairness |
| Observability | Prometheus counters, latency histogram, breaker-state gauge; `/healthz`, `/v1/backends` | You can't operate what you can't see |

## Measured, not claimed

Closed-loop load test (64 workers, in-process ASGI, 3 mock backends at
30/45/60 ms base latency, one with a 2% failure rate). Reproduce with
`python loadtest/loadtest.py`.

| Scenario | Throughput | p50 | p95 |
|---|---|---|---|
| v1 batcher, cache off | 271 req/s | 223 ms | 355 ms |
| + overlapped batch dispatch | **687 req/s (2.5×)** | **83 ms** | 152 ms |
| 30% repeat traffic, naive O(n) cache scan | 214 req/s | 87 ms | 1,209 ms |
| + exact-match fast path + inverted index | **429 req/s (2×)** | **4 ms** | 639 ms |

Two production-style lessons came out of the load tests:

1. **Head-of-line blocking in the batcher.** v1 awaited each batch
   inline, serializing batches per backend. Dispatching flushes as
   bounded concurrent tasks (semaphore-capped) gave 2.5× throughput
   and cut p50 by 63%.
2. **The cache was blocking the event loop.** A full cosine scan of 2k
   entries in pure Python cost ~30 ms per miss — *on the hot path of
   every request*. An O(1) exact-match check plus an inverted index
   over each entry's top-8 embedding dimensions shrank the scored
   candidate set by ~100×, cutting p50 from 87 ms to 4 ms at a 57%
   hit rate.

Remaining p95 in the cached scenario is backend saturation under
closed-loop load, not gateway overhead — visible in `/v1/backends`
EWMA per backend.

## Run it

```bash
pip install ".[dev,metrics]"
pytest -q                                   # 32 tests: unit + e2e failover
python loadtest/loadtest.py                 # measure it yourself
uvicorn lattice.server:create_app --factory # serve (mock backends by default)
```

```bash
curl -s localhost:8000/v1/completions \
  -H 'content-type: application/json' -H 'x-api-key: demo' \
  -d '{"prompt": "explain two-phase commit", "temperature": 0}'
```

Docker: `docker build -t lattice . && docker run -p 8000:8000 lattice`

## Honest limitations

- Gateway-side **micro**-batching, not token-level continuous batching —
  that belongs inside the engine (vLLM does it well); Lattice batches at
  the request level in front of it.
- Semantic cache trades exactness for speed: trigram embeddings catch
  re-phrasings, not paraphrases. The `embed()` boundary exists so a
  real embedding model + FAISS can drop in.
- Single-node. Multi-node Lattice would need shared breaker state and
  cache (Redis) — the interfaces are deliberately narrow to allow it.

## Layout

```
lattice/
  server.py     # FastAPI wiring: the request lifecycle lives here
  router.py     # P2C + EWMA + outstanding-request scoring
  breaker.py    # circuit breaker state machine (injectable clock)
  batcher.py    # adaptive micro-batching, bounded overlap
  cache.py      # semantic cache: sparse embeddings + inverted index
  ratelimit.py  # token buckets per API key
  backends.py   # Backend protocol, MockBackend, HTTPBackend
  metrics.py    # Prometheus (no-op fallback)
tests/          # 32 tests, no sleeps: injectable clocks throughout
loadtest/       # the closed-loop harness behind the numbers above
```
