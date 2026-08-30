# Scaling Roadmap

This document tracks the path from the current single-request, sequential MVP
(as constrained by `CLAUDE.md`'s Strict Guardrails) toward a service that can
handle real concurrent traffic. It is a **planning doc, not a spec** — it
changes as phases land or assumptions turn out wrong.

`CLAUDE.md` should only ever describe the architecture as it exists *today*.
This file is where the "not yet" work lives. When a phase below ships:

1. Move any guardrail it removes out of `CLAUDE.md`'s "Strict Guardrails"
   section.
2. Add whatever new pattern it introduces to `CLAUDE.md`'s "Coding Standards
   & Architecture" section.
3. Mark the phase `DONE` in this file (keep it here for history — don't
   delete completed phases, they explain *why* the code looks the way it
   does).

Phases are ordered by dependency, not just priority — later phases build on
earlier ones.

---

## Phase 0 — Safety nets
**Status:** DONE

No architecture change. Fixes real correctness/reliability gaps in the
current synchronous pipeline.

- Retry with backoff on transient `HttpError`s from the YouTube API
  (`youtube/client.py`) and on transient errors from the LLM provider call
  (`llm/classifier.py`).
- If a batch fails classification partway through a request, return the
  results already classified plus a per-batch error marker instead of
  failing the entire request (currently one bad batch loses everything
  classified so far in that request).
- Add an overall request-level timeout/ceiling so a pathological video
  (extremely high comment count) can't run unbounded.

**Implementation notes:**
- Retry/backoff uses `tenacity` (already a transitive dep via `langchain`,
  now a direct dep). `youtube/client.py::_execute_page_request` retries on
  429/5xx `HttpError`s only; `llm/classifier.py::_invoke_with_retry` retries
  on a duck-typed transient-error heuristic (status code or exception class
  name) since the four provider SDKs don't share an exception hierarchy.
  3–4 attempts, exponential backoff, `reraise=True` so the final failure
  surfaces as the original exception type.
- `classify_comments` now returns a `ClassificationSummary` (see
  `schemas.py`) instead of a `(results, dropped_count)` tuple, so a
  batch-level failure (`failed_batch_count` / `failed_comment_count`) is
  reported distinctly from a per-comment drop inside a successful batch
  (`dropped_comment_count`).
- The request-level ceiling is a single `time.monotonic()` deadline computed
  once in `main.py` and threaded through both `fetch_all_top_level_comments`
  and `classify_comments` (`deadline` param on both). It's a wall-clock
  budget, not a comment-count cap — pagination and batching still run to
  exhaustion within the budget, per the existing pagination/batching rules.
  Reported back as `timed_out` / `timed_out_comment_count` on the response.
- New settings: `request_timeout_ceiling_seconds` (default 900s).

**CLAUDE.md impact:** done — see "Coding Standards & Architecture" (Retry/
Backoff, Per-Batch Failure Isolation, Request-Level Deadline) and "Known
Behaviors" (Dropped Comments note now distinguishes drop vs. batch failure).

---

## Phase 1 — Caching
**Status:** DONE

- Cache classification results keyed by `video_id`, short TTL (e.g. 1 hour).
- In-memory (`functools.lru_cache` / a small TTL dict) is enough for the
  single-process MVP; revisit if Phase 4 introduces multiple worker
  processes, since in-memory caches don't share across processes.

**Why this order:** highest ROI relative to effort — this is the single
biggest lever on YouTube quota and LLM cost, and requires no changes to the
request/response contract.

**Implementation notes:**
- Added `cache.py`: a minimal thread-safe `TTLCache` (dict + lock, lazy
  expiry on read). `main.py` holds one process-wide instance keyed by
  `video_id`, checked before the fetch step and populated after the full
  response is built.
- New setting: `cache_ttl_seconds` (default 3600s / 1 hour).
- Decided explicitly: a short-lived, in-memory, per-process cache does
  **not** count as "persistent storage" under guardrail #1 — it holds no
  state across restarts, writes nothing to disk, and isn't a datastore.
  Guardrail #1 in `CLAUDE.md` has been reworded to say so explicitly.
- Interaction with Phase 0: a response with `timed_out` or
  `failed_batch_count > 0` is deliberately **not** cached, so a partial
  result from a transient blip doesn't get re-served verbatim for the rest
  of the TTL — the next request tries again from scratch instead.

**CLAUDE.md impact:** done — guardrail #1 reworded (see "Strict
Guardrails"); "Response Caching" pattern added to "Coding Standards &
Architecture".

---

## Phase 2 — Async job model
**Status:** not started

- `POST /classify-video` returns a job ID immediately (202 Accepted) instead
  of blocking until classification finishes.
- Add `GET /jobs/{job_id}` to poll status and fetch the result once ready.
- Requires some job state store — in-memory dict is fine to start, but note
  this reopens the "no persistent storage" guardrail question from Phase 1.

**Why this order:** a prerequisite for real concurrency — without it, every
in-flight classification occupies a worker thread for its full duration
(minutes), which caps concurrent users at the size of the thread pool
regardless of anything else.

**CLAUDE.md impact when done:** this is a breaking API change — update the
"Core pipeline" description and the `curl` example under Commands, and add
the new endpoint's contract to Coding Standards.

---

## Phase 3 — Concurrent batch classification
**Status:** not started

- Classify batches concurrently instead of sequentially, bounded by a
  concurrency limit / semaphore that respects the LLM provider's rate limits
  (RPM/TPM vary by provider — see `llm/provider.py`).
- Directly removes the "Concurrency in Batching" guardrail.

**Why this order:** doing this before Phase 2 just makes requests finish
faster while still blocking a worker thread for the (now shorter) duration —
real benefit only shows up once requests are async.

**CLAUDE.md impact when done:** remove guardrail #5 ("Concurrency in
Batching") from Strict Guardrails entirely; document the concurrency-limit
pattern in Coding Standards so future provider additions in
`llm/provider.py` account for it.

---

## Phase 4 — Horizontal scaling
**Status:** not started

- Move job state and the classification work itself out of the API
  process: a task queue (Celery/RQ + Redis, or equivalent) with multiple
  worker processes.
- Address the shared-quota bottleneck: either a higher YouTube API quota, a
  pool of API keys, or explicit backpressure when quota is close to
  exhausted.

**CLAUDE.md impact when done:** this is a significant enough architecture
change that it likely needs its own top-level section in CLAUDE.md (e.g.
"Job Queue Architecture"), not just an edit to an existing bullet.

---

## Phase 5 — Observability & abuse controls
**Status:** not started

- Structured logging across the pipeline (currently none).
- Basic metrics (requests, YouTube quota consumed, LLM tokens/cost, job
  queue depth).
- Rate limiting and/or an API key on `/classify-video` — right now the
  endpoint is open, and every request costs real YouTube quota and LLM
  money.

**CLAUDE.md impact when done:** add an "Observability" section; the "No
Auth" guardrail should be narrowed to "no *user* accounts" if a service-level
API key is introduced, since those are different guarantees.

---

## Phase 6 — Load testing
**Status:** not started

- Validate actual concurrent-user and comment-volume capacity against a
  target, using the deployed Phase 4/5 architecture.
- Feed results back into this doc — either close out the roadmap or add a
  Phase 7 for whatever bottleneck shows up next (most likely candidate: LLM
  provider rate limits under real concurrency).
