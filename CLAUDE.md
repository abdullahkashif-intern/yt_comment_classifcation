# Project Context
This is an API service that ingests a YouTube video URL, fetches **all** top-level comments via the YouTube Data API (paginated — not a sample, not capped), and uses an LLM to classify each comment into one or more categories: `appreciation`, `question`, `comparison`, `funny`, or `criticism`.

**Core pipeline:** Link in -> Paginate through all comments -> Batch (~25) -> LLM Classification (sequential) -> JSON out.

# Tech Stack & Libraries
*   **Python:** Core language.
*   **FastAPI:** API framework.
*   **uv:** Package and environment management.
*   **YouTube Data API:** For fetching comments (Do NOT scrape YouTube).
*   **LangChain:** For LLM provider abstraction (Anthropic, Gemini, Groq, HuggingFace).
*   **Pydantic:** Strict schema enforcement for LLM outputs.

# Environment Variables (.env)
*   `YOUTUBE_API_KEY`
*   `LLM_PROVIDER` (anthropic, gemini, groq, or huggingface)
*   `LLM_MODEL` (exact model name from provider docs — verify current, don't assume from memory)
*   Provider Key (`ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY`, or `HUGGINGFACEHUB_API_TOKEN`)

# Commands (For AI Execution)
*   **Install/Sync:** `uv sync`
*   **Run Server:** `uv run uvicorn main:app --reload --port 8000`
*   **Run Tests:** `pytest` (Always run this after modifying logic or prompts, specifically testing against `tests/fixtures/sample_comments.json`).
*   **Test API:** `curl -X POST localhost:8000/classify-video -d '{"video_url": "..."}'`

# Coding Standards & Architecture
*   **Type Hinting & Schemas:** Use Pydantic to enforce all data shapes going into and coming out of the LLM. Do not write manual text-parsing logic for LLM outputs.
*   **Separation of Concerns:** Keep YouTube fetching logic and LLM classification logic in completely separate files. They should only share basic data shapes.
*   **LLM Provider Logic:** Keep all "which AI company are we using" logic strictly isolated in `llm/provider.py`. Do not scatter provider-specific code across the codebase.
* **UI Layer:** Keep frontend code in its own directory (e.g. `web/` or `frontend/`), fully separate from YouTube-fetching and LLM-classification logic. The UI only calls the existing `/classify-video` endpoint — no business logic duplicated client-side.
*   **Pagination:** Fully paginate through YouTube's `commentThreads.list` using `nextPageToken` until exhausted. Do not stop at a fixed comment count.
*   **Batching:** Comments must be batched (~25 at a time) before sending to the LLM. Do not send all comments in a single request.
*   **Retry/Backoff:** Transient failures (YouTube 429/5xx in `youtube/client.py`, provider rate-limit/timeout/connection/5xx errors in `llm/classifier.py`) are retried with exponential backoff via `tenacity` before being allowed to propagate. Non-transient errors (404, 400, `commentsDisabled`, bad request schema) are not retried. Each call site defines its own transient-error predicate rather than sharing one, since YouTube's `HttpError` and each LLM provider's exception hierarchy differ.
*   **Per-Batch Failure Isolation:** In `classify_comments`, a batch that still errors after retries are exhausted does not fail the whole request — it's counted separately (`failed_batch_count` / `failed_comment_count` in `ClassificationSummary`) and processing continues with the next batch. This is distinct from `dropped_comment_count`, which covers a *successful* batch call that omitted a comment.
*   **Request-Level Deadline:** `/classify-video` computes a single `time.monotonic()`-based deadline (`settings.request_timeout_ceiling_seconds`, default 900s) shared across fetching and classifying. If reached, remaining work is skipped rather than run unbounded, and the response reports `timed_out` / `timed_out_comment_count`. This is a wall-clock safety ceiling, not a comment-count cap — it doesn't relax the pagination/batching rules above.
*   **Response Caching:** `/classify-video` caches the full response in-process, keyed by `video_id`, via the `TTLCache` in `cache.py` (`settings.cache_ttl_seconds`, default 1 hour). In-memory only, lost on restart, not shared across worker processes — see `docs/scaling-roadmap.md` Phase 4 for when that stops being sufficient.

# Strict Guardrails (DO NOT BUILD THESE)
Unless explicitly asked to change these rules, do NOT implement:
1.  **Databases:** No durable/persistent storage (disk-backed DB, files, external cache like Redis). A short-lived, in-memory, per-process cache (see "Response Caching" below) does not count as persistent storage — it holds no state across restarts and is not a datastore.
2.  **Auth:** No user accounts or login systems.
3.  **Multi-Video Handling:** One video per request only.
4.  **Nested Comments:** Only process top-level comments, ignore reply threads.
5.  **Concurrency in Batching:** Process batches sequentially for now, not concurrently.

# Known Behaviors (Do Not Attempt to "Fix")
*   **High Latency & Quota Usage:** Fetching all comments from popular videos (tens of thousands) takes real time and uses real YouTube API quota. This is expected, not a bug.
*   **Dropped Comments:** If the LLM occasionally skips a comment in a *successful* batch call, ignore it for now — tracked as `dropped_comment_count`. If an entire batch fails after retries are exhausted, that's tracked separately as `failed_batch_count` / `failed_comment_count` (see "Per-Batch Failure Isolation" above) — don't conflate the two when debugging a count discrepancy.
*   **Sarcasm:** Handle sarcasm (e.g., "wow SO helpful 🙄" -> `criticism`) purely through LLM system prompt examples, NEVER through special-case Python logic.
*   **Provider Package Drift:** LangChain provider packages version independently and ship fast. If `with_structured_output` misbehaves, check the installed package version before assuming it's a prompt bug.

# Verification
*   `tests/fixtures/sample_comments.json` — ~20 hand-labeled comments (sarcasm, multi-label, non-English). Re-run classification against this fixture after any prompt or provider change.
*   Structured output guarantees valid *shape*, not correct *labels*. Verify labels manually against the fixture — don't assert exact-match on LLM output in tests.

# Roadmap (Not Now)
Only after the above works reliably end-to-end:
*   Concurrent batch classification for speed.
*   Proper reconciliation when the LLM drops a comment.
*   See @docs/scaling-roadmap.md for the full phased plan.