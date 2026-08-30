"""
llm/classifier.py — LLM-powered batch classifier for YouTube comments.

This module takes a flat list of raw Comment objects and returns a list of
ClassifiedComment objects, each enriched with one or more category labels
assigned by the language model.

Key design decisions:
  - Batching: Comments are grouped into batches of BATCH_SIZE (25) to stay
    within context-window and token-output limits while minimising the total
    number of LLM API calls.
  - Structured output: `model.with_structured_output(BatchClassificationResult)`
    instructs LangChain to enforce that the model returns well-formed JSON
    matching the Pydantic schema, rather than free-form text.
  - Graceful dropping: If the LLM omits a comment or returns it with an empty
    category list, that comment is silently dropped and counted separately
    rather than failing the entire request. This is by design — some providers
    (e.g. Groq) may reject tool-call schemas server-side for certain edge cases.
"""

import time  # Standard library: monotonic clock for deadline checks

from pydantic import BaseModel, Field
# BaseModel — base class for Pydantic data models used to describe the expected LLM response shape
# Field     — allows attaching extra metadata (e.g. default_factory) to model fields
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
# tenacity — retry/backoff decorator used to ride out transient LLM provider errors (Phase 0)

from llm.provider import get_chat_model              # Factory that returns the configured LangChain chat model
from schemas import Category, ClassificationSummary, ClassifiedComment, Comment  # Shared domain types

# Maximum number of attempts (including the first) for a single batch's LLM
# call before a transient error is allowed to propagate to the batch loop,
# where it is caught and the batch is marked failed rather than raised.
_MAX_ATTEMPTS = 3

# Exception class-name substrings that indicate a transient provider error
# (rate limiting, timeouts, transport/connection issues, server-side 5xx).
# Matched by name rather than isinstance because the four supported provider
# SDKs (Anthropic, Gemini, Groq, HuggingFace) each raise their own exception
# hierarchy — see CLAUDE.md's "Provider Package Drift" note. This is a
# best-effort heuristic, not an exhaustive list.
_TRANSIENT_ERROR_NAME_PATTERNS = (
    "RateLimit",
    "Timeout",
    "Connection",
    "ServiceUnavailable",
    "InternalServerError",
    "Overloaded",
)


def _is_transient_llm_error(exc: BaseException) -> bool:
    """Best-effort, provider-agnostic check for whether an error is transient."""
    status = getattr(exc, "status_code", None)
    if status in (429, 500, 502, 503, 504):
        return True
    name = type(exc).__name__
    return any(pattern in name for pattern in _TRANSIENT_ERROR_NAME_PATTERNS)

# Number of comments sent to the LLM in a single API call.
# 25 is a practical sweet spot: large enough to amortise per-call overhead,
# small enough to avoid hitting context or output-token limits on most models.
BATCH_SIZE = 25

# The system-level instruction injected into every LLM call.
# It enumerates the five allowed categories, explains multi-label rules,
# and gives a worked example of sarcasm detection — all in plain English so
# the prompt is model-agnostic and works across providers.
SYSTEM_PROMPT = """You are classifying YouTube comments into one or more of these categories:
- appreciation: praise, gratitude, positive reactions
- question: asking something, seeking clarification or info
- comparison: comparing to other things, videos, products, or creators
- funny: jokes, memes, humorous remarks
- criticism: negative feedback, complaints, or disagreement — including sarcasm that reads \
as positive on the surface but means the opposite (e.g. "wow SO helpful 🙄" -> criticism)

Rules:
- A comment can have multiple categories if it genuinely fits more than one.
- Every comment in the input must appear exactly once in the output, matched by comment_id.
- Judge sarcasm by tone and context, not just literal words.
- Comments may be in any language; classify based on meaning, not the language itself.
"""


class CommentClassificationItem(BaseModel):
    """Represents the LLM's classification result for a single comment."""

    comment_id: str  # Must match one of the input comment IDs so results can be correlated

    # No min_length constraint on `categories`: some providers (e.g. Groq) reject the entire
    # structured tool-call server-side if *any* item in the batch fails schema validation.
    # Allowing an empty list means the model can "abstain" on a hard-to-classify comment
    # without blowing up the entire batch — the classifier then counts it as "dropped".
    categories: list[Category] = Field(default_factory=list)  # Zero or more valid Category labels


class BatchClassificationResult(BaseModel):
    """Top-level structured output expected from the LLM for a full batch."""

    classifications: list[CommentClassificationItem]  # One entry per input comment


def _chunk(items: list[Comment], size: int):
    """Split `items` into consecutive sublists of at most `size` elements.

    This is a simple generator so batches are produced lazily without
    materialising the entire split list in memory at once.

    Args:
        items: The full list of Comment objects to split.
        size:  Maximum number of items per chunk (BATCH_SIZE in practice).

    Yields:
        Successive subslices of `items`.
    """
    for i in range(0, len(items), size):
        yield items[i : i + size]  # Python slice is safe even when i+size > len(items)


@retry(
    retry=retry_if_exception(_is_transient_llm_error),
    stop=stop_after_attempt(_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,  # On final failure, raise the original provider exception
)
def _invoke_with_retry(structured_model, messages) -> "BatchClassificationResult":
    """Invoke the structured model for one batch, retrying transient errors."""
    return structured_model.invoke(messages)


def classify_comments(
    comments: list[Comment], deadline: float | None = None
) -> ClassificationSummary:
    """Classify a list of Comments using the configured LLM, in batches.

    Each batch is formatted as a numbered list of "[comment_id] text" lines and
    sent to the model. The model's structured response is correlated back to the
    original comments by comment_id. Comments not present in the response or
    returned with no categories are counted as dropped.

    Transient per-batch LLM errors are retried with exponential backoff. If a
    batch still fails after retries are exhausted, that batch is marked failed
    and processing continues with the next batch — one bad batch no longer
    loses every comment classified so far in the request.

    Args:
        comments: All raw Comment objects to classify.
        deadline: Optional `time.monotonic()` timestamp. If reached before a
            batch starts, that batch and all remaining batches are skipped
            (counted as timed-out) instead of continuing to run unbounded.

    Returns:
        A ClassificationSummary with the successfully classified comments and
        counts of dropped, failed, and timed-out comments.
    """
    # Obtain the LangChain chat model configured in settings
    model = get_chat_model()

    # Wrap the model with structured-output enforcement.
    # LangChain will use tool-calling / JSON mode to ensure the model's
    # response can be parsed into a BatchClassificationResult Pydantic object.
    structured_model = model.with_structured_output(BatchClassificationResult)

    results: list[ClassifiedComment] = []  # Accumulates successfully classified comments
    dropped_count = 0                       # Running tally of comments skipped by the LLM
    failed_batch_count = 0                  # Batches that errored out after retries were exhausted
    failed_comment_count = 0                # Comments belonging to those failed batches
    timed_out_comment_count = 0             # Comments never attempted because the deadline was hit

    for batch in _chunk(comments, BATCH_SIZE):
        if deadline is not None and time.monotonic() > deadline:
            # Ceiling reached — stop starting new batches, but don't lose what's
            # already been classified. Remaining comments are reported as timed out.
            timed_out_comment_count += len(batch)
            continue

        # Format the batch as a simple numbered list the LLM can parse reliably:
        # "[comment_id_1] First comment text here"
        # "[comment_id_2] Second comment text here"
        batch_input = "\n".join(f"[{c.comment_id}] {c.text}" for c in batch)

        try:
            # Invoke the LLM with the system prompt + user-formatted batch
            response: BatchClassificationResult = _invoke_with_retry(
                structured_model,
                [
                    {"role": "system", "content": SYSTEM_PROMPT},  # Category definitions and rules
                    {"role": "user", "content": batch_input},       # The actual comments to classify
                ],
            )
        except Exception:
            # Retries exhausted (or a non-transient error) — don't fail the whole
            # request, just this batch. The comments it contained are reported
            # separately from "dropped" so the discrepancy is diagnosable.
            failed_batch_count += 1
            failed_comment_count += len(batch)
            continue

        # Build a lookup dict from comment_id → ClassificationItem for O(1) access
        by_id = {item.comment_id: item for item in response.classifications}

        for comment in batch:
            # Look up the LLM's output for this specific comment
            classification = by_id.get(comment.comment_id)

            if classification is None or not classification.categories:
                # LLM omitted the comment or returned an empty category list — drop it
                dropped_count += 1
                continue  # Skip to the next comment in the batch

            # Merge the original comment fields with the LLM-assigned categories
            results.append(
                ClassifiedComment(
                    comment_id=comment.comment_id,  # Preserve original ID
                    author=comment.author,           # Preserve original author display name
                    text=comment.text,               # Preserve original comment text
                    categories=classification.categories,  # One or more Category labels from the LLM
                )
            )

    return ClassificationSummary(
        results=results,
        dropped_comment_count=dropped_count,
        failed_batch_count=failed_batch_count,
        failed_comment_count=failed_comment_count,
        timed_out_comment_count=timed_out_comment_count,
    )
