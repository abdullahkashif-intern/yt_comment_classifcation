"""
schemas.py — Shared Pydantic data models for the YouTube Comment Classifier.

This module defines all the data structures (schemas) that flow through the
application. It acts as the single source of truth for data shapes, ensuring
that the YouTube client, LLM classifier, FastAPI routes, and Streamlit frontend
all speak the same language. Using Pydantic BaseModel gives automatic validation,
serialisation to/from JSON, and IDE auto-completion across the codebase.
"""

from typing import Literal  # Literal allows restricting a type to a fixed set of string values

from pydantic import BaseModel  # BaseModel is the foundation for all Pydantic data models

# Category is a union of exactly five allowed string values.
# Any value outside this set will be rejected at validation time.
# This type is shared by the LLM classifier's system prompt and the response models.
Category = Literal["appreciation", "question", "comparison", "funny", "criticism"]


class Comment(BaseModel):
    """A raw, unclassified YouTube comment as returned by the YouTube Data API."""

    comment_id: str  # Unique ID assigned by YouTube (e.g. "UgxABC123")
    author: str      # Display name of the comment author
    text: str        # Plain-text body of the comment


class ClassifiedComment(BaseModel):
    """A comment that has been labelled by the LLM with one or more categories."""

    comment_id: str             # Same ID as the original Comment — used to correlate results
    author: str                 # Preserved from the original Comment unchanged
    text: str                   # Preserved from the original Comment unchanged
    categories: list[Category]  # One or more Category labels assigned by the LLM


class VideoRequest(BaseModel):
    """Request body accepted by the POST /classify-video endpoint."""

    video_url: str  # Any supported YouTube URL format or bare video ID


class ClassificationSummary(BaseModel):
    """Aggregate result of running classify_comments() over a full comment set.

    Separates three distinct reasons a comment can end up unclassified so the
    discrepancy is diagnosable instead of collapsing into one "dropped" count:
      - dropped:   the LLM returned the comment with no categories (a successful
                   batch call that just skipped this one comment).
      - failed:    the comment's batch errored out after retries were exhausted.
      - timed out: the comment's batch was never attempted because the
                   request-level deadline was already reached.
    """

    results: list[ClassifiedComment]  # Comments that received at least one category label
    dropped_comment_count: int        # See docstring above
    failed_batch_count: int           # Number of batches that failed after retries were exhausted
    failed_comment_count: int         # Comments belonging to those failed batches
    timed_out_comment_count: int      # Comments never attempted because the deadline was hit


class ClassifyVideoResponse(BaseModel):
    """Full response returned by the POST /classify-video endpoint."""

    video_id: str                    # Extracted 11-character YouTube video ID
    total_comments_fetched: int      # Total raw comments retrieved from the YouTube API
    total_comments_classified: int   # Comments that received at least one category from the LLM
    dropped_comment_count: int       # Comments the LLM returned with no categories (skipped)
    failed_batch_count: int          # Batches that errored out after retries were exhausted
    failed_comment_count: int        # Comments belonging to those failed batches
    timed_out_comment_count: int     # Comments never attempted because the request-level deadline was hit
    timed_out: bool                  # True if the request-level ceiling cut fetching and/or classifying short
    results: list[ClassifiedComment] # Ordered list of every successfully classified comment
