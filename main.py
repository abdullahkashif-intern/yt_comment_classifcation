"""
main.py — FastAPI application entry point and route definitions.

This module wires together the three main layers of the backend:
  1. YouTube client  — fetches raw comments via the YouTube Data API
  2. LLM classifier  — sends those comments to an LLM for category labelling
  3. FastAPI routing  — exposes the result over HTTP for the Streamlit frontend

It intentionally stays thin: all business logic lives in the youtube and llm
packages, and all data shapes are defined in schemas.py. The single route
/classify-video accepts a video URL, orchestrates the fetch→classify pipeline,
and returns a structured JSON response. Errors from downstream layers are
translated into appropriate HTTP status codes so the frontend can surface
meaningful messages to the user.
"""

import time  # Standard library: monotonic clock for the request-level deadline (Phase 0)

from fastapi import FastAPI, HTTPException
# FastAPI   — the ASGI web framework
# HTTPException — raised to return an HTTP error response with a status code and detail

from cache import TTLCache                           # Phase 1: in-memory TTL cache for responses
from config import settings                         # Typed singleton with all env-var config
from llm.classifier import classify_comments        # Batches comments through the LLM pipeline
from schemas import ClassifyVideoResponse, VideoRequest  # Request/response Pydantic models
from youtube.client import (
    CommentsDisabledError,          # Raised when the video has comments turned off
    VideoNotFoundError,             # Raised when YouTube returns a 404 for the video
    extract_video_id,               # Parses any YouTube URL format into a bare video ID
    fetch_all_top_level_comments,   # Pages through the YouTube API to collect all comments
)

# Create the FastAPI application instance.
# The title appears in the auto-generated /docs (Swagger UI) interface.
app = FastAPI(title="YouTube Comment Classifier")

# Phase 1: process-wide cache of full responses, keyed by video_id.
# In-memory and single-process — see cache.py's docstring for the tradeoffs.
_response_cache: TTLCache[ClassifyVideoResponse] = TTLCache(ttl_seconds=settings.cache_ttl_seconds)


@app.post("/classify-video", response_model=ClassifyVideoResponse)
def classify_video(request: VideoRequest) -> ClassifyVideoResponse:
    """Fetch and classify every top-level comment on a YouTube video.

    Accepts a VideoRequest body containing any supported YouTube URL format.
    Returns a ClassifyVideoResponse with per-comment category labels and
    aggregate counts (total fetched, classified, dropped, failed, and timed out).
    """

    # --- Step 1: Parse the video URL into a bare 11-character video ID ---
    try:
        video_id = extract_video_id(request.video_url)
    except ValueError as e:
        # extract_video_id raises ValueError for unrecognised URL patterns
        raise HTTPException(status_code=400, detail=str(e)) from e

    # --- Step 2: Serve from cache if this video was classified recently (Phase 1) ---
    cached_response = _response_cache.get(video_id)
    if cached_response is not None:
        return cached_response

    # Request-level deadline (Phase 0): a single wall-clock budget shared across
    # both the fetch and classify phases, so a pathological video can't run
    # unbounded. Not a comment-count cap — pagination and batching still run
    # to exhaustion within this time budget.
    deadline = time.monotonic() + settings.request_timeout_ceiling_seconds

    # --- Step 3: Fetch all top-level comments from the YouTube Data API ---
    try:
        comments, fetch_timed_out = fetch_all_top_level_comments(
            video_id, settings.youtube_api_key, deadline=deadline
        )
    except CommentsDisabledError as e:
        # 422 Unprocessable Entity: the request was valid but the video can't be processed
        raise HTTPException(status_code=422, detail=str(e)) from e
    except VideoNotFoundError as e:
        # 404 Not Found: the video ID does not correspond to an existing video
        raise HTTPException(status_code=404, detail=str(e)) from e

    # --- Step 4: Send all comments through the LLM classification pipeline ---
    summary = classify_comments(comments, deadline=deadline)

    # --- Step 5: Build, cache, and return the structured response ---
    response = ClassifyVideoResponse(
        video_id=video_id,
        total_comments_fetched=len(comments),                      # Raw count before classification
        total_comments_classified=len(summary.results),            # Comments that received at least one label
        dropped_comment_count=summary.dropped_comment_count,       # LLM returned the comment with no labels
        failed_batch_count=summary.failed_batch_count,              # Batches that errored after retries
        failed_comment_count=summary.failed_comment_count,          # Comments in those failed batches
        timed_out_comment_count=summary.timed_out_comment_count,    # Comments never attempted (deadline hit)
        timed_out=fetch_timed_out or summary.timed_out_comment_count > 0,  # Ceiling cut this request short
        results=summary.results,                                    # Full list of ClassifiedComment objects
    )
    # Only cache clean results — a response with unretried failures or a
    # deadline cutoff shouldn't be served verbatim to every request for the
    # next hour; let the next request try again from scratch instead.
    if not response.timed_out and response.failed_batch_count == 0:
        _response_cache.set(video_id, response)
    return response
