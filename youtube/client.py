"""
youtube/client.py — YouTube Data API v3 client for fetching video comments.

This module is the sole point of contact with the YouTube Data API. It provides
two public utilities:

  - extract_video_id(video_url): Normalises any supported YouTube URL format
    (watch links, shortened youtu.be links, embed/shorts/v paths, or a bare
    11-character ID) into the canonical video ID string.

  - fetch_all_top_level_comments(video_id, api_key): Pages through the YouTube
    commentThreads.list endpoint, collecting every top-level comment for the
    given video and returning them as a list of Comment objects.

Two custom exceptions (CommentsDisabledError, VideoNotFoundError) allow callers
to react to specific API failure modes with the right HTTP status codes rather
than catching generic HttpErrors.
"""

import re                                    # Standard library: regular expression support
import time                                  # Standard library: monotonic clock for deadline checks
from urllib.parse import parse_qs, urlparse  # Standard library: URL decomposition utilities
# parse_qs  — converts a query string (e.g. "v=abc123") into a dict
# urlparse  — splits a URL into scheme, hostname, path, query, etc.

from googleapiclient.discovery import build  # Builds an authenticated API client for any Google API
from googleapiclient.errors import HttpError  # Represents an HTTP-level error from any Google API
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
# tenacity — retry/backoff decorator used to ride out transient YouTube API errors (Phase 0)

from schemas import Comment  # Shared Pydantic model for a single YouTube comment

# Maximum number of attempts (including the first) for a single page request
# before a transient error is allowed to propagate.
_MAX_ATTEMPTS = 4


def _is_transient_http_error(exc: BaseException) -> bool:
    """Decide whether an HttpError is worth retrying.

    429 (rate limited) and 5xx (server-side) errors are transient — a retry
    with backoff is likely to succeed. Everything else (404 not found, 400
    bad request, 403 commentsDisabled/quotaExceeded) is a permanent failure
    for this request and must not be retried.
    """
    if not isinstance(exc, HttpError):
        return False
    status = exc.resp.status if exc.resp is not None else None
    return status == 429 or (status is not None and 500 <= status < 600)

# Pre-compiled pattern list used by extract_video_id for quick regex matching.
# Currently contains one entry: a pattern that matches a bare 11-character video ID
# (alphanumeric, hyphens, and underscores) so the function short-circuits before
# attempting URL parsing when the input is already just an ID.
_VIDEO_ID_PATTERNS = [
    re.compile(r"^[\w-]{11}$"),  # bare video ID (e.g. "dQw4w9WgXcQ")
]


class CommentsDisabledError(Exception):
    """Raised when the YouTube API signals that comments are disabled for a video.

    Callers should map this to an HTTP 422 Unprocessable Entity response.
    """
    pass


class VideoNotFoundError(Exception):
    """Raised when the YouTube API returns a 404 for the requested video ID.

    Callers should map this to an HTTP 404 Not Found response.
    """
    pass


def extract_video_id(video_url: str) -> str:
    """Parse a YouTube URL (or bare ID) and return the 11-character video ID.

    Supported formats:
      - Bare ID:   "dQw4w9WgXcQ"
      - Watch URL: "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
      - Short URL: "https://youtu.be/dQw4w9WgXcQ"
      - Embed URL: "https://www.youtube.com/embed/dQw4w9WgXcQ"
      - Shorts:    "https://www.youtube.com/shorts/dQw4w9WgXcQ"
      - /v/ path:  "https://www.youtube.com/v/dQw4w9WgXcQ"

    Raises:
        ValueError: If none of the known patterns match the input.
    """
    video_url = video_url.strip()  # Remove any accidental leading/trailing whitespace

    # Fast path: input is already a valid bare video ID — return it immediately
    if _VIDEO_ID_PATTERNS[0].match(video_url):
        return video_url

    # Decompose the URL into its components for further inspection
    parsed = urlparse(video_url)

    # Handle youtu.be shortened links — the video ID is the first path segment
    if parsed.hostname in ("youtu.be",):
        return parsed.path.lstrip("/")  # Strip the leading "/" to get just the ID

    # Handle full youtube.com URLs
    if parsed.hostname and "youtube.com" in parsed.hostname:

        # Standard watch URL: extract the "v" query parameter
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]  # parse_qs returns lists
            if video_id:
                return video_id

        # Path-based formats (/embed/, /shorts/, /v/) — ID is the segment after the prefix
        for prefix in ("/embed/", "/shorts/", "/v/"):
            if parsed.path.startswith(prefix):
                # Slice off the prefix, then split on "/" in case there are trailing segments
                return parsed.path[len(prefix):].split("/")[0]

    # Nothing matched — surface a descriptive error to the caller
    raise ValueError(f"Could not extract a video ID from URL: {video_url}")


@retry(
    retry=retry_if_exception(_is_transient_http_error),
    stop=stop_after_attempt(_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,  # On final failure, raise the original HttpError rather than a tenacity wrapper
)
def _execute_page_request(request):
    """Execute a single commentThreads.list page request, retrying transient failures.

    Isolated into its own function so the retry decorator only wraps the
    network call itself, not the per-item parsing that follows it.
    """
    return request.execute()


def fetch_all_top_level_comments(
    video_id: str, api_key: str, deadline: float | None = None
) -> tuple[list[Comment], bool]:
    """Retrieve every top-level comment thread for a YouTube video via the Data API.

    Uses pagination (nextPageToken) to collect all available comments, up to
    100 per API call, and returns them as a flat list of Comment objects.
    Transient per-page failures (rate limiting, 5xx) are retried with
    exponential backoff before giving up.

    Args:
        video_id: The 11-character YouTube video ID.
        api_key:  A valid YouTube Data API v3 key with quota available.
        deadline: Optional `time.monotonic()` timestamp. If pagination is
            still running after this point, it stops early and returns
            whatever has been collected so far instead of running unbounded
            against a video with an extremely high comment count.

    Returns:
        A tuple of (comments collected so far, whether the deadline cut
        pagination short before all pages were fetched).

    Raises:
        CommentsDisabledError: If the video has comments turned off.
        VideoNotFoundError:    If the video ID does not exist.
        HttpError:             For any other unexpected (non-transient, or
                                transient-but-exhausted-retries) API error.
    """
    # Build the YouTube Data API v3 service object using the provided developer key
    youtube = build("youtube", "v3", developerKey=api_key)

    comments: list[Comment] = []       # Accumulates Comment objects across all pages
    page_token: str | None = None      # None on the first request; populated by the API for subsequent pages
    timed_out = False                  # Set True if the deadline stops pagination before exhaustion

    while True:
        if deadline is not None and time.monotonic() > deadline:
            timed_out = True
            break  # Stop paginating — return whatever has been collected so far

        # Construct the commentThreads.list API request for a single page
        request = youtube.commentThreads().list(
            part="snippet",             # Only request the "snippet" resource part (contains the comment text)
            videoId=video_id,           # Filter to this specific video
            maxResults=100,             # Maximum allowed per page by the API
            pageToken=page_token,       # Omitted (None) on the first call; drives pagination afterwards
            textFormat="plainText",     # Return plain text instead of HTML-encoded text
        )

        try:
            response = _execute_page_request(request)  # Fire the HTTP request, retrying transient errors
        except HttpError as e:
            # Inspect the structured error details to identify specific failure modes
            reason = ""
            if e.error_details:
                reason = e.error_details[0].get("reason", "")  # e.g. "commentsDisabled", "forbidden"

            if reason == "commentsDisabled":
                # Comments are turned off for this video — re-raise as a domain-specific error
                raise CommentsDisabledError(f"Comments are disabled for video {video_id}") from e

            if e.resp.status == 404:
                # The video ID does not correspond to a real video
                raise VideoNotFoundError(f"Video not found: {video_id}") from e

            raise  # All other API errors propagate unchanged to the caller

        # Extract the top-level comment from each thread item in the page
        for item in response.get("items", []):
            # Each "item" is a commentThread; the actual comment lives one level deeper
            top_level = item["snippet"]["topLevelComment"]["snippet"]
            comments.append(
                Comment(
                    comment_id=item["snippet"]["topLevelComment"]["id"],  # Unique comment ID
                    author=top_level.get("authorDisplayName", ""),        # Channel display name
                    text=top_level.get("textOriginal", ""),               # Raw comment body text
                )
            )

        # Check whether another page of results exists
        page_token = response.get("nextPageToken")
        if not page_token:
            break  # No more pages — all comments have been collected

    return comments, timed_out
