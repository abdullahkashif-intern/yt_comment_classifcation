"""
tests/test_classifier.py — Integration test for the LLM comment classifier.

This test loads a small, hand-curated fixture of YouTube comments (with
human-assigned expected categories), sends them through the real classify_comments
pipeline, and verifies structural invariants about the output:
  - Every input comment must appear in either the results or the dropped count.
  - Every returned comment must have at least one category.
  - Every category must be one of the five valid values.

It deliberately does NOT assert that the model assigned the *correct* labels —
that would make the test brittle and model-dependent. Instead, it prints a
side-by-side comparison table (expected vs. actual) so a developer can review
label quality manually after a run, as recommended in CLAUDE.md.

Run with:  pytest tests/test_classifier.py -s   (the -s flag shows print output)
"""

import json      # Standard library: for parsing the fixture JSON file
import sys       # Standard library: used to reconfigure stdout encoding on Windows
from pathlib import Path  # Standard library: cross-platform path manipulation

# Windows consoles default to cp1252, which cannot encode emoji characters that
# appear in many YouTube comments. Reconfigure stdout to UTF-8 (with replacement
# for any remaining unencodable characters) so the print table doesn't crash.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from llm.classifier import classify_comments  # The function under test
from schemas import Comment                   # Pydantic model used to build the input list

# Absolute path to the fixture file, derived from this test file's location so
# the test works regardless of the working directory pytest is invoked from.
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_comments.json"

# Set of all valid category strings — used to assert that the LLM's output
# stays within the allowed vocabulary defined in schemas.py and the system prompt.
VALID_CATEGORIES = {"appreciation", "question", "comparison", "funny", "criticism"}


def _load_fixture() -> list[dict]:
    """Read and parse the sample_comments.json fixture file.

    Returns:
        A list of dicts, each containing:
          - comment_id (str)
          - author (str)
          - text (str)
          - expected_categories (list[str]) — human-assigned ground truth labels
    """
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))  # UTF-8 to handle emoji in comments


def test_classify_sample_comments():
    """Structural integration test for classify_comments().

    Sends the fixture comments through the live LLM pipeline and asserts
    shape/count invariants only. Label correctness is verified manually
    by inspecting the printed comparison table — see CLAUDE.md for rationale.
    """
    fixture = _load_fixture()  # Load the hand-curated test data from the JSON file

    # Convert each fixture dict into a Comment Pydantic object as classify_comments expects
    comments = [
        Comment(comment_id=row["comment_id"], author=row["author"], text=row["text"])
        for row in fixture
    ]

    # Run the full classification pipeline (makes real LLM API calls)
    summary = classify_comments(comments)
    results = summary.results

    # --- Structural invariants ---
    # Every input comment must be accounted for: classified, dropped, failed, or timed out
    assert len(results) + summary.dropped_comment_count + summary.failed_comment_count + \
        summary.timed_out_comment_count == len(comments)

    for result in results:
        # Each returned comment must have been assigned at least one category
        assert result.categories, f"{result.comment_id} came back with no categories"
        # All assigned categories must be within the allowed vocabulary
        assert set(result.categories).issubset(VALID_CATEGORIES)

    # --- Manual review table ---
    # Build a lookup dict so we can find each result by comment_id in O(1)
    results_by_id = {r.comment_id: r for r in results}

    # Print a header row for the comparison table
    print(f"\n{'comment_id':<6} {'expected':<35} {'actual':<35} text")

    for row in fixture:
        result = results_by_id.get(row["comment_id"])
        # If the comment was dropped, represent it as ["DROPPED"] for visibility
        actual = result.categories if result else ["DROPPED"]
        print(
            f"{row['comment_id']:<6} "
            f"{str(row['expected_categories']):<35} "  # Human-assigned ground truth
            f"{str(actual):<35} "                       # LLM-assigned output
            f"{row['text'][:50]}"                       # First 50 chars of the comment text
        )

    # Remind the developer that dropped comments are expected in some edge cases
    if summary.dropped_comment_count:
        print(f"\n{summary.dropped_comment_count} comment(s) dropped by the LLM — see CLAUDE.md known behaviors.")
    if summary.failed_batch_count:
        print(f"\n{summary.failed_batch_count} batch(es) failed after retries "
              f"({summary.failed_comment_count} comment(s) affected).")
