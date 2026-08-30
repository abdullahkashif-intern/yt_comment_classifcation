"""
frontend/app.py — Streamlit web UI for the YouTube Comment Classifier.

This module is the entire frontend of the application. It is a single-page
Streamlit app that:
  1. Accepts a YouTube video URL from the user via a text input field.
  2. POSTs that URL to the backend FastAPI service (/classify-video endpoint).
  3. Displays summary metrics (total fetched, classified, dropped).
  4. Renders an interactive bar chart of category counts.
  5. Shows a filterable table of every classified comment with its labels.

The app communicates with the backend over HTTP using the `requests` library,
so it can run on a separate process or container from the FastAPI server. The
backend URL is configurable via the API_BASE_URL environment variable or the
sidebar text input, defaulting to http://localhost:8000 for local development.
"""

import os  # Standard library: used to read the API_BASE_URL environment variable

import pandas as pd  # DataFrame operations for reshaping and displaying comment data
import requests      # HTTP client for calling the FastAPI backend
import streamlit as st  # The Streamlit framework — every `st.*` call renders a UI element

# Determine the backend base URL. The environment variable takes precedence
# so that Docker / cloud deployments can override it without code changes.
# The fallback points to localhost for local development.
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

# Configure the browser tab title, favicon emoji, and wide-column layout.
# This must be the first Streamlit call in the script.
st.set_page_config(page_title="YouTube Comment Classifier", page_icon="🎬", layout="wide")

# Render the main page heading and a one-line description beneath it
st.title("YouTube Comment Classifier")
st.caption("Fetches every top-level comment on a video and classifies each one via the LLM pipeline.")

# --- Sidebar: settings panel ---
with st.sidebar:
    st.subheader("Settings")
    # Allow the user to override the backend URL at runtime (useful when the
    # backend is deployed to a remote server rather than localhost)
    api_base_url = st.text_input("Backend API URL", value=API_BASE_URL)

# --- Main input row ---
# Primary text input for the YouTube URL
video_url = st.text_input("YouTube video URL", placeholder="https://www.youtube.com/watch?v=...")
# Primary action button — styled as "primary" to make it visually prominent
submit = st.button("Classify comments", type="primary")

# --- Form submission logic ---
if submit:
    # Guard: require a non-empty URL before making any API call
    if not video_url:
        st.warning("Enter a YouTube video URL first.")
        st.stop()  # Halt further execution for this run — nothing else should render

    # Show a spinner while the backend is working; classification can take
    # a long time for videos with thousands of comments.
    with st.spinner("Fetching and classifying comments — this can take a while for popular videos..."):
        try:
            # POST the video URL to the backend classify endpoint as JSON
            response = requests.post(
                f"{api_base_url}/classify-video",  # Constructed from the sidebar URL input
                json={"video_url": video_url},      # Serialised as {"video_url": "..."}
                timeout=None,                       # No client-side timeout — let the backend control it
            )
        except requests.RequestException as e:
            # Network-level failure (DNS, refused connection, etc.) — tell the user clearly
            st.error(f"Could not reach the backend at {api_base_url}: {e}")
            st.stop()  # Stop rendering; nothing useful can be shown without a response

    # --- Error handling for non-2xx HTTP responses ---
    if not response.ok:
        try:
            # Try to extract the structured "detail" field from FastAPI's JSON error body
            detail = response.json().get("detail", response.text)
        except ValueError:
            # Response body is not valid JSON — fall back to raw text
            detail = response.text
        st.error(f"Request failed ({response.status_code}): {detail}")
        st.stop()  # Stop rendering; the error message is the only thing to show

    # --- Parse the successful response ---
    data = response.json()     # Full ClassifyVideoResponse dict
    results = data["results"]  # List of ClassifiedComment dicts

    # --- Summary metrics row ---
    col1, col2, col3 = st.columns(3)  # Three equal-width columns for the metric cards
    col1.metric("Comments fetched", data["total_comments_fetched"])       # Raw API count
    col2.metric("Comments classified", data["total_comments_classified"]) # LLM-labelled count
    col3.metric("Dropped by LLM", data["dropped_comment_count"])          # Comments with no label

    # --- Chart and table (only shown when at least one result exists) ---
    if results:
        # Convert the list of dicts to a DataFrame for easy manipulation and display
        df = pd.DataFrame(results)
        # `categories` is currently a list per row; join to a comma-separated string
        # so Streamlit's table renderer can display it as plain text.
        df["categories"] = df["categories"].apply(", ".join)

        # Build a frequency count dict: { "appreciation": 12, "funny": 7, ... }
        category_counts: dict[str, int] = {}
        for row in results:
            for category in row["categories"]:
                # Increment the counter for this category, defaulting to 0 if unseen
                category_counts[category] = category_counts.get(category, 0) + 1

        # --- Bar chart ---
        st.subheader("Category breakdown")
        # pd.Series converts the dict to a named Series that st.bar_chart can render
        st.bar_chart(pd.Series(category_counts, name="count"))

        # --- Filterable comment table ---
        st.subheader("Classified comments")
        all_categories = sorted(category_counts.keys())  # Alphabetically sorted for consistent ordering
        # Multi-select widget — returns a list of categories the user wants to filter by
        selected_categories = st.multiselect("Filter by category", options=all_categories)

        # Default to showing all rows; apply filter only when categories are selected
        filtered_df = df
        if selected_categories:
            # Create a boolean mask: True for rows whose comma-separated categories
            # contain at least one of the selected filter values
            mask = df["categories"].apply(
                lambda cats: any(c in cats.split(", ") for c in selected_categories)
            )
            filtered_df = df[mask]  # Apply the mask to keep only matching rows

        # Render the final table — only show author, text, and categories columns
        st.dataframe(
            filtered_df[["author", "text", "categories"]],
            use_container_width=True,  # Stretch to fill the full page width
            hide_index=True,           # Don't show the numeric DataFrame index column
        )
    else:
        # Edge case: backend succeeded but returned zero classified comments
        st.info("No comments were classified for this video.")
