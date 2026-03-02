"""
============================================================================
AIT-204 Deep Learning | Topic 4: Natural Language Processing
ACTIVITY 4 — Part B: Frontend Web Application (Streamlit)
============================================================================

FRONTEND / BACKEND ARCHITECTURE
──────────────────────────────────────────────────────────────────────────
This file is the FRONTEND. It contains zero model logic.

    ┌──────────────────────────────────────────────────────────────────┐
    │  FRONTEND  (THIS FILE — Streamlit)                               │
    │    Creates the web page.  Handles user input.                    │
    │    Calls service.predict() / service.compare().                  │
    │    Formats the returned dicts as visual components.              │
    ├──────────────────────────────────────────────────────────────────┤
    │  BACKEND   (model_service.py)                                    │
    │    Loads model. Runs PyTorch inference. Returns dicts.           │
    │    Zero UI code. Independently testable.                         │
    └──────────────────────────────────────────────────────────────────┘

============================================================================
"""

import streamlit as st
from model_service import SentimentService


# =========================================================================
# PAGE CONFIGURATION
# Must be the first Streamlit call in the script.
# =========================================================================
st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
)


# =========================================================================
# STEP 1: INITIALIZE THE BACKEND SERVICE (cached)
# =========================================================================
@st.cache_resource
def load_service():
    # Backend service loads vocab + model weights and provides predict/compare.
    return SentimentService()


service = load_service()


# =========================================================================
# APP HEADER
# =========================================================================
st.title("Movie Review Sentiment Analyzer")
st.caption(
    "AIT-204 Deep Learning · Topic 4 · "
    "Built with PyTorch + Streamlit · Trained on movie reviews"
)
st.divider()


# =========================================================================
# STEP 2: TAB NAVIGATION (frontend routing between features)
# =========================================================================
tab1, tab2 = st.tabs(["Sentiment Analysis", "Translation Comparison"])


# =========================================================================
# TAB 1 — SENTIMENT ANALYSIS
# =========================================================================
with tab1:
    st.subheader("Analyze a Movie Review")
    st.caption(
        "Type or paste a movie review. "
        "The backend runs the full NLP pipeline and returns a prediction."
    )

    review = st.text_area(
        "Movie Review",
        placeholder="Type or paste a movie review here...",
        height=120,
    )

    if st.button("Analyze Sentiment", type="primary"):
        if not review.strip():
            st.warning("Please enter a review before clicking Analyze.")
        else:
            result = service.predict(review)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sentiment", result["sentiment"])
            with col2:
                st.metric("Confidence", f"{result['confidence']:.1%}")

            st.progress(
                float(result["positive_score"]),
                text=f"Positive score: {result['positive_score']:.3f}",
            )

            with st.expander("Preprocessing Pipeline"):
                st.write("**Cleaned:**", result["cleaned"])
                st.write("**Tokens:**", result["tokens"])
                st.write("**Encoded IDs:**", result["encoded"])
                st.caption(
                    f"Vocabulary coverage: "
                    f"{result['known_count']}/{len(result['tokens'])} tokens known"
                )


# =========================================================================
# TAB 2 — TRANSLATION COMPARISON
# =========================================================================
with tab2:
    st.subheader("Compare Original vs. Translated")
    st.caption(
        "Paste a review and its round-trip translation "
        "(English → other language → English back). "
        "The model scores both and shows the sentiment shift."
    )

    in1, in2 = st.columns(2)
    with in1:
        original = st.text_area("Original (English)", height=140)
    with in2:
        translated = st.text_area("Round-trip Translation", height=140)

    if st.button("Compare Sentiments", type="primary"):
        if not original.strip() or not translated.strip():
            st.warning("Please enter both texts before comparing.")
        else:
            result = service.compare(original, translated)

            # --- headline comparison ---
            c1, c2 = st.columns(2)
            with c1:
                st.metric(
                    "Original",
                    result["original"]["sentiment"],
                    f"{result['original']['positive_score']:.3f} pos",
                )
                st.caption(f"Confidence: {result['original']['confidence']:.1%}")

            with c2:
                st.metric(
                    "Translated",
                    result["translated"]["sentiment"],
                    f"{result['translated']['positive_score']:.3f} pos",
                )
                st.caption(f"Confidence: {result['translated']['confidence']:.1%}")

            # --- delta indicator ---
            if result.get("changed"):
                st.warning(f"Sentiment CHANGED (delta: {result['delta']:+.3f})")
            else:
                st.success(f"Sentiment preserved (delta: {result['delta']:+.3f})")

            # --- word-level differences (if provided by backend) ---
            if result.get("lost_words"):
                st.write("**Words lost in translation:**", result["lost_words"])
            if result.get("new_words"):
                st.write("**New words from translation:**", result["new_words"])

            # --- optional: preprocessing details for each text ---
            with st.expander("Preprocessing: Original"):
                st.write("**Cleaned:**", result["original"]["cleaned"])
                st.write("**Tokens:**", result["original"]["tokens"])
                st.write("**Encoded IDs:**", result["original"]["encoded"])
                st.caption(
                    f"Vocabulary coverage: "
                    f"{result['original']['known_count']}/{len(result['original']['tokens'])} tokens known"
                )

            with st.expander("Preprocessing: Translated"):
                st.write("**Cleaned:**", result["translated"]["cleaned"])
                st.write("**Tokens:**", result["translated"]["tokens"])
                st.write("**Encoded IDs:**", result["translated"]["encoded"])
                st.caption(
                    f"Vocabulary coverage: "
                    f"{result['translated']['known_count']}/{len(result['translated']['tokens'])} tokens known"
                )


# =========================================================================
# FOOTER
# =========================================================================
st.divider()
st.caption(
    "AIT-204 Deep Learning · Topic 4 · Grand Canyon University  |  "
    "Architecture: Embedding → AvgPool → FC → Sigmoid"
)