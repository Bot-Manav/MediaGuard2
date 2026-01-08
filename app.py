"""
Streamlit app for MediaGuard.
"""

import os
import traceback
from typing import Optional

import streamlit as st
from modules.ai_analysis import AIAnalysisEngine


def _env_status():
    keys = [
        "AZURE_CONTENT_SAFETY_ENDPOINT",
        "AZURE_CONTENT_SAFETY_KEY",
    ]
    found = {k: os.environ.get(k) for k in keys}
    missing = [k for k, v in found.items() if not v]
    return found, missing


def main():
    st.set_page_config(page_title="MediaGuard — Content Safety Analyzer")
    st.title("MediaGuard — Image & Text Safety Analyzer")

    found, missing = _env_status()
    if missing:
        st.warning("Missing environment variables: " + ", ".join(missing))

    uploaded_file = st.file_uploader(
        "Upload an image (optional)",
        type=["png", "jpg", "jpeg", "bmp", "webp"]
    )

    text_input = st.text_area("Optional text to analyze", height=160)

    if st.button("Analyze"):
        if not uploaded_file and not text_input:
            st.warning("Provide at least an image or some text.")
            return

        engine = AIAnalysisEngine(
            content_safety_endpoint=found.get("AZURE_CONTENT_SAFETY_ENDPOINT"),
            content_safety_key=found.get("AZURE_CONTENT_SAFETY_KEY"),
        )

        img_bytes: Optional[bytes] = None
        if uploaded_file:
            uploaded_file.seek(0)
            img_bytes = uploaded_file.read()

        try:
            with st.spinner("Running analysis..."):
                result = engine.analyze(
                    image=img_bytes,
                    text=text_input or None
                )

            if result.get("analysis_failed"):
                st.error(result.get("error"))
                return

            st.subheader("Summary")
            st.write("**Classification:**", result["classification"])
            st.write("**Risk:**", f"{result['risk_percentage']}%")

            if result.get("image"):
                st.subheader("Image Analysis")
                if result["image"]["analysis_failed"]:
                    st.error(result["image"]["error"])
                else:
                    st.json(result["image"]["categories"])

            if result.get("text"):
                st.subheader("Text Analysis")
                if result["text"]["analysis_failed"]:
                    st.error(result["text"]["error"])
                else:
                    st.json(result["text"]["categories"])

            st.subheader("Raw Output")
            st.json(result)

        except Exception:
            st.error("Unexpected error")
            st.text(traceback.format_exc())


if __name__ == "__main__":
    main()
