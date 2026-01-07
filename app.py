"""
Streamlit app for MediaGuard.

Usage:
  streamlit run app.py --server.port=8000 --server.address=0.0.0.0
"""

import logging
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
    port = os.environ.get("PORT", "8501")
    os.environ.setdefault("STREAMLIT_SERVER_PORT", port)

    st.set_page_config(page_title="MediaGuard — Image & Text Safety Analyzer")
    st.title("MediaGuard — Image & Text Safety Analyzer")
    st.write("Upload an image and/or paste text. Both are optional; provide at least one to analyze.")

    found, missing = _env_status()
    if missing:
        st.info("Some environment variables are missing; features depending on them will be disabled.")
        st.write("Missing:", ", ".join(missing))

    col1, col2 = st.columns([1, 2])
    uploaded_file: Optional[st.runtime.uploaded_file_manager.UploadedFile] = None
    text_input: Optional[str] = None

    with col1:
        uploaded_file = st.file_uploader(
            "Upload an image (optional)",
            type=["png", "jpg", "jpeg", "bmp", "webp"]
        )

    with col2:
        text_input = st.text_area("Optional text to analyze", height=160)

    st.markdown("---")
    analyze = st.button("Analyze")

    if not analyze:
        st.info("Ready. Provide inputs and click Analyze.")

    if analyze:
        if not uploaded_file and not text_input:
            st.warning("Please provide at least an image or some text to analyze.")
            return

        engine = AIAnalysisEngine(
            content_safety_endpoint=found.get("AZURE_CONTENT_SAFETY_ENDPOINT"),
            content_safety_key=found.get("AZURE_CONTENT_SAFETY_KEY"),
        )

        # Convert uploaded image to bytes
        img_bytes: Optional[bytes] = None
        if uploaded_file is not None:
            try:
                uploaded_file.seek(0)
                img_bytes = uploaded_file.read()
            except Exception:
                st.error("Failed to read uploaded image.")
                img_bytes = None

        try:
            with st.spinner("Running analysis..."):
                result = engine.analyze(image=img_bytes, text=text_input or None)

            if result.get("analysis_failed"):
                st.error("Analysis failed: " + str(result.get("error")))
                st.expander("Error details").write(result.get("error"))
                return

            # Display inputs
            st.subheader("Inputs")
            if uploaded_file:
                try:
                    st.image(uploaded_file, use_column_width=True)
                except Exception:
                    st.write("(Could not render image preview)")
            else:
                st.info("No image provided.")

            if text_input:
                st.write("**Text provided:**")
                st.write(text_input)
            else:
                st.info("No text provided.")

            # Display results
            st.markdown("---")
            st.subheader("Summary")
            st.write(f"**Classification:** {result.get('classification')}")
            st.write(f"**Risk:** {result.get('risk_percentage')}%")

            st.markdown("---")
            st.subheader("Image Analysis")
            if result.get("image"):
                if result["image"].get("analysis_failed"):
                    st.error("Image analysis failed: " + str(result["image"].get("error")))
                else:
                    st.write("**Risk:**", f"{round(result['image'].get('risk', 0)*100,2)}%")
                    st.write("**Confidence:**", f"{round(result['image'].get('confidence', 0)*100,2)}%")
                    st.write("**Categories:**")
                    st.json(result["image"].get("categories", {}))
            else:
                st.info("No image analysis available.")

            st.markdown("---")
            st.subheader("Text Analysis")
            if result.get("text"):
                if result["text"].get("analysis_failed"):
                    st.error("Text analysis failed: " + str(result["text"].get("error")))
                else:
                    st.write("**Risk:**", f"{round(result['text'].get('risk',0)*100,2)}%")
                    st.write("**Confidence:**", f"{round(result['text'].get('confidence',0)*100,2)}%")
                    st.json(result["text"].get("categories", {}))
            else:
                st.info("No text analysis available.")

            st.markdown("---")
            st.subheader("Full Response")
            st.json(result)

        except Exception:
            st.error("Unexpected error while running analysis")
            with st.expander("Traceback"):
                st.text(traceback.format_exc())


if __name__ == "__main__":
    main()
