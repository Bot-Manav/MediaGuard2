"""
Streamlit app for MediaGuard.

Usage:
  streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
"""

import os
import traceback
import streamlit as st

from modules.ai_analysis import AIAnalysisEngine


# ===============================
# ENV CHECK
# ===============================
def env_status():
    keys = [
        "AZURE_CONTENT_SAFETY_ENDPOINT",
        "AZURE_CONTENT_SAFETY_KEY",
        "AZURE_LANGUAGE_ENDPOINT",
        "AZURE_LANGUAGE_KEY",
    ]
    found = {k: os.getenv(k) for k in keys}
    missing = [k for k, v in found.items() if not v]
    return found, missing


# ===============================
# MAIN APP
# ===============================
def main():
    # Azure App Service compatibility
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    st.set_page_config(
        page_title="MediaGuard",
        page_icon="🛡️",
        layout="centered",
    )

    st.title("🛡️ MediaGuard")
    st.caption("AI-powered image & text safety analysis using Azure AI")

    found, missing = env_status()

    if missing:
        st.warning("Some Azure environment variables are missing:")
        st.code("\n".join(missing))

    # ===============================
    # INPUTS
    # ===============================
    uploaded_file = st.file_uploader(
        "Upload an image (optional)",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
    )

    text_input = st.text_area(
        "Paste text to analyze (optional)",
        height=160,
        placeholder="Enter any text you want to check for safety risks…",
    )

    st.markdown("---")

    if not st.button("Analyze"):
        st.info("Provide at least one input and click **Analyze**.")
        return

    if not uploaded_file and not text_input.strip():
        st.error("You must provide either an image, text, or both.")
        return

    # ===============================
    # PREPARE INPUTS
    # ===============================
    image_bytes = None
    if uploaded_file:
        try:
            uploaded_file.seek(0)
            image_bytes = uploaded_file.read()
        except Exception:
            st.error("Failed to read uploaded image.")
            return

    # ===============================
    # RUN ANALYSIS
    # ===============================
    engine = AIAnalysisEngine()

    try:
        with st.spinner("Analyzing content…"):
            result = engine.analyze(
                image_bytes=image_bytes,
                text=text_input.strip() or None,
            )
    except Exception:
        st.error("Unexpected error during analysis")
        st.text(traceback.format_exc())
        return

    # ===============================
    # HANDLE FAILURE
    # ===============================
    if result.get("classification") == "analysis_failed":
        st.error("Analysis failed")
        st.code(result.get("error", "Unknown error"))
        return

    # ===============================
    # DISPLAY INPUTS
    # ===============================
    st.markdown("## Inputs")

    if uploaded_file:
        st.image(uploaded_file, use_column_width=True)
    else:
        st.info("No image provided")

    if text_input.strip():
        st.markdown("**Text:**")
        st.write(text_input)
    else:
        st.info("No text provided")

    # ===============================
    # SUMMARY
    # ===============================
    st.markdown("---")
    st.markdown("## 🧠 Analysis Summary")

    st.metric(
        label="Overall Risk",
        value=f"{int(result['risk'] * 100)}%",
    )

    st.write("**Classification:**", result["classification"])

    # ===============================
    # CATEGORY BREAKDOWN
    # ===============================
    st.markdown("---")
    st.markdown("## 📊 Category Breakdown")

    if result.get("categories"):
        st.json(result["categories"])
    else:
        st.info("No category data available")

    # ===============================
    # RAW DEBUG (OPTIONAL)
    # ===============================
    with st.expander("🔍 Full Raw Response"):
        st.json(result)


if __name__ == "__main__":
    main()
