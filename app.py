"""Streamlit app for MediaGuard.

Usage:
  streamlit run app.py --server.port=8000 --server.address=0.0.0.0
"""
import logging
import io
from PIL import Image
import streamlit as st

from modules.ai_analysis import AIAnalysisEngine


required_envs = [
    "AZURE_CONTENT_SAFETY_ENDPOINT",
    "AZURE_CONTENT_SAFETY_KEY",
    "AZURE_LANGUAGE_ENDPOINT",
    "AZURE_LANGUAGE_KEY"
]

missing = [var for var in required_envs if not os.getenv(var)]
if missing:
    st.error(f"Missing required environment variables: {', '.join(missing)}")
    raise SystemExit(f"Missing env vars: {', '.join(missing)}")
else:
    st.info("All required environment variables are set.")

# ------------------ Logging ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------ Streamlit Config ------------------
st.set_page_config(page_title="MediaGuard", layout="centered")
st.title("MediaGuard — Image & Text Safety Analyzer")
st.markdown(
    "Upload an image and optionally enter text. MediaGuard will analyze them "
    "using Microsoft Azure AI Content Safety and Language models."
)

# ------------------ Inputs ------------------
uploader = st.file_uploader(
    "Choose an image", type=["png", "jpg", "jpeg", "webp", "bmp"]
)

text_input = st.text_area(
    "Optional text to analyze (max 5000 characters):",
    placeholder="Enter text here...",
    height=150,
)

# ------------------ Analysis ------------------
if uploader is not None or (text_input and text_input.strip()):
    try:
        img_bytes = None
        image = None
        if uploader is not None:
            img_bytes = uploader.read()
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            st.image(image, caption="Uploaded image", use_column_width=True)

        st.info("Sending data to Azure AI for analysis...")

        engine = AIAnalysisEngine()
        result = engine.analyze(img_bytes if img_bytes else None, text_input.strip() if text_input else None)

        # ------------------ Result Handling ------------------
        if result.get("classification") == "analysis_failed":
            st.error("Analysis failed: " + result.get("error", "Unknown error"))
            st.json(result)
        else:
            st.success(
                f"Classification: {result['classification']} — Risk: {result['risk']} — Confidence: {result['confidence']}"
            )

            # Provider
            st.subheader("Provider")
            st.write(result.get("provider"))

            # Image categories breakdown
            if "categories" in result:
                st.subheader("Image Category Breakdown")
                st.json(result["categories"])

            # Text sentiment analysis
            if "text_analysis" in result:
                st.subheader("Text Sentiment Analysis")
                st.json(result["text_analysis"])

            # Full raw response
            st.subheader("Full Response")
            st.json(result)

    except Exception as e:
        logger.exception("Unhandled error in app during analysis")
        st.error("Unexpected error: " + str(e))

else:
    st.info("Upload an image or enter text to begin analysis.")

