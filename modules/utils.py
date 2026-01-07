import io
import os
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def image_to_bytes(img) -> bytes:
    """Accept a PIL Image, file-like, or bytes and return raw bytes.

    This function does not perform any analysis - just conversion.
    """
    if img is None:
        raise ValueError("No image provided")

    # If it's already bytes
    if isinstance(img, (bytes, bytearray)):
        return bytes(img)

    # If it's a file-like with read()
    if hasattr(img, "read"):
        data = img.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        return data

    # If it's a PIL Image
    if isinstance(img, Image.Image):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    raise TypeError("Unsupported image type for conversion to bytes")


def require_env(var_name: str) -> str:
    """Read an environment variable and raise a clear error if missing.

    The app must not use defaults or hardcode secrets.
    """
    val = os.getenv(var_name)
    if not val:
        logger.error("Missing required environment variable: %s", var_name)
        raise EnvironmentError(f"Missing required environment variable: {var_name}")
    return val
