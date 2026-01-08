"""
AI analysis engine for MediaGuard.

Uses Azure Content Safety for:
- Image moderation
- Text moderation

Severity (0–4) is normalized to risk (0.0–1.0).
"""

import os
import logging
import base64
from typing import Optional, Any, Dict

import requests

logger = logging.getLogger(__name__)


class AIAnalysisEngine:
    def __init__(
        self,
        content_safety_endpoint: Optional[str] = None,
        content_safety_key: Optional[str] = None,
    ):
        self.endpoint = (
            content_safety_endpoint
            or os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT")
        )
        self.key = (
            content_safety_key
            or os.getenv("AZURE_CONTENT_SAFETY_KEY")
        )

        if self.endpoint:
            self.endpoint = self.endpoint.rstrip("/")

    # ---------------------------
    # PUBLIC ENTRY
    # ---------------------------
    def analyze(
        self,
        image: Optional[Any] = None,
        text: Optional[str] = None,
    ) -> Dict[str, Any]:

        result = {
            "analysis_failed": False,
            "error": None,
            "image": None,
            "text": None,
            "final_risk": 0.0,
            "risk_percentage": 0.0,
            "classification": "unknown",
        }

        if image is None and not text:
            return self._fail_global("No image or text provided")

        risks = []

        try:
            if image is not None:
                img_res = self._analyze_image(image)
                result["image"] = img_res
                if not img_res["analysis_failed"]:
                    risks.append(img_res["risk"])

            if text:
                txt_res = self._analyze_text(text)
                result["text"] = txt_res
                if not txt_res["analysis_failed"]:
                    risks.append(txt_res["risk"])

            max_risk = max(risks) if risks else 0.0
            result["final_risk"] = max_risk
            result["risk_percentage"] = round(max_risk * 100, 2)
            result["classification"] = self._classify(max_risk)

        except Exception as e:
            logger.exception("Fatal analysis error")
            return self._fail_global(str(e))

        return result

    # ---------------------------
    # IMAGE ANALYSIS (AZURE-CORRECT)
    # ---------------------------
    def _analyze_image(self, image: Any) -> Dict[str, Any]:
        if not self.endpoint or not self.key:
            return self._fail("Missing Content Safety credentials")

        try:
            if isinstance(image, (bytes, bytearray)):
                image_bytes = image
            elif hasattr(image, "read"):
                image.seek(0)
                image_bytes = image.read()
            elif isinstance(image, str) and os.path.exists(image):
                with open(image, "rb") as f:
                    image_bytes = f.read()
            else:
                return self._fail("Unsupported image input type")
        except Exception as e:
            return self._fail(f"Failed to read image: {e}")

        if not image_bytes:
            return self._fail("Empty image provided")

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "image": {
                "content": image_b64
            }
        }

        headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": "application/json",
        }

        url = f"{self.endpoint}/contentsafety/image:analyze?api-version=2023-10-01"

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            return self._parse_content_safety(resp.json())
        except requests.RequestException as e:
            return self._fail(f"Content Safety image request failed: {e}")
        except ValueError:
            return self._fail("Invalid JSON response from image moderation")

    # ---------------------------
    # TEXT ANALYSIS
    # ---------------------------
    def _analyze_text(self, text: str) -> Dict[str, Any]:
        if not self.endpoint or not self.key:
            return self._fail("Missing Content Safety credentials")

        text = text.strip()
        if not text:
            return self._fail("Empty text input")

        payload = {"text": text}

        headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": "application/json",
        }

        url = f"{self.endpoint}/contentsafety/text:analyze?api-version=2023-10-01"

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            return self._parse_content_safety(resp.json())
        except requests.RequestException as e:
            return self._fail(f"Content Safety text request failed: {e}")
        except ValueError:
            return self._fail("Invalid JSON response from text moderation")

    # ---------------------------
    # SHARED PARSER
    # ---------------------------
    def _parse_content_safety(self, data: Dict[str, Any]) -> Dict[str, Any]:
        categories = {}
        max_severity = 0

        for item in data.get("categoriesAnalysis", []):
            name = item.get("category")
            severity = item.get("severity", 0)
            if name:
                categories[name.lower()] = severity
                max_severity = max(max_severity, severity)

        risk = max_severity / 4.0

        return {
            "analysis_failed": False,
            "raw": data,
            "categories": categories,
            "risk": round(risk, 3),
            "confidence": round(risk, 3),
        }

    # ---------------------------
    # HELPERS
    # ---------------------------
    @staticmethod
    def _classify(risk: float) -> str:
        if risk >= 0.7:
            return "unsafe"
        if risk >= 0.4:
            return "sensitive"
        return "safe"

    @staticmethod
    def _fail(msg: str) -> Dict[str, Any]:
        return {
            "analysis_failed": True,
            "error": msg,
            "risk": 0.0,
            "confidence": 0.0,
            "categories": {},
        }

    @staticmethod
    def _fail_global(msg: str) -> Dict[str, Any]:
        return {
            "analysis_failed": True,
            "error": msg,
            "image": None,
            "text": None,
            "final_risk": 0.0,
            "risk_percentage": 0.0,
            "classification": "unknown",
        }
