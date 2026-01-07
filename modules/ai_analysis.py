"""
AI analysis engine for MediaGuard.

- Image analysis via Azure Content Safety
- Text analysis via Azure Language
- Fully defensive: never returns None risk
"""

import os
import logging
import mimetypes
from typing import Optional, Dict, Any

import requests

logger = logging.getLogger(__name__)


class AIAnalysisEngine:
    def __init__(self):
        self.content_safety_endpoint = os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT")
        self.content_safety_key = os.getenv("AZURE_CONTENT_SAFETY_KEY")
        self.language_endpoint = os.getenv("AZURE_LANGUAGE_ENDPOINT")
        self.language_key = os.getenv("AZURE_LANGUAGE_KEY")

    # =========================
    # PUBLIC ENTRY
    # =========================
    def analyze(
        self,
        image_bytes: Optional[bytes] = None,
        text: Optional[str] = None,
    ) -> Dict[str, Any]:

        image_risk = 0.0
        text_risk = 0.0
        categories = {}

        # ---------- IMAGE ----------
        if image_bytes:
            img = self._analyze_image(image_bytes)
            image_risk = float(img.get("risk", 0.0))
            categories.update(img.get("categories", {}))

        # ---------- TEXT ----------
        if text:
            txt = self._analyze_text(text)
            text_risk = float(txt.get("risk", 0.0))
            categories.update(txt.get("categories", {}))

        # ---------- FINAL ----------
        final_risk = max(image_risk, text_risk)
        final_risk = self._clamp(final_risk)

        return {
            "risk": final_risk,
            "classification": self._classify(final_risk),
            "categories": categories,
        }

    # =========================
    # IMAGE ANALYSIS
    # =========================
    def _analyze_image(self, image_bytes: bytes) -> Dict[str, Any]:
        if not self.content_safety_endpoint or not self.content_safety_key:
            return {"risk": 0.0, "categories": {}}

        headers = {
            "Ocp-Apim-Subscription-Key": self.content_safety_key,
        }

        files = {
            "file": ("image.jpg", image_bytes, "image/jpeg"),
        }

        try:
            r = requests.post(
                self.content_safety_endpoint,
                headers=headers,
                files=files,
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.warning("Image analysis failed: %s", e)
            return {"risk": 0.0, "categories": {}}

        categories = data.get("categories", {})
        risk = self._extract_max_score(categories)

        return {
            "risk": risk,
            "categories": categories,
        }

    # =========================
    # TEXT ANALYSIS
    # =========================
    def _analyze_text(self, text: str) -> Dict[str, Any]:
        if not self.language_endpoint or not self.language_key:
            return {"risk": 0.0, "categories": {}}

        payload = {
            "documents": [
                {"id": "1", "language": "en", "text": text[:5000]}
            ]
        }

        headers = {
            "Ocp-Apim-Subscription-Key": self.language_key,
            "Content-Type": "application/json",
        }

        try:
            r = requests.post(
                self.language_endpoint,
                headers=headers,
                json=payload,
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.warning("Text analysis failed: %s", e)
            return {"risk": 0.0, "categories": {}}

        docs = data.get("documents", [])
        if not docs:
            return {"risk": 0.0, "categories": {}}

        scores = docs[0].get("confidenceScores", {})
        negative = scores.get("negative", 0.0)

        return {
            "risk": self._clamp(negative),
            "categories": {"negative_sentiment": negative},
        }

    # =========================
    # HELPERS
    # =========================
    @staticmethod
    def _extract_max_score(categories: Dict[str, Any]) -> float:
        values = []
        for v in categories.values():
            if isinstance(v, dict):
                values.extend(v.values())
            elif isinstance(v, (int, float)):
                values.append(v)
        return AIAnalysisEngine._clamp(max(values) if values else 0.0)

    @staticmethod
    def _clamp(value: Any) -> float:
        try:
            v = float(value)
        except Exception:
            return 0.0
        return max(0.0, min(1.0, v))

    @staticmethod
    def _classify(risk: float) -> str:
        if risk >= 0.7:
            return "unsafe"
        if risk >= 0.3:
            return "sensitive"
        return "safe"
