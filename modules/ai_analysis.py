import os
import time
import requests
from typing import Dict, Any


class AIAnalysisEngine:
    def __init__(self):
        self.content_safety_endpoint = os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT")
        self.content_safety_key = os.getenv("AZURE_CONTENT_SAFETY_KEY")
        self.language_endpoint = os.getenv("AZURE_LANGUAGE_ENDPOINT")
        self.language_key = os.getenv("AZURE_LANGUAGE_KEY")

    # ===============================
    # PUBLIC ENTRY POINT
    # ===============================
    def analyze(self, image_bytes: bytes = None, text: str = None) -> Dict[str, Any]:
        results = []

        if text:
            results.append(self._analyze_text(text))

        if image_bytes:
            results.append(self._analyze_image(image_bytes))

        if not results:
            return self._fail("No input provided")

        if any(r.get("analysis_failed") for r in results):
            return next(r for r in results if r.get("analysis_failed"))

        max_risk = max(r["risk"] for r in results)
        merged_categories = {}

        for r in results:
            for k, v in r["categories"].items():
                merged_categories[k] = max(merged_categories.get(k, 0), v)

        return {
            "classification": self._classify(max_risk),
            "risk": round(max_risk, 3),
            "confidence": round(max_risk, 3),
            "categories": merged_categories,
            "provider": "Azure AI Content Safety",
            "raw": results,
        }

    # ===============================
    # TEXT ANALYSIS
    # ===============================
    def _analyze_text(self, text: str) -> Dict[str, Any]:
        if not self.language_endpoint or not self.language_key:
            return self._fail("Azure Language credentials missing")

        submit_url = (
            self.language_endpoint.rstrip("/")
            + "/language/analyze-text/jobs?api-version=2023-10-01"
        )

        headers = {
            "Ocp-Apim-Subscription-Key": self.language_key,
            "Content-Type": "application/json",
        }

        payload = {
            "kind": "ContentSafety",
            "analysisInput": {
                "documents": [{"id": "1", "language": "en", "text": text}]
            },
        }

        try:
            submit = requests.post(submit_url, headers=headers, json=payload)
            submit.raise_for_status()
            job_url = submit.headers["operation-location"]

            for _ in range(10):
                time.sleep(1)
                poll = requests.get(
                    job_url,
                    headers={"Ocp-Apim-Subscription-Key": self.language_key},
                )
                poll.raise_for_status()
                data = poll.json()

                if data["status"] == "succeeded":
                    break
            else:
                return self._fail("Text analysis timed out")

            analysis = data["results"]["documents"][0]["contentSafety"]

        except Exception as e:
            return self._fail(f"Text analysis failed: {e}")

        categories = {}
        max_severity = 0.0

        for name, info in analysis.items():
            sev = float(info.get("severity", 0))
            categories[name.lower()] = round(sev / 6.0, 3)
            max_severity = max(max_severity, sev)

        risk = max_severity / 6.0

        return {
            "risk": round(risk, 3),
            "categories": categories,
        }

    # ===============================
    # IMAGE ANALYSIS
    # ===============================
    def _analyze_image(self, image_bytes: bytes) -> Dict[str, Any]:
        if not self.content_safety_endpoint or not self.content_safety_key:
            return self._fail("Azure Content Safety credentials missing")

        url = (
            self.content_safety_endpoint.rstrip("/")
            + "/contentsafety/image:analyze?api-version=2023-10-01"
        )

        headers = {
            "Ocp-Apim-Subscription-Key": self.content_safety_key,
            "Content-Type": "application/octet-stream",
        }

        try:
            resp = requests.post(url, headers=headers, data=image_bytes, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return self._fail(f"Image analysis failed: {e}")

        categories = {}
        max_severity = 0.0

        for item in data.get("categoriesAnalysis", []):
            sev = float(item.get("severity", 0))
            cat = item.get("category", "").lower().replace(" ", "_")
            categories[cat] = round(sev / 6.0, 3)
            max_severity = max(max_severity, sev)

        risk = max_severity / 6.0

        return {
            "risk": round(risk, 3),
            "categories": categories,
        }

    # ===============================
    # HELPERS
    # ===============================
    def _classify(self, risk: float) -> str:
        if risk >= 0.7:
            return "high_risk"
        if risk >= 0.4:
            return "medium_risk"
        return "low_risk"

    def _fail(self, msg: str) -> Dict[str, Any]:
        return {
            "classification": "analysis_failed",
            "confidence": 0,
            "risk": None,
            "error": msg,
            "categories": {
                "violence": 0,
                "sexual": 0,
                "self_harm": 0,
                "hate": 0,
            },
        }
