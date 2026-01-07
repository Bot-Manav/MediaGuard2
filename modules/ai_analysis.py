"""AI analysis engine for MediaGuard.

All inference comes from Azure AI JSON responses. No local models or heuristics.
Integrates Azure AI Content Safety for images and Azure AI Language for text sentiment.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional
import requests

from .utils import image_to_bytes, require_env

logger = logging.getLogger(__name__)


class AIAnalysisEngine:
    """Analyze images and text using Microsoft Azure AI services.

    Image Analysis: Azure Content Safety
    Endpoint: POST {AZURE_CONTENT_SAFETY_ENDPOINT}/contentsafety/image:analyze?api-version=2023-10-01
    Response: {"categoriesAnalysis": [{"category":"Violence","severity":0-6}, ...]}

        Text Sentiment: Azure AI Language (async job)
        Endpoint: POST {AZURE_LANGUAGE_ENDPOINT}/language/analyze-text/jobs?api-version=2023-10-01-preview
        Response (jobs):
        {
            "tasks": {
                "items": [
                    {
                        "results": {
                            "documents": [ {"sentiment": "positive|negative|neutral|mixed"} ]
                        }
                    }
                ]
            }
        }

    Final risk score = max(image_risk, text_risk)
    On any API/schema error the engine returns an `analysis_failed` response.
    """

    PROVIDER = "Microsoft Azure AI"

    # Map Azure category names (case-insensitive) to our output keys
    CATEGORY_MAP = {
        "violence": "violence",
        "sexual": "sexual",
        "selfharm": "self_harm",
        "self_harm": "self_harm",
        "hate": "hate",
    }

    MAX_SEVERITY = 6.0

    # Sentiment to risk mapping
    SENTIMENT_RISK_MAP = {
        "negative": 0.6,
        "mixed": 0.4,
        "neutral": 0.0,
        "positive": 0.0,
    }

    def __init__(self) -> None:
        # Azure Content Safety credentials
        self.content_safety_endpoint = require_env("AZURE_CONTENT_SAFETY_ENDPOINT").rstrip("/")
        self.content_safety_key = require_env("AZURE_CONTENT_SAFETY_KEY")

        # Azure Language credentials
        self.language_endpoint = require_env("AZURE_LANGUAGE_ENDPOINT").rstrip("/")
        self.language_key = require_env("AZURE_LANGUAGE_KEY")

        # Optional Content Safety region header
        self.content_safety_region = os.getenv("AZURE_CONTENT_SAFETY_REGION")

        # Build full endpoints
        self.image_endpoint = f"{self.content_safety_endpoint}/contentsafety/image:analyze?api-version=2023-10-01"
        # use jobs path for async responses
        self.text_endpoint = f"{self.language_endpoint}/language/analyze-text/jobs?api-version=2023-10-01-preview"

    def analyze(self, image, text: Optional[str] = None) -> Dict[str, Any]:
        """Analyze image and optional text using Azure AI services.
        
        Args:
            image: Image data (file path, bytes, or file-like object)
            text: Optional text content for sentiment analysis
            
        Returns:
            Analysis result with classification, risk score, and detailed breakdown
        """
        # Analyze image with Azure Content Safety
        image_result = self._analyze_image(image)

        # If image analysis failed, return early
        if image_result.get("classification") == "analysis_failed":
            return image_result

        # Get image risk score (0.0-1.0)
        image_risk = float(image_result.get("confidence", 0.0))
        
        # Analyze text if provided
        text_analysis = None
        text_risk = 0.0
        
        if text:
            text_result = self._analyze_text(text)
            # If text analysis failed, log and continue with image-only risk
            if text_result.get("classification") == "analysis_failed":
                logger.warning("Text analysis failed: %s", text_result.get("error"))
            elif text_result.get("status") == "ok":
                try:
                    sentiment = text_result["sentiment"]
                    risk_val = float(text_result["risk"])
                    text_analysis = {"sentiment": sentiment, "risk": risk_val}
                    text_risk = risk_val
                except Exception:
                    logger.exception("Malformed text analysis result")
            else:
                logger.error("Unexpected text analysis response: %s", json.dumps(text_result)[:800])
        
        # Fusion logic: final risk = max(image_risk, text_risk)
        final_risk = max(image_risk, text_risk)
        final_classification = self._score_to_classification(final_risk)

        # Confidence decreases as risk increases
        confidence = round(1.0 - float(final_risk), 2)

        # Build final result
        result = {
            "classification": final_classification,
            "confidence": confidence,
            "risk": int(round(final_risk * 100)),
            "provider": self.PROVIDER,
            "categories": image_result["categories"],
        }
        
        # Include text analysis if available
        if text_analysis:
            result["text_analysis"] = text_analysis
        
        return result

    def _analyze_image(self, image) -> Dict[str, Any]:
        """Analyze image using Azure Content Safety."""
        try:
            payload = image_to_bytes(image)
        except Exception as e:
            logger.exception("Failed converting image to bytes")
            return {
                "classification": "analysis_failed",
                "confidence": 0,
                "risk": None,
                "error": f"Image conversion failed: {str(e)}",
                "categories": {"violence": 0.0, "sexual": 0.0, "self_harm": 0.0, "hate": 0.0},
            }

        headers = {
            "Ocp-Apim-Subscription-Key": self.content_safety_key,
            "Content-Type": "application/octet-stream",
        }
        # Optional region header
        if self.content_safety_region:
            headers["Ocp-Apim-Subscription-Region"] = self.content_safety_region

        try:
            resp = requests.post(self.image_endpoint, data=payload, headers=headers, timeout=30)
        except requests.RequestException:
            logger.exception("Azure Content Safety request failed")
            return {
                "classification": "analysis_failed",
                "confidence": 0,
                "risk": None,
                "error": "Azure Content Safety request failed",
                "categories": {"violence": 0.0, "sexual": 0.0, "self_harm": 0.0, "hate": 0.0},
            }

        if resp.status_code != 200:
            logger.error("Azure Content Safety returned non-200: %s %s", resp.status_code, resp.text[:400])
            return {
                "classification": "analysis_failed",
                "confidence": 0,
                "risk": None,
                "error": f"Azure Content Safety request failed: status={resp.status_code}",
                "categories": {"violence": 0.0, "sexual": 0.0, "self_harm": 0.0, "hate": 0.0},
            }

        try:
            data = resp.json()
        except ValueError:
            logger.exception("Azure Content Safety returned non-JSON response")
            return {
                "classification": "analysis_failed",
                "confidence": 0,
                "risk": None,
                "error": "Invalid Azure Content Safety response: not JSON",
                "categories": {"violence": 0.0, "sexual": 0.0, "self_harm": 0.0, "hate": 0.0},
            }

        # Parse categoriesAnalysis array
        categories_raw = (data.get("categoriesAnalysis") or data.get("categories_analysis"))
        if not isinstance(categories_raw, list):
            logger.error("Missing or invalid 'categoriesAnalysis' in response: %s", json.dumps(data)[:800])
            return {
                "classification": "analysis_failed",
                "confidence": 0,
                "risk": None,
                "error": "Invalid Azure Content Safety response",
                "categories": {"violence": 0.0, "sexual": 0.0, "self_harm": 0.0, "hate": 0.0},
            }

        parsed: Dict[str, float] = {"violence": 0.0, "sexual": 0.0, "self_harm": 0.0, "hate": 0.0}

        valid_found = False
        for item in categories_raw:
            if not isinstance(item, dict):
                continue
            cat = item.get("category")
            sev = item.get("severity")
            if not isinstance(cat, str):
                continue
            # normalize category token
            key = cat.replace(" ", "").lower()
            mapped = self.CATEGORY_MAP.get(key)
            if mapped is None:
                continue
            # severity must be numeric between 0 and MAX_SEVERITY
            try:
                sev_f = float(sev)
            except (TypeError, ValueError):
                logger.error("Non-numeric severity for category %s: %s", cat, sev)
                return {
                    "classification": "analysis_failed",
                    "confidence": 0,
                    "risk": None,
                    "error": "Invalid Azure Content Safety response",
                    "categories": {"violence": 0.0, "sexual": 0.0, "self_harm": 0.0, "hate": 0.0},
                }
            if not (0.0 <= sev_f <= self.MAX_SEVERITY):
                logger.error("Severity out of range for category %s: %s", cat, sev_f)
                return {
                    "classification": "analysis_failed",
                    "confidence": 0,
                    "risk": None,
                    "error": "Invalid Azure Content Safety response",
                    "categories": {"violence": 0.0, "sexual": 0.0, "self_harm": 0.0, "hate": 0.0},
                }

            # normalize to 0..1
            norm = sev_f / self.MAX_SEVERITY
            parsed[mapped] = round(norm, 3)
            valid_found = True

        if not valid_found:
            logger.error("No valid category severities found in response: %s", json.dumps(data)[:800])
            return {
                "classification": "analysis_failed",
                "confidence": 0,
                "risk": None,
                "error": "Invalid Azure Content Safety response",
                "categories": {"violence": 0.0, "sexual": 0.0, "self_harm": 0.0, "hate": 0.0},
            }

        # compute max normalized score
        max_score = max(parsed.values())

        return {
            "status": "ok",
            "confidence": max_score,
            "categories": parsed,
        }

    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text sentiment using Azure AI Language."""
        if not text or not text.strip():
            logger.warning("Empty text provided for sentiment analysis")
            return {
                "classification": "analysis_failed",
                "error": "Empty text",
            }

        headers = {
            "Ocp-Apim-Subscription-Key": self.language_key,
            "Content-Type": "application/json",
        }

        payload = {
            "kind": "SentimentAnalysis",
            "parameters": {"modelVersion": "latest"},
            "analysisInput": {
                "documents": [
                    {"id": "1", "language": "en", "text": text[:5000]}
                ]
            }
        }

        try:
            resp = requests.post(self.text_endpoint, json=payload, headers=headers, timeout=30)
        except requests.RequestException:
            logger.exception("Azure AI Language request failed")
            return {
                "classification": "analysis_failed",
                "error": "Azure AI Language request failed",
            }

        if resp.status_code != 200:
            logger.error("Azure AI Language returned non-200: %s %s", resp.status_code, resp.text[:400])
            return {
                "classification": "analysis_failed",
                "error": f"Azure AI Language request failed: status={resp.status_code}",
            }

        try:
            data = resp.json()
        except ValueError:
            logger.exception("Azure AI Language returned non-JSON response")
            return {
                "classification": "analysis_failed",
                "error": "Invalid Azure AI Language response: not JSON",
            }

        # Parse async job response: tasks.items[].results.documents
        try:
            tasks = data.get("tasks") if isinstance(data, dict) else None
            if not tasks or not isinstance(tasks, dict):
                logger.error("Missing 'tasks' in language response: %s", json.dumps(data)[:800])
                return {"classification": "analysis_failed", "error": "Invalid Azure AI Language response"}

            items = tasks.get("items")
            if not items or not isinstance(items, list) or len(items) == 0:
                logger.error("Missing 'tasks.items' in language response: %s", json.dumps(data)[:800])
                return {"classification": "analysis_failed", "error": "Invalid Azure AI Language response"}

            # find first item with results.documents
            doc_list = None
            for it in items:
                if not isinstance(it, dict):
                    continue
                results = it.get("results")
                if not results or not isinstance(results, dict):
                    continue
                documents = results.get("documents")
                if documents and isinstance(documents, list) and len(documents) > 0:
                    doc_list = documents
                    break

            if not doc_list:
                logger.error("No documents found in language task results: %s", json.dumps(data)[:800])
                return {"classification": "analysis_failed", "error": "Invalid Azure AI Language response"}

            document = doc_list[0]
            sentiment = (document.get("sentiment") or "").lower()
            if sentiment not in self.SENTIMENT_RISK_MAP:
                logger.error("Unknown sentiment value: %s", sentiment)
                return {"classification": "analysis_failed", "error": "Invalid Azure AI Language response"}

            risk = self.SENTIMENT_RISK_MAP[sentiment]
            return {"status": "ok", "sentiment": sentiment, "risk": risk}

        except Exception:
            logger.exception("Error parsing Azure AI Language job response")
            return {"classification": "analysis_failed", "error": "Invalid Azure AI Language response"}

    def _score_to_classification(self, score: float) -> str:
        if score <= 0.2:
            return "safe"
        if score <= 0.6:
            return "sensitive"
        return "unsafe"
