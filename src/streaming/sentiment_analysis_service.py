#!/usr/bin/env python3
"""Sentiment Analysis Service for Multi-Language Sentiment Analysis Engine.
This Kafka-based service consumes preprocessed text, performs model inference,
then publishes sentiment results to downstream topics for aggregation and APIs.

Author: Gabriel Demetrios Lafis
Version: 0.1.0

Overview
- Consumes: preprocessed_text
- Produces: sentiment_results
- Dependencies: Hugging Face Transformers or custom model registry (lazy-loaded)

Design Notes
- This module is intentionally lightweight and pluggable. Replace the stubbed
  ModelRegistry and SentimentModel with your production implementations.
- Add observability hooks (Prometheus, OpenTelemetry) at the indicated points.
- Add retries/DLQ via Kafka settings or custom logic in _process_messages.

Future Integrations
- Feature store for input enrichment and ABSA features
- Online model registry (MLflow) and A/B testing of models
- Batch backfill job using the same processing core
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from kafka import KafkaConsumer, KafkaProducer

# Optional heavy deps (transformers, torch) can be imported lazily in ModelRegistry


@dataclass
class SentimentResult:
    """Structured result for a single sentiment inference."""
    id: Optional[str]
    text: str
    language: Optional[str]
    sentiment: str
    score: float
    model_name: str
    timestamp: Optional[str]
    source: str = "unknown"
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "language": self.language,
            "sentiment": self.sentiment,
            "score": self.score,
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "source": self.source,
            "metadata": self.metadata or {},
        }


class SentimentModel:
    """Abstracts a sentiment model. Replace with HF pipelines or custom models.

    Example (future):
        from transformers import pipeline
        self._pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base")
    """

    def __init__(self, model_name: str = "stub-xlm-r"):
        self.model_name = model_name
        self.logger = logging.getLogger(self.__class__.__name__)
        # TODO: Load real model here
        self._loaded = True

    def predict(self, text: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Return label and score. Replace with real model inference.
        For demonstration, uses a trivial heuristic.
        """
        if not text:
            return {"label": "neutral", "score": 0.0}
        # Very naive heuristic placeholder
        txt = text.lower()
        if any(x in txt for x in ["great", "bom", "ótimo", "excelente", "love"]):
            return {"label": "positive", "score": 0.92}
        if any(x in txt for x in ["bad", "ruim", "horrível", "terrible", "hate"]):
            return {"label": "negative", "score": 0.91}
        return {"label": "neutral", "score": 0.55}


class ModelRegistry:
    """Simple registry that picks a model by language or name.

    Replace with MLflow, custom registry, or a local cache of HF models.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._cache: Dict[str, SentimentModel] = {}

    def get(self, language: Optional[str]) -> SentimentModel:
        # Example strategy: choose multilingual default for None/unknown
        key = language or "multilingual"
        if key not in self._cache:
            # TODO: Map language -> model_name (e.g., "en" -> "distilbert-base-uncased-finetuned-sst-2-english")
            model_name = "xlm-r-multilingual-stub"
            self._cache[key] = SentimentModel(model_name=model_name)
        return self._cache[key]


class SentimentAnalysisService:
    """Kafka-based sentiment analysis service."""

    def __init__(self, kafka_config: Dict[str, Any]):
        self.kafka_config = kafka_config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.registry = ModelRegistry()
        self.consumer: Optional[KafkaConsumer] = None
        self.producer: Optional[KafkaProducer] = None

    def start(self) -> None:
        self.logger.info("Starting sentiment analysis service...")
        self.consumer = KafkaConsumer(
            "preprocessed_text",
            bootstrap_servers=self.kafka_config["bootstrap_servers"],
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            group_id=self.kafka_config.get("group_id", "sentiment_group"),
            enable_auto_commit=True,
            auto_offset_reset=self.kafka_config.get("auto_offset_reset", "latest"),
        )
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_config["bootstrap_servers"],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks=self.kafka_config.get("acks", "all"),
            linger_ms=self.kafka_config.get("linger_ms", 5),
        )
        self.logger.info("Sentiment analysis service started successfully")
        self._process_messages()

    def _process_messages(self) -> None:
        for message in self.consumer:
            try:
                data = message.value
                result = self._analyze(data)
                if result:
                    self.producer.send("sentiment_results", value=result.to_dict())
                    # TODO: Add metrics increment here (e.g., Prometheus counter)
            except Exception as e:
                self.logger.exception(f"Error processing message: {e}")
                # TODO: Consider sending to DLQ or retrying

    def _analyze(self, data: Dict[str, Any]) -> Optional[SentimentResult]:
        try:
            text = data.get("text")
            if not text:
                return None
            language = data.get("language")
            model = self.registry.get(language)
            pred = model.predict(text=text, language=language)
            return SentimentResult(
                id=data.get("id"),
                text=text,
                language=language,
                sentiment=pred.get("label", "neutral"),
                score=float(pred.get("score", 0.0)),
                model_name=model.model_name,
                timestamp=data.get("timestamp"),
                source=data.get("source", "unknown"),
                metadata=data.get("metadata", {}),
            )
        except Exception as e:
            self.logger.error(f"Error during analysis: {e}")
            return None


def main() -> None:
    """Entrypoint to run the sentiment analysis service.

    Usage
        python src/streaming/sentiment_analysis_service.py

    Env Overrides
        KAFKA_BOOTSTRAP_SERVERS: Comma-separated list of brokers
        KAFKA_GROUP_ID: Consumer group id
        KAFKA_AUTO_OFFSET_RESET: earliest|latest
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    kafka_bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").split(",")
    kafka_config = {
        "bootstrap_servers": kafka_bootstrap,
        "group_id": os.getenv("KAFKA_GROUP_ID", "sentiment_group"),
        "auto_offset_reset": os.getenv("KAFKA_AUTO_OFFSET_RESET", "latest"),
        "acks": os.getenv("KAFKA_ACKS", "all"),
        "linger_ms": int(os.getenv("KAFKA_LINGER_MS", "5")),
    }

    service = SentimentAnalysisService(kafka_config)
    try:
        service.start()
    except KeyboardInterrupt:
        logging.info("Sentiment analysis service stopped by user")
    except Exception as e:
        logging.error(f"Sentiment analysis service failed: {e}")


if __name__ == "__main__":
    main()
