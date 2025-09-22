#!/usr/bin/env python3
"""Aggregation Service for Multi-Language Sentiment Analysis Engine.
Consumes sentiment_results from Kafka, computes rolling and windowed aggregates,
and publishes aggregated insights to downstream topics and/or data stores.

Author: Gabriel Demetrios Lafis
Version: 0.1.0

Overview
- Consumes: sentiment_results
- Produces: aggregated_insights (optional)
- Optional sinks: Redis, Elasticsearch/ClickHouse, PostgreSQL

Future Integrations
- Exactly-once semantics via idempotent sinks and transactional producers
- Time-window aggregations with Kafka Streams or Flink (migration path noted)
- Expose Prometheus metrics for aggregation latencies and counters
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple

from kafka import KafkaConsumer, KafkaProducer


@dataclass
class AggregateSnapshot:
    """Represents a snapshot of aggregated sentiment counts and averages."""
    total: int
    positive: int
    negative: int
    neutral: int
    avg_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "positive": self.positive,
            "negative": self.negative,
            "neutral": self.neutral,
            "avg_score": self.avg_score,
        }


class RollingAggregator:
    """In-memory rolling aggregator with simple decay/window semantics.

    For production, consider Kafka Streams, Flink, or materialized views in
    an OLAP database. This class is a minimal reference implementation.
    """

    def __init__(self, maxlen: int = 10_000):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.events: Deque[Tuple[str, float]] = deque(maxlen=maxlen)
        self.counts: Dict[str, int] = defaultdict(int)
        self.score_sum: float = 0.0

    def update(self, label: str, score: float) -> None:
        # Evict oldest if at capacity
        if len(self.events) == self.events.maxlen:
            old_label, old_score = self.events[0]
            self.counts[old_label] -= 1
            self.score_sum -= old_score
        self.events.append((label, score))
        self.counts[label] += 1
        self.score_sum += score

    def snapshot(self) -> AggregateSnapshot:
        total = len(self.events)
        pos = self.counts.get("positive", 0)
        neg = self.counts.get("negative", 0)
        neu = self.counts.get("neutral", 0)
        avg = (self.score_sum / total) if total else 0.0
        return AggregateSnapshot(total=total, positive=pos, negative=neg, neutral=neu, avg_score=avg)


class AggregationService:
    """Kafka-based service that aggregates sentiment results."""

    def __init__(self, kafka_config: Dict[str, Any]):
        self.kafka_config = kafka_config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.consumer: Optional[KafkaConsumer] = None
        self.producer: Optional[KafkaProducer] = None
        self.aggregator = RollingAggregator(maxlen=int(os.getenv("AGG_MAXLEN", "10000")))
        self.publish_topic = os.getenv("AGG_PUBLISH_TOPIC", "aggregated_insights")
        self.publish_interval_sec = float(os.getenv("AGG_PUBLISH_INTERVAL_SEC", "5"))
        self._last_publish_ts = 0.0

    def start(self) -> None:
        self.logger.info("Starting aggregation service...")
        self.consumer = KafkaConsumer(
            "sentiment_results",
            bootstrap_servers=self.kafka_config["bootstrap_servers"],
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            group_id=self.kafka_config.get("group_id", "aggregation_group"),
            enable_auto_commit=True,
            auto_offset_reset=self.kafka_config.get("auto_offset_reset", "latest"),
        )
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_config["bootstrap_servers"],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks=self.kafka_config.get("acks", "all"),
            linger_ms=self.kafka_config.get("linger_ms", 5),
        )
        self.logger.info("Aggregation service started successfully")
        self._process_messages()

    def _maybe_publish(self) -> None:
        now = time.time()
        if now - self._last_publish_ts >= self.publish_interval_sec:
            snap = self.aggregator.snapshot().to_dict()
            self.producer.send(self.publish_topic, value={"type": "rolling_snapshot", "data": snap, "ts": now})
            # TODO: push snapshot to Redis/Elasticsearch if configured
            self._last_publish_ts = now

    def _process_messages(self) -> None:
        for message in self.consumer:
            try:
                data = message.value
                label = str(data.get("sentiment", "neutral")).lower()
                score = float(data.get("score", 0.0))
                self.aggregator.update(label=label, score=score)
                self._maybe_publish()
            except Exception as e:
                self.logger.exception(f"Error processing message: {e}")


def main() -> None:
    """Entrypoint for the aggregation service.

    Usage
        python src/streaming/aggregation_service.py

    Env Overrides
        AGG_MAXLEN: Rolling window length (default 10000)
        AGG_PUBLISH_TOPIC: Topic to publish snapshots (default aggregated_insights)
        AGG_PUBLISH_INTERVAL_SEC: Publish interval seconds (default 5)
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    kafka_bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").split(",")
    kafka_config = {
        "bootstrap_servers": kafka_bootstrap,
        "group_id": os.getenv("KAFKA_GROUP_ID", "aggregation_group"),
        "auto_offset_reset": os.getenv("KAFKA_AUTO_OFFSET_RESET", "latest"),
        "acks": os.getenv("KAFKA_ACKS", "all"),
        "linger_ms": int(os.getenv("KAFKA_LINGER_MS", "5")),
    }

    service = AggregationService(kafka_config)
    try:
        service.start()
    except KeyboardInterrupt:
        logging.info("Aggregation service stopped by user")
    except Exception as e:
        logging.error(f"Aggregation service failed: {e}")


if __name__ == "__main__":
    main()
