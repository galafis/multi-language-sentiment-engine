#!/usr/bin/env python3
"""Preprocessing Service for Multi-Language Sentiment Analysis Engine.

This service consumes raw text from Kafka topics, performs cleaning and tokenization,
and publishes preprocessed text to downstream topics for sentiment analysis.

Author: Gabriel Demetrios Lafis
Version: 0.1.0
"""

import logging
import json
from typing import Dict, Any, Optional
from kafka import KafkaConsumer, KafkaProducer
import re
from langdetect import detect


class TextPreprocessor:
    """Text preprocessing utilities for multi-language sentiment analysis."""
    
    def __init__(self):
        """Initialize the text preprocessor."""
        self.logger = logging.getLogger(__name__)
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for sentiment analysis.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned and normalized text
        """
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def detect_language(self, text: str) -> Optional[str]:
        """Detect the language of the input text.
        
        Args:
            text: Input text
            
        Returns:
            Detected language code or None if detection fails
        """
        try:
            return detect(text)
        except Exception as e:
            self.logger.warning(f"Language detection failed: {e}")
            return None


class PreprocessingService:
    """Kafka-based preprocessing service for sentiment analysis pipeline."""
    
    def __init__(self, kafka_config: Dict[str, Any]):
        """Initialize the preprocessing service.
        
        Args:
            kafka_config: Kafka configuration dictionary
        """
        self.kafka_config = kafka_config
        self.logger = logging.getLogger(__name__)
        self.preprocessor = TextPreprocessor()
        
        # Initialize Kafka consumer and producer
        self.consumer = None
        self.producer = None
        
    def start(self):
        """Start the preprocessing service."""
        self.logger.info("Starting preprocessing service...")
        
        # Initialize Kafka consumer
        self.consumer = KafkaConsumer(
            'raw_text_stream',
            bootstrap_servers=self.kafka_config['bootstrap_servers'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id='preprocessing_group'
        )
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_config['bootstrap_servers'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        self.logger.info("Preprocessing service started successfully")
        
        # Start processing messages
        self._process_messages()
    
    def _process_messages(self):
        """Process incoming messages from Kafka."""
        for message in self.consumer:
            try:
                data = message.value
                processed_data = self._process_text(data)
                
                if processed_data:
                    # Send to preprocessed topic
                    self.producer.send('preprocessed_text', value=processed_data)
                    self.logger.debug(f"Processed text with ID: {processed_data.get('id')}")
                    
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
    
    def _process_text(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single text item.
        
        Args:
            data: Input data dictionary containing text and metadata
            
        Returns:
            Processed data dictionary or None if processing fails
        """
        try:
            text = data.get('text', '')
            if not text:
                return None
                
            # Clean the text
            cleaned_text = self.preprocessor.clean_text(text)
            
            # Detect language if not provided
            language = data.get('language') or self.preprocessor.detect_language(cleaned_text)
            
            return {
                'id': data.get('id'),
                'text': cleaned_text,
                'original_text': text,
                'language': language,
                'timestamp': data.get('timestamp'),
                'source': data.get('source', 'unknown'),
                'metadata': data.get('metadata', {})
            }
            
        except Exception as e:
            self.logger.error(f"Error processing text: {e}")
            return None


def main():
    """Main function to run the preprocessing service."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Kafka configuration
    kafka_config = {
        'bootstrap_servers': ['localhost:9092']
    }
    
    # Create and start service
    service = PreprocessingService(kafka_config)
    
    try:
        service.start()
    except KeyboardInterrupt:
        logging.info("Preprocessing service stopped by user")
    except Exception as e:
        logging.error(f"Preprocessing service failed: {e}")


if __name__ == '__main__':
    main()
