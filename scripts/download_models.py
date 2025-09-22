#!/usr/bin/env python3
"""
Model Download Script for Multi-Language Sentiment Analysis Engine

This script downloads and caches pre-trained transformer models required
for the sentiment analysis engine. It supports various model types and
handles download failures gracefully.

Author: Gabriel Demetrios Lafis
Project: Multi-Language Sentiment Analysis Engine
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Tuple

try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification,
        pipeline
    )
    from huggingface_hub import snapshot_download
except ImportError as e:
    print(f"Error: Required dependencies not installed. Please run: pip install transformers huggingface_hub")
    print(f"Original error: {e}")
    sys.exit(1)


class ModelDownloader:
    """Downloads and manages pre-trained sentiment analysis models."""
    
    def __init__(self, models_dir: str = "models", cache_dir: str = None):
        """
        Initialize the model downloader.
        
        Args:
            models_dir: Directory to store downloaded models
            cache_dir: HuggingFace cache directory (optional)
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.cache_dir = cache_dir
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Define models to download
        self.models_config = {
            "multilingual_base": {
                "model_name": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
                "description": "XLM-RoBERTa base model for multilingual sentiment analysis",
                "languages": ["en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "ar"]
            },
            "english_base": {
                "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "description": "RoBERTa base model fine-tuned for English sentiment analysis",
                "languages": ["en"]
            },
            "bert_multilingual": {
                "model_name": "nlptown/bert-base-multilingual-uncased-sentiment",
                "description": "BERT multilingual model for sentiment analysis",
                "languages": ["en", "es", "fr", "de", "it", "pt", "nl", "ru"]
            },
            "finbert": {
                "model_name": "ProsusAI/finbert",
                "description": "FinBERT model for financial sentiment analysis",
                "languages": ["en"],
                "domain": "finance"
            }
        }
    
    def download_model(self, model_key: str) -> bool:
        """
        Download a specific model.
        
        Args:
            model_key: Key identifying the model in models_config
            
        Returns:
            bool: True if download successful, False otherwise
        """
        if model_key not in self.models_config:
            self.logger.error(f"Unknown model key: {model_key}")
            return False
            
        model_info = self.models_config[model_key]
        model_name = model_info["model_name"]
        
        self.logger.info(f"Downloading {model_key}: {model_info['description']}")
        self.logger.info(f"Model: {model_name}")
        
        try:
            # Download tokenizer
            self.logger.info("Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.cache_dir
            )
            
            # Download model
            self.logger.info("Downloading model...")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                cache_dir=self.cache_dir
            )
            
            # Save locally
            local_path = self.models_dir / model_key
            local_path.mkdir(exist_ok=True)
            
            tokenizer.save_pretrained(local_path)
            model.save_pretrained(local_path)
            
            # Test the model
            self.logger.info("Testing model...")
            test_pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer
            )
            
            # Test with a simple sentence
            test_result = test_pipeline("This is a great product!")
            self.logger.info(f"Test result: {test_result}")
            
            self.logger.info(f"Successfully downloaded and saved {model_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download {model_key}: {str(e)}")
            return False
    
    def download_all_models(self) -> Dict[str, bool]:
        """
        Download all configured models.
        
        Returns:
            Dict[str, bool]: Results of each model download
        """
        results = {}
        
        self.logger.info(f"Starting download of {len(self.models_config)} models...")
        
        for model_key in self.models_config.keys():
            results[model_key] = self.download_model(model_key)
            
        # Summary
        successful = sum(results.values())
        total = len(results)
        
        self.logger.info(f"Download complete: {successful}/{total} models successful")
        
        if successful < total:
            self.logger.warning("Some models failed to download. Check logs above for details.")
            
        return results
    
    def list_available_models(self) -> None:
        """
        Print information about available models.
        """
        print("\n=== Available Models ===")
        for key, info in self.models_config.items():
            print(f"\n{key}:")
            print(f"  Model: {info['model_name']}")
            print(f"  Description: {info['description']}")
            print(f"  Languages: {', '.join(info['languages'])}")
            if 'domain' in info:
                print(f"  Domain: {info['domain']}")
    
    def verify_models(self) -> Dict[str, bool]:
        """
        Verify that downloaded models are accessible.
        
        Returns:
            Dict[str, bool]: Verification results for each model
        """
        results = {}
        
        self.logger.info("Verifying downloaded models...")
        
        for model_key in self.models_config.keys():
            local_path = self.models_dir / model_key
            
            if not local_path.exists():
                self.logger.warning(f"Model {model_key} not found at {local_path}")
                results[model_key] = False
                continue
                
            try:
                # Try to load the model
                tokenizer = AutoTokenizer.from_pretrained(local_path)
                model = AutoModelForSequenceClassification.from_pretrained(local_path)
                
                # Quick test
                test_pipeline = pipeline(
                    "sentiment-analysis",
                    model=model,
                    tokenizer=tokenizer
                )
                test_pipeline("Test sentence")
                
                self.logger.info(f"Model {model_key} verified successfully")
                results[model_key] = True
                
            except Exception as e:
                self.logger.error(f"Model {model_key} verification failed: {str(e)}")
                results[model_key] = False
                
        return results


def main():
    """
    Main function to handle command line execution.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download pre-trained models for sentiment analysis"
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory to store downloaded models (default: models)"
    )
    parser.add_argument(
        "--cache-dir",
        help="HuggingFace cache directory (optional)"
    )
    parser.add_argument(
        "--model",
        help="Download specific model only (use --list to see available models)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify downloaded models"
    )
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(args.models_dir, args.cache_dir)
    
    if args.list:
        downloader.list_available_models()
        return
    
    if args.verify:
        results = downloader.verify_models()
        failed = [k for k, v in results.items() if not v]
        if failed:
            print(f"\nFailed models: {', '.join(failed)}")
            sys.exit(1)
        else:
            print("\nAll models verified successfully!")
        return
    
    if args.model:
        success = downloader.download_model(args.model)
        if not success:
            sys.exit(1)
    else:
        results = downloader.download_all_models()
        failed = [k for k, v in results.items() if not v]
        if failed:
            print(f"\nFailed downloads: {', '.join(failed)}")
            sys.exit(1)
    
    print("\nModel download completed successfully!")
    print("You can now start the sentiment analysis services.")


if __name__ == "__main__":
    main()
