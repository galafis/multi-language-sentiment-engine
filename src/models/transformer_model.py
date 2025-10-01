"""
Multi-Language Sentiment Analysis Engine
Transformer Model Implementation

This module implements the core transformer-based sentiment analysis models.
It supports multiple pre-trained models from Hugging Face for different languages
and provides a unified interface for sentiment prediction.

Author: Gabriel Demetrios Lafis
Date: June 2025
"""

import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoConfig,
    pipeline
)
from transformers.pipelines import Pipeline
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentimentModel:
    """
    A wrapper class for transformer-based sentiment analysis models.
    Supports multiple languages and models from Hugging Face.
    """
    
    # Mapping of language codes to recommended models
    LANGUAGE_MODEL_MAPPING = {
        'en': 'distilbert-base-uncased-finetuned-sst-2-english',
        'pt': 'neuralmind/bert-base-portuguese-cased',
        'es': 'dccuchile/bert-base-spanish-wwm-cased',
        'fr': 'camembert-base',
        'de': 'dbmdz/bert-base-german-cased',
        'it': 'dbmdz/bert-base-italian-cased',
        'nl': 'wietsedv/bert-base-dutch-cased',
        'zh': 'bert-base-chinese',
        'ja': 'cl-tohoku/bert-base-japanese',
        'ko': 'klue/bert-base',
        'ru': 'DeepPavlov/rubert-base-cased',
        'ar': 'asafaya/bert-base-arabic',
        'hi': 'monsoon-nlp/hindi-bert',
        # Default multilingual model for other languages
        'default': 'xlm-roberta-base'
    }
    
    # Emotion labels for emotion detection
    EMOTION_LABELS = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
    
    def __init__(
        self, 
        model_name_or_path: Optional[str] = None,
        language: str = 'en',
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None
    ):
        """
        Initialize the sentiment model.
        
        Args:
            model_name_or_path: Hugging Face model name or path to local model
            language: ISO language code (e.g., 'en', 'fr')
            device: Device to run the model on ('cpu', 'cuda:0', etc.)
            cache_dir: Directory to cache downloaded models
            token: Hugging Face authentication token (string) or boolean (legacy).
        """
        self.language = language.lower()
        
        # Select model based on language if not specified
        if model_name_or_path is None:
            model_name_or_path = self.LANGUAGE_MODEL_MAPPING.get(
                self.language, 
                self.LANGUAGE_MODEL_MAPPING['default']
            )
            logger.info(f"Using model {model_name_or_path} for language {language}")
        
        self.model_name = model_name_or_path
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                token=token
            )
            
            self.config = AutoConfig.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                token=token
            )
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                config=self.config,
                cache_dir=cache_dir,
                token=token
            ).to(self.device)
            
            # Create sentiment analysis pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info(f"Successfully loaded model {model_name_or_path}")
            
        except Exception as e:
            logger.error(f"Error loading model {model_name_or_path}: {str(e)}")
            raise
        
        # Determine label mapping based on model
        self._setup_label_mapping()
    
    def _setup_label_mapping(self):
        """Set up the mapping between model outputs and sentiment labels."""
        # Try to get label mapping from config
        if hasattr(self.config, "id2label"):
            self.id2label = self.config.id2label
        else:
            # Default mapping for binary sentiment
            self.id2label = {0: "negative", 1: "positive"}
        
        self.label2id = {v: k for k, v in self.id2label.items()}
        logger.info(f"Label mapping: {self.id2label}")
    
    def predict(
        self, 
        text: Union[str, List[str]], 
        batch_size: int = 8,
        return_all_scores: bool = False
    ) -> List[Dict]:
        """
        Predict sentiment for the given text(s).
        
        Args:
            text: Input text or list of texts
            batch_size: Batch size for processing
            return_all_scores: Whether to return scores for all labels
            
        Returns:
            List of dictionaries with sentiment predictions
        """
        if isinstance(text, str):
            text = [text]
        
        try:
            # Use the pipeline for prediction
            results = self.sentiment_pipeline(
                text, 
                batch_size=batch_size,
                top_k=None if return_all_scores else 1
            )
            
            # Format results
            formatted_results = []
            for i, result in enumerate(results):
                if return_all_scores:
                    # Convert to dictionary with label -> score mapping
                    scores = {item['label']: item['score'] for item in result}
                    
                    # Find the label with highest score
                    label = max(scores, key=scores.get)
                    score = scores[label]
                    
                    formatted_results.append({
                        'text': text[i],
                        'label': label,
                        'score': score,
                        'all_scores': scores
                    })
                else:
                    formatted_results.append({
                        'text': text[i],
                        'label': result['label'],
                        'score': result['score']
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def predict_emotion(self, text: Union[str, List[str]], batch_size: int = 8) -> List[Dict]:
        """
        Predict emotion for the given text(s).
        This is a specialized method for emotion detection models.
        
        Args:
            text: Input text or list of texts
            batch_size: Batch size for processing
            
        Returns:
            List of dictionaries with emotion predictions
        """
        # Check if this is an emotion model
        if not any(emotion in self.model_name.lower() for emotion in ['emotion', 'affect']):
            logger.warning("This model may not be trained for emotion detection")
        
        # Use the standard prediction method but return all scores
        results = self.predict(text, batch_size=batch_size, return_all_scores=True)
        
        # Format results specifically for emotions
        for result in results:
            # Extract emotion with highest score
            emotion = result['label']
            score = result['score']
            
            # Add emotion-specific fields
            result['emotion'] = emotion
            result['emotion_score'] = score
            
            # Map to basic emotion categories if possible
            if 'all_scores' in result:
                result['emotions'] = result['all_scores']
                del result['all_scores']
        
        return results
    
    def predict_aspect_based(
        self, 
        text: str, 
        aspects: List[str]
    ) -> Dict[str, Dict]:
        """
        Perform aspect-based sentiment analysis.
        
        Args:
            text: Input text
            aspects: List of aspects to analyze
            
        Returns:
            Dictionary mapping aspects to sentiment predictions
        """
        results = {}
        
        for aspect in aspects:
            # Create aspect-focused text
            aspect_text = f"Aspect: {aspect}. Text: {text}"
            
            # Get sentiment for this aspect
            sentiment = self.predict(aspect_text)[0]
            
            results[aspect] = {
                'label': sentiment['label'],
                'score': sentiment['score']
            }
        
        return {
            'text': text,
            'aspects': results
        }
    
    def batch_predict(
        self, 
        texts: List[str], 
        batch_size: int = 16
    ) -> List[Dict]:
        """
        Predict sentiment for a batch of texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            List of dictionaries with sentiment predictions
        """
        return self.predict(texts, batch_size=batch_size)
    
    def detect_sarcasm(self, text: str) -> Dict:
        """
        Detect sarcasm in the given text.
        This is a specialized method that works best with models fine-tuned for sarcasm detection.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sarcasm prediction
        """
        # Check if this is a sarcasm detection model
        if 'sarcasm' not in self.model_name.lower():
            logger.warning("This model may not be trained for sarcasm detection")
        
        result = self.predict(text)[0]
        
        # Map the result to sarcasm detection format
        return {
            'text': text,
            'is_sarcastic': result['label'] == 'SARCASM' or result['label'] == 'positive',
            'confidence': result['score']
        }
    
    def save_model(self, output_dir: str):
        """
        Save the model and tokenizer to the specified directory.
        
        Args:
            output_dir: Directory to save the model
        """
        os.makedirs(output_dir, exist_ok=True)
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model saved to {output_dir}")
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """
        Load a model from a local directory.
        
        Args:
            model_path: Path to the saved model
            **kwargs: Additional arguments to pass to the constructor
            
        Returns:
            SentimentModel instance
        """
        return cls(model_name_or_path=model_path, **kwargs)


class MultiLanguageSentimentModel:
    """
    A manager class that handles multiple sentiment models for different languages.
    Automatically selects the appropriate model based on the detected language.
    """
    
    def __init__(
        self,
        languages: List[str] = None,
        model_mapping: Dict[str, str] = None,
        device: str = None,
        cache_dir: str = None
    ):
        """
        Initialize the multi-language sentiment model manager.
        
        Args:
            languages: List of language codes to support
            model_mapping: Custom mapping of language codes to model names
            device: Device to run models on
            cache_dir: Directory to cache downloaded models
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        
        # Use default language mapping if none provided
        self.model_mapping = model_mapping or SentimentModel.LANGUAGE_MODEL_MAPPING
        
        # If languages not specified, use all from mapping
        self.languages = languages or list(self.model_mapping.keys())
        
        # Remove 'default' from languages if present
        if 'default' in self.languages:
            self.languages.remove('default')
        
        # Initialize models dictionary
        self.models = {}
        
        # Load default multilingual model
        logger.info("Loading default multilingual model")
        self.default_model = SentimentModel(
            model_name_or_path=self.model_mapping.get('default', 'xlm-roberta-base'),
            device=self.device,
            cache_dir=self.cache_dir
        )
    
    def load_model(self, language: str):
        """
        Load a model for the specified language.
        
        Args:
            language: Language code
        """
        if language not in self.models:
            model_name = self.model_mapping.get(language)
            
            if model_name:
                logger.info(f"Loading model for language: {language}")
                self.models[language] = SentimentModel(
                    model_name_or_path=model_name,
                    language=language,
                    device=self.device,
                    cache_dir=self.cache_dir
                )
            else:
                logger.warning(f"No specific model found for language {language}, using default")
                self.models[language] = self.default_model
    
    def get_model(self, language: str) -> SentimentModel:
        """
        Get the appropriate model for the specified language.
        Loads the model if not already loaded.
        
        Args:
            language: Language code
            
        Returns:
            SentimentModel instance
        """
        language = language.lower()
        
        # Load model if not already loaded
        if language not in self.models:
            self.load_model(language)
        
        # Return language-specific model or default
        return self.models.get(language, self.default_model)
    
    def predict(
        self, 
        text: Union[str, List[str]], 
        language: str = None
    ) -> List[Dict]:
        """
        Predict sentiment for the given text(s).
        
        Args:
            text: Input text or list of texts
            language: Language code (if None, will use language detection)
            
        Returns:
            List of dictionaries with sentiment predictions
        """
        if language is None:
            # Use language detection
            language = self.detect_language(text)
        
        model = self.get_model(language)
        results = model.predict(text)
        
        # Add language information to results
        for result in results:
            result['language'] = language
        
        return results
    
    def detect_language(self, text: Union[str, List[str]]) -> str:
        """
        Detect the language of the given text.
        
        Args:
            text: Input text or list of texts
            
        Returns:
            Detected language code
        """
        # This is a simplified language detection
        # In a real implementation, use a proper language detection library
        # like langdetect, fastText, or a specialized model
        
        # For now, default to English
        logger.warning("Using simplified language detection, defaulting to English")
        return 'en'
    
    def predict_batch(
        self, 
        texts: List[Dict[str, str]]
    ) -> List[Dict]:
        """
        Predict sentiment for a batch of texts with specified languages.
        
        Args:
            texts: List of dictionaries with 'text' and 'language' keys
            
        Returns:
            List of dictionaries with sentiment predictions
        """
        results = []
        
        # Group texts by language for batch processing
        language_groups = {}
        for i, item in enumerate(texts):
            lang = item.get('language', self.detect_language(item['text']))
            if lang not in language_groups:
                language_groups[lang] = []
            language_groups[lang].append((i, item['text']))
        
        # Process each language group
        for lang, text_group in language_groups.items():
            indices, lang_texts = zip(*text_group)
            
            # Get predictions for this language
            model = self.get_model(lang)
            lang_results = model.predict(list(lang_texts))
            
            # Add language information and original index
            for i, result in zip(indices, lang_results):
                result['language'] = lang
                result['original_index'] = i
                results.append(result)
        
        # Sort results back to original order
        results.sort(key=lambda x: x.pop('original_index'))
        return results


# Example usage
if __name__ == "__main__":
    # Example with single language model
    model = SentimentModel(language='en')
    result = model.predict("I love this product! It's amazing.")
    print(result)
    
    # Example with multi-language model
    multi_model = MultiLanguageSentimentModel(languages=['en', 'fr', 'es'])
    
    texts = [
        {"text": "I love this product!", "language": "en"},
        {"text": "J'adore ce produit!", "language": "fr"},
        {"text": "¡Me encanta este producto!", "language": "es"}
    ]
    
    results = multi_model.predict_batch(texts)
    print(results)

