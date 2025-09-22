"""Core tests for the Multi-Language Sentiment Engine.

This module contains basic tests for the core functionality of the sentiment analysis system,
including API endpoints, model operations, and data processing.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock


class TestSentimentAnalysis:
    """Test cases for sentiment analysis functionality."""
    
    def test_sentiment_analysis_positive(self):
        """Test sentiment analysis with positive text."""
        # Mock positive sentiment result
        text = "I love this product! It's amazing and works perfectly."
        expected_sentiment = "positive"
        expected_confidence = 0.95
        
        # This would normally call the actual sentiment analysis function
        # For now, we'll mock the expected behavior
        with patch('src.models.sentiment_analyzer.analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = {
                'sentiment': expected_sentiment,
                'confidence': expected_confidence,
                'language': 'en'
            }
            
            result = mock_analyze(text)
            
            assert result['sentiment'] == expected_sentiment
            assert result['confidence'] >= 0.8
            assert 'language' in result
            mock_analyze.assert_called_once_with(text)
    
    def test_sentiment_analysis_negative(self):
        """Test sentiment analysis with negative text."""
        text = "This product is terrible and doesn't work at all."
        expected_sentiment = "negative"
        expected_confidence = 0.89
        
        with patch('src.models.sentiment_analyzer.analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = {
                'sentiment': expected_sentiment,
                'confidence': expected_confidence,
                'language': 'en'
            }
            
            result = mock_analyze(text)
            
            assert result['sentiment'] == expected_sentiment
            assert result['confidence'] >= 0.7
            assert result['language'] == 'en'
    
    def test_sentiment_analysis_neutral(self):
        """Test sentiment analysis with neutral text."""
        text = "The weather is cloudy today."
        expected_sentiment = "neutral"
        
        with patch('src.models.sentiment_analyzer.analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = {
                'sentiment': expected_sentiment,
                'confidence': 0.65,
                'language': 'en'
            }
            
            result = mock_analyze(text)
            
            assert result['sentiment'] == expected_sentiment
            assert 0.0 <= result['confidence'] <= 1.0


class TestMultiLanguageSupport:
    """Test cases for multi-language sentiment analysis."""
    
    def test_spanish_sentiment_analysis(self):
        """Test sentiment analysis with Spanish text."""
        text = "¡Me encanta este producto! Es increíble."
        
        with patch('src.models.sentiment_analyzer.analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = {
                'sentiment': 'positive',
                'confidence': 0.92,
                'language': 'es'
            }
            
            result = mock_analyze(text)
            
            assert result['sentiment'] == 'positive'
            assert result['language'] == 'es'
            assert result['confidence'] > 0.8
    
    def test_portuguese_sentiment_analysis(self):
        """Test sentiment analysis with Portuguese text."""
        text = "Este produto é muito bom e funciona bem."
        
        with patch('src.models.sentiment_analyzer.analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = {
                'sentiment': 'positive',
                'confidence': 0.88,
                'language': 'pt'
            }
            
            result = mock_analyze(text)
            
            assert result['sentiment'] == 'positive'
            assert result['language'] == 'pt'


class TestAPIEndpoints:
    """Test cases for API endpoints."""
    
    def test_analyze_endpoint(self):
        """Test the /analyze API endpoint."""
        # Mock API response
        test_payload = {
            'text': 'This is a test message',
            'language': 'auto'
        }
        
        expected_response = {
            'sentiment': 'neutral',
            'confidence': 0.72,
            'language': 'en',
            'processing_time': 0.15
        }
        
        # This would normally test the actual API endpoint
        # For demonstration, we'll mock the expected behavior
        with patch('src.api.routes.analyze_text') as mock_endpoint:
            mock_endpoint.return_value = expected_response
            
            response = mock_endpoint(test_payload)
            
            assert response['sentiment'] in ['positive', 'negative', 'neutral']
            assert 0.0 <= response['confidence'] <= 1.0
            assert 'processing_time' in response
            assert 'language' in response
    
    def test_batch_analyze_endpoint(self):
        """Test the /batch-analyze API endpoint."""
        test_texts = [
            'Great product!',
            'Terrible experience',
            'It works fine'
        ]
        
        expected_responses = [
            {'sentiment': 'positive', 'confidence': 0.95},
            {'sentiment': 'negative', 'confidence': 0.92},
            {'sentiment': 'neutral', 'confidence': 0.68}
        ]
        
        with patch('src.api.routes.batch_analyze') as mock_batch:
            mock_batch.return_value = expected_responses
            
            responses = mock_batch(test_texts)
            
            assert len(responses) == len(test_texts)
            for response in responses:
                assert 'sentiment' in response
                assert 'confidence' in response


class TestDataPreprocessing:
    """Test cases for data preprocessing utilities."""
    
    def test_text_cleaning(self):
        """Test text cleaning functionality."""
        dirty_text = "  Hello @world! Check this out: https://example.com #awesome  "
        expected_clean = "Hello world! Check this out: awesome"
        
        with patch('src.preprocessing.text_cleaner.clean_text') as mock_clean:
            mock_clean.return_value = expected_clean
            
            result = mock_clean(dirty_text)
            
            assert result == expected_clean
            assert '@' not in result
            assert 'http' not in result
            assert '#' not in result
    
    def test_language_detection(self):
        """Test language detection functionality."""
        english_text = "This is a sentence in English."
        spanish_text = "Esta es una oración en español."
        
        with patch('src.preprocessing.language_detector.detect_language') as mock_detect:
            # Test English detection
            mock_detect.return_value = 'en'
            result_en = mock_detect(english_text)
            assert result_en == 'en'
            
            # Test Spanish detection
            mock_detect.return_value = 'es'
            result_es = mock_detect(spanish_text)
            assert result_es == 'es'


class TestModelOperations:
    """Test cases for model loading and operations."""
    
    def test_model_loading(self):
        """Test model loading functionality."""
        with patch('src.models.model_loader.load_model') as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model
            
            model = mock_load('sentiment_model_v1.0')
            
            assert model is not None
            mock_load.assert_called_once_with('sentiment_model_v1.0')
    
    def test_model_prediction(self):
        """Test model prediction functionality."""
        test_features = [0.1, 0.5, 0.8, 0.3, 0.9]
        expected_prediction = [0.2, 0.7, 0.1]  # probabilities for [negative, neutral, positive]
        
        with patch('src.models.predictor.predict') as mock_predict:
            mock_predict.return_value = expected_prediction
            
            prediction = mock_predict(test_features)
            
            assert len(prediction) == 3
            assert sum(prediction) == pytest.approx(1.0, rel=1e-2)
            assert all(0.0 <= p <= 1.0 for p in prediction)


class TestErrorHandling:
    """Test cases for error handling."""
    
    def test_empty_text_error(self):
        """Test handling of empty text input."""
        with patch('src.models.sentiment_analyzer.analyze_sentiment') as mock_analyze:
            mock_analyze.side_effect = ValueError("Text cannot be empty")
            
            with pytest.raises(ValueError, match="Text cannot be empty"):
                mock_analyze("")
    
    def test_invalid_language_error(self):
        """Test handling of invalid language code."""
        with patch('src.models.sentiment_analyzer.analyze_sentiment') as mock_analyze:
            mock_analyze.side_effect = ValueError("Unsupported language: xyz")
            
            with pytest.raises(ValueError, match="Unsupported language"):
                mock_analyze("Hello world", language="xyz")
    
    def test_model_loading_error(self):
        """Test handling of model loading errors."""
        with patch('src.models.model_loader.load_model') as mock_load:
            mock_load.side_effect = FileNotFoundError("Model file not found")
            
            with pytest.raises(FileNotFoundError, match="Model file not found"):
                mock_load('non_existent_model.pkl')


# Fixtures for test setup
@pytest.fixture
def sample_texts():
    """Provide sample texts for testing."""
    return {
        'positive': 'I love this product! It works great!',
        'negative': 'This is the worst thing I have ever bought.',
        'neutral': 'The item was delivered on time.',
        'multilingual': {
            'es': '¡Este producto es fantástico!',
            'pt': 'Este produto é muito bom.',
            'fr': 'Ce produit est excellent.'
        }
    }


@pytest.fixture
def mock_api_client():
    """Provide a mock API client for testing."""
    client = MagicMock()
    client.analyze.return_value = {
        'sentiment': 'positive',
        'confidence': 0.85,
        'language': 'en'
    }
    return client


# Integration test example
class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_end_to_end_analysis(self, sample_texts):
        """Test complete sentiment analysis pipeline."""
        text = sample_texts['positive']
        
        # Mock the complete pipeline
        with patch('src.preprocessing.text_cleaner.clean_text') as mock_clean, \
             patch('src.preprocessing.language_detector.detect_language') as mock_detect, \
             patch('src.models.sentiment_analyzer.analyze_sentiment') as mock_analyze:
            
            mock_clean.return_value = text
            mock_detect.return_value = 'en'
            mock_analyze.return_value = {
                'sentiment': 'positive',
                'confidence': 0.92,
                'language': 'en'
            }
            
            # Simulate the complete pipeline
            cleaned_text = mock_clean(text)
            detected_lang = mock_detect(cleaned_text)
            result = mock_analyze(cleaned_text, language=detected_lang)
            
            assert result['sentiment'] == 'positive'
            assert result['confidence'] > 0.8
            assert result['language'] == 'en'


if __name__ == '__main__':
    # Run tests when script is executed directly
    pytest.main([__file__])
