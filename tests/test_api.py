import pytest
from fastapi.testclient import TestClient
from src.api.rest_api import app, sentiment_model, USE_CACHE, redis_client
import json

# Create a TestClient for the FastAPI app
client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "models_loaded" in response.json()

def test_analyze_sentiment_en():
    text = "I love this product!"
    response = client.post(
        "/analyze",
        headers={
            "Authorization": "Bearer demo"
        },
        json={
            "text": text,
            "language": "en"
        }
    )
    assert response.status_code == 200
    assert "label" in response.json()
    assert "score" in response.json()
    assert response.json()["language"] == "en"
    assert response.json()["label"] in ["POSITIVE", "NEGATIVE", "neutral", "LABEL_0", "LABEL_1"]

def test_analyze_sentiment_pt():
    text = "Eu amo este produto!"
    response = client.post(
        "/analyze",
        headers={
            "Authorization": "Bearer demo"
        },
        json={
            "text": text,
            "language": "pt"
        }
    )
    assert response.status_code == 200
    assert "label" in response.json()
    assert "score" in response.json()
    assert response.json()["language"] == "pt"
    assert response.json()["label"] in ["POSITIVE", "NEGATIVE", "neutral", "LABEL_0", "LABEL_1"]

def test_analyze_sentiment_auto_language():
    text = "This is a test sentence."
    response = client.post(
        "/analyze",
        headers={
            "Authorization": "Bearer demo"
        },
        json={
            "text": text
        }
    )
    assert response.status_code == 200
    assert "label" in response.json()
    assert "score" in response.json()
    assert response.json()["language"] == "en" # Default language detection is 'en'

def test_analyze_sentiment_unauthorized():
    text = "I love this product!"
    response = client.post(
        "/analyze",
        json={
            "text": text,
            "language": "en"
        }
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"

def test_analyze_sentiment_invalid_token():
    text = "I love this product!"
    response = client.post(
        "/analyze",
        headers={
            "Authorization": "Bearer invalid_token"
        },
        json={
            "text": text,
            "language": "en"
        }
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid authentication credentials"

def test_batch_analyze_sentiment():
    texts = [
        {"text": "I love this!", "language": "en"},
        {"text": "Eu gosto disto.", "language": "pt"}
    ]
    response = client.post(
        "/analyze/batch",
        headers={
            "Authorization": "Bearer demo"
        },
        json={
            "texts": texts
        }
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    assert all("label" in r and "score" in r and "language" in r for r in results)

def test_emotion_detection():
    text = "I am so happy today!"
    response = client.post(
        "/analyze/emotion",
        headers={
            "Authorization": "Bearer demo"
        },
        json={
            "text": text,
            "language": "en"
        }
    )
    assert response.status_code == 200
    assert "emotion" in response.json()
    assert "emotion_score" in response.json()
    assert response.json()["language"] == "en"

def test_aspect_sentiment_analysis():
    text = "The food was great but the service was slow."
    aspects = ["food", "service"]
    response = client.post(
        "/analyze/aspect",
        headers={
            "Authorization": "Bearer demo"
        },
        json={
            "text": text,
            "aspects": aspects,
            "language": "en"
        }
    )
    assert response.status_code == 200
    result = response.json()
    assert "text" in result
    assert "aspects" in result
    assert "food" in result["aspects"]
    assert "service" in result["aspects"]
    assert result["aspects"]["food"]["label"].upper() in ["POSITIVE", "NEGATIVE", "NEUTRAL"]
    assert result["aspects"]["service"]["label"].upper() in ["POSITIVE", "NEGATIVE", "NEUTRAL"]

def test_sarcasm_detection():
    text = "Oh, great, another Monday!"
    response = client.post(
        "/analyze/sarcasm",
        headers={
            "Authorization": "Bearer demo"
        },
        json={
            "text": text,
            "language": "en"
        }
    )
    assert response.status_code == 200
    assert "is_sarcastic" in response.json()
    assert "confidence" in response.json()
    assert response.json()["language"] == "en"

# Test caching functionality
@pytest.mark.skipif(not USE_CACHE, reason="Redis cache not enabled")
def test_caching_sentiment_analysis():
    text = "This is a cached test sentence."
    language = "en"
    cache_key = f"sentiment:{language}:{text}"

    # Clear cache for this key before test
    redis_client.delete(cache_key)

    # First request - should not be from cache
    response1 = client.post(
        "/analyze",
        headers={
            "Authorization": "Bearer demo"
        },
        json={
            "text": text,
            "language": language
        }
    )
    assert response1.status_code == 200
    result1 = response1.json()
    assert "label" in result1

    # Second request - should be from cache
    response2 = client.post(
        "/analyze",
        headers={
            "Authorization": "Bearer demo"
        },
        json={
            "text": text,
            "language": language
        }
    )
    assert response2.status_code == 200
    result2 = response2.json()
    assert result1 == result2

    # Verify it was cached
    cached_value = redis_client.get(cache_key)
    assert cached_value is not None
    assert json.loads(cached_value) == result1

# Test authentication
def test_login_success():
    response = client.post(
        "/token",
        data={
            "username": "demo",
            "password": "password123"
        }
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"

def test_login_failure():
    response = client.post(
        "/token",
        data={
            "username": "demo",
            "password": "wrong_password"
        }
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect username or password"


