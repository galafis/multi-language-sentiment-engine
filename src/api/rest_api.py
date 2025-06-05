"""
Multi-Language Sentiment Analysis Engine
REST API Implementation

This module implements the FastAPI-based REST API for the sentiment analysis engine.
It provides endpoints for sentiment analysis, emotion detection, and aspect-based analysis.

Author: Gabriel Demetrios Lafis
Date: June 2025
"""

import os
import time
import json
import logging
from typing import Dict, List, Optional, Union, Any

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
import uvicorn
import redis
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge

# Import the sentiment model
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.transformer_model import MultiLanguageSentimentModel, SentimentModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize metrics
REQUESTS = Counter('sentiment_api_requests_total', 'Total number of API requests', ['endpoint', 'status'])
LATENCY = Histogram('sentiment_api_latency_seconds', 'API request latency in seconds', ['endpoint'])
MODEL_USAGE = Counter('sentiment_model_usage_total', 'Total number of model uses', ['model', 'language'])
ACTIVE_REQUESTS = Gauge('sentiment_api_active_requests', 'Number of active requests')

# Initialize Redis client for caching (if available)
try:
    redis_client = redis.Redis(
        host=os.environ.get('REDIS_HOST', 'localhost'),
        port=int(os.environ.get('REDIS_PORT', 6379)),
        db=0,
        decode_responses=True
    )
    redis_client.ping()  # Test connection
    USE_CACHE = True
    logger.info("Redis cache connected successfully")
except Exception as e:
    logger.warning(f"Redis cache not available: {str(e)}")
    USE_CACHE = False
    redis_client = None

# Initialize the sentiment model
LANGUAGES = ['en', 'pt', 'es', 'fr', 'de', 'it', 'zh', 'ja', 'ru', 'ar']
MODEL_CACHE_DIR = os.environ.get('MODEL_CACHE_DIR', './model_cache')

try:
    sentiment_model = MultiLanguageSentimentModel(
        languages=LANGUAGES,
        cache_dir=MODEL_CACHE_DIR
    )
    logger.info("Sentiment model initialized successfully")
except Exception as e:
    logger.error(f"Error initializing sentiment model: {str(e)}")
    raise

# Create FastAPI app
app = FastAPI(
    title="Multi-Language Sentiment Analysis API",
    description="API for analyzing sentiment in multiple languages using transformer models",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# OAuth2 for simple authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Define API models
class SentimentRequest(BaseModel):
    text: str = Field(..., description="Text to analyze")
    language: Optional[str] = Field(None, description="ISO language code (e.g., 'en', 'fr')")

class BatchSentimentRequest(BaseModel):
    texts: List[Dict[str, str]] = Field(..., description="List of texts with language codes")

class AspectSentimentRequest(BaseModel):
    text: str = Field(..., description="Text to analyze")
    aspects: List[str] = Field(..., description="List of aspects to analyze")
    language: Optional[str] = Field(None, description="ISO language code (e.g., 'en', 'fr')")

class EmotionRequest(BaseModel):
    text: str = Field(..., description="Text to analyze")
    language: Optional[str] = Field(None, description="ISO language code (e.g., 'en', 'fr')")

class SarcasmRequest(BaseModel):
    text: str = Field(..., description="Text to analyze")
    language: Optional[str] = Field("en", description="ISO language code (e.g., 'en', 'fr')")

class TokenRequest(BaseModel):
    username: str
    password: str

# Simple user database for demo purposes
# In production, use a proper authentication system
USERS = {
    "demo": {
        "username": "demo",
        "password": "password123",
    }
}

def get_current_user(token: str = Depends(oauth2_scheme)):
    """Simple token validation for demo purposes."""
    if token in USERS:
        return USERS[token]
    raise HTTPException(
        status_code=401,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

# Middleware for metrics
@app.middleware("http")
async def add_metrics(request: Request, call_next):
    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    
    try:
        response = await call_next(request)
        status = response.status_code
        endpoint = request.url.path
        
        # Record metrics
        REQUESTS.labels(endpoint=endpoint, status=status).inc()
        LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)
        
        return response
    finally:
        ACTIVE_REQUESTS.dec()

# Authentication endpoint
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = USERS.get(form_data.username)
    if not user or form_data.password != user["password"]:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"access_token": user["username"], "token_type": "bearer"}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": list(sentiment_model.models.keys())}

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    return prometheus_client.generate_latest()

# Sentiment analysis endpoint
@app.post("/analyze")
async def analyze_sentiment(
    request: SentimentRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    try:
        # Check cache if enabled
        if USE_CACHE:
            cache_key = f"sentiment:{request.language or 'auto'}:{request.text}"
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
        
        # Get language
        language = request.language or sentiment_model.detect_language(request.text)
        
        # Get model for language
        model = sentiment_model.get_model(language)
        
        # Record model usage
        MODEL_USAGE.labels(model=model.model_name, language=language).inc()
        
        # Predict sentiment
        result = model.predict(request.text)[0]
        result['language'] = language
        
        # Cache result in background
        if USE_CACHE:
            background_tasks.add_task(
                redis_client.setex,
                cache_key,
                3600,  # 1 hour expiration
                json.dumps(result)
            )
        
        return result
    
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch sentiment analysis endpoint
@app.post("/analyze/batch")
async def analyze_batch(
    request: BatchSentimentRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    try:
        # Process batch request
        results = sentiment_model.predict_batch(request.texts)
        
        # Record model usage for each text
        for result in results:
            MODEL_USAGE.labels(
                model=sentiment_model.get_model(result['language']).model_name,
                language=result['language']
            ).inc()
        
        return {"results": results}
    
    except Exception as e:
        logger.error(f"Error analyzing batch sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Emotion detection endpoint
@app.post("/analyze/emotion")
async def analyze_emotion(
    request: EmotionRequest,
    current_user: Dict = Depends(get_current_user)
):
    try:
        # Get language
        language = request.language or sentiment_model.detect_language(request.text)
        
        # Get model for language
        model = sentiment_model.get_model(language)
        
        # Record model usage
        MODEL_USAGE.labels(model=model.model_name, language=language).inc()
        
        # Predict emotion
        result = model.predict_emotion(request.text)[0]
        result['language'] = language
        
        return result
    
    except Exception as e:
        logger.error(f"Error analyzing emotion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Aspect-based sentiment analysis endpoint
@app.post("/analyze/aspect")
async def analyze_aspect_sentiment(
    request: AspectSentimentRequest,
    current_user: Dict = Depends(get_current_user)
):
    try:
        # Get language
        language = request.language or sentiment_model.detect_language(request.text)
        
        # Get model for language
        model = sentiment_model.get_model(language)
        
        # Record model usage
        MODEL_USAGE.labels(model=model.model_name, language=language).inc()
        
        # Predict aspect-based sentiment
        result = model.predict_aspect_based(request.text, request.aspects)
        result['language'] = language
        
        return result
    
    except Exception as e:
        logger.error(f"Error analyzing aspect sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Sarcasm detection endpoint
@app.post("/analyze/sarcasm")
async def detect_sarcasm(
    request: SarcasmRequest,
    current_user: Dict = Depends(get_current_user)
):
    try:
        # Get language
        language = request.language or "en"  # Default to English for sarcasm
        
        # Get model for language
        model = sentiment_model.get_model(language)
        
        # Record model usage
        MODEL_USAGE.labels(model=model.model_name, language=language).inc()
        
        # Detect sarcasm
        result = model.detect_sarcasm(request.text)
        result['language'] = language
        
        return result
    
    except Exception as e:
        logger.error(f"Error detecting sarcasm: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the API server
if __name__ == "__main__":
    uvicorn.run(
        "rest_api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True
    )

