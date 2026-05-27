
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from model import ModelWrapper
import logging
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Model API", description="API for machine learning model inference", version="1.0.0")

model_wrapper = ModelWrapper()
model_wrapper.load_model()

class PredictionRequest(BaseModel):
    features: List[float]
    metadata: Dict[str, Any] = {}

class BatchPredictionRequest(BaseModel):
    data: List[List[float]]

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float = None
    request_id: str = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

@app.get("/", response_model=Dict[str, str])
async def root():
    return {"message": "ML Model API is running", "docs": "/docs", "health": "/health"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", model_loaded=model_wrapper.is_loaded())

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        logger.info(f"Received prediction request with {len(request.features)} features")
        features_array = np.array(request.features).reshape(1, -1)
        prediction = model_wrapper.predict(features_array)
        
        confidence = None
        if hasattr(model_wrapper.model, 'predict_proba'):
            probabilities = model_wrapper.model.predict_proba(features_array)
            confidence = float(np.max(probabilities))
        
        request_id = request.metadata.get("request_id", str(uuid.uuid4()))
        
        return PredictionResponse(
            prediction=float(prediction[0]),
            confidence=confidence,
            request_id=request_id
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(request: BatchPredictionRequest):
    try:
        logger.info(f"Received batch prediction request with {len(request.data)} samples")
        predictions = []
        for features in request.data:
            features_array = np.array(features).reshape(1, -1)
            pred = model_wrapper.predict(features_array)
            
            confidence = None
            if hasattr(model_wrapper.model, 'predict_proba'):
                probabilities = model_wrapper.model.predict_proba(features_array)
                confidence = float(np.max(probabilities))
            
            predictions.append(PredictionResponse(
                prediction=float(pred[0]),
                confidence=confidence,
                request_id=str(uuid.uuid4())
            ))
        
        return predictions
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model/info")
async def model_info():
    return {
        "model_type": str(type(model_wrapper.model)),
        "is_loaded": model_wrapper.is_loaded(),
        "features_expected": model_wrapper.feature_count
    }