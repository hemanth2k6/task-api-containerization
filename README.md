# ML Model API - FastAPI + Docker

A production-ready machine learning API with FastAPI, scikit-learn, and Docker containerization.

## Requirements Covered

- FastAPI endpoints for model inference
- Docker containerization with Dockerfile
- Complete local setup instructions
- Example requests and responses

## Quick Start

### Option 1: Run Locally

pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000

### Option 2: Run with Docker

docker build -t ml-model-api .
docker run -d -p 8000:8000 --name ml-api-container ml-model-api
docker ps

## API Endpoints

GET /health - Health check
POST /predict - Single prediction
POST /predict/batch - Batch predictions
GET /model/info - Model information
GET /docs - Interactive Swagger UI

## Example Requests & Responses

Health Check:
curl http://localhost:8000/health
Response: {"status": "healthy", "model_loaded": true}

Single Prediction:
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"features\": [0.5]\*20}"
Response: {"prediction": 0, "confidence": 0.97, "request_id": "550e8400-e29b-41d4-a716-446655440000"}

Batch Prediction:
curl -X POST http://localhost:8000/predict/batch -H "Content-Type: application/json" -d "{\"data\": [[0.5]*20, [1.0]*20]}"
Response: [{"prediction": 0, "confidence": 0.97}, {"prediction": 1, "confidence": 0.95}]

## Testing

python test_request.py

## Demo Screenshots

1.png - API server running with uvicorn
2.png - Prediction request and response (curl)
3.png - Docker container running (docker ps)

## Project Structure

ml-model-api/
├── app.py
├── model.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── test_request.py
└── README.md

## Dependencies

FastAPI 0.104.1, Uvicorn 0.24.0, scikit-learn 1.3.0, numpy 1.24.3, joblib 1.3.2

Submitted to Alfido Tech - Task 3: Model API with Docker
