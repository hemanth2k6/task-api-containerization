
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health Check: {response.json()}")
    return response.json()

def test_single_prediction():
    features = [0.5] * 20
    payload = {"features": features, "metadata": {"request_id": "test_001"}}
    response = requests.post(f"{BASE_URL}/predict", json=payload, headers={"Content-Type": "application/json"})
    print(f"Single Prediction: {response.json()}")
    return response.json()

def test_batch_prediction():
    data = [[0.5] * 20, [1.0] * 20, [-0.5] * 20]
    payload = {"data": data}
    response = requests.post(f"{BASE_URL}/predict/batch", json=payload, headers={"Content-Type": "application/json"})
    print(f"Batch Prediction: {response.json()}")
    return response.json()

def test_model_info():
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Model Info: {response.json()}")
    return response.json()

if __name__ == "__main__":
    print("Testing ML Model API\n" + "="*50)
    try:
        test_health()
        print("-"*50)
        test_model_info()
        print("-"*50)
        test_single_prediction()
        print("-"*50)
        test_batch_prediction()
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to API. Make sure the server is running.")
        print("Run: docker-compose up or uvicorn app:app --reload")