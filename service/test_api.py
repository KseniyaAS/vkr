import requests
import time
import json

BASE_URL = "http://127.0.0.1:8000"

def test_single_prediction():
    payload = {
        "Описание задачи": "Прошу завести номенклатуру от менеджера для ИНН 7701231231",
        "Регион": "МСК"
    }
    
    print("\nTesting single request...")
    start = time.perf_counter()
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    elapsed = time.perf_counter() - start
    
    if response.status_code == 200:
        print(f"Success! Prediction: {response.json()['prediction']}")
        print(f"Server-side latency: {response.json()['latency_ms']} ms")
        print(f"Client-side total time: {elapsed*1000:.2f} ms")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def test_batch_prediction():
    payload = [
        {"Описание задачи": "Отчет по продажам за март месяц", "Регион": "СПб"},
        {"Описание задачи": "Обновить цены поставщика Ромашка", "Регион": "НСК"}
    ]
    
    print("\nTesting batch request...")
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    if response.status_code == 200:
        results = response.json()
        for idx, res in enumerate(results):
            print(f"Item {idx}: {res['prediction']} (Worker: {res['responsible_worker']})")
    else:
        print(f"Error: {response.status_code}")

if __name__ == "__main__":
    try:
        # Check health
        health = requests.get(f"{BASE_URL}/health").json()
        print(f"Service status: {health['status']}")
        
        test_single_prediction()
        test_batch_prediction()
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to {BASE_URL}. Is the server running?")
