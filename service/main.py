from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import time
import json
from pathlib import Path
from typing import List, Union
import pandas as pd
import numpy as np

# Import our pipeline logic from the test folder
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "test"))
from thesis_pipeline import load_artifacts, DEFAULT_TEXT_COLUMN, DEFAULT_REGION_COLUMN

app = FastAPI(
    title="Ticket Categorization API",
    description="Microservice for automated ticket classification using TF-IDF and LinearSVC",
    version="1.0.0"
)

# Configuration
MODEL_PATH = Path("../best_transformer_model.pkl")
ROLES_PATH = Path("../roles.json")
THRESHOLD = 0.55

# SLA Config
SLA_CONFIG = {
    "Заведение номенклатуры от менеджера": 1,
    "Запрос на обновление цен (сведенный поставщик)": 3,
    "Запрос на сведение поставщика": 14,
    "Запрос отчета": 14,
    "Изменение складской программы": 3,
    "Изменение/Добавление номенклатуры 1С8 ТХ": 7,
    "Расчет Спек закупки": 3
}

# Global Artifacts (Loaded once at startup)
PIPELINE = None
WORKER_MAP = {}

class TicketRequest(BaseModel):
    text: str = Field(..., alias="Описание задачи", example="Прошу завести новую номенклатуру для ИНН 7701234567")
    region: str = Field("не указан", alias="Регион", example="Москва")

class TicketResponse(BaseModel):
    prediction: str
    confidence: float
    source: str
    responsible_worker: str
    time_to_complete_days: int
    latency_ms: float

@app.on_event("startup")
async def startup_event():
    global PIPELINE, WORKER_MAP
    
    # 1. Load Model
    if not MODEL_PATH.exists():
        print(f"CRITICAL: Model {MODEL_PATH} not found!")
        return
    PIPELINE = load_artifacts(MODEL_PATH)
    print(f"Model loaded: {MODEL_PATH}")

    # 2. Load Roles & Prep Worker Map
    if ROLES_PATH.exists():
        with open(ROLES_PATH, "r", encoding="utf-8") as f:
            roles_data = json.load(f)
            WORKER_MAP = {cat: list(workers.keys())[0] for cat, workers in roles_data.items() if workers}
    print(f"Roles loaded: {len(WORKER_MAP)} categories mapped.")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": PIPELINE is not None}

@app.post("/predict", response_model=Union[TicketResponse, List[TicketResponse]])
async def predict_ticket(request: Union[TicketRequest, List[TicketRequest]]):
    if PIPELINE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    is_batch = isinstance(request, list)
    requests = request if is_batch else [request]
    results = []

    for req in requests:
        start_time = time.perf_counter()
        
        # Prepare input for pipeline
        X_input = pd.DataFrame([{
            DEFAULT_TEXT_COLUMN: req.text,
            DEFAULT_REGION_COLUMN: req.region
        }])

        # Inference
        predict_proba = getattr(PIPELINE, "predict_proba", None)
        if predict_proba:
            probas = predict_proba(X_input)[0]
            max_idx = np.argmax(probas)
            max_proba = float(probas[max_idx])
            label = PIPELINE.classes_[max_idx]
            source = "ml"
            if max_proba < THRESHOLD:
                label = "manual_review"
                source = "threshold_fallback"
        else:
            label = PIPELINE.predict(X_input)[0]
            max_proba = 1.0
            source = "ml"

        latency_ms = (time.perf_counter() - start_time) * 1000

        results.append(TicketResponse(
            prediction=label,
            confidence=max_proba,
            source=source,
            responsible_worker=WORKER_MAP.get(label, "Unassigned"),
            time_to_complete_days=SLA_CONFIG.get(label, 7),
            latency_ms=round(latency_ms, 4)
        ))

    return results if is_batch else results[0]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
